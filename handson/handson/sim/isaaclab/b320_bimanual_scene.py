"""Scene and asset contract for the registered B320 benchmark tasks.

This module keeps Isaac imports inside spawn functions so registration,
configuration, and interface tests do not start Kit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .hs005_scene_collision import (
    COLLISION_GEOMETRY_BY_SCENE,
    HAND_COLLISION_CONTACT_OFFSET,
    MODERN_APARTMENT_SCENE,
    SCENE_COLLISION_POLICY,
    SCENE_COLLISION_ROOT_NAME,
    collision_repair_boxes,
)

from .compat import quat_wxyz_to_isaaclab
from .eval_comb06_season_and_stir_scene import (
    TASK_NAME as XY6_TASK_NAME,
    make_task_spec as make_xy6_task_spec,
)
from .gen6_audio_mixer_scene import (
    TASK_NAME as GEN6_TASK_NAME,
    task_spec_from_reference as gen6_task_spec_from_reference,
)


from .xy4_contract import (
    TASK_NAME as XY4_TASK_NAME, TASK_LABEL as XY4_TASK_LABEL,
    TASK_PROMPT as XY4_TASK_PROMPT, TIME_LIMIT_SECONDS as XY4_TIME_LIMIT,
    X_SUCCESS_LABEL, BOTTLE_RETAINED_LABEL,
)
from .xy4_scene import MUG_HOME_XY

B320_SCENE_XML = "arena_arm_hand_b320_bimanual_benchmark_rokae_wuji1.xml"
B320_BC004_SCENE_XML = "arena_arm_hand_b320_bc004_carton_carry_rokae_wuji1.xml"
B320_ASSET_ROOT_PATH = "/World/Task/B320Benchmark"
B320_TABLE_PATH = f"{B320_ASSET_ROOT_PATH}/worktable"
# Four PhysX substeps per 30 Hz teleoperation command keep B320 aligned with
# the 120 Hz PhysX contract used by the other collection tasks. Canonical
# cameras are captured after the control step, outside the online physics loop.
B320_PHYSICS_HZ = 120
B320_CONTROL_HZ = 30
B320_TABLE_TOP_Z = 0.89
B320_D08_TABLE_DIFFUSE = (0.70, 0.70, 0.66)
B320_D08_LEG_DIFFUSE = (0.44, 0.45, 0.47)
B320_BC017_STUDY_TABLE_DIFFUSE = (0.30, 0.15, 0.065)
B320_TOWEL_TASK_NAME = "b320_towel_corner_synchronized_spread"
B320_BC014_TASK_NAME = "t30_pour_seasonings"
B320_BC017_TASK_NAME = "t15_open_drawers"
B320_KA020_TASK_NAME = "t32_press_soap_pump"
B320_KA021_TASK_NAME = "t05_fill_cup"
B320_KA005_2_TASK_NAME = "t38_set_dual_knobs"
KA020_DISPENSER_ASSET_BUNDLE_KEY = "cc_by_mike_nov_dishwashing_dispenser"
KA020_SPONGE_ASSET_BUNDLE_KEY = "cc_by_requital_dish_sponge"
KA020_DISPENSER_VISUAL_RELATIVE_USD = (
    f"{KA020_DISPENSER_ASSET_BUNDLE_KEY}/dishwashing_dispenser_visual.usd"
)
KA020_SPONGE_VISUAL_RELATIVE_USD = (
    f"{KA020_SPONGE_ASSET_BUNDLE_KEY}/dish_sponge_only_visual.usd"
)
KA020_DISPENSER_SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "dish-washing-liquid-dispenser-94907db381e5497ea2db0d5b1e8412c0"
)
KA020_DISPENSER_SOURCE_AUTHOR = "Mike Nov"
KA020_SPONGE_SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "sponge-for-washing-dishes-7ec3822721dd416aacc6308a7ed46c2c"
)
KA020_SPONGE_SOURCE_AUTHOR = "Requital (Azamkhon)"
KA020_SOURCE_LICENSE = "CC BY 4.0"
KA020_SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
# Both selected visuals keep their authored Y-up geometry. A +90 degree X
# rotation makes source +Y vertical in the task's Z-up frame. The sponge-only
# derivative additionally keeps its converted default root transform identity,
# so this runtime rotation cannot double-apply a hidden source-root rotation.
KA020_SOURCE_ORIENTATION_WXYZ = (
    math.sqrt(0.5),
    math.sqrt(0.5),
    0.0,
    0.0,
)
KA020_SPONGE_RUNTIME_ORIENTATION_WXYZ = KA020_SOURCE_ORIENTATION_WXYZ
KA020_DISPENSER_VISUAL_SCALE = 0.035
KA020_SPONGE_VISUAL_SCALE = 0.045
# Bounds below are measured from the hash-locked glTF closures after node
# transforms, before each asset's runtime orientation and unit scale.
KA020_DISPENSER_SOURCE_BOUNDS = (
    (-1.2147976690, 0.6083819151, -1.0452475433),
    (1.2143529654, 7.2391861772, 2.4101088245),
)
KA020_SPONGE_SOURCE_BOUNDS = (
    (-1.0542048167, 0.0, -0.5937442081),
    (1.0542048167, 0.3659545209, 0.5937442081),
)
KA020_BOTTLE_BODY_COLLIDER_SIZE_M = (0.0850, 0.0750, 0.1595)
KA020_BOTTLE_BODY_COLLIDER_CENTER_M = (0.0, 0.0, 0.1010)
KA020_BOTTLE_NECK_COLLIDER_RADIUS_M = 0.0208
KA020_BOTTLE_NECK_COLLIDER_HEIGHT_M = 0.0550
KA020_BOTTLE_NECK_COLLIDER_CENTER_M = (0.0, 0.0, 0.2045)
KA020_PUMP_HEAD_COLLIDER_SIZE_M = (0.0356, 0.1002, 0.0200)
KA020_PUMP_HEAD_CENTER_LOCAL_M = (0.0, -0.0343, 0.2434)
KA020_PUMP_OUTLET_OFFSET_M = (0.0, -0.0501, 0.0)
KA020_PUMP_TRAVEL_M = 0.018
KA020_SPONGE_PLAN_HALF_EXTENTS_M = (0.04745, 0.02670)
KA020_SPONGE_THICKNESS_M = 0.01650
KA020_SPONGE_COLLIDER_SIZE_M = (0.0949, 0.0165, 0.0534)
KA020_SPONGE_COLLIDER_CENTER_M = (0.0, 0.00825, 0.0)
KA020_SPONGE_RENDER_COMPONENT_COUNT = 5
KA020_SPONGE_RENDER_TRIANGLE_COUNT = 3180
KA020_SPONGE_REMOVED_HOLDER_TRIANGLE_COUNT = 45508
KA020_FLOW_PROXY_PATH = f"{B320_ASSET_ROOT_PATH}/soap_flow_proxy"
# The selected pump head is the source's long top/nozzle mesh.  Every other
# downloaded component stays on the bottle body so the prismatic link does not
# duplicate visible geometry.
KA020_MOVING_VISUAL_COMPONENT = "Cube_001_6"
KA020_MOVING_VISUAL_SOURCE_PRIM = (
    f"/World/root/GLTF_SceneRootNode/{KA020_MOVING_VISUAL_COMPONENT}"
)
KA020_MOVING_MATERIAL_SOURCE_PRIM = "/World/Looks/material_0"
BC014_ASSET_BUNDLE_KEY = "cc_by_anthony_yanez_salt_pepper_shakers"
BC014_ASSET_SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "salt-and-pepper-shakers-5f2adebdda754e40915d1330a61fa70c"
)
BC014_ASSET_LICENSE = "CC BY 4.0"
BC014_ASSET_ARCHIVE_SHA256 = (
    "a60fe7e5289422527695254baf4c6d98463fd71a3c5c5cd0c6e88696dcb9594d"
)
BC014_JAR_TARGET_HEIGHT_M = 0.096
BC014_JAR_TABLE_CLEARANCE_M = 0.0005
# These are the deterministic, non-randomized fallback centers. Formal
# collection resets replace them with seeded samples inside the corresponding
# hand's comfortable workspace.
BC014_GOAL_CENTERS_XY = {
    "salt_jar": (-0.23, 0.29),
    "pepper_jar": (-0.23, -0.29),
}
# The 100 mm square leaves a clear visual margin around the 56--57 mm shaker
# bodies. Left/right workspace sampling keeps the two zones non-overlapping.
BC014_GOAL_ZONE_SIZE_M = (0.100, 0.100, 0.006)
BC014_GOAL_SPACING_M = 0.58
BC014_GOAL_MARKER_PATHS = {
    "salt_jar": f"{B320_ASSET_ROOT_PATH}/salt_goal",
    "pepper_jar": f"{B320_ASSET_ROOT_PATH}/pepper_goal",
}
BC014_FLOW_PROXY_PATHS = {
    "salt_jar": f"{B320_ASSET_ROOT_PATH}/salt_water_flow_proxy",
    "pepper_jar": f"{B320_ASSET_ROOT_PATH}/pepper_water_flow_proxy",
}
KA021_ASSET_BUNDLE_KEY = "cc_ka021_hs005_kitchen_anilai_white_mug"
# Keep the historical mug/HS005 bundle intact (its aggregate hash is part of
# the B320 release contract) and install the downloaded BlenderKit faucet as a
# separate runtime bundle.  The task still uses the mug from the former and
# the static/moving faucet derivatives from the latter.
KA021_FAUCET_ASSET_BUNDLE_KEY = "blenderkit_kitchen_faucet"
KA021_FAUCET_STATIC_VISUAL_RELATIVE_USD = (
    f"{KA021_FAUCET_ASSET_BUNDLE_KEY}/faucet_static.usd"
)
KA021_FAUCET_HANDLE_VISUAL_RELATIVE_USD = (
    f"{KA021_FAUCET_ASSET_BUNDLE_KEY}/faucet_handle.usd"
)
KA021_FAUCET_SOURCE_URL = (
    "https://www.blendkit.com/asset-gallery-detail/"
    "b372d45e-2fbe-41cf-bf10-d0fd70dd2574/"
)
KA021_FAUCET_SOURCE_UUID = "0390e86f-2ffd-4745-b25c-d48397c67a51"
KA021_FAUCET_SOURCE_VERSION_ID = "876c68b4-1341-46ad-8491-3b9762dccd87"
KA021_FAUCET_SOURCE_CREATOR = "Cody Flynn"
KA021_FAUCET_SOURCE_LICENSE = "BlenderKit Royalty Free"
KA021_FAUCET_SOURCE_LICENSE_URL = "https://www.blendkit.com/docs/licenses/"
KA021_FAUCET_SOURCE_SHA256 = (
    "17faafad74eb833f5cecee9c84b13964bff92492fb1fbe3d795cea6e189183bf"
)
KA021_FAUCET_STATIC_SHA256 = (
    "7319c3b385f80cc47624b6bbbc7d72607633d9d0311f6c5af668d1cc4a3822ec"
)
KA021_FAUCET_HANDLE_SHA256 = (
    "7ac9a7f647c015eb8444253d5d1c47bcb81aa96efd018a876cc252160c9468e7"
)
KA021_HS005_SCENE_PATH = (
    Path(__file__).resolve().parents[1]
    / "envs/usd/hs005_sink/scene/hs005_assets.usda"
)

# BlenderKit's authored model is Z-up.  Its static base is kept on the
# reviewed HS005 counter plane and the local +X spout is yawed into the sink.
# The body and handle *physics roots* remain identity-oriented; only the
# visual wrappers carry this yaw.  The revolute joint below uses the same
# yawed source-Y frame (with its sign flipped) so the handle and HandleBase
# stay coaxial while a negative angle opens toward the robot side.
KA021_FAUCET_MODEL_SCALE = 0.90
KA021_FAUCET_MODEL_YAW_DEG = 139.20831019081422
KA021_FAUCET_MODEL_YAW_RAD = math.radians(KA021_FAUCET_MODEL_YAW_DEG)
KA021_FAUCET_MODEL_ORIENTATION_WXYZ = (
    math.cos(0.5 * KA021_FAUCET_MODEL_YAW_RAD),
    0.0,
    0.0,
    math.sin(0.5 * KA021_FAUCET_MODEL_YAW_RAD),
)
# The source Empty's local +Y is the physical cartridge/lever axis.  Add 180
# degrees around Z to choose the equivalent axis sign for the task's negative
# opening DOF: q<0 then moves the lever top along the source +X direction,
# which has a strong world -X component toward the robot workspace.
KA021_FAUCET_JOINT_FRAME_YAW_RAD = KA021_FAUCET_MODEL_YAW_RAD + math.pi
KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ = (
    math.cos(0.5 * KA021_FAUCET_JOINT_FRAME_YAW_RAD),
    0.0,
    0.0,
    math.sin(0.5 * KA021_FAUCET_JOINT_FRAME_YAW_RAD),
)
KA021_FAUCET_JOINT_AXIS_WORLD_W = (
    -math.sin(KA021_FAUCET_JOINT_FRAME_YAW_RAD),
    math.cos(KA021_FAUCET_JOINT_FRAME_YAW_RAD),
    0.0,
)
KA021_FAUCET_INITIAL_OPEN_DIRECTION_W = (
    math.cos(KA021_FAUCET_MODEL_YAW_RAD),
    math.sin(KA021_FAUCET_MODEL_YAW_RAD),
    0.0,
)
KA021_FAUCET_BASE_POSITION_W = (0.027, -0.275, 1.002)
KA021_FAUCET_NATIVE_HANDLE_PIVOT_M = (
    0.0,
    0.09024405479431152,
    0.06231890618801117,
)
KA021_FAUCET_NATIVE_OUTLET_M = (
    0.2126704454421997,
    0.0,
    0.22968795120716095,
)


def _ka021_model_local_point(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map one BlenderKit source point into the base body's local frame."""

    x, y, z = (float(value) for value in point)
    c = math.cos(KA021_FAUCET_MODEL_YAW_RAD)
    s = math.sin(KA021_FAUCET_MODEL_YAW_RAD)
    scale = KA021_FAUCET_MODEL_SCALE
    return (
        scale * (c * x - s * y),
        scale * (s * x + c * y),
        scale * z,
    )


def _ka021_model_world_point(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    local = _ka021_model_local_point(point)
    return tuple(
        float(KA021_FAUCET_BASE_POSITION_W[index] + local[index])
        for index in range(3)
    )


KA021_FAUCET_HANDLE_PIVOT_W = _ka021_model_world_point(
    KA021_FAUCET_NATIVE_HANDLE_PIVOT_M
)
KA021_FLOW_PROXY_PATH = f"{B320_ASSET_ROOT_PATH}/faucet_water_flow_proxy"
KA021_OUTLET_POSITION_W = _ka021_model_world_point(KA021_FAUCET_NATIVE_OUTLET_M)
KA021_SHARED_SCENE_ROOT_PATH = (
    f"{B320_ASSET_ROOT_PATH}/faucet_fixture/hs005_realistic_kitchen"
)
KA021_SHARED_SCENE_COLLISION_ROOT_PATH = (
    f"{KA021_SHARED_SCENE_ROOT_PATH}/{SCENE_COLLISION_ROOT_NAME}"
)
KA021_SAFE_BASIN_FLOOR_PATH = (
    f"{KA021_SHARED_SCENE_ROOT_PATH}/Sink/Collider"
)
(
    KA021_SAFE_BASIN_FLOOR_CENTER_W,
    KA021_SAFE_BASIN_FLOOR_SIZE_M,
) = COLLISION_GEOMETRY_BY_SCENE[MODERN_APARTMENT_SCENE]["floor_box"]
KA021_BASIN_FLOOR_TOP_Z = (
    float(KA021_SAFE_BASIN_FLOOR_CENTER_W[2])
    + 0.5 * float(KA021_SAFE_BASIN_FLOOR_SIZE_M[2])
)
KA021_HAND_CONTACT_OFFSET = HAND_COLLISION_CONTACT_OFFSET
KA021_SHARED_SCENE_COLLISION_POLICY = SCENE_COLLISION_POLICY
KA021_SHARED_SCENE_COLLISION_SPECS = tuple(
    (name, size, center)
    for name, center, size in collision_repair_boxes(MODERN_APARTMENT_SCENE)
)
# Conservative world-XY bounds of the actually visible, deep HS005 basin.
# These are inset by about 1 mm from the measured Sink_Sink4Metal_001_0
# bowl extrema (x=-0.359..0.067 m, y=-0.229..0.240 m).  The ribbed drainboard
# begins on the negative-Y side beyond this opening.  It may overlap the
# replacement floor in top view, but it is not part of the usable basin.
KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY = (
    (-0.358, -0.228),
    (0.066, 0.238),
)
# KA021's goal pad remains a small task-owned support marker on top of the
# shared scene countertop.  The sink floor/walls/countertop themselves are
# owned by the HS005 scene layer and are not recreated by this task.
KA021_CUP_PARK_SUPPORT_PATH = (
    f"{B320_ASSET_ROOT_PATH}/faucet_fixture/cup_park_support_collision"
)
KA021_CUP_PARK_MARKER_PATH = f"{B320_ASSET_ROOT_PATH}/cup_park_marker"
# The composed HS005 visual counter is not on the generic B320 table plane:
# its measured top face is z=1.011315 m.  Keep the physical pad on that same
# visible plane.  This smaller pad is fully inside the left counter polygon,
# outside the sink rim, with enough clearance for the 81 mm mug foot.
KA021_COUNTER_SURFACE_Z = float(
    COLLISION_GEOMETRY_BY_SCENE[MODERN_APARTMENT_SCENE]["countertop_z"]
)
KA021_CUP_PARK_SUPPORT_SIZE_M = (0.100, 0.120, 0.020)
KA021_CUP_PARK_SUPPORT_CENTER_W = (
    -0.445,
    -0.10,
    KA021_COUNTER_SURFACE_Z - 0.010,
)
KA021_CUP_PARK_POSITION_W = (-0.445, -0.10, KA021_COUNTER_SURFACE_Z)
# Public aliases expose the four shared countertop slabs to KA021 metrics and
# evidence without authoring another set of task-local colliders.
KA021_COUNTER_SURFACE_COLLISION_PATH = (
    KA021_SHARED_SCENE_COLLISION_ROOT_PATH
)
KA021_COUNTER_SURFACE_COLLISION_SPECS = tuple(
    spec
    for spec in KA021_SHARED_SCENE_COLLISION_SPECS
    if spec[0].endswith("_countertop")
)
KA021_CUP_PARK_TOLERANCE_M = 0.060
KA021_CUP_PARK_Z_TOLERANCE_M = 0.014
KA021_CUP_RIM_HEIGHT_M = 0.09387
KA021_CUP_OPENING_RADIUS_M = 0.030
KA021_MUG_HANDLE_COLLIDER_RADIUS_M = 0.0045
# The mug visual's converted runtime frame is source-scaled by 0.006.  These
# three center-line segments follow the modeled side handle in that frame and
# keep the handle physically present without closing the cup mouth.
KA021_MUG_HANDLE_CENTERLINE_SOURCE_M = (
    ((-7.0, 0.0, 3.2), (-10.5, 0.0, 3.2)),
    ((-10.5, 0.0, 3.2), (-10.5, 0.0, 12.8)),
    ((-10.5, 0.0, 12.8), (-7.0, 0.0, 12.8)),
)
KA021_CUP_UPRIGHT_TOLERANCE_DEG = 15.0
KA021_CUP_UP_AXIS_LOCAL = (0.0, 0.0, 1.0)
KA021_MIN_OUTLET_RIM_GAP_M = 0.040
KA021_MAX_OUTLET_RIM_GAP_M = 0.190
KA021_FLOW_VISIBLE_FRACTION = 0.35
KA021_SUCCESS_OPEN_FRACTION = 0.70
KA021_FAUCET_OPEN_AXIS = "Y"
KA021_FAUCET_OPEN_LIMIT_DEG = -45.0
KA021_FAUCET_OPEN_TRAVEL_RAD = math.radians(abs(KA021_FAUCET_OPEN_LIMIT_DEG))
KA021_FAUCET_RESET_POSITION_RAD = 0.0
# Retain the benchmark's coarse outward direction for task-language and
# compatibility consumers.  The source-aligned joint's exact initial motion
# is exposed separately as ``KA021_FAUCET_INITIAL_OPEN_DIRECTION_W``.
KA021_FAUCET_OUTWARD_DIRECTION_W = (-1.0, 0.0, 0.0)
KA021_FAUCET_COLUMN_COLLIDER_RADIUS_M = 0.023
KA021_FAUCET_ARCH_COLLIDER_RADIUS_M = 0.023
# Center-line samples follow the BlenderKit tube in its authored frame.  They
# are transformed into task/world coordinates below and then expressed in the
# fixed body's local frame when the collider prims are authored.  Four open
# capsules preserve the arch void and avoid the former broad AABB collider.
KA021_FAUCET_NATIVE_ARCH_CENTERLINE_M = (
    (0.000, 0.000, 0.105),
    (0.008, 0.000, 0.335),
    (0.060, 0.000, 0.402),
    (0.172, 0.000, 0.374),
    (0.213, 0.000, 0.260),
)
KA021_FAUCET_ARCH_COLLIDER_SEGMENTS_W = tuple(
    (
        _ka021_model_world_point(start),
        _ka021_model_world_point(end),
    )
    for start, end in zip(
        KA021_FAUCET_NATIVE_ARCH_CENTERLINE_M[:-1],
        KA021_FAUCET_NATIVE_ARCH_CENTERLINE_M[1:],
    )
)
# A small probe sits in the arch opening.  It is kept as evidence for the
# clearance regression and is not itself a collision shape.
KA021_FAUCET_ARCH_CLEARANCE_PROBE_CENTER_W = _ka021_model_world_point(
    (0.095, 0.000, 0.300)
)
KA021_FAUCET_ARCH_CLEARANCE_PROBE_HALF_EXTENTS_M = (0.008, 0.008, 0.008)
# The source lever is about 82 mm tall.  This box follows its yawed visual and
# leaves a small contact margin without swallowing the fixed handle base.
KA021_FAUCET_GRIP_COLLIDER_SIZE_M = (0.024, 0.024, 0.085)
KA021_FAUCET_GRIP_COLLIDER_CENTER_LOCAL_M = (0.0, 0.0, 0.0285)
_KA005_2_KNOB_MASS_KG = 0.09
_KA005_2_KNOB_ANGULAR_DAMPING = 0.40
_KA005_2_JOINT_DRIVE_DAMPING = 0.10
_KA005_2_JOINT_FRICTION = 0.065
_KA005_2_GRIP_STATIC_FRICTION = 0.85
_KA005_2_GRIP_DYNAMIC_FRICTION = 0.65

B320_TASK_NAMES = (
    XY6_TASK_NAME,
    "t10_lift_carton",
    "t20_pass_and_orient_pen",
    B320_BC014_TASK_NAME,
    B320_BC017_TASK_NAME,
    "t01_align_blocks_in_midair",
    B320_TOWEL_TASK_NAME,
    "t45_stir_pot",
    B320_KA020_TASK_NAME,
    B320_KA021_TASK_NAME,
    "b320_synchronized_double_cabinet_open",
    B320_KA005_2_TASK_NAME,
    GEN6_TASK_NAME,
)
B320_FAUCET_TASK_NAMES = frozenset({B320_KA021_TASK_NAME, XY4_TASK_NAME})
B320_TASK_NAMES = (*B320_TASK_NAMES, XY4_TASK_NAME)
B320_RIGID_TASK_NAMES = frozenset(set(B320_TASK_NAMES) - {B320_TOWEL_TASK_NAME})

BC011_PEN_VARIANTS = (
    "cc_by_black_pen_high_quality/pen.usda",
    "cc_by_the_pen/pen.usda",
)
# The two licensed source models were normalized to the same root frame and
# length, but their authored tips face opposite directions.  Keep this mapping
# explicit so reset randomization cannot silently invert the success predicate.
BC011_PEN_TIP_AXIS_LOCAL = {
    "cc_by_black_pen_high_quality/pen.usda": (-1.0, 0.0, 0.0),
    "cc_by_the_pen/pen.usda": (1.0, 0.0, 0.0),
}
# Kept as an alias for downstream checks that predate the public reset-time
# randomization contract.
_BC011_PEN_VARIANTS = BC011_PEN_VARIANTS
# Both randomized source assets are normalized to a 0.16 m longest axis.  The
# Wuji1 index finger measures about 0.112 m from its MCP joint to the outside
# of the distal tip, so the old pens were only 1.42 index-finger lengths and
# looked undersized in-hand.  A 0.20 m pen restores a realistic 1.78 ratio.
# Keep the scale uniform so each source's authored proportions and original
# convex-hull collider remain aligned.
BC011_SOURCE_PEN_LENGTH_M = 0.160
BC011_TARGET_PEN_LENGTH_M = 0.200
BC011_WUJI1_INDEX_REFERENCE_LENGTH_M = 0.112381
BC011_PEN_SCALE = (
    BC011_TARGET_PEN_LENGTH_M / BC011_SOURCE_PEN_LENGTH_M,
) * 3
BC011_PEN_TABLE_CLEARANCE_M = 0.00125


def bc011_tip_up_goal_pose_wxyz(variant: str) -> tuple[float, ...]:
    """Return the variant-specific pose whose authored pen tip points up."""

    tip_axis = BC011_PEN_TIP_AXIS_LOCAL[variant]
    return (
        -0.05,
        0.0,
        1.05,
        math.sqrt(0.5),
        0.0,
        -tip_axis[0] * math.sqrt(0.5),
        0.0,
    )


@dataclass(frozen=True)
class B320ObjectSpec:
    name: str
    relative_usd: str
    home_translation: tuple[float, float, float]
    home_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass_kg: float = 0.1
    dynamic: bool = True
    expected_colliders: int = 1
    static_friction: float = 1.2
    dynamic_friction: float = 1.0
    linear_damping: float = 0.10
    angular_damping: float = 0.65
    sleep_threshold: float = 0.01
    contact_offset: float = 0.002

    @property
    def source_path(self) -> Path:
        return B320_ASSET_DIRECTORY / self.relative_usd

    @property
    def prim_path(self) -> str:
        return f"{B320_ASSET_ROOT_PATH}/{self.name}"


@dataclass(frozen=True)
class B320TaskSpec:
    task_id: str
    task_name: str
    label: str
    prompt: str
    time_limit_seconds: int
    stable_frames: int
    left_role: str
    right_role: str
    object_specs: tuple[B320ObjectSpec, ...] = ()
    fixture_kind: str = "table"
    goal_pose_wxyz: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    success_labels: tuple[str, ...] = ()
    source_asset_keys: tuple[str, ...] = ()
    dof_names: tuple[str, ...] = ()


B320_ASSET_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "b320_bimanual_benchmark"
    / "assets"
)
KA021_FAUCET_ASSET_DIRECTORY = B320_ASSET_DIRECTORY / KA021_FAUCET_ASSET_BUNDLE_KEY

# FU001's selected visual/physics source is kept in its own bundle.  The
# historical ``dv_double_cabinet_44781`` bundle below remains available for
# provenance and for the no-argument legacy validators, but the canonical
# FU001 spawn never falls back to it.  Keeping these paths in the production
# module (rather than reaching into ``taskdev`` at spawn time) makes collection
# workers relocatable while the task-dev archive remains the provenance
# authority.
FU001_ASSET_BUNDLE_KEY = "polyhaven_painted_wooden_cabinet"
FU001_ASSET_DIRECTORY = B320_ASSET_DIRECTORY / FU001_ASSET_BUNDLE_KEY
FU001_RUNTIME_DIRECTORY = FU001_ASSET_DIRECTORY / "runtime"
FU001_ASSET_MANIFEST_PATH = FU001_ASSET_DIRECTORY / "asset_manifest.json"
FU001_ASSET_LICENSE_PATH = FU001_ASSET_DIRECTORY / "LICENSE.txt"
FU001_ASSET_SOURCE_ARCHIVE_PATH = (
    Path(__file__).resolve().parents[1]
    / "envs/usd/fu001_cabinet"
    / "assets/polyhaven_painted_wooden_cabinet/source"
    / "painted_wooden_cabinet_4k.gltf.zip"
)
FU001_ASSET_ARCHIVE_SHA256 = (
    "dec8c52b933705c3332619faf6ec8618f5e8ca4ff57c24f25016e9876eba0ddf"
)
FU001_ASSET_SOURCE_URL = "https://polyhaven.com/a/painted_wooden_cabinet"
FU001_ASSET_LICENSE = "CC0 1.0"
FU001_ASSET_LICENSE_URL = "https://polyhaven.com/license"
FU001_RUNTIME_COMPONENTS = {
    "body": FU001_RUNTIME_DIRECTORY / "painted_cabinet_body.usd",
    "door_left": FU001_RUNTIME_DIRECTORY / "painted_cabinet_door_left.usd",
    "door_right": FU001_RUNTIME_DIRECTORY / "painted_cabinet_door_right.usd",
}
# The source glTF is Y-up.  This quaternion maps source (x,y,z) to runtime
# (-z,-x,y), so the source +Y hinge axis becomes runtime +Z.
FU001_RUNTIME_ORIENTATION_WXYZ = (0.5, 0.5, -0.5, -0.5)
FU001_RUNTIME_SCALE = 0.55
FU001_HINGE_AXIS_SOURCE = "+Y"
FU001_HINGE_AXIS_RUNTIME = "+Z"
FU001_ROOT_HOME = (
    -0.15,
    0.0,
    B320_TABLE_TOP_Z + 0.001 - FU001_RUNTIME_SCALE * (-0.0003708677),
)
FU001_DOOR_SOURCE_PIVOTS = {
    "cabinet_door_left": (-0.5682284236, 0.5324437022, 0.2114711404),
    "cabinet_door_right": (0.5677881241, 0.5334159732, 0.2114711404),
}
FU001_DOOR_SOURCE_NODES = {
    "cabinet_door_left": "painted_wooden_cabinet_door_01",
    "cabinet_door_right": "painted_wooden_cabinet_door_02",
}
FU001_DOOR_COMPONENT_KEYS = {
    "cabinet_door_left": "door_left",
    "cabinet_door_right": "door_right",
}
FU001_DOOR_JOINT_LIMITS_DEG = {
    "target_joint_revolute": (0.0, 90.0),
    "joint_11": (-90.0, 0.0),
}
FU001_SOURCE_MESH_COLLISION_APPROXIMATION = "convexDecomposition"
B320_KA005_2_ASSET_DIRECTORY = B320_ASSET_DIRECTORY / "ka005_2_dual_knob"
B320_KA005_2_ASSET_MANIFEST_SHA256 = (
    "51bbb35362e6e513cfe2e70c66f0bf9ae3892ebec53db20dbf43cf9f497cb6de"
)
B320_KA005_2_TARGET_MARKER_PATHS = {
    "left": f"{B320_ASSET_ROOT_PATH}/panel_base/target_marker_left",
    "right": f"{B320_ASSET_ROOT_PATH}/panel_base/target_marker_right",
}

# The furniture source bundles were originally exported with their visual
# texture directory beside ``mobility_base.usd``.  Keep the expected names
# explicit so the strict runtime validator can reject an incomplete release
# before Hydra composes unresolved shader references.
_B320_FURNITURE_TEXTURE_NAMES = (
    "texture_0.jpg",
    "texture_1.jpg",
    "texture_2.png",
    "texture_3.jpg",
    "texture_4.jpg",
    "texture_5.png",
)

_RX_90_WXYZ = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
_RZ_180_WXYZ = (0.0, 0.0, 0.0, 1.0)

B320_TASK_SPECS: dict[str, B320TaskSpec] = {
    "t10_lift_carton": B320TaskSpec(
        task_id="10",
        task_name="t10_lift_carton",
        label="B320-BC004 Low-Compression Carton Carry",
        prompt=(
            "Support opposite lower sides of the carton with both hands, lift the "
            "entire carton clear of the table without either side exceeding 12 N, "
            "and keep it continuously airborne for 3 seconds."
        ),
        time_limit_seconds=22,
        # BC004 qualifies the three-second hold at the 120 Hz physics-substep
        # rate.  The generic control-frame debounce is therefore one frame.
        stable_frames=1,
        left_role="carton",
        right_role="carton",
        object_specs=(
            B320ObjectSpec(
                "carton",
                "dv_carton/model_carton.usd",
                (-0.30, 0.0, B320_TABLE_TOP_Z),
                mass_kg=0.75,
                # Match the current D08 single-object stability baseline while
                # keeping the carton material's higher intended grip friction.
                static_friction=1.25,
                dynamic_friction=1.0,
                linear_damping=0.40,
                angular_damping=2.00,
                sleep_threshold=0.020,
                contact_offset=0.001,
            ),
        ),
        goal_pose_wxyz=(-0.30, 0.0, B320_TABLE_TOP_Z + 0.010, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Carton pose and carton-table contact channels are physically readable",
            "The carton's lowest transformed corner clears the table by at least 10 mm",
            "Left and right hands physically support opposite lower carton sides",
            "Both side-force peaks stay at or below 12 N",
            "Every 120 Hz physics substep remains valid for 3 continuous seconds",
        ),
        source_asset_keys=("dv_carton",),
    ),
    "t20_pass_and_orient_pen": B320TaskSpec(
        task_id="20",
        task_name="t20_pass_and_orient_pen",
        label="B320-BC011 Slim Object Stable Handover",
        prompt=(
            "Touch the pen anywhere with the right hand only, then with both hands "
            "at the same time, then with the left hand only, and finish with the "
            "left hand holding the pen tip-up at 90 +/- 2 degrees from horizontal; "
            "after both hands have touched it, do not drop the pen."
        ),
        time_limit_seconds=32,
        # BC011 succeeds on the first control frame that reaches the complete
        # ordered handoff and tip-up state.  The terminal success result is the
        # latch; no scripted pose snap or post-contact freezing is applied.
        stable_frames=1,
        left_role="pen",
        right_role="pen",
        object_specs=(
            B320ObjectSpec(
                "pen",
                "cc_by_black_pen_high_quality/pen.usda",
                # Start supported.  The clearance scales with the authored
                # one-millimetre lower bound so the 1.25x collider touches the
                # table without beginning in penetration.
                (-0.30, -0.18, B320_TABLE_TOP_Z + BC011_PEN_TABLE_CLEARANCE_M),
                scale=BC011_PEN_SCALE,
                mass_kg=0.03,
                expected_colliders=1,
                # Match the proven thin-object grasp material.  The object
                # material uses max-combine below, so these values dominate
                # the generic hand skin without changing the hand or unrelated
                # B320 objects.
                static_friction=2.5,
                dynamic_friction=2.0,
            ),
        ),
        # The non-randomized default uses the black pen whose tip is local -X.
        goal_pose_wxyz=bc011_tip_up_goal_pose_wxyz(BC011_PEN_VARIANTS[0]),
        success_labels=(
            "Any positive right-hand contact anywhere on the pen occurred alone first",
            "Any positive contact from both hands occurred at the same time",
            "Any positive left-hand contact occurred alone after the overlap",
            "Left hand holds the lifted pen tip-up at 90 +/- 2 degrees without a drop",
        ),
        source_asset_keys=("target_marker",),
    ),
    B320_BC014_TASK_NAME: B320TaskSpec(
        task_id="30",
        task_name="t30_pour_seasonings",
        label="B320-BC014 Synchronized Salt/Pepper Sprinkling",
        prompt=(
            "The salt jar and cyan target are randomized in the left-hand comfort "
            "zone; the pepper jar and orange target are randomized in the right-"
            "hand comfort zone. Use the assigned hands, tilt each cap downward, "
            "and overlap both blue proxy streams in their matching targets. "
            "Finish before either jar pours for more than four seconds."
        ),
        time_limit_seconds=16,
        stable_frames=12,
        left_role="salt_jar",
        right_role="pepper_jar",
        object_specs=(
            B320ObjectSpec(
                "salt_jar",
                f"{BC014_ASSET_BUNDLE_KEY}/salt_jar.usda",
                (-0.30, 0.18, B320_TABLE_TOP_Z + BC014_JAR_TABLE_CLEARANCE_M),
                mass_kg=0.12,
                expected_colliders=2,
                contact_offset=0.0015,
            ),
            B320ObjectSpec(
                "pepper_jar",
                f"{BC014_ASSET_BUNDLE_KEY}/pepper_jar.usda",
                (-0.30, -0.18, B320_TABLE_TOP_Z + BC014_JAR_TABLE_CLEARANCE_M),
                mass_kg=0.12,
                expected_colliders=2,
                contact_offset=0.0015,
            ),
        ),
        goal_pose_wxyz=(
            -0.22,
            0.0,
            B320_TABLE_TOP_Z + BC014_JAR_TABLE_CLEARANCE_M,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        success_labels=(
            "Left hand manipulated salt and right hand manipulated pepper",
            "Salt proxy stream accumulated in the salt target zone",
            "Pepper proxy stream accumulated in the pepper target zone",
            "Both proxy streams hit their zones together long enough",
        ),
        source_asset_keys=(
            BC014_ASSET_BUNDLE_KEY,
            "target_marker",
            "water_flow_proxy",
        ),
    ),
    B320_BC017_TASK_NAME: B320TaskSpec(
        task_id="15",
        task_name=B320_BC017_TASK_NAME,
        label="B320-BC017 Dual Drawer Synchronized Open",
        prompt=(
            "Use each assigned hand to pull the highlighted drawer on its side "
            "beyond 60 percent travel; each side independently selects one of "
            "three drawers per episode, and both thresholds must be crossed "
            "within 1.2 seconds."
        ),
        time_limit_seconds=30,
        stable_frames=1,
        left_role="left_drawer",
        right_role="right_drawer",
        fixture_kind="bc017_dedicated_runtime",
        success_labels=(
            "All six drawer DOFs, twelve physical contact views, four selected "
            "semantic contact channels, and fixed-fixture truth are readable",
            "The left drawer crosses strictly beyond 60 percent with only its assigned left hand on the preceding and crossing substeps",
            "The right drawer crosses strictly beyond 60 percent with only its assigned right hand on the preceding and crossing substeps",
            "The two qualifying 120 Hz crossings occur no more than 144 substeps (1.2 seconds) apart",
            "Both drawers remain strictly beyond 60 percent for two consecutive 120 Hz samples without a physical failure",
        ),
        source_asset_keys=("cc0_polyhaven_metal_office_desk",),
        dof_names=(
            "bc017_left_drawer_prismatic_joint",
            "bc017_left_middle_drawer_prismatic_joint",
            "bc017_left_bottom_drawer_prismatic_joint",
            "bc017_right_drawer_prismatic_joint",
            "bc017_right_middle_drawer_prismatic_joint",
            "bc017_right_bottom_drawer_prismatic_joint",
        ),
    ),
    "t01_align_blocks_in_midair": B320TaskSpec(
        task_id="01",
        task_name="t01_align_blocks_in_midair",
        label="B320-BC021 Two-Block Midair Face Mating",
        prompt=(
            "Lift both assigned blocks with their assigned hands, then mate their "
            "marked faces in midair and hold the physical contact stable for 0.4 seconds."
        ),
        time_limit_seconds=60,
        # The dedicated BC021 state machine owns the 48 consecutive 120 Hz
        # samples; the generic control-step debounce consumes one frame.
        stable_frames=1,
        left_role="block_a",
        right_role="block_b",
        fixture_kind="bc021_dedicated_runtime",
        success_labels=(
            "Current assigned LA/RB contacts are readable and crossed LB/RA contacts are clear",
            "Both blocks' lowest geometry points are at least 80 mm above the table",
            "Marked-face normal, lateral, and axial errors are within the final geometry limits",
            "The marked patches remain in reported physical AB contact for 0.4 seconds",
        ),
    ),
    B320_TOWEL_TASK_NAME: B320TaskSpec(
        task_id="B320-BC025",
        task_name=B320_TOWEL_TASK_NAME,
        label="B320-BC025 Towel Corner Synchronized Spread",
        prompt=(
            "Physically contact the assigned upper-left and upper-right Terry Cloth "
            "corners with the matching hands, spread the towel without "
            "overstretching it, place at least three corners inside the target "
            "frame with over 80 percent coverage, then release together."
        ),
        time_limit_seconds=45,
        stable_frames=15,
        left_role="towel_left_corner",
        right_role="towel_right_corner",
        fixture_kind="towel_surface",
        goal_pose_wxyz=(-0.05, 0.0, B320_TABLE_TOP_Z + 0.004, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Assigned left/right hands physically contacted the upper-left/right corners",
            "Towel stretch stays below 15 percent and assigned end distance is within 10 percent",
            "At least three corners are inside the target frame",
            "Projected target coverage exceeds 80 percent",
            "Both hands release within one second and state holds 0.5 seconds",
        ),
        source_asset_keys=("bc025-towel-01", "Pillow024_collision_parameters", "target_marker"),
    ),
    "t45_stir_pot": B320TaskSpec(
        task_id="45",
        task_name="t45_stir_pot",
        label="B320-KA012 Dutch Oven Stabilized Stirring",
        prompt=(
            "With the removable lid set beside the open Dutch oven, keep the pot "
            "fixed by its left handle with the left hand while the right hand moves "
            "the spoon head through two full ordered circles."
        ),
        time_limit_seconds=40,
        stable_frames=12,
        left_role="dutch_oven",
        right_role="spoon",
        object_specs=(
            B320ObjectSpec(
                "dutch_oven",
                "polyhaven_pot_enamel_01/pot_base.usd",
                (-0.195, 0.08, B320_TABLE_TOP_Z + 0.001),
                _RX_90_WXYZ,
                scale=(0.78, 0.78, 0.78),
                mass_kg=2.80,
                static_friction=1.40,
                dynamic_friction=1.10,
                linear_damping=0.35,
                angular_damping=1.20,
                contact_offset=0.0015,
            ),
            B320ObjectSpec(
                "dutch_oven_lid",
                "polyhaven_pot_enamel_01/pot_lid.usd",
                (0.13, 0.36, B320_TABLE_TOP_Z + 0.001),
                _RX_90_WXYZ,
                scale=(0.78, 0.78, 0.78),
                mass_kg=0.55,
                static_friction=1.30,
                dynamic_friction=1.05,
                linear_damping=0.30,
                angular_damping=1.40,
                contact_offset=0.0015,
            ),
            B320ObjectSpec(
                "spoon",
                "dexverse_authored_spoon/dexverse_authored_spoon.usd",
                (-0.30, -0.20, B320_TABLE_TOP_Z),
                _RX_90_WXYZ,
                mass_kg=0.08,
                expected_colliders=2,
            ),
        ),
        goal_pose_wxyz=(-0.195, 0.08, B320_TABLE_TOP_Z + 0.08, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Left hand continuously stabilizes the Dutch oven",
            "Right hand physically owns the spoon",
            "Spoon head crosses all six ordered angular sectors",
            "Cumulative stirring angle exceeds 650 degrees",
            "Dutch oven translation after stirring begins stays below 25 mm",
        ),
        source_asset_keys=(
            "polyhaven_pot_enamel_01",
            "dexverse_authored_spoon",
        ),
    ),
    B320_KA020_TASK_NAME: B320TaskSpec(
        task_id="32",
        task_name=B320_KA020_TASK_NAME,
        label="B320-KA020 Sponge-Stabilized Soap Dispense",
        prompt=(
            "Place the dish sponge under the dish-soap bottle, hold the bottle with "
            "the right hand, align its outlet over the sponge, and "
            "physically depress the visible pump head with the right hand."
        ),
        time_limit_seconds=90,
        stable_frames=12,
        left_role="sponge",
        right_role="soap_bottle",
        object_specs=(
            B320ObjectSpec(
                "sponge",
                KA020_SPONGE_VISUAL_RELATIVE_USD,
                (-0.21, 0.18, B320_TABLE_TOP_Z + 0.002),
                KA020_SPONGE_RUNTIME_ORIENTATION_WXYZ,
                scale=(KA020_SPONGE_VISUAL_SCALE,) * 3,
                mass_kg=0.045,
                expected_colliders=1,
                static_friction=1.35,
                dynamic_friction=1.10,
                linear_damping=0.28,
                angular_damping=1.10,
                contact_offset=0.0015,
            ),
            B320ObjectSpec(
                "soap_bottle",
                KA020_DISPENSER_VISUAL_RELATIVE_USD,
                (
                    -0.21,
                    -0.18,
                    B320_TABLE_TOP_Z
                    + 0.002
                    - KA020_DISPENSER_SOURCE_BOUNDS[0][1]
                    * KA020_DISPENSER_VISUAL_SCALE,
                ),
                scale=(KA020_DISPENSER_VISUAL_SCALE,) * 3,
                mass_kg=0.36,
                expected_colliders=2,
                static_friction=1.25,
                dynamic_friction=1.00,
                linear_damping=0.22,
                angular_damping=1.20,
                contact_offset=0.0015,
            ),
        ),
        goal_pose_wxyz=(-0.18, 0.12, B320_TABLE_TOP_Z + 0.08, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Right hand physically owns the dish-soap bottle",
            "Right hand physically contacts the moving pump head",
            "Nozzle outlet projects inside the measured sponge face",
            "Nozzle outlet stays 20-80 mm above the sponge surface",
            "Visible pump head reaches at least 75 percent travel",
        ),
        source_asset_keys=(
            KA020_SPONGE_ASSET_BUNDLE_KEY,
            KA020_DISPENSER_ASSET_BUNDLE_KEY,
        ),
        dof_names=("soap_dispense_joint",),
    ),
    B320_KA021_TASK_NAME: B320TaskSpec(
        task_id="05",
        task_name="t05_fill_cup",
        label="B320-KA021 Cup-Stabilized Faucet Open",
        prompt=(
            "Lift and hold the white ceramic mug under the faucet outlet with its local +Z "
            "axis within 15 degrees, inclusive, of world +Z. Pull the right faucet lever "
            "outward toward the robots along its source-aligned hinge (with a world -X "
            "component) beyond 70 percent of its 45-degree flow travel; left-right "
            "temperature motion does not count. Keep the faucet open while the visible "
            "stream enters the mug for 15 "
            "consecutive control frames (0.5 seconds), then place the upright mug "
            "on the marked counter pad outside the sink and close the faucet."
        ),
        time_limit_seconds=22,
        stable_frames=15,
        left_role="ceramic_mug",
        right_role="faucet_handle",
        object_specs=(
            B320ObjectSpec(
                "ceramic_mug",
                f"{KA021_ASSET_BUNDLE_KEY}/ceramic_white_mug.usd",
                (-0.15, 0.02, KA021_BASIN_FLOOR_TOP_Z),
                (1.0, 0.0, 0.0, 0.0),
                scale=(0.006, 0.006, 0.006),
                mass_kg=0.18,
                expected_colliders=2,
                static_friction=1.35,
                dynamic_friction=1.10,
                linear_damping=0.60,
                angular_damping=12.00,
                sleep_threshold=0.025,
                contact_offset=0.0015,
            ),
        ),
        fixture_kind="ka021_hs005_realistic_faucet",
        goal_pose_wxyz=(*KA021_OUTLET_POSITION_W, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Left hand contacts the mug during the fill phase",
            "Right hand contacts the faucet lever during the fill phase",
            "Mug opening contains the outlet projection within 30 mm during filling",
            "Mug local +Z remains within 15 degrees during filling",
            "Outlet remains 40-190 mm above the mug rim during filling",
            "Faucet stays open and visible water stream enters the mug for 0.5 seconds",
            "After filling, mug is parked on the marked counter pad outside the sink",
            "Faucet is closed after the mug is parked",
        ),
        source_asset_keys=(KA021_ASSET_BUNDLE_KEY, KA021_FAUCET_ASSET_BUNDLE_KEY),
        dof_names=("RevoluteJoint_faucet_3_right",),
    ),
    "b320_synchronized_double_cabinet_open": B320TaskSpec(
        task_id="B320-FU001",
        task_name="b320_synchronized_double_cabinet_open",
        label="B320-FU001 Synchronized Double Cabinet Open",
        prompt=(
            "Use the left hand on the left cabinet door and the right hand on the "
            "right door, open both beyond 80 percent, cross the threshold within "
            "one second, then release both handles while the doors remain open."
        ),
        time_limit_seconds=32,
        stable_frames=12,
        left_role="cabinet_door_left",
        right_role="cabinet_door_right",
        fixture_kind="cabinet_44781",
        goal_pose_wxyz=(-0.10, 0.0, 1.25, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Left and right hands contact their assigned doors",
            "target_joint_revolute exceeds 80 percent travel",
            "joint_11 exceeds 80 percent travel",
            "Threshold times differ by at most one second",
            "Both hands release after threshold within one second",
            "Both doors remain in the success range for 0.4 seconds",
        ),
        source_asset_keys=(FU001_ASSET_BUNDLE_KEY,),
        dof_names=("target_joint_revolute", "joint_11"),
    ),
    B320_KA005_2_TASK_NAME: B320TaskSpec(
        task_id="38",
        task_name=B320_KA005_2_TASK_NAME,
        label="B320-KA005 Synchronized Dual-Knob Setting-2",
        prompt=(
            "Use the left hand on the left knob and the right hand on the right "
            "knob, turn them in their assigned directions in either order, set both displayed targets, "
            "release both knobs, and keep the settings stable."
        ),
        time_limit_seconds=40,
        # The task-specific state machine qualifies every physical hold at
        # 120 Hz; the generic success debounce therefore consumes one frame.
        stable_frames=1,
        left_role="knob_left",
        right_role="knob_right",
        fixture_kind="dual_knob_box_2",
        goal_pose_wxyz=(-0.14, 0.0, 0.915, 1.0, 0.0, 0.0, 0.0),
        success_labels=(
            "Two named revolute DOFs and all four hand-knob contact channels are readable",
            "Left hand contacts the left knob and right hand contacts the right knob",
            "Left turns counter-clockwise and right turns clockwise",
            "Both knobs reach their displayed targets within 5 degrees, in either order",
            "All hand-to-knob contacts release after target acquisition",
            "Both released knobs remain in range and stationary for 0.5 seconds",
        ),
        source_asset_keys=("project_boxy", "rotary_knob_0x_anas"),
        dof_names=("left_knob_joint", "right_knob_joint"),
    ),
}

B320_TASK_SPECS[XY6_TASK_NAME] = make_xy6_task_spec(
    B320_TASK_SPECS[B320_BC014_TASK_NAME],
    B320_TASK_SPECS["t45_stir_pot"],
)
B320_TASK_SPECS[XY4_TASK_NAME] = replace(
    B320_TASK_SPECS[B320_KA021_TASK_NAME],
    task_id="XY4", task_name=XY4_TASK_NAME, label=XY4_TASK_LABEL,
    prompt=XY4_TASK_PROMPT, time_limit_seconds=XY4_TIME_LIMIT,
    object_specs=tuple(replace(obj, home_translation=(*MUG_HOME_XY, obj.home_translation[2]))
                       for obj in B320_TASK_SPECS[B320_KA021_TASK_NAME].object_specs),
    success_labels=(X_SUCCESS_LABEL, *B320_TASK_SPECS[B320_KA021_TASK_NAME].success_labels,
                    BOTTLE_RETAINED_LABEL),
)
B320_TASK_SPECS[GEN6_TASK_NAME] = gen6_task_spec_from_reference(
    B320_TASK_SPECS[B320_KA005_2_TASK_NAME]
)
B320_DUAL_KNOB_TASK_NAMES = frozenset({B320_KA005_2_TASK_NAME, GEN6_TASK_NAME})

B320_SUCCESS_CONDITION_LABELS = {
    name: spec.success_labels for name, spec in B320_TASK_SPECS.items()
}
B320_TASK_SCENES = {name: B320_SCENE_XML for name in B320_TASK_NAMES}
B320_TASK_SCENES["t10_lift_carton"] = B320_BC004_SCENE_XML

_REQUIRED_SOURCES = tuple(
    sorted(
        {
            object_spec.relative_usd
            for task_spec in B320_TASK_SPECS.values()
            for object_spec in task_spec.object_specs
        }
        | {
            "dv_double_cabinet_44781/mobility.usd",
            "dv_double_cabinet_44781/meta.json",
        }
    )
)
_DERIVED_GLBS = {
    "dexverse_authored_spoon/dexverse_authored_spoon.usd": (
        "dexverse_authored_spoon/dexverse_authored_spoon.glb",
        "d6828940e4570f8bc4050315e3e77511089866d17babe510bafc8ed28f1c26ad",
    ),
    "dexverse_authored_sponge/dexverse_authored_sponge.usd": (
        "dexverse_authored_sponge/dexverse_authored_sponge.glb",
        "ff55877962c3a35d9872eb448cdf94fa531b1c824cb19311a1141bb958a46dcc",
    ),
}
B320_SOURCE_BUNDLE_SHA256 = {
    "dv_carton": "5d7d738d184acbf32080023021ebc7d79df1c86611fd1bacbd66ea48868574b6",
    "dexverse_pen": "70cd998340b0c2b2c0092741b275c1678faa7d17559b506d82b29f00b4d6086b",
    BC014_ASSET_BUNDLE_KEY: (
        "efba30ea08e24bbed9b5d638a62ad274c6cba33ef418c6d1a71f18a7c83a1412"
    ),
    "polyhaven_pot_enamel_01": (
        "947de82d7d9624f0e387c8b033c7f14188e641e1cdfbfb2052b9d93eb740988d"
    ),
    KA021_ASSET_BUNDLE_KEY: (
        "5dc27d83107a8688f0a810013682fff74d1b24ff9557b59e7000c70e51dd6535"
    ),
    "dexverse_authored_spoon": "d3cdb721ea12f82fbcb342aeb0ee36d449d60729538104d491bd9745c44bb555",
    "dexverse_authored_sponge": "1cdf56103878aea6f42dfbd1df4eec489aebce2ce55cfbe4302db171c475f14d",
    "dexverse_dish_soap": "3159190e10f8612799543706f0d96df27d6184e7ba51bd46975ca7014b85c646",
    KA020_DISPENSER_ASSET_BUNDLE_KEY: (
        "38e3d924205b4dab12c2e3d93efb607d21491caa8cdc54e5d1787ee1c235fdee"
    ),
    KA020_SPONGE_ASSET_BUNDLE_KEY: (
        "57630e7cc80c7ac314582ed75917b507f54434fae7a82c1d27badd3b2a512aa0"
    ),
    "dv_double_cabinet_44781": "5fad3badbbcdb614084a204fe54d41996b03147daa9f36683a9e9dad9af6c081",
}


def _source_bundle_sha256(directory_name: str) -> str:
    """Reproduce the remote sorted-sha256sum aggregate for one source bundle."""

    directory = B320_ASSET_DIRECTORY / directory_name
    if not directory.is_dir():
        raise FileNotFoundError(f"B320 source asset directory is missing: {directory}")
    derived_outputs = set(_DERIVED_GLBS)
    rows: list[str] = []
    for path in sorted(
        directory.rglob("*"),
        key=lambda candidate: candidate.relative_to(directory).as_posix(),
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if f"{directory_name}/{relative}" in derived_outputs:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  ./{relative}\n")
    if not rows:
        raise ValueError(f"B320 source asset directory is empty: {directory}")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _furniture_texture_paths(directory_name: str) -> tuple[Path, ...]:
    texture_dir = B320_ASSET_DIRECTORY / directory_name / "configuration" / "materials" / "textures"
    return tuple(texture_dir / name for name in _B320_FURNITURE_TEXTURE_NAMES)


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without importing Isaac/Kit."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ka021_float_tuple_matches(value: Any, expected: tuple[float, ...]) -> bool:
    """Compare manifest numeric tuples without importing NumPy/pytest."""

    if not isinstance(value, (list, tuple)) or len(value) != len(expected):
        return False
    try:
        return all(abs(float(actual) - float(wanted)) <= 1.0e-9 for actual, wanted in zip(value, expected))
    except (TypeError, ValueError):
        return False


def validate_b320_ka021_faucet_asset_files() -> tuple[Path, ...]:
    """Validate the separately downloaded BlenderKit faucet derivatives.

    The historical mug/HS005 bundle is intentionally checked by
    :func:`validate_b320_asset_files` and must keep its reviewed aggregate
    hash.  This validator therefore owns only the new faucet closure and does
    not add it to ``B320_SOURCE_BUNDLE_SHA256``.  The source ``.blend`` is
    retained in the user download/output area (the BlenderKit license permits
    a derived task asset but not reselling the raw model); the generated USDs
    are locked by their reviewed hashes here.
    """

    manifest_path = KA021_FAUCET_ASSET_DIRECTORY / "asset_manifest.json"
    attribution_path = KA021_FAUCET_ASSET_DIRECTORY / "ATTRIBUTION.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"KA021 BlenderKit faucet manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"KA021 BlenderKit faucet manifest is unreadable: {manifest_path}"
        ) from exc

    if manifest.get("schema") != "handson.ka021.blenderkit-kitchen-faucet-asset":
        raise ValueError(
            "KA021 BlenderKit faucet manifest schema is invalid: "
            f"{manifest.get('schema')!r}"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("KA021 BlenderKit faucet manifest schema version changed")
    if manifest.get("task_id") != "B320-KA021" or manifest.get(
        "task_name"
    ) != B320_KA021_TASK_NAME:
        raise ValueError("KA021 BlenderKit faucet manifest identity is invalid")
    if manifest.get("asset_bundle") != KA021_FAUCET_ASSET_BUNDLE_KEY:
        raise ValueError("KA021 BlenderKit faucet bundle binding changed")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("KA021 BlenderKit faucet provenance is missing")
    expected_provenance = {
        "platform": "BlenderKit",
        "title": "Kitchen Faucet",
        "creator": KA021_FAUCET_SOURCE_CREATOR,
        "asset_uuid": KA021_FAUCET_SOURCE_UUID,
        "version_id": KA021_FAUCET_SOURCE_VERSION_ID,
        "source_url": KA021_FAUCET_SOURCE_URL,
        "license": KA021_FAUCET_SOURCE_LICENSE,
        "license_url": KA021_FAUCET_SOURCE_LICENSE_URL,
        "source_sha256": KA021_FAUCET_SOURCE_SHA256,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"KA021 BlenderKit faucet provenance binding changed for {key}: "
                f"expected={expected!r} actual={provenance.get(key)!r}"
            )

    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict):
        raise ValueError("KA021 BlenderKit faucet conversion record is missing")
    if conversion.get("split_method") != "authored_object_boundary":
        raise ValueError("KA021 faucet must be split at the authored object boundary")
    if conversion.get("static_objects") != [
        "FaucetBase",
        "FaucetMain",
        "FaucetHead",
        "Buttons",
        "BrassTip",
        "SprayNozzle",
        "HandleBase",
    ] or conversion.get("moving_objects") != ["Handle"]:
        raise ValueError("KA021 BlenderKit faucet object split changed")
    if conversion.get("moving_parent_pivot") != "Empty":
        raise ValueError("KA021 BlenderKit faucet handle pivot binding changed")
    if not _ka021_float_tuple_matches(
        conversion.get("pivot_native_m", ()),
        KA021_FAUCET_NATIVE_HANDLE_PIVOT_M,
    ):
        raise ValueError("KA021 BlenderKit faucet native pivot changed")

    components = manifest.get("runtime_components")
    if not isinstance(components, list):
        raise ValueError("KA021 BlenderKit faucet runtime component list is missing")
    expected_components = {
        "faucet_static.usd": (
            KA021_FAUCET_STATIC_SHA256,
            7,
            {
                "FaucetBase",
                "FaucetMain",
                "FaucetHead",
                "Buttons",
                "BrassTip",
                "SprayNozzle",
                "HandleBase",
            },
        ),
        "faucet_handle.usd": (
            KA021_FAUCET_HANDLE_SHA256,
            1,
            {"Handle"},
        ),
    }
    required: list[Path] = [manifest_path]
    seen: set[str] = set()
    for item in components:
        if not isinstance(item, dict):
            raise ValueError("KA021 BlenderKit faucet component entry is invalid")
        relative = str(item.get("path", ""))
        if relative not in expected_components or Path(relative).name != relative:
            raise ValueError(f"KA021 faucet component path is unsafe or unknown: {relative!r}")
        if relative in seen:
            raise ValueError(f"KA021 faucet component is duplicated: {relative}")
        seen.add(relative)
        expected_hash, expected_mesh_count, expected_objects = expected_components[relative]
        path = KA021_FAUCET_ASSET_DIRECTORY / relative
        if not path.is_file() or path.stat().st_size <= 1024:
            raise FileNotFoundError(f"KA021 BlenderKit faucet component is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash or item.get("sha256") != expected_hash:
            raise ValueError(
                f"KA021 faucet component hash mismatch for {relative}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        if int(item.get("mesh_count", -1)) != expected_mesh_count:
            raise ValueError(f"KA021 faucet mesh count changed for {relative}")
        if set(item.get("objects", ())) != expected_objects:
            raise ValueError(f"KA021 faucet object list changed for {relative}")
        required.append(path)
    if seen != set(expected_components):
        raise ValueError(
            "KA021 BlenderKit faucet runtime components changed: "
            f"expected={sorted(expected_components)} actual={sorted(seen)}"
        )

    coordinate = manifest.get("coordinate_contract")
    if not isinstance(coordinate, dict) or coordinate.get("source_up_axis") != "Z" or coordinate.get(
        "runtime_up_axis"
    ) != "Z":
        raise ValueError("KA021 BlenderKit faucet coordinate contract changed")
    if not _ka021_float_tuple_matches(
        coordinate.get("pivot_native_m", ()),
        KA021_FAUCET_NATIVE_HANDLE_PIVOT_M,
    ):
        raise ValueError("KA021 BlenderKit faucet coordinate pivot changed")

    physics = manifest.get("physics")
    if not isinstance(physics, dict) or physics.get("visual_components_have_physics_api") is not False:
        raise ValueError("KA021 visual derivatives must remain physics-free")
    if physics.get("joint_type") != "revolute" or physics.get(
        "joint_name"
    ) != "RevoluteJoint_faucet_3_right":
        raise ValueError("KA021 faucet joint lineage changed")
    if physics.get("joint_axis_runtime") != KA021_FAUCET_OPEN_AXIS:
        raise ValueError("KA021 faucet joint axis token changed")
    if physics.get("joint_frame") != "yawed_source_y_axis_with_sign_flip":
        raise ValueError("KA021 faucet joint frame contract changed")
    if not _ka021_float_tuple_matches(
        physics.get("joint_frame_orientation_wxyz", ()),
        KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ,
    ):
        raise ValueError("KA021 faucet joint frame orientation changed")
    if not _ka021_float_tuple_matches(
        physics.get("joint_axis_world_w", ()),
        KA021_FAUCET_JOINT_AXIS_WORLD_W,
    ):
        raise ValueError("KA021 faucet world joint axis changed")
    if not _ka021_float_tuple_matches(
        physics.get("initial_open_direction_world_w", ()),
        KA021_FAUCET_INITIAL_OPEN_DIRECTION_W,
    ):
        raise ValueError("KA021 faucet opening direction changed")

    if not attribution_path.is_file() or attribution_path.stat().st_size == 0:
        raise FileNotFoundError(f"KA021 faucet attribution is missing: {attribution_path}")
    attribution = attribution_path.read_text(encoding="utf-8")
    for token in (
        KA021_FAUCET_SOURCE_URL,
        KA021_FAUCET_SOURCE_CREATOR,
        KA021_FAUCET_SOURCE_LICENSE,
    ):
        if token not in attribution:
            raise ValueError(f"KA021 faucet attribution is incomplete: {token}")
    required.append(attribution_path)

    # Isaac's pxr package is not present in the lightweight registration test
    # environment.  When available, additionally prove that the two visual
    # layers do not accidentally carry a rigid body, collider, or joint API.
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:  # pragma: no cover - exercised outside Isaac runtime
        Usd = None
        UsdPhysics = None
    if Usd is not None and UsdPhysics is not None:
        for relative in sorted(expected_components):
            stage = Usd.Stage.Open(str(KA021_FAUCET_ASSET_DIRECTORY / relative))
            if stage is None:
                raise ValueError(f"KA021 faucet USD cannot be opened: {relative}")
            physics_prims = [
                prim
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.IsA(UsdPhysics.Joint)
            ]
            if physics_prims:
                raise ValueError(
                    f"KA021 visual USD unexpectedly contains physics APIs: {relative}"
                )
    return tuple(dict.fromkeys(required))


def _fu001_manifest_path() -> Path:
    """Resolve the selected FU001 manifest, preferring the canonical bundle."""

    # Releases produced before the canonical asset tree was copied can still
    # use the task-dev manifest as the provenance record.  The runtime files
    # themselves are *always* required from ``FU001_RUNTIME_DIRECTORY``; this
    # fallback never substitutes the historical cabinet geometry.
    taskdev_manifest = (
        Path(__file__).resolve().parents[1]
        / "envs/usd/fu001_cabinet"
        / "assets/polyhaven_painted_wooden_cabinet/asset_manifest.json"
    )
    for candidate in (FU001_ASSET_MANIFEST_PATH, taskdev_manifest):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "FU001 visual dependency closure is incomplete; selected Poly Haven "
        f"asset manifest is missing (checked {FU001_ASSET_MANIFEST_PATH} and "
        f"{taskdev_manifest})"
    )


def validate_b320_fu001_asset_files() -> tuple[Path, ...]:
    """Validate the selected Poly Haven FU001 runtime closure.

    This check deliberately remains pure Python: registration and CPU contract
    tests can prove the selected archive/runtime hashes without starting Kit.
    It also fails closed on a missing texture or a manifest that advertises a
    primitive placeholder collision.  The historical 44781 bundle is not
    consulted here.
    """

    manifest_path = _fu001_manifest_path()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"FU001 visual dependency closure has an unreadable manifest: "
            f"{manifest_path}"
        ) from exc
    if manifest.get("task_id") != "B320-FU001" or manifest.get(
        "task_name"
    ) != "b320_synchronized_double_cabinet_open":
        raise ValueError(
            "FU001 selected asset manifest identity is invalid: "
            f"{manifest_path}"
        )
    if manifest.get("schema") != "handson.fu001.polyhaven-painted-cabinet-asset":
        raise ValueError(
            "FU001 selected asset manifest schema is invalid: "
            f"{manifest.get('schema')!r}"
        )

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("FU001 selected asset provenance record is missing")
    if provenance.get("archive_sha256") != FU001_ASSET_ARCHIVE_SHA256:
        raise ValueError(
            "FU001 selected archive hash binding changed: "
            f"expected={FU001_ASSET_ARCHIVE_SHA256} "
            f"actual={provenance.get('archive_sha256')!r}"
        )
    if provenance.get("source_url") != FU001_ASSET_SOURCE_URL:
        raise ValueError("FU001 selected asset source URL binding changed")
    if provenance.get("license") != FU001_ASSET_LICENSE or provenance.get(
        "license_url"
    ) != FU001_ASSET_LICENSE_URL:
        raise ValueError("FU001 selected asset license binding changed")

    # The downloaded archive is retained beside the task-dev provenance.  A
    # hash mismatch is always fatal; this prevents a same-named replacement
    # from silently passing the selected-asset gate.
    archive = FU001_ASSET_SOURCE_ARCHIVE_PATH
    if not archive.is_file() or archive.stat().st_size == 0:
        raise FileNotFoundError(
            "FU001 visual dependency closure is incomplete; selected archive "
            f"is missing: {archive}"
        )
    archive_hash = _sha256_file(archive)
    if archive_hash != FU001_ASSET_ARCHIVE_SHA256:
        raise ValueError(
            "FU001 selected archive SHA-256 mismatch: "
            f"expected={FU001_ASSET_ARCHIVE_SHA256} actual={archive_hash}"
        )

    components = manifest.get("runtime_components")
    if not isinstance(components, list):
        raise ValueError("FU001 selected runtime component list is missing")
    component_by_path: dict[str, dict[str, Any]] = {}
    required_paths: list[Path] = [manifest_path, archive]
    for item in components:
        if not isinstance(item, dict):
            raise ValueError("FU001 runtime component entry is not an object")
        relative = str(item.get("path", ""))
        expected_hash = str(item.get("sha256", ""))
        if not relative or "/" in relative or "\\" in relative:
            raise ValueError(f"FU001 runtime component path is unsafe: {relative!r}")
        path = FU001_RUNTIME_DIRECTORY / relative
        if not path.is_file() or path.stat().st_size <= 1024:
            raise FileNotFoundError(
                "FU001 visual dependency closure is incomplete; runtime "
                f"component is missing: {path}"
            )
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"FU001 runtime component hash mismatch for {relative}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        if item.get("collision") != (
            "source_exact_mesh_topology_then_physx_convex_decomposition"
        ):
            raise ValueError(
                "FU001 runtime component collision lineage is not source-derived: "
                f"{relative}"
            )
        component_by_path[relative] = item
        required_paths.append(path)
    expected_components = {
        "painted_cabinet_body.usd",
        "painted_cabinet_door_left.usd",
        "painted_cabinet_door_right.usd",
    }
    if set(component_by_path) != expected_components:
        raise ValueError(
            "FU001 selected runtime components changed: "
            f"expected={sorted(expected_components)} actual={sorted(component_by_path)}"
        )

    collision = manifest.get("collision")
    if not isinstance(collision, dict) or collision.get("placeholder_primitives"):
        raise ValueError(
            "FU001 selected asset advertises a primitive placeholder collision"
        )
    if collision.get("approximation") != FU001_SOURCE_MESH_COLLISION_APPROXIMATION:
        raise ValueError("FU001 selected collision approximation is not convexDecomposition")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("FU001 selected source geometry record is missing")
    if source.get("hinge_axis_source") != FU001_HINGE_AXIS_SOURCE or source.get(
        "hinge_axis_runtime"
    ) != f"{FU001_HINGE_AXIS_RUNTIME} after +90 degrees about X":
        raise ValueError("FU001 selected hinge-axis mapping changed")
    scene_frame = manifest.get("scene_frame")
    if not isinstance(scene_frame, dict) or (
        scene_frame.get("converter_source_to_runtime")
        != "(x,y,z) -> (x,-z,y)"
        or scene_frame.get("extra_rotation") != "Rz(-90 degrees)"
        or scene_frame.get("source_to_world") != "(x,y,z) -> (-z,-x,y)"
        or scene_frame.get("orientation_wxyz")
        != list(FU001_RUNTIME_ORIENTATION_WXYZ)
    ):
        raise ValueError("FU001 selected scene-frame mapping changed")
    source_members = source.get("members")
    if not isinstance(source_members, list):
        raise ValueError("FU001 selected source member list is missing")
    # Keep the runtime texture closure byte-identical to the reviewed source
    # textures.  Material references in the converted USD are relative to this
    # directory, so an absent or substituted image would be a visual failure.
    for item in source_members:
        if not isinstance(item, dict):
            raise ValueError("FU001 source member entry is not an object")
        relative = str(item.get("path", ""))
        if not relative.startswith("textures/"):
            continue
        filename = Path(relative).name
        runtime_texture = FU001_RUNTIME_DIRECTORY / "textures" / filename
        expected_hash = str(item.get("sha256", ""))
        if not runtime_texture.is_file() or runtime_texture.stat().st_size == 0:
            raise FileNotFoundError(
                "FU001 visual dependency closure is incomplete; texture is missing: "
                f"{runtime_texture}"
            )
        actual_hash = _sha256_file(runtime_texture)
        if actual_hash != expected_hash:
            raise ValueError(
                f"FU001 runtime texture hash mismatch for {filename}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        required_paths.append(runtime_texture)

    if FU001_ASSET_LICENSE_PATH.is_file():
        required_paths.append(FU001_ASSET_LICENSE_PATH)
    return tuple(dict.fromkeys(required_paths))


def validate_b320_visual_dependencies(task_name: str | None = None) -> tuple[Path, ...]:
    """Require the complete material closure for the integrated cabinet."""

    directory_by_task = {
        "b320_synchronized_double_cabinet_open": "dv_double_cabinet_44781",
    }
    if task_name == "b320_synchronized_double_cabinet_open":
        return validate_b320_fu001_asset_files()
    if task_name is not None and task_name not in directory_by_task:
        raise ValueError(
            "No B320 furniture visual-dependency manifest exists for "
            f"task {task_name!r}; refuse to infer or borrow another bundle."
        )
    directory_names = (
        (directory_by_task[task_name],)
        if task_name in directory_by_task
        else ("dv_double_cabinet_44781",)
        if task_name is None
        else ()
    )
    paths = tuple(
        path
        for directory_name in directory_names
        for path in _furniture_texture_paths(directory_name)
    )
    missing = []
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(path)
        except OSError:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "B320 furniture visual dependency closure is incomplete; "
            "obtain the authoritative release textures before Isaac spawn: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def validate_b320_ka005_2_asset_files() -> tuple[Path, ...]:
    """Validate the complete clean-room Project Boxy/Rotary Knob closure."""

    manifest_path = B320_KA005_2_ASSET_DIRECTORY / "asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"B320-KA005-2 asset manifest is missing: {manifest_path}"
        )
    actual_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_manifest_hash != B320_KA005_2_ASSET_MANIFEST_SHA256:
        raise ValueError(
            "B320-KA005-2 asset manifest hash mismatch: "
            f"expected={B320_KA005_2_ASSET_MANIFEST_SHA256} "
            f"actual={actual_manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task_id") != "B320-KA005-2" or manifest.get(
        "task_name"
    ) != B320_KA005_2_TASK_NAME:
        raise ValueError("B320-KA005-2 asset manifest identity is invalid")
    expected_files = {
        **manifest.get("generated_files", {}),
        **manifest.get("runtime_dependencies", {}),
    }
    required = [manifest_path]
    for relative, expected_hash in sorted(expected_files.items()):
        path = B320_KA005_2_ASSET_DIRECTORY / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"B320-KA005-2 asset dependency is missing: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"B320-KA005-2 asset hash mismatch for {relative}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        required.append(path)
    return tuple(required)


def validate_b320_asset_files(
    *,
    require_visual_dependencies: bool = False,
) -> tuple[Path, ...]:
    """Fail closed on source/hash/slot errors and optionally visual dependencies.

    The lightweight default keeps registration/CPU tests independent of the
    large texture payload.  Furniture scene spawns call
    :func:`validate_b320_visual_dependencies` for their selected fixture so a
    missing USD material closure cannot reach Hydra and masquerade as a
    runtime crash.
    """

    required = [B320_ASSET_DIRECTORY / relative for relative in _REQUIRED_SOURCES]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "B320 asset bundle is incomplete: " + ", ".join(map(str, missing))
        )
    for directory_name, expected_hash in B320_SOURCE_BUNDLE_SHA256.items():
        actual_hash = _source_bundle_sha256(directory_name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"B320 source bundle hash mismatch for {directory_name}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
    for derived_relative, (source_relative, expected_hash) in _DERIVED_GLBS.items():
        source = B320_ASSET_DIRECTORY / source_relative
        derived = B320_ASSET_DIRECTORY / derived_relative
        if not derived.is_file() or derived.stat().st_size < 1024:
            raise FileNotFoundError(f"B320 derived USD is missing: {derived}")
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"B320 derived USD source hash mismatch for {source}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
    expected_slots = {
        "dv_double_cabinet_44781/meta.json": {
            "target_joint_revolute": "joint_1",
            "joint_11": "joint_0",
        },
    }
    for relative, expected in expected_slots.items():
        metadata = json.loads((B320_ASSET_DIRECTORY / relative).read_text())
        actual = {
            str(slot["slot_joint"]): str(slot["original_joint"])
            for slot in metadata["slots"]
            if not bool(slot["is_dummy"]) and str(slot["slot_joint"]) in expected
        }
        if actual != expected:
            raise ValueError(
                f"B320 furniture slot contract changed for {relative}: "
                f"expected={expected!r} actual={actual!r}"
            )
    if require_visual_dependencies:
        validate_b320_visual_dependencies()
    return tuple(required)


@dataclass
class B320SpawnResult:
    body_paths: dict[str, str] = field(default_factory=dict)
    articulation_root_paths: tuple[str, ...] = ()
    static_paths: tuple[str, ...] = ()
    dof_names: tuple[str, ...] = ()
    collider_counts: dict[str, int] = field(default_factory=dict)
    # JSON-safe, typed spawn-boundary evidence.  Most B320 tasks leave this
    # empty; dedicated runtimes use it instead of attaching dynamic attributes
    # that disappear under dataclass serialization.
    evidence: dict[str, Any] = field(default_factory=dict)


def _wxyz(value: tuple[float, float, float, float]) -> Any:
    from pxr import Gf

    w, x, y, z = (float(component) for component in value)
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))


def _set_xform(
    prim: Any,
    *,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    scale: tuple[float, float, float] | None = None,
) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    orient_op = xform.AddOrientOp()
    quaternion = _wxyz(orientation_wxyz)
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        quaternion = Gf.Quatf(
            float(quaternion.GetReal()),
            Gf.Vec3f(*map(float, quaternion.GetImaginary())),
        )
    orient_op.Set(quaternion)
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3d(*map(float, scale)))


def _descendants(stage: Any, root_path: str) -> list[Any]:
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    return list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))


def _bind_material(
    prim_path: str,
    *,
    sim_utils: Any,
    static_friction: float = 1.2,
    dynamic_friction: float = 1.0,
    friction_combine_mode: str = "max",
) -> None:
    material_path = f"{prim_path}/physicsMaterial"
    cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
        friction_combine_mode=friction_combine_mode,
        restitution_combine_mode="min",
    )
    cfg.func(material_path, cfg)
    sim_utils.bind_physics_material(
        prim_path,
        material_path,
        stronger_than_descendants=True,
    )


def _configure_dynamic_source(
    stage: Any,
    sim_utils: Any,
    object_spec: B320ObjectSpec,
) -> tuple[str, int]:
    from isaaclab.sim import schemas
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    rigid_prims = [
        prim
        for prim in _descendants(stage, object_spec.prim_path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(rigid_prims) > 1:
        raise RuntimeError(
            f"B320 rigid object {object_spec.name} composed multiple rigid roots: "
            + ", ".join(str(prim.GetPath()) for prim in rigid_prims)
        )
    body_prim = rigid_prims[0] if rigid_prims else stage.GetPrimAtPath(object_spec.prim_path)
    if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(body_prim)

    colliders = [
        prim
        for prim in _descendants(stage, object_spec.prim_path)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not colliders:
        for prim in _descendants(stage, object_spec.prim_path):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr().Set(0.002)
            physx_collision.CreateRestOffsetAttr().Set(0.0)
            colliders.append(prim)
    if len(colliders) < object_spec.expected_colliders:
        raise RuntimeError(
            f"B320 collider contract failed for {object_spec.name}: "
            f"expected>={object_spec.expected_colliders} actual={len(colliders)}"
        )

    body_path = str(body_prim.GetPath())
    schemas.define_rigid_body_properties(
        prim_path=body_path,
        cfg=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=float(object_spec.linear_damping),
            angular_damping=float(object_spec.angular_damping),
            max_depenetration_velocity=0.8,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=float(object_spec.sleep_threshold),
            stabilization_threshold=0.002,
        ),
    )
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        body_prim
    ).CreateEnableSpeculativeCCDAttr().Set(True)
    schemas.define_mass_properties(
        prim_path=body_path,
        cfg=sim_utils.MassPropertiesCfg(mass=float(object_spec.mass_kg)),
    )
    _bind_material(
        object_spec.prim_path,
        sim_utils=sim_utils,
        static_friction=float(object_spec.static_friction),
        dynamic_friction=float(object_spec.dynamic_friction),
    )
    body_prim.SetCustomDataByKey("handson:formalTruth", True)
    body_prim.SetCustomDataByKey("handson:logicalName", object_spec.name)
    return body_path, len(colliders)


def _replace_ka020_sponge_collision(
    stage: Any,
    sim_utils: Any,
    *,
    sponge_path: str,
) -> str:
    """Replace source-mesh cooking with one measured sponge-only box.

    The downloaded source contains a sponge inside a holder.  The referenced
    render USD has already removed every holder face and recentered only the
    five sponge shells on a Y-up support frame.  Use those isolated, measured
    post-scale dimensions for the physical hull as well, so no invisible bowl
    or source-wide convex hull can be lifted with the sponge.
    """

    from pxr import PhysxSchema, UsdPhysics

    for prim in _descendants(stage, sponge_path):
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    # The rigid-body root carries the uniform visual scale, which USD also
    # applies to child collision primitives. Author the box in unscaled root
    # coordinates so its composed PhysX dimensions remain measured metres.
    collider_size_local = tuple(
        value / KA020_SPONGE_VISUAL_SCALE
        for value in KA020_SPONGE_COLLIDER_SIZE_M
    )
    collider_center_local = tuple(
        value / KA020_SPONGE_VISUAL_SCALE
        for value in KA020_SPONGE_COLLIDER_CENTER_M
    )
    collision_path = _invisible_box_collider(
        stage,
        sim_utils,
        path=f"{sponge_path}/selected_geometry_collision",
        size=collider_size_local,
        translation=collider_center_local,
    )
    collision = stage.GetPrimAtPath(collision_path)
    collision.SetCustomDataByKey("handson:logicalName", "dish_sponge")
    collision.SetCustomDataByKey("handson:collisionOnly", True)
    collision.SetCustomDataByKey(
        "handson:derivedSizeM",
        KA020_SPONGE_COLLIDER_SIZE_M,
    )
    collision.SetCustomDataByKey("handson:holderGeometryPresent", False)
    return collision_path


def _spawn_worktable(stage: Any, sim_utils: Any, *, task_name: str) -> str:
    from pxr import UsdGeom, UsdPhysics

    d08_aligned = task_name == "t10_lift_carton"
    bc017_study = task_name == B320_BC017_TASK_NAME
    table_rigid_kwargs = (
        {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            )
        }
        if d08_aligned
        else {}
    )
    table_cfg = sim_utils.CuboidCfg(
        size=(
            (0.80, 1.60, 0.060)
            if d08_aligned
            else (0.92, 1.52, 0.060)
            if bc017_study
            else (0.84, 1.28, 0.060)
        ),
        **table_rigid_kwargs,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.00 if d08_aligned or bc017_study else 0.90,
            dynamic_friction=0.80,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(
                B320_BC017_STUDY_TABLE_DIFFUSE
                if bc017_study
                else B320_D08_TABLE_DIFFUSE
                if d08_aligned
                else (0.38, 0.30, 0.22)
            ),
            roughness=0.88 if bc017_study else 0.92 if d08_aligned else 0.78,
        ),
    )
    table_cfg.func(
        B320_TABLE_PATH,
        table_cfg,
        translation=(-0.15, 0.0, B320_TABLE_TOP_Z - 0.030),
    )
    prim = stage.GetPrimAtPath(B320_TABLE_PATH)
    prim.SetCustomDataByKey("handson:fixture", "worktable")
    prim.SetCustomDataByKey("handson:d08Aligned", bool(d08_aligned))
    prim.SetCustomDataByKey("handson:bc017StudySupport", bool(bc017_study))
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    if d08_aligned:
        leg_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=B320_D08_LEG_DIFFUSE,
            roughness=0.72,
            metallic=0.18,
        )
        for name, translation in (
            ("front_left", (0.17, 0.72, 0.43)),
            ("front_right", (0.17, -0.72, 0.43)),
            ("rear_left", (-0.47, 0.72, 0.43)),
            ("rear_right", (-0.47, -0.72, 0.43)),
        ):
            leg_path = f"{B320_TABLE_PATH}_{name}_leg"
            leg_cfg = sim_utils.CylinderCfg(
                radius=0.04,
                height=0.86,
                visual_material=leg_material,
            )
            leg_cfg.func(leg_path, leg_cfg, translation=translation)
            stage.GetPrimAtPath(leg_path).SetCustomDataByKey(
                "handson:d08AlignedVisual", True
            )
    return B320_TABLE_PATH


def _spawn_goal_marker(
    stage: Any,
    sim_utils: Any,
    *,
    name: str,
    translation: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    highlight: bool = False,
) -> str:
    path = f"{B320_ASSET_ROOT_PATH}/{name}"
    fill_material_kwargs: dict[str, Any] = {
        "diffuse_color": color,
        "opacity": 0.72 if highlight else 0.45,
        "roughness": 0.38 if highlight else 0.85,
    }
    if highlight:
        fill_material_kwargs["emissive_color"] = tuple(
            min(1.0, 0.42 * channel) for channel in color
        )
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(**fill_material_kwargs),
    )
    cfg.func(path, cfg, translation=translation)
    marker_prim = stage.GetPrimAtPath(path)
    marker_prim.SetCustomDataByKey("handson:goalMarker", True)
    marker_prim.SetCustomDataByKey("handson:highlighted", bool(highlight))
    marker_prim.SetCustomDataByKey(
        "handson:goalZoneSizeM",
        tuple(float(value) for value in size),
    )

    if highlight:
        # Outline the exact success-predicate boundary.  The bars are visual
        # Cuboids only (no collision/rigid properties) and sit just above the
        # translucent fill so the target remains legible against the BC014
        # kitchen island from both the operator and canonical cameras.
        border_width = min(0.008, 0.16 * min(size[0], size[1]))
        border_height = max(0.004, size[2])
        border_z = translation[2] + 0.5 * (size[2] + border_height)
        border_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            emissive_color=tuple(min(1.0, 0.85 * channel) for channel in color),
            opacity=1.0,
            roughness=0.24,
        )
        border_specs = (
            (
                "front",
                (size[0], border_width, border_height),
                (
                    translation[0],
                    translation[1] + 0.5 * (size[1] - border_width),
                    border_z,
                ),
            ),
            (
                "rear",
                (size[0], border_width, border_height),
                (
                    translation[0],
                    translation[1] - 0.5 * (size[1] - border_width),
                    border_z,
                ),
            ),
            (
                "left",
                (border_width, size[1], border_height),
                (
                    translation[0] - 0.5 * (size[0] - border_width),
                    translation[1],
                    border_z,
                ),
            ),
            (
                "right",
                (border_width, size[1], border_height),
                (
                    translation[0] + 0.5 * (size[0] - border_width),
                    translation[1],
                    border_z,
                ),
            ),
        )
        for edge_name, edge_size, edge_translation in border_specs:
            edge_path = f"{path}_highlight_{edge_name}"
            edge_cfg = sim_utils.CuboidCfg(
                size=edge_size,
                visual_material=border_material,
            )
            edge_cfg.func(
                edge_path,
                edge_cfg,
                translation=edge_translation,
            )
            edge_prim = stage.GetPrimAtPath(edge_path)
            edge_prim.SetCustomDataByKey("handson:goalHighlight", True)
            edge_prim.SetCustomDataByKey("handson:visualOnly", True)
    return path


def _spawn_bc014_flow_proxy(
    stage: Any,
    sim_utils: Any,
    *,
    role: str,
) -> str:
    """Author a render-only D09-style water stream for one shaker.

    BC014 intentionally has no salt/pepper particle simulation.  This proxy
    reuses the established D09 water-stream visual language: a translucent
    blue cylinder whose live transform is driven by the shaker-cap ray.  No
    rigid or collision properties are authored, so it cannot affect PhysX.
    """

    from pxr import UsdGeom

    path = BC014_FLOW_PROXY_PATHS[role]
    cfg = sim_utils.CylinderCfg(
        radius=0.006,
        height=1.0,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.08, 0.55, 0.98),
            emissive_color=(0.02, 0.12, 0.28),
            opacity=0.72,
            roughness=0.20,
        ),
    )
    cfg.func(
        path,
        cfg,
        translation=(0.0, 0.0, B320_TABLE_TOP_Z - 1.0),
    )
    prim = stage.GetPrimAtPath(path)
    prim.SetCustomDataByKey("handson:flowProxy", "water_stream")
    prim.SetCustomDataByKey("handson:seasoningRole", role)
    prim.SetCustomDataByKey("handson:visualOnly", True)
    UsdGeom.Imageable(prim).MakeInvisible()
    return path


def _spawn_ka021_flow_proxy(stage: Any, sim_utils: Any) -> str:
    """Author the render-only stream that responds to the faucet joint."""

    from pxr import UsdGeom

    cfg = sim_utils.CylinderCfg(
        radius=0.0055,
        height=1.0,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.06, 0.55, 1.00),
            emissive_color=(0.02, 0.14, 0.34),
            opacity=0.76,
            roughness=0.16,
        ),
    )
    cfg.func(
        KA021_FLOW_PROXY_PATH,
        cfg,
        translation=(0.0, 0.0, B320_TABLE_TOP_Z - 1.0),
    )
    prim = stage.GetPrimAtPath(KA021_FLOW_PROXY_PATH)
    prim.SetCustomDataByKey("handson:flowProxy", "faucet_water_stream")
    prim.SetCustomDataByKey("handson:drivenBy", "RevoluteJoint_faucet_3_right")
    prim.SetCustomDataByKey("handson:visualOnly", True)
    UsdGeom.Imageable(prim).MakeInvisible()
    return KA021_FLOW_PROXY_PATH


def _spawn_direct_object(
    stage: Any,
    sim_utils: Any,
    object_spec: B320ObjectSpec,
    *,
    bc011_pen_variant_index: int | None = None,
) -> tuple[str, int]:
    relative_usd = (
        BC011_PEN_VARIANTS[int(bc011_pen_variant_index or 0)]
        if object_spec.name == "pen"
        else object_spec.relative_usd
    )
    source_path = B320_ASSET_DIRECTORY / relative_usd
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(source_path.resolve()),
        scale=object_spec.scale,
        semantic_tags=[
            ("class", object_spec.name),
            ("instance", object_spec.name),
        ],
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=float(object_spec.contact_offset),
            rest_offset=0.0,
        ),
    )
    cfg.func(
        object_spec.prim_path,
        cfg,
        translation=object_spec.home_translation,
        orientation=quat_wxyz_to_isaaclab(object_spec.home_orientation_wxyz),
    )
    sim_utils.make_uninstanceable(object_spec.prim_path, stage=stage)
    sim_utils.update_stage()
    root = stage.GetPrimAtPath(object_spec.prim_path)
    root.SetCustomDataByKey("handson:sourceAsset", relative_usd)
    root.SetCustomDataByKey(
        "handson:sourceSha256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    if object_spec.name == "pen":
        root.SetCustomDataByKey("handson:penVariant", relative_usd)
        root.SetCustomDataByKey(
            "handson:penTargetLengthM", BC011_TARGET_PEN_LENGTH_M
        )
        root.SetCustomDataByKey(
            "handson:referenceIndexLengthM",
            BC011_WUJI1_INDEX_REFERENCE_LENGTH_M,
        )
        print(
            "[HandsOn/IsaacLab] BC011 hand-referenced pen size: "
            f"variant={relative_usd}, "
            f"source_length={BC011_SOURCE_PEN_LENGTH_M:.3f}m, "
            f"target_length={BC011_TARGET_PEN_LENGTH_M:.3f}m, "
            f"index_reference={BC011_WUJI1_INDEX_REFERENCE_LENGTH_M:.3f}m, "
            f"scale={object_spec.scale}",
            flush=True,
        )
    return _configure_dynamic_source(stage, sim_utils, object_spec)


def _spawn_articulated_source(
    stage: Any,
    sim_utils: Any,
    object_spec: B320ObjectSpec,
) -> tuple[str, int]:
    """Spawn an authored articulation without collapsing its rigid links."""

    if not object_spec.source_path.is_file():
        raise FileNotFoundError(object_spec.source_path)
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(object_spec.source_path.resolve()),
        scale=object_spec.scale,
        semantic_tags=[
            ("class", object_spec.name),
            ("instance", object_spec.name),
        ],
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
    )
    cfg.func(
        object_spec.prim_path,
        cfg,
        translation=object_spec.home_translation,
        orientation=quat_wxyz_to_isaaclab(object_spec.home_orientation_wxyz),
    )
    sim_utils.make_uninstanceable(object_spec.prim_path, stage=stage)
    sim_utils.update_stage()
    root = stage.GetPrimAtPath(object_spec.prim_path)
    root.SetCustomDataByKey("handson:sourceAsset", object_spec.relative_usd)
    root.SetCustomDataByKey(
        "handson:sourceSha256",
        hashlib.sha256(object_spec.source_path.read_bytes()).hexdigest(),
    )
    from pxr import UsdPhysics

    rigid_count = sum(
        prim.HasAPI(UsdPhysics.RigidBodyAPI)
        for prim in _descendants(stage, object_spec.prim_path)
    )
    collider_count = sum(
        prim.HasAPI(UsdPhysics.CollisionAPI)
        for prim in _descendants(stage, object_spec.prim_path)
    )
    if rigid_count < 2:
        raise RuntimeError(
            f"B320 articulated source {object_spec.name} has only "
            f"{rigid_count} rigid bodies"
        )
    if collider_count < object_spec.expected_colliders:
        raise RuntimeError(
            f"B320 collider contract failed for {object_spec.name}: "
            f"expected>={object_spec.expected_colliders} actual={collider_count}"
        )
    _bind_material(object_spec.prim_path, sim_utils=sim_utils)
    return object_spec.prim_path, int(collider_count)


def _reference_visual(
    stage: Any,
    *,
    path: str,
    source: Path,
    source_prim: str,
    scale: float,
    instanceable: bool = True,
) -> str:
    from pxr import Gf, Sdf, UsdGeom

    visual = UsdGeom.Xform.Define(stage, path).GetPrim()
    visual.GetReferences().AddReference(
        Sdf.Reference(assetPath=str(source.resolve()), primPath=source_prim)
    )
    xformable = UsdGeom.Xformable(visual)
    # Referencing a component root (``/World``) can compose the source root's
    # translate/orient/scale ops directly onto this prim.  Do not attempt to
    # author a duplicate scale op in that case; duplicate names are rejected
    # by USD before the caller gets a chance to normalize the transform.
    scale_ops = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if scale_ops:
        scale_ops[-1].Set(Gf.Vec3d(scale, scale, scale))
    else:
        xformable.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    # Cabinet components carry their own source-derived CollisionAPI meshes.
    # Leave that composition uninstanceable so the runtime can inspect and
    # tune the referenced colliders per body; the historical callers retain
    # the old instanceable default.
    visual.SetInstanceable(bool(instanceable))
    return path


def _reference_visual_preserving_source_root(
    stage: Any,
    *,
    path: str,
    source: Path,
    source_prim: str,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    scale: float,
) -> str:
    """Reference a visual below a transformed wrapper without clobbering its root.

    Some Isaac 5.1 glTF conversions carry a required axis transform on their
    default prim. Referencing that prim directly onto the task wrapper and then
    authoring task xform ops replaces the source transform, which laid KA020's
    dispenser on its side. A child reference preserves the complete source
    hierarchy while the parent owns task placement, orientation, and scale.
    """

    from pxr import Sdf, UsdGeom

    wrapper = UsdGeom.Xform.Define(stage, path).GetPrim()
    _set_xform(
        wrapper,
        translation=translation,
        orientation_wxyz=orientation_wxyz,
        scale=(scale, scale, scale),
    )
    asset_path = f"{path}/asset"
    asset = UsdGeom.Xform.Define(stage, asset_path).GetPrim()
    asset.GetReferences().AddReference(
        Sdf.Reference(assetPath=str(source.resolve()), primPath=source_prim)
    )
    asset.SetInstanceable(False)
    return asset_path


def _invisible_box_collider(
    stage: Any,
    sim_utils: Any,
    *,
    path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] | None = None,
) -> str:
    """Create an invisible cuboid collider in the caller's local frame.

    Scene code keeps quaternions in the repository-wide ``(w, x, y, z)``
    convention.  IsaacLab 3's shape spawners switched their public argument
    to ``(x, y, z, w)`` (IsaacLab 2 still accepts ``(w, x, y, z)``), so the
    compatibility conversion belongs at this boundary rather than at each
    call site.
    """
    from pxr import UsdGeom

    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.1,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
    )
    if orientation is None:
        cfg.func(path, cfg, translation=translation)
    else:
        cfg.func(
            path,
            cfg,
            translation=translation,
            orientation=quat_wxyz_to_isaaclab(orientation),
        )
    UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    return path


def _invisible_cylinder_collider(
    stage: Any,
    sim_utils: Any,
    *,
    path: str,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
) -> str:
    from pxr import UsdGeom

    cfg = sim_utils.CylinderCfg(
        radius=float(radius),
        height=float(height),
        axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0015,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.25,
            dynamic_friction=1.05,
            restitution=0.0,
        ),
    )
    cfg.func(path, cfg, translation=translation)
    UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    return path


def _replace_ka021_mug_collision(
    stage: Any,
    sim_utils: Any,
    *,
    mug_path: str,
    source_scale: float,
) -> tuple[str, ...]:
    """Leave the mug cavity open while retaining a stable bottom support.

    The reviewed mug USD historically used one solid cylinder.  That shape is
    useful for resting the object but blocks a hand/finger from entering the
    visible opening.  Disable that cylinder and author four thin wall boxes in
    the mug's source frame.  Because the mug root carries the 0.006 runtime
    scale, the dimensions below are expressed in source units and become
    approximately 81 mm outside diameter / 3 mm wall thickness in the scene.
    The authored recessed foot remains enabled as the fifth collider.
    """

    from pxr import UsdPhysics

    old_body = stage.GetPrimAtPath(f"{mug_path}/KA021MugCollision")
    if old_body.IsValid() and old_body.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(old_body).CreateCollisionEnabledAttr().Set(False)
        old_body.SetCustomDataByKey("handson:disabledForOpenCavity", True)

    # The reviewed mug asset carries a separate recessed flat foot.  Keep it
    # explicitly enabled after the source cylinder is disabled; otherwise an
    # importer/instance override can leave the open-cavity walls with no
    # downward-facing support and the mug will fall through the park pad.
    foot = stage.GetPrimAtPath(f"{mug_path}/KA021MugCollisionFoot")
    if not foot.IsValid() or not foot.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(
            "KA021 mug source is missing its recessed foot collision: "
            f"{mug_path}/KA021MugCollisionFoot"
        )
    UsdPhysics.CollisionAPI(foot).CreateCollisionEnabledAttr().Set(True)
    foot.SetCustomDataByKey("handson:logicalName", "mug_recessed_foot")
    foot.SetCustomDataByKey("handson:openCavity", True)

    scale = max(float(source_scale), 1.0e-6)
    outer = 0.0405 / scale
    wall = 0.0030 / scale
    height = 0.0864 / scale
    center_z = 0.0432 / scale
    center_xy = outer - 0.5 * wall
    inner_side = 2.0 * max(outer - wall, wall)
    long_size = (2.0 * outer, wall, height)
    short_size = (wall, inner_side, height)
    colliders: list[str] = []
    for name, size, translation in (
        ("north", long_size, (0.0, center_xy, center_z)),
        ("south", long_size, (0.0, -center_xy, center_z)),
        ("east", short_size, (center_xy, 0.0, center_z)),
        ("west", short_size, (-center_xy, 0.0, center_z)),
    ):
        path = f"{mug_path}/KA021MugWallCollision_{name}"
        _invisible_box_collider(
            stage,
            sim_utils,
            path=path,
            size=size,
            translation=translation,
        )
        prim = stage.GetPrimAtPath(path)
        prim.SetCustomDataByKey("handson:logicalName", "mug_open_wall")
        prim.SetCustomDataByKey("handson:openCavity", True)
        colliders.append(path)
    handle_radius = KA021_MUG_HANDLE_COLLIDER_RADIUS_M / scale
    for index, (start, end) in enumerate(KA021_MUG_HANDLE_CENTERLINE_SOURCE_M):
        path = f"{mug_path}/KA021MugHandleCollision_{index}"
        _invisible_capsule_segment_collider(
            stage,
            sim_utils,
            path=path,
            start=start,
            end=end,
            radius=handle_radius,
        )
        prim = stage.GetPrimAtPath(path)
        prim.SetCustomDataByKey("handson:logicalName", "mug_handle")
        prim.SetCustomDataByKey("handson:openCavity", True)
        colliders.append(path)
    return tuple(colliders)


def _invisible_capsule_segment_collider(
    stage: Any,
    sim_utils: Any,
    *,
    path: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> str:
    """Author a Z-axis capsule rotated onto a caller-frame line segment.

    ``start``/``end`` are intentionally interpreted in the parent prim's
    frame.  Most callers use world coordinates, while fixed-body fixtures such
    as KA021 pass coordinates local to an identity-oriented body root.  The
    helper records the supplied values verbatim; the caller can add an
    explicitly transformed world-frame record when diagnostics need it.
    """

    from pxr import UsdGeom

    delta = tuple(float(end[index] - start[index]) for index in range(3))
    length = math.sqrt(sum(component * component for component in delta))
    if length <= 1.0e-9:
        raise ValueError(f"capsule segment is degenerate: {start!r} -> {end!r}")
    direction = tuple(component / length for component in delta)
    # Shortest-arc quaternion mapping local +Z onto the segment direction.
    # For the antiparallel singularity, rotate 180 degrees around local +X.
    quaternion_norm = math.sqrt(max(2.0 * (1.0 + direction[2]), 0.0))
    if quaternion_norm <= 1.0e-9:
        orientation_wxyz = (0.0, 1.0, 0.0, 0.0)
    else:
        orientation_wxyz = (
            (1.0 + direction[2]) / quaternion_norm,
            -direction[1] / quaternion_norm,
            direction[0] / quaternion_norm,
            0.0,
        )
    midpoint = tuple(
        0.5 * (float(start[index]) + float(end[index])) for index in range(3)
    )
    cfg = sim_utils.CapsuleCfg(
        radius=float(radius),
        height=float(length),
        axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0015,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.25,
            dynamic_friction=1.05,
            restitution=0.0,
        ),
    )
    cfg.func(
        path,
        cfg,
        translation=midpoint,
        orientation=quat_wxyz_to_isaaclab(orientation_wxyz),
    )
    prim = stage.GetPrimAtPath(path)
    UsdGeom.Imageable(prim).MakeInvisible()
    prim.SetCustomDataByKey("handson:segmentStartW", start)
    prim.SetCustomDataByKey("handson:segmentEndW", end)
    prim.SetCustomDataByKey("handson:segmentRadiusM", float(radius))
    return path


def _apply_dynamic_body(
    stage: Any,
    sim_utils: Any,
    *,
    path: str,
    mass_kg: float,
    kinematic: bool = False,
) -> None:
    from isaaclab.sim import schemas
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    schemas.define_rigid_body_properties(
        prim_path=path,
        cfg=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic,
            disable_gravity=kinematic,
            linear_damping=0.12,
            angular_damping=0.75,
            max_depenetration_velocity=0.6,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    )
    if not kinematic:
        from pxr import PhysxSchema

        PhysxSchema.PhysxRigidBodyAPI.Apply(
            prim
        ).CreateEnableSpeculativeCCDAttr().Set(True)
    schemas.define_mass_properties(
        prim_path=path,
        cfg=sim_utils.MassPropertiesCfg(mass=float(mass_kg)),
    )


def _fix_articulation_base_to_world(stage: Any, base_path: str) -> str:
    """Use PhysX's fixed-base articulation convention for Isaac Sim 5.1."""

    from pxr import PhysxSchema, Sdf, UsdPhysics

    base = stage.GetPrimAtPath(base_path)
    if base.HasAPI(UsdPhysics.ArticulationRootAPI):
        base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    rigid = UsdPhysics.RigidBodyAPI(base)
    rigid.CreateKinematicEnabledAttr().Set(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(base).CreateDisableGravityAttr().Set(True)

    fixed_path = f"{base_path}/world_fixed_joint"
    fixed = UsdPhysics.FixedJoint.Define(stage, fixed_path)
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
    root = fixed.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(
        root
    ).CreateEnabledSelfCollisionsAttr().Set(False)
    return fixed_path


def _author_joint(
    stage: Any,
    *,
    path: str,
    joint_type: str,
    body0: str,
    body1: str,
    axis: str,
    local_pos0: tuple[float, float, float],
    local_rot1_wxyz: tuple[float, float, float, float],
    lower: float,
    upper: float,
    damping: float,
    max_force: float,
    stiffness: float = 0.0,
    local_rot0_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> str:
    from pxr import Gf, UsdPhysics

    joint_schema = (
        UsdPhysics.PrismaticJoint.Define(stage, path)
        if joint_type == "prismatic"
        else UsdPhysics.RevoluteJoint.Define(stage, path)
    )
    prim = joint_schema.GetPrim()
    joint = UsdPhysics.Joint(prim)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, local_pos0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    w0, x0, y0, z0 = local_rot0_wxyz
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(w0, Gf.Vec3f(x0, y0, z0)))
    w, x, y, z = local_rot1_wxyz
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    joint_schema.CreateAxisAttr().Set(axis)
    joint_schema.CreateLowerLimitAttr().Set(float(lower))
    joint_schema.CreateUpperLimitAttr().Set(float(upper))
    drive_name = "linear" if joint_type == "prismatic" else "angular"
    drive = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    return path


def _spawn_dual_knob_box_2(
    stage: Any,
    sim_utils: Any,
    *,
    task_spec: B320TaskSpec,
) -> B320SpawnResult:
    """Author the low, side-by-side clean-room KA005-2 fixture."""

    from pxr import Gf, PhysxSchema, UsdGeom

    validate_b320_ka005_2_asset_files()
    table_path = _spawn_worktable(stage, sim_utils, task_name=task_spec.task_name)
    enclosure_visual = B320_KA005_2_ASSET_DIRECTORY / "project_boxy_enclosure_visual.usda"
    fixed_visual = B320_KA005_2_ASSET_DIRECTORY / "rotary_knob_fixed_visual.usda"
    moving_visual = B320_KA005_2_ASSET_DIRECTORY / "rotary_knob_moving_visual.usda"

    root_home = (-0.14, 0.0, B320_TABLE_TOP_Z + 0.025)
    base_path = f"{B320_ASSET_ROOT_PATH}/panel_base"
    base = UsdGeom.Xform.Define(stage, base_path).GetPrim()
    _set_xform(
        base,
        translation=root_home,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(stage, sim_utils, path=base_path, mass_kg=2.4)
    base.SetCustomDataByKey("handson:logicalName", "panel_base")
    base.SetCustomDataByKey("handson:formalTruth", True)
    base.SetCustomDataByKey("handson:sourceAsset", "project_boxy")
    articulation_root_path = _fix_articulation_base_to_world(stage, base_path)

    enclosure_path = _reference_visual(
        stage,
        path=f"{base_path}/project_boxy_visual",
        source=enclosure_visual,
        source_prim="/ProjectBoxyEnclosureVisual",
        scale=1.0,
    )
    _set_xform(
        stage.GetPrimAtPath(enclosure_path),
        translation=(0.0, 0.0, 0.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    _invisible_box_collider(
        stage,
        sim_utils,
        path=f"{base_path}/collision_proxy",
        size=(0.210, 0.490, 0.050),
        translation=(0.0, 0.0, 0.0),
    )

    static_paths = [table_path, enclosure_path]
    knob_centers = {"left": (0.0, 0.135, 0.025), "right": (0.0, -0.135, 0.025)}
    neutral_tick = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.72, 0.75, 0.78),
        metallic=0.18,
        roughness=0.42,
    )
    target_tick = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.55, 0.03),
        emissive_color=(0.45, 0.12, 0.0),
        roughness=0.28,
    )
    for side, center in knob_centers.items():
        fixed_path = _reference_visual(
            stage,
            path=f"{base_path}/{side}_fixed_bezel",
            source=fixed_visual,
            source_prim="/RotaryKnobFixedVisual",
            scale=1.0,
        )
        _set_xform(
            stage.GetPrimAtPath(fixed_path),
            translation=center,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
        static_paths.append(fixed_path)

        for angle_deg in range(-100, 101, 20):
            angle = math.radians(float(angle_deg))
            radius = 0.040
            tick_path = f"{base_path}/{side}_tick_{angle_deg:+d}".replace("+", "p").replace("-", "n")
            tick_cfg = sim_utils.CuboidCfg(
                size=(0.0075, 0.0018, 0.0012),
                visual_material=neutral_tick,
            )
            tick_cfg.func(
                tick_path,
                tick_cfg,
                translation=(
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                    center[2] + 0.0010,
                ),
                orientation=(math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5)),
            )
            stage.GetPrimAtPath(tick_path).SetCustomDataByKey("handson:visualOnly", True)
            static_paths.append(tick_path)

        marker_path = B320_KA005_2_TARGET_MARKER_PATHS[side]
        marker = UsdGeom.Xform.Define(stage, marker_path).GetPrim()
        _set_xform(
            marker,
            translation=center,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        marker_cfg = sim_utils.CuboidCfg(
            size=(0.0100, 0.0035, 0.0024),
            visual_material=target_tick,
        )
        marker_cfg.func(
            f"{marker_path}/visual",
            marker_cfg,
            translation=(0.044, 0.0, 0.0020),
        )
        marker.SetCustomDataByKey("handson:targetMarker", side)
        marker.SetCustomDataByKey("handson:visualOnly", True)
        static_paths.append(marker_path)

    body_paths: dict[str, str] = {"panel_base": base_path}
    collider_counts = {"panel_base": 1}
    for side, center in knob_centers.items():
        logical = f"knob_{side}"
        body_path = f"{B320_ASSET_ROOT_PATH}/{logical}"
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        world_position = tuple(root_home[index] + center[index] for index in range(3))
        _set_xform(
            body,
            translation=world_position,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        _apply_dynamic_body(
            stage,
            sim_utils,
            path=body_path,
            mass_kg=_KA005_2_KNOB_MASS_KG,
        )
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        physx_body.CreateLinearDampingAttr().Set(0.08)
        physx_body.CreateAngularDampingAttr().Set(
            _KA005_2_KNOB_ANGULAR_DAMPING
        )
        body.SetCustomDataByKey("handson:logicalName", logical)
        body.SetCustomDataByKey("handson:formalTruth", True)
        body.SetCustomDataByKey("handson:sourceAsset", "rotary_knob_0x_anas")
        moving_path = _reference_visual(
            stage,
            path=f"{body_path}/source_visual",
            source=moving_visual,
            source_prim="/RotaryKnobMovingVisual",
            scale=1.0,
        )
        _set_xform(
            stage.GetPrimAtPath(moving_path),
            translation=(0.0, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
        grip_path = _invisible_cylinder_collider(
            stage,
            sim_utils,
            path=f"{body_path}/grip_collision",
            radius=0.030,
            height=0.0422,
            translation=(0.0, 0.0, 0.0272),
        )
        _bind_material(
            grip_path,
            sim_utils=sim_utils,
            static_friction=_KA005_2_GRIP_STATIC_FRICTION,
            dynamic_friction=_KA005_2_GRIP_DYNAMIC_FRICTION,
            friction_combine_mode="min",
        )
        joint_path = _author_joint(
            stage,
            path=f"{B320_ASSET_ROOT_PATH}/{side}_knob_joint",
            joint_type="revolute",
            body0=base_path,
            body1=body_path,
            axis="Z",
            local_pos0=center,
            local_rot1_wxyz=(1.0, 0.0, 0.0, 0.0),
            lower=-100.0,
            upper=100.0,
            stiffness=0.0,
            damping=_KA005_2_JOINT_DRIVE_DAMPING,
            max_force=0.25,
        )
        joint_prim = stage.GetPrimAtPath(joint_path)
        PhysxSchema.PhysxJointAPI.Apply(
            joint_prim
        ).CreateJointFrictionAttr().Set(_KA005_2_JOINT_FRICTION)
        body_paths[logical] = body_path
        collider_counts[logical] = 1

    return B320SpawnResult(
        body_paths=body_paths,
        articulation_root_paths=(articulation_root_path,),
        static_paths=tuple(static_paths),
        dof_names=task_spec.dof_names,
        collider_counts=collider_counts,
    )


def _spawn_legacy_double_cabinet_fixture(
    stage: Any,
    sim_utils: Any,
    *,
    task_spec: B320TaskSpec,
) -> B320SpawnResult:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    scale = 0.55
    source = (
        B320_ASSET_DIRECTORY
        / "dv_double_cabinet_44781/configuration/mobility_base.usd"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    root_home = (-0.10, 0.0, 1.2477)
    base_path = f"{B320_ASSET_ROOT_PATH}/fixture_base"
    base = UsdGeom.Xform.Define(stage, base_path).GetPrim()
    _set_xform(
        base,
        translation=root_home,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(stage, sim_utils, path=base_path, mass_kg=20.0)
    articulation_root_path = _fix_articulation_base_to_world(stage, base_path)
    _reference_visual(
        stage,
        path=f"{base_path}/source_visual",
        source=source,
        source_prim="/visuals/base_link",
        scale=scale,
    )
    # A rear collision wall gives the fixture physical presence without
    # blocking either commanded door or adding hundreds of mesh hulls.
    _invisible_box_collider(
        stage,
        sim_utils,
        path=f"{base_path}/rear_collision",
        size=(0.08, 0.78, 0.72),
        translation=(0.10, 0.0, 0.0),
    )

    source_quat = (0.4999999701976776, 0.4999999701976776, -0.4999999701976776, -0.4999999701976776)
    inverse_source_quat = (
        source_quat[0],
        -source_quat[1],
        -source_quat[2],
        -source_quat[3],
    )
    link_rows = (
        (
            "cabinet_door_left",
            "target_link_revolute",
            (-0.3347701728, 0.6694047451, 0.5983409882),
            (-0.0109392758, -1.1408910439, -0.0155292470),
            (0.6731147831, -0.2792049636, 0.0759348823),
            "target_joint_revolute",
            0.0,
            90.0,
        ),
        (
            "cabinet_door_right",
            "link_11",
            (-0.3320541978, -0.6814305186, 0.5983409882),
            (-0.6738695540, -1.1408910439, -0.0128132719),
            (0.0101834914, -0.2792049636, 0.0786508347),
            "joint_11",
            -90.0,
            0.0,
        ),
    )
    joint_type = "revolute"
    axis = "Z"
    mass = 0.90
    stiffness = 35.0
    damping = 7.0
    max_force = 80.0

    body_paths: dict[str, str] = {"fixture_base": base_path}
    collider_counts: dict[str, int] = {"fixture_base": 1}
    for logical, source_link, local_position, local_min, local_max, dof_name, lower, upper in link_rows:
        body_path = f"{B320_ASSET_ROOT_PATH}/{logical}"
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        scaled_position = tuple(float(value) * scale for value in local_position)
        world_position = tuple(root_home[index] + scaled_position[index] for index in range(3))
        _set_xform(
            body,
            translation=world_position,
            orientation_wxyz=source_quat,
        )
        _apply_dynamic_body(stage, sim_utils, path=body_path, mass_kg=mass)
        body.SetCustomDataByKey("handson:logicalName", logical)
        body.SetCustomDataByKey("handson:formalTruth", True)
        local_center = tuple(
            0.5 * (float(low) + float(high)) * scale
            for low, high in zip(local_min, local_max)
        )
        local_size = tuple(
            max(0.008, (float(high) - float(low)) * scale)
            for low, high in zip(local_min, local_max)
        )
        _reference_visual(
            stage,
            path=f"{body_path}/source_visual",
            source=source,
            source_prim=f"/visuals/{source_link}",
            scale=scale,
        )
        _invisible_box_collider(
            stage,
            sim_utils,
            path=f"{body_path}/collision_proxy",
            size=local_size,
            translation=local_center,
        )
        _author_joint(
            stage,
            path=f"{B320_ASSET_ROOT_PATH}/{dof_name}",
            joint_type=joint_type,
            body0=base_path,
            body1=body_path,
            axis=axis,
            local_pos0=scaled_position,
            local_rot1_wxyz=inverse_source_quat,
            lower=lower,
            upper=upper,
            stiffness=stiffness,
            damping=damping,
            max_force=max_force,
        )
        body_paths[logical] = body_path
        collider_counts[logical] = 1

    base.SetCustomDataByKey("handson:sourceAsset", str(source.relative_to(B320_ASSET_DIRECTORY)))
    base.SetCustomDataByKey("handson:fixtureScale", scale)
    return B320SpawnResult(
        body_paths=body_paths,
        articulation_root_paths=(articulation_root_path,),
        static_paths=(f"{base_path}/source_visual",),
        dof_names=task_spec.dof_names,
        collider_counts=collider_counts,
    )


def _fu001_runtime_point(
    point: tuple[float, float, float],
    *,
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """Map a source Y-up point into the selected FU001 runtime frame.

    ``FU001_RUNTIME_ORIENTATION_WXYZ`` is intentionally represented here by
    its exact axis map as well.  Keeping this helper dependency-free makes the
    placement contract testable without importing pxr/Kit.
    """

    x, y, z = (float(value) * float(scale) for value in point)
    return (-z, -x, y)


def _fu001_collision_paths(stage: Any, root_path: str) -> tuple[str, ...]:
    """Return source-derived collision prims composed below one FU001 link."""

    from pxr import UsdPhysics

    return tuple(
        str(prim.GetPath())
        for prim in _descendants(stage, root_path)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _spawn_polyhaven_double_cabinet_fixture(
    stage: Any,
    sim_utils: Any,
    *,
    task_spec: B320TaskSpec,
) -> B320SpawnResult:
    """Spawn the selected Poly Haven cabinet as one fixed body and two links.

    The converted component USDs retain the source node translations.  The
    fixed body references ``/World`` directly; each door reference is placed
    under a rigid link at its world-space source pivot and receives an equal
    and opposite *scaled* local translation.  That explicit compensation is
    what keeps the source hinge at the joint origin instead of translating it
    twice.
    """

    from pxr import UsdGeom, UsdPhysics

    # The caller validates the closure before this function.  Re-check the
    # component paths at the composition boundary so direct callers cannot
    # accidentally compose the legacy bundle.
    for role, source in FU001_RUNTIME_COMPONENTS.items():
        if not source.is_file():
            raise FileNotFoundError(
                "FU001 selected runtime component is missing: " f"{role}={source}"
            )

    table_path = _spawn_worktable(stage, sim_utils, task_name=task_spec.task_name)
    root_home = tuple(float(value) for value in FU001_ROOT_HOME)
    base_path = f"{B320_ASSET_ROOT_PATH}/fixture_base"
    base = UsdGeom.Xform.Define(stage, base_path).GetPrim()
    # Keep the physical fixture frame world-aligned.  The source-to-runtime
    # quaternion is applied to the visual/collision component, while the
    # revolute axis remains an unambiguous world +Z joint axis.
    _set_xform(
        base,
        translation=root_home,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(stage, sim_utils, path=base_path, mass_kg=20.0)
    articulation_root_path = _fix_articulation_base_to_world(stage, base_path)
    base.SetCustomDataByKey("handson:logicalName", "fixture_base")
    base.SetCustomDataByKey("handson:formalTruth", True)
    base.SetCustomDataByKey("handson:sourceAsset", FU001_ASSET_BUNDLE_KEY)
    base.SetCustomDataByKey("handson:sourceUrl", FU001_ASSET_SOURCE_URL)
    base.SetCustomDataByKey("handson:sourceLicense", FU001_ASSET_LICENSE)
    base.SetCustomDataByKey("handson:sourceLicenseUrl", FU001_ASSET_LICENSE_URL)
    base.SetCustomDataByKey("handson:sourceArchiveSha256", FU001_ASSET_ARCHIVE_SHA256)
    base.SetCustomDataByKey("handson:sourceHingeAxis", FU001_HINGE_AXIS_SOURCE)
    base.SetCustomDataByKey("handson:runtimeHingeAxis", FU001_HINGE_AXIS_RUNTIME)
    base.SetCustomDataByKey(
        "handson:runtimeOrientationWxyz",
        tuple(float(value) for value in FU001_RUNTIME_ORIENTATION_WXYZ),
    )
    base.SetCustomDataByKey("handson:runtimeScale", FU001_RUNTIME_SCALE)

    body_visual_path = _reference_visual(
        stage,
        path=f"{base_path}/source_visual",
        source=FU001_RUNTIME_COMPONENTS["body"],
        source_prim="/World",
        scale=1.0,
        instanceable=False,
    )
    _set_xform(
        stage.GetPrimAtPath(body_visual_path),
        translation=(0.0, 0.0, 0.0),
        orientation_wxyz=FU001_RUNTIME_ORIENTATION_WXYZ,
        scale=(FU001_RUNTIME_SCALE,) * 3,
    )
    _bind_material(
        body_visual_path,
        sim_utils=sim_utils,
        static_friction=1.15,
        dynamic_friction=0.95,
    )
    # Let referenced source meshes carry the physical collision.  A primitive
    # rear wall or door box would change the contact geometry and is expressly
    # not part of the selected asset contract.
    if hasattr(sim_utils, "update_stage"):
        sim_utils.update_stage()
    body_collision_paths = _fu001_collision_paths(stage, base_path)
    if len(body_collision_paths) < 3:
        raise RuntimeError(
            "FU001 selected cabinet body did not compose its three "
            "source-derived collision meshes: "
            f"{body_collision_paths!r}"
        )

    body_paths: dict[str, str] = {"fixture_base": base_path}
    collider_counts: dict[str, int] = {"fixture_base": len(body_collision_paths)}
    door_visual_paths: dict[str, str] = {}
    door_collision_paths: dict[str, tuple[str, ...]] = {}
    door_world_pivots: dict[str, tuple[float, float, float]] = {}

    inverse_runtime_quat = (
        FU001_RUNTIME_ORIENTATION_WXYZ[0],
        -FU001_RUNTIME_ORIENTATION_WXYZ[1],
        -FU001_RUNTIME_ORIENTATION_WXYZ[2],
        -FU001_RUNTIME_ORIENTATION_WXYZ[3],
    )
    door_rows = (
        (
            "cabinet_door_left",
            "target_joint_revolute",
            "door_left",
        ),
        (
            "cabinet_door_right",
            "joint_11",
            "door_right",
        ),
    )
    for logical, joint_name, component_key in door_rows:
        source_pivot = FU001_DOOR_SOURCE_PIVOTS[logical]
        scaled_runtime_pivot = _fu001_runtime_point(
            source_pivot,
            scale=FU001_RUNTIME_SCALE,
        )
        world_pivot = tuple(
            root_home[index] + scaled_runtime_pivot[index]
            for index in range(3)
        )
        door_world_pivots[logical] = world_pivot
        body_path = f"{B320_ASSET_ROOT_PATH}/{logical}"
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        # The link frame is source-oriented (the same q that maps source
        # geometry to runtime).  This matches the existing B320 fallback pose
        # and lets the joint's localRot1 inverse keep its axis at world +Z.
        _set_xform(
            body,
            translation=world_pivot,
            orientation_wxyz=FU001_RUNTIME_ORIENTATION_WXYZ,
        )
        _apply_dynamic_body(stage, sim_utils, path=body_path, mass_kg=0.90)
        body.SetCustomDataByKey("handson:logicalName", logical)
        body.SetCustomDataByKey("handson:formalTruth", True)
        body.SetCustomDataByKey("handson:sourceAsset", FU001_ASSET_BUNDLE_KEY)
        body.SetCustomDataByKey("handson:sourceUrl", FU001_ASSET_SOURCE_URL)
        body.SetCustomDataByKey("handson:sourceLicense", FU001_ASSET_LICENSE)
        body.SetCustomDataByKey("handson:sourceNode", FU001_DOOR_SOURCE_NODES[logical])
        body.SetCustomDataByKey("handson:sourcePivotM", source_pivot)
        body.SetCustomDataByKey("handson:hingeAxisSource", FU001_HINGE_AXIS_SOURCE)
        body.SetCustomDataByKey("handson:hingeAxisRuntime", FU001_HINGE_AXIS_RUNTIME)
        body.SetCustomDataByKey("handson:jointName", joint_name)
        body.SetCustomDataByKey(
            "handson:jointLimitsDeg",
            FU001_DOOR_JOINT_LIMITS_DEG[joint_name],
        )

        visual_path = _reference_visual(
            stage,
            path=f"{body_path}/source_visual",
            source=FU001_RUNTIME_COMPONENTS[component_key],
            source_prim="/World",
            scale=1.0,
            instanceable=False,
        )
        # The referenced source node already translates by ``source_pivot``.
        # It is cancelled in the link's source-oriented frame exactly once;
        # the rigid link world pose above supplies the pivot translation.  The
        # compensation is scaled because the parent scale also scales the
        # referenced node's translation.
        # The parent scale op is applied to the referenced child's translation
        # as well as its mesh.  Cancel the *scaled* source pivot, otherwise a
        # residual (1 - scale) pivot would remain and visibly offset the door.
        pivot_compensation = tuple(
            -FU001_RUNTIME_SCALE * float(value) for value in source_pivot
        )
        _set_xform(
            stage.GetPrimAtPath(visual_path),
            translation=pivot_compensation,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=(FU001_RUNTIME_SCALE,) * 3,
        )
        stage.GetPrimAtPath(visual_path).SetCustomDataByKey(
            "handson:sourcePivotCompensationM", pivot_compensation
        )
        stage.GetPrimAtPath(visual_path).SetCustomDataByKey(
            "handson:sourceNodeTranslationRetained", True
        )
        _bind_material(
            visual_path,
            sim_utils=sim_utils,
            static_friction=1.15,
            dynamic_friction=0.95,
        )
        door_visual_paths[logical] = visual_path
        if hasattr(sim_utils, "update_stage"):
            sim_utils.update_stage()
        collision_paths = _fu001_collision_paths(stage, body_path)
        if len(collision_paths) < 1:
            raise RuntimeError(
                f"FU001 selected {logical} did not compose its source-derived "
                "door collision mesh"
            )
        door_collision_paths[logical] = collision_paths

        lower, upper = FU001_DOOR_JOINT_LIMITS_DEG[joint_name]
        _author_joint(
            stage,
            path=f"{B320_ASSET_ROOT_PATH}/{joint_name}",
            joint_type="revolute",
            body0=base_path,
            body1=body_path,
            axis="Z",
            # Base is world-aligned, so this is the transformed/scaled source
            # pivot in the base frame (not a second local door translation).
            local_pos0=scaled_runtime_pivot,
            local_rot1_wxyz=inverse_runtime_quat,
            lower=lower,
            upper=upper,
            stiffness=35.0,
            damping=7.0,
            max_force=80.0,
        )
        body_paths[logical] = body_path
        collider_counts[logical] = len(collision_paths)

    # The selected cabinet source intentionally models the closed leaves as
    # overlaying the front frame (the visual back of each leaf sits inside the
    # shell by roughly 20--30 mm).  Solving that source-internal visual overlap
    # as a physical contact wedges the reduced-coordinate links at the first
    # PhysX reset.  Keep the source-derived collision meshes for contacts with
    # hands/table, but filter only the internal shell/leaf pair.  This is the
    # same narrow collision-group policy used by the integrated FU006 cabinet:
    # it is not a proxy collider and it does not filter either leaf against the
    # robot or support.
    internal_group_root = f"{B320_ASSET_ROOT_PATH}/fu001_internal_collision_groups"
    shell_group = UsdPhysics.CollisionGroup.Define(
        stage, f"{internal_group_root}/shell"
    )
    shell_collection = shell_group.GetCollidersCollectionAPI()
    shell_collection.CreateIncludesRel().AddTarget(
        stage.GetPrimAtPath(base_path).GetPath()
    )
    leaf_group = UsdPhysics.CollisionGroup.Define(
        stage, f"{internal_group_root}/leaves"
    )
    leaf_collection = leaf_group.GetCollidersCollectionAPI()
    for logical, _joint_name, _component_key in door_rows:
        leaf_collection.CreateIncludesRel().AddTarget(
            stage.GetPrimAtPath(body_paths[logical]).GetPath()
        )
    shell_group.CreateFilteredGroupsRel().AddTarget(leaf_group.GetPath())
    leaf_group.CreateFilteredGroupsRel().AddTarget(shell_group.GetPath())
    shell_group.GetPrim().SetCustomDataByKey(
        "handson:purpose", "fu001_source_internal_shell_leaf_filter"
    )
    leaf_group.GetPrim().SetCustomDataByKey(
        "handson:purpose", "fu001_source_internal_shell_leaf_filter"
    )

    # Collision groups are a coarse, stage-level filter.  Isaac Sim 5.1 can
    # resolve a collection rooted at a rigid body differently when the body
    # is an articulation link, so also author the exact source-collider pairs
    # with FilteredPairsAPI.  This keeps the source meshes as the contact
    # truth for hands/table while deterministically removing only the known
    # source-internal shell/leaf overlap.  The explicit pair list is part of
    # the scene evidence and is intentionally not a blanket self-collision
    # disable on either link.
    for shell_path in body_collision_paths:
        shell_collider = stage.GetPrimAtPath(shell_path)
        shell_pairs = UsdPhysics.FilteredPairsAPI.Apply(shell_collider)
        shell_rel = shell_pairs.CreateFilteredPairsRel()
        for logical, _joint_name, _component_key in door_rows:
            for leaf_path in door_collision_paths[logical]:
                shell_rel.AddTarget(leaf_path)
    for logical, _joint_name, _component_key in door_rows:
        for leaf_path in door_collision_paths[logical]:
            leaf_collider = stage.GetPrimAtPath(leaf_path)
            leaf_pairs = UsdPhysics.FilteredPairsAPI.Apply(leaf_collider)
            leaf_rel = leaf_pairs.CreateFilteredPairsRel()
            for shell_path in body_collision_paths:
                leaf_rel.AddTarget(shell_path)

    static_paths = (table_path, body_visual_path)
    evidence = {
        "asset_bundle": FU001_ASSET_BUNDLE_KEY,
        "asset_manifest": str(_fu001_manifest_path()),
        "source_archive": str(FU001_ASSET_SOURCE_ARCHIVE_PATH),
        "source_archive_sha256": FU001_ASSET_ARCHIVE_SHA256,
        "source_url": FU001_ASSET_SOURCE_URL,
        "license": FU001_ASSET_LICENSE,
        "license_url": FU001_ASSET_LICENSE_URL,
        "runtime_components": {
            key: str(path) for key, path in FU001_RUNTIME_COMPONENTS.items()
        },
        "runtime_orientation_wxyz": list(FU001_RUNTIME_ORIENTATION_WXYZ),
        "runtime_scale": FU001_RUNTIME_SCALE,
        "source_to_runtime": "(x,y,z) -> (-z,-x,y)",
        "hinge_axis_source": FU001_HINGE_AXIS_SOURCE,
        "hinge_axis_runtime": FU001_HINGE_AXIS_RUNTIME,
        "root_home_w": list(root_home),
        "body_visual_path": body_visual_path,
        "door_visual_paths": dict(door_visual_paths),
        "door_source_pivots_m": {
            key: list(value) for key, value in FU001_DOOR_SOURCE_PIVOTS.items()
        },
        "door_world_pivots_w": {
            key: list(value) for key, value in door_world_pivots.items()
        },
        "joint_axis": {
            "target_joint_revolute": "Z",
            "joint_11": "Z",
        },
        "joint_limits_deg": {
            key: list(value) for key, value in FU001_DOOR_JOINT_LIMITS_DEG.items()
        },
        "collision_policy": "source_mesh_topology_then_physx_convex_decomposition",
        "internal_collision_filter": {
            "type": "UsdPhysics.FilteredPairsAPI",
            "scope": "source cabinet shell colliders <-> source door colliders",
            "reason": "source closed-pose shell/leaf mesh overlap; preserve hand/table contacts",
            "shell_paths": list(body_collision_paths),
            "leaf_paths": {
                key: list(value) for key, value in door_collision_paths.items()
            },
        },
        "collision_paths": {
            "fixture_base": list(body_collision_paths),
            **{key: list(value) for key, value in door_collision_paths.items()},
        },
        "placeholder_collision": False,
        "door_pivot_compensation": "source node translation cancelled once in link visual parent",
    }
    return B320SpawnResult(
        body_paths=body_paths,
        articulation_root_paths=(articulation_root_path,),
        static_paths=static_paths,
        dof_names=task_spec.dof_names,
        collider_counts=collider_counts,
        evidence=evidence,
    )


def _spawn_double_cabinet_fixture(
    stage: Any,
    sim_utils: Any,
    *,
    task_spec: B320TaskSpec,
) -> B320SpawnResult:
    """Dispatch the cabinet fixture without silently borrowing an asset."""

    if task_spec.task_name == "b320_synchronized_double_cabinet_open":
        return _spawn_polyhaven_double_cabinet_fixture(
            stage,
            sim_utils,
            task_spec=task_spec,
        )
    # Keep the historical implementation available for downstream callers
    # that explicitly construct a legacy cabinet spec; FU001 never reaches it.
    return _spawn_legacy_double_cabinet_fixture(
        stage,
        sim_utils,
        task_spec=task_spec,
    )


def _configure_faucet(
    stage: Any,
    result: B320SpawnResult,
    *,
    root_path: str,
) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics

    base = next(
        (
            prim
            for prim in _descendants(stage, root_path)
            if prim.GetName() == "E_body_1" and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        None,
    )
    handle = next(
        (
            prim
            for prim in _descendants(stage, root_path)
            if prim.GetName() == "E_switch_2_7" and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        None,
    )
    if base is None or handle is None:
        raise RuntimeError("B320 faucet body/handle contract is incomplete")
    articulation_root_path = _fix_articulation_base_to_world(
        stage,
        str(base.GetPath()),
    )
    target_joint = next(
        (
            prim
            for prim in _descendants(stage, root_path)
            if prim.GetName() == "RevoluteJoint_faucet_3_right"
            and prim.IsA(UsdPhysics.RevoluteJoint)
        ),
        None,
    )
    other_joint = next(
        (
            prim
            for prim in _descendants(stage, root_path)
            if prim.GetName() == "RevoluteJoint_faucet_3_left"
            and prim.IsA(UsdPhysics.RevoluteJoint)
        ),
        None,
    )
    if target_joint is None or other_joint is None:
        raise RuntimeError("B320 faucet joint contract is incomplete")
    UsdPhysics.RevoluteJoint(other_joint).CreateLowerLimitAttr().Set(0.0)
    UsdPhysics.RevoluteJoint(other_joint).CreateUpperLimitAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Apply(target_joint, "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(0.08)
    drive.CreateMaxForceAttr().Set(3.0)
    result.body_paths["faucet_fixture"] = str(base.GetPath())
    result.body_paths["faucet_base"] = str(base.GetPath())
    result.body_paths["faucet_handle"] = str(handle.GetPath())
    result.articulation_root_paths = (articulation_root_path,)
    result.dof_names = ("RevoluteJoint_faucet_3_right",)


def _partition_ka020_downloaded_visual(
    stage: Any,
    *,
    visual_path: str,
    moving_only: bool,
) -> None:
    """Keep the selected source mesh on exactly one rigid link."""

    from pxr import UsdGeom

    scene_root = stage.GetPrimAtPath(f"{visual_path}/root/GLTF_SceneRootNode")
    if scene_root is None or not scene_root.IsValid():
        raise RuntimeError(
            f"KA020 selected visual hierarchy is incomplete: {visual_path}"
        )
    found_moving = False
    for component in scene_root.GetChildren():
        is_moving = component.GetName() == KA020_MOVING_VISUAL_COMPONENT
        found_moving = bool(found_moving or is_moving)
        imageable = UsdGeom.Imageable(component)
        if bool(is_moving) == bool(moving_only):
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    if not found_moving:
        raise RuntimeError(
            "KA020 downloaded pump-head component is missing: "
            f"{KA020_MOVING_VISUAL_COMPONENT}"
        )


def _reference_ka020_pump_visual(
    stage: Any,
    *,
    path: str,
    source: Path,
) -> str:
    """Reference only the moving source component on the pump rigid link.

    Referencing the complete dispenser a second time below another dynamic
    rigid body crashes PhysX GPU scene creation in Isaac Sim 5.1, even when
    every non-pump component is hidden. Compose just the reviewed pump-head
    subtree and explicitly bring along its bound material instead.
    """

    from pxr import Sdf, UsdGeom, UsdShade

    wrapper = UsdGeom.Xform.Define(stage, path).GetPrim()
    _set_xform(
        wrapper,
        translation=tuple(-value for value in KA020_PUMP_HEAD_CENTER_LOCAL_M),
        orientation_wxyz=KA020_SOURCE_ORIENTATION_WXYZ,
        scale=(KA020_DISPENSER_VISUAL_SCALE,) * 3,
    )
    asset_path = f"{path}/asset"
    asset = UsdGeom.Xform.Define(stage, asset_path).GetPrim()
    asset.GetReferences().AddReference(
        Sdf.Reference(
            assetPath=str(source.resolve()),
            primPath=KA020_MOVING_VISUAL_SOURCE_PRIM,
        )
    )
    asset.SetInstanceable(False)

    material_path = f"{path}/material_0"
    material = UsdShade.Material.Define(stage, material_path)
    material.GetPrim().GetReferences().AddReference(
        Sdf.Reference(
            assetPath=str(source.resolve()),
            primPath=KA020_MOVING_MATERIAL_SOURCE_PRIM,
        )
    )
    mesh_path = f"{asset_path}/Object_16/Object_6"
    mesh = stage.GetPrimAtPath(mesh_path)
    if not mesh.IsValid() or not mesh.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"KA020 pump visual mesh is missing: {mesh_path}")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    return asset_path


def _spawn_ka020_flow_proxy(stage: Any, sim_utils: Any) -> str:
    """Author a render-only soap stream driven by physical pump travel."""

    from pxr import UsdGeom

    cfg = sim_utils.CylinderCfg(
        radius=0.0032,
        height=1.0,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.88, 0.98),
            emissive_color=(0.04, 0.16, 0.20),
            opacity=0.72,
            roughness=0.20,
        ),
    )
    cfg.func(
        KA020_FLOW_PROXY_PATH,
        cfg,
        translation=(0.0, 0.0, B320_TABLE_TOP_Z - 1.0),
    )
    prim = stage.GetPrimAtPath(KA020_FLOW_PROXY_PATH)
    prim.SetCustomDataByKey("handson:flowProxy", "soap_stream")
    prim.SetCustomDataByKey("handson:drivenBy", "soap_dispense_joint")
    prim.SetCustomDataByKey("handson:visualOnly", True)
    UsdGeom.Imageable(prim).MakeInvisible()
    return KA020_FLOW_PROXY_PATH


def _spawn_ka020_soap_dispenser(
    stage: Any,
    sim_utils: Any,
    result: B320SpawnResult,
    *,
    bottle_spec: B320ObjectSpec,
) -> str:
    """Spawn the selected bottle and directly pressable physical pump."""

    from pxr import UsdGeom

    source = bottle_spec.source_path
    if not source.is_file():
        raise FileNotFoundError(source)

    bottle_path = bottle_spec.prim_path
    bottle = UsdGeom.Xform.Define(stage, bottle_path).GetPrim()
    _set_xform(
        bottle,
        translation=bottle_spec.home_translation,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(
        stage,
        sim_utils,
        path=bottle_path,
        mass_kg=bottle_spec.mass_kg,
    )
    bottle.SetCustomDataByKey("handson:logicalName", "soap_bottle")
    bottle.SetCustomDataByKey("handson:formalTruth", True)
    bottle.SetCustomDataByKey(
        "handson:sourceAsset", KA020_DISPENSER_ASSET_BUNDLE_KEY
    )
    bottle.SetCustomDataByKey(
        "handson:sourceSha256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    fixed_visual_path = _reference_visual_preserving_source_root(
        stage,
        path=f"{bottle_path}/source_visual",
        source=source,
        source_prim="/World",
        translation=(0.0, 0.0, 0.0),
        orientation_wxyz=KA020_SOURCE_ORIENTATION_WXYZ,
        scale=KA020_DISPENSER_VISUAL_SCALE,
    )
    _partition_ka020_downloaded_visual(
        stage,
        visual_path=fixed_visual_path,
        moving_only=False,
    )

    body_collision = _invisible_box_collider(
        stage,
        sim_utils,
        path=f"{bottle_path}/body_collision",
        size=KA020_BOTTLE_BODY_COLLIDER_SIZE_M,
        translation=KA020_BOTTLE_BODY_COLLIDER_CENTER_M,
    )
    neck_collision = _invisible_cylinder_collider(
        stage,
        sim_utils,
        path=f"{bottle_path}/neck_collision",
        radius=KA020_BOTTLE_NECK_COLLIDER_RADIUS_M,
        height=KA020_BOTTLE_NECK_COLLIDER_HEIGHT_M,
        translation=KA020_BOTTLE_NECK_COLLIDER_CENTER_M,
    )
    for collider_path, logical_name in (
        (body_collision, "soap_bottle_body"),
        (neck_collision, "soap_bottle_neck"),
    ):
        collider = stage.GetPrimAtPath(collider_path)
        collider.SetCustomDataByKey("handson:logicalName", logical_name)
        collider.SetCustomDataByKey("handson:collisionOnly", True)

    # Keep this as a maximal-coordinate PhysX joint. Isaac Sim 5.1's GPU
    # solver crashes during the first reset when the free bottle is promoted
    # to an articulation root. The same physical prismatic constraint is
    # stable without ArticulationRootAPI and remains directly pressable.
    pad_path = f"{B320_ASSET_ROOT_PATH}/soap_dispense_pad"
    pad = UsdGeom.Xform.Define(stage, pad_path).GetPrim()
    pad_translation = tuple(
        float(bottle_spec.home_translation[index]
        + KA020_PUMP_HEAD_CENTER_LOCAL_M[index])
        for index in range(3)
    )
    _set_xform(
        pad,
        translation=pad_translation,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(stage, sim_utils, path=pad_path, mass_kg=0.025)
    _reference_ka020_pump_visual(
        stage,
        path=f"{pad_path}/source_visual",
        source=source,
    )
    pump_collision = _invisible_box_collider(
        stage,
        sim_utils,
        path=f"{pad_path}/pump_head_collision",
        size=KA020_PUMP_HEAD_COLLIDER_SIZE_M,
        translation=(0.0, 0.0, 0.0),
    )
    pad.SetCustomDataByKey("handson:logicalName", "soap_dispense_pad")
    pad.SetCustomDataByKey("handson:semanticRole", "moving_pump_head")
    pad.SetCustomDataByKey("handson:formalTruth", True)
    pad.SetCustomDataByKey(
        "handson:sourceAsset", KA020_DISPENSER_ASSET_BUNDLE_KEY
    )
    pump_collision_prim = stage.GetPrimAtPath(pump_collision)
    pump_collision_prim.SetCustomDataByKey(
        "handson:logicalName", "soap_pump_head"
    )
    pump_collision_prim.SetCustomDataByKey("handson:collisionOnly", True)
    _author_joint(
        stage,
        path=f"{B320_ASSET_ROOT_PATH}/soap_dispense_joint",
        joint_type="prismatic",
        body0=bottle_path,
        body1=pad_path,
        axis="Z",
        local_pos0=KA020_PUMP_HEAD_CENTER_LOCAL_M,
        local_rot1_wxyz=(1.0, 0.0, 0.0, 0.0),
        lower=-KA020_PUMP_TRAVEL_M,
        upper=0.0,
        # Keep the 25 g head within about 1 mm of its visible rest pose under
        # gravity.  Max force remains capped at 4 N so a physical finger can
        # still depress the complete 18 mm travel.
        stiffness=260.0,
        damping=4.0,
        max_force=4.0,
    )
    _bind_material(
        bottle_path,
        sim_utils=sim_utils,
        static_friction=bottle_spec.static_friction,
        dynamic_friction=bottle_spec.dynamic_friction,
    )
    result.body_paths["soap_bottle"] = bottle_path
    result.body_paths["soap_dispense_pad"] = pad_path
    result.articulation_root_paths = ()
    result.dof_names = ("soap_dispense_joint",)
    result.collider_counts["soap_bottle"] = 2
    result.collider_counts["soap_dispense_pad"] = 1
    result.evidence.update(
        {
            "ka020_selected_assets": [
                KA020_DISPENSER_ASSET_BUNDLE_KEY,
                KA020_SPONGE_ASSET_BUNDLE_KEY,
            ],
            "ka020_license": KA020_SOURCE_LICENSE,
            "ka020_dispenser_source_url": KA020_DISPENSER_SOURCE_URL,
            "ka020_sponge_source_url": KA020_SPONGE_SOURCE_URL,
            "ka020_moving_visual_component": KA020_MOVING_VISUAL_COMPONENT,
            "ka020_moving_visual_source_prim": KA020_MOVING_VISUAL_SOURCE_PRIM,
            "ka020_joint_runtime": "maximal_coordinate_prismatic",
            "ka020_pump_travel_readback": "rigid_body_relative_pose",
            "ka020_pump_head_center_local_m": list(
                KA020_PUMP_HEAD_CENTER_LOCAL_M
            ),
            "ka020_pump_outlet_offset_m": list(KA020_PUMP_OUTLET_OFFSET_M),
            "ka020_pump_travel_m": KA020_PUMP_TRAVEL_M,
            "ka020_dispenser_source_root_preserved": True,
            "ka020_collision_only_paths": [
                body_collision,
                neck_collision,
                pump_collision,
            ],
        }
    )
    return _spawn_ka020_flow_proxy(stage, sim_utils)


def _spawn_ka021_hs005_faucet(
    stage: Any,
    sim_utils: Any,
    *,
    task_spec: B320TaskSpec,
) -> B320SpawnResult:
    """Reuse HS005's kitchen while articulating the downloaded faucet."""

    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics

    table_path = _spawn_worktable(stage, sim_utils, task_name=task_spec.task_name)
    # HS005's realistic counter and open sink visually replace the generic B320
    # tabletop.  The generic table collider is broader than the sink task and
    # would overlap the task-only basin proxy below, so keep neither its brown
    # render material nor its collision in this composition.
    table_prim = stage.GetPrimAtPath(table_path)
    UsdGeom.Imageable(table_prim).MakeInvisible()
    table_colliders = tuple(
        prim
        for prim in _descendants(stage, table_path)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if not table_colliders:
        raise RuntimeError("KA021 worktable is missing its expected collision prim")
    for table_collider in table_colliders:
        UsdPhysics.CollisionAPI(
            table_collider
        ).CreateCollisionEnabledAttr().Set(False)
    fixture_root_path = f"{B320_ASSET_ROOT_PATH}/faucet_fixture"
    UsdGeom.Xform.Define(stage, fixture_root_path)
    if not KA021_HS005_SCENE_PATH.is_file():
        raise FileNotFoundError(KA021_HS005_SCENE_PATH)

    kitchen_path = f"{fixture_root_path}/hs005_realistic_kitchen"
    if kitchen_path != KA021_SHARED_SCENE_ROOT_PATH:
        raise RuntimeError(
            "KA021 shared-scene path drifted from its collision contract: "
            f"{kitchen_path}"
        )
    kitchen = UsdGeom.Xform.Define(stage, kitchen_path).GetPrim()
    kitchen.GetReferences().AddReference(
        Sdf.Reference(
            assetPath=str(KA021_HS005_SCENE_PATH.resolve()),
            primPath="/HS005Assets",
        )
    )
    kitchen.SetCustomDataByKey("handson:sharedFromTask", "B320-HS005")
    kitchen.SetCustomDataByKey("handson:sourceCreator", "Visthétique")
    kitchen.SetCustomDataByKey("handson:sourceLicense", "CC BY 4.0")
    kitchen.SetCustomDataByKey("handson:realisticKitchen", True)
    for prop_name in ("Bottle", "Brush", "DishSoap", "Sponge"):
        prop = stage.GetPrimAtPath(f"{kitchen_path}/{prop_name}")
        if not prop.IsValid():
            raise RuntimeError(f"HS005 shared prop is missing: {prop_name}")
        prop.SetActive(task_spec.task_name == XY4_TASK_NAME)

    if task_spec.task_name == XY4_TASK_NAME:
        from .xy4_scene import configure_scrub_props
        configure_scrub_props(stage, kitchen_path)

    source_sink_collider_path = f"{kitchen_path}/Sink/Collider"
    source_sink_collider = stage.GetPrimAtPath(source_sink_collider_path)
    if not source_sink_collider.IsValid() or not source_sink_collider.HasAPI(
        UsdPhysics.CollisionAPI
    ):
        raise RuntimeError(
            "HS005 shared sink no longer exposes its reviewed collision prim: "
            f"{source_sink_collider_path}"
        )
    # The shared HS005 layer now owns its reviewed analytic basin floor, four
    # full-height walls, and open countertop ring.  KA021 consumes and verifies
    # that scene contract instead of disabling the floor and authoring a second,
    # task-local approximation that could drift from HS005.
    floor_collision = UsdPhysics.CollisionAPI(source_sink_collider)
    floor_enabled = floor_collision.GetCollisionEnabledAttr()
    if floor_enabled.IsValid() and floor_enabled.Get() is False:
        raise RuntimeError(
            f"KA021 shared basin floor is disabled: {source_sink_collider_path}"
        )
    source_sink_collider.SetCustomDataByKey(
        "handson:sharedSceneCollisionFloor", True
    )
    safe_basin_floor_path = source_sink_collider_path

    shared_collision_root = stage.GetPrimAtPath(
        KA021_SHARED_SCENE_COLLISION_ROOT_PATH
    )
    if (
        not shared_collision_root.IsValid()
        or shared_collision_root.GetCustomDataByKey("physics_policy")
        != KA021_SHARED_SCENE_COLLISION_POLICY
    ):
        raise RuntimeError(
            "KA021 shared HS005 scene collision policy is missing or stale: "
            f"{KA021_SHARED_SCENE_COLLISION_ROOT_PATH}"
        )
    shared_scene_collision_paths: list[str] = []
    counter_surface_collision_paths: list[str] = []
    for name, _size, _center in KA021_SHARED_SCENE_COLLISION_SPECS:
        collision_path = f"{KA021_SHARED_SCENE_COLLISION_ROOT_PATH}/{name}"
        collision_prim = stage.GetPrimAtPath(collision_path)
        if not collision_prim.IsValid() or not collision_prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            raise RuntimeError(
                f"KA021 shared HS005 scene collider is missing: {collision_path}"
            )
        enabled = UsdPhysics.CollisionAPI(
            collision_prim
        ).GetCollisionEnabledAttr()
        if enabled.IsValid() and enabled.Get() is False:
            raise RuntimeError(
                f"KA021 shared HS005 scene collider is disabled: {collision_path}"
            )
        shared_scene_collision_paths.append(collision_path)
        if name.endswith("_countertop"):
            counter_surface_collision_paths.append(collision_path)
    park_support_path = _invisible_box_collider(
        stage,
        sim_utils,
        path=KA021_CUP_PARK_SUPPORT_PATH,
        size=KA021_CUP_PARK_SUPPORT_SIZE_M,
        translation=KA021_CUP_PARK_SUPPORT_CENTER_W,
    )
    park_support = stage.GetPrimAtPath(park_support_path)
    park_support.SetCustomDataByKey("handson:logicalName", "cup_park_support")
    park_support.SetCustomDataByKey("handson:formalTruth", True)
    park_support.SetCustomDataByKey("handson:initialHandSafe", True)
    park_support.SetCustomDataByKey(
        "handson:purpose", "post_fill_counter_support_beside_sink"
    )
    park_marker_path = _spawn_goal_marker(
        stage,
        sim_utils,
        name="cup_park_marker",
        translation=(
            KA021_CUP_PARK_POSITION_W[0],
            KA021_CUP_PARK_POSITION_W[1],
            KA021_CUP_PARK_POSITION_W[2] + 0.004,
        ),
        size=(
            KA021_CUP_PARK_SUPPORT_SIZE_M[0],
            KA021_CUP_PARK_SUPPORT_SIZE_M[1],
            0.006,
        ),
        color=(0.95, 0.68, 0.08),
        highlight=True,
    )
    stage.GetPrimAtPath(park_marker_path).SetCustomDataByKey(
        "handson:purpose", "post_fill_mug_placement"
    )

    original_faucet_path = (
        f"{kitchen_path}/KitchenBackground/Visual/Sink_Faucet3_001_0"
    )
    original_faucet = stage.GetPrimAtPath(original_faucet_path)
    if not original_faucet.IsValid():
        raise RuntimeError(
            "HS005 realistic kitchen no longer exposes its reviewed faucet prim"
        )
    # Hide the composed HS005 faucet so the downloaded BlenderKit model appears
    # once and its independently authored handle can move.
    UsdGeom.Imageable(original_faucet).MakeInvisible()
    base_path = f"{fixture_root_path}/E_body_1"
    base = UsdGeom.Xform.Define(stage, base_path).GetPrim()
    _set_xform(
        base,
        translation=KA021_FAUCET_BASE_POSITION_W,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(stage, sim_utils, path=base_path, mass_kg=2.0)
    articulation_root = _fix_articulation_base_to_world(stage, base_path)
    base.SetCustomDataByKey("handson:logicalName", "faucet_base")
    base.SetCustomDataByKey("handson:formalTruth", True)
    base.SetCustomDataByKey(
        "handson:sourceAsset",
        KA021_FAUCET_STATIC_VISUAL_RELATIVE_USD,
    )
    base.SetCustomDataByKey("handson:sourceLicense", KA021_FAUCET_SOURCE_LICENSE)
    base.SetCustomDataByKey("handson:sourceCreator", KA021_FAUCET_SOURCE_CREATOR)
    base.SetCustomDataByKey("handson:sourceUrl", KA021_FAUCET_SOURCE_URL)
    base.SetCustomDataByKey(
        "handson:sourceSha256",
        KA021_FAUCET_SOURCE_SHA256,
    )
    base.SetCustomDataByKey("handson:sharedScene", "B320-HS005")
    base.SetCustomDataByKey("handson:authoredArticulation", True)
    base.SetCustomDataByKey(
        "handson:visualYawDeg",
        KA021_FAUCET_MODEL_YAW_DEG,
    )
    base.SetCustomDataByKey("handson:visualScale", KA021_FAUCET_MODEL_SCALE)

    static_source = (
        B320_ASSET_DIRECTORY / KA021_FAUCET_STATIC_VISUAL_RELATIVE_USD
    )
    handle_source = (
        B320_ASSET_DIRECTORY / KA021_FAUCET_HANDLE_VISUAL_RELATIVE_USD
    )
    for source in (static_source, handle_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    visual_path = f"{base_path}/hs005_source_faucet_static_visual"
    visual = UsdGeom.Xform.Define(stage, visual_path).GetPrim()
    source_visual = UsdGeom.Xform.Define(
        stage,
        f"{visual_path}/source",
    ).GetPrim()
    source_visual.GetReferences().AddReference(
        Sdf.Reference(assetPath=str(static_source.resolve()), primPath="/World")
    )
    _set_xform(
        visual,
        translation=(0.0, 0.0, 0.0),
        orientation_wxyz=KA021_FAUCET_MODEL_ORIENTATION_WXYZ,
        scale=(
            KA021_FAUCET_MODEL_SCALE,
            KA021_FAUCET_MODEL_SCALE,
            KA021_FAUCET_MODEL_SCALE,
        ),
    )
    visual.SetCustomDataByKey("handson:visualSource", "BlenderKit")

    # The body root is translated to the real counter mount but intentionally
    # has identity orientation.  All analytic collision coordinates below are
    # therefore local to that root; the same yaw/scale is carried by the
    # visual wrapper and by the moving handle wrapper.
    column_center_local = _ka021_model_local_point((0.0, 0.0, 0.065))
    outlet_local = _ka021_model_local_point(KA021_FAUCET_NATIVE_OUTLET_M)
    arch_segments_local = tuple(
        (
            _ka021_model_local_point(start),
            _ka021_model_local_point(end),
        )
        for start, end in zip(
            KA021_FAUCET_NATIVE_ARCH_CENTERLINE_M[:-1],
            KA021_FAUCET_NATIVE_ARCH_CENTERLINE_M[1:],
        )
    )
    _invisible_cylinder_collider(
        stage,
        sim_utils,
        path=f"{base_path}/column_collision",
        radius=KA021_FAUCET_COLUMN_COLLIDER_RADIUS_M,
        height=0.135,
        translation=column_center_local,
    )
    spout_arch_collision_paths_list: list[str] = []
    for index, ((start_local, end_local), (start_world, end_world)) in enumerate(
        zip(arch_segments_local, KA021_FAUCET_ARCH_COLLIDER_SEGMENTS_W)
    ):
        path = _invisible_capsule_segment_collider(
            stage,
            sim_utils,
            path=f"{base_path}/spout_arch_collision_{index}",
            start=start_local,
            end=end_local,
            radius=KA021_FAUCET_ARCH_COLLIDER_RADIUS_M,
        )
        # The collider is authored below an identity-oriented body root, so
        # the helper receives local points.  Keep both frames explicit in USD
        # custom data rather than labeling local coordinates as world values.
        arch_prim = stage.GetPrimAtPath(path)
        arch_prim.SetCustomDataByKey("handson:segmentStartLocal", start_local)
        arch_prim.SetCustomDataByKey("handson:segmentEndLocal", end_local)
        arch_prim.SetCustomDataByKey("handson:segmentStartW", start_world)
        arch_prim.SetCustomDataByKey("handson:segmentEndW", end_world)
        spout_arch_collision_paths_list.append(path)
    spout_arch_collision_paths = tuple(spout_arch_collision_paths_list)
    _invisible_cylinder_collider(
        stage,
        sim_utils,
        path=f"{base_path}/outlet_collision",
        radius=0.018,
        height=0.038,
        translation=(outlet_local[0], outlet_local[1], outlet_local[2] + 0.019),
    )

    outlet_path = f"{base_path}/metric_outlet"
    outlet = UsdGeom.Xform.Define(stage, outlet_path).GetPrim()
    _set_xform(
        outlet,
        translation=outlet_local,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    outlet.SetCustomDataByKey("handson:outletPositionW", KA021_OUTLET_POSITION_W)
    outlet.SetCustomDataByKey("handson:metricMarker", True)
    outlet.SetCustomDataByKey(
        "handson:sourcePrim",
        "Sink_Out_001_0/Sink_Out_001_0",
    )

    handle_path = f"{fixture_root_path}/E_switch_2_7"
    handle = UsdGeom.Xform.Define(stage, handle_path).GetPrim()
    _set_xform(
        handle,
        translation=KA021_FAUCET_HANDLE_PIVOT_W,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    _apply_dynamic_body(stage, sim_utils, path=handle_path, mass_kg=0.14)
    # The BlenderKit USD has no native PhysX cartridge.  Keep the authored
    # runtime handle gravity-balanced so it cannot open itself before the
    # operator touches it.  Success still requires continuous right-hand
    # contact with this physical body.
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        handle
    ).CreateDisableGravityAttr().Set(True)
    handle.SetCustomDataByKey("handson:logicalName", "faucet_handle")
    handle.SetCustomDataByKey("handson:formalTruth", True)
    handle.SetCustomDataByKey("handson:authoredFromSource", True)
    handle.SetCustomDataByKey(
        "handson:sourceAsset",
        KA021_FAUCET_HANDLE_VISUAL_RELATIVE_USD,
    )
    handle.SetCustomDataByKey("handson:sourceLicense", KA021_FAUCET_SOURCE_LICENSE)
    handle.SetCustomDataByKey("handson:sourceCreator", KA021_FAUCET_SOURCE_CREATOR)
    handle.SetCustomDataByKey("handson:sourceUrl", KA021_FAUCET_SOURCE_URL)
    handle.SetCustomDataByKey("handson:sourceSha256", KA021_FAUCET_SOURCE_SHA256)
    handle.SetCustomDataByKey("handson:sharedScene", "B320-HS005")
    handle.SetCustomDataByKey("handson:closedDirection", "toward faucet body")
    handle.SetCustomDataByKey(
        "handson:openMotion",
        "pull outward along the yawed source axis; initial motion has world -X component",
    )
    handle.SetCustomDataByKey(
        "handson:temperatureMotion", "left-right motion is not the flow joint"
    )
    handle.SetCustomDataByKey("handson:gravityBalancedCartridge", True)

    handle_visual_path = f"{handle_path}/hs005_source_lever_visual"
    handle_visual = UsdGeom.Xform.Define(stage, handle_visual_path).GetPrim()
    handle_visual_source = UsdGeom.Xform.Define(
        stage,
        f"{handle_visual_path}/source",
    ).GetPrim()
    handle_visual_source.GetReferences().AddReference(
        Sdf.Reference(assetPath=str(handle_source.resolve()), primPath="/World")
    )
    _set_xform(
        handle_visual,
        translation=tuple(
            -_ka021_model_local_point(KA021_FAUCET_NATIVE_HANDLE_PIVOT_M)[index]
            for index in range(3)
        ),
        orientation_wxyz=KA021_FAUCET_MODEL_ORIENTATION_WXYZ,
        scale=(
            KA021_FAUCET_MODEL_SCALE,
            KA021_FAUCET_MODEL_SCALE,
            KA021_FAUCET_MODEL_SCALE,
        ),
    )
    handle_visual.SetCustomDataByKey("handson:visualSource", "BlenderKit")
    _invisible_box_collider(
        stage,
        sim_utils,
        path=f"{handle_path}/source_lever_grip_collision",
        size=KA021_FAUCET_GRIP_COLLIDER_SIZE_M,
        translation=KA021_FAUCET_GRIP_COLLIDER_CENTER_LOCAL_M,
        orientation=KA021_FAUCET_MODEL_ORIENTATION_WXYZ,
    )

    joint_name = "RevoluteJoint_faucet_3_right"
    _author_joint(
        stage,
        path=f"{fixture_root_path}/{joint_name}",
        joint_type="revolute",
        body0=base_path,
        body1=handle_path,
        axis=KA021_FAUCET_OPEN_AXIS,
        local_pos0=_ka021_model_local_point(KA021_FAUCET_NATIVE_HANDLE_PIVOT_M),
        local_rot0_wxyz=KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ,
        local_rot1_wxyz=KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ,
        lower=KA021_FAUCET_OPEN_LIMIT_DEG,
        upper=0.0,
        damping=0.16,
        max_force=3.5,
    )

    cup_spec = next(spec for spec in task_spec.object_specs if spec.name == "ceramic_mug")
    cup_path, cup_colliders = _spawn_direct_object(stage, sim_utils, cup_spec)
    mug_wall_colliders = _replace_ka021_mug_collision(
        stage,
        sim_utils,
        mug_path=cup_path,
        source_scale=float(cup_spec.scale[0]),
    )
    flow_path = _spawn_ka021_flow_proxy(stage, sim_utils)
    appearance = None
    if task_spec.task_name == XY4_TASK_NAME:
        from .xy4_object_appearance import apply_xy4_object_appearance
        appearance = apply_xy4_object_appearance(
            stage, kitchen_path=kitchen_path,
            faucet_paths=(visual_path, handle_visual_path),
            mug_path=cup_path, flow_path=flow_path,
        )
    return B320SpawnResult(
        body_paths={
            "faucet_fixture": base_path,
            "faucet_base": base_path,
            "faucet_handle": handle_path,
            "ceramic_mug": cup_path,
            **({"hs005_bottle": f"{kitchen_path}/Bottle",
                "hs005_brush": f"{kitchen_path}/Brush"}
               if task_spec.task_name == XY4_TASK_NAME else {}),
        },
        articulation_root_paths=(articulation_root,),
        static_paths=(
            table_path,
            kitchen_path,
            safe_basin_floor_path,
            *counter_surface_collision_paths,
            park_support_path,
            park_marker_path,
            visual_path,
            outlet_path,
            flow_path,
        ),
        dof_names=(joint_name,),
        collider_counts={
            "faucet_fixture": 2 + len(spout_arch_collision_paths),
            "faucet_base": 2 + len(spout_arch_collision_paths),
            "faucet_handle": 1,
            "safe_open_basin": 1,
            "hs005_shared_scene_collision": len(
                shared_scene_collision_paths
            ),
            "hs005_counter_surface": len(counter_surface_collision_paths),
            "cup_park_support": 1,
            "ceramic_mug": int(cup_colliders + len(mug_wall_colliders)),
        },
        evidence={
            # ``asset_bundle`` remains the mug/HS005 provenance key for
            # backwards-compatible consumers; the actual faucet derivatives
            # are recorded explicitly beside it.
            "asset_bundle": KA021_ASSET_BUNDLE_KEY,
            **({"xy4_object_appearance": appearance} if appearance else {}),
            "faucet_asset_bundle": KA021_FAUCET_ASSET_BUNDLE_KEY,
            "faucet_asset_manifest": f"{KA021_FAUCET_ASSET_BUNDLE_KEY}/asset_manifest.json",
            "source_licenses": ["CC BY 4.0", KA021_FAUCET_SOURCE_LICENSE],
            "faucet_source_creator": KA021_FAUCET_SOURCE_CREATOR,
            "faucet_source_url": KA021_FAUCET_SOURCE_URL,
            "faucet_source_sha256": KA021_FAUCET_SOURCE_SHA256,
            "shared_scene": "B320-HS005",
            "reuses_hs005_realistic_kitchen": True,
            "hs005_scene_path": str(KA021_HS005_SCENE_PATH),
            "source_open_sink_collision": source_sink_collider_path,
            "source_open_sink_collision_enabled": True,
            "shared_scene_collision_policy": (
                KA021_SHARED_SCENE_COLLISION_POLICY
            ),
            "shared_scene_collisions": list(shared_scene_collision_paths),
            "safe_open_basin_floor_collision": safe_basin_floor_path,
            "safe_open_basin_floor_size_m": list(KA021_SAFE_BASIN_FLOOR_SIZE_M),
            "safe_open_basin_floor_center_w": list(
                KA021_SAFE_BASIN_FLOOR_CENTER_W
            ),
            "visible_basin_opening_bounds_xy_m": [
                list(values)
                for values in KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY
            ],
            "visible_basin_excludes_drainboard": True,
            "counter_surface_collisions": list(
                counter_surface_collision_paths
            ),
            "counter_surface_collision_specs": [
                {
                    "name": name,
                    "size_m": list(size),
                    "center_w": list(center),
                    "top_z_m": KA021_COUNTER_SURFACE_Z,
                }
                for name, size, center in KA021_COUNTER_SURFACE_COLLISION_SPECS
            ],
            "cup_park_support_collision": park_support_path,
            "cup_park_support_size_m": list(KA021_CUP_PARK_SUPPORT_SIZE_M),
            "cup_park_support_center_w": list(KA021_CUP_PARK_SUPPORT_CENTER_W),
            "cup_park_marker": park_marker_path,
            "cup_park_region": "outside_sink_left_counter",
            "cup_park_surface_z_m": KA021_COUNTER_SURFACE_Z,
            "cup_park_position_w": list(KA021_CUP_PARK_POSITION_W),
            "cup_park_tolerance_m": KA021_CUP_PARK_TOLERANCE_M,
            "cup_park_z_tolerance_m": KA021_CUP_PARK_Z_TOLERANCE_M,
            "fill_requires_faucet_open": True,
            "fill_hold_frames": task_spec.stable_frames,
            "fill_hold_seconds": task_spec.stable_frames / B320_CONTROL_HZ,
            "cup_park_condition_independent": True,
            "faucet_close_condition_independent": True,
            "mug_open_cavity_collision": True,
            "mug_open_cavity_colliders": list(mug_wall_colliders),
            "mug_handle_collision_radius_m": KA021_MUG_HANDLE_COLLIDER_RADIUS_M,
            "mug_handle_collision_centerline_source_m": [
                [list(start), list(end)]
                for start, end in KA021_MUG_HANDLE_CENTERLINE_SOURCE_M
            ],
            "source_faucet_articulated": False,
            "runtime_lever_authored": True,
            "runtime_handle_visual_uses_source_geometry": True,
            "runtime_split_method": "authored_object_boundary",
            "runtime_handle_native_pivot_m": list(
                KA021_FAUCET_NATIVE_HANDLE_PIVOT_M
            ),
            "runtime_handle_pivot_w": list(KA021_FAUCET_HANDLE_PIVOT_W),
            "runtime_base_position_w": list(KA021_FAUCET_BASE_POSITION_W),
            "runtime_visual_yaw_deg": KA021_FAUCET_MODEL_YAW_DEG,
            "runtime_visual_scale": KA021_FAUCET_MODEL_SCALE,
            "runtime_lever_gravity_balanced": True,
            "runtime_lever_open_axis": KA021_FAUCET_OPEN_AXIS,
            "runtime_joint_frame": "yawed_source_y_axis_with_sign_flip",
            "runtime_joint_frame_orientation_wxyz": list(
                KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ
            ),
            "runtime_joint_axis_world_w": list(KA021_FAUCET_JOINT_AXIS_WORLD_W),
            "runtime_initial_open_direction_world_w": list(
                KA021_FAUCET_INITIAL_OPEN_DIRECTION_W
            ),
            "runtime_lever_open_limit_deg": KA021_FAUCET_OPEN_LIMIT_DEG,
            "runtime_lever_reset_position_rad": KA021_FAUCET_RESET_POSITION_RAD,
            "runtime_lever_open_travel_rad": KA021_FAUCET_OPEN_TRAVEL_RAD,
            "runtime_lever_outward_direction_w": list(
                KA021_FAUCET_OUTWARD_DIRECTION_W
            ),
            "runtime_lateral_motion_role": "temperature_only_not_scored",
            "runtime_handle_grip_collider_size_m": list(
                KA021_FAUCET_GRIP_COLLIDER_SIZE_M
            ),
            "runtime_column_collider_radius_m": (
                KA021_FAUCET_COLUMN_COLLIDER_RADIUS_M
            ),
            "runtime_spout_arch_collider_paths": list(
                spout_arch_collision_paths
            ),
            "runtime_spout_arch_collider_radius_m": (
                KA021_FAUCET_ARCH_COLLIDER_RADIUS_M
            ),
            "runtime_spout_arch_collider_segments_w": [
                [list(start), list(end)]
                for start, end in KA021_FAUCET_ARCH_COLLIDER_SEGMENTS_W
            ],
            "runtime_spout_arch_clearance_probe_center_w": list(
                KA021_FAUCET_ARCH_CLEARANCE_PROBE_CENTER_W
            ),
            "runtime_spout_arch_clearance_probe_half_extents_m": list(
                KA021_FAUCET_ARCH_CLEARANCE_PROBE_HALF_EXTENTS_M
            ),
            "outlet_position_w": list(KA021_OUTLET_POSITION_W),
            "water_proxy": KA021_FLOW_PROXY_PATH,
        },
    )


def spawn_b320_task_assets(
    stage: Any,
    sim_utils: Any,
    task_name: str,
    *,
    bc011_pen_variant_index: int | None = None,
) -> B320SpawnResult:
    """Author one collision-valid B320 scene before the first PhysX reset."""

    if task_name == B320_BC017_TASK_NAME:
        raise RuntimeError(
            "B320-BC017 generic scene boundary is fail-closed; use "
            "b320_bc017_scene.spawn_bc017_task_assets through "
            "HandsOnB320BC017IsaacLabEnv"
        )

    from pxr import UsdGeom

    try:
        task_spec = B320_TASK_SPECS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported B320 task: {task_name!r}") from exc
    if task_name == XY6_TASK_NAME:
        from .eval_comb06_season_and_stir_scene import validate_asset_files

        validate_asset_files(task_spec)
    elif task_name == "b320_synchronized_double_cabinet_open":
        # FU001 is a self-contained selected-asset release.  Do not require
        # the historical 44781 source bundle just because the generic fixture
        # kind retains its old name for API compatibility.
        validate_b320_fu001_asset_files()
    elif task_name == XY4_TASK_NAME:
        from .xy4_scene import validate_xy4_asset_files
        validate_xy4_asset_files()
    elif task_spec.fixture_kind == "ka021_hs005_realistic_faucet":
        # Validate the new BlenderKit visual closure before composing any
        # kitchen prims.  The mug/HS005 source bundle is still checked by the
        # generic validator below; keeping the two gates separate preserves
        # its reviewed aggregate hash.
        validate_b320_ka021_faucet_asset_files()
        validate_b320_asset_files()
    elif task_spec.fixture_kind != "dual_knob_box_2":
        validate_b320_asset_files()
    if task_spec.fixture_kind == "cabinet_44781" and task_name != "b320_synchronized_double_cabinet_open":
        validate_b320_visual_dependencies(task_name)
    root = UsdGeom.Xform.Define(stage, B320_ASSET_ROOT_PATH).GetPrim()
    root.SetCustomDataByKey("handson:task", task_name)
    root.SetCustomDataByKey("handson:taskId", task_spec.task_id)

    if task_spec.fixture_kind == "gen6_audio_mixer":
        from .gen6_audio_mixer_scene import spawn_task_assets

        return spawn_task_assets(stage, sim_utils, task_spec=task_spec)
    if task_spec.fixture_kind == "dual_knob_box_2":
        return _spawn_dual_knob_box_2(stage, sim_utils, task_spec=task_spec)
    if task_spec.fixture_kind == "cabinet_44781":
        return _spawn_double_cabinet_fixture(
            stage,
            sim_utils,
            task_spec=task_spec,
        )
    if task_spec.fixture_kind == "ka021_hs005_realistic_faucet":
        return _spawn_ka021_hs005_faucet(
            stage,
            sim_utils,
            task_spec=task_spec,
        )

    static_paths = [_spawn_worktable(stage, sim_utils, task_name=task_name)]
    result = B320SpawnResult(static_paths=tuple(static_paths))
    for object_spec in task_spec.object_specs:
        if task_name == B320_KA020_TASK_NAME and object_spec.name == "soap_bottle":
            # KA020 keeps the downloaded bottle and pump head on separate
            # rigid links; the generic one-body importer would collapse the
            # visible pump into eleven unrelated convex hulls.
            continue
        if object_spec.name == "faucet_fixture":
            body_path, collider_count = _spawn_articulated_source(
                stage,
                sim_utils,
                object_spec,
            )
        else:
            body_path, collider_count = _spawn_direct_object(
                stage,
                sim_utils,
                object_spec,
                bc011_pen_variant_index=bc011_pen_variant_index,
            )
        if task_name == B320_KA020_TASK_NAME and object_spec.name == "sponge":
            sponge_collision = _replace_ka020_sponge_collision(
                stage,
                sim_utils,
                sponge_path=body_path,
            )
            collider_count = 1
            result.evidence["ka020_sponge_collision_path"] = sponge_collision
            result.evidence["ka020_sponge_collider_size_m"] = list(
                KA020_SPONGE_COLLIDER_SIZE_M
            )
            result.evidence["ka020_sponge_collider_center_m"] = list(
                KA020_SPONGE_COLLIDER_CENTER_M
            )
            result.evidence["ka020_sponge_render_component_count"] = (
                KA020_SPONGE_RENDER_COMPONENT_COUNT
            )
            result.evidence["ka020_sponge_render_triangle_count"] = (
                KA020_SPONGE_RENDER_TRIANGLE_COUNT
            )
            result.evidence["ka020_sponge_removed_holder_triangle_count"] = (
                KA020_SPONGE_REMOVED_HOLDER_TRIANGLE_COUNT
            )
            result.evidence["ka020_sponge_holder_geometry_present"] = False
        result.body_paths[object_spec.name] = body_path
        result.collider_counts[object_spec.name] = collider_count

    if task_name == XY6_TASK_NAME:
        from .xy6_object_appearance import apply_xy6_object_appearance

        result.evidence["xy6_object_appearance"] = apply_xy6_object_appearance(
            stage, body_paths=result.body_paths,
        )

    if task_name == B320_BC014_TASK_NAME:
        for name, role, color in (
            ("salt_goal", "salt_jar", (0.00, 0.72, 1.00)),
            ("pepper_goal", "pepper_jar", (1.00, 0.32, 0.02)),
        ):
            goal_x, goal_y = BC014_GOAL_CENTERS_XY[role]
            static_paths.append(
                _spawn_goal_marker(
                    stage,
                    sim_utils,
                    name=name,
                    translation=(
                        goal_x,
                        goal_y,
                        B320_TABLE_TOP_Z + 0.003,
                    ),
                    size=BC014_GOAL_ZONE_SIZE_M,
                    color=color,
                    highlight=True,
                )
            )
    if task_name in {B320_BC014_TASK_NAME, XY6_TASK_NAME}:
        for role in ("salt_jar", "pepper_jar"):
            static_paths.append(
                _spawn_bc014_flow_proxy(
                    stage,
                    sim_utils,
                    role=role,
                )
            )
    elif task_name == B320_KA020_TASK_NAME:
        bottle_spec = next(
            object_spec
            for object_spec in task_spec.object_specs
            if object_spec.name == "soap_bottle"
        )
        static_paths.append(
            _spawn_ka020_soap_dispenser(
                stage,
                sim_utils,
                result,
                bottle_spec=bottle_spec,
            )
        )
    result.static_paths = tuple(static_paths)
    result.dof_names = result.dof_names or task_spec.dof_names
    print(
        "[HandsOn/IsaacLab] B320 scene authored: "
        f"task={task_name}, bodies={result.body_paths}, "
        f"colliders={result.collider_counts}, dofs={result.dof_names}",
        flush=True,
    )
    return result


__all__ = [
    "BC014_ASSET_ARCHIVE_SHA256",
    "BC014_ASSET_BUNDLE_KEY",
    "BC014_ASSET_LICENSE",
    "BC014_ASSET_SOURCE_URL",
    "BC014_GOAL_CENTERS_XY",
    "BC014_FLOW_PROXY_PATHS",
    "BC014_GOAL_MARKER_PATHS",
    "BC014_GOAL_SPACING_M",
    "BC014_GOAL_ZONE_SIZE_M",
    "BC014_JAR_TABLE_CLEARANCE_M",
    "BC014_JAR_TARGET_HEIGHT_M",
    "B320_BC014_TASK_NAME",
    "B320_BC017_TASK_NAME",
    "B320_ASSET_DIRECTORY",
    "B320_ASSET_ROOT_PATH",
    "B320_BC004_SCENE_XML",
    "B320_CONTROL_HZ",
    "B320_KA005_2_ASSET_DIRECTORY",
    "B320_KA005_2_ASSET_MANIFEST_SHA256",
    "B320_KA005_2_TARGET_MARKER_PATHS",
    "B320_KA005_2_TASK_NAME",
    "B320_KA020_TASK_NAME",
    "B320_KA021_TASK_NAME",
    "B320_PHYSICS_HZ",
    "B320_RIGID_TASK_NAMES",
    "B320_SCENE_XML",
    "B320_SOURCE_BUNDLE_SHA256",
    "B320_SUCCESS_CONDITION_LABELS",
    "B320_TABLE_PATH",
    "B320_TABLE_TOP_Z",
    "B320_TASK_NAMES",
    "B320_TASK_SCENES",
    "B320_TASK_SPECS",
    "B320_TOWEL_TASK_NAME",
    "FU001_ASSET_ARCHIVE_SHA256",
    "FU001_ASSET_BUNDLE_KEY",
    "FU001_ASSET_DIRECTORY",
    "FU001_ASSET_LICENSE",
    "FU001_ASSET_LICENSE_URL",
    "FU001_ASSET_MANIFEST_PATH",
    "FU001_ASSET_SOURCE_ARCHIVE_PATH",
    "FU001_ASSET_SOURCE_URL",
    "FU001_DOOR_JOINT_LIMITS_DEG",
    "FU001_DOOR_SOURCE_NODES",
    "FU001_DOOR_SOURCE_PIVOTS",
    "FU001_HINGE_AXIS_RUNTIME",
    "FU001_HINGE_AXIS_SOURCE",
    "FU001_ROOT_HOME",
    "FU001_RUNTIME_COMPONENTS",
    "FU001_RUNTIME_DIRECTORY",
    "FU001_RUNTIME_ORIENTATION_WXYZ",
    "FU001_RUNTIME_SCALE",
    "KA020_DISPENSER_ASSET_BUNDLE_KEY",
    "KA020_DISPENSER_SOURCE_AUTHOR",
    "KA020_DISPENSER_SOURCE_URL",
    "KA020_DISPENSER_VISUAL_RELATIVE_USD",
    "KA020_DISPENSER_VISUAL_SCALE",
    "KA020_FLOW_PROXY_PATH",
    "KA020_MOVING_VISUAL_COMPONENT",
    "KA020_PUMP_HEAD_CENTER_LOCAL_M",
    "KA020_PUMP_HEAD_COLLIDER_SIZE_M",
    "KA020_PUMP_OUTLET_OFFSET_M",
    "KA020_PUMP_TRAVEL_M",
    "KA020_SOURCE_LICENSE",
    "KA020_SOURCE_LICENSE_URL",
    "KA020_SOURCE_ORIENTATION_WXYZ",
    "KA020_SPONGE_ASSET_BUNDLE_KEY",
    "KA020_SPONGE_COLLIDER_CENTER_M",
    "KA020_SPONGE_COLLIDER_SIZE_M",
    "KA020_SPONGE_PLAN_HALF_EXTENTS_M",
    "KA020_SPONGE_REMOVED_HOLDER_TRIANGLE_COUNT",
    "KA020_SPONGE_RENDER_COMPONENT_COUNT",
    "KA020_SPONGE_RENDER_TRIANGLE_COUNT",
    "KA020_SPONGE_RUNTIME_ORIENTATION_WXYZ",
    "KA020_SPONGE_SOURCE_AUTHOR",
    "KA020_SPONGE_SOURCE_URL",
    "KA020_SPONGE_THICKNESS_M",
    "KA020_SPONGE_VISUAL_RELATIVE_USD",
    "KA020_SPONGE_VISUAL_SCALE",
    "KA021_ASSET_BUNDLE_KEY",
    "KA021_FAUCET_ASSET_BUNDLE_KEY",
    "KA021_FAUCET_ASSET_DIRECTORY",
    "KA021_FAUCET_BASE_POSITION_W",
    "KA021_FAUCET_MODEL_ORIENTATION_WXYZ",
    "KA021_FAUCET_MODEL_SCALE",
    "KA021_FAUCET_MODEL_YAW_DEG",
    "KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ",
    "KA021_FAUCET_JOINT_AXIS_WORLD_W",
    "KA021_FAUCET_INITIAL_OPEN_DIRECTION_W",
    "KA021_FAUCET_NATIVE_ARCH_CENTERLINE_M",
    "KA021_FAUCET_HANDLE_PIVOT_W",
    "KA021_FAUCET_NATIVE_HANDLE_PIVOT_M",
    "KA021_FAUCET_NATIVE_OUTLET_M",
    "KA021_FAUCET_SOURCE_CREATOR",
    "KA021_FAUCET_SOURCE_LICENSE",
    "KA021_FAUCET_SOURCE_LICENSE_URL",
    "KA021_FAUCET_SOURCE_SHA256",
    "KA021_FAUCET_SOURCE_URL",
    "KA021_FAUCET_SOURCE_UUID",
    "KA021_FAUCET_SOURCE_VERSION_ID",
    "KA021_FAUCET_STATIC_VISUAL_RELATIVE_USD",
    "KA021_FAUCET_HANDLE_VISUAL_RELATIVE_USD",
    "KA021_FAUCET_STATIC_SHA256",
    "KA021_FAUCET_HANDLE_SHA256",
    "KA021_CUP_OPENING_RADIUS_M",
    "KA021_CUP_PARK_POSITION_W",
    "KA021_CUP_PARK_MARKER_PATH",
    "KA021_COUNTER_SURFACE_COLLISION_PATH",
    "KA021_COUNTER_SURFACE_COLLISION_SPECS",
    "KA021_COUNTER_SURFACE_Z",
    "KA021_CUP_PARK_SUPPORT_CENTER_W",
    "KA021_CUP_PARK_SUPPORT_PATH",
    "KA021_CUP_PARK_SUPPORT_SIZE_M",
    "KA021_CUP_PARK_TOLERANCE_M",
    "KA021_CUP_PARK_Z_TOLERANCE_M",
    "KA021_CUP_RIM_HEIGHT_M",
    "KA021_BASIN_FLOOR_TOP_Z",
    "KA021_HAND_CONTACT_OFFSET",
    "KA021_MUG_HANDLE_COLLIDER_RADIUS_M",
    "KA021_MUG_HANDLE_CENTERLINE_SOURCE_M",
    "KA021_CUP_UP_AXIS_LOCAL",
    "KA021_CUP_UPRIGHT_TOLERANCE_DEG",
    "KA021_FAUCET_ARCH_CLEARANCE_PROBE_CENTER_W",
    "KA021_FAUCET_ARCH_CLEARANCE_PROBE_HALF_EXTENTS_M",
    "KA021_FAUCET_ARCH_COLLIDER_RADIUS_M",
    "KA021_FAUCET_ARCH_COLLIDER_SEGMENTS_W",
    "KA021_FAUCET_COLUMN_COLLIDER_RADIUS_M",
    "KA021_FAUCET_GRIP_COLLIDER_CENTER_LOCAL_M",
    "KA021_FAUCET_GRIP_COLLIDER_SIZE_M",
    "KA021_FAUCET_OPEN_AXIS",
    "KA021_FAUCET_OPEN_LIMIT_DEG",
    "KA021_FAUCET_OPEN_TRAVEL_RAD",
    "KA021_FAUCET_OUTWARD_DIRECTION_W",
    "KA021_FAUCET_RESET_POSITION_RAD",
    "KA021_FLOW_PROXY_PATH",
    "KA021_FLOW_VISIBLE_FRACTION",
    "KA021_MAX_OUTLET_RIM_GAP_M",
    "KA021_MIN_OUTLET_RIM_GAP_M",
    "KA021_OUTLET_POSITION_W",
    "KA021_SAFE_BASIN_FLOOR_CENTER_W",
    "KA021_SAFE_BASIN_FLOOR_PATH",
    "KA021_SAFE_BASIN_FLOOR_SIZE_M",
    "KA021_SHARED_SCENE_COLLISION_POLICY",
    "KA021_SHARED_SCENE_COLLISION_ROOT_PATH",
    "KA021_SHARED_SCENE_COLLISION_SPECS",
    "KA021_SHARED_SCENE_ROOT_PATH",
    "KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY",
    "KA021_SUCCESS_OPEN_FRACTION",
    "B320ObjectSpec",
    "B320SpawnResult",
    "B320TaskSpec",
    "spawn_b320_task_assets",
    "validate_b320_asset_files",
    "validate_b320_fu001_asset_files",
    "validate_b320_ka005_2_asset_files",
    "validate_b320_ka021_faucet_asset_files",
    "validate_b320_visual_dependencies",
]
