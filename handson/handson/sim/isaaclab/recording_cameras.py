"""Shared production camera geometry for native Isaac collection."""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import numpy as np


CAMERA_PROFILE_ENV = "HANDSON_ISAACSIM_CAMERA_PROFILE"
LEGACY_CAMERA_PROFILE_NAME = "legacy_v6"
REAL_V1_CAMERA_PROFILE_NAME = "real_v1"
DEFAULT_CAMERA_PROFILE_NAME = "handson"


def _camera_profile_name_from_env() -> str:
    raw_value = os.environ.get(CAMERA_PROFILE_ENV)
    value = (raw_value if raw_value is not None else DEFAULT_CAMERA_PROFILE_NAME)
    value = value.strip().lower()
    if value not in {"", DEFAULT_CAMERA_PROFILE_NAME}:
        raise ValueError(f"{CAMERA_PROFILE_ENV} no longer selects an alternate camera; remove it for HandsOn")
    return DEFAULT_CAMERA_PROFILE_NAME


CAMERA_PROFILE_NAME = _camera_profile_name_from_env()
CAMERA_PROFILE_VERSION = (
    "camera_pose_v7_real_v1"
    if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
    else "camera_pose_v6"
)
HEAD_RECORDING_CAMERA_EYE = (-0.675, 0.0, 1.5885)
HEAD_REAL_V1_CAMERA_TARGET = (-0.15, 0.0, 1.0)
HEAD_LEGACY_CAMERA_QUAT_WXYZ_OPENGL = (
    0.6221650,
    0.3360220,
    -0.3360220,
    -0.6221650,
)
# Real-v1 keeps the established optical origin but aims the optical axis at the
# tabletop work area (-0.15, 0.0, 1.00).  The legacy axis intersects z=1.0 at
# x=+0.22, which leaves over half of a 16:9 frame above the real tabletop.
HEAD_REAL_V1_CAMERA_QUAT_WXYZ_OPENGL = (
    0.6607228661203985,
    0.25188349327743953,
    -0.25188349327743953,
    -0.6607228661203985,
)
HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL = (
    HEAD_REAL_V1_CAMERA_QUAT_WXYZ_OPENGL
    if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
    else HEAD_LEGACY_CAMERA_QUAT_WXYZ_OPENGL
)
# The real-v1 head projection comes from the Gemini 336 color calibration
# exported for device CP96563000E6 at 1920x1080.  The production stream is
# already rectified, so Isaac renders the equivalent pinhole projection.
HEAD_GEMINI336_COLOR_CALIBRATION_SIZE = (1920, 1080)
HEAD_GEMINI336_COLOR_K = (
    (1034.736938477, 0.0, 967.177856445),
    (0.0, 1034.913818359, 546.876098633),
    (0.0, 0.0, 1.0),
)
HEAD_GEMINI336_COLOR_D = (
    0.00365191837773,
    -0.0413195490837,
    0.000175553956069,
    0.0000243501162913,
    0.0290885977447,
    0.0,
    0.0,
    0.0,
)
HEAD_GEMINI336_COLOR_HFOV_DEG = 85.7085979243538
HEAD_GEMINI336_COLOR_VFOV_DEG = 55.109536510547784
HEAD_RECORDING_CAMERA_FOVY_DEG = (
    HEAD_GEMINI336_COLOR_VFOV_DEG
    if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
    else 80.0
)
HEAD_RECORDING_CAMERA_HFOV_DEG_AT_16_9 = (
    HEAD_GEMINI336_COLOR_HFOV_DEG
    if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
    else math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(HEAD_RECORDING_CAMERA_FOVY_DEG) / 2.0)
            * (16.0 / 9.0)
        )
    )
)

# Kit's free perspective camera cannot safely use the physical head-camera
# optical origin because that point sits inside the imported center-head
# visual envelope.  Keep the interactive/default viewport behind the robot
# and aimed in the same robot-to-table (+X) direction instead.  This is an
# operator framing aid only; recorded ``main`` frames continue to use the
# calibrated physical pose above.
OPERATOR_VIEWPORT_CAMERA_EYE = (-1.35, 0.0, 1.58)
OPERATOR_VIEWPORT_CAMERA_TARGET = (-0.12, 0.0, 1.02)

# D09 is a front-facing coffee station.  Aim between the back cabinet, the
# machine, and the serving tray; the old target sat on the cabinet and pushed
# the machine/tray to the bottom edge of the opening view.
D09_OPERATOR_VIEWPORT_CAMERA_EYE = (-0.88, -1.12, 1.34)
D09_OPERATOR_VIEWPORT_CAMERA_TARGET = (-0.16, 0.08, 1.00)


def _wrist_camera_output_size_from_env(
    default: tuple[int, int] = (848, 480),
) -> tuple[int, int]:
    raw = os.environ.get("HANDSON_ISAACLAB_WRIST_CAMERA_OUTPUT_SIZE", "").strip()
    if not raw:
        return default
    normalized = raw.lower().replace("×", "x").replace(",", "x")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) != 2:
        return default
    try:
        width, height = (int(parts[0]), int(parts[1]))
    except ValueError:
        return default
    if width <= 0 or height <= 0:
        return default
    return width, height


WRIST_RECORDING_CAMERA_OUTPUT_SIZE = _wrist_camera_output_size_from_env()
WRIST_RECORDING_CAMERA_NEAR_M = 0.03
RECORDING_CAMERA_VERTICAL_APERTURE = 24.0

# This tabletop rectangle is inside camera_pose_v6 with at least 15% vertical
# and 29% horizontal image margin for object centers between z=0.89 and 1.08 m.
# Every corner is also within 2 cm of the shipped ROKAE+Wuji reachability point
# cloud for at least one arm at the nominal grasp height.
RECORDING_OBJECT_XY_BOUNDS = np.asarray(
    [
        [-0.40, -0.30],
        [-0.12, 0.30],
    ],
    dtype=np.float64,
)

WRIST_REALTEK_FISHEYE_K_1920X1080 = (
    (613.8387, 0.0, 960.4092),
    (0.0, 613.4801, 547.4983),
    (0.0, 0.0, 1.0),
)
# Pose of each optical frame relative to the imported circular wrist-camera
# mount body. Quaternions use wxyz and the camera convention is OpenGL.
WRIST_LEGACY_CAMERA_MOUNT_POSES = {
    "left": np.asarray(
        [
            -0.048382115059890526,
            -0.000546846019311461,
            -0.0688498020281539,
            0.1321841588,
            0.9382781453,
            -0.3182675885,
            -0.0294484690,
        ],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [
            0.04838211451979743,
            -0.0005468457310754686,
            -0.06884980202764532,
            0.1854105380,
            0.9291137054,
            0.2905687037,
            -0.1339420892,
        ],
        dtype=np.float64,
    ),
}

# The real-v1 optical centers are moved 10% toward the palm origin while
# preserving the asset-aligned V6 optical axes.  Open-palm silhouettes in the
# serial-bound real recordings are about 10% larger than the V6 projection;
# this 6.7 mm correction matches that rigid palm scale without narrowing the
# independently calibrated Realtek field of view.
WRIST_REAL_V1_CAMERA_MOUNT_POSES = {
    "left": np.asarray(
        [
            -0.04883645289013458,
            -0.007057225869637224,
            -0.0702918302891323,
            0.1321841588,
            0.9382781453,
            -0.3182675885,
            -0.0294484690,
        ],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [
            0.04801127166738761,
            -0.007219660160393951,
            -0.06893309269743961,
            0.1854105380,
            0.9291137054,
            0.2905687037,
            -0.1339420892,
        ],
        dtype=np.float64,
    ),
}
WRIST_RECORDING_CAMERA_MOUNT_POSES = (
    WRIST_REAL_V1_CAMERA_MOUNT_POSES
    if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
    else WRIST_LEGACY_CAMERA_MOUNT_POSES
)

# Fallback poses relative to the palm frame for runtimes where the imported
# circular mount body cannot be resolved.
WRIST_LEGACY_CAMERA_LOCAL_POSES = {
    "left": np.asarray(
        [
            0.03816295,
            0.03816295,
            0.03942423,
            0.0868936043,
            -0.7568897559,
            0.6477050058,
            -0.0067545863,
        ],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [
            0.03816295,
            -0.03816295,
            0.03942423,
            0.0067467084,
            -0.6477736234,
            0.7568310313,
            -0.0868942163,
        ],
        dtype=np.float64,
    ),
}
WRIST_REAL_V1_CAMERA_LOCAL_POSES = {
    side: np.concatenate((pose[:3] * 0.9, pose[3:]))
    for side, pose in WRIST_LEGACY_CAMERA_LOCAL_POSES.items()
}
WRIST_RECORDING_CAMERA_LOCAL_POSES = (
    WRIST_REAL_V1_CAMERA_LOCAL_POSES
    if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
    else WRIST_LEGACY_CAMERA_LOCAL_POSES
)

WRIST_REALTEK_FISHEYE_K_848X480 = (
    (271.112092500, 0.0, 423.901563333),
    (0.0, 272.657822222, 243.054800000),
    (0.0, 0.0, 1.0),
)
WRIST_REALTEK_FISHEYE_D = (-0.185015, 0.014974, -0.000507, 0.0)
WRIST_UNDISTORT_K_NEW_848X480 = (
    (355.778243619, 0.0, 423.5),
    (0.0, 355.778243619, 239.5),
    (0.0, 0.0, 1.0),
)

# The real robot binds these two cameras by serial number.  Their 2560x1440
# lens-center-aligned streams are rectified independently before recording;
# scaling to 848x480 preserves each camera's measured angular coverage.
WRIST_REAL_V1_INTRINSICS_BY_SIDE = {
    "left": {
        "serial": "YHT61A5530378DF4B220C918DBD",
        "calibration_resolution": (2560, 1440),
        "K_calibration": (
            (821.5402825803739, 0.0, 1279.5),
            (0.0, 821.6158728058153, 719.5),
            (0.0, 0.0, 1.0),
        ),
        "D": (-0.194086311788323, 0.019631590042487743, -0.0011301287790118795, 0.0),
        "K_new_848x480": (
            (272.13521860474884, 0.0, 423.5),
            (0.0, 273.87195760193845, 239.5),
            (0.0, 0.0, 1.0),
        ),
        "horizontal_fov_deg": 114.61300428236564,
        "vertical_fov_deg": 82.45758869076539,
    },
    "right": {
        "serial": "YHTC6BDA658F9FFC8594A329130",
        "calibration_resolution": (2560, 1440),
        "K_calibration": (
            (806.0585365379185, 0.0, 1279.5),
            (0.0, 806.4790027474348, 719.5),
            (0.0, 0.0, 1.0),
        ),
        "D": (-0.17994672050478522, 0.01798947122078772, -0.0024138934981154324, 0.0),
        "K_new_848x480": (
            (267.0068902281855, 0.0, 423.5),
            (0.0, 268.8263342491449, 239.5),
            (0.0, 0.0, 1.0),
        ),
        "horizontal_fov_deg": 115.60003092730064,
        "vertical_fov_deg": 83.51502231749416,
    },
}

WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE = {
    side: (
        float(values["horizontal_fov_deg"])
        if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
        else 100.0
    )
    for side, values in WRIST_REAL_V1_INTRINSICS_BY_SIDE.items()
}
# Compatibility scalar for metadata consumers that predate per-side profiles.
WRIST_RECORDING_CAMERA_HFOV_DEG = float(
    np.mean(tuple(WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE.values()))
)


# The released calibration is shared with the native lens schema.
from .calibrated_cameras import profile as _calibration_profile, metadata as _calibration_metadata
_CALIBRATION = _calibration_profile()
CAMERA_PROFILE_VERSION = "handson"
HEAD_RECORDING_CAMERA_EYE = tuple(_CALIBRATION["position_m"])
HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL = tuple(_CALIBRATION["quaternion_wxyz_opengl"])
HEAD_RECORDING_CAMERA_FOVY_DEG = math.degrees(2 * math.atan(480 / (2 * _CALIBRATION["runtime_K"][1][1])))
HEAD_RECORDING_CAMERA_HFOV_DEG_AT_16_9 = math.degrees(2 * math.atan(854 / (2 * _CALIBRATION["runtime_K"][0][0])))
WRIST_RECORDING_CAMERA_LOCAL_POSES = {s: np.asarray(p, dtype=np.float64) for s, p in {'left': [np.float64(0.06877913524957757), np.float64(0.029143365994456466), np.float64(-0.034815742831402215), np.float64(0.11603727405557109), np.float64(0.4215872822962284), np.float64(0.8964224363769866), np.float64(0.07229336066182039)], 'right': [np.float64(0.06378304461597961), np.float64(-0.022633119403028243), np.float64(-0.03375643894785378), np.float64(0.0814413420599562), np.float64(0.8905403202224125), np.float64(0.4447701080241547), np.float64(0.049847736857536865)]}.items()}
WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE = {s: float(v["horizontal_fov_deg"]) for s, v in WRIST_REAL_V1_INTRINSICS_BY_SIDE.items()}
WRIST_RECORDING_CAMERA_HFOV_DEG = float(np.mean(tuple(WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE.values())))


def constrain_recording_object_xy(xy: np.ndarray) -> np.ndarray:
    """Clamp tabletop object centers to the V6 camera/arm shared workspace."""
    value = np.asarray(xy, dtype=np.float64)
    return np.clip(
        value,
        RECORDING_OBJECT_XY_BOUNDS[0],
        RECORDING_OBJECT_XY_BOUNDS[1],
    )


def focal_length_for_vertical_fov(
    fovy_degrees: float,
    vertical_aperture: float = RECORDING_CAMERA_VERTICAL_APERTURE,
) -> float:
    return float(
        vertical_aperture
        / (2.0 * math.tan(math.radians(float(fovy_degrees)) / 2.0))
    )


def focal_length_for_horizontal_fov(
    hfov_degrees: float,
    horizontal_aperture: float,
) -> float:
    return float(
        horizontal_aperture
        / (2.0 * math.tan(math.radians(float(hfov_degrees)) / 2.0))
    )


def _inherited_recording_camera_profile() -> dict[str, Any]:
    """Return JSON-safe metadata for every newly collected Isaac episode."""
    return {
        "version": CAMERA_PROFILE_VERSION,
        "name": CAMERA_PROFILE_NAME,
        "default_for_new_collection": (
            CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
        ),
        "legacy_compatibility_only": (
            CAMERA_PROFILE_NAME == LEGACY_CAMERA_PROFILE_NAME
        ),
        "head": {
            "device": "Orbbec Gemini 336",
            "production_device_serial": "CP96563000CF",
            "calibration_device_serial": "CP96563000E6",
            "calibration_is_same_model_proxy": True,
            "calibration_applied": (
                CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
            ),
            "projection": "pinhole",
            "eye_xyz_m": list(HEAD_RECORDING_CAMERA_EYE),
            "quat_wxyz_opengl": list(HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL),
            "real_v1_alignment_target_xyz_m": (
                list(HEAD_REAL_V1_CAMERA_TARGET)
                if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
                else None
            ),
            "extrinsic_source": (
                "manual_alignment_against_real_head_reference"
                if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
                else "legacy_v6"
            ),
            "vertical_fov_deg": HEAD_RECORDING_CAMERA_FOVY_DEG,
            "horizontal_fov_deg_at_16_9": (
                HEAD_RECORDING_CAMERA_HFOV_DEG_AT_16_9
            ),
            "calibration_resolution": list(HEAD_GEMINI336_COLOR_CALIBRATION_SIZE),
            "K_calibration": [list(row) for row in HEAD_GEMINI336_COLOR_K],
            "D_calibration": list(HEAD_GEMINI336_COLOR_D),
            "factory_stream_is_rectified": True,
            "fisheye_undistortion": False,
            "principal_point_runtime": "centered_omniverse_pinhole",
            "calibrated_principal_point_offset_unsupported": True,
        },
        "wrist": {
            "device": "Realtek 0bda:5883",
            "lens": "approximately 180-degree fisheye",
            "distortion_model": "opencv_fisheye",
            "fleet_mean_intrinsics": CAMERA_PROFILE_NAME == LEGACY_CAMERA_PROFILE_NAME,
            "calibration_resolution": [1920, 1080],
            "K_calibration": [
                list(row)
                for row in WRIST_REALTEK_FISHEYE_K_1920X1080
            ],
            "raw_input_resolution": list(WRIST_RECORDING_CAMERA_OUTPUT_SIZE),
            "output_resolution": list(WRIST_RECORDING_CAMERA_OUTPUT_SIZE),
            "output_projection": "pinhole",
            "horizontal_fov_deg": WRIST_RECORDING_CAMERA_HFOV_DEG,
            "rendered_frames_are_already_rectified": True,
            "K_scaled": [list(row) for row in WRIST_REALTEK_FISHEYE_K_848X480],
            "D": list(WRIST_REALTEK_FISHEYE_D),
            "K_new": [list(row) for row in WRIST_UNDISTORT_K_NEW_848X480],
            "intrinsics_by_side": {
                side: {
                    "serial": values["serial"],
                    "calibration_resolution": list(values["calibration_resolution"]),
                    "K_calibration": [list(row) for row in values["K_calibration"]],
                    "D": list(values["D"]),
                    "K_new": [list(row) for row in values["K_new_848x480"]],
                    "horizontal_fov_deg": float(values["horizontal_fov_deg"]),
                    "vertical_fov_deg": float(values["vertical_fov_deg"]),
                }
                for side, values in WRIST_REAL_V1_INTRINSICS_BY_SIDE.items()
            },
            "mount_pose_xyz_quat_wxyz_by_side": {
                side: pose.tolist()
                for side, pose in WRIST_RECORDING_CAMERA_MOUNT_POSES.items()
            },
            "extrinsic_source": (
                "real_open_palm_silhouette_alignment"
                if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
                else "legacy_v6"
            ),
            "palm_distance_scale_vs_legacy": (
                0.9
                if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
                else 1.0
            ),
            "near_clip_m": WRIST_RECORDING_CAMERA_NEAR_M,
        },
        "visibility": {
            "main": (
                "wrist housings, adapters, and link-7 shells visible"
                if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
                else "shared low-latency obstruction mask"
            ),
            "wrist": "near-field camera housing and connector visuals hidden",
            "default_capture_strategy": (
                "split_head_wrist_render_groups"
                if CAMERA_PROFILE_NAME == DEFAULT_CAMERA_PROFILE_NAME
                else "single_shared_render_group"
            ),
            "override_env": "HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY",
        },
    }


def recording_camera_profile() -> dict[str, Any]:
    result = _calibration_metadata(_inherited_recording_camera_profile())
    result["head"].update(projection="opencvPinhole", calibrated_principal_point_offset_unsupported=False)
    result["head"].pop("real_v1_alignment_target_xyz_m", None)
    result["wrist"].update(extrinsic_source="calibrated_camera_to_palm", camera_to_palm_opengl=_CALIBRATION["wrist_camera_to_palm_opengl"])
    result["wrist"].pop("mount_pose_xyz_quat_wxyz_by_side", None)
    result["wrist"].pop("palm_distance_scale_vs_legacy", None)
    return result


def wrist_camera_obstruction_paths(
    stage,
    root_path: str = "/World",
) -> tuple[str, ...]:
    """Return visual prims hidden only while wrist sensors capture a frame."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return ()
    debug = os.environ.get("HANDSON_DEBUG_WRIST_PRIMS") == "1"
    paths: list[str] = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        path = str(prim.GetPath()).lower()
        is_wrist_obstruction = (
            "wrist_fisheye_flange_assembly" in name
            or ("wrist_fisheye" in path and "flange_assembly" in path)
            or (
                (
                    "wuji_direct_adapter_mount" in path
                    or "direct-adapter-mount" in path
                    or "direct_adapter" in path
                )
                and "visual" in path
            )
            or (
                (
                    "link7_left" in path
                    or "link7_right" in path
                    or "rokae_link7" in path
                )
                and "visual" in path
            )
            or path.endswith("/attachment_left/visuals")
            or path.endswith("/attachment_right/visuals")
            or path.endswith("/left_wrist_fisheye_mount/visuals")
            or path.endswith("/right_wrist_fisheye_mount/visuals")
        )
        if not is_wrist_obstruction:
            continue
        # Visibility changes on collision/rigid prims (or their parents)
        # trigger a USD resync in Isaac Sim 6.  Once PhysX tensor views exist,
        # that resync invalidates every articulation view.  Hide only pure
        # visual subtrees; collision housings stay present and simply do not
        # participate in the wrist-camera obstruction mask.
        contains_physics = "collision" in path or any(
            descendant.HasAPI(UsdPhysics.CollisionAPI)
            or descendant.HasAPI(UsdPhysics.RigidBodyAPI)
            or descendant.IsA(UsdPhysics.Joint)
            or "collision" in str(descendant.GetPath()).lower()
            for descendant in Usd.PrimRange(
                prim,
                Usd.TraverseInstanceProxies(),
            )
        )
        if contains_physics:
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        if debug:
            print(
                f"[HandsOn/camera/debug] wrist-only obstruction {prim.GetPath()}",
                flush=True,
            )
        paths.append(str(prim.GetPath()))
    return tuple(dict.fromkeys(paths))


def wrist_obstruction_visibility_records(
    stage,
) -> tuple[tuple[str, Any], ...]:
    from pxr import UsdGeom

    records = []
    for path in wrist_camera_obstruction_paths(stage):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        visibility = imageable.GetVisibilityAttr().Get()
        records.append((path, visibility or UsdGeom.Tokens.inherited))
    return tuple(records)


def set_wrist_obstructions_hidden(
    stage,
    records: tuple[tuple[str, Any], ...],
    hidden: bool,
) -> None:
    from pxr import UsdGeom

    for path, original in records:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        visibility = UsdGeom.Imageable(prim).GetVisibilityAttr()
        visibility.Set(UsdGeom.Tokens.invisible if hidden else original)
