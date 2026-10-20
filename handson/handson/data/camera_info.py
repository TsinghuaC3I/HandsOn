"""Camera metadata sidecars for saved HandsOn MP4 captures."""

from __future__ import annotations

import datetime as _datetime
import json
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FISHEYE_INTRINSICS_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "cameras"
    / "realtek_0bda_5883_fleet_mean_848x480.yaml"
)

FISHEYE_CAMERA_KEYS = {
    "wrist",
    "wrist_left",
    "wrist_right",
    "handcam_rgb",
    "handcam_rgb_left",
    "handcam_rgb_right",
    "wrist_fisheye_left",
    "wrist_fisheye_right",
}

FISHEYE_CAMERA_ALIASES = {
    "wrist": ["wrist_fisheye_right", "wrist_fisheye_left", "handcam_rgb"],
    "wrist_left": ["wrist_fisheye_left", "handcam_rgb_left"],
    "wrist_right": ["wrist_fisheye_right", "handcam_rgb_right"],
}

CAMERA_ID_ATTR_CANDIDATES = {
    "main": (
        "_main_camera_id",
        "_head_main_camera_id",
        "_head_camera_id",
        "_front_camera_id",
        "_ego_camera_id",
    ),
    "front": ("_front_camera_id",),
    "ego": ("_front_camera_id", "_ego_camera_id"),
    "random_camera": ("_front_camera_id",),
    "wrist": ("_wrist_camera_id", "_wrist_right_camera_id"),
    "wrist_left": ("_wrist_left_camera_id",),
    "wrist_right": ("_wrist_right_camera_id", "_wrist_camera_id"),
}

CAMERA_NAME_CANDIDATES = {
    "main": ("head_main", "front", "back", "ego"),
    "front": ("front", "back"),
    "ego": ("ego", "back", "front"),
    "random_camera": ("front", "back"),
    "wrist": ("wrist_fisheye_right", "handcam_rgb", "handcam_rgb_right"),
    "wrist_left": ("wrist_fisheye_left", "handcam_rgb_left"),
    "wrist_right": ("wrist_fisheye_right", "handcam_rgb_right", "handcam_rgb"),
}


@dataclass(frozen=True)
class FisheyeIntrinsics:
    path: Path
    model: str
    image_width: int
    image_height: int
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    new_K: np.ndarray | None
    balance: float | None
    fov_scale: float | None
    rms: float | None
    lens_center: np.ndarray | None
    lens_radius: float | None
    camera: dict[str, Any]


def _node_real(fs, name: str) -> float | None:
    node = fs.getNode(name)
    if node.empty():
        return None
    return float(node.real())


def _node_matrix(fs, name: str) -> np.ndarray | None:
    node = fs.getNode(name)
    if node.empty():
        return None
    mat = node.mat()
    if mat is None:
        return None
    return np.asarray(mat, dtype=np.float64)


def _node_string(node, default: str = "") -> str:
    if node.empty():
        return default
    value = node.string()
    return str(value) if value is not None else default


def load_fisheye_intrinsics(path: str | Path) -> FisheyeIntrinsics:
    """Load OpenCV FileStorage fisheye intrinsics YAML."""
    import cv2

    intrinsics_path = Path(path).expanduser()
    fs = cv2.FileStorage(str(intrinsics_path), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise FileNotFoundError(intrinsics_path)

        K = _node_matrix(fs, "K")
        D = _node_matrix(fs, "D")
        R = _node_matrix(fs, "R")
        new_K = _node_matrix(fs, "new_K")
        image_width = int(_node_real(fs, "image_width") or 0)
        image_height = int(_node_real(fs, "image_height") or 0)
        if K is None or D is None or image_width <= 0 or image_height <= 0:
            raise ValueError(f"could not parse K/D/image size from {intrinsics_path}")

        camera = {}
        camera_node = fs.getNode("camera")
        if not camera_node.empty():
            for key in ("name", "vendor_id", "product_id", "serial", "usb_port", "device"):
                value = _node_string(camera_node.getNode(key))
                if value:
                    camera[key] = value

        return FisheyeIntrinsics(
            path=intrinsics_path,
            model=_node_string(fs.getNode("model"), "fisheye"),
            image_width=image_width,
            image_height=image_height,
            K=K,
            D=D,
            R=R if R is not None else np.eye(3, dtype=np.float64),
            new_K=new_K,
            balance=_node_real(fs, "balance"),
            fov_scale=_node_real(fs, "fov_scale"),
            rms=_node_real(fs, "rms"),
            lens_center=_node_matrix(fs, "lens_center"),
            lens_radius=_node_real(fs, "lens_radius"),
            camera=camera,
        )
    finally:
        fs.release()


def _scale_camera_matrix(
    K: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]
) -> np.ndarray:
    source_w, source_h = source_size
    target_w, target_h = target_size
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[0, 2] = (out[0, 2] + 0.5) * sx - 0.5
    out[1, 1] *= sy
    out[1, 2] = (out[1, 2] + 0.5) * sy - 0.5
    return out


def _matrix_list(value: np.ndarray | None) -> list | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).tolist()


def _fov_deg(size_px: int, focal_px: float) -> float | None:
    if focal_px <= 0:
        return None
    return math.degrees(2.0 * math.atan(float(size_px) / (2.0 * float(focal_px))))


def _fisheye_theta_distorted(theta: float, distortion: np.ndarray) -> float:
    coeffs = np.zeros(4, dtype=np.float64)
    flat = np.asarray(distortion, dtype=np.float64).reshape(-1)
    coeffs[: min(4, flat.size)] = flat[:4]
    theta2 = theta * theta
    radial = (
        1.0
        + coeffs[0] * theta2
        + coeffs[1] * theta2 * theta2
        + coeffs[2] * theta2 * theta2 * theta2
        + coeffs[3] * theta2 * theta2 * theta2 * theta2
    )
    return theta * radial


def _fisheye_theta_from_distorted_radius(
    radius_over_focal: float, distortion: np.ndarray
) -> float:
    if radius_over_focal <= 0.0:
        return 0.0

    lo = 0.0
    hi = math.pi * 0.5
    if _fisheye_theta_distorted(hi, distortion) <= radius_over_focal:
        return hi

    for _ in range(80):
        mid = (lo + hi) * 0.5
        if _fisheye_theta_distorted(mid, distortion) < radius_over_focal:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _fisheye_axis_fov_deg(
    *,
    size_px: int,
    principal_px: float,
    focal_px: float,
    distortion: np.ndarray,
) -> float | None:
    if focal_px <= 0.0 or size_px <= 1:
        return None
    near_radius = max(float(principal_px), 0.0) / float(focal_px)
    far_radius = max(float(size_px - 1) - float(principal_px), 0.0) / float(focal_px)
    near_theta = _fisheye_theta_from_distorted_radius(near_radius, distortion)
    far_theta = _fisheye_theta_from_distorted_radius(far_radius, distortion)
    return math.degrees(near_theta + far_theta)


def fisheye_fov_degrees_from_intrinsics(
    intr: FisheyeIntrinsics,
) -> dict[str, float | None]:
    """Return angular coverage implied by an OpenCV fisheye calibration."""
    fx = float(intr.K[0, 0])
    fy = float(intr.K[1, 1])
    cx = float(intr.K[0, 2])
    cy = float(intr.K[1, 2])
    return {
        "horizontal": _fisheye_axis_fov_deg(
            size_px=intr.image_width,
            principal_px=cx,
            focal_px=fx,
            distortion=intr.D,
        ),
        "vertical": _fisheye_axis_fov_deg(
            size_px=intr.image_height,
            principal_px=cy,
            focal_px=fy,
            distortion=intr.D,
        ),
    }


def fisheye_mujoco_fovy_from_intrinsics(
    path: str | Path = DEFAULT_FISHEYE_INTRINSICS_PATH,
) -> float | None:
    """Best-effort MuJoCo pinhole fovy for a calibrated wrist fisheye camera."""
    intr = load_fisheye_intrinsics(path)
    fovs = fisheye_fov_degrees_from_intrinsics(intr)
    vertical = fovs.get("vertical")
    if vertical is not None and 0.0 < float(vertical) < 179.0:
        return float(vertical)
    return _fov_deg(intr.image_height, float(intr.K[1, 1]))


def _fisheye_theta_from_distorted_radius_array(
    radius_over_focal: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    coeffs = np.zeros(4, dtype=np.float64)
    flat = np.asarray(distortion, dtype=np.float64).reshape(-1)
    coeffs[: min(4, flat.size)] = flat[:4]

    radius = np.asarray(radius_over_focal, dtype=np.float64)
    lo = np.zeros_like(radius)
    hi = np.full_like(radius, math.pi * 0.5 - 1e-6)
    for _ in range(48):
        mid = (lo + hi) * 0.5
        theta2 = mid * mid
        theta_d = mid * (
            1.0
            + coeffs[0] * theta2
            + coeffs[1] * theta2 * theta2
            + coeffs[2] * theta2 * theta2 * theta2
            + coeffs[3] * theta2 * theta2 * theta2 * theta2
        )
        use_upper = theta_d < radius
        lo = np.where(use_upper, mid, lo)
        hi = np.where(use_upper, hi, mid)
    theta = (lo + hi) * 0.5
    return np.where(radius <= 1e-12, 0.0, theta)


@lru_cache(maxsize=16)
def _fisheye_rectilinear_remap(
    width: int,
    height: int,
    fovy_deg: float,
    intrinsics_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build maps from a MuJoCo pinhole frame into the calibrated fisheye frame."""
    intr = load_fisheye_intrinsics(intrinsics_path)
    target_size = (int(width), int(height))
    scaled_K = _scale_camera_matrix(
        intr.K, (intr.image_width, intr.image_height), target_size
    )

    fx = float(scaled_K[0, 0])
    fy = float(scaled_K[1, 1])
    cx = float(scaled_K[0, 2])
    cy = float(scaled_K[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"invalid fisheye focal length in {intrinsics_path}")

    yy, xx = np.indices((int(height), int(width)), dtype=np.float64)
    xd = (xx - cx) / fx
    yd = (yy - cy) / fy
    rd = np.sqrt(xd * xd + yd * yd)
    theta = _fisheye_theta_from_distorted_radius_array(rd, intr.D)

    half_angle = math.radians(float(fovy_deg)) * 0.5
    if half_angle <= 0.0 or half_angle >= math.pi * 0.5:
        raise ValueError(f"invalid source pinhole fovy: {fovy_deg}")
    source_focal = (float(height) * 0.5) / math.tan(half_angle)
    source_cx = (float(width) - 1.0) * 0.5
    source_cy = (float(height) - 1.0) * 0.5

    ray_radius = np.tan(theta)
    scale = np.zeros_like(rd)
    valid_radius = rd > 1e-12
    scale[valid_radius] = ray_radius[valid_radius] / rd[valid_radius]

    map_x = source_focal * (xd * scale) + source_cx
    map_y = source_focal * (yd * scale) + source_cy
    map_x[~valid_radius] = source_cx
    map_y[~valid_radius] = source_cy
    return map_x.astype(np.float32), map_y.astype(np.float32)


def fisheye_distort_rectilinear_frame(
    frame: np.ndarray,
    *,
    fovy_deg: float | None = None,
    intrinsics_path: str | Path = DEFAULT_FISHEYE_INTRINSICS_PATH,
) -> np.ndarray:
    """Apply calibrated OpenCV fisheye projection to a rendered pinhole RGB frame."""
    arr = np.asarray(frame)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return arr

    intrinsics = str(Path(intrinsics_path).expanduser())
    if fovy_deg is None:
        fovy_deg = fisheye_mujoco_fovy_from_intrinsics(intrinsics)
    if fovy_deg is None:
        return arr

    import cv2

    height, width = int(arr.shape[0]), int(arr.shape[1])
    map_x, map_y = _fisheye_rectilinear_remap(
        width,
        height,
        round(float(fovy_deg), 8),
        intrinsics,
    )
    return cv2.remap(
        np.ascontiguousarray(arr),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def fisheye_distort_rectilinear_depth(
    depth: np.ndarray,
    *,
    fovy_deg: float | None = None,
    intrinsics_path: str | Path = DEFAULT_FISHEYE_INTRINSICS_PATH,
) -> np.ndarray:
    """Apply the RGB fisheye remap to depth while preserving invalid pixels."""
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2:
        return array

    intrinsics = str(Path(intrinsics_path).expanduser())
    if fovy_deg is None:
        fovy_deg = fisheye_mujoco_fovy_from_intrinsics(intrinsics)
    if fovy_deg is None:
        return array

    import cv2

    height, width = int(array.shape[0]), int(array.shape[1])
    map_x, map_y = _fisheye_rectilinear_remap(
        width,
        height,
        round(float(fovy_deg), 8),
        intrinsics,
    )
    finite = np.isfinite(array).astype(np.uint8)
    filled = np.where(np.isfinite(array), array, 0.0).astype(
        np.float32,
        copy=False,
    )
    remapped = cv2.remap(
        np.ascontiguousarray(filled),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.remap(
        finite,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    remapped = np.asarray(remapped, dtype=np.float32)
    remapped[valid == 0] = np.inf
    return remapped


def _fisheye_intrinsics_dict(
    intr: FisheyeIntrinsics, target_size: tuple[int, int]
) -> dict[str, Any]:
    target_w, target_h = target_size
    source_size = (intr.image_width, intr.image_height)
    scaled_K = _scale_camera_matrix(intr.K, source_size, target_size)
    scaled_new_K = (
        _scale_camera_matrix(intr.new_K, source_size, target_size)
        if intr.new_K is not None
        else None
    )

    sx = float(target_w) / float(intr.image_width)
    sy = float(target_h) / float(intr.image_height)
    lens_center = None
    if intr.lens_center is not None:
        center = np.asarray(intr.lens_center, dtype=np.float64).reshape(-1)
        if center.size >= 2:
            lens_center = [float(center[0] * sx), float(center[1] * sy)]

    lens_radius = None
    if intr.lens_radius is not None:
        lens_radius = float(intr.lens_radius * ((sx + sy) * 0.5))

    angular_fov = fisheye_fov_degrees_from_intrinsics(intr)

    return {
        "source": str(intr.path),
        "model": intr.model,
        "distortion_model": "equidistant" if intr.model == "fisheye" else intr.model,
        "source_image_width": intr.image_width,
        "source_image_height": intr.image_height,
        "image_width": int(target_w),
        "image_height": int(target_h),
        "K": _matrix_list(scaled_K),
        "D": _matrix_list(np.asarray(intr.D, dtype=np.float64).reshape(-1)),
        "R": _matrix_list(intr.R),
        "new_K": _matrix_list(scaled_new_K),
        "raw_K": _matrix_list(intr.K),
        "raw_D": _matrix_list(np.asarray(intr.D, dtype=np.float64).reshape(-1)),
        "raw_new_K": _matrix_list(intr.new_K),
        "horizontal_fov_deg_from_K": _fov_deg(target_w, float(scaled_K[0, 0])),
        "vertical_fov_deg_from_K": _fov_deg(target_h, float(scaled_K[1, 1])),
        "horizontal_fov_deg_from_fisheye_model": angular_fov.get("horizontal"),
        "vertical_fov_deg_from_fisheye_model": angular_fov.get("vertical"),
        "mujoco_equivalent_fovy_deg": angular_fov.get("vertical"),
        "balance": intr.balance,
        "fov_scale": intr.fov_scale,
        "rms": intr.rms,
        "lens_center": lens_center,
        "lens_radius": lens_radius,
        "camera": intr.camera,
        "scaling": {
            "method": "nonuniform_scale_to_saved_video_frame",
            "scale_x": sx,
            "scale_y": sy,
        },
    }


def _pinhole_intrinsics_from_fovy(
    fovy_deg: float | None, target_size: tuple[int, int]
) -> dict[str, Any] | None:
    if fovy_deg is None:
        return None
    target_w, target_h = target_size
    half_angle = math.radians(float(fovy_deg)) * 0.5
    if half_angle <= 0.0 or half_angle >= math.pi * 0.5:
        return None
    focal = (float(target_h) * 0.5) / math.tan(half_angle)
    K = [
        [focal, 0.0, (float(target_w) - 1.0) * 0.5],
        [0.0, focal, (float(target_h) - 1.0) * 0.5],
        [0.0, 0.0, 1.0],
    ]
    return {
        "model": "mujoco_pinhole",
        "image_width": int(target_w),
        "image_height": int(target_h),
        "fovy_deg": float(fovy_deg),
        "K": K,
        "D": [0.0, 0.0, 0.0, 0.0],
        "horizontal_fov_deg": _fov_deg(target_w, focal),
        "vertical_fov_deg": float(fovy_deg),
    }


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _camera_id_by_name(model, name: str) -> int:
    try:
        return int(model.camera(name).id)
    except Exception:
        pass
    try:
        import mujoco

        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name))
    except Exception:
        return -1


def _mapped_camera_id(raw_env, key: str) -> int | None:
    mapping = getattr(raw_env, "camera_key_to_id", None)
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    try:
        return int(mapping[key])
    except Exception:
        return None


def _attr_camera_id(raw_env, key: str) -> int | None:
    for attr in CAMERA_ID_ATTR_CANDIDATES.get(key, ()):
        value = getattr(raw_env, attr, None)
        if value is None:
            continue
        try:
            cam_id = int(value)
        except Exception:
            continue
        if cam_id >= 0:
            return cam_id
    return None


def _named_camera_id(raw_env, key: str) -> int | None:
    model = getattr(raw_env, "_model", None) or getattr(raw_env, "model", None)
    if model is None:
        return None
    for name in CAMERA_NAME_CANDIDATES.get(key, ()):
        cam_id = _camera_id_by_name(model, name)
        if cam_id >= 0:
            return cam_id
    return None


def _camera_ids_from_keys(raw_env, image_keys: list[str]) -> tuple[int, ...] | None:
    cam_ids: list[int] = []
    for key in image_keys:
        cam_id = _mapped_camera_id(raw_env, key)
        if cam_id is None:
            cam_id = _attr_camera_id(raw_env, key)
        if cam_id is None:
            cam_id = _named_camera_id(raw_env, key)
        if cam_id is None or cam_id < 0:
            return None
        cam_ids.append(int(cam_id))
    return tuple(cam_ids)


def _env_viewer_and_camera_ids(raw_env, image_keys: list[str]):
    viewer = getattr(raw_env, "_viewer", None)
    if viewer is None:
        viewer = getattr(raw_env, "_mj_viewer", None)

    key_cam_ids = _camera_ids_from_keys(raw_env, image_keys)
    if key_cam_ids is not None:
        return viewer, key_cam_ids

    cam_ids = getattr(raw_env, "camera_id", None)
    if cam_ids is not None:
        cam_ids = tuple(int(c) for c in cam_ids)
    else:
        wrist = getattr(raw_env, "_wrist_camera_id", None)
        front = getattr(raw_env, "_front_camera_id", None)
        if wrist is None or front is None or len(image_keys) != 2:
            return viewer, None
        cam_ids = (int(wrist), int(front))

    if len(cam_ids) != len(image_keys):
        return viewer, None
    return viewer, cam_ids


def _mujoco_name(model, obj_type, obj_id: int) -> str | None:
    try:
        import mujoco

        return mujoco.mj_id2name(model, obj_type, int(obj_id))
    except Exception:
        return None


def describe_camera_views(env, image_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Return MuJoCo camera pose metadata keyed by saved video camera key."""
    raw_env = _unwrap_env(env)
    model = getattr(raw_env, "_model", None) or getattr(raw_env, "model", None)
    data = getattr(raw_env, "_data", None) or getattr(raw_env, "data", None)
    if model is None:
        return {}

    _viewer, cam_ids = _env_viewer_and_camera_ids(raw_env, image_keys)
    if cam_ids is None:
        return {}

    try:
        import mujoco
    except Exception:
        mujoco = None

    views: dict[str, dict[str, Any]] = {}
    for key, cam_id in zip(image_keys, cam_ids):
        if cam_id < 0:
            continue

        body_id = int(model.cam_bodyid[cam_id]) if hasattr(model, "cam_bodyid") else -1
        cam_name = None
        body_name = None
        if mujoco is not None:
            cam_name = _mujoco_name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            body_name = _mujoco_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        view = {
            "camera_id": int(cam_id),
            "mjcf_camera_name": cam_name,
            "equivalent_mjcf_camera_names": FISHEYE_CAMERA_ALIASES.get(key, []),
            "parent_body_id": body_id,
            "parent_body_name": body_name,
            "local_pos": np.asarray(model.cam_pos[cam_id], dtype=np.float64).tolist(),
            "local_quat_wxyz": np.asarray(model.cam_quat[cam_id], dtype=np.float64).tolist(),
            "fovy_deg": float(model.cam_fovy[cam_id]),
        }
        if data is not None and hasattr(data, "cam_xpos"):
            view["world_pos_at_save"] = np.asarray(
                data.cam_xpos[cam_id], dtype=np.float64
            ).tolist()
        if data is not None and hasattr(data, "cam_xmat"):
            view["world_xmat_at_save"] = np.asarray(
                data.cam_xmat[cam_id], dtype=np.float64
            ).reshape(3, 3).tolist()
        views[key] = view
    return views


def _is_fisheye_camera_key(camera_key: str, view: dict[str, Any] | None) -> bool:
    key = camera_key.lower()
    if key in FISHEYE_CAMERA_KEYS or "wrist" in key or "fisheye" in key:
        return True
    if view:
        name = str(view.get("mjcf_camera_name") or "").lower()
        if name in FISHEYE_CAMERA_KEYS or "wrist_fisheye" in name:
            return True
    return False


def _saved_wrist_profile(
    wrist_profile: dict[str, Any], camera_key: str
) -> dict[str, Any]:
    """Resolve a serial-specific rectified projection when one is recorded."""

    if wrist_profile.get("fleet_mean_intrinsics", True):
        return wrist_profile
    key = camera_key.lower()
    side = "left" if "left" in key else "right" if "right" in key else None
    per_side = wrist_profile.get("intrinsics_by_side", {})
    selected = per_side.get(side, {}) if side is not None else {}
    if not selected:
        return wrist_profile
    merged = dict(wrist_profile)
    merged.update(selected)
    return merged


def write_video_camera_info(
    *,
    videos_dir: str | Path,
    camera_keys: list[str],
    first_frames: dict[str, np.ndarray],
    intrinsics_yaml: str | Path | None,
    camera_views: dict[str, dict[str, Any]] | None = None,
    camera_profile: dict[str, Any] | None = None,
) -> list[str]:
    """Write aggregate and per-video JSON camera sidecars."""
    videos_path = Path(videos_dir)
    videos_path.mkdir(parents=True, exist_ok=True)
    camera_views = camera_views or {}
    camera_profile = camera_profile or {}

    intrinsics = None
    intrinsics_error = None
    if intrinsics_yaml:
        try:
            intrinsics = load_fisheye_intrinsics(intrinsics_yaml)
        except Exception as exc:
            intrinsics_error = str(exc)

    generated_at = _datetime.datetime.now(_datetime.UTC).isoformat()
    aggregate: dict[str, Any] = {
        "schema": "handson_video_camera_info.v1",
        "generated_at": generated_at,
        "camera_info": {},
    }
    if intrinsics_error:
        aggregate["fisheye_intrinsics_warning"] = intrinsics_error

    saved_paths: list[str] = []
    for camera_key in camera_keys:
        frame = first_frames.get(camera_key)
        if frame is None:
            continue
        frame_arr = np.asarray(frame)
        if frame_arr.ndim < 2:
            continue

        height, width = int(frame_arr.shape[0]), int(frame_arr.shape[1])
        view = camera_views.get(camera_key, {})
        target_size = (width, height)
        is_wrist = _is_fisheye_camera_key(camera_key, view)
        head_profile = camera_profile.get("head", {})
        wrist_profile = camera_profile.get("wrist", {})
        saved_wrist_profile = _saved_wrist_profile(wrist_profile, camera_key)
        fovy_deg = view.get("fovy_deg")
        if fovy_deg is None and not is_wrist:
            fovy_deg = head_profile.get("vertical_fov_deg")
        info: dict[str, Any] = {
            "schema": "handson_video_camera_info.v1",
            "generated_at": generated_at,
            "camera_key": camera_key,
            "video_file": f"{camera_key}.mp4",
            "frame": {"width": width, "height": height},
            "sim_view": view,
            "sim_pinhole_intrinsics": _pinhole_intrinsics_from_fovy(
                fovy_deg,
                target_size,
            ),
        }
        profile_version = camera_profile.get("version")
        if profile_version:
            info["recording_camera_profile_version"] = profile_version
        if (
            is_wrist
            and saved_wrist_profile.get("rendered_frames_are_already_rectified")
        ):
            info["saved_projection"] = {
                "model": saved_wrist_profile.get("output_projection", "pinhole"),
                "image_width": width,
                "image_height": height,
                "horizontal_fov_deg": saved_wrist_profile.get(
                    "horizontal_fov_deg"
                ),
                "K": saved_wrist_profile.get("K_new"),
                "D": [0.0, 0.0, 0.0, 0.0],
                "already_rectified": True,
                "must_not_rectify_again": True,
            }

        if intrinsics is not None and is_wrist:
            info["physical_fisheye_intrinsics"] = _fisheye_intrinsics_dict(
                intrinsics, target_size
            )

        sidecar_path = videos_path / f"{camera_key}_camera_info.json"
        sidecar_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved_paths.append(str(sidecar_path))
        aggregate["camera_info"][camera_key] = info

    aggregate_path = videos_path / "camera_info.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    saved_paths.append(str(aggregate_path))
    return saved_paths
