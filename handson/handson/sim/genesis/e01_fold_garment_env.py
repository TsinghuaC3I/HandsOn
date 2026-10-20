"""Genesis FEM.Cloth + libuipc environment for E01 garment folding."""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..isaaclab.e_collection_scene import (
    E01_TASK_NAME,
    garmentlab_asset_root_candidates,
)
from .mjcf_assets import prepare_detached_rokae_wuji_hand_mjcf


_TABLE_TOP_Z = 0.89
_TABLE_X_BOUNDS = (-0.55, 0.25)
_TABLE_Y_BOUNDS = (-0.80, 0.80)
_FOLD_RATIO_THRESHOLD = 0.48
_AREA_RATIO_THRESHOLD = 0.70
_HAND_RELEASE_DISTANCE = 0.065
_STABLE_RMS_SPEED = 0.035
_STABLE_MAX_SPEED = 0.20
_SUCCESS_STEPS = 15
_IMAGE_SHAPE = (640, 640, 3)
_HAND_DIM = 20
_HAND_TABLE_CLEARANCE = 0.001
_ROKAE_REACH_RADIUS = 0.88
_ROKAE_REACH_MAX_POINT_DIST = 0.08
_ROKAE_REACH_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "envs"
    / "c01_rokae_wuji_reachability.npz"
)
_WUJI_HAND_MJCF_DIR = (
    Path(__file__).resolve().parents[1] / "envs" / "xmls" / "wuji_hand1" / "mjcf"
)
_ROKAE_ARM_ROOT_POS = {
    "right": np.asarray([-0.75, -0.065, 1.34], dtype=np.float64),
    "left": np.asarray([-0.75, 0.065, 1.34], dtype=np.float64),
}
_ROKAE_HAND_WORKSPACE_BOUNDS = {
    "right": (
        np.asarray([-0.42, -0.72, _TABLE_TOP_Z + 0.001], dtype=np.float64),
        np.asarray([0.18, -0.02, 1.38], dtype=np.float64),
    ),
    "left": (
        np.asarray([-0.42, 0.02, _TABLE_TOP_Z + 0.001], dtype=np.float64),
        np.asarray([0.18, 0.72, 1.38], dtype=np.float64),
    ),
}
_GENESIS_GARMENT_RELATIVE_PATH = Path(
    "Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top578/"
    "TCLC_Top578_genesis.obj"
)
# This is the exact Cotton_40s_Poplin garment used by DexGarmentLab's
# Fold_Tops/data_0 example. Genesis FEM rotates Mesh morphs around the mesh
# COM, so this translation differs from PBD's origin-pivoted placement.
_GENESIS_GARMENT_SCALE = 0.0087
_GENESIS_GARMENT_POSITION = (-0.150024, -1.160873, 1.005735)
_GENESIS_GARMENT_EULER = (0.0, 0.0, 90.0)

_RIGHT_HOME_TCP = np.asarray(
    [-0.3715, -0.2395, 0.9975, 0.70710678, 0.0, 0.70710678, 0.0],
    dtype=np.float64,
)
_LEFT_HOME_TCP = np.asarray(
    [-0.3715, 0.2395, 0.9975, 0.70710678, 0.0, 0.70710678, 0.0],
    dtype=np.float64,
)
_RIGHT_BUILD_TCP = _RIGHT_HOME_TCP.copy()
_RIGHT_BUILD_TCP[2] = 1.30
_LEFT_BUILD_TCP = _LEFT_HOME_TCP.copy()
_LEFT_BUILD_TCP[2] = 1.30
_ARM_HOME = {
    "right": np.asarray(
        [-math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, 0.0, 0.0],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, 0.0, 0.0],
        dtype=np.float64,
    ),
}
_ARM_KP = np.asarray([380, 380, 300, 300, 105, 105, 105], dtype=np.float64)
_ARM_KV = np.asarray([52, 52, 44, 44, 18, 18, 18], dtype=np.float64)
_WUJI_HAND_HOME = np.zeros(_HAND_DIM, dtype=np.float64)
_WUJI_HAND_HOME[0] = 0.10

_HEAD_CAMERA_EYE = (-0.675, 0.0, 1.5885)
_HEAD_CAMERA_LOOKAT = (0.161244, 0.0, 1.040143)
_WRIST_CAMERA_MOUNT_POSES = {
    "left": np.asarray(
        [0.0, 0.0160052526, 0.0452529768, 0.0524277738, 0.3054651071, 0.9370572242, -0.1608295853],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [0.0, 0.0160052526, 0.0452529768, 0.0524423546, 0.3055500602, -0.9370295266, 0.1608248315],
        dtype=np.float64,
    ),
}
_WRIST_CAMERA_PALM_POSES = {
    "left": np.asarray(
        [0.03816295, 0.03816295, 0.03942423, 0.0868936043, -0.7568897559, 0.6477050058, -0.0067545863],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [0.03816295, -0.03816295, 0.03942423, 0.0067467084, -0.6477736234, 0.7568310313, -0.0868942163],
        dtype=np.float64,
    ),
}

_GENESIS_INIT_LOCK = threading.Lock()
_GENESIS_INITIALIZED = False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def _env_vector(
    name: str, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    values = tuple(float(value.strip()) for value in raw.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain three comma-separated values")
    return values


def _interface_garment_nodes() -> np.ndarray:
    x_values = np.linspace(-0.31, 0.31, 21)
    y_values = np.linspace(-0.19, 0.19, 13)
    return np.asarray(
        [
            (-0.15 + x, y, _TABLE_TOP_Z + 0.004)
            for y in y_values
            for x in x_values
            if abs(y) <= 0.115 or abs(x) <= 0.17
        ],
        dtype=np.float64,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _is_wuji_ipc_contact_link(name: str, side: str) -> bool:
    """Keep every authored collision-bearing hand segment out to the tips."""
    return name == f"{side}_palm_link" or any(
        name == f"{side}_finger{finger}_link{link}"
        for finger in range(1, 6)
        # link1 has no collision geom in the authored Wuji primitive MJCF.
        for link in range(2, 5)
    )


def _limit_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_translation: float,
    max_rotation: float,
) -> np.ndarray:
    """Limit an IPC kinematic target step in translation and rotation."""
    current = np.asarray(current, dtype=np.float64).reshape(7)
    target = np.asarray(target, dtype=np.float64).reshape(7).copy()

    delta = target[:3] - current[:3]
    distance = float(np.linalg.norm(delta))
    if max_translation > 0.0 and distance > max_translation:
        target[:3] = current[:3] + delta * (max_translation / distance)

    current_quat = current[3:7] / max(float(np.linalg.norm(current[3:7])), 1e-12)
    target_norm = float(np.linalg.norm(target[3:7]))
    target_quat = (
        target[3:7] / target_norm if target_norm > 1e-12 else current_quat.copy()
    )
    dot = float(np.clip(np.dot(current_quat, target_quat), -1.0, 1.0))
    if dot < 0.0:
        target_quat = -target_quat
        dot = -dot
    angle = 2.0 * math.acos(min(1.0, dot))
    if max_rotation > 0.0 and angle > max_rotation:
        fraction = max_rotation / angle
        half_angle = math.acos(min(1.0, dot))
        if half_angle < 1e-8:
            blended = (1.0 - fraction) * current_quat + fraction * target_quat
        else:
            denominator = math.sin(half_angle)
            blended = (
                math.sin((1.0 - fraction) * half_angle) / denominator * current_quat
                + math.sin(fraction * half_angle) / denominator * target_quat
            )
        target_quat = blended / max(float(np.linalg.norm(blended)), 1e-12)
    target[3:7] = target_quat
    return target


@lru_cache(maxsize=4)
def _cached_rokae_reach_points(path_string: str) -> dict[str, np.ndarray]:
    """Load the shipped C01/Plan1 palm-position reach samples once."""
    path = Path(path_string)
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as cache:
            result: dict[str, np.ndarray] = {}
            for side in ("right", "left"):
                key = f"{side}_points"
                if key not in cache.files:
                    continue
                points = np.asarray(cache[key], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3:
                    continue
                lower, upper = _ROKAE_HAND_WORKSPACE_BOUNDS[side]
                root = _ROKAE_ARM_ROOT_POS[side]
                valid = np.all(np.isfinite(points), axis=1)
                valid &= np.all(points >= lower, axis=1)
                valid &= np.all(points <= upper, axis=1)
                valid &= np.linalg.norm(points - root, axis=1) <= (
                    _ROKAE_REACH_RADIUS + 1e-9
                )
                points = np.ascontiguousarray(points[valid])
                if points.size:
                    points.setflags(write=False)
                    result[side] = points
            return result
    except (OSError, ValueError, KeyError):
        return {}


def _load_rokae_reach_points(path: Path | None = None) -> dict[str, np.ndarray]:
    resolved = (path or _ROKAE_REACH_CACHE_PATH).expanduser().resolve()
    return dict(_cached_rokae_reach_points(str(resolved)))


def _project_rokae_hand_position(
    side: str,
    target: np.ndarray,
    fallback: np.ndarray,
    *,
    reach_points: np.ndarray | None = None,
    minimum_z: float | None = None,
    step_origin: np.ndarray | None = None,
    max_step: float | None = None,
) -> tuple[np.ndarray, bool, float]:
    """Project a detached palm into Plan1's sampled ROKAE reach envelope.

    The box and radius are a cheap fallback for installations without the
    shipped cache.  Like Plan1, a query inside the sampled 8 cm reach envelope
    remains continuous; only a query outside that envelope snaps to a sampled
    ROKAE palm position.  A dynamic minimum z and step ball let the caller
    combine reach, table clearance, and a per-frame motion limit in one
    inexpensive nearest-neighbour scan (roughly 8k points per side).
    """
    if side not in _ROKAE_HAND_WORKSPACE_BOUNDS:
        raise ValueError(f"Unsupported ROKAE side: {side!r}")
    requested = np.asarray(target, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(requested)):
        requested = np.asarray(fallback, dtype=np.float64).reshape(3).copy()
    projected = requested.copy()
    base_lower, upper = _ROKAE_HAND_WORKSPACE_BOUNDS[side]
    lower = base_lower.copy()
    if minimum_z is not None and np.isfinite(minimum_z):
        lower[2] = min(upper[2], max(lower[2], float(minimum_z)))
    root = _ROKAE_ARM_ROOT_POS[side]
    # Box and radial projections are convex.  Two alternating passes are
    # sufficient for this particular root/box geometry and retain a deterministic
    # fallback when the sampled cache is deliberately disabled.
    for _ in range(2):
        projected = np.clip(projected, lower, upper)
        offset = projected - root
        distance = float(np.linalg.norm(offset))
        if distance > _ROKAE_REACH_RADIUS and distance > 1e-12:
            projected = root + offset * (_ROKAE_REACH_RADIUS / distance)
    projected = np.clip(projected, lower, upper)

    if reach_points is None:
        reach_points = _load_rokae_reach_points().get(side)
    if reach_points is not None:
        points = np.asarray(reach_points, dtype=np.float64).reshape(-1, 3)
        valid = np.all(np.isfinite(points), axis=1)
        valid &= np.all(points >= lower, axis=1)
        valid &= np.all(points <= upper, axis=1)
        valid &= np.linalg.norm(points - root, axis=1) <= (
            _ROKAE_REACH_RADIUS + 1e-9
        )
        safe_points = points[valid]
        if safe_points.size:
            delta = safe_points - projected
            nearest_index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
            nearest_distance = float(np.linalg.norm(delta[nearest_index]))
            projected_within_step = True
            origin: np.ndarray | None = None
            if (
                step_origin is not None
                and max_step is not None
                and np.isfinite(max_step)
                and max_step > 0.0
            ):
                origin = np.asarray(step_origin, dtype=np.float64).reshape(3)
                projected_within_step = bool(
                    np.linalg.norm(projected - origin) <= float(max_step) + 1e-9
                )
                within_step = np.linalg.norm(safe_points - origin, axis=1) <= (
                    float(max_step) + 1e-9
                )
            else:
                within_step = np.ones(safe_points.shape[0], dtype=bool)

            if (
                nearest_distance <= _ROKAE_REACH_MAX_POINT_DIST
                and projected_within_step
            ):
                pass
            elif np.any(within_step):
                candidates = safe_points[within_step]
                candidate_delta = candidates - projected
                index = int(
                    np.argmin(np.einsum("ij,ij->i", candidate_delta, candidate_delta))
                )
                projected = candidates[index].copy()
            elif origin is not None:
                # The dynamic table floor and step ball have no joint cache
                # candidate.  Hold position; the caller will either confirm it
                # remains safe for the new orientation/hand or roll the whole
                # command back to the previous safe state.
                projected = origin.copy()

    correction = float(np.linalg.norm(projected - requested))
    return projected, bool(correction > 1e-6), correction


class _WujiPrimitiveCollisionGuard:
    """Fast exact FK bounds for the independent Wuji primitive colliders."""

    def __init__(self, side: str, primitive_mjcf: Path) -> None:
        if side not in {"right", "left"}:
            raise ValueError(f"Unsupported Wuji side: {side!r}")
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - HandsOn requires MuJoCo
            raise RuntimeError(
                "MuJoCo is required for the Genesis Wuji table-clearance guard"
            ) from exc

        self.side = side
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(primitive_mjcf.resolve()))
        self.data = mujoco.MjData(self.model)
        self._palm_body_id = int(self.model.body(f"{side}_palm_link").id)
        self._joint_ids = np.asarray(
            [
                int(self.model.joint(f"{side}_finger{finger}_joint{joint}").id)
                for finger in range(1, 6)
                for joint in range(1, 5)
            ],
            dtype=int,
        )
        self._qpos_ids = np.asarray(
            [int(self.model.jnt_qposadr[joint_id]) for joint_id in self._joint_ids],
            dtype=int,
        )
        ranges = np.asarray(self.model.jnt_range[self._joint_ids], dtype=np.float64)
        # Position actuators carry the command range used by Plan1.  Use the
        # intersection with the joint range so direct Genesis qpos writes cannot
        # bypass either authored limit.
        for actuator_id in range(int(self.model.nu)):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            matches = np.flatnonzero(self._joint_ids == joint_id)
            if not matches.size:
                continue
            control_range = np.asarray(
                self.model.actuator_ctrlrange[actuator_id], dtype=np.float64
            )
            if np.all(np.isfinite(control_range)) and control_range[0] < control_range[1]:
                index = int(matches[0])
                ranges[index, 0] = max(ranges[index, 0], control_range[0])
                ranges[index, 1] = min(ranges[index, 1], control_range[1])
        self.joint_ranges = ranges
        self._collision_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in range(int(self.model.ngeom))
                if int(self.model.geom_contype[geom_id]) != 0
                or int(self.model.geom_conaffinity[geom_id]) != 0
            ],
            dtype=int,
        )
        if not self._collision_geom_ids.size:
            raise RuntimeError(f"No primitive collision geoms found for {side} Wuji hand")

    def _geom_vertical_extent(self, geom_id: int) -> float:
        mujoco = self._mujoco
        geom_type = int(self.model.geom_type[geom_id])
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(
            3, 3
        )
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(size[0])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            return float(size[0] + abs(rotation[2, 2]) * size[1])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return float(np.sum(np.abs(rotation[2]) * size[:3]))
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            axial_z = abs(float(rotation[2, 2]))
            radial_z = math.sqrt(max(0.0, 1.0 - axial_z * axial_z))
            return float(axial_z * size[1] + radial_z * size[0])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return float(np.linalg.norm(rotation[2] * size[:3]))
        # The authored collision set is primitive-only.  Keep a conservative
        # bound if a future asset adds another supported MuJoCo geom type.
        return float(self.model.geom_rbound[geom_id])

    def minimum_collision_z(self, pose: np.ndarray, hand: np.ndarray) -> float:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        hand = np.asarray(hand, dtype=np.float64).reshape(_HAND_DIM)
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(hand)):
            raise ValueError("Wuji collision FK requires finite pose and joint values")
        quat = pose[3:7]
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-12:
            raise ValueError("Wuji collision FK requires a non-zero quaternion")
        self.model.body_pos[self._palm_body_id] = pose[:3]
        self.model.body_quat[self._palm_body_id] = quat / quat_norm
        self.data.qpos[self._qpos_ids] = np.clip(
            hand, self.joint_ranges[:, 0], self.joint_ranges[:, 1]
        )
        self._mujoco.mj_forward(self.model, self.data)
        return float(
            min(
                float(self.data.geom_xpos[geom_id, 2])
                - self._geom_vertical_extent(int(geom_id))
                for geom_id in self._collision_geom_ids
            )
        )

    def project_above_table(
        self,
        pose: np.ndarray,
        hand: np.ndarray,
        *,
        table_top_z: float = _TABLE_TOP_Z,
        clearance: float = _HAND_TABLE_CLEARANCE,
    ) -> tuple[np.ndarray, bool, float, float]:
        projected = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        floor_z = float(table_top_z) + max(0.0, float(clearance))
        minimum_z = self.minimum_collision_z(projected, hand)
        signed_correction = floor_z - minimum_z
        correction = max(0.0, signed_correction)
        required_palm_z = float(projected[2] + signed_correction)
        projected[2] += correction
        return (
            projected,
            bool(correction > 1e-9),
            correction,
            required_palm_z,
        )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def _resolve_genesis_garment_obj() -> Path:
    configured = os.environ.get("HANDSON_GENESIS_E01_GARMENT_OBJ", "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        candidate = next(
            (
                root / _GENESIS_GARMENT_RELATIVE_PATH
                for root in garmentlab_asset_root_candidates()
                if (root / _GENESIS_GARMENT_RELATIVE_PATH).is_file()
            ),
            garmentlab_asset_root_candidates()[0]
            / _GENESIS_GARMENT_RELATIVE_PATH,
        )
    if not candidate.is_file():
        raise FileNotFoundError(
            "Genesis E01 needs the official DexGarmentLab TCLC_Top578 "
            "Cotton_40s_Poplin OBJ, "
            f"but it is missing: {candidate}. Run "
            "scripts/export_garmentlab_usd_mesh.py with an Isaac/pxr Python runtime."
        )
    return candidate


class HandsOnE01GenesisEnv(gym.Env):
    """Bimanual E01 task using Genesis World, FEM.Cloth, and libuipc IPC."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_name: str = E01_TASK_NAME,
        *,
        render_mode: str = "human",
        randomize: bool = False,
        time_limit: float = 180.0,
        hz: int = 30,
        image_obs: bool = True,
        launch_genesis: bool = True,
        show_native_viewer: bool = False,
        physics_dt: float | None = None,
        **_: Any,
    ) -> None:
        if task_name != E01_TASK_NAME:
            raise ValueError(
                f"Genesis E01 environment supports {E01_TASK_NAME!r}, got {task_name!r}"
            )
        if render_mode not in {"human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode={render_mode!r}")

        self.task_name = task_name
        self.render_mode = render_mode
        self.randomize = bool(randomize)
        self.time_limit = float(time_limit)
        self.hz = int(hz)
        self.image_obs = bool(image_obs)
        # Genesis 1.2's native viewer shares an OpenGL framebuffer with the
        # off-screen cameras used by the recorder and can remain black after
        # those cameras render.  The collection path displays the canonical
        # front frame through OpenCV instead; direct callers can opt back in.
        self.show_native_viewer = bool(show_native_viewer)
        self.physics_dt = float(
            _env_float("HANDSON_GENESIS_E01_PHYSICS_DT", 1.0 / 30.0)
            if physics_dt is None
            else physics_dt
        )
        if self.physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be positive, got {self.physics_dt}")
        self.control_dt = 1.0 / max(1, self.hz)
        self._n_substeps = max(1, int(round(self.control_dt / self.physics_dt)))
        self._max_tcp_translation_step = _env_float(
            "HANDSON_GENESIS_E01_MAX_TCP_TRANSLATION_STEP", 0.060
        )
        self._max_tcp_rotation_step = math.radians(
            _env_float("HANDSON_GENESIS_E01_MAX_TCP_ROTATION_STEP_DEG", 10.0)
        )
        self._max_hand_joint_step = _env_float(
            "HANDSON_GENESIS_E01_MAX_HAND_JOINT_STEP", 0.12
        )

        # Teleoperation wrapper feature detection relies on these attributes.
        self.arm_type = "rokae"
        self.hand_type = "wuji1"
        self._hand_target_control = True

        self._random = np.random.RandomState()
        self._scene: Any | None = None
        self._cloth: Any | None = None
        self._robots: dict[str, Any] = {}
        self._ee_links: dict[str, Any] = {}
        self._arm_dofs: dict[str, list[int]] = {}
        self._hand_base_qs: dict[str, list[int]] = {}
        self._hand_dofs: dict[str, list[int]] = {}
        self._hand_links: dict[str, list[Any]] = {"right": [], "left": []}
        self._cameras: dict[str, Any] = {}
        self._genesis_warnings: list[str] = []
        self._asset_path: Path | None = None
        reach_cache_path = Path(
            os.environ.get(
                "HANDSON_GENESIS_E01_REACH_CACHE_PATH",
                str(_ROKAE_REACH_CACHE_PATH),
            )
        )
        self._rokae_reach_points = (
            _load_rokae_reach_points(reach_cache_path)
            if _env_bool("HANDSON_GENESIS_E01_STATIC_REACH_CACHE", True)
            else {}
        )
        missing_reach_sides = {
            "right",
            "left",
        }.difference(self._rokae_reach_points)
        if missing_reach_sides:
            self._genesis_warnings.append(
                "ROKAE static reach cache unavailable for "
                + ", ".join(sorted(missing_reach_sides))
                + "; using box/radius fallback"
            )
        self._hand_collision_guards = {
            side: _WujiPrimitiveCollisionGuard(
                side, _WUJI_HAND_MJCF_DIR / f"{side}_primitive.xml"
            )
            for side in ("right", "left")
        }
        self._hand_joint_ranges = {
            side: guard.joint_ranges.copy()
            for side, guard in self._hand_collision_guards.items()
        }

        fallback = _interface_garment_nodes()
        self._nodes = fallback.copy()
        self._velocities = np.zeros_like(fallback)
        self._base_nodes = fallback.copy()
        self._reset_nodes = fallback.copy()
        self._fold_axis = np.asarray([1.0, 0.0], dtype=np.float64)
        midpoint = fallback.shape[0] // 2
        self._fold_masks = (
            np.arange(midpoint, dtype=np.int64),
            np.arange(midpoint, fallback.shape[0], dtype=np.int64),
        )
        self._keypoint_indices = np.arange(8, dtype=np.int64)
        self._baseline_half_distance = 1.0
        self._baseline_area = 1.0
        self._initial_state = np.zeros(6, dtype=np.float64)
        self.delta_h = np.zeros(1, dtype=np.float64)
        self._last_metrics: dict[str, Any] = {}
        self._right_tcp_pose = _RIGHT_HOME_TCP.copy()
        self._left_tcp_pose = _LEFT_HOME_TCP.copy()
        self._right_commanded_tcp_pose = _RIGHT_HOME_TCP.copy()
        self._left_commanded_tcp_pose = _LEFT_HOME_TCP.copy()
        self._right_hand = _WUJI_HAND_HOME.copy()
        self._left_hand = _WUJI_HAND_HOME.copy()
        self._arm_reach_guard_active = {"right": False, "left": False}
        self._arm_reach_guard_distance = {"right": 0.0, "left": 0.0}
        self._hand_table_guard_active = {"right": False, "left": False}
        self._hand_table_guard_distance = {"right": 0.0, "left": 0.0}
        self._constraint_hold_active = {"right": False, "left": False}
        self._hand_minimum_collision_z = {
            "right": float("inf"),
            "left": float("inf"),
        }
        self.env_step = 0
        self._success_counter = 0
        self._configure_tracking(self._nodes)

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14 + 2 * _HAND_DIM,),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()

        if launch_genesis:
            self._setup_genesis()

    # ------------------------------------------------------------------
    # Genesis scene construction
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_genesis(gs: Any) -> None:
        global _GENESIS_INITIALIZED
        with _GENESIS_INIT_LOCK:
            if _GENESIS_INITIALIZED:
                return
            gs.init(
                backend=gs.cpu,
                precision="32",
                logging_level=os.environ.get(
                    "HANDSON_GENESIS_LOG_LEVEL", "warning"
                ),
            )
            _GENESIS_INITIALIZED = True

    def _setup_genesis(self) -> None:
        try:
            import genesis as gs
        except ImportError as exc:
            raise RuntimeError(
                "Genesis E01 runtime is unavailable. Use the isolated Python at "
                "/home/user/下载/handson/.handson_genesis/env/bin/python or set "
                "HANDSON_GENESIS_PYTHON."
            ) from exc

        self._initialize_genesis(gs)
        self._asset_path = _resolve_genesis_garment_obj()

        coupler_options = gs.options.IPCCouplerOptions(
            # libuipc otherwise permits 1024 Newton iterations.  A difficult
            # animated-hand CCD frame can then block teleoperation for tens of
            # seconds.  Keep a finite iteration budget and tighter convergence
            # defaults for this dual 20-DoF-hand scene.
            newton_max_iterations=max(
                1,
                int(_env_float("HANDSON_GENESIS_IPC_NEWTON_MAX_ITERATIONS", 2)),
            ),
            newton_tolerance=_env_float(
                "HANDSON_GENESIS_IPC_NEWTON_TOLERANCE", 0.05
            ),
            newton_translation_tolerance=_env_float(
                "HANDSON_GENESIS_IPC_TRANSLATION_TOLERANCE", 0.1
            ),
            newton_semi_implicit_enable=False,
            n_linesearch_iterations=8,
            linear_system_tolerance=1.0e-3,
            contact_enable=True,
            contact_d_hat=_env_float("HANDSON_GENESIS_IPC_D_HAT", 0.001),
            contact_resistance=_env_float(
                "HANDSON_GENESIS_IPC_CONTACT_RESISTANCE", 1.0e8
            ),
            # The collected hands are kinematic targets.  Cloth reaction must
            # not push the lightweight display articulation away from the
            # tracker pose; IPC contact itself remains fully enabled.
            two_way_coupling=_env_bool(
                "HANDSON_GENESIS_IPC_TWO_WAY_COUPLING", False
            ),
            enable_rigid_rigid_contact=False,
        )
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.physics_dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            coupler_options=coupler_options,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=_HEAD_CAMERA_EYE,
                camera_lookat=_HEAD_CAMERA_LOOKAT,
                camera_fov=80,
                refresh_rate=max(30, self.hz),
            ),
            show_viewer=(
                self.render_mode == "human" and self.show_native_viewer
            ),
        )

        static_material = gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=_env_float("HANDSON_GENESIS_E01_RIGID_FRICTION", 0.8),
        )
        self._scene.add_entity(
            morph=gs.morphs.Plane(),
            material=static_material,
            surface=gs.surfaces.Plastic(color=(0.78, 0.80, 0.82, 1.0)),
        )
        self._scene.add_entity(
            morph=gs.morphs.Box(
                pos=(-0.15, 0.0, 0.86),
                size=(0.8, 1.6, 0.06),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                coup_type="ipc_only",
                coup_friction=_env_float(
                    "HANDSON_GENESIS_E01_RIGID_FRICTION", 0.8
                ),
            ),
            surface=gs.surfaces.Plastic(color=(0.72, 0.72, 0.68, 1.0)),
        )

        xml_dir = Path(__file__).resolve().parents[1] / "envs" / "xmls"
        hand_xml_dir = xml_dir / "wuji_hand1" / "mjcf"
        for side in ("right", "left"):
            # Keep the canonical black Wuji hand from the complete collection
            # asset, but extract only its palm/finger subtree.  The ROKAE arm
            # is absent from the live scene and completed offline from the
            # recorded palm trajectory after a successful episode.
            hand_xml = prepare_detached_rokae_wuji_hand_mjcf(
                xml_dir / f"rokae_wuji1_{side}.xml",
                hand_xml_dir / f"{side}_primitive.xml",
                side=side,
            )
            material = gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_friction=_env_float(
                    "HANDSON_GENESIS_E01_HAND_FRICTION", 0.9
                ),
                gravity_compensation=1.0,
            )
            hand_entity = self._scene.add_entity(
                morph=gs.morphs.MJCF(
                    file=str(hand_xml),
                    pos=(
                        _RIGHT_BUILD_TCP[:3]
                        if side == "right"
                        else _LEFT_BUILD_TCP[:3]
                    ),
                    quat=(
                        _RIGHT_BUILD_TCP[3:7]
                        if side == "right"
                        else _LEFT_BUILD_TCP[3:7]
                    ),
                    requires_jac_and_IK=False,
                ),
                material=material,
            )
            coupled_names = tuple(
                link.name
                for link in hand_entity.links
                if _is_wuji_ipc_contact_link(link.name, side)
            )
            if not coupled_names:
                raise RuntimeError(
                    f"Genesis parsed no IPC hand links from {hand_xml}"
                )
            hand_entity.material.coup_links = coupled_names
            self._robots[side] = hand_entity

        self._cloth = self._scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(self._asset_path),
                scale=_env_float(
                    "HANDSON_GENESIS_E01_GARMENT_SCALE",
                    _GENESIS_GARMENT_SCALE,
                ),
                pos=_env_vector(
                    "HANDSON_GENESIS_E01_GARMENT_POSITION",
                    _GENESIS_GARMENT_POSITION,
                ),
                euler=_env_vector(
                    "HANDSON_GENESIS_E01_GARMENT_EULER",
                    _GENESIS_GARMENT_EULER,
                ),
            ),
            material=gs.materials.FEM.Cloth(
                E=_env_float("HANDSON_GENESIS_E01_YOUNGS_MODULUS", 1.0e4),
                nu=_env_float("HANDSON_GENESIS_E01_POISSON_RATIO", 0.30),
                rho=_env_float("HANDSON_GENESIS_E01_DENSITY", 300.0),
                # The source material is Cotton_40s_Poplin. At 0.5 mm, two
                # authored layers violate libuipc's rest-distance check; the
                # measured-style 0.2 mm thickness both fits poplin and passes
                # IPC validation without changing the downloaded geometry.
                thickness=_env_float("HANDSON_GENESIS_E01_THICKNESS", 0.0002),
                bending_stiffness=_env_float(
                    "HANDSON_GENESIS_E01_BENDING_STIFFNESS", 10.0
                ),
                friction_mu=_env_float("HANDSON_GENESIS_E01_CLOTH_FRICTION", 0.7),
            ),
            surface=gs.surfaces.Plastic(color=(0.34, 0.56, 0.88, 1.0)),
        )

        self._cameras["front"] = self._scene.add_camera(
            res=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[0]),
            pos=_HEAD_CAMERA_EYE,
            lookat=_HEAD_CAMERA_LOOKAT,
            fov=80,
            GUI=False,
        )
        for side in ("left", "right"):
            self._cameras[f"wrist_{side}"] = self._scene.add_camera(
                res=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[0]),
                pos=(0.0, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                GUI=False,
            )

        try:
            self._scene.build()
        except Exception as exc:
            raise RuntimeError(
                "Genesis/libuipc could not build E01. IPC rejects garment rest "
                "meshes that already self-intersect or violate the configured "
                f"thickness. Asset: {self._asset_path}"
            ) from exc

        self._bind_robot_interfaces()
        self._initialize_robot_home()
        self._attach_wrist_cameras()
        self._refresh_state(
            compute_metrics=False,
            reset_nodal_velocity=True,
        )
        self._right_commanded_tcp_pose = self._right_tcp_pose.copy()
        self._left_commanded_tcp_pose = self._left_tcp_pose.copy()
        self._base_nodes = self._nodes.copy()
        self._reset_nodes = self._nodes.copy()
        self._configure_tracking(self._base_nodes)
        self._last_metrics = self._garment_metrics()

    def _bind_robot_interfaces(self) -> None:
        for side, robot in self._robots.items():
            # Live Plan-style collection has no ROKAE articulation and never
            # performs foreground arm IK.  Keep the mapping present for
            # callers that inspect it, but it is intentionally empty.
            self._arm_dofs[side] = []
            self._hand_base_qs[side] = [
                int(index)
                for index in robot.get_joint(
                    f"{side}_palm_free"
                ).qs_idx_local
            ]
            self._hand_dofs[side] = [
                int(
                    robot.get_joint(
                        f"{side}_finger{finger}_joint{joint}"
                    ).dofs_idx_local[0]
                )
                for finger in range(1, 6)
                for joint in range(1, 5)
            ]
            self._ee_links[side] = robot.get_link(f"{side}_palm_link")
            self._hand_links[side] = [
                link
                for link in robot.links
                if link.name == f"{side}_palm_link"
                or (
                    link.name.startswith(f"{side}_finger")
                    and "_link" in link.name
                )
            ]

    def _initialize_robot_home(self) -> None:
        for side, robot in self._robots.items():
            hand = _WUJI_HAND_HOME.copy()
            home_pose = _RIGHT_HOME_TCP if side == "right" else _LEFT_HOME_TCP
            dofs = self._hand_dofs[side]
            robot.set_dofs_kp(
                np.full(_HAND_DIM, 8.0),
                dofs_idx_local=dofs,
            )
            robot.set_dofs_kv(
                np.full(_HAND_DIM, 0.12),
                dofs_idx_local=dofs,
            )
            robot.set_dofs_position(hand, dofs_idx_local=dofs)
            robot.control_dofs_position(hand, dofs_idx_local=dofs)
            # Set the floating base last: Genesis' articulated-DOF forward
            # pass may otherwise restore the MJCF build pose.
            robot.set_qpos(
                home_pose,
                qs_idx_local=self._hand_base_qs[side],
                zero_velocity=True,
            )
            if side == "right":
                self._right_hand = hand.copy()
            else:
                self._left_hand = hand.copy()

    def _set_kinematic_hand_target(
        self,
        side: str,
        pose: np.ndarray,
        hand: np.ndarray,
    ) -> None:
        """Place a detached hand directly, without any online arm IK."""
        robot = self._robots.get(side)
        if robot is None:
            return
        hand_dofs = self._hand_dofs[side]
        # Plan collection treats the glove angles as kinematic hand state.
        # Position control remains enabled to hold that state during the rigid
        # substep, but there is no actuator lag in the recorded contact pose.
        robot.set_dofs_position(hand, dofs_idx_local=hand_dofs)
        robot.control_dofs_position(hand, dofs_idx_local=hand_dofs)
        robot.set_qpos(
            pose,
            qs_idx_local=self._hand_base_qs[side],
            zero_velocity=True,
        )

    def _attach_wrist_cameras(self) -> None:
        for side, robot in self._robots.items():
            try:
                link = robot.get_link(f"{side}_wrist_fisheye_mount")
                local_pose = _WRIST_CAMERA_MOUNT_POSES[side]
            except Exception:
                link = self._ee_links[side]
                local_pose = _WRIST_CAMERA_PALM_POSES[side]
                self._genesis_warnings.append(
                    f"{side} wrist mount link unavailable; camera attached to palm"
                )
            self._cameras[f"wrist_{side}"].attach(
                link, _pose_to_matrix(local_pose)
            )

    # ------------------------------------------------------------------
    # Gym API and robot control
    # ------------------------------------------------------------------

    def _parse_bimanual_hand_angles(
        self, action: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        right = np.asarray(action.get("right", np.zeros(27)), dtype=np.float64).ravel()
        left = np.asarray(action.get("left", np.zeros(27)), dtype=np.float64).ravel()
        return right[7 : 7 + _HAND_DIM], left[7 : 7 + _HAND_DIM]

    @staticmethod
    def _split_action(action: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(action, Mapping):
            right = np.asarray(action.get("right", np.zeros(27)), dtype=np.float64).ravel()
            left = np.asarray(action.get("left", np.zeros(27)), dtype=np.float64).ravel()
        else:
            flat = np.asarray(action, dtype=np.float64).ravel()
            if flat.size < 54:
                raise ValueError(f"Expected a 54-value E01 action, got {flat.size}")
            right = np.concatenate([flat[:7], flat[14:34]])
            left = np.concatenate([flat[7:14], flat[34:54]])
        if right.size < 27 or left.size < 27:
            raise ValueError(
                f"Expected right/left 27-value actions, got {right.size}/{left.size}"
            )
        return right[:27], left[:27]

    def _sanitize_hand_target(
        self,
        side: str,
        requested: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        """Apply Plan1's finite/range boundary before the IPC joint rate limit."""
        ranges = self._hand_joint_ranges[side]
        current = np.asarray(current, dtype=np.float64).reshape(_HAND_DIM).copy()
        current = np.where(np.isfinite(current), current, _WUJI_HAND_HOME)
        current = np.clip(current, ranges[:, 0], ranges[:, 1])
        target = np.asarray(requested, dtype=np.float64).reshape(_HAND_DIM).copy()
        target = np.where(np.isfinite(target), target, current)
        target = np.clip(target, ranges[:, 0], ranges[:, 1])
        if self._max_hand_joint_step > 0.0:
            target = np.clip(
                target,
                current - self._max_hand_joint_step,
                current + self._max_hand_joint_step,
            )
        return np.clip(target, ranges[:, 0], ranges[:, 1])

    def _apply_action(self, action: Any) -> None:
        right, left = self._split_action(action)
        for side, side_action in (("right", right), ("left", left)):
            current_pose = (
                self._right_commanded_tcp_pose
                if side == "right"
                else self._left_commanded_tcp_pose
            )
            current_hand = self._right_hand if side == "right" else self._left_hand
            requested_pose = side_action[:7].astype(np.float64, copy=True)
            raw_requested_position = requested_pose[:3].copy()
            requested_pose[:3], reach_active, reach_distance = (
                _project_rokae_hand_position(
                    side,
                    requested_pose[:3],
                    current_pose[:3],
                    reach_points=self._rokae_reach_points.get(
                        side, np.empty((0, 3), dtype=np.float64)
                    ),
                )
            )
            pose = _limit_pose_step(
                current_pose,
                requested_pose,
                max_translation=self._max_tcp_translation_step,
                max_rotation=self._max_tcp_rotation_step,
            )
            hand = self._sanitize_hand_target(
                side,
                side_action[7:27],
                current_hand,
            )
            guard = self._hand_collision_guards[side]
            (
                pose,
                table_active,
                table_correction,
                required_palm_z,
            ) = guard.project_above_table(
                pose,
                hand,
                table_top_z=_TABLE_TOP_Z,
                clearance=_HAND_TABLE_CLEARANCE,
            )

            # Interpolation between two IK samples may leave a non-convex reach
            # set.  Select the final sample only after the rate-limited rotation
            # and finger state are known, jointly enforcing the dynamic table
            # floor and the Cartesian step ball.  A second pass absorbs numeric
            # tolerance without giving up any of the three constraints.
            post_reach_active = False
            post_reach_distance = 0.0
            floor_z = _TABLE_TOP_Z + _HAND_TABLE_CLEARANCE
            for _ in range(2):
                (
                    pose[:3],
                    corrected,
                    correction,
                ) = _project_rokae_hand_position(
                    side,
                    pose[:3],
                    current_pose[:3],
                    reach_points=self._rokae_reach_points.get(
                        side, np.empty((0, 3), dtype=np.float64)
                    ),
                    minimum_z=required_palm_z,
                    step_origin=current_pose[:3],
                    max_step=self._max_tcp_translation_step,
                )
                post_reach_active = post_reach_active or corrected
                post_reach_distance = max(post_reach_distance, correction)
                final_minimum_z = guard.minimum_collision_z(pose, hand)
                deficit = floor_z - final_minimum_z
                if deficit <= 1e-9:
                    break
                required_palm_z = pose[2] + deficit
                table_active = True
                table_correction += deficit
            else:
                final_minimum_z = guard.minimum_collision_z(pose, hand)
            step_distance = float(np.linalg.norm(pose[:3] - current_pose[:3]))
            step_exceeded = bool(
                self._max_tcp_translation_step > 0.0
                and step_distance > self._max_tcp_translation_step + 1e-9
            )
            unsafe_table = bool(final_minimum_z < floor_z - 1e-9)
            constraint_hold = bool(step_exceeded or unsafe_table)
            if constraint_hold:
                # There is no joint cache/table/step candidate for this frame.
                # Roll back the complete command, including rotation and fingers;
                # lifting only the palm would silently leave the reach cache and
                # could violate the same step limit it was meant to protect.
                pose = np.asarray(current_pose, dtype=np.float64).copy()
                hand = np.asarray(current_hand, dtype=np.float64).copy()
                final_minimum_z = guard.minimum_collision_z(pose, hand)
                if final_minimum_z < floor_z - 1e-9:
                    raise RuntimeError(
                        f"Previous {side} Wuji command is below the table safety floor"
                    )
                table_active = table_active or unsafe_table

            self._arm_reach_guard_active[side] = bool(
                reach_active or post_reach_active
            )
            self._arm_reach_guard_distance[side] = max(
                float(reach_distance),
                float(post_reach_distance),
                float(np.linalg.norm(pose[:3] - raw_requested_position))
                if reach_active
                else 0.0,
            )
            self._hand_table_guard_active[side] = bool(table_active)
            self._hand_table_guard_distance[side] = float(table_correction)
            self._constraint_hold_active[side] = constraint_hold
            self._hand_minimum_collision_z[side] = float(final_minimum_z)
            if side == "right":
                self._right_commanded_tcp_pose = pose.copy()
                self._right_tcp_pose = pose
                self._right_hand = hand
            else:
                self._left_commanded_tcp_pose = pose.copy()
                self._left_tcp_pose = pose
                self._left_hand = hand

            self._set_kinematic_hand_target(side, pose, hand)

    def reset(self, seed: int | None = None, **_: Any):
        super().reset(seed=seed)
        if seed is not None:
            self._random.seed(seed)
        self.env_step = 0
        self._success_counter = 0
        self._last_metrics = {}
        self.delta_h[:] = 0.0
        self._initial_state[:] = 0.0
        self._right_tcp_pose = _RIGHT_HOME_TCP.copy()
        self._left_tcp_pose = _LEFT_HOME_TCP.copy()
        self._right_commanded_tcp_pose = _RIGHT_HOME_TCP.copy()
        self._left_commanded_tcp_pose = _LEFT_HOME_TCP.copy()
        self._right_hand = _WUJI_HAND_HOME.copy()
        self._left_hand = _WUJI_HAND_HOME.copy()
        self._arm_reach_guard_active = {"right": False, "left": False}
        self._arm_reach_guard_distance = {"right": 0.0, "left": 0.0}
        self._hand_table_guard_active = {"right": False, "left": False}
        self._hand_table_guard_distance = {"right": 0.0, "left": 0.0}
        self._constraint_hold_active = {"right": False, "left": False}
        self._hand_minimum_collision_z = {
            "right": self._hand_collision_guards["right"].minimum_collision_z(
                self._right_commanded_tcp_pose, self._right_hand
            ),
            "left": self._hand_collision_guards["left"].minimum_collision_z(
                self._left_commanded_tcp_pose, self._left_hand
            ),
        }
        if self.randomize:
            warning = (
                "libuipc cloth uses whole-scene reset; per-episode garment "
                "teleport randomization is disabled for Genesis E01"
            )
            if warning not in self._genesis_warnings:
                self._genesis_warnings.append(warning)
        if self._scene is not None:
            self._scene.reset()
            self._initialize_robot_home()
            self._scene.step()
            self._set_kinematic_hand_target(
                "right", self._right_commanded_tcp_pose, self._right_hand
            )
            self._set_kinematic_hand_target(
                "left", self._left_commanded_tcp_pose, self._left_hand
            )
        self._refresh_state(reset_nodal_velocity=True)
        self._right_commanded_tcp_pose = self._right_tcp_pose.copy()
        self._left_commanded_tcp_pose = self._left_tcp_pose.copy()
        self._reset_nodes = self._nodes.copy()
        self._last_metrics = self._garment_metrics()
        info = {"succeed": False, "manual_reset": False}
        info.update(self._task_info())
        return self._compute_observation(), info

    def step(self, action: Any):
        start_time = time.time()
        self._apply_action(action)
        if self._scene is not None:
            for _ in range(self._n_substeps):
                self._scene.step()
            # The rigid substep may impart tiny floating-base motion from the
            # finger actuators.  Present and record the exact tracker target;
            # IPC already consumed the kinematic sweep for this frame.
            self._set_kinematic_hand_target(
                "right", self._right_commanded_tcp_pose, self._right_hand
            )
            self._set_kinematic_hand_target(
                "left", self._left_commanded_tcp_pose, self._left_hand
            )
        self._refresh_state(
            elapsed_physics_dt=self._n_substeps * self.physics_dt,
        )
        success_now = bool(self._last_metrics["success_now"])
        self._success_counter = self._success_counter + 1 if success_now else 0
        success = self._success_counter >= _SUCCESS_STEPS
        failed = self._compute_failure()
        reward = 1.0 if success else 0.0
        self.env_step += 1
        terminated = bool(
            success
            or failed
            or self.env_step * self.control_dt >= self.time_limit
        )
        info = {
            "succeed": success,
            "manual_reset": False,
            "isaac_warnings": (),
            "genesis_warnings": tuple(self._genesis_warnings),
        }
        if failed:
            info["failed"] = True
        info.update(self._task_info())
        observation = self._compute_observation()
        if self.hz > 0:
            time.sleep(max(0.0, self.control_dt - (time.time() - start_time)))
        return observation, reward, terminated, False, info

    def close(self) -> None:
        if self._scene is not None:
            viewer = getattr(self._scene, "viewer", None)
            if viewer is not None and hasattr(viewer, "stop"):
                try:
                    viewer.stop()
                except Exception:
                    pass
        self._scene = None
        self._cloth = None
        self._robots.clear()
        self._cameras.clear()

    # ------------------------------------------------------------------
    # State, rendering, and task contract
    # ------------------------------------------------------------------

    def _refresh_state(
        self,
        *,
        compute_metrics: bool = True,
        elapsed_physics_dt: float | None = None,
        reset_nodal_velocity: bool = False,
    ) -> None:
        if self._cloth is not None:
            previous_nodes = self._nodes
            state = self._cloth.get_state()
            nodes = _to_numpy(state.pos).reshape(-1, 3).astype(np.float64)
            # Genesis 1.2.3's IPC coupler retrieves FEM positions but does not
            # copy libuipc velocities back into FEMEntity state.  Derive the
            # task's stability signal from consecutive retrieved positions so
            # a moving garment cannot be reported as perfectly stationary.
            if (
                reset_nodal_velocity
                or elapsed_physics_dt is None
                or elapsed_physics_dt <= 0.0
                or previous_nodes.shape != nodes.shape
            ):
                velocities = np.zeros_like(nodes)
            else:
                velocities = (nodes - previous_nodes) / elapsed_physics_dt
            self._nodes = nodes
            self._velocities = velocities
        for side in ("right", "left"):
            link = self._ee_links.get(side)
            if link is None:
                continue
            pos = _to_numpy(link.get_pos()).reshape(-1, 3)[0]
            quat = _to_numpy(link.get_quat()).reshape(-1, 4)[0]
            pose = np.concatenate([pos, quat]).astype(np.float64)
            if side == "right":
                self._right_tcp_pose = pose
            else:
                self._left_tcp_pose = pose
        if compute_metrics:
            self._last_metrics = self._garment_metrics()

    @staticmethod
    def _rgb_array(value: Any, name: str) -> np.ndarray:
        array = _to_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.shape[-1:] == (4,):
            array = array[..., :3]
        if array.dtype.kind == "f":
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8, copy=False)
        if array.shape != _IMAGE_SHAPE:
            raise RuntimeError(
                f"Genesis {name} camera returned {array.shape}; expected {_IMAGE_SHAPE}"
            )
        return np.ascontiguousarray(array)

    def render(self):
        if not self._cameras:
            blank = np.zeros(_IMAGE_SHAPE, dtype=np.uint8)
            return blank.copy(), blank.copy(), blank.copy()
        images: dict[str, np.ndarray] = {}
        for camera_index, name in enumerate(
            ("front", "wrist_left", "wrist_right")
        ):
            camera = self._cameras[name]
            rgb, _depth, _segmentation, _normal = camera.render(
                rgb=True,
                depth=False,
                segmentation=False,
                normal=False,
                antialiasing=False,
                # A forced render refreshes the visual state of every camera
                # attached to this scene.  Repeating that whole-scene update
                # for the two wrist captures wastes most of a display frame.
                force_render=camera_index == 0,
            )
            images[name] = self._rgb_array(rgb, name)
        # Genesis' viewer and off-screen cameras share the rasterizer.  The
        # final off-screen render can leave the on-screen framebuffer cleared
        # until the next viewer refresh, which appears as a permanently black
        # main window in the recorder loop.  Redraw after all camera captures
        # so the human viewer presents the same current scene state.
        if self.render_mode == "human" and self._scene is not None:
            viewer = getattr(self._scene, "viewer", None)
            if viewer is not None:
                viewer.update(auto_refresh=True, force=False)
        return images["front"], images["wrist_left"], images["wrist_right"]

    def get_end_effector_pose_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _pose_to_matrix(self._right_tcp_pose),
            _pose_to_matrix(self._left_tcp_pose),
        )

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        initial = np.asarray(
            state_parts.get("garment_initial_state", np.zeros(6)),
            dtype=np.float64,
        ).ravel()
        delta = np.asarray(
            state_parts.get("table_delta_height", np.zeros(1)),
            dtype=np.float64,
        ).ravel()
        if np.any(np.abs(initial[:6]) > 1e-8) or np.any(np.abs(delta[:1]) > 1e-8):
            raise ValueError(
                "Genesis/libuipc only supports restoring the build-time E01 cloth "
                "state; arbitrary nodal teleport is intentionally disabled."
            )
        observation, _info = self.reset()
        return observation

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
            "garment_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "garment_keypoints": spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float64),
            "garment_aabb": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "fold_metrics": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "garment_initial_state": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        result: dict[str, spaces.Space] = {"state": spaces.Dict(state_spaces)}
        if self.image_obs:
            image_space = lambda: spaces.Box(0, 255, shape=_IMAGE_SHAPE, dtype=np.uint8)
            result["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": image_space(),
                    "wrist_left": image_space(),
                    "wrist_right": image_space(),
                }
            )
        return spaces.Dict(result)

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._last_metrics or self._garment_metrics()
        gripper = np.zeros(32, dtype=np.float64)
        gripper[:16] = self._right_hand[:16]
        gripper[16:] = self._left_hand[:16]
        state = {
            "tcp_pose": np.concatenate([self._right_tcp_pose, self._left_tcp_pose]),
            "gripper_pose": gripper,
            "garment_pose": self._garment_pose(),
            "garment_keypoints": self._nodes[self._keypoint_indices].reshape(-1).copy(),
            "garment_aabb": np.concatenate([metrics["bounds_min"], metrics["bounds_max"]]),
            "fold_metrics": np.asarray(
                [
                    metrics["fold_ratio"],
                    metrics["area_ratio"],
                    metrics["half_centroid_distance"],
                    metrics["z_span"],
                    metrics["hand_distance"],
                    metrics["rms_nodal_speed"],
                    metrics["max_nodal_speed"],
                    metrics["compact_fold"],
                    metrics["on_table"],
                    metrics["in_workspace"],
                    metrics["hands_released"],
                    self._success_counter / _SUCCESS_STEPS,
                ],
                dtype=np.float64,
            ),
            "garment_initial_state": self._initial_state.copy(),
            "table_delta_height": self.delta_h.copy(),
        }
        observation: dict[str, Any] = {"state": state}
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    # ------------------------------------------------------------------
    # Garment metrics (same task contract as the Isaac E01 backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _oriented_xy_area(nodes: np.ndarray) -> float:
        xy = np.asarray(nodes, dtype=np.float64)[:, :2]
        centered = xy - np.mean(xy, axis=0)
        covariance = centered.T @ centered / max(1, xy.shape[0] - 1)
        _values, vectors = np.linalg.eigh(covariance)
        extent = np.ptp(centered @ vectors, axis=0)
        return float(max(1e-9, extent[0] * extent[1]))

    def _configure_tracking(self, nodes: np.ndarray) -> None:
        nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
        centered_xy = nodes[:, :2] - np.mean(nodes[:, :2], axis=0)
        covariance = centered_xy.T @ centered_xy / max(1, nodes.shape[0] - 1)
        _values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        self._fold_axis = axis.astype(np.float64)
        projection = centered_xy @ self._fold_axis
        median = float(np.median(projection))
        low = np.flatnonzero(projection <= median).astype(np.int64)
        high = np.flatnonzero(projection > median).astype(np.int64)
        self._fold_masks = (low, high)

        perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
        directions = (
            axis,
            -axis,
            perpendicular,
            -perpendicular,
            axis + perpendicular,
            axis - perpendicular,
            -axis + perpendicular,
            -axis - perpendicular,
        )
        selected: list[int] = []
        for direction in directions:
            index = int(np.argmax(centered_xy @ direction))
            if index not in selected:
                selected.append(index)
        for index in np.argsort(np.linalg.norm(centered_xy, axis=1))[::-1]:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == 8:
                break
        while len(selected) < 8:
            selected.append(selected[-1] if selected else 0)
        self._keypoint_indices = np.asarray(selected[:8], dtype=np.int64)

        low_center = np.mean(nodes[low, :2], axis=0)
        high_center = np.mean(nodes[high, :2], axis=0)
        self._baseline_half_distance = max(
            1e-6, float(np.linalg.norm(low_center - high_center))
        )
        self._baseline_area = max(1e-6, self._oriented_xy_area(nodes))

    def _garment_pose(self) -> np.ndarray:
        centroid = np.mean(self._nodes, axis=0)
        centered = self._nodes[:, :2] - centroid[:2]
        covariance = centered.T @ centered / max(1, self._nodes.shape[0] - 1)
        _values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        if axis[0] < 0.0:
            axis = -axis
        yaw = math.atan2(float(axis[1]), float(axis[0]))
        return np.asarray(
            [*centroid, math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
            dtype=np.float64,
        )

    def _minimum_hand_distance(self, nodes: np.ndarray) -> float:
        samples = nodes[:: max(1, nodes.shape[0] // 512)]
        points: list[np.ndarray] = []
        for links in self._hand_links.values():
            for link in links:
                points.append(_to_numpy(link.get_pos()).reshape(-1, 3)[0])
        if not points:
            points = [self._right_tcp_pose[:3], self._left_tcp_pose[:3]]
        return float(
            min(np.min(np.linalg.norm(samples - point, axis=1)) for point in points)
        )

    def _garment_metrics(self) -> dict[str, Any]:
        lower, upper = self._fold_masks
        lower_center = np.mean(self._nodes[lower, :2], axis=0)
        upper_center = np.mean(self._nodes[upper, :2], axis=0)
        half_distance = float(np.linalg.norm(lower_center - upper_center))
        fold_ratio = half_distance / self._baseline_half_distance
        area_ratio = self._oriented_xy_area(self._nodes) / self._baseline_area
        speeds = np.linalg.norm(self._velocities, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(speeds))))
        max_speed = float(np.max(speeds))
        bounds_min = np.min(self._nodes, axis=0)
        bounds_max = np.max(self._nodes, axis=0)
        centroid = np.mean(self._nodes, axis=0)
        hand_distance = self._minimum_hand_distance(self._nodes)
        compact = bool(
            fold_ratio <= _FOLD_RATIO_THRESHOLD
            and area_ratio <= _AREA_RATIO_THRESHOLD
        )
        on_table = bool(
            bounds_min[2] >= _TABLE_TOP_Z - 0.025
            and bounds_max[2] <= _TABLE_TOP_Z + 0.24
        )
        in_workspace = bool(
            bounds_min[0] >= _TABLE_X_BOUNDS[0] - 0.02
            and bounds_max[0] <= _TABLE_X_BOUNDS[1] + 0.02
            and bounds_min[1] >= _TABLE_Y_BOUNDS[0] - 0.02
            and bounds_max[1] <= _TABLE_Y_BOUNDS[1] + 0.02
        )
        hands_released = bool(hand_distance >= _HAND_RELEASE_DISTANCE)
        stable = bool(
            rms_speed <= _STABLE_RMS_SPEED and max_speed <= _STABLE_MAX_SPEED
        )
        displacement = np.linalg.norm(self._nodes - self._reset_nodes, axis=1)
        return {
            "fold_ratio": float(fold_ratio),
            "area_ratio": float(area_ratio),
            "half_centroid_distance": half_distance,
            "z_span": float(bounds_max[2] - bounds_min[2]),
            "hand_distance": hand_distance,
            "rms_nodal_speed": rms_speed,
            "max_nodal_speed": max_speed,
            "max_reset_displacement": float(np.max(displacement)),
            "compact_fold": compact,
            "on_table": on_table,
            "in_workspace": in_workspace,
            "hands_released": hands_released,
            "stable": stable,
            "centroid": centroid,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "success_now": bool(
                compact and on_table and in_workspace and hands_released and stable
            ),
        }

    def _compute_failure(self) -> bool:
        metrics = self._last_metrics or self._garment_metrics()
        center = metrics["centroid"]
        return bool(
            metrics["bounds_min"][2] < 0.55
            or center[0] < -0.90
            or center[0] > 0.60
            or abs(float(center[1])) > 1.10
        )

    def _task_info(self) -> dict[str, Any]:
        metrics = self._last_metrics or self._garment_metrics()
        return {
            "task_id": "E01",
            "task_name": self.task_name,
            "sim_backend": "genesis",
            "cloth_solver": "libuipc_ipc",
            "garment_asset": str(self._asset_path or "DexGarmentLab/TCLC_Top578"),
            "garment_node_count": int(self._nodes.shape[0]),
            "success_counter": int(self._success_counter),
            "success_steps_required": _SUCCESS_STEPS,
            "live_control_mode": "detached_hands_offline_arm_ik",
            "arm_ik_deferred": True,
            "arm_reach_cache_active": bool(
                self._rokae_reach_points.keys() >= {"right", "left"}
            ),
            "arm_reach_guard_right_active": bool(
                self._arm_reach_guard_active["right"]
            ),
            "arm_reach_guard_left_active": bool(
                self._arm_reach_guard_active["left"]
            ),
            "arm_reach_guard_right_distance": float(
                self._arm_reach_guard_distance["right"]
            ),
            "arm_reach_guard_left_distance": float(
                self._arm_reach_guard_distance["left"]
            ),
            "hand_table_guard_right_active": bool(
                self._hand_table_guard_active["right"]
            ),
            "hand_table_guard_left_active": bool(
                self._hand_table_guard_active["left"]
            ),
            "hand_table_guard_right_distance": float(
                self._hand_table_guard_distance["right"]
            ),
            "hand_table_guard_left_distance": float(
                self._hand_table_guard_distance["left"]
            ),
            "hand_minimum_collision_z_right": float(
                self._hand_minimum_collision_z["right"]
            ),
            "hand_minimum_collision_z_left": float(
                self._hand_minimum_collision_z["left"]
            ),
            "constraint_hold_right_active": bool(
                self._constraint_hold_active["right"]
            ),
            "constraint_hold_left_active": bool(
                self._constraint_hold_active["left"]
            ),
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"centroid", "bounds_min", "bounds_max"}
            },
        }


__all__ = ["HandsOnE01GenesisEnv"]
