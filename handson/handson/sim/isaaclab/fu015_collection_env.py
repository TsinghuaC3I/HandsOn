"""Native Isaac Sim 5.1 / Isaac Lab runtime for B320-FU015."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .fu015_contract import (
    ASSET_RELATIVE_PATH,
    ASSET_SHA256,
    BASE_BODY_PATH,
    BASE_LIFT_MAX_M,
    BASE_TRANSLATION_MAX_M,
    CONTROL_HZ,
    DISPLAY_BODY_PATH,
    EXPLOSION_ANGULAR_SPEED_RADPS,
    EXPLOSION_LINEAR_SPEED_MPS,
    FORBIDDEN_CONTACT_FORCE_N,
    FU015BilateralOwnerGate,
    FU015PhysicalReadback,
    FU015ResetState,
    HINGE_PATH,
    HINGE_TRAVEL_DEG,
    LAPTOP_ROOT_PATH,
    PHYSICS_HZ,
    REQUIRED_CONTACT_FORCE_N,
    RENDER_RESOLUTION_HW,
    STABLE_ANGULAR_SPEED_RADPS,
    STABLE_LINEAR_SPEED_MPS,
    SUCCESS_HOLD_STEPS,
    TARGET_TOLERANCE_DEG,
    TASK_NAME,
    TOUCHPAD_COLLIDER_PATH,
    evaluate_physical_readback,
    sample_reset_state,
)
from .realistic_kitchen_scene import (
    REALISTIC_KITCHEN_ASSET_BYTES as _KITCHEN_PREVIEW_ASSET_BYTES,
    REALISTIC_KITCHEN_ASSET_RELATIVE_PATH as _KITCHEN_PREVIEW_ASSET_RELATIVE_PATH,
    REALISTIC_KITCHEN_ASSET_SHA256 as _KITCHEN_PREVIEW_ASSET_SHA256,
    REALISTIC_KITCHEN_ISLAND_MAX_M as _KITCHEN_PREVIEW_ISLAND_EXPECTED_MAX_M,
    REALISTIC_KITCHEN_ISLAND_MIN_M as _KITCHEN_PREVIEW_ISLAND_EXPECTED_MIN_M,
    REALISTIC_KITCHEN_LICENSE as _KITCHEN_PREVIEW_LICENSE,
    REALISTIC_KITCHEN_PREVIEW_ENV,
    REALISTIC_KITCHEN_PREVIEW_ROOT,
    REALISTIC_KITCHEN_SCALE as _KITCHEN_PREVIEW_SCALE,
    REALISTIC_KITCHEN_SOURCE_NODE as _KITCHEN_PREVIEW_SOURCE_NODE,
    REALISTIC_KITCHEN_SOURCE_URL as _KITCHEN_PREVIEW_SOURCE_URL,
    REALISTIC_KITCHEN_TEXTURE_BYTES as _KITCHEN_PREVIEW_TEXTURE_BYTES,
    REALISTIC_KITCHEN_TEXTURE_COUNT as _KITCHEN_PREVIEW_TEXTURE_COUNT,
    REALISTIC_KITCHEN_TRANSLATE_M as _KITCHEN_PREVIEW_TRANSLATE_M,
    realistic_kitchen_preview_enabled,
    validate_realistic_kitchen_asset,
)


_TABLE_SUPPORT_GAP_M = 0.001
# With the reachable fixed-wrist trajectory, stiffness 6.0, 9.0, and 18.0
# settled at 95.29, 95.49, and 96.02 degrees while all role contacts and the
# base remained stable.  At stiffness 18 the spring output is already about
# 2.82 against the 3.0 cap.  Keep the spring decisively inside a modest 4.0
# limit so motion remains bounded by that explicit contact-gated force.
_HINGE_DRIVE_STIFFNESS = 36.0
_HINGE_DRIVE_DAMPING = 1.2
_HINGE_DRIVE_MAX_FORCE = 4.0
_BASE_FIXED_JOINT_PATH = f"{BASE_BODY_PATH}/world_fixed_joint"
# A downward approach must not accumulate an opening-drive target merely
# because it touches the closed display.  Requiring 0.2 mm of upward TCP
# command per 30 Hz frame rejects stationary/downward tracker noise while
# remaining far below the intentional hinge-arc motion used for collection.
_RIGHT_OPENING_INTENT_MIN_DELTA_M = 0.0002
# At the closed stop, contact plus a deliberate lift is required before the
# joint brake releases.  Away from this small window, exact right-display
# contact alone permits physical hinge motion.
_HINGE_CLOSED_LOCK_WINDOW_DEG = 0.25
# The former 0.15 deg/substep advanced 0.6 degrees within one control interval.
# At the live 13.53-degree reset the display then moved away from the held wrist
# between action updates, repeatedly broke the exact-owner gate, and fell back
# around 17 degrees.  Limit the change to 0.4 degrees per control interval so
# the tracked hand can retain contact.  Even from the 11-degree reset boundary,
# this rate can reach 105 degrees and complete the 15-step hold before the
# unchanged 660-step timeout.
_ASSIST_DEGREES_PER_PHYSICS_STEP = 0.40 * CONTROL_HZ / PHYSICS_HZ
_DISPLAY_TOP_EDGE_BAND_M = 0.045
_TOUCHPAD_POINT_MARGIN_M = 0.004
_FRONT_PROXY_POINT_MARGIN_M = 0.001
# Bounds are expressed directly in E_body_1's rigid-body frame.  They cover
# only the front 45 mm and the full 6.5 mm thickness of the audited
# P_1e5d57b88817687f shell; the mesh's ancestor transform must not be applied
# a second time when this child collider is authored under E_body_1.
_BASE_FRONT_PROXY_LOWER_M = (-0.1641000062, -0.0020000001, -0.0023000001)
_BASE_FRONT_PROXY_UPPER_M = (0.1641000062, 0.0429999999, 0.0041999999)
_BASE_FRONT_PROXY_CENTER_M = tuple(
    0.5 * (lower + upper)
    for lower, upper in zip(_BASE_FRONT_PROXY_LOWER_M, _BASE_FRONT_PROXY_UPPER_M)
)
_BASE_FRONT_PROXY_SIZE_M = tuple(
    upper - lower
    for lower, upper in zip(_BASE_FRONT_PROXY_LOWER_M, _BASE_FRONT_PROXY_UPPER_M)
)
_BASE_COLLISION_PROXY_PATH = f"{BASE_BODY_PATH}/FU015BaseCollisionProxy"
_BASE_COLLISION_PROXY_CENTER_M = (0.0, 0.11295, -0.00106395)
_BASE_COLLISION_PROXY_SIZE_M = (0.3282, 0.2299, 0.0105279)
_DISPLAY_COLLISION_PROXY_PATH = (
    f"{DISPLAY_BODY_PATH}/FU015DisplayCollisionProxy"
)
_DISPLAY_COLLISION_PROXY_CENTER_M = (0.1264, 0.00355, 0.11255)
_DISPLAY_COLLISION_PROXY_SIZE_M = (0.3282, 0.0135, 0.2299)
_MATERIAL_PATH = "/World/PhysicsMaterials/FU015Laptop"
_LIGHT_ROOT = "/World/Task/FU015Lights"
_KITCHEN_PREVIEW_ENV = REALISTIC_KITCHEN_PREVIEW_ENV
_KITCHEN_PREVIEW_ROOT = REALISTIC_KITCHEN_PREVIEW_ROOT
_KITCHEN_PREVIEW_ASSET_ROOT = f"{_KITCHEN_PREVIEW_ROOT}/Asset"
# Keep FU015 on the shared ABC robot-root contract.  Task-specific asymmetric
# base offsets made the reset palms differ by 100 mm in X even though their
# mirrored joint homes and heights matched.
_RIGHT_ROBOT_BASE_X_OFFSET_M = 0.0
_LEFT_ROBOT_BASE_X_OFFSET_M = 0.0


def _fu015_env_flag(name: str, default: bool = False) -> bool:
    if name != REALISTIC_KITCHEN_PREVIEW_ENV:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        if raw == "1":
            return True
        if raw == "0":
            return False
        raise ValueError(f"{name} must be exactly '1' or '0'; got {raw!r}")
    return realistic_kitchen_preview_enabled(default=default)


def summarize_contact_forces_by_filter(
    filter_paths: tuple[str, ...],
    filter_indices: np.ndarray,
    forces: np.ndarray,
) -> dict[str, float]:
    """Aggregate diagnostic contact force without changing failure semantics."""

    indices = np.asarray(filter_indices, dtype=np.int64).reshape(-1)
    magnitudes = np.asarray(forces, dtype=np.float64).reshape(-1)
    if indices.shape != magnitudes.shape:
        raise ValueError("FU015 contact filter indices and forces must align")
    if indices.size and (
        int(np.min(indices)) < 0 or int(np.max(indices)) >= len(filter_paths)
    ):
        raise ValueError("FU015 contact filter index is outside the path contract")
    totals = np.zeros(len(filter_paths), dtype=np.float64)
    np.add.at(totals, indices, magnitudes)
    return {
        path: float(force)
        for path, force in zip(filter_paths, totals)
        if force > 0.0
    }


def summarize_contact_points_by_filter(
    filter_paths: tuple[str, ...],
    filter_indices: np.ndarray,
    points: np.ndarray,
    forces: np.ndarray,
    *,
    max_points_per_path: int = 8,
) -> dict[str, list[list[float]]]:
    """Retain bounded world-space points for nonzero diagnostic contacts."""

    indices = np.asarray(filter_indices, dtype=np.int64).reshape(-1)
    positions = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    magnitudes = np.asarray(forces, dtype=np.float64).reshape(-1)
    if indices.shape != magnitudes.shape or positions.shape[0] != indices.size:
        raise ValueError("FU015 contact points, filter indices, and forces must align")
    if int(max_points_per_path) <= 0:
        raise ValueError("FU015 diagnostic point bound must be positive")
    if indices.size and (
        int(np.min(indices)) < 0 or int(np.max(indices)) >= len(filter_paths)
    ):
        raise ValueError("FU015 contact filter index is outside the path contract")

    result: dict[str, list[list[float]]] = {}
    for filter_index, point, force in zip(indices, positions, magnitudes):
        if force <= 0.0:
            continue
        samples = result.setdefault(filter_paths[int(filter_index)], [])
        if len(samples) < int(max_points_per_path):
            samples.append([float(value) for value in point])
    return result


def summarize_contact_manifold_points_by_filter(
    filter_paths: tuple[str, ...],
    filter_indices: np.ndarray,
    points: np.ndarray,
    *,
    max_points_per_path: int = 8,
) -> dict[str, list[list[float]]]:
    """Retain bounded manifold points even when their normal force is zero."""

    indices = np.asarray(filter_indices, dtype=np.int64).reshape(-1)
    return summarize_contact_points_by_filter(
        filter_paths,
        indices,
        points,
        np.ones(indices.shape, dtype=np.float64),
        max_points_per_path=max_points_per_path,
    )


def points_in_body_box(
    world_points: np.ndarray,
    body_pose_wxyz_m: np.ndarray,
    lower_m: tuple[float, float, float],
    upper_m: tuple[float, float, float],
    *,
    margin_m: float = 0.0,
) -> np.ndarray:
    """Test world points against a box expressed in a rigid body's frame."""

    points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(body_pose_wxyz_m, dtype=np.float64).reshape(7)
    quaternion = pose[3:7]
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("FU015 body pose quaternion must be finite and nonzero")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    local_points = (points - pose[:3]) @ rotation
    lower = np.asarray(lower_m, dtype=np.float64) - float(margin_m)
    upper = np.asarray(upper_m, dtype=np.float64) + float(margin_m)
    return np.all(local_points >= lower, axis=1) & np.all(
        local_points <= upper, axis=1
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fu015_kitchen_preview_asset(
    repository_root: Path | None = None,
) -> Path:
    """Backward-compatible FU015 name for the shared asset validator."""

    return validate_realistic_kitchen_asset(repository_root)


def _quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64).reshape(4)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _transform_composed_root_pose(
    reference_pose_xyzw: np.ndarray,
    reference_reset: FU015ResetState,
    target_reset: FU015ResetState,
) -> np.ndarray:
    """Move a composed articulation root between randomized task frames.

    ``reference_pose_xyzw`` is the cached PhysX root-link pose after the USD
    asset was composed with ``reference_reset``.  Its XY is therefore not the
    task-root XY: it also contains the source asset's rigid-body offset.  Move
    that offset with the task frame instead of overwriting it.
    """

    pose = np.asarray(reference_pose_xyzw, dtype=np.float64).reshape(7).copy()
    if (
        float(reference_reset.root_x_m) == float(target_reset.root_x_m)
        and float(reference_reset.root_y_m) == float(target_reset.root_y_m)
        and float(reference_reset.root_yaw_deg) == float(target_reset.root_yaw_deg)
    ):
        return pose

    reference_yaw = math.radians(float(reference_reset.root_yaw_deg))
    target_yaw = math.radians(float(target_reset.root_yaw_deg))

    def rotation_xy(yaw: float) -> np.ndarray:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)

    reference_origin = np.asarray(
        [reference_reset.root_x_m, reference_reset.root_y_m], dtype=np.float64
    )
    target_origin = np.asarray(
        [target_reset.root_x_m, target_reset.root_y_m], dtype=np.float64
    )
    body_offset_in_task_frame = rotation_xy(reference_yaw).T @ (
        pose[:2] - reference_origin
    )
    pose[:2] = target_origin + rotation_xy(target_yaw) @ body_offset_in_task_frame

    yaw_delta = target_yaw - reference_yaw
    delta_quat = np.asarray(
        [0.0, 0.0, math.sin(yaw_delta / 2.0), math.cos(yaw_delta / 2.0)],
        dtype=np.float64,
    )
    pose[3:7] = _quat_mul_xyzw(delta_quat, pose[3:7])
    return pose


def _configure_fu015_robot_cfg(robot_cfg: Any, side: str) -> Any:
    """Place both fixed arms within FU015's audited randomized workspace."""

    offset_m = {
        "right": _RIGHT_ROBOT_BASE_X_OFFSET_M,
        "left": _LEFT_ROBOT_BASE_X_OFFSET_M,
    }.get(side, 0.0)
    if offset_m:
        init_state = robot_cfg.init_state
        position = tuple(float(value) for value in init_state.pos)
        init_state.pos = (
            position[0] + offset_m,
            position[1],
            position[2],
        )
    return robot_cfg


def _author_fu015_fixed_base_articulation(stage: Any, base: Any) -> Any:
    """Convert the imported floating laptop into a native fixed-base tree."""

    from pxr import PhysxSchema, UsdPhysics

    # PhysX only classifies the tree as fixed-base when the articulation API
    # lives on the inbound world joint.  Applying a separate constraint to a
    # floating articulation (the former kinematic-anchor implementation) leaves
    # the root movable under hand/table impulses.
    if base.HasAPI(UsdPhysics.ArticulationRootAPI):
        base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if base.HasAPI(PhysxSchema.PhysxArticulationAPI):
        base.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    UsdPhysics.RigidBodyAPI(base).CreateKinematicEnabledAttr().Set(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(base).CreateDisableGravityAttr().Set(True)

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, _BASE_FIXED_JOINT_PATH)
    fixed_joint.CreateBody1Rel().SetTargets([base.GetPath()])
    fixed_joint.CreateJointEnabledAttr().Set(True)
    fixed_root = fixed_joint.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(fixed_root)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(fixed_root)
    articulation.CreateEnabledSelfCollisionsAttr(False)
    return fixed_joint


def _fu015_is_articulation_link_path(path: str) -> bool:
    return str(path) in {BASE_BODY_PATH, DISPLAY_BODY_PATH}


def _right_tcp_opening_intent(
    previous_z_m: float | None,
    current_z_m: float,
    *,
    min_delta_m: float = _RIGHT_OPENING_INTENT_MIN_DELTA_M,
) -> bool:
    """Return whether the right-hand command intentionally lifts the lid."""

    if previous_z_m is None:
        return False
    previous = float(previous_z_m)
    current = float(current_z_m)
    threshold = float(min_delta_m)
    if not (math.isfinite(previous) and math.isfinite(current)):
        return False
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("FU015 opening-intent threshold must be finite and nonnegative")
    return current - previous + 1.0e-12 >= threshold


def _hinge_motion_allowed(
    *,
    right_display_contact: bool,
    hinge_angle_deg: float,
    opening_intent: bool,
) -> bool:
    """Allow lid motion only under real display contact and safe intent."""

    angle = float(hinge_angle_deg)
    if not right_display_contact or not math.isfinite(angle):
        return False
    # Contact alone must not let a downward press open a fully closed lid.
    # Once the lid is visibly open, contact may move it in either direction.
    return bool(
        angle < -_HINGE_CLOSED_LOCK_WINDOW_DEG or opening_intent
    )


def _next_hinge_drive_target_deg(
    *,
    current_angle_deg: float,
    drive_target_deg: float,
    task_target_deg: float,
    assist_increment_deg: float,
) -> float:
    """Advance the opening assist without ever retracting an overshot lid."""

    target = float(np.clip(drive_target_deg, -HINGE_TRAVEL_DEG, 0.0))
    if math.isfinite(current_angle_deg):
        # Negative angles are more open.  If the hand opens beyond the current
        # drive target, follow that physical angle so releasing the lid cannot
        # spring it back toward the task's scoring target.
        target = min(
            target,
            float(np.clip(current_angle_deg, -HINGE_TRAVEL_DEG, 0.0)),
        )
    scoring_target = -float(np.clip(task_target_deg, 0.0, HINGE_TRAVEL_DEG))
    if assist_increment_deg > 0.0 and target > scoring_target:
        target = max(scoring_target, target - float(assist_increment_deg))
    return float(np.clip(target, -HINGE_TRAVEL_DEG, 0.0))


class HandsOnFU015IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Physical laptop-opening environment with a reset-positioned fixed base.

    The only task assist is an explicitly reported PhysX hinge drive target.
    It advances after, and only after, right-display contact is read from the
    exact side-specific contact view.  The native fixed-base articulation root
    is repositioned only in the reset transaction; no object pose or joint
    state is written after action zero.
    """

    _runtime_task_name = TASK_NAME

    def _initial_laptop_reset(self, seed: int):
        return sample_reset_state(seed)

    def __init__(self, task_name: str = TASK_NAME, *args: Any, **kwargs: Any):
        if task_name != self._runtime_task_name:
            raise ValueError(f"Laptop runtime requires {self._runtime_task_name!r}, got {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).strip().lower()
        if physics_backend not in {"physx", "isaacsim", "isaac"}:
            raise ValueError("B320-FU015 is Isaac Sim 5.1 PhysX-only; Newton is forbidden")
        kwargs["physics_backend"] = "physx"
        kwargs.setdefault("physics_dt", 1.0 / PHYSICS_HZ)
        kwargs.setdefault("control_dt", 1.0 / CONTROL_HZ)
        kwargs.setdefault("hz", CONTROL_HZ)
        kwargs.setdefault("time_limit", 22.0)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)
        self._fu015_kitchen_preview_enabled = _fu015_env_flag(
            _KITCHEN_PREVIEW_ENV,
            False,
        )
        self._fu015_kitchen_preview_mesh_count = 0
        self._fu015_kitchen_preview_island_bounds_m: dict[str, list[float]] = {}

        # FU015 records three RTX camera streams in disposable Isaac workers.
        # The ordinary SimulationApp teardown can spin forever after capture;
        # use the base runtime's existing skip-cleanup close path so the fresh
        # collection supervisor can observe a clean worker exit and start its
        # independent replay process.
        self._fast_shutdown_on_close = True

        requested_seed = 0 if kwargs.get("seed") is None else int(kwargs["seed"])
        self._fu015_reset_serial = -1
        self._fu015_reset = self._initial_laptop_reset(requested_seed)
        # The cached articulation root state is composed from this immutable
        # task frame during super().__init__.  Later resets are transformed
        # relative to it so the asset's internal base-link offset is retained.
        self._fu015_composed_reset = self._fu015_reset
        self._fu015_articulation_view: Any | None = None
        self._fu015_base_view: Any | None = None
        self._fu015_display_view: Any | None = None
        self._fu015_contact_views: dict[str, Any] = {}
        self._fu015_contact_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._fu015_robot_table_force_by_path_n: dict[str, float] = {}
        self._fu015_robot_table_points_by_path_m: dict[
            str, list[list[float]]
        ] = {}
        self._fu015_raw_contact_diagnostic: dict[str, Any] = {}
        self._fu015_reset_base_position = np.asarray(
            [self._fu015_reset.root_x_m, self._fu015_reset.root_y_m, 0.891],
            dtype=np.float64,
        )
        self._fu015_reset_base_pose_wxyz = np.concatenate(
            [
                self._fu015_reset_base_position,
                np.asarray([1.0, 0.0, 0.0, 0.0]),
            ]
        )
        self._fu015_stability_probe_substeps = 0
        self._fu015_last_metrics: dict[str, float | bool] = {}
        self._fu015_last_readback: FU015PhysicalReadback | None = None
        self._fu015_failure_latched = False
        self._fu015_failure_reported = False
        self._fu015_bilateral_owner_gate = FU015BilateralOwnerGate()
        self._fu015_drive_target_deg = float(self._fu015_reset.hinge_angle_deg)
        self._fu015_assist_active = False
        self._fu015_assist_substeps = 0
        self._fu015_previous_right_tcp_z_m: float | None = None
        self._fu015_right_command_delta_z_m = 0.0
        self._fu015_right_opening_intent = False
        self._fu015_right_display_contact_active = False
        self._fu015_hinge_contact_lock_active = True
        self._fu015_hinge_lock_angle_deg = float(
            self._fu015_reset.hinge_angle_deg
        )
        self._fu015_hinge_limit_signature: tuple[bool, float] | None = None
        self._fu015_base_nominal_mass = 1.0
        self._fu015_display_nominal_mass = 0.5
        self._fu015_mass_attrs: dict[str, Any] = {}
        self._fu015_material_api: Any | None = None
        self._fu015_contact_contract_ready = False
        self._fu015_preview_transit_reported = False
        self._fu015_preview_left_role_reported = False
        self._fu015_preview_right_role_reported = False
        self._fu015_preview_terminal_reported = False
        self._fu015_preview_left_transit_peak_force_n = 0.0
        self._fu015_preview_right_transit_peak_force_n = 0.0
        # The first GPU run produced an invalid canonical signature through
        # the full-report path while all four exact contact views initialized.
        # Prefer that public RigidContactView payload so the recorded signature
        # remains path-based and fail-closed.
        self._physx_prefer_tensor_contact_capture = True
        super().__init__(task_name, *args, **kwargs)

    def _configure_task_robot_cfg(self, robot_cfg: Any, side: str) -> Any:
        robot_cfg = super()._configure_task_robot_cfg(robot_cfg, side)
        return _configure_fu015_robot_cfg(robot_cfg, side)

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "laptop_base_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "laptop_display_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "hinge_angle": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "base_displacement": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "bimanual_role_contacts": spaces.Box(-np.inf, np.inf, shape=(5,)),
                "fu015_task_state": spaces.Box(-np.inf, np.inf, shape=(17,)),
            }
        )
        return observation_space

    def _spawn_fu015_macbook_visual_variant(
        self,
        *,
        base_prim: Any,
        display_prim: Any,
    ) -> None:
        """Replace the canonical laptop meshes with one render-only MacBook variant."""
        from pathlib import Path

        from pxr import Usd, UsdGeom

        if self._stage is None:
            raise RuntimeError("FU015 stage is unavailable before visual composition")
        assets_dir = (
            Path(__file__).resolve().parents[1]
            / "envs"
            / "usd"
            / "t16_open_laptop"
        )
        variants = (
            ("Apple MacBook Pro 16-inch 2021", "macbook_pro_16_2021"),
            ("MacBook Pro M5 Max 16-inch 2026", "macbook_pro_m5_max_16_2026"),
        )
        variant_name, variant_dirname = variants[
            int(self._fu015_reset.seed) % len(variants)
        ]
        variant_dir = assets_dir / variant_dirname
        base_visual_asset = variant_dir / "macbook_base_visual.usdc"
        display_visual_asset = variant_dir / "macbook_display_visual.usdc"
        missing_assets = [
            path
            for path in (base_visual_asset, display_visual_asset)
            if not path.is_file()
        ]
        if missing_assets:
            raise FileNotFoundError(
                "FU015 MacBook visual assets are missing: "
                + ", ".join(str(path) for path in missing_assets)
            )

        # Visibility does not affect the canonical collision meshes or rigid bodies.
        for rigid_prim in (base_prim, display_prim):
            for prim in Usd.PrimRange(rigid_prim):
                if prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Imageable(prim).MakeInvisible()

        base_visual = UsdGeom.Xform.Define(
            self._stage, f"{BASE_BODY_PATH}/MacBookVisual"
        )
        display_visual = UsdGeom.Xform.Define(
            self._stage, f"{DISPLAY_BODY_PATH}/MacBookVisual"
        )
        base_visual.GetPrim().GetReferences().AddReference(base_visual_asset.as_posix())
        display_visual.GetPrim().GetReferences().AddReference(
            display_visual_asset.as_posix()
        )
        self._fu015_laptop_visual_variant = variant_name
        self._fu015_laptop_variant_dir = variant_dir
        print(f"[HandsOn/FU015] Laptop visual variant: {variant_name}")

    def _spawn_fu015_macbook_collision_variant(self, sim_utils: Any) -> None:
        """Replace the box proxies with compact convex hulls derived from the variant."""
        from pxr import Usd, UsdGeom, UsdPhysics

        if self._stage is None:
            raise RuntimeError("FU015 stage is unavailable before collision composition")
        variant_dir = getattr(self, "_fu015_laptop_variant_dir", None)
        if variant_dir is None:
            raise RuntimeError("FU015 visual variant must be selected before collision setup")

        collision_specs = (
            (
                _BASE_COLLISION_PROXY_PATH,
                f"{BASE_BODY_PATH}/MacBookAssetCollision",
                variant_dir / "macbook_base_collision.usdc",
            ),
            (
                _DISPLAY_COLLISION_PROXY_PATH,
                f"{DISPLAY_BODY_PATH}/MacBookAssetCollision",
                variant_dir / "macbook_display_collision.usdc",
            ),
        )
        missing_assets = [
            asset_path
            for _, _, asset_path in collision_specs
            if not asset_path.is_file()
        ]
        if missing_assets:
            raise FileNotFoundError(
                "FU015 MacBook collision assets are missing: "
                + ", ".join(str(path) for path in missing_assets)
            )

        for proxy_path, collision_path, asset_path in collision_specs:
            proxy = self._stage.GetPrimAtPath(proxy_path)
            if not proxy.IsValid():
                raise RuntimeError(f"FU015 collision proxy is unavailable: {proxy_path}")

            inherited_relationships = []
            for relationship in proxy.GetRelationships():
                if relationship.GetName().startswith("material:binding"):
                    inherited_relationships.append(
                        (relationship.GetName(), relationship.GetTargets())
                    )
            inherited_physx_attributes = []
            for attribute in proxy.GetAttributes():
                if attribute.GetName().startswith("physxCollision:"):
                    value = attribute.Get()
                    if value is not None:
                        inherited_physx_attributes.append(
                            (
                                attribute.GetName(),
                                attribute.GetTypeName(),
                                attribute.IsCustom(),
                                value,
                            )
                        )

            UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(False)
            collision_root = UsdGeom.Xform.Define(self._stage, collision_path)
            collision_root.GetPrim().GetReferences().AddReference(asset_path.as_posix())
            mesh_prims = [
                prim
                for prim in Usd.PrimRange(collision_root.GetPrim())
                if prim.IsA(UsdGeom.Mesh)
            ]
            if len(mesh_prims) != 1:
                raise RuntimeError(
                    f"FU015 collision asset must contain one hull: {asset_path}"
                )
            hull = mesh_prims[0]
            UsdGeom.Imageable(hull).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(hull).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(hull).CreateApproximationAttr().Set(
                "convexHull"
            )
            for relationship_name, targets in inherited_relationships:
                hull.CreateRelationship(relationship_name).SetTargets(targets)
            for name, type_name, custom, value in inherited_physx_attributes:
                hull.CreateAttribute(name, type_name, custom=custom).Set(value)

    def _spawn_fu015_kitchen_preview(self) -> None:
        """Backward-compatible entry point for the shared kitchen layer."""

        self._spawn_realistic_kitchen_preview()

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("FU015 stage is unavailable before asset composition")

        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

        package_root = Path(__file__).resolve().parents[2]
        asset_path = package_root / ASSET_RELATIVE_PATH
        if not asset_path.is_file():
            raise FileNotFoundError(f"FU015 exact asset is missing: {asset_path}")
        actual_sha = _sha256(asset_path)
        if actual_sha != ASSET_SHA256:
            raise RuntimeError(
                f"FU015 asset SHA mismatch: expected {ASSET_SHA256}, got {actual_sha}"
            )

        root = UsdGeom.Xform.Define(self._stage, LAPTOP_ROOT_PATH).GetPrim()
        root.GetReferences().ClearReferences()
        root.GetReferences().AddReference(str(asset_path.resolve()))
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(
                float(self._fu015_reset.root_x_m),
                float(self._fu015_reset.root_y_m),
                0.89 + _TABLE_SUPPORT_GAP_M,
            )
        )
        xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(
            float(self._fu015_reset.root_yaw_deg)
        )

        base = self._stage.GetPrimAtPath(BASE_BODY_PATH)
        display = self._stage.GetPrimAtPath(DISPLAY_BODY_PATH)
        joint = self._stage.GetPrimAtPath(HINGE_PATH)
        touchpad = self._stage.GetPrimAtPath(TOUCHPAD_COLLIDER_PATH)
        if not all(prim.IsValid() for prim in (base, display, joint, touchpad)):
            raise RuntimeError("FU015 composed base/display/hinge/touchpad is incomplete")
        hinge_joint = UsdPhysics.RevoluteJoint(joint)
        hinge_joint.CreateLowerLimitAttr(-float(HINGE_TRAVEL_DEG))
        hinge_joint.CreateUpperLimitAttr(0.0)
        self._spawn_fu015_macbook_visual_variant(
            base_prim=base,
            display_prim=display,
        )

        # A world inbound joint with the articulation API on the joint itself is
        # PhysX's fixed-base convention.  Root tensor writes can still reposition
        # this fixed articulation during reset, but contacts cannot move it while
        # an episode is running.
        _author_fu015_fixed_base_articulation(self._stage, base)
        # The source display is intentionally gravity-compensated.  It remains
        # a dynamic articulation link and all opening motion is solved by PhysX.
        PhysxSchema.PhysxRigidBodyAPI.Apply(display).CreateDisableGravityAttr(True)
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateTargetPositionAttr(float(self._fu015_reset.hinge_angle_deg))
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(_HINGE_DRIVE_STIFFNESS)
        drive.CreateDampingAttr(_HINGE_DRIVE_DAMPING)
        drive.CreateMaxForceAttr(_HINGE_DRIVE_MAX_FORCE)
        if joint.HasAPI(PhysxSchema.JointStateAPI, "angular"):
            joint_state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        else:
            joint_state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        joint_state.GetPositionAttr().Set(float(self._fu015_reset.hinge_angle_deg))
        joint_state.GetVelocityAttr().Set(0.0)

        sim_utils.make_uninstanceable(LAPTOP_ROOT_PATH, stage=self._stage)
        # The source asset authors eight overlapping base mesh colliders and
        # five overlapping display colliders. Several cannot be cooked as GPU
        # convexes, which forces CPU collision detection and makes two 60 Hz
        # substeps exceed one 30 Hz command period. Preserve all render meshes
        # and region geometry, but replace their collision APIs with one exact
        # rigid-local bounding box per articulation link.
        source_colliders_disabled = 0
        for rigid_prim in (base, display):
            for collider_prim in Usd.PrimRange(rigid_prim):
                if not collider_prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = UsdPhysics.CollisionAPI(collider_prim)
                enabled_attr = collision_api.GetCollisionEnabledAttr()
                if not enabled_attr or not enabled_attr.IsValid():
                    enabled_attr = collision_api.CreateCollisionEnabledAttr()
                enabled_attr.Set(False)
                source_colliders_disabled += 1

        for proxy_path, center, size in (
            (
                _BASE_COLLISION_PROXY_PATH,
                _BASE_COLLISION_PROXY_CENTER_M,
                _BASE_COLLISION_PROXY_SIZE_M,
            ),
            (
                _DISPLAY_COLLISION_PROXY_PATH,
                _DISPLAY_COLLISION_PROXY_CENTER_M,
                _DISPLAY_COLLISION_PROXY_SIZE_M,
            ),
        ):
            proxy = UsdGeom.Cube.Define(self._stage, proxy_path)
            proxy.CreateSizeAttr(1.0)
            proxy.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*center)
            )
            proxy.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size))
            UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
            proxy_collision = UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
            proxy_collision.CreateCollisionEnabledAttr(True)

        print(
            "[HandsOn/FU015] GPU collision proxies ready: "
            f"source_meshes_disabled={source_colliders_disabled}, "
            "analytic_boxes=2 (base and display; no auxiliary front-edge strip)",
            flush=True,
        )
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self._fu015_reset.friction),
            dynamic_friction=float(self._fu015_reset.friction),
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(_MATERIAL_PATH, material_cfg)
        sim_utils.bind_physics_material(
            LAPTOP_ROOT_PATH,
            _MATERIAL_PATH,
            stage=self._stage,
            stronger_than_descendants=True,
        )
        material_prim = self._stage.GetPrimAtPath(_MATERIAL_PATH)
        self._fu015_material_api = UsdPhysics.MaterialAPI(material_prim)

        for label, prim, fallback_mass in (
            ("base", base, 1.0),
            ("display", display, 0.5),
        ):
            mass_api = UsdPhysics.MassAPI(prim)
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_attr = mass_api.GetMassAttr()
            if not mass_attr or not mass_attr.IsValid():
                mass_attr = mass_api.CreateMassAttr()
            authored_mass = mass_attr.Get()
            nominal_mass = (
                float(authored_mass)
                if authored_mass is not None and float(authored_mass) > 0.0
                else fallback_mass
            )
            mass_attr.Set(nominal_mass * float(self._fu015_reset.mass_scale))
            self._fu015_mass_attrs[label] = mass_attr
            setattr(self, f"_fu015_{label}_nominal_mass", nominal_mass)

        for prim in (base, display):
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage, prim.GetPath()
                )
            else:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)

        dome = UsdLux.DomeLight.Define(self._stage, f"{_LIGHT_ROOT}/DomeLight")
        dome.CreateIntensityAttr(450.0)
        dome.CreateColorAttr(Gf.Vec3f(0.86, 0.90, 1.0))
        distant = UsdLux.DistantLight.Define(
            self._stage, f"{_LIGHT_ROOT}/DistantLight"
        )
        distant.CreateIntensityAttr(1800.0)
        distant.CreateAngleAttr(2.0)
        distant.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 15.0, 25.0))

        self._configure_fu015_render_mode()
        self._spawn_fu015_macbook_collision_variant(sim_utils)
        if self.task_name == "eval_comb02_open_laptop_and_click_mouse":
            from .xy2_appearance import apply_xy2_visual_pilot, xy2_visual_pilot_enabled

            if xy2_visual_pilot_enabled():
                report = apply_xy2_visual_pilot(
                    self._stage, self._find_rigid_body_prim("mousepad"),
                    base_visual_path=f"{BASE_BODY_PATH}/MacBookVisual",
                    display_visual_path=f"{DISPLAY_BODY_PATH}/MacBookVisual",
                )
                self.xy2_appearance_report = report
                print(f"[HandsOn/XY2] Visual pilot: {json.dumps(report, sort_keys=True)}", flush=True)

    def _configure_fu015_render_mode(self) -> None:
        from .runtime_display import isaac_ultra_rendering_enabled

        # Task construction follows the shared offline renderer setup. Do
        # not undo that setup or an explicitly selected PathTracing mode.
        if self._path_tracing_capture or isaac_ultra_rendering_enabled():
            return
        try:
            import carb

            settings = carb.settings.get_settings()
            if settings.get("/rtx/rendermode") != "PathTracing":
                settings.set("/rtx/rendermode", "RayTracedLighting")
        except Exception as exc:
            self._add_task_state_warning(f"Could not select RayTracedLighting: {exc}")

    def _robot_rigid_body_paths(self, side: str, *, hand_only: bool) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            name = str(prim.GetName()).lower()
            if hand_only and not (
                name == f"{side}_palm_link" or name.startswith(f"{side}_finger")
            ):
                continue
            paths.append(str(prim.GetPath()))
        return paths

    def _create_fu015_contact_view(
        self,
        physics_view: Any,
        key: str,
        sensor_path: str,
        filter_paths: list[str],
    ) -> None:
        if not filter_paths:
            raise RuntimeError(f"FU015 contact filter {key} resolved no rigid bodies")
        view = physics_view.create_rigid_contact_view(
            sensor_path,
            filter_patterns=filter_paths,
            max_contact_data_count=max(128, len(filter_paths) * 16),
        )
        if (
            view is None
            or int(view.sensor_count) <= 0
            or int(view.filter_count) != len(filter_paths)
        ):
            raise RuntimeError(
                f"FU015 contact view {key} is empty "
                f"(filters={len(filter_paths)})"
            )
        self._fu015_contact_views[key] = view
        self._fu015_contact_paths[key] = (sensor_path, tuple(filter_paths))

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            raise RuntimeError("FU015 PhysX tensor simulation view is unavailable")
        self._fu015_articulation_view = physics_view.create_articulation_view(
            _BASE_FIXED_JOINT_PATH
        )
        self._fu015_base_view = physics_view.create_rigid_body_view(BASE_BODY_PATH)
        self._fu015_display_view = physics_view.create_rigid_body_view(DISPLAY_BODY_PATH)
        if (
            int(self._fu015_articulation_view.count) != 1
            or int(self._fu015_base_view.count) != 1
            or int(self._fu015_display_view.count) != 1
        ):
            raise RuntimeError("FU015 physical articulation/body views are incomplete")
        articulation_metatype = self._fu015_articulation_view.get_metatype(0)
        if not bool(articulation_metatype.fixed_base):
            raise RuntimeError(
                "FU015 laptop parsed as a floating articulation; fixed base is required"
            )
        print(
            "[HandsOn/FU015] Native fixed-base articulation ready: fixed_base=True",
            flush=True,
        )

        left_hand = self._robot_rigid_body_paths("left", hand_only=True)
        right_hand = self._robot_rigid_body_paths("right", hand_only=True)
        all_robot = self._robot_rigid_body_paths("left", hand_only=False)
        all_robot.extend(self._robot_rigid_body_paths("right", hand_only=False))
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            raise RuntimeError("FU015 table rigid body is unavailable")
        self._create_fu015_contact_view(
            physics_view, "left_base", BASE_BODY_PATH, left_hand
        )
        self._create_fu015_contact_view(
            physics_view, "left_display", DISPLAY_BODY_PATH, left_hand
        )
        self._create_fu015_contact_view(
            physics_view, "right_base", BASE_BODY_PATH, right_hand
        )
        self._create_fu015_contact_view(
            physics_view, "right_display", DISPLAY_BODY_PATH, right_hand
        )
        self._create_fu015_contact_view(
            physics_view, "robot_table", str(table_prim.GetPath()), all_robot
        )
        self._fu015_contact_contract_ready = True
        print(
            "[HandsOn/FU015] Exact contact contract ready: "
            f"left_hand={len(left_hand)}, right_hand={len(right_hand)}, "
            f"robot_table_filters={len(all_robot)}",
            flush=True,
        )

    def _capture_task_scene_state(self, task_mjcf: Path) -> None:
        """Cache the laptop only through its reduced-coordinate articulation."""

        super()._capture_task_scene_state(task_mjcf)
        before = len(self._task_reset_body_states)
        self._task_reset_body_states[:] = [
            state
            for state in self._task_reset_body_states
            if not _fu015_is_articulation_link_path(state[0])
        ]
        remaining_links = [
            state[0]
            for state in self._task_reset_body_states
            if _fu015_is_articulation_link_path(state[0])
        ]
        if remaining_links:
            raise RuntimeError(
                "FU015 reset cache retained non-root articulation links: "
                f"{remaining_links}"
            )
        print(
            "[HandsOn/FU015] Reset cache uses articulation-only laptop restore: "
            f"filtered_rigid_links={before - len(self._task_reset_body_states)}",
            flush=True,
        )

    def reset(self, seed: int | None = None, **kwargs: Any):
        if seed is not None:
            self._fu015_reset_serial = -1
        return super().reset(seed=seed, **kwargs)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._fu015_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._fu015_reset_serial
        self._fu015_reset = self._initial_laptop_reset(episode_seed)
        self._fu015_drive_target_deg = float(self._fu015_reset.hinge_angle_deg)
        self._fu015_failure_latched = False
        self._fu015_failure_reported = False
        self._fu015_bilateral_owner_gate.reset()
        self._fu015_assist_active = False
        self._fu015_assist_substeps = 0
        self._fu015_previous_right_tcp_z_m = float(self._right_tcp_pose[2])
        self._fu015_right_command_delta_z_m = 0.0
        self._fu015_right_opening_intent = False
        self._fu015_right_display_contact_active = False
        self._fu015_hinge_contact_lock_active = True
        self._fu015_hinge_lock_angle_deg = float(
            self._fu015_reset.hinge_angle_deg
        )
        self._fu015_hinge_limit_signature = None
        self._fu015_last_metrics = {}
        self._fu015_last_readback = None
        self._fu015_robot_table_force_by_path_n = {}
        self._fu015_robot_table_points_by_path_m = {}
        self._fu015_raw_contact_diagnostic = {}
        self._fu015_preview_transit_reported = False
        self._fu015_preview_left_role_reported = False
        self._fu015_preview_right_role_reported = False
        self._fu015_preview_terminal_reported = False
        self._fu015_preview_left_transit_peak_force_n = 0.0
        self._fu015_preview_right_transit_peak_force_n = 0.0
        self._fu015_stability_probe_substeps = 0

    def _apply_fu015_randomized_dynamics(self) -> None:
        friction = float(self._fu015_reset.friction)
        material_api = self._fu015_material_api
        if material_api:
            static_attr = material_api.GetStaticFrictionAttr()
            dynamic_attr = material_api.GetDynamicFrictionAttr()
            if not static_attr or not static_attr.IsValid():
                static_attr = material_api.CreateStaticFrictionAttr()
            if not dynamic_attr or not dynamic_attr.IsValid():
                dynamic_attr = material_api.CreateDynamicFrictionAttr()
            static_attr.Set(friction)
            dynamic_attr.Set(friction)
        for label, nominal in (
            ("base", self._fu015_base_nominal_mass),
            ("display", self._fu015_display_nominal_mass),
        ):
            attr = self._fu015_mass_attrs.get(label)
            if attr is not None:
                attr.Set(float(nominal) * float(self._fu015_reset.mass_scale))

    def _set_hinge_contact_lock(
        self,
        locked: bool,
        *,
        hold_angle_deg: float | None = None,
    ) -> None:
        """Brake the hinge at its live angle without writing hinge state."""

        desired_locked = bool(locked)
        desired_hold_deg = self._fu015_hinge_lock_angle_deg
        if desired_locked and hold_angle_deg is not None:
            candidate = float(hold_angle_deg)
            if math.isfinite(candidate):
                desired_hold_deg = float(
                    np.clip(candidate, -HINGE_TRAVEL_DEG, 0.0)
                )
        desired_signature = (desired_locked, desired_hold_deg)
        self._fu015_hinge_contact_lock_active = desired_locked
        self._fu015_hinge_lock_angle_deg = desired_hold_deg
        view = self._fu015_articulation_view
        if view is None or self._torch is None:
            return
        applied = self._fu015_hinge_limit_signature
        if (
            applied is not None
            and applied[0] is desired_locked
            and (
                not desired_locked
                or math.isclose(applied[1], desired_hold_deg, abs_tol=1.0e-6)
            )
        ):
            return
        limits = view.get_dof_limits().clone()
        if limits.ndim != 3 or int(limits.shape[1]) < 1:
            raise RuntimeError("FU015 hinge articulation exposes no DOF limits")
        if desired_locked:
            hold_angle_rad = math.radians(desired_hold_deg)
            limits[:, 0, 0] = hold_angle_rad
            limits[:, 0, 1] = hold_angle_rad
        else:
            limits[:, 0, 0] = math.radians(-HINGE_TRAVEL_DEG)
            limits[:, 0, 1] = 0.0
        indices = self._torch.arange(
            int(view.count), dtype=self._torch.long, device="cpu"
        )
        try:
            view.set_dof_limits(limits.cpu(), indices)
        except Exception as exc:
            raise RuntimeError(
                "FU015 could not update the GPU PhysX contact-gated hinge limit"
            ) from exc
        self._fu015_hinge_limit_signature = desired_signature
        if desired_locked and not math.isclose(
            self._fu015_drive_target_deg,
            desired_hold_deg,
            abs_tol=1.0e-6,
        ):
            # Remove stored spring energy at contact loss so the next contact
            # resumes from the held angle instead of releasing an old target.
            self._set_hinge_drive_target(desired_hold_deg)

    def _set_hinge_drive_target(self, target_deg: float) -> None:
        self._fu015_drive_target_deg = float(
            np.clip(target_deg, -HINGE_TRAVEL_DEG, 0.0)
        )
        view = self._fu015_articulation_view
        if view is None or self._torch is None:
            return
        # USD DriveAPI attributes may only be authored before the first PhysX
        # reset. Mutating targetPosition afterwards is illegal when the Direct
        # GPU API is enabled. The tensor target is the supported runtime path
        # and does not teleport the hinge state.
        targets = view.get_dof_positions().clone()
        if targets.ndim != 2 or int(targets.shape[1]) < 1:
            raise RuntimeError(
                "FU015 hinge articulation exposes no writable DOF target"
            )
        targets[:, 0] = math.radians(self._fu015_drive_target_deg)
        indices = self._torch.arange(
            int(view.count), dtype=self._torch.long, device=self._isaac_device
        )
        try:
            view.set_dof_position_targets(targets, indices)
        except Exception as exc:
            raise RuntimeError(
                "FU015 could not write the GPU PhysX hinge drive target"
            ) from exc

    def _apply_task_pose_randomization(self) -> None:
        if self._torch is None:
            return
        entry = next(
            (
                value
                for value in self._task_reset_articulation_states
                if value[0] == _BASE_FIXED_JOINT_PATH
            ),
            None,
        )
        if entry is None:
            self._add_task_state_warning("FU015 articulation reset state is unavailable")
            return
        path, view, indices, root_pose, root_velocity, joint_pos, joint_vel = entry
        pose_np = root_pose.detach().cpu().numpy().copy()
        pose_np[0] = _transform_composed_root_pose(
            pose_np[0], self._fu015_composed_reset, self._fu015_reset
        )
        pose_tensor = self._torch.as_tensor(
            pose_np, dtype=root_pose.dtype, device=root_pose.device
        )
        positions = joint_pos.clone()
        velocities = self._torch.zeros_like(joint_vel)
        positions[:, 0] = math.radians(float(self._fu015_reset.hinge_angle_deg))
        zero_root_velocity = self._torch.zeros_like(root_velocity)
        view.set_root_transforms(pose_tensor, indices)
        view.set_root_velocities(zero_root_velocity, indices)
        view.set_dof_positions(positions, indices)
        view.set_dof_velocities(velocities, indices)
        self._record_authored_task_reset_write(
            path,
            transform_xyzw=pose_tensor,
            velocity=zero_root_velocity,
            dof_position=positions,
            dof_velocity=velocities,
        )
        self._fu015_reset_base_position = pose_np[0, :3].copy()
        self._fu015_reset_base_pose_wxyz = np.concatenate(
            [pose_np[0, :3], pose_np[0, 6:7], pose_np[0, 3:6]]
        )
        self._apply_fu015_randomized_dynamics()
        self._set_hinge_drive_target(float(self._fu015_reset.hinge_angle_deg))
        print(
            "[HandsOn/FU015] Seeded reset: "
            f"seed={self._fu015_reset.seed}, "
            f"xy=({self._fu015_reset.root_x_m:+.4f},"
            f"{self._fu015_reset.root_y_m:+.4f})m, "
            f"yaw={self._fu015_reset.root_yaw_deg:+.2f}deg, "
            f"hinge={self._fu015_reset.hinge_angle_deg:+.2f}deg, "
            f"target={self._fu015_reset.target_open_deg:.0f}deg, "
            f"friction={self._fu015_reset.friction:.3f}, "
            f"mass_scale={self._fu015_reset.mass_scale:.3f}",
            flush=True,
        )

    @staticmethod
    def _view_numpy(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(shape)

    def _fu015_body_pose(self, view: Any | None, fallback: np.ndarray) -> np.ndarray:
        if view is None:
            return np.asarray(fallback, dtype=np.float64).reshape(7).copy()
        try:
            transform = self._view_numpy(view.get_transforms()[0], (7,))
            return np.concatenate([transform[:3], transform[6:7], transform[3:6]])
        except Exception:
            return np.asarray(fallback, dtype=np.float64).reshape(7).copy()

    def _fu015_body_speeds(self, view: Any | None) -> tuple[float, float]:
        velocity = self._fu015_body_velocity(view)
        return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))

    def _fu015_body_velocity(self, view: Any | None) -> np.ndarray:
        if view is None:
            return np.zeros(6, dtype=np.float64)
        try:
            return self._view_numpy(view.get_velocities()[0], (6,))
        except Exception:
            return np.full(6, math.inf, dtype=np.float64)

    def _fu015_hinge_angle(self) -> float:
        view = self._fu015_articulation_view
        if view is None:
            return math.radians(float(self._fu015_reset.hinge_angle_deg))
        try:
            positions = self._view_numpy(view.get_dof_positions(), (int(view.count), -1))
            return float(positions[0, 0])
        except Exception:
            return math.nan

    def _fu015_hinge_target_angle(self) -> float:
        view = self._fu015_articulation_view
        if view is None:
            return math.radians(float(self._fu015_drive_target_deg))
        try:
            targets = self._view_numpy(
                view.get_dof_position_targets(), (int(view.count), -1)
            )
            return float(targets[0, 0])
        except Exception:
            return math.nan

    def _fu015_hinge_velocity(self) -> float:
        view = self._fu015_articulation_view
        if view is None:
            return 0.0
        try:
            velocities = self._view_numpy(
                view.get_dof_velocities(), (int(view.count), -1)
            )
            return float(velocities[0, 0])
        except Exception:
            return math.nan

    def _world_aabb(self, path: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self._stage is None:
            return None
        try:
            from pxr import Usd, UsdGeom

            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return None
            bbox = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            ).ComputeWorldBound(prim)
            box_range = bbox.ComputeAlignedRange()
            return (
                np.asarray(box_range.GetMin(), dtype=np.float64),
                np.asarray(box_range.GetMax(), dtype=np.float64),
            )
        except Exception:
            return None

    def _contact_samples_with_filter_indices(
        self,
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        view = self._fu015_contact_views.get(key)
        if view is None:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )
        try:
            normal_forces, points, _normals, _separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )
            normal_forces = self._view_numpy(normal_forces, (-1,))
            points = self._view_numpy(points, (-1, 3))
            counts = self._view_numpy(counts, (int(view.sensor_count), int(view.filter_count))).astype(np.int64)
            starts = self._view_numpy(starts, counts.shape).astype(np.int64)
            selected_points: list[np.ndarray] = []
            selected_forces: list[np.ndarray] = []
            selected_filter_indices: list[np.ndarray] = []
            filter_count = int(view.filter_count)
            for flat_index, (count, start) in enumerate(
                zip(counts.ravel(), starts.ravel())
            ):
                if count <= 0:
                    continue
                stop = int(start + count)
                selected_points.append(points[int(start):stop])
                selected_forces.append(np.abs(normal_forces[int(start):stop]))
                selected_filter_indices.append(
                    np.full(int(count), flat_index % filter_count, dtype=np.int64)
                )
            if not selected_points:
                return (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.int64),
                )
            return (
                np.concatenate(selected_points),
                np.concatenate(selected_forces),
                np.concatenate(selected_filter_indices),
            )
        except Exception as exc:
            self._add_task_state_warning(f"FU015 contact readback {key} failed: {exc}")
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )

    def _contact_samples(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        points, forces, _filter_indices = self._contact_samples_with_filter_indices(
            key
        )
        return points, forces

    @staticmethod
    def _force_in_mask(forces: np.ndarray, mask: np.ndarray) -> float:
        if forces.size == 0 or mask.size == 0:
            return 0.0
        return float(np.sum(forces[np.asarray(mask, dtype=bool)]))

    def _fu015_hand_body_positions_m(self, side: str) -> dict[str, list[float]]:
        """Return compact physical hand-link positions for failure diagnosis."""

        robot = self._robots.get(side)
        if robot is None:
            return {}
        try:
            body_positions = self._view_numpy(
                robot.data.body_pose_w[0, :, :3],
                (len(robot.body_names), 3),
            )
        except Exception:
            return {}
        result: dict[str, list[float]] = {}
        for index, name in enumerate(robot.body_names):
            normalized_name = str(name).lower()
            if normalized_name == f"{side}_palm_link" or (
                normalized_name.startswith(f"{side}_finger")
                and normalized_name.endswith(("_link3", "_link4"))
            ):
                result[str(name)] = [
                    float(value) for value in body_positions[index]
                ]
        return result

    def _fu015_hand_joint_diagnostic(self, side: str) -> dict[str, Any]:
        """Expose physical hand tracking and joint-limit margins."""

        robot = self._robots.get(side)
        joint_ids = list(self._hand_joint_ids.get(side, ()))
        if robot is None or not joint_ids:
            return {}
        try:
            positions = self._view_numpy(
                robot.data.joint_pos[0, joint_ids],
                (len(joint_ids),),
            )
            target_tensor = self._joint_targets.get(side)
            targets = (
                positions.copy()
                if target_tensor is None
                else self._view_numpy(
                    target_tensor[0, joint_ids],
                    (len(joint_ids),),
                )
            )
            limits = self._view_numpy(
                robot.data.soft_joint_pos_limits[0, joint_ids, :],
                (len(joint_ids), 2),
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "joint_names": [str(robot.joint_names[index]) for index in joint_ids],
            "joint_positions_rad": positions.tolist(),
            "joint_targets_rad": targets.tolist(),
            "joint_lower_margins_rad": (positions - limits[:, 0]).tolist(),
            "joint_upper_margins_rad": (limits[:, 1] - positions).tolist(),
        }

    def _fu015_tcp_pose_diagnostic(self, side: str) -> dict[str, Any]:
        """Compare the commanded TCP with the physical palm pose."""

        target = self._right_tcp_pose if side == "right" else self._left_tcp_pose
        try:
            actual = self._native_end_effector_pose_matrix(side, target)
        except Exception:
            return {"target_pose_wxyz_m": np.asarray(target).tolist()}
        target_position = np.asarray(target[:3], dtype=np.float64)
        actual_position = np.asarray(actual[:3, 3], dtype=np.float64)
        return {
            "target_pose_wxyz_m": np.asarray(target).tolist(),
            "actual_pose_matrix_m": np.asarray(actual).tolist(),
            "position_error_m": float(
                np.linalg.norm(actual_position - target_position)
            ),
        }

    def _fu015_arm_joint_diagnostic(self, side: str) -> dict[str, Any]:
        """Expose joint-limit and Jacobian evidence for DLS reach failures."""

        robot = self._robots.get(side)
        arm_ids = list(self._arm_joint_ids.get(side, ()))
        if robot is None or len(arm_ids) != 7:
            return {}
        try:
            positions = self._view_numpy(
                robot.data.joint_pos[0, arm_ids],
                (len(arm_ids),),
            )
            target_tensor = self._joint_targets.get(side)
            targets = (
                positions.copy()
                if target_tensor is None
                else self._view_numpy(
                    target_tensor[0, arm_ids],
                    (len(arm_ids),),
                )
            )
            limits = self._view_numpy(
                robot.data.soft_joint_pos_limits[0, arm_ids, :],
                (len(arm_ids), 2),
            )
            jacobian = self._view_numpy(
                robot.root_physx_view.get_jacobians()[
                    0,
                    self._ee_jacobi_indices[side],
                    :,
                    arm_ids,
                ],
                (6, len(arm_ids)),
            )
            singular_values = np.linalg.svd(
                jacobian,
                compute_uv=False,
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        lower_margin = positions - limits[:, 0]
        upper_margin = limits[:, 1] - positions
        names = [str(robot.joint_names[index]) for index in arm_ids]
        smallest = float(np.min(singular_values))
        largest = float(np.max(singular_values))
        return {
            "joint_names": names,
            "joint_positions_deg": np.rad2deg(positions).tolist(),
            "joint_targets_deg": np.rad2deg(targets).tolist(),
            "joint_lower_margins_deg": np.rad2deg(lower_margin).tolist(),
            "joint_upper_margins_deg": np.rad2deg(upper_margin).tolist(),
            "jacobian_singular_values": singular_values.tolist(),
            "jacobian_condition_number": (
                math.inf if smallest <= 1.0e-12 else largest / smallest
            ),
        }

    def _fu015_contact_forces(self) -> tuple[float, float, float, float, float]:
        left_points, left_forces, left_filter_indices = (
            self._contact_samples_with_filter_indices("left_base")
        )
        left_display_points, left_display_forces, left_display_filter_indices = (
            self._contact_samples_with_filter_indices("left_display")
        )
        left_paths = self._fu015_contact_paths.get("left_base", ("", ()))[1]
        left_display_paths = self._fu015_contact_paths.get(
            "left_display", ("", ())
        )[1]
        right_base_points, right_base_forces = self._contact_samples("right_base")
        display_points, display_forces, display_filter_indices = (
            self._contact_samples_with_filter_indices("right_display")
        )
        right_display_paths = self._fu015_contact_paths.get(
            "right_display", ("", ())
        )[1]
        table_points, table_forces, table_filter_indices = (
            self._contact_samples_with_filter_indices("robot_table")
        )
        robot_table_paths = self._fu015_contact_paths.get(
            "robot_table", ("", ())
        )[1]
        self._fu015_robot_table_force_by_path_n = summarize_contact_forces_by_filter(
            robot_table_paths,
            table_filter_indices,
            table_forces,
        )
        self._fu015_robot_table_points_by_path_m = summarize_contact_points_by_filter(
            robot_table_paths,
            table_filter_indices,
            table_points,
            table_forces,
        )
        display_aabb = self._world_aabb(DISPLAY_BODY_PATH)
        touchpad_aabb = self._world_aabb(TOUCHPAD_COLLIDER_PATH)
        self._fu015_raw_contact_diagnostic = {
            "left_base_force_n": float(np.sum(left_forces)),
            "left_base_points_m": left_points.tolist(),
            "left_base_force_by_path_n": summarize_contact_forces_by_filter(
                left_paths, left_filter_indices, left_forces
            ),
            "left_base_points_by_path_m": summarize_contact_points_by_filter(
                left_paths, left_filter_indices, left_points, left_forces
            ),
            "left_base_manifold_points_by_path_m": (
                summarize_contact_manifold_points_by_filter(
                    left_paths, left_filter_indices, left_points
                )
            ),
            "left_display_force_n": float(np.sum(left_display_forces)),
            "left_display_points_m": left_display_points.tolist(),
            "left_display_force_by_path_n": summarize_contact_forces_by_filter(
                left_display_paths,
                left_display_filter_indices,
                left_display_forces,
            ),
            "left_display_points_by_path_m": summarize_contact_points_by_filter(
                left_display_paths,
                left_display_filter_indices,
                left_display_points,
                left_display_forces,
            ),
            "left_display_manifold_points_by_path_m": (
                summarize_contact_manifold_points_by_filter(
                    left_display_paths,
                    left_display_filter_indices,
                    left_display_points,
                )
            ),
            "right_base_force_n": float(np.sum(right_base_forces)),
            "right_base_points_m": right_base_points.tolist(),
            "right_display_force_n": float(np.sum(display_forces)),
            "right_display_points_m": display_points.tolist(),
            "right_display_force_by_path_n": summarize_contact_forces_by_filter(
                right_display_paths,
                display_filter_indices,
                display_forces,
            ),
            "right_display_points_by_path_m": summarize_contact_points_by_filter(
                right_display_paths,
                display_filter_indices,
                display_points,
                display_forces,
            ),
            "right_display_manifold_points_by_path_m": (
                summarize_contact_manifold_points_by_filter(
                    right_display_paths,
                    display_filter_indices,
                    display_points,
                )
            ),
        }
        if display_aabb is None or touchpad_aabb is None:
            return (
                0.0,
                0.0,
                math.inf,
                float(np.sum(right_base_forces)),
                float(np.sum(table_forces)),
            )

        touch_min, touch_max = touchpad_aabb
        display_min, display_max = display_aabb
        base_pose = self._fu015_body_pose(
            self._fu015_base_view,
            self._fu015_reset_base_pose_wxyz,
        )
        left_front_mask = points_in_body_box(
            left_points,
            base_pose,
            _BASE_FRONT_PROXY_LOWER_M,
            _BASE_FRONT_PROXY_UPPER_M,
            margin_m=_FRONT_PROXY_POINT_MARGIN_M,
        )
        left_touchpad_mask = (
            np.all(left_points >= touch_min - _TOUCHPAD_POINT_MARGIN_M, axis=1)
            & np.all(left_points <= touch_max + _TOUCHPAD_POINT_MARGIN_M, axis=1)
            if left_points.size
            else np.zeros(0, dtype=bool)
        )
        left_role_mask = left_front_mask & ~left_touchpad_mask
        # At every hinge angle the free display edge is the highest part of
        # the dynamic lid.  This remains a contact-point test, not proximity.
        right_top_mask = (
            display_points[:, 2] >= display_max[2] - _DISPLAY_TOP_EDGE_BAND_M
            if display_points.size
            else np.zeros(0, dtype=bool)
        )
        # Also reject contacts below the lower third of the display envelope.
        if display_points.size:
            right_top_mask &= display_points[:, 2] >= (
                display_min[2] + 0.60 * (display_max[2] - display_min[2])
            )
        return (
            self._force_in_mask(left_forces, left_role_mask),
            self._force_in_mask(display_forces, right_top_mask),
            self._force_in_mask(left_forces, left_touchpad_mask),
            float(np.sum(right_base_forces)),
            float(np.sum(table_forces)),
        )

    def _physical_readback(self) -> FU015PhysicalReadback:
        base_fallback = np.concatenate(
            [self._fu015_reset_base_position, np.asarray([1.0, 0.0, 0.0, 0.0])]
        )
        display_fallback = base_fallback.copy()
        base_pose = self._fu015_body_pose(self._fu015_base_view, base_fallback)
        _display_pose = self._fu015_body_pose(self._fu015_display_view, display_fallback)
        base_linear, base_angular = self._fu015_body_speeds(self._fu015_base_view)
        display_linear, display_angular = self._fu015_body_speeds(self._fu015_display_view)
        contacts = self._fu015_contact_forces()
        return FU015PhysicalReadback(
            hinge_angle_rad=self._fu015_hinge_angle(),
            base_position_m=tuple(float(value) for value in base_pose[:3]),
            reset_base_position_m=tuple(
                float(value) for value in self._fu015_reset_base_position
            ),
            base_linear_speed_mps=base_linear,
            base_angular_speed_radps=base_angular,
            display_linear_speed_mps=display_linear,
            display_angular_speed_radps=display_angular,
            left_base_contact_force_n=contacts[0],
            right_display_contact_force_n=contacts[1],
            left_touchpad_contact_force_n=contacts[2],
            right_base_contact_force_n=contacts[3],
            robot_table_contact_force_n=contacts[4],
            left_any_base_contact_force_n=float(
                self._fu015_raw_contact_diagnostic.get(
                    "left_base_force_n", contacts[0]
                )
            ),
        )

    def _refresh_fu015_metrics(self) -> dict[str, float | bool]:
        self._fu015_last_readback = self._physical_readback()
        self._fu015_last_metrics = evaluate_physical_readback(
            self._fu015_last_readback,
            target_open_deg=self._fu015_reset.target_open_deg,
        )
        return self._fu015_last_metrics

    def _latch_fu015_failure(
        self,
        metrics: Mapping[str, float | bool],
    ) -> bool:
        """Persist failures at each physical readback where they are observed."""

        was_latched = bool(self._fu015_failure_latched)
        self._fu015_failure_latched = bool(
            self._fu015_failure_latched or metrics["failure"]
        )
        if (
            self._fu015_failure_latched
            and not was_latched
            and not self._fu015_failure_reported
        ):
            self._fu015_failure_reported = True
            diagnostic = {
                **metrics,
                "physical_readback": (
                    asdict(self._fu015_last_readback)
                    if self._fu015_last_readback is not None
                    else None
                ),
                "robot_table_contact_force_by_path_n": (
                    self._fu015_robot_table_force_by_path_n
                ),
                "robot_table_contact_points_by_path_m": (
                    self._fu015_robot_table_points_by_path_m
                ),
                "raw_contact_diagnostic": self._fu015_raw_contact_diagnostic,
                "left_hand_body_positions_m": (
                    self._fu015_hand_body_positions_m("left")
                ),
                "left_hand_joint_diagnostic": (
                    self._fu015_hand_joint_diagnostic("left")
                ),
                "left_arm_joint_diagnostic": (
                    self._fu015_arm_joint_diagnostic("left")
                ),
                "right_hand_body_positions_m": (
                    self._fu015_hand_body_positions_m("right")
                ),
                "left_tcp_diagnostic": self._fu015_tcp_pose_diagnostic("left"),
                "right_tcp_diagnostic": self._fu015_tcp_pose_diagnostic("right"),
                "left_command_tcp_pose": self._left_tcp_pose.tolist(),
                "right_command_tcp_pose": self._right_tcp_pose.tolist(),
                "base_aabb_m": (
                    [bound.tolist() for bound in base_aabb]
                    if (base_aabb := self._world_aabb(BASE_BODY_PATH)) is not None
                    else None
                ),
                "display_aabb_m": (
                    [bound.tolist() for bound in display_aabb]
                    if (
                        display_aabb := self._world_aabb(DISPLAY_BODY_PATH)
                    ) is not None
                    else None
                ),
                "touchpad_aabb_m": (
                    [bound.tolist() for bound in touchpad_aabb]
                    if (
                        touchpad_aabb := self._world_aabb(
                            TOUCHPAD_COLLIDER_PATH
                        )
                    ) is not None
                    else None
                ),
            }
            print(
                f"[HandsOn/FU015] Failure latched at action step {self.env_step}: "
                f"{diagnostic}",
                flush=True,
            )
        return self._fu015_failure_latched

    def _before_task_physics_substep(self) -> None:
        control_substeps = max(1, int(self._n_substeps))
        if self._fu015_stability_probe_substeps % control_substeps == 0:
            current_right_tcp_z_m = float(self._right_tcp_pose[2])
            previous_right_tcp_z_m = self._fu015_previous_right_tcp_z_m
            self._fu015_right_opening_intent = _right_tcp_opening_intent(
                previous_right_tcp_z_m,
                current_right_tcp_z_m,
            )
            self._fu015_right_command_delta_z_m = (
                0.0
                if previous_right_tcp_z_m is None
                or not math.isfinite(float(previous_right_tcp_z_m))
                or not math.isfinite(current_right_tcp_z_m)
                else current_right_tcp_z_m - float(previous_right_tcp_z_m)
            )
            if math.isfinite(current_right_tcp_z_m):
                self._fu015_previous_right_tcp_z_m = current_right_tcp_z_m

        self._fu015_right_display_contact_active = bool(
            self._fu015_last_metrics.get("right_display_contact", False)
        )
        hinge_angle_deg = float(np.rad2deg(self._fu015_hinge_angle()))
        motion_allowed = _hinge_motion_allowed(
            right_display_contact=self._fu015_right_display_contact_active,
            hinge_angle_deg=hinge_angle_deg,
            opening_intent=self._fu015_right_opening_intent,
        )
        if motion_allowed:
            self._set_hinge_contact_lock(False)
        elif self._fu015_hinge_contact_lock_active:
            self._set_hinge_contact_lock(
                True,
                hold_angle_deg=self._fu015_hinge_lock_angle_deg,
            )
        else:
            # This is a dynamic joint brake: the current angle becomes equal
            # lower/upper PhysX limits.  No pose or velocity is teleported.
            self._set_hinge_contact_lock(True, hold_angle_deg=hinge_angle_deg)

    def _after_task_physics_substep(self) -> None:
        self._fu015_stability_probe_substeps += 1
        # All five role-contact views synchronize PhysX tensor data back to
        # the CPU. Reading them after every 60 Hz substep, then reading them
        # twice again in success/failure, made the nominal 30 Hz control loop
        # run at roughly 10-18 Hz. One final-substep snapshot is sufficient:
        # it is the state recorded for this control transition and is reused
        # by failure, success, task-info and observation consumers below.
        control_substeps = max(1, int(self._n_substeps))
        if self._fu015_stability_probe_substeps % control_substeps:
            return
        metrics = self._refresh_fu015_metrics()
        if self._fu015_stability_probe_substeps % max(1, PHYSICS_HZ * 5) == 0:
            fallback = self._fu015_reset_base_pose_wxyz
            readback = self._fu015_last_readback
            print(
                "[HandsOn/FU015] "
                + json.dumps(
                    {
                        "event": "physical_substep_probe",
                        "reset_seed": int(self._fu015_reset.seed),
                        "env_step": int(self.env_step),
                        "physics_substep": int(self._fu015_stability_probe_substeps),
                        "base_reset_pose_wxyz_m": fallback.tolist(),
                        "base_pose_wxyz_m": self._fu015_body_pose(
                            self._fu015_base_view, fallback
                        ).tolist(),
                        "base_velocity_mps_radps": self._fu015_body_velocity(
                            self._fu015_base_view
                        ).tolist(),
                        "display_pose_wxyz_m": self._fu015_body_pose(
                            self._fu015_display_view, fallback
                        ).tolist(),
                        "display_velocity_mps_radps": self._fu015_body_velocity(
                            self._fu015_display_view
                        ).tolist(),
                        "hinge_angle_deg": float(
                            np.rad2deg(self._fu015_hinge_angle())
                        ),
                        "hinge_target_readback_deg": float(
                            np.rad2deg(self._fu015_hinge_target_angle())
                        ),
                        "hinge_velocity_radps": self._fu015_hinge_velocity(),
                        "right_tcp_command_delta_z_m": float(
                            self._fu015_right_command_delta_z_m
                        ),
                        "right_opening_intent": bool(
                            self._fu015_right_opening_intent
                        ),
                        "right_display_contact_gate_active": bool(
                            self._fu015_right_display_contact_active
                        ),
                        "hinge_contact_lock_active": bool(
                            self._fu015_hinge_contact_lock_active
                        ),
                        "hinge_lock_angle_deg": float(
                            self._fu015_hinge_lock_angle_deg
                        ),
                        "left_base_contact_force_n": (
                            float(readback.left_base_contact_force_n)
                            if readback is not None
                            else 0.0
                        ),
                        "right_display_contact_force_n": (
                            float(readback.right_display_contact_force_n)
                            if readback is not None
                            else 0.0
                        ),
                        "right_base_contact_force_n": (
                            float(readback.right_base_contact_force_n)
                            if readback is not None
                            else 0.0
                        ),
                        "raw_right_display_force_n": float(
                            self._fu015_raw_contact_diagnostic.get(
                                "right_display_force_n", 0.0
                            )
                        ),
                        "raw_left_display_force_n": float(
                            self._fu015_raw_contact_diagnostic.get(
                                "left_display_force_n", 0.0
                            )
                        ),
                        "raw_left_display_points_m": (
                            self._fu015_raw_contact_diagnostic.get(
                                "left_display_points_m", []
                            )
                        ),
                        "raw_left_base_force_by_path_n": (
                            self._fu015_raw_contact_diagnostic.get(
                                "left_base_force_by_path_n", {}
                            )
                        ),
                        "raw_left_display_force_by_path_n": (
                            self._fu015_raw_contact_diagnostic.get(
                                "left_display_force_by_path_n", {}
                            )
                        ),
                        "raw_left_display_points_by_path_m": (
                            self._fu015_raw_contact_diagnostic.get(
                                "left_display_points_by_path_m", {}
                            )
                        ),
                        "raw_left_base_manifold_points_by_path_m": (
                            self._fu015_raw_contact_diagnostic.get(
                                "left_base_manifold_points_by_path_m", {}
                            )
                        ),
                        "raw_left_display_manifold_points_by_path_m": (
                            self._fu015_raw_contact_diagnostic.get(
                                "left_display_manifold_points_by_path_m", {}
                            )
                        ),
                        "raw_right_display_points_m": (
                            self._fu015_raw_contact_diagnostic.get(
                                "right_display_points_m", []
                            )
                        ),
                        "raw_right_display_force_by_path_n": (
                            self._fu015_raw_contact_diagnostic.get(
                                "right_display_force_by_path_n", {}
                            )
                        ),
                        "raw_right_display_manifold_points_by_path_m": (
                            self._fu015_raw_contact_diagnostic.get(
                                "right_display_manifold_points_by_path_m", {}
                            )
                        ),
                        "left_hand_body_positions_m": (
                            self._fu015_hand_body_positions_m("left")
                        ),
                        "right_hand_body_positions_m": (
                            self._fu015_hand_body_positions_m("right")
                        ),
                        "left_tcp_diagnostic": (
                            self._fu015_tcp_pose_diagnostic("left")
                        ),
                        "right_tcp_diagnostic": (
                            self._fu015_tcp_pose_diagnostic("right")
                        ),
                        "right_arm_joint_diagnostic": (
                            self._fu015_arm_joint_diagnostic("right")
                        ),
                        "left_arm_joint_diagnostic": (
                            self._fu015_arm_joint_diagnostic("left")
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        self._latch_fu015_failure(metrics)
        owner_sample = bool(
            not self._fu015_failure_latched
            and metrics["right_display_contact"]
            and metrics["base_stable"]
        )
        owner_ready = False
        for _ in range(control_substeps):
            owner_ready = self._fu015_bilateral_owner_gate.update(owner_sample)
        # Ownership/contact remains the success gate.  Opening intent is a
        # separate control gate: a hand pressing down on the closed lid can
        # own the contact, but must never wind up the opening spring.
        self._fu015_assist_active = bool(
            owner_ready and self._fu015_right_opening_intent
        )
        assist_increment_deg = 0.0
        if self._fu015_assist_active:
            self._fu015_assist_substeps += control_substeps
            assist_increment_deg = (
                _ASSIST_DEGREES_PER_PHYSICS_STEP * control_substeps
            )
        next_target_deg = _next_hinge_drive_target_deg(
            current_angle_deg=float(np.rad2deg(self._fu015_hinge_angle())),
            drive_target_deg=self._fu015_drive_target_deg,
            task_target_deg=float(self._fu015_reset.target_open_deg),
            assist_increment_deg=assist_increment_deg,
        )
        if not math.isclose(
            next_target_deg,
            self._fu015_drive_target_deg,
            abs_tol=1e-6,
        ):
            self._set_hinge_drive_target(next_target_deg)

    def _compute_success_now(self) -> bool:
        metrics = self._fu015_last_metrics or self._refresh_fu015_metrics()
        return bool(
            metrics["success_now"]
            and self._fu015_bilateral_owner_gate.ready
            and not self._fu015_failure_latched
        )

    def _compute_failure(self) -> bool:
        metrics = self._fu015_last_metrics or self._refresh_fu015_metrics()
        return self._latch_fu015_failure(metrics)

    def _success_steps_required(self) -> int:
        return SUCCESS_HOLD_STEPS

    def _emit_fu015_preview_acceptance_marker(
        self,
        metrics: Mapping[str, float | bool],
    ) -> None:
        """Emit machine-readable, task-local evidence for bounded previews."""

        filter_counts = {
            "left_hand": len(
                self._fu015_contact_paths.get("left_base", ("", ()))[1]
            ),
            "right_hand": len(
                self._fu015_contact_paths.get("right_display", ("", ()))[1]
            ),
            "robot_table": len(
                self._fu015_contact_paths.get("robot_table", ("", ()))[1]
            ),
        }
        left_table_force = float(
            sum(
                force
                for path, force in self._fu015_robot_table_force_by_path_n.items()
                if "/LeftRobot/" in path
            )
        )
        right_table_force = float(
            sum(
                force
                for path, force in self._fu015_robot_table_force_by_path_n.items()
                if "/RightRobot/" in path
            )
        )
        if self.env_step <= 45:
            self._fu015_preview_left_transit_peak_force_n = max(
                self._fu015_preview_left_transit_peak_force_n,
                left_table_force,
            )
            self._fu015_preview_right_transit_peak_force_n = max(
                self._fu015_preview_right_transit_peak_force_n,
                right_table_force,
            )

        payload: dict[str, Any] | None = None
        if self.env_step >= 45 and not self._fu015_preview_transit_reported:
            self._fu015_preview_transit_reported = True
            payload = {
                "event": "transit_clearance",
                "task_id": "B320-FU015",
                "env_step": int(self.env_step),
                "filter_counts": filter_counts,
                "left_robot_table_peak_force_n": (
                    self._fu015_preview_left_transit_peak_force_n
                ),
                "right_robot_table_peak_force_n": (
                    self._fu015_preview_right_transit_peak_force_n
                ),
                "robot_table_clear": bool(metrics["robot_table_clear"]),
            }

        if self.env_step >= 150 and not self._fu015_preview_left_role_reported:
            self._fu015_preview_left_role_reported = True
            readback = self._fu015_last_readback
            pose_fallback = np.concatenate(
                [
                    self._fu015_reset_base_position,
                    np.asarray([1.0, 0.0, 0.0, 0.0]),
                ]
            )
            payload = {
                "event": "left_role_checkpoint",
                "task_id": "B320-FU015",
                "env_step": int(self.env_step),
                "left_base_contact": bool(metrics["left_base_contact"]),
                "left_base_contact_force_n": (
                    float(readback.left_base_contact_force_n)
                    if readback is not None
                    else 0.0
                ),
                "hinge_angle_deg": float(np.rad2deg(self._fu015_hinge_angle())),
                "hinge_drive_target_deg": float(self._fu015_drive_target_deg),
                "robot_table_clear": bool(metrics["robot_table_clear"]),
                "left_hand_body_positions_m": (
                    self._fu015_hand_body_positions_m("left")
                ),
                "left_hand_joint_diagnostic": (
                    self._fu015_hand_joint_diagnostic("left")
                ),
                "left_arm_joint_diagnostic": (
                    self._fu015_arm_joint_diagnostic("left")
                ),
                "left_command_tcp_pose": self._left_tcp_pose.tolist(),
                "right_hand_body_positions_m": (
                    self._fu015_hand_body_positions_m("right")
                ),
                "right_command_tcp_pose": self._right_tcp_pose.tolist(),
                "base_body_pose_wxyz_m": self._fu015_body_pose(
                    self._fu015_base_view, pose_fallback
                ).tolist(),
                "display_body_pose_wxyz_m": self._fu015_body_pose(
                    self._fu015_display_view, pose_fallback
                ).tolist(),
                "raw_left_base_force_n": float(
                    self._fu015_raw_contact_diagnostic.get(
                        "left_base_force_n", 0.0
                    )
                ),
                "raw_left_base_points_m": self._fu015_raw_contact_diagnostic.get(
                    "left_base_points_m", []
                ),
            }

        if self.env_step >= 195 and not self._fu015_preview_right_role_reported:
            self._fu015_preview_right_role_reported = True
            readback = self._fu015_last_readback
            pose_fallback = np.concatenate(
                [
                    self._fu015_reset_base_position,
                    np.asarray([1.0, 0.0, 0.0, 0.0]),
                ]
            )
            payload = {
                "event": "right_role_checkpoint",
                "task_id": "B320-FU015",
                "env_step": int(self.env_step),
                "left_base_contact": bool(metrics["left_base_contact"]),
                "right_display_contact": bool(metrics["right_display_contact"]),
                "right_display_contact_force_n": (
                    float(readback.right_display_contact_force_n)
                    if readback is not None
                    else 0.0
                ),
                "hinge_angle_deg": float(np.rad2deg(self._fu015_hinge_angle())),
                "hinge_drive_target_deg": float(self._fu015_drive_target_deg),
                "right_base_clear": bool(metrics["right_base_clear"]),
                "robot_table_clear": bool(metrics["robot_table_clear"]),
                "right_hand_body_positions_m": (
                    self._fu015_hand_body_positions_m("right")
                ),
                "right_command_tcp_pose": self._right_tcp_pose.tolist(),
                "base_body_pose_wxyz_m": self._fu015_body_pose(
                    self._fu015_base_view, pose_fallback
                ).tolist(),
                "display_body_pose_wxyz_m": self._fu015_body_pose(
                    self._fu015_display_view, pose_fallback
                ).tolist(),
                "raw_right_base_force_n": float(
                    self._fu015_raw_contact_diagnostic.get(
                        "right_base_force_n", 0.0
                    )
                ),
                "raw_right_display_force_n": float(
                    self._fu015_raw_contact_diagnostic.get(
                        "right_display_force_n", 0.0
                    )
                ),
                "raw_right_display_points_m": (
                    self._fu015_raw_contact_diagnostic.get(
                        "right_display_points_m", []
                    )
                ),
            }

        succeed = self._success_counter >= SUCCESS_HOLD_STEPS
        failed = bool(self._fu015_failure_latched)
        if (succeed or failed) and not self._fu015_preview_terminal_reported:
            self._fu015_preview_terminal_reported = True
            payload = {
                "event": "terminal",
                "task_id": "B320-FU015",
                "env_step": int(self.env_step),
                "filter_counts": filter_counts,
                "terminated": True,
                "succeed": bool(succeed and not failed),
                "failed": failed,
                "success_counter": int(self._success_counter),
                "success_steps_required": SUCCESS_HOLD_STEPS,
                "left_base_contact": bool(metrics["left_base_contact"]),
                "right_display_contact": bool(metrics["right_display_contact"]),
                "robot_table_clear": bool(metrics["robot_table_clear"]),
            }

        if payload is not None:
            print(
                "[HandsOn/FU015][PREVIEW_ACCEPTANCE] "
                + json.dumps(payload, sort_keys=True, separators=(",", ":")),
                flush=True,
            )

    def _task_info(self) -> dict[str, Any]:
        metrics = self._fu015_last_metrics or self._refresh_fu015_metrics()
        self._emit_fu015_preview_acceptance_marker(metrics)
        return {
            **{f"fu015_{key}": value for key, value in metrics.items()},
            "fu015_success_counter": int(self._success_counter),
            "fu015_success_steps_required": SUCCESS_HOLD_STEPS,
            "fu015_failure_latched": bool(self._fu015_failure_latched),
            "fu015_reset_seed": int(self._fu015_reset.seed),
            "fu015_reset_root_xy_m": [
                float(self._fu015_reset.root_x_m),
                float(self._fu015_reset.root_y_m),
            ],
            "fu015_reset_root_yaw_deg": float(self._fu015_reset.root_yaw_deg),
            "fu015_reset_hinge_deg": float(self._fu015_reset.hinge_angle_deg),
            "fu015_target_open_deg": float(self._fu015_reset.target_open_deg),
            "fu015_reset_friction": float(self._fu015_reset.friction),
            "fu015_reset_mass_scale": float(self._fu015_reset.mass_scale),
            "fu015_right_robot_base_x_offset_m": (
                _RIGHT_ROBOT_BASE_X_OFFSET_M
            ),
            "fu015_left_robot_base_x_offset_m": _LEFT_ROBOT_BASE_X_OFFSET_M,
            "fu015_hinge_drive_stiffness": _HINGE_DRIVE_STIFFNESS,
            "fu015_hinge_drive_max_force": _HINGE_DRIVE_MAX_FORCE,
            "fu015_hinge_drive_target_deg": float(self._fu015_drive_target_deg),
            "fu015_contact_gated_assist_active": bool(self._fu015_assist_active),
            "fu015_contact_gated_assist_substeps": int(self._fu015_assist_substeps),
            "fu015_right_tcp_command_delta_z_m": float(
                self._fu015_right_command_delta_z_m
            ),
            "fu015_right_opening_intent": bool(self._fu015_right_opening_intent),
            "fu015_right_display_contact_gate_active": bool(
                self._fu015_right_display_contact_active
            ),
            "fu015_hinge_contact_lock_active": bool(
                self._fu015_hinge_contact_lock_active
            ),
            "fu015_hinge_lock_angle_deg": float(
                self._fu015_hinge_lock_angle_deg
            ),
            "fu015_bilateral_owner_substeps": int(
                self._fu015_bilateral_owner_gate.consecutive_substeps
            ),
            "fu015_bilateral_owner_ready": bool(
                self._fu015_bilateral_owner_gate.ready
            ),
            "fu015_contact_contract_ready": bool(self._fu015_contact_contract_ready),
            "fu015_contact_view_paths": self._fu015_contact_paths,
            "fu015_robot_table_contact_force_by_path_n": (
                self._fu015_robot_table_force_by_path_n
            ),
            "fu015_kitchen_preview_enabled": bool(
                self._fu015_kitchen_preview_enabled
            ),
            "fu015_kitchen_preview_asset_sha256": (
                _KITCHEN_PREVIEW_ASSET_SHA256
                if self._fu015_kitchen_preview_enabled
                else None
            ),
            "fu015_kitchen_preview_source_url": (
                _KITCHEN_PREVIEW_SOURCE_URL
                if self._fu015_kitchen_preview_enabled
                else None
            ),
            "fu015_kitchen_preview_license": (
                _KITCHEN_PREVIEW_LICENSE
                if self._fu015_kitchen_preview_enabled
                else None
            ),
            "fu015_kitchen_preview_mesh_count": int(
                self._fu015_kitchen_preview_mesh_count
            ),
            "fu015_kitchen_preview_island_bounds_m": dict(
                self._fu015_kitchen_preview_island_bounds_m
            ),
            "fu015_kitchen_preview_collision_mode": (
                "canonical_table_and_floor_only"
                if self._fu015_kitchen_preview_enabled
                else "disabled"
            ),
            "fu015_object_state_writes_after_action_zero": 0,
        }

    def _task_status_text(self) -> str:
        metrics = self._fu015_last_metrics
        return (
            f"屏幕 {float(metrics.get('hinge_deg', 0.0)):.1f}/"
            f"{self._fu015_reset.target_open_deg:.0f}°"
            f"  |  稳定保持 {self._success_counter}/{SUCCESS_HOLD_STEPS}"
            f"  |  底座 已固定"
            f"  右手开盖 {'✓' if metrics.get('right_display_contact') else '—'}"
        )

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._fu015_last_metrics or self._refresh_fu015_metrics()
        rows = [
            self._condition_row(
                "fu015_hinge",
                f"本轮目标：屏幕打开到 {self._fu015_reset.target_open_deg:.0f}°"
                f"（±{TARGET_TOLERANCE_DEG:.0f}°）",
                bool(metrics["hinge_target_ok"]),
            ),
            self._condition_row("fu015_right_display", "右手接触屏幕前缘并开盖", bool(metrics["right_display_contact"])),
            self._condition_row("fu015_base_stable", "随机化后的底座保持固定", bool(metrics["base_stable"])),
            self._condition_row("fu015_velocity_stable", "屏幕到达目标后保持稳定", bool(metrics["velocity_stable"])),
        ]
        return self._append_success_hold_row(rows)

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._fu015_last_metrics or self._refresh_fu015_metrics()
        base_pose = self._fu015_body_pose(
            self._fu015_base_view,
            np.concatenate(
                [self._fu015_reset_base_position, np.asarray([1.0, 0.0, 0.0, 0.0])]
            ),
        )
        display_pose = self._fu015_body_pose(self._fu015_display_view, base_pose)
        state = observation["state"]
        state["laptop_base_pose"] = base_pose
        state["laptop_display_pose"] = display_pose
        state["hinge_angle"] = np.asarray([self._fu015_hinge_angle()], dtype=np.float64)
        state["base_displacement"] = np.asarray(
            [
                float(metrics["base_displacement_m"]),
                float(metrics["base_lift_m"]),
                float(metrics["base_stable"]),
            ],
            dtype=np.float64,
        )
        state["bimanual_role_contacts"] = np.asarray(
            [
                float(metrics["left_base_contact"]),
                float(metrics["right_display_contact"]),
                float(not metrics["left_touchpad_clear"]),
                float(not metrics["right_base_clear"]),
                float(not metrics["robot_table_clear"]),
            ],
            dtype=np.float64,
        )
        state["fu015_task_state"] = np.asarray(
            [
                float(self._fu015_reset.seed),
                float(self._fu015_reset.root_x_m),
                float(self._fu015_reset.root_y_m),
                float(self._fu015_reset.root_yaw_deg),
                float(self._fu015_reset.hinge_angle_deg),
                float(self._fu015_reset.target_open_deg),
                float(self._fu015_reset.friction),
                float(self._fu015_reset.mass_scale),
                float(self._fu015_drive_target_deg),
                float(self._fu015_assist_active),
                float(self._fu015_assist_substeps),
                float(self._success_counter),
                float(self._fu015_failure_latched),
                float(metrics["hinge_error_deg"]),
                float(metrics["base_displacement_m"]),
                float(metrics["base_lift_m"]),
                float(metrics["success_now"]),
            ],
            dtype=np.float64,
        )
        return observation

    def restore_recorded_initial_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        """Restore the single laptop articulation root and hinge for replay."""

        base_pose = np.asarray(
            state_parts.get("laptop_base_pose", np.zeros(0)), dtype=np.float64
        ).ravel()
        hinge = np.asarray(
            state_parts.get("hinge_angle", np.zeros(0)), dtype=np.float64
        ).ravel()
        restored = False
        if (
            base_pose.size >= 7
            and hinge.size >= 1
            and self._fu015_articulation_view is not None
            and self._torch is not None
        ):
            view = self._fu015_articulation_view
            indices = self._torch.arange(
                int(view.count), dtype=self._torch.long, device=self._isaac_device
            )
            transforms = view.get_root_transforms().clone()
            velocities = self._torch.zeros_like(view.get_root_velocities())
            positions = view.get_dof_positions().clone()
            dof_velocities = self._torch.zeros_like(view.get_dof_velocities())
            transforms[:, :3] = self._torch.as_tensor(
                base_pose[:3], dtype=transforms.dtype, device=transforms.device
            )
            transforms[:, 3:7] = self._torch.as_tensor(
                [base_pose[4], base_pose[5], base_pose[6], base_pose[3]],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            positions[:, 0] = float(hinge[0])
            view.set_root_transforms(transforms, indices)
            view.set_root_velocities(velocities, indices)
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(dof_velocities, indices)
            self._fu015_reset_base_position = base_pose[:3].copy()
            self._set_hinge_drive_target(float(np.rad2deg(hinge[0])))
            restored = True
        self._refresh_object_poses()
        return {
            "restored": restored,
            "restored_task_articulation": BASE_BODY_PATH if restored else "",
            "restored_hinge": bool(restored),
            "nonroot_display_state_write": False,
        }


__all__ = ["HandsOnFU015IsaacLabEnv"]
