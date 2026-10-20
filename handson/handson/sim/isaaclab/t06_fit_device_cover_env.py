"""Native IsaacLab environment for G02 bimanual device closure."""

from __future__ import annotations

import math
from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .t06_fit_device_cover_scene import (
    G02_ASSET_SPECS,
    G02_BODY_ANGULAR_SPEED_TOLERANCE_RAD_S,
    G02_BODY_COVER_MAX_PENETRATION_M,
    G02_BODY_HEIGHT_M,
    G02_BODY_LINEAR_SPEED_TOLERANCE_M_S,
    G02_BODY_RESET_XY_BOUNDS_M,
    G02_COVER_RISER_PATH,
    G02_COVER_SIDE_MIN_OFFSET_M,
    G02_COVER_RESET_XY_BOUNDS_M,
    G02_DYNAMIC_CONTACT_OFFSET_M,
    G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES,
    G02_DUAL_CONTACT_MIN_CONTROL_FRAMES,
    G02_DUAL_TRANSPORT_MIN_DISTANCE_M,
    G02_DUAL_TRANSPORT_TARGET_XY_TOLERANCE_M,
    G02_PHYSICS_HZ,
    G02_POSITION_TOLERANCE_M,
    G02_RESET_YAW_BOUNDS_DEG,
    G02_ROTATION_TOLERANCE_DEG,
    G02_SINGLE_HAND_GRACE_CONTROL_FRAMES,
    G02_STABLE_CONTROL_FRAMES,
    G02_TASK_NAME,
    spawn_g02_repair_assets,
)
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE


_G02_OBJECT_CONTACT_FORCE_THRESHOLD = 0.02
_G02_HAND_CONTACT_FORCE_THRESHOLD = 0.10
_G02_RESET_FOOTPRINT_RADII = {
    "device_body": float(math.hypot(0.091, 0.125)),
    "device_cover": float(math.hypot(0.090, 0.130)),
}
_G02_RESET_CLEARANCE_M = 0.010


def _normalize_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _quaternion_angle_degrees(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = _normalize_quaternion_wxyz(lhs)
    right = _normalize_quaternion_wxyz(rhs)
    cosine_half = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(cosine_half)))


def _relative_quaternion_wxyz(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    rw, rx, ry, rz = _normalize_quaternion_wxyz(reference)
    vw, vx, vy, vz = _normalize_quaternion_wxyz(value)
    return _normalize_quaternion_wxyz(
        np.asarray(
            [
                rw * vw + rx * vx + ry * vy + rz * vz,
                rw * vx - rx * vw - ry * vz + rz * vy,
                rw * vy + rx * vz - ry * vw - rz * vx,
                rw * vz - rx * vy + ry * vx - rz * vw,
            ],
            dtype=np.float64,
        )
    )


def _quaternion_matrix_wxyz(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(value)
    return np.asarray(
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


class HandsOnG02RepairCloseDeviceIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Require physical two-hand manipulation and a stable seated cover."""

    _TASK_NAME = G02_TASK_NAME

    def __init__(self, task_name: str = G02_TASK_NAME, **kwargs: Any):
        if task_name != self._TASK_NAME:
            raise ValueError(
                f"G02 environment requires {self._TASK_NAME!r}, got {task_name!r}"
            )
        self._g02_asset_paths: tuple[str, ...] = ()
        self._g02_contact_views: dict[str, Any] = {}
        self._g02_contact_available = {
            "body_cover": False,
            "left_cover": False,
            "right_cover": False,
        }
        self._g02_contact_forces = {
            "body_cover": 0.0,
            "left_cover": 0.0,
            "right_cover": 0.0,
        }
        self._g02_cover_side_detail_readable = {"left": False, "right": False}
        self._g02_cover_side_forces = {"left": 0.0, "right": 0.0}
        self._g02_left_cover_contact = False
        self._g02_right_cover_contact = False
        self._g02_left_cover_side_contact = False
        self._g02_right_cover_side_contact = False
        self._g02_single_hand_contact_frames = 0
        self._g02_contact_attempt_invalid = False
        self._g02_dual_contact_frames = 0
        self._g02_dual_contact_dropout_frames = 0
        self._g02_dual_contact_duration_achieved = False
        self._g02_dual_contact_anchor_pose: np.ndarray | None = None
        self._g02_dual_contact_last_pose: np.ndarray | None = None
        self._g02_dual_transport_distance_m = 0.0
        self._g02_dual_transport_near_target = False
        self._g02_dual_transport_completed = False
        self._g02_body_cover_penetration_available = False
        self._g02_body_cover_penetration_m = 0.0
        self._g02_body_reset_pose: np.ndarray | None = None
        self._g02_sampled_reset_xy: dict[str, np.ndarray] = {}
        self._g02_last_metrics: dict[str, float | bool | int] = {}
        kwargs.setdefault("physics_dt", 1.0 / G02_PHYSICS_HZ)
        super().__init__(task_name=task_name, **kwargs)

    @staticmethod
    def _fallback_g02_pose(name: str) -> np.ndarray:
        spec = next(spec for spec in G02_ASSET_SPECS if spec.name == name)
        return np.asarray(
            [
                *spec.translation,
                0.7071067811865476,
                0.0,
                0.0,
                -0.7071067811865475,
            ],
            dtype=np.float64,
        )

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("G02 stage is unavailable before asset spawn")
        self._g02_asset_paths = spawn_g02_repair_assets(self._stage, sim_utils)
        print(
            "[HandsOn/IsaacLab] G02 authored USDs spawned: "
            + ", ".join(self._g02_asset_paths),
            flush=True,
        )

    def _task_extra_pose_names(self) -> set[str]:
        return super()._task_extra_pose_names() | {"cover_pickup_riser"}

    def _configure_task_collision_safety(
        self,
        sim_utils: Any,
    ) -> tuple[int, int]:
        bodies, colliders = super()._configure_task_collision_safety(sim_utils)
        if self._stage is None:
            return bodies, colliders
        from pxr import PhysxSchema, Usd, UsdPhysics

        configured_bodies = 0
        configured_colliders = 0
        specs_by_name = {spec.name: spec for spec in G02_ASSET_SPECS}
        for name in ("device_body", "device_cover"):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not configure G02 collision safety for {name}"
                )
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            usd_rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not usd_rigid_api:
                usd_rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)

            # Imported layers may carry a stronger rigid-mode opinion than the
            # spawn config.  Author on the resolved rigid root and read the
            # values back before PhysX parses the stage.
            expected_kinematic = specs_by_name[name].kinematic
            kinematic_attr = usd_rigid_api.GetKinematicEnabledAttr()
            if not kinematic_attr or not kinematic_attr.IsValid():
                kinematic_attr = usd_rigid_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(expected_kinematic)
            disable_gravity_attr = rigid_api.GetDisableGravityAttr()
            if not disable_gravity_attr or not disable_gravity_attr.IsValid():
                disable_gravity_attr = rigid_api.CreateDisableGravityAttr()
            disable_gravity_attr.Set(expected_kinematic)
            if bool(kinematic_attr.Get()) != expected_kinematic:
                raise RuntimeError(
                    f"G02 {name} kinematic readback does not match the task contract"
                )
            if bool(disable_gravity_attr.Get()) != expected_kinematic:
                raise RuntimeError(
                    f"G02 {name} gravity readback does not match the task contract"
                )
            # Isaac Sim 5.1 GPU PhysX ignores scene-level swept CCD.  Keep the
            # GPU-supported predictive mode scoped to the two task objects;
            # applying it to hand links could make release contacts feel sticky.
            speculative_ccd_attr = rigid_api.GetEnableSpeculativeCCDAttr()
            if not speculative_ccd_attr or not speculative_ccd_attr.IsValid():
                speculative_ccd_attr = rigid_api.CreateEnableSpeculativeCCDAttr()
            speculative_ccd_attr.Set(True)
            report_api = (
                PhysxSchema.PhysxContactReportAPI.Get(self._stage, prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(prim)
            )
            report_api.CreateThresholdAttr().Set(0.0)
            for descendant in Usd.PrimRange(prim):
                if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(
                        descendant
                    )
                contact_attr = collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = collision_api.CreateContactOffsetAttr()
                contact_attr.Set(G02_DYNAMIC_CONTACT_OFFSET_M)
                rest_attr = collision_api.GetRestOffsetAttr()
                if not rest_attr or not rest_attr.IsValid():
                    rest_attr = collision_api.CreateRestOffsetAttr()
                rest_attr.Set(0.0)
                configured_colliders += 1
            configured_bodies += 1
        if configured_bodies != 2 or configured_colliders != 2:
            raise RuntimeError(
                "G02 predictive collision safety is incomplete: "
                f"bodies={configured_bodies}/2, colliders={configured_colliders}/2"
            )
        return bodies + configured_bodies, colliders + configured_colliders

    def _capture_task_scene_state(self, task_mjcf: Any) -> None:
        """Fail closed unless both manipulated objects remain dynamic."""

        super()._capture_task_scene_state(task_mjcf)
        if self._stage is None or not self._robots:
            return
        body_prim = self._find_rigid_body_prim("device_body")
        cover_prim = self._find_rigid_body_prim("device_cover")
        riser_prim = self._find_rigid_body_prim("cover_pickup_riser")
        if body_prim is None or cover_prim is None or riser_prim is None:
            raise RuntimeError("G02 live rigid roots are unavailable after PhysX parse")
        body_path = str(body_prim.GetPath())
        cover_path = str(cover_prim.GetPath())
        riser_path = str(riser_prim.GetPath())
        if body_path in self._task_kinematic_body_paths:
            raise RuntimeError(
                "G02 device_body was incorrectly parsed as kinematic: "
                f"{body_path}"
            )
        if cover_path in self._task_kinematic_body_paths:
            raise RuntimeError(
                "G02 device_cover was incorrectly parsed as kinematic: "
                f"{cover_path}"
            )
        if riser_path not in self._task_kinematic_body_paths:
            raise RuntimeError(
                "G02 cover pickup riser was not parsed as kinematic: "
                f"{riser_path}"
            )
        print(
            "[HandsOn/IsaacLab] G02 rigid-mode readback: "
            f"device_body={body_path} kinematic=False, "
            f"device_cover={cover_path} kinematic=False, "
            f"cover_pickup_riser={riser_path} kinematic=True",
            flush=True,
        )

    def _g02_hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if root is None or not root.IsValid():
            return []
        prefix = f"{side}_finger"
        palm = f"{side}_palm_link"
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (prim.GetName() == palm or prim.GetName().startswith(prefix))
        ]

    def _init_task_specific_runtime(self) -> None:
        self._g02_contact_views = {}
        if not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        body_prim = self._find_rigid_body_prim("device_body")
        cover_prim = self._find_rigid_body_prim("device_cover")
        if physics_sim_view is None or body_prim is None or cover_prim is None:
            self._add_task_state_warning("G02 contact reporting is unavailable")
            return

        bindings = {
            "body_cover": (str(cover_prim.GetPath()), [str(body_prim.GetPath())]),
            "left_cover": (
                str(cover_prim.GetPath()),
                self._g02_hand_body_paths("left"),
            ),
            "right_cover": (
                str(cover_prim.GetPath()),
                self._g02_hand_body_paths("right"),
            ),
        }
        for name, (sensor, filters) in bindings.items():
            if not filters:
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    sensor,
                    filter_patterns=filters,
                    max_contact_data_count=max(64, len(filters) * 8),
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._g02_contact_views[name] = view
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not initialize G02 {name} contact view: {exc}"
                )
        missing = sorted(set(bindings) - set(self._g02_contact_views))
        if missing:
            self._add_task_state_warning(
                "G02 contact reporting is incomplete and success will fail closed: "
                + ",".join(missing)
            )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._g02_cover_side_detail_readable = {"left": False, "right": False}
        self._g02_cover_side_forces = {"left": 0.0, "right": 0.0}
        self._g02_left_cover_contact = False
        self._g02_right_cover_contact = False
        self._g02_left_cover_side_contact = False
        self._g02_right_cover_side_contact = False
        self._g02_single_hand_contact_frames = 0
        self._g02_contact_attempt_invalid = False
        self._g02_dual_contact_frames = 0
        self._g02_dual_contact_dropout_frames = 0
        self._g02_dual_contact_duration_achieved = False
        self._g02_dual_contact_anchor_pose = None
        self._g02_dual_contact_last_pose = None
        self._g02_dual_transport_distance_m = 0.0
        self._g02_dual_transport_near_target = False
        self._g02_dual_transport_completed = False
        self._g02_body_cover_penetration_available = False
        self._g02_body_cover_penetration_m = 0.0
        self._g02_body_reset_pose = None
        self._g02_last_metrics = {}
        for name in self._g02_contact_forces:
            self._g02_contact_forces[name] = 0.0
            self._g02_contact_available[name] = name in self._g02_contact_views

    def _task_specific_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        """Sample broad, reachable XY layouts for both dynamic objects."""

        self._g02_sampled_reset_xy = {}
        if not self.randomize:
            return None
        bounds_by_name = {
            "device_body": G02_BODY_RESET_XY_BOUNDS_M,
            "device_cover": G02_COVER_RESET_XY_BOUNDS_M,
        }
        required = (
            _G02_RESET_FOOTPRINT_RADII["device_body"]
            + _G02_RESET_FOOTPRINT_RADII["device_cover"]
            + _G02_RESET_CLEARANCE_M
        )
        best: dict[str, np.ndarray] | None = None
        best_distance = -math.inf
        for _ in range(128):
            candidate = {}
            for name, bounds in bounds_by_name.items():
                lower = np.asarray(
                    [bounds[0][0], bounds[1][0]],
                    dtype=np.float64,
                )
                upper = np.asarray(
                    [bounds[0][1], bounds[1][1]],
                    dtype=np.float64,
                )
                candidate[name] = self._random.uniform(lower, upper)
            # The two broad, opposite-side tabletop zones preserve a meaningful
            # cooperative transport while allowing 40 cm of independent depth
            # variation instead of jittering around one front-edge anchor.
            distance = float(
                np.linalg.norm(candidate["device_body"] - candidate["device_cover"])
            )
            if distance > best_distance:
                best = candidate
                best_distance = distance
            if distance >= required:
                self._g02_sampled_reset_xy = {
                    name: xy.copy() for name, xy in candidate.items()
                }
                return candidate
        if best is None or best_distance < required:
            raise RuntimeError(
                "G02 could not sample a reachable collision-clear reset layout"
            )
        self._g02_sampled_reset_xy = {
            name: xy.copy() for name, xy in best.items()
        }
        return best

    def _apply_task_pose_randomization(self) -> None:
        """Move the fixed pickup riser beneath the broadly randomized cover."""

        self._g02_sampled_reset_xy = {}
        super()._apply_task_pose_randomization()
        cover_xy = self._g02_sampled_reset_xy.get("device_cover")
        if cover_xy is None:
            return
        riser_state = next(
            (
                (path, view, indices, pose, velocity)
                for path, view, indices, pose, velocity
                in self._task_reset_body_states
                if path == G02_COVER_RISER_PATH
            ),
            None,
        )
        if riser_state is None:
            raise RuntimeError("G02 cover pickup riser reset state is unavailable")
        path, view, indices, reset_pose, reset_velocity = riser_state
        pose_tensor = reset_pose.clone()
        pose_tensor[:, :2] = self._torch.as_tensor(
            cover_xy,
            dtype=pose_tensor.dtype,
            device=pose_tensor.device,
        )
        self._write_rigid_reset_state(
            path,
            view,
            indices,
            pose_tensor,
            self._torch.zeros_like(reset_velocity),
        )

    def _g02_pose(self, name: str) -> np.ndarray:
        fallback = self._fallback_g02_pose(name)
        return np.asarray(self._poses.get(name, fallback), dtype=np.float64).reshape(7)

    def _g02_body_cover_penetration(self, view: Any | None) -> tuple[float, bool]:
        """Read the current body/cover penetration from public PhysX buffers."""

        if view is None:
            return 0.0, False
        try:
            data = view.get_contact_data(self._sim_dt)
            if len(data) != 6:
                return 0.0, False
            separations = data[3]
            counts = data[4]
            starts = data[5]
            if hasattr(separations, "detach"):
                separations = separations.detach().cpu().numpy()
            if hasattr(counts, "detach"):
                counts = counts.detach().cpu().numpy()
            if hasattr(starts, "detach"):
                starts = starts.detach().cpu().numpy()
            separations = np.asarray(separations, dtype=np.float64).reshape(-1)
            raw_counts = np.asarray(counts).reshape(-1)
            raw_starts = np.asarray(starts).reshape(-1)
            if (
                raw_counts.shape != raw_starts.shape
                or not np.all(np.isfinite(raw_counts))
                or not np.all(np.isfinite(raw_starts))
                or not np.all(raw_counts == np.floor(raw_counts))
                or not np.all(raw_starts == np.floor(raw_starts))
            ):
                return 0.0, False
            counts = raw_counts.astype(np.int64, copy=False)
            starts = raw_starts.astype(np.int64, copy=False)
            if np.any(counts < 0) or np.any(starts < 0):
                return 0.0, False
            selected: list[np.ndarray] = []
            for count, start in zip(counts, starts):
                stop = int(start + count)
                if stop > separations.size:
                    return 0.0, False
                if count:
                    selected.append(separations[int(start):stop])
            if not selected:
                return 0.0, True
            values = np.concatenate(selected)
            if not np.all(np.isfinite(values)):
                return 0.0, False
            return max(0.0, -float(np.min(values))), True
        except Exception:
            return 0.0, False

    def _g02_contact_points(
        self,
        view: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return finite detailed-contact rows for one hand/cover view."""

        empty_points = np.zeros((0, 3), dtype=np.float64)
        empty_forces = np.zeros(0, dtype=np.float64)
        if view is None:
            return empty_points, empty_forces, False

        def as_numpy(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        try:
            data = view.get_contact_data(self._sim_dt)
            if not isinstance(data, (tuple, list)) or len(data) != 6:
                return empty_points, empty_forces, False
            forces_np = np.abs(
                as_numpy(data[0]).astype(np.float64).reshape(-1)
            )
            points_np = (
                as_numpy(data[1]).astype(np.float64).reshape(-1, 3)
            )
            raw_counts = as_numpy(data[4]).reshape(-1)
            raw_starts = as_numpy(data[5]).reshape(-1)
            if (
                raw_counts.shape != raw_starts.shape
                or not np.all(np.isfinite(raw_counts))
                or not np.all(np.isfinite(raw_starts))
                or not np.all(raw_counts == np.floor(raw_counts))
                or not np.all(raw_starts == np.floor(raw_starts))
            ):
                return empty_points, empty_forces, False
            counts_np = raw_counts.astype(np.int64, copy=False)
            starts_np = raw_starts.astype(np.int64, copy=False)
            if np.any(counts_np < 0) or np.any(starts_np < 0):
                return empty_points, empty_forces, False
            selected_points: list[np.ndarray] = []
            selected_forces: list[np.ndarray] = []
            for count, start in zip(counts_np, starts_np, strict=True):
                stop = int(start + count)
                if stop > len(points_np) or stop > len(forces_np):
                    return empty_points, empty_forces, False
                if count:
                    selected_points.append(points_np[int(start):stop])
                    selected_forces.append(forces_np[int(start):stop])
            valid_points = (
                np.concatenate(selected_points, axis=0)
                if selected_points
                else empty_points
            )
            valid_forces = (
                np.concatenate(selected_forces)
                if selected_forces
                else empty_forces
            )
            if not np.all(np.isfinite(valid_points)) or not np.all(
                np.isfinite(valid_forces)
            ):
                return empty_points, empty_forces, False
            return valid_points, valid_forces, True
        except Exception:
            return empty_points, empty_forces, False

    def _g02_opposite_cover_side_contacts(
        self,
        left_view: Any | None,
        right_view: Any | None,
        cover: np.ndarray,
    ) -> tuple[bool, bool, float, float, bool, bool]:
        """Require force-bearing hand contacts on either opposite edge pair."""

        rotation = _quaternion_matrix_wxyz(cover[3:7])
        contact_centres: dict[str, np.ndarray] = {}
        contact_forces: dict[str, float] = {}
        readable_by_side: dict[str, bool] = {}
        for side, view in (("left", left_view), ("right", right_view)):
            points, forces, readable = self._g02_contact_points(view)
            readable_by_side[side] = readable
            total_force = float(np.sum(forces)) if readable else 0.0
            contact_forces[side] = total_force
            if (
                readable
                and len(points)
                and total_force >= _G02_HAND_CONTACT_FORCE_THRESHOLD
            ):
                weights = forces / total_force
                world_centre = np.sum(points * weights[:, None], axis=0)
                contact_centres[side] = rotation.T @ (
                    world_centre - cover[:3]
                )

        opposite = False
        if len(contact_centres) == 2:
            left = contact_centres["left"]
            right = contact_centres["right"]
            for axis in (0, 1):
                if (
                    abs(float(left[axis])) >= G02_COVER_SIDE_MIN_OFFSET_M
                    and abs(float(right[axis])) >= G02_COVER_SIDE_MIN_OFFSET_M
                    and float(left[axis] * right[axis]) < 0.0
                ):
                    opposite = True
                    break
        return (
            opposite,
            opposite,
            contact_forces.get("left", 0.0),
            contact_forces.get("right", 0.0),
            readable_by_side.get("left", False),
            readable_by_side.get("right", False),
        )

    def _reset_g02_active_transport_attempt(self) -> None:
        self._g02_dual_contact_frames = 0
        self._g02_dual_contact_dropout_frames = 0
        self._g02_dual_contact_anchor_pose = None
        self._g02_dual_contact_last_pose = None
        if not self._g02_dual_transport_completed:
            self._g02_dual_contact_duration_achieved = False
            self._g02_dual_transport_distance_m = 0.0
            self._g02_dual_transport_near_target = False

    def _advance_g02_cooperative_transport(
        self,
        *,
        body: np.ndarray,
        cover: np.ndarray,
    ) -> None:
        """Advance the recoverable two-hand contact/transport attempt."""

        left = bool(self._g02_left_cover_contact)
        right = bool(self._g02_right_cover_contact)
        both_opposite_sides = bool(
            self._g02_left_cover_side_contact
            and self._g02_right_cover_side_contact
        )
        if self._g02_dual_transport_completed:
            self._g02_dual_contact_dropout_frames = 0
            if not left and not right:
                self._g02_single_hand_contact_frames = 0
                self._g02_contact_attempt_invalid = False
            elif left == right:
                self._g02_single_hand_contact_frames = 0
            else:
                self._g02_single_hand_contact_frames += 1
            return

        active_dual_attempt = self._g02_dual_contact_anchor_pose is not None
        if not left and not right:
            self._g02_single_hand_contact_frames = 0
            self._g02_contact_attempt_invalid = False
            if active_dual_attempt:
                self._g02_dual_contact_dropout_frames += 1
                if (
                    self._g02_dual_contact_dropout_frames
                    <= G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES
                ):
                    return
            self._reset_g02_active_transport_attempt()
            return

        if left != right:
            if active_dual_attempt:
                self._g02_dual_contact_dropout_frames += 1
                if (
                    self._g02_dual_contact_dropout_frames
                    <= G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES
                ):
                    return
                self._reset_g02_active_transport_attempt()
            self._g02_single_hand_contact_frames += 1
            if (
                self._g02_single_hand_contact_frames
                > G02_SINGLE_HAND_GRACE_CONTROL_FRAMES
            ):
                # This invalidates only the current grasp attempt.  It is a
                # recoverable progress guard, never an episode termination.
                self._g02_contact_attempt_invalid = True
                self._g02_dual_contact_duration_achieved = False
                self._g02_dual_transport_distance_m = 0.0
                self._g02_dual_transport_near_target = False
                self._g02_dual_transport_completed = False
            return

        self._g02_single_hand_contact_frames = 0
        if self._g02_contact_attempt_invalid:
            self._reset_g02_active_transport_attempt()
            return
        if not both_opposite_sides:
            if active_dual_attempt:
                self._g02_dual_contact_dropout_frames += 1
                if (
                    self._g02_dual_contact_dropout_frames
                    <= G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES
                ):
                    return
            self._reset_g02_active_transport_attempt()
            return

        resumed_after_dropout = self._g02_dual_contact_dropout_frames > 0
        self._g02_dual_contact_dropout_frames = 0
        if self._g02_dual_contact_anchor_pose is None:
            self._g02_dual_contact_anchor_pose = cover.copy()
            self._g02_dual_contact_last_pose = cover.copy()
            self._g02_dual_contact_frames = 0
            self._g02_dual_transport_distance_m = 0.0
            self._g02_dual_transport_near_target = False
        elif resumed_after_dropout and self._g02_dual_contact_last_pose is not None:
            # A cover displacement observed while one/both contact streams
            # were absent is not cooperative transport.  Shift the reference
            # by that unobserved delta so it cannot be credited on reacquire.
            self._g02_dual_contact_anchor_pose[:3] += (
                cover[:3] - self._g02_dual_contact_last_pose[:3]
            )
        self._g02_dual_contact_frames += 1
        self._g02_dual_transport_distance_m = max(
            self._g02_dual_transport_distance_m,
            float(
                np.linalg.norm(
                    cover[:3] - self._g02_dual_contact_anchor_pose[:3]
                )
            ),
        )
        xy_error = float(np.linalg.norm(cover[:2] - body[:2]))
        vertical_offset = float(cover[2] - body[2])
        self._g02_dual_transport_near_target = bool(
            xy_error <= G02_DUAL_TRANSPORT_TARGET_XY_TOLERANCE_M
            and vertical_offset
            >= G02_BODY_HEIGHT_M - G02_POSITION_TOLERANCE_M
        )
        self._g02_dual_contact_last_pose = cover.copy()
        if (
            self._g02_dual_contact_frames
            >= G02_DUAL_CONTACT_MIN_CONTROL_FRAMES
        ):
            self._g02_dual_contact_duration_achieved = True
        if (
            self._g02_dual_contact_duration_achieved
            and self._g02_dual_transport_distance_m
            >= G02_DUAL_TRANSPORT_MIN_DISTANCE_M
            and self._g02_dual_transport_near_target
        ):
            self._g02_dual_transport_completed = True

    def _read_g02_contacts(self, *, update_latches: bool) -> None:
        for name in self._g02_contact_forces:
            view = self._g02_contact_views.get(name)
            available = view is not None
            force = self._c04_contact_force(view) if available else 0.0
            self._g02_contact_available[name] = available
            self._g02_contact_forces[name] = float(force)
        penetration, penetration_available = self._g02_body_cover_penetration(
            self._g02_contact_views.get("body_cover")
        )
        self._g02_body_cover_penetration_m = penetration
        self._g02_body_cover_penetration_available = penetration_available
        self._g02_left_cover_contact = bool(
            self._g02_contact_available["left_cover"]
            and self._g02_contact_forces["left_cover"]
            >= _G02_HAND_CONTACT_FORCE_THRESHOLD
        )
        self._g02_right_cover_contact = bool(
            self._g02_contact_available["right_cover"]
            and self._g02_contact_forces["right_cover"]
            >= _G02_HAND_CONTACT_FORCE_THRESHOLD
        )
        cover = self._g02_pose("device_cover")
        (
            self._g02_left_cover_side_contact,
            self._g02_right_cover_side_contact,
            self._g02_cover_side_forces["left"],
            self._g02_cover_side_forces["right"],
            self._g02_cover_side_detail_readable["left"],
            self._g02_cover_side_detail_readable["right"],
        ) = self._g02_opposite_cover_side_contacts(
            self._g02_contact_views.get("left_cover"),
            self._g02_contact_views.get("right_cover"),
            cover,
        )
        if update_latches:
            self._advance_g02_cooperative_transport(
                body=self._g02_pose("device_body"),
                cover=cover,
            )

    def _g02_metrics(
        self,
        *,
        update_contacts: bool = False,
    ) -> dict[str, float | bool | int]:
        if update_contacts:
            self._read_g02_contacts(update_latches=True)
        body = self._g02_pose("device_body")
        cover = self._g02_pose("device_cover")
        if self._g02_body_reset_pose is None:
            self._g02_body_reset_pose = body.copy()
        reset_body = self._g02_body_reset_pose
        xy_error = float(np.linalg.norm(cover[:2] - body[:2]))
        vertical_offset = float(cover[2] - body[2])
        seating_error = abs(vertical_offset - G02_BODY_HEIGHT_M)
        rotation_error = _quaternion_angle_degrees(body[3:7], cover[3:7])
        body_translation = float(np.linalg.norm(body[:3] - reset_body[:3]))
        body_rotation = _quaternion_angle_degrees(reset_body[3:7], body[3:7])
        body_velocity = self._physics_body_velocity("device_body")
        body_velocity_available = bool(
            body_velocity is not None
            and np.asarray(body_velocity).shape == (6,)
            and np.all(np.isfinite(body_velocity))
        )
        if body_velocity_available:
            body_velocity = np.asarray(body_velocity, dtype=np.float64)
            body_linear_speed = float(np.linalg.norm(body_velocity[:3]))
            body_angular_speed = float(np.linalg.norm(body_velocity[3:]))
        else:
            # Keep observation leaves finite while the availability flag makes
            # the stability predicate fail closed.
            body_linear_speed = -1.0
            body_angular_speed = -1.0
        object_contact = bool(
            self._g02_contact_available["body_cover"]
            and self._g02_contact_forces["body_cover"]
            >= _G02_OBJECT_CONTACT_FORCE_THRESHOLD
        )
        both_sides_contact = bool(
            self._g02_left_cover_side_contact
            and self._g02_right_cover_side_contact
        )
        no_cover_hand_contact = bool(
            not self._g02_left_cover_contact
            and not self._g02_right_cover_contact
        )
        metrics: dict[str, float | bool | int] = {
            "xy_error_m": xy_error,
            "vertical_offset_m": vertical_offset,
            "seating_error_m": seating_error,
            "rotation_error_deg": rotation_error,
            "body_translation_m": body_translation,
            "body_rotation_deg": body_rotation,
            "body_linear_speed_m_s": body_linear_speed,
            "body_angular_speed_rad_s": body_angular_speed,
            "body_velocity_available": body_velocity_available,
            "body_stable": bool(
                body_velocity_available
                and body_linear_speed <= G02_BODY_LINEAR_SPEED_TOLERANCE_M_S
                and body_angular_speed <= G02_BODY_ANGULAR_SPEED_TOLERANCE_RAD_S
            ),
            "body_cover_contact": object_contact,
            "body_cover_penetration_m": self._g02_body_cover_penetration_m,
            "body_cover_penetration_available": (
                self._g02_body_cover_penetration_available
            ),
            "body_cover_penetration_ok": bool(
                self._g02_body_cover_penetration_available
                and self._g02_body_cover_penetration_m
                <= G02_BODY_COVER_MAX_PENETRATION_M
            ),
            "left_cover_contact": self._g02_left_cover_contact,
            "right_cover_contact": self._g02_right_cover_contact,
            "left_cover_side_contact": self._g02_left_cover_side_contact,
            "right_cover_side_contact": self._g02_right_cover_side_contact,
            "both_cover_sides_contact": both_sides_contact,
            "single_hand_contact_frames": self._g02_single_hand_contact_frames,
            "single_hand_grace_ok": not self._g02_contact_attempt_invalid,
            "dual_contact_frames": self._g02_dual_contact_frames,
            "dual_contact_dropout_frames": (
                self._g02_dual_contact_dropout_frames
            ),
            "dual_contact_duration_achieved": (
                self._g02_dual_contact_duration_achieved
            ),
            "dual_transport_distance_m": self._g02_dual_transport_distance_m,
            "dual_transport_near_target": self._g02_dual_transport_near_target,
            "dual_transport_completed": self._g02_dual_transport_completed,
            "cover_contact_mode_valid": bool(
                no_cover_hand_contact or both_sides_contact
            ),
        }
        self._g02_last_metrics = metrics
        return metrics

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
            "device_body_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "device_cover_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "cover_to_body": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "alignment_metrics": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            "contact_metrics": spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float64),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        observation_spaces: dict[str, spaces.Space] = {"state": spaces.Dict(state_spaces)}
        if self.image_obs:
            head = lambda: spaces.Box(0, 255, shape=self._image_shape, dtype=np.uint8)
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_shape = (wrist_height, wrist_width, 3)
            wrist = lambda: spaces.Box(0, 255, shape=wrist_shape, dtype=np.uint8)
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": head(),
                    "wrist_left": wrist(),
                    "wrist_right": wrist(),
                }
            )
        return spaces.Dict(observation_spaces)

    def _compute_observation(self) -> dict[str, Any]:
        body = self._g02_pose("device_body")
        cover = self._g02_pose("device_cover")
        metrics = self._g02_metrics()
        tcp = np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(np.float64)
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[:16]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[:16]
        relative = np.concatenate(
            [cover[:3] - body[:3], _relative_quaternion_wxyz(body[3:7], cover[3:7])]
        )
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "device_body_pose": body.copy(),
            "device_cover_pose": cover.copy(),
            "cover_to_body": relative.astype(np.float64),
            "alignment_metrics": np.asarray(
                [
                    metrics["xy_error_m"],
                    metrics["vertical_offset_m"],
                    metrics["seating_error_m"],
                    metrics["rotation_error_deg"],
                    metrics["body_linear_speed_m_s"],
                    metrics["body_angular_speed_rad_s"],
                    float(metrics["body_stable"]),
                    self._success_counter / max(1, G02_STABLE_CONTROL_FRAMES),
                ],
                dtype=np.float64,
            ),
            "contact_metrics": np.asarray(
                [
                    float(self._g02_contact_available["body_cover"]),
                    float(self._g02_contact_available["left_cover"]),
                    float(self._g02_contact_available["right_cover"]),
                    self._g02_contact_forces["body_cover"],
                    self._g02_contact_forces["left_cover"],
                    self._g02_contact_forces["right_cover"],
                    float(self._g02_cover_side_detail_readable["left"]),
                    float(self._g02_cover_side_detail_readable["right"]),
                    float(self._g02_left_cover_contact),
                    float(self._g02_right_cover_contact),
                    float(self._g02_left_cover_side_contact),
                    float(self._g02_right_cover_side_contact),
                    float(metrics["both_cover_sides_contact"]),
                    float(self._g02_single_hand_contact_frames),
                    float(self._g02_contact_attempt_invalid),
                    float(self._g02_dual_contact_frames),
                    self._g02_dual_transport_distance_m,
                    float(self._g02_dual_transport_near_target),
                    float(self._g02_dual_transport_completed),
                ],
                dtype=np.float64,
            ),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
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

    def _compute_success_now(self) -> bool:
        metrics = self._g02_metrics(update_contacts=True)
        return bool(
            metrics["dual_transport_completed"]
            and metrics["cover_contact_mode_valid"]
            and metrics["body_stable"]
            and float(metrics["xy_error_m"]) <= G02_POSITION_TOLERANCE_M
            and float(metrics["seating_error_m"]) <= G02_POSITION_TOLERANCE_M
            and float(metrics["rotation_error_deg"]) <= G02_ROTATION_TOLERANCE_DEG
            and metrics["body_cover_contact"]
            and metrics["body_cover_penetration_ok"]
        )

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._g02_metrics()
        rows = [
            self._condition_row(
                "g02_opposite_side_dual_contact",
                "Both hands held opposite cover sides for at least 0.2 seconds",
                bool(metrics["dual_contact_duration_achieved"]),
            ),
            self._condition_row(
                "g02_cooperative_cover_transport",
                "Cover moved at least 100 mm during continuous two-hand contact and reached above device_body",
                bool(metrics["dual_transport_completed"]),
            ),
            self._condition_row(
                "g02_cover_contact_mode",
                "Current cover contact is either both opposite sides or neither hand",
                bool(metrics["cover_contact_mode_valid"]),
            ),
            self._condition_row(
                "g02_body_stable",
                "device_body is dynamically stable at its current pose",
                bool(metrics["body_stable"]),
            ),
            self._condition_row(
                "g02_xy_aligned",
                "device_cover horizontal error is at most 12 mm",
                float(metrics["xy_error_m"]) <= G02_POSITION_TOLERANCE_M,
            ),
            self._condition_row(
                "g02_interface_seated",
                "device_cover seating-height error is at most 12 mm",
                float(metrics["seating_error_m"]) <= G02_POSITION_TOLERANCE_M,
            ),
            self._condition_row(
                "g02_rotation_aligned",
                "device_cover rotation error is at most 8 degrees",
                float(metrics["rotation_error_deg"]) <= G02_ROTATION_TOLERANCE_DEG,
            ),
            self._condition_row(
                "g02_body_cover_contact",
                "device_body and device_cover have physical contact",
                bool(metrics["body_cover_contact"]),
            ),
            self._condition_row(
                "g02_body_cover_penetration",
                "device_body/device_cover penetration is at most 3 mm",
                bool(metrics["body_cover_penetration_ok"]),
            ),
        ]
        return self._append_success_hold_row(rows)

    def _task_info(self) -> dict[str, Any]:
        metrics = self._g02_metrics()
        info = super()._task_info()
        info.update(
            {
                "task_id": "G02",
                "task_name": G02_TASK_NAME,
                "g02_source_scene": "scene_06_electronics_repair",
                "g02_active_objects": ("device_body", "device_cover"),
                "g02_position_tolerance_m": G02_POSITION_TOLERANCE_M,
                "g02_rotation_tolerance_deg": G02_ROTATION_TOLERANCE_DEG,
                "g02_body_reset_xy_bounds_m": G02_BODY_RESET_XY_BOUNDS_M,
                "g02_cover_reset_xy_bounds_m": G02_COVER_RESET_XY_BOUNDS_M,
                "g02_reset_yaw_bounds_deg": G02_RESET_YAW_BOUNDS_DEG,
                "g02_body_linear_speed_tolerance_m_s": (
                    G02_BODY_LINEAR_SPEED_TOLERANCE_M_S
                ),
                "g02_body_angular_speed_tolerance_rad_s": (
                    G02_BODY_ANGULAR_SPEED_TOLERANCE_RAD_S
                ),
                "g02_body_cover_max_penetration_m": (
                    G02_BODY_COVER_MAX_PENETRATION_M
                ),
                "g02_cover_side_min_offset_m": G02_COVER_SIDE_MIN_OFFSET_M,
                "g02_single_hand_grace_control_frames": (
                    G02_SINGLE_HAND_GRACE_CONTROL_FRAMES
                ),
                "g02_dual_contact_min_control_frames": (
                    G02_DUAL_CONTACT_MIN_CONTROL_FRAMES
                ),
                "g02_dual_contact_dropout_grace_control_frames": (
                    G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES
                ),
                "g02_dual_transport_min_distance_m": (
                    G02_DUAL_TRANSPORT_MIN_DISTANCE_M
                ),
                "g02_dual_transport_target_xy_tolerance_m": (
                    G02_DUAL_TRANSPORT_TARGET_XY_TOLERANCE_M
                ),
                "g02_contact_forces": dict(self._g02_contact_forces),
                "g02_contact_available": dict(self._g02_contact_available),
                "g02_cover_side_detail_readable": dict(
                    self._g02_cover_side_detail_readable
                ),
                "g02_cover_side_forces": dict(self._g02_cover_side_forces),
                "g02_left_cover_contact": bool(
                    self._g02_left_cover_contact
                ),
                "g02_right_cover_contact": bool(
                    self._g02_right_cover_contact
                ),
                "g02_contact_attempt_invalid": bool(
                    self._g02_contact_attempt_invalid
                ),
                "g02_dual_transport_completed": bool(
                    self._g02_dual_transport_completed
                ),
                "g02_metrics": dict(metrics),
                "g02_physical_replay_certified": False,
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        contract.update(
            {
                "g02_rigid_bodies_in_formal_truth": (
                    "device_body",
                    "device_cover",
                ),
                "g02_requires_fresh_process_replay": True,
                "g02_physical_replay_certified": False,
            }
        )
        return contract


__all__ = [
    "HandsOnG02RepairCloseDeviceIsaacLabEnv",
    "_G02_HAND_CONTACT_FORCE_THRESHOLD",
    "_G02_OBJECT_CONTACT_FORCE_THRESHOLD",
]
