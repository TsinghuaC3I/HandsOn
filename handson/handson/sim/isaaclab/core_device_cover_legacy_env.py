"""Native IsaacLab environment for G02 bimanual device closure."""

from __future__ import annotations

import math
from typing import Any

from gymnasium import spaces
import numpy as np

from .core_device_cover_legacy_abc import HandsOnABCIsaacLabEnv
from .core_device_cover_legacy_scene import (
    G02_ASSET_SPECS,
    G02_BODY_HEIGHT_M,
    G02_PHYSICS_HZ,
    G02_POSITION_TOLERANCE_M,
    G02_ROTATION_TOLERANCE_DEG,
    G02_STABLE_CONTROL_FRAMES,
    G02_TASK_NAME,
    spawn_g02_repair_assets,
)
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE


_G02_OBJECT_CONTACT_FORCE_THRESHOLD = 0.02
_G02_HAND_CONTACT_FORCE_THRESHOLD = 0.10
_G02_BODY_TRANSLATION_TOLERANCE_M = 0.030
_G02_BODY_ROTATION_TOLERANCE_DEG = 12.0
_G02_RESET_XY_JITTER_M = 0.025
_G02_COVER_CRADLE_JITTER_M = 0.002
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
            "left_body": False,
            "right_cover": False,
        }
        self._g02_contact_forces = {
            "body_cover": 0.0,
            "left_body": 0.0,
            "right_cover": 0.0,
        }
        self._g02_left_body_contact_seen = False
        self._g02_right_cover_contact_seen = False
        self._g02_body_reset_pose: np.ndarray | None = None
        self._g02_last_metrics: dict[str, float | bool] = {}
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

    def _configure_task_collision_safety(
        self,
        sim_utils: Any,
    ) -> tuple[int, int]:
        bodies, colliders = super()._configure_task_collision_safety(sim_utils)
        if self._stage is None:
            return bodies, colliders
        from pxr import PhysxSchema

        configured = 0
        for name in ("device_body", "device_cover"):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not configure G02 contact reporting for {name}"
                )
                continue
            report_api = (
                PhysxSchema.PhysxContactReportAPI.Get(self._stage, prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(prim)
            )
            report_api.CreateThresholdAttr().Set(0.0)
            configured += 1
        return bodies + configured, colliders

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
            "left_body": (
                str(body_prim.GetPath()),
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
        self._g02_left_body_contact_seen = False
        self._g02_right_cover_contact_seen = False
        self._g02_body_reset_pose = None
        self._g02_last_metrics = {}
        for name in self._g02_contact_forces:
            self._g02_contact_forces[name] = 0.0
            self._g02_contact_available[name] = name in self._g02_contact_views

    def _task_specific_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        """Sample reachable +/-25 mm jitter while preserving hand ownership."""

        if not self.randomize:
            return None
        centres = {
            spec.name: np.asarray(spec.translation[:2], dtype=np.float64)
            for spec in G02_ASSET_SPECS
            if spec.dynamic
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
            for name, centre in centres.items():
                jitter = (
                    _G02_COVER_CRADLE_JITTER_M
                    if name == "device_cover"
                    else _G02_RESET_XY_JITTER_M
                )
                candidate[name] = centre + self._random.uniform(
                    -jitter,
                    jitter,
                    size=2,
                )
            # Left owns +Y body and right owns -Y cover.  Prevent randomized
            # resets from crossing the centre line into the other hand's zone.
            if candidate["device_body"][1] < 0.13:
                continue
            if candidate["device_cover"][1] > -0.13:
                continue
            distance = float(
                np.linalg.norm(candidate["device_body"] - candidate["device_cover"])
            )
            if distance > best_distance:
                best = candidate
                best_distance = distance
            if distance >= required:
                return candidate
        if best is None or best_distance < required:
            raise RuntimeError(
                "G02 could not sample a reachable collision-clear reset layout"
            )
        return best

    def _g02_pose(self, name: str) -> np.ndarray:
        fallback = self._fallback_g02_pose(name)
        return np.asarray(self._poses.get(name, fallback), dtype=np.float64).reshape(7)

    def _read_g02_contacts(self, *, update_latches: bool) -> None:
        for name in self._g02_contact_forces:
            view = self._g02_contact_views.get(name)
            available = view is not None
            force = self._c04_contact_force(view) if available else 0.0
            self._g02_contact_available[name] = available
            self._g02_contact_forces[name] = float(force)
        if update_latches:
            self._g02_left_body_contact_seen = bool(
                self._g02_left_body_contact_seen
                or self._g02_contact_forces["left_body"]
                >= _G02_HAND_CONTACT_FORCE_THRESHOLD
            )
            self._g02_right_cover_contact_seen = bool(
                self._g02_right_cover_contact_seen
                or self._g02_contact_forces["right_cover"]
                >= _G02_HAND_CONTACT_FORCE_THRESHOLD
            )

    def _g02_metrics(self, *, update_contacts: bool = False) -> dict[str, float | bool]:
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
        object_contact = bool(
            self._g02_contact_available["body_cover"]
            and self._g02_contact_forces["body_cover"]
            >= _G02_OBJECT_CONTACT_FORCE_THRESHOLD
        )
        metrics: dict[str, float | bool] = {
            "xy_error_m": xy_error,
            "vertical_offset_m": vertical_offset,
            "seating_error_m": seating_error,
            "rotation_error_deg": rotation_error,
            "body_translation_m": body_translation,
            "body_rotation_deg": body_rotation,
            "body_stable": bool(
                body_translation <= _G02_BODY_TRANSLATION_TOLERANCE_M
                and body_rotation <= _G02_BODY_ROTATION_TOLERANCE_DEG
            ),
            "body_cover_contact": object_contact,
            "left_body_contact_seen": self._g02_left_body_contact_seen,
            "right_cover_contact_seen": self._g02_right_cover_contact_seen,
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
            "contact_metrics": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float64),
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
                    metrics["body_translation_m"],
                    metrics["body_rotation_deg"],
                    float(metrics["body_stable"]),
                    self._success_counter / max(1, G02_STABLE_CONTROL_FRAMES),
                ],
                dtype=np.float64,
            ),
            "contact_metrics": np.asarray(
                [
                    float(self._g02_contact_available["body_cover"]),
                    float(self._g02_contact_available["left_body"]),
                    float(self._g02_contact_available["right_cover"]),
                    self._g02_contact_forces["body_cover"],
                    self._g02_contact_forces["left_body"],
                    self._g02_contact_forces["right_cover"],
                    float(self._g02_left_body_contact_seen),
                    float(self._g02_right_cover_contact_seen),
                    float(metrics["body_cover_contact"]),
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
            metrics["left_body_contact_seen"]
            and metrics["right_cover_contact_seen"]
            and metrics["body_stable"]
            and float(metrics["xy_error_m"]) <= G02_POSITION_TOLERANCE_M
            and float(metrics["seating_error_m"]) <= G02_POSITION_TOLERANCE_M
            and float(metrics["rotation_error_deg"]) <= G02_ROTATION_TOLERANCE_DEG
            and metrics["body_cover_contact"]
        )

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._g02_metrics()
        rows = [
            self._condition_row(
                "g02_left_stabilized_body",
                "Left hand has physically stabilized device_body",
                bool(metrics["left_body_contact_seen"]),
            ),
            self._condition_row(
                "g02_right_manipulated_cover",
                "Right hand has physically manipulated device_cover",
                bool(metrics["right_cover_contact_seen"]),
            ),
            self._condition_row(
                "g02_body_stable",
                "device_body remains near its reset pose",
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
                "g02_contact_forces": dict(self._g02_contact_forces),
                "g02_contact_available": dict(self._g02_contact_available),
                "g02_left_body_contact_seen": bool(self._g02_left_body_contact_seen),
                "g02_right_cover_contact_seen": bool(self._g02_right_cover_contact_seen),
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
    "_G02_BODY_ROTATION_TOLERANCE_DEG",
    "_G02_BODY_TRANSLATION_TOLERANCE_M",
    "_G02_HAND_CONTACT_FORCE_THRESHOLD",
    "_G02_OBJECT_CONTACT_FORCE_THRESHOLD",
]
