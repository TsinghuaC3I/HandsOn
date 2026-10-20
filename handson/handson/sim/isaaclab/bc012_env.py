"""Isaac Sim 5.1 PhysX collection runtime for B320-BC012."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from ...tasks.t13_load_cans.scene import (
    BIN_ANCHOR_XY,
    BIN_NAME,
    BIN_SUPPORT_CLEARANCE,
    BIN_XY_JITTER,
    CAN_ANCHOR_XY,
    CAN_BIN_CLEARANCE,
    CAN_CAN_CLEARANCE,
    CAN_NAMES,
    CAN_SUPPORT_CLEARANCE,
    CAN_XY_JITTER,
    CAN_YAW_DEGREES,
    GRASP_D6_AXES,
    GRASP_D6_FREE_LIMIT_BY_AXIS,
    GRASP_D6_LOCK_LIMIT_BY_AXIS,
    HAND_CLEARANCE,
    HAND_CLEAR_DISTANCE,
    HAND_CONTACT_FORCE_THRESHOLD,
    LEFT_CAN_NAME,
    PARTITION_CENTERS_LOCAL,
    RESET_MAX_ATTEMPTS,
    RIGHT_CAN_NAME,
    STABILITY_STEPS,
    TABLE_TOP_Z,
    TARGET_PARTITION_INDEX_BY_SIDE,
    TASK_NAME,
    TASK_ROOT,
    WORKSPACE_HIGH_XY,
    WORKSPACE_LOW_XY,
    evaluate_bin_load,
    joint_local_frame_pair,
    matrix_to_quat_wxyz,
    partition_centers_world,
    pose_matrix,
    spawn_bc012_assets,
)
from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _C_TOOL_WUJI_POWER_GRIP,
    _matrix_to_quat_wxyz,
    _randomized_physx_pose,
)
from .b320_bimanual_scene import B320_CONTROL_HZ, B320_PHYSICS_HZ
from .compat import mjcf_robot_root_link_path


_OWNER_CONTINUITY_POSITION_TOLERANCE_M = 0.020
_OWNER_CONTINUITY_ANGLE_TOLERANCE_RAD = math.radians(12.0)
_CAN_CENTER_IN_POWER_GRIP_PALM_M = np.asarray(
    [0.055, 0.0, 0.055], dtype=np.float64
)
_RELEASE_DROP_HEIGHT_M = 0.0
_OUTSIDE_CROSS_X_LOCAL_M = -0.225
_OUTSIDE_CROSS_Y_LOCAL_M = 0.10
_RIGHT_OUTSIDE_CROSS_Z_LOCAL_M = 0.34
_LEFT_OUTSIDE_CROSS_Z_LOCAL_M = 0.34
_RIGHT_STAGED_CLEAR_X_LOCAL_M = 0.10
_PLACEMENT_COMMAND_X_LOCAL_M = 0.05
_PLACEMENT_COMMAND_Y_LOCAL_M = 0.046
_PLACEMENT_PALM_COMPENSATION_LOCAL_M = {
    "right": np.asarray([0.005, -0.07, 0.0], dtype=np.float64),
    "left": np.asarray([0.055, 0.06, 0.0], dtype=np.float64),
}
_PRE_RELEASE_HAND_MAX = 0.25
_BIN_ENTRY_Z_LOCAL_M = 0.22
_PHYSICAL_GRASP_CONFIRM_FRAMES = 2
_PHYSICAL_RELEASE_CONFIRM_FRAMES = 2
_PHYSICAL_GRASP_MIN_LIFT_M = 0.015


class BC012IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Dual-arm, two-can, crossed-partition collection environment."""

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if str(task_name) != TASK_NAME:
            raise ValueError(
                f"BC012IsaacLabEnv requires task_name={TASK_NAME!r}, "
                f"got {task_name!r}"
            )
        self._bc012_scene_manifest: dict[str, Any] = {}
        self._bc012_contact_views: dict[str, dict[str, Any]] = {
            name: {} for name in CAN_NAMES
        }
        self._bc012_grasp_owner: dict[str, bool] = {
            "left": False,
            "right": False,
        }
        self._bc012_physical_grasp_seen = {"left": False, "right": False}
        self._bc012_grasp_contact_streak = {"left": 0, "right": 0}
        self._bc012_release_contact_streak = {"left": 0, "right": 0}
        self._bc012_grasp_frame_armed = {"left": False, "right": False}
        self._bc012_release_pending = {"left": False, "right": False}
        self._bc012_joint_frame_alignment_error_m = {"left": 0.0, "right": 0.0}
        self._bc012_grasp_frame_staged_at: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._bc012_release_frame_staged_at: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._bc012_release_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._bc012_event_history: list[dict[str, Any]] = []
        self._bc012_grasp_attempt_count = {"left": 0, "right": 0}
        self._bc012_grasp_errors: list[str] = []
        self._bc012_owner_contact_force = {"left": 0.0, "right": 0.0}
        self._bc012_owner_contact_available = {"left": False, "right": False}
        self._bc012_owner_relative_transform: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._bc012_joint_enabled_readback = {"left": False, "right": False}
        self._bc012_owner_contact_filtered = {"left": False, "right": False}
        self._bc012_owner_continuity_ok = {"left": False, "right": False}
        self._bc012_owner_continuity_position_error_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc012_owner_continuity_angle_error_rad = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc012_min_palm_to_can_center_m = {
            "left": math.inf,
            "right": math.inf,
        }
        self._bc012_automatic_start_source = {
            "left": "unplanned",
            "right": "unplanned",
        }
        self._bc012_replay_assist_controlled = False
        self._bc012_control_frame = 0
        self._bc012_failed = False
        self._bc012_last_metrics: dict[str, Any] = {}
        kwargs.setdefault("physics_backend", "physx")
        kwargs.setdefault("physics_dt", 1.0 / B320_PHYSICS_HZ)
        kwargs.setdefault("hz", B320_CONTROL_HZ)
        kwargs.setdefault("control_dt", 1.0 / float(kwargs["hz"]))
        kwargs.setdefault("time_limit", 30.0)
        kwargs.setdefault("render_resolution", (480, 640))
        kwargs.setdefault("wrist_camera_output_size", (640, 480))
        super().__init__(task_name=task_name, **kwargs)

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "left_can_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "right_can_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "open_bin_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "partition_centers": spaces.Box(-np.inf, np.inf, shape=(2, 3)),
                "bin_load_metrics": spaces.Box(-np.inf, np.inf, shape=(12,)),
                "bc012_task_state": spaces.Box(-np.inf, np.inf, shape=(14,)),
            }
        )
        return observation_space

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        del sim_utils
        if self._stage is None:
            raise RuntimeError("BC012 asset authoring requires a USD stage")
        self._bc012_scene_manifest = spawn_bc012_assets(self._stage)

    def _configure_task_specific_pre_physx(self) -> bool:
        # BC012 is a physics-only grasp task.  In particular, do not author the
        # former can-to-palm D6 joints: the Wuji fingers and can colliders own
        # transport, and assigned-hand contact history owns event semantics.
        self._bc012_scene_manifest["grasp_mode"] = "physical_contacts"
        return False

    def _smooth_grasp_soft_hand_enabled(self) -> bool:
        """Use source Wuji drive/friction values for unassisted can grasps."""

        return bool(
            getattr(self, "_physics_backend", "") == "physx"
            and getattr(self, "hand_type", "") in {"wuji1", "wuji2"}
        )

    def _init_task_specific_runtime(self) -> None:
        self._bc012_contact_views = {name: {} for name in CAN_NAMES}
        if not self._robots or self._stage is None:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            self._add_task_state_warning("BC012 PhysX contact view is unavailable")
            return

        from pxr import Usd, UsdPhysics

        hand_paths_by_side: dict[str, list[str]] = {"left": [], "right": []}
        for side in ("left", "right"):
            root = self._stage.GetPrimAtPath(
                f"/World/HandsOnRobots/{side.capitalize()}Robot"
            )
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                name = str(prim.GetName())
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if name == f"{side}_palm_link" or name.startswith(
                    f"{side}_finger"
                ):
                    hand_paths_by_side[side].append(str(prim.GetPath()))
        hand_paths = [
            path
            for side in ("left", "right")
            for path in hand_paths_by_side[side]
        ]

        bin_path = str(self._bc012_scene_manifest.get("bin_root") or "")
        errors: list[str] = []
        for name in CAN_NAMES:
            owner_side = "left" if name == LEFT_CAN_NAME else "right"
            can_path = str(
                self._bc012_scene_manifest.get("can_roots", {}).get(name) or ""
            )
            if not can_path:
                errors.append(f"{name}: missing can root")
                continue
            try:
                support_view = physics_view.create_rigid_contact_view(
                    can_path,
                    filter_patterns=[bin_path],
                    max_contact_data_count=128,
                )
                hand_view = physics_view.create_rigid_contact_view(
                    can_path,
                    filter_patterns=hand_paths,
                    max_contact_data_count=max(128, len(hand_paths) * 8),
                )
                owner_view = physics_view.create_rigid_contact_view(
                    can_path,
                    filter_patterns=hand_paths_by_side[owner_side],
                    max_contact_data_count=max(
                        64, len(hand_paths_by_side[owner_side]) * 8
                    ),
                )
                if (
                    support_view is None
                    or int(support_view.sensor_count) < 1
                    or int(support_view.filter_count) < 1
                ):
                    errors.append(f"{name}: empty bin-support contact view")
                else:
                    self._bc012_contact_views[name]["support"] = support_view
                if (
                    hand_view is None
                    or int(hand_view.sensor_count) < 1
                    or int(hand_view.filter_count) < 1
                ):
                    errors.append(f"{name}: empty hand contact view")
                else:
                    self._bc012_contact_views[name]["hand"] = hand_view
                if (
                    owner_view is None
                    or int(owner_view.sensor_count) < 1
                    or int(owner_view.filter_count) < 1
                ):
                    errors.append(f"{name}: empty assigned-{owner_side}-hand contact view")
                else:
                    self._bc012_contact_views[name]["owner"] = owner_view
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            self._add_task_state_warning(
                "BC012 contact ownership is incomplete: " + "; ".join(errors)
            )

    def _clear_bc012_runtime_state(self) -> None:
        self._bc012_grasp_owner = {"left": False, "right": False}
        self._bc012_physical_grasp_seen = {"left": False, "right": False}
        self._bc012_grasp_contact_streak = {"left": 0, "right": 0}
        self._bc012_release_contact_streak = {"left": 0, "right": 0}
        self._bc012_grasp_frame_armed = {"left": False, "right": False}
        self._bc012_release_pending = {"left": False, "right": False}
        self._bc012_joint_frame_alignment_error_m = {"left": 0.0, "right": 0.0}
        self._bc012_grasp_frame_staged_at = {"left": None, "right": None}
        self._bc012_release_frame_staged_at = {"left": None, "right": None}
        self._bc012_release_frame = {"left": None, "right": None}
        self._bc012_event_history = []
        self._bc012_grasp_attempt_count = {"left": 0, "right": 0}
        self._bc012_grasp_errors = []
        self._bc012_owner_contact_force = {"left": 0.0, "right": 0.0}
        self._bc012_owner_contact_available = {"left": False, "right": False}
        self._bc012_owner_relative_transform = {"left": None, "right": None}
        self._bc012_joint_enabled_readback = {"left": False, "right": False}
        self._bc012_owner_contact_filtered = {"left": False, "right": False}
        self._bc012_owner_continuity_ok = {"left": False, "right": False}
        self._bc012_owner_continuity_position_error_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc012_owner_continuity_angle_error_rad = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc012_min_palm_to_can_center_m = {
            "left": math.inf,
            "right": math.inf,
        }
        self._bc012_automatic_start_source = {
            "left": "unplanned",
            "right": "unplanned",
        }
        self._bc012_control_frame = 0
        self._bc012_failed = False
        self._bc012_last_metrics = {}

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._clear_bc012_runtime_state()

    def _reset_scene_state(self) -> None:
        super()._reset_scene_state()

    @staticmethod
    def _outside_bin_footprint_margin(
        can_xy: np.ndarray,
        bin_xy: np.ndarray,
        can_radius: float,
    ) -> float:
        delta = np.abs(np.asarray(can_xy) - np.asarray(bin_xy))
        outside = np.maximum(delta - np.asarray([0.19, 0.14]) - can_radius, 0.0)
        if np.any(outside > 0.0):
            return float(np.linalg.norm(outside))
        penetration = np.asarray([0.19, 0.14]) + can_radius - delta
        return -float(np.min(penetration))

    def _bc012_layout_margin(
        self,
        layout: Mapping[str, np.ndarray],
        hand_points: Sequence[np.ndarray],
    ) -> float:
        radius = float(self._bc012_scene_manifest.get("can_collision_radius", 0.033))
        left_xy = np.asarray(layout[LEFT_CAN_NAME], dtype=np.float64)
        right_xy = np.asarray(layout[RIGHT_CAN_NAME], dtype=np.float64)
        bin_xy = np.asarray(layout[BIN_NAME], dtype=np.float64)
        margins = [
            float(np.linalg.norm(left_xy - right_xy)) - CAN_CAN_CLEARANCE,
            self._outside_bin_footprint_margin(left_xy, bin_xy, radius)
            - CAN_BIN_CLEARANCE,
            self._outside_bin_footprint_margin(right_xy, bin_xy, radius)
            - CAN_BIN_CLEARANCE,
        ]
        for can_xy in (left_xy, right_xy):
            margins.extend(
                float(np.linalg.norm(can_xy - np.asarray(point)[:2]))
                - HAND_CLEARANCE
                for point in hand_points
            )
            margins.extend(
                [
                    float(can_xy[0] - WORKSPACE_LOW_XY[0]),
                    float(WORKSPACE_HIGH_XY[0] - can_xy[0]),
                    float(can_xy[1] - WORKSPACE_LOW_XY[1]),
                    float(WORKSPACE_HIGH_XY[1] - can_xy[1]),
                ]
            )
        return float(min(margins))

    def _bc012_reset_layout(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        fallback = {
            LEFT_CAN_NAME: CAN_ANCHOR_XY[LEFT_CAN_NAME].copy(),
            RIGHT_CAN_NAME: CAN_ANCHOR_XY[RIGHT_CAN_NAME].copy(),
            BIN_NAME: BIN_ANCHOR_XY.copy(),
        }
        if not self.randomize:
            return fallback, {name: 0.0 for name in fallback}
        hand_points = (
            *self._reset_robot_body_points("left"),
            *self._reset_robot_body_points("right"),
        )
        best: dict[str, np.ndarray] | None = None
        best_margin = -math.inf
        for _ in range(RESET_MAX_ATTEMPTS):
            common = self._random.uniform(-0.010, 0.010, size=2)
            candidate = {
                LEFT_CAN_NAME: CAN_ANCHOR_XY[LEFT_CAN_NAME]
                + common
                + self._random.uniform(-CAN_XY_JITTER, CAN_XY_JITTER, size=2),
                RIGHT_CAN_NAME: CAN_ANCHOR_XY[RIGHT_CAN_NAME]
                + common
                + self._random.uniform(-CAN_XY_JITTER, CAN_XY_JITTER, size=2),
                BIN_NAME: BIN_ANCHOR_XY
                + common
                + self._random.uniform(-BIN_XY_JITTER, BIN_XY_JITTER, size=2),
            }
            margin = self._bc012_layout_margin(candidate, hand_points)
            if margin > best_margin:
                best = {name: value.copy() for name, value in candidate.items()}
                best_margin = margin
            if margin >= 0.005:
                return candidate, {
                    LEFT_CAN_NAME: float(
                        self._random.uniform(-CAN_YAW_DEGREES, CAN_YAW_DEGREES)
                    ),
                    RIGHT_CAN_NAME: float(
                        self._random.uniform(-CAN_YAW_DEGREES, CAN_YAW_DEGREES)
                    ),
                    BIN_NAME: 0.0,
                }
        if best is not None and best_margin >= 0.0:
            return best, {
                LEFT_CAN_NAME: 0.0,
                RIGHT_CAN_NAME: 0.0,
                BIN_NAME: 0.0,
            }
        fallback_margin = self._bc012_layout_margin(fallback, hand_points)
        if fallback_margin < 0.0:
            raise RuntimeError(
                f"BC012 deterministic reset fallback is not collision clear: {fallback_margin:.6f}m"
            )
        return fallback, {name: 0.0 for name in fallback}

    def _apply_task_pose_randomization(self) -> None:
        if not self._task_reset_body_states:
            return
        layout, yaw_by_name = self._bc012_reset_layout()
        reset_by_path = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity in self._task_reset_body_states
        }
        source_low = np.asarray(
            self._bc012_scene_manifest.get("can_source_low", [0.0, 0.0, -0.05]),
            dtype=np.float64,
        )
        applied: list[str] = []
        for name in (*CAN_NAMES, BIN_NAME):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                raise RuntimeError(f"BC012 reset body is unavailable: {name}")
            path = str(prim.GetPath())
            state = reset_by_path.get(path)
            if state is None:
                raise RuntimeError(f"BC012 reset truth is unavailable: {path}")
            view, indices, root_pose, root_velocity = state
            randomized = _randomized_physx_pose(
                root_pose.detach().cpu().numpy(),
                xy_offset=np.zeros(2, dtype=np.float64),
                z_offset=0.0,
                yaw_degrees=float(yaw_by_name[name]),
            )
            randomized[:, :2] = layout[name]
            randomized[:, 2] = (
                TABLE_TOP_Z + BIN_SUPPORT_CLEARANCE
                if name == BIN_NAME
                else TABLE_TOP_Z + CAN_SUPPORT_CLEARANCE - float(source_low[2])
            )
            pose_tensor = self._torch.as_tensor(
                randomized,
                dtype=root_pose.dtype,
                device=root_pose.device,
            )
            self._write_rigid_reset_state(
                path,
                view,
                indices,
                pose_tensor,
                self._torch.zeros_like(root_velocity),
            )
            applied.append(name)
        print(
            "[HandsOn/BC012] seeded reset "
            + ", ".join(
                f"{name}=({layout[name][0]:+.4f},{layout[name][1]:+.4f})/yaw={yaw_by_name[name]:+.2f}"
                for name in applied
            ),
            flush=True,
        )

    def _bc012_palm_prim_path(self, side: str) -> str | None:
        if self._stage is None:
            return None
        palm_path = str(
            self._bc012_scene_manifest.get("palm_paths", {}).get(side) or ""
        )
        if not palm_path or not self._stage.GetPrimAtPath(palm_path).IsValid():
            return None
        return palm_path

    def _set_bc012_owner_contact_filter(self, side: str, enabled: bool) -> bool:
        """Avoid an over-constrained can/owner-hand loop while joint-owned."""

        if self._stage is None:
            self._bc012_owner_contact_filtered[side] = False
            return False
        try:
            from pxr import Sdf, UsdPhysics

            can_name = LEFT_CAN_NAME if side == "left" else RIGHT_CAN_NAME
            can_prim = self._find_rigid_body_prim(can_name)
            robot_path = mjcf_robot_root_link_path(
                f"/World/HandsOnRobots/{side.capitalize()}Robot",
                side,
            )
            robot_prim = self._stage.GetPrimAtPath(robot_path)
            palm_path = self._bc012_palm_prim_path(side)
            if can_prim is None or not robot_prim.IsValid() or palm_path is None:
                self._bc012_owner_contact_filtered[side] = False
                return False
            attachment_path = Sdf.Path(palm_path).GetParentPath().AppendChild(
                f"attachment_{side}"
            )
            attachment_prim = self._stage.GetPrimAtPath(attachment_path)
            if not attachment_prim.IsValid():
                self._bc012_owner_contact_filtered[side] = False
                return False
            owner_paths = (robot_prim.GetPath(), attachment_prim.GetPath())
            filtered_pairs = UsdPhysics.FilteredPairsAPI(can_prim)
            if enabled:
                if not filtered_pairs:
                    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(can_prim)
                relation = filtered_pairs.CreateFilteredPairsRel()
                for owner_path in owner_paths:
                    relation.AddTarget(owner_path)
                self._bc012_owner_contact_filtered[side] = True
                return True
            if filtered_pairs:
                relation = filtered_pairs.GetFilteredPairsRel()
                for owner_path in owner_paths:
                    relation.RemoveTarget(owner_path)
            self._bc012_owner_contact_filtered[side] = False
            return True
        except Exception as exc:
            self._bc012_owner_contact_filtered[side] = False
            self._add_task_state_warning(
                f"Could not {'enable' if enabled else 'disable'} BC012 "
                f"{side} owner contact filter: {exc}"
            )
            return False

    @staticmethod
    def _bc012_d6_limit_attributes(joint: Any) -> list[tuple[Any, Any]]:
        """Return the six preauthored D6 low/high attribute pairs."""

        from pxr import UsdPhysics

        pairs: list[tuple[Any, Any]] = []
        for axis in GRASP_D6_AXES:
            limit = UsdPhysics.LimitAPI(joint.GetPrim(), axis)
            low = limit.GetLowAttr()
            high = limit.GetHighAttr()
            if not low.IsValid() or not high.IsValid():
                raise RuntimeError(f"preauthored D6 limit is missing for {axis}")
            pairs.append((low, high))
        return pairs

    @classmethod
    def _bc012_d6_locked(cls, joint: Any) -> bool:
        return all(
            math.isclose(
                float(low.Get()),
                -GRASP_D6_LOCK_LIMIT_BY_AXIS[axis],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                float(high.Get()),
                GRASP_D6_LOCK_LIMIT_BY_AXIS[axis],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for axis, (low, high) in zip(
                GRASP_D6_AXES,
                cls._bc012_d6_limit_attributes(joint),
            )
        )

    def _set_bc012_grasp_joint(
        self,
        side: str,
        enabled: bool,
        *,
        record_release: bool = True,
        require_owner_contact: bool = True,
        refresh_joint_frame: bool = True,
        owner_relative_transform: Sequence[Sequence[float]] | None = None,
        joint_frame_body0_world: Sequence[Sequence[float]] | None = None,
        joint_frame_body1_world: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        previous = bool(self._bc012_grasp_owner.get(side, False))
        if self._stage is None:
            self._bc012_grasp_owner[side] = False
            if enabled:
                self._record_bc012_grasp_error(side, "USD stage is unavailable")
            return False
        joint_path = str(
            self._bc012_scene_manifest.get("joint_paths", {}).get(side) or ""
        )
        if not joint_path:
            if enabled:
                self._record_bc012_grasp_error(side, "joint path is unavailable")
            return False
        try:
            from pxr import Sdf, UsdPhysics

            joint = UsdPhysics.Joint(self._stage.GetPrimAtPath(joint_path))
            if not joint:
                if enabled:
                    self._record_bc012_grasp_error(
                        side, f"D6 joint schema is invalid at {joint_path}"
                    )
                return False
            limit_attributes = self._bc012_d6_limit_attributes(joint)
            if enabled:
                if require_owner_contact and (
                    not self._bc012_owner_contact_available[side]
                    or self._bc012_owner_contact_force[side]
                    < HAND_CONTACT_FORCE_THRESHOLD
                ):
                    return False
                palm_path = self._bc012_palm_prim_path(side)
                if palm_path is None:
                    return False
                targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
                if targets != [palm_path]:
                    raise RuntimeError(
                        "BC012 preauthored grasp owner changed after first PhysX: "
                        f"expected={palm_path}, observed={targets}"
                    )
                if not self._bc012_grasp_frame_armed[side]:
                    raise RuntimeError(
                        "BC012 grasp joint cannot lock before a staged live frame"
                    )
                can_name = LEFT_CAN_NAME if side == "left" else RIGHT_CAN_NAME
                can_matrix = self._current_body_matrix(can_name)
                palm_matrix = self._robot_link_pose_matrix(side)
                if can_matrix is None or palm_matrix is None:
                    return False
                owner_relative = (
                    np.linalg.inv(palm_matrix) @ can_matrix
                    if owner_relative_transform is None
                    else np.asarray(
                        owner_relative_transform,
                        dtype=np.float64,
                    ).reshape(4, 4)
                )
                if not np.all(np.isfinite(owner_relative)):
                    raise RuntimeError("BC012 owner-relative transform is non-finite")
                # The generic D6 joint is present for the entire simulation.
                # Locking its already-authored limits avoids destroying and
                # recreating PhysX joint topology at grasp time.
                with Sdf.ChangeBlock():
                    if refresh_joint_frame and not self._stage_bc012_grasp_joint_frame(
                        side,
                        record_stage=False,
                        body0_world=joint_frame_body0_world,
                        body1_world=joint_frame_body1_world,
                    ):
                        return False
                    for axis, (low, high) in zip(
                        GRASP_D6_AXES, limit_attributes
                    ):
                        lock_limit = GRASP_D6_LOCK_LIMIT_BY_AXIS[axis]
                        low.Set(-lock_limit)
                        high.Set(lock_limit)
                enabled_readback = self._bc012_d6_locked(joint)
                self._bc012_joint_enabled_readback[side] = enabled_readback
                if not enabled_readback:
                    return False
                if not self._set_bc012_owner_contact_filter(side, True):
                    with Sdf.ChangeBlock():
                        for axis, (low, high) in zip(
                            GRASP_D6_AXES, limit_attributes
                        ):
                            free_limit = GRASP_D6_FREE_LIMIT_BY_AXIS[axis]
                            low.Set(-free_limit)
                            high.Set(free_limit)
                    self._bc012_joint_enabled_readback[side] = False
                    raise RuntimeError("owner contact filter could not be enabled")
                self._bc012_grasp_owner[side] = True
                self._bc012_owner_relative_transform[side] = owner_relative
                self._bc012_owner_continuity_ok[side] = True
                self._bc012_owner_continuity_position_error_m[side] = 0.0
                self._bc012_owner_continuity_angle_error_rad[side] = 0.0
                self._bc012_release_pending[side] = False
                if not previous:
                    self._bc012_event_history.append(
                        {
                            "frame": int(self._bc012_control_frame),
                            "event": "grasp",
                            "side": side,
                            "owner_contact_force": float(
                                self._bc012_owner_contact_force[side]
                            ),
                            "owner_contact_available": bool(
                                self._bc012_owner_contact_available[side]
                            ),
                            "joint_enabled_readback": enabled_readback,
                        }
                    )
                return True
            # Opening all six preauthored limits releases the can without
            # disabling or reconstructing the live D6 joint.
            with Sdf.ChangeBlock():
                for axis, (low, high) in zip(GRASP_D6_AXES, limit_attributes):
                    free_limit = GRASP_D6_FREE_LIMIT_BY_AXIS[axis]
                    low.Set(-free_limit)
                    high.Set(free_limit)
            enabled_readback = self._bc012_d6_locked(joint)
            self._bc012_joint_enabled_readback[side] = enabled_readback
            if enabled_readback:
                self._bc012_grasp_owner[side] = True
                self._bc012_failed = True
                self._record_bc012_grasp_error(
                    side, "D6 grasp limits remained locked after release"
                )
                return False
            if not record_release:
                self._set_bc012_owner_contact_filter(side, False)
            self._bc012_grasp_owner[side] = False
            self._bc012_owner_relative_transform[side] = None
            self._bc012_owner_continuity_ok[side] = False
            self._bc012_grasp_frame_armed[side] = False
            self._bc012_release_pending[side] = False
            if previous and record_release:
                self._bc012_release_frame[side] = int(self._bc012_control_frame)
                self._bc012_event_history.append(
                    {"frame": int(self._bc012_control_frame), "event": "release", "side": side}
                )
            return True
        except Exception as exc:
            detail = (
                f"Could not {'lock' if enabled else 'release'} BC012 "
                f"{side} D6 grasp joint: {exc}"
            )
            self._add_task_state_warning(detail)
            if enabled:
                self._record_bc012_grasp_error(side, str(exc), warn=False)
            return False

    def install_bc012_replay_assist_state(
        self,
        recorded_next_state: Sequence[float],
        *,
        recorded_can_poses: Mapping[str, Sequence[float]],
        recorded_palm_states: Mapping[str, Sequence[float]],
    ) -> int:
        del recorded_next_state, recorded_can_poses, recorded_palm_states
        raise RuntimeError(
            "BC012 grasp assist was removed; replay must reproduce physical "
            "contact events from recorded actions"
        )

    def begin_bc012_replay_assist_control(self) -> None:
        """Reject legacy replay paths that could silently restore D6 assists."""

        raise RuntimeError(
            "BC012 grasp assist was removed; replay assist control is unavailable"
        )

    @staticmethod
    def _bc012_relative_pose_error(
        reference: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        reference_matrix = np.asarray(reference, dtype=np.float64).reshape(4, 4)
        current_matrix = np.asarray(current, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(reference_matrix)) or not np.all(
            np.isfinite(current_matrix)
        ):
            return math.inf, math.inf
        position_error = float(
            np.linalg.norm(current_matrix[:3, 3] - reference_matrix[:3, 3])
        )
        rotation_delta = reference_matrix[:3, :3].T @ current_matrix[:3, :3]
        cosine = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
        return position_error, float(math.acos(cosine))

    def _bc012_owner_continuity(self, side: str, can_name: str) -> bool:
        """Verify the authored joint and live palm-to-can transform still agree."""

        reference = self._bc012_owner_relative_transform.get(side)
        can_matrix = self._current_body_matrix(can_name)
        palm_matrix = self._robot_link_pose_matrix(side)
        joint_path = str(
            self._bc012_scene_manifest.get("joint_paths", {}).get(side) or ""
        )
        enabled_readback = False
        if self._stage is not None and joint_path:
            try:
                from pxr import UsdPhysics

                joint = UsdPhysics.Joint(self._stage.GetPrimAtPath(joint_path))
                enabled_readback = bool(joint and self._bc012_d6_locked(joint))
            except Exception:
                enabled_readback = False
        self._bc012_joint_enabled_readback[side] = enabled_readback
        if reference is None or can_matrix is None or palm_matrix is None:
            self._bc012_owner_continuity_position_error_m[side] = math.inf
            self._bc012_owner_continuity_angle_error_rad[side] = math.inf
            self._bc012_owner_continuity_ok[side] = False
            return False
        current_relative = np.linalg.inv(palm_matrix) @ can_matrix
        position_error, angle_error = self._bc012_relative_pose_error(
            reference, current_relative
        )
        continuity_ok = bool(
            enabled_readback
            and position_error <= _OWNER_CONTINUITY_POSITION_TOLERANCE_M
            and angle_error <= _OWNER_CONTINUITY_ANGLE_TOLERANCE_RAD
        )
        self._bc012_owner_continuity_position_error_m[side] = position_error
        self._bc012_owner_continuity_angle_error_rad[side] = angle_error
        self._bc012_owner_continuity_ok[side] = continuity_ok
        return continuity_ok

    def _stage_bc012_grasp_joint_frame(
        self,
        side: str,
        *,
        record_stage: bool = True,
        body0_world: Sequence[Sequence[float]] | None = None,
        body1_world: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        """Write coincident live body frames while leaving the joint state unchanged."""

        if self._stage is None:
            return False
        joint_path = str(
            self._bc012_scene_manifest.get("joint_paths", {}).get(side) or ""
        )
        can_name = LEFT_CAN_NAME if side == "left" else RIGHT_CAN_NAME
        can_path = str(
            self._bc012_scene_manifest.get("can_roots", {}).get(can_name) or ""
        )
        palm_path = self._bc012_palm_prim_path(side)
        can_matrix = (
            self._current_body_matrix(can_name)
            if body0_world is None
            else np.asarray(body0_world, dtype=np.float64).reshape(4, 4)
        )
        palm_matrix = (
            self._robot_link_pose_matrix(side)
            if body1_world is None
            else np.asarray(body1_world, dtype=np.float64).reshape(4, 4)
        )
        if (
            not joint_path
            or not can_path
            or palm_path is None
            or can_matrix is None
            or palm_matrix is None
        ):
            return False
        try:
            from pxr import Gf, Sdf, UsdPhysics

            joint = UsdPhysics.Joint(self._stage.GetPrimAtPath(joint_path))
            if not joint:
                raise RuntimeError(f"D6 joint schema is invalid at {joint_path}")
            body0_targets = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            body1_targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            if body0_targets != [can_path] or body1_targets != [palm_path]:
                raise RuntimeError(
                    "BC012 preauthored grasp topology changed after first PhysX: "
                    f"body0={body0_targets}, body1={body1_targets}"
                )
            body0_frame, body1_frame = joint_local_frame_pair(
                can_matrix, palm_matrix
            )
            world0 = can_matrix @ body0_frame
            world1 = palm_matrix @ body1_frame
            alignment_error = float(
                np.linalg.norm(world0[:3, 3] - world1[:3, 3])
            )
            if alignment_error > 1.0e-6:
                raise RuntimeError(
                    "computed grasp frames are not coincident: "
                    f"{alignment_error:.9f}m"
                )
            attributes = (
                joint.GetLocalPos0Attr(),
                joint.GetLocalRot0Attr(),
                joint.GetLocalPos1Attr(),
                joint.GetLocalRot1Attr(),
            )
            if not all(attribute.IsValid() for attribute in attributes):
                raise RuntimeError("preauthored local joint-frame attributes are missing")
            body0_quaternion = matrix_to_quat_wxyz(body0_frame[:3, :3])
            body1_quaternion = matrix_to_quat_wxyz(body1_frame[:3, :3])
            # PhysX observes USD change notices.  Publish the complete pair as
            # one notice so it can never rebuild the joint from a half-updated
            # body0/body1 frame and emit a disjointed-transform warning.
            with Sdf.ChangeBlock():
                attributes[0].Set(
                    Gf.Vec3f(*(float(value) for value in body0_frame[:3, 3]))
                )
                attributes[1].Set(
                    Gf.Quatf(*(float(value) for value in body0_quaternion))
                )
                attributes[2].Set(
                    Gf.Vec3f(*(float(value) for value in body1_frame[:3, 3]))
                )
                attributes[3].Set(
                    Gf.Quatf(*(float(value) for value in body1_quaternion))
                )
            self._bc012_grasp_frame_armed[side] = True
            if record_stage:
                if self._bc012_grasp_owner[side]:
                    self._bc012_release_frame_staged_at[side] = int(
                        self._bc012_control_frame
                    )
                elif self._bc012_grasp_attempt_count[side] == 0:
                    self._bc012_grasp_frame_staged_at[side] = int(
                        self._bc012_control_frame
                    )
            self._bc012_joint_frame_alignment_error_m[side] = max(
                float(self._bc012_joint_frame_alignment_error_m[side]),
                alignment_error,
            )
            return True
        except Exception as exc:
            self._record_bc012_grasp_error(side, str(exc))
            self._bc012_grasp_frame_armed[side] = False
            return False

    def _record_bc012_grasp_error(
        self,
        side: str,
        detail: str,
        *,
        warn: bool = True,
    ) -> None:
        message = f"{side}: {detail}"
        if message not in self._bc012_grasp_errors:
            self._bc012_grasp_errors.append(message)
        if warn:
            self._add_task_state_warning(f"BC012 grasp assist unavailable: {message}")

    def _postprocess_hand_action(self) -> None:
        self._bc012_control_frame += 1
        for side, can_name in (("left", LEFT_CAN_NAME), ("right", RIGHT_CAN_NAME)):
            palm = self._robot_link_pose_matrix(side)
            can_matrix = self._current_body_matrix(can_name)
            local_center = np.asarray(
                self._bc012_scene_manifest.get("can_local_center", [0.0, 0.0, 0.05]),
                dtype=np.float64,
            )
            if palm is None or can_matrix is None:
                continue
            can_center = (can_matrix @ np.asarray([*local_center, 1.0]))[:3]
            distance = float(np.linalg.norm(palm[:3, 3] - can_center))
            owner_force, owner_available = self._bc012_contact_force(
                can_name, "owner"
            )
            self._bc012_owner_contact_force[side] = owner_force
            self._bc012_owner_contact_available[side] = owner_available
            self._bc012_min_palm_to_can_center_m[side] = min(
                float(self._bc012_min_palm_to_can_center_m[side]),
                distance,
            )

            contact_ready = bool(
                owner_available and owner_force >= HAND_CONTACT_FORCE_THRESHOLD
            )
            source_low = np.asarray(
                self._bc012_scene_manifest.get("can_source_low", [0.0, 0.0, -0.05]),
                dtype=np.float64,
            )
            rest_center_z = float(
                TABLE_TOP_Z
                + CAN_SUPPORT_CLEARANCE
                - source_low[2]
                + local_center[2]
            )
            lifted = bool(
                float(can_center[2])
                >= rest_center_z + _PHYSICAL_GRASP_MIN_LIFT_M
            )

            if not self._bc012_physical_grasp_seen[side]:
                if not (contact_ready and lifted):
                    self._bc012_grasp_contact_streak[side] = 0
                    self._bc012_grasp_frame_armed[side] = False
                    self._bc012_grasp_frame_staged_at[side] = None
                    continue
                self._bc012_grasp_contact_streak[side] += 1
                self._bc012_grasp_frame_armed[side] = True
                if self._bc012_grasp_frame_staged_at[side] is None:
                    self._bc012_grasp_frame_staged_at[side] = int(
                        self._bc012_control_frame
                    )
                if (
                    self._bc012_grasp_contact_streak[side]
                    < _PHYSICAL_GRASP_CONFIRM_FRAMES
                ):
                    continue
                self._bc012_physical_grasp_seen[side] = True
                self._bc012_grasp_attempt_count[side] += 1
                self._bc012_grasp_frame_armed[side] = False
                grasp_frame = int(
                    self._bc012_grasp_frame_staged_at[side]
                    if self._bc012_grasp_frame_staged_at[side] is not None
                    else self._bc012_control_frame
                )
                self._bc012_event_history.append(
                    {
                        "frame": grasp_frame,
                        "observed_frame": int(self._bc012_control_frame),
                        "event": "grasp",
                        "side": side,
                        "source": "assigned_hand_contact",
                        "contact_force": float(owner_force),
                        "lifted": True,
                    }
                )
                continue

            if self._bc012_release_frame[side] is not None:
                continue
            if contact_ready:
                self._bc012_release_contact_streak[side] = 0
                self._bc012_release_pending[side] = False
                self._bc012_release_frame_staged_at[side] = None
                continue

            self._bc012_release_contact_streak[side] += 1
            self._bc012_release_pending[side] = True
            if self._bc012_release_frame_staged_at[side] is None:
                self._bc012_release_frame_staged_at[side] = int(
                    self._bc012_control_frame
                )
            side_points = self._a06_hand_points(side)
            side_hand_clear = bool(
                side_points
                and min(
                    float(np.linalg.norm(np.asarray(point) - can_center))
                    for point in side_points
                )
                >= HAND_CLEAR_DISTANCE
            )
            if (
                self._bc012_release_contact_streak[side]
                < _PHYSICAL_RELEASE_CONFIRM_FRAMES
                or not side_hand_clear
            ):
                continue
            release_frame = int(
                self._bc012_release_frame_staged_at[side]
                if self._bc012_release_frame_staged_at[side] is not None
                else self._bc012_control_frame
            )
            self._bc012_release_frame[side] = release_frame
            self._bc012_release_pending[side] = False
            self._bc012_event_history.append(
                {
                    "frame": release_frame,
                    "observed_frame": int(self._bc012_control_frame),
                    "event": "release",
                    "side": side,
                    "source": "assigned_hand_contact_loss",
                }
            )

    def _bc012_contact_force(self, name: str, role: str) -> tuple[float, bool]:
        view = self._bc012_contact_views.get(name, {}).get(role)
        if view is None:
            return 0.0, False
        return float(self._c04_contact_force(view)), True

    def _bc012_hand_clearance(self, can_pose: np.ndarray) -> tuple[float, bool]:
        can_matrix = pose_matrix(can_pose)
        local_center = np.asarray(
            self._bc012_scene_manifest.get("can_local_center", [0.0, 0.0, 0.05]),
            dtype=np.float64,
        )
        center = (can_matrix @ np.asarray([*local_center, 1.0]))[:3]
        distances = [
            float(np.linalg.norm(np.asarray(point) - center))
            for side in ("left", "right")
            for point in self._a06_hand_points(side)
        ]
        # A finite fail-closed sentinel keeps interface-only observations and
        # JSON/Zarr episode fields valid while still exceeding every clearance
        # threshold.  Runtime success separately requires contact availability.
        minimum = min(distances, default=1.0e6)
        return minimum, bool(minimum >= HAND_CLEAR_DISTANCE)

    def _bc012_metrics(self) -> dict[str, Any]:
        left_pose = self._poses.get(LEFT_CAN_NAME, self._fallback_pose(LEFT_CAN_NAME))
        right_pose = self._poses.get(RIGHT_CAN_NAME, self._fallback_pose(RIGHT_CAN_NAME))
        bin_pose = self._poses.get(BIN_NAME, self._fallback_pose(BIN_NAME))
        left_velocity = self._physics_body_velocity(LEFT_CAN_NAME)
        right_velocity = self._physics_body_velocity(RIGHT_CAN_NAME)
        left_support_force, left_support_available = self._bc012_contact_force(
            LEFT_CAN_NAME, "support"
        )
        right_support_force, right_support_available = self._bc012_contact_force(
            RIGHT_CAN_NAME, "support"
        )
        left_hand_force, left_hand_available = self._bc012_contact_force(
            LEFT_CAN_NAME, "hand"
        )
        right_hand_force, right_hand_available = self._bc012_contact_force(
            RIGHT_CAN_NAME, "hand"
        )
        left_hand_distance, left_hand_clear = self._bc012_hand_clearance(left_pose)
        right_hand_distance, right_hand_clear = self._bc012_hand_clearance(right_pose)
        metrics = evaluate_bin_load(
            left_pose,
            right_pose,
            bin_pose,
            left_velocity=(left_velocity if left_velocity is not None else (1.0e6,) * 6),
            right_velocity=(right_velocity if right_velocity is not None else (1.0e6,) * 6),
            can_local_center=self._bc012_scene_manifest.get("can_local_center", [0.0, 0.0, 0.05]),
            left_supported=left_support_available and left_support_force > 1.0e-4,
            right_supported=right_support_available and right_support_force > 1.0e-4,
            left_hand_clear=left_hand_available and left_hand_clear,
            right_hand_clear=right_hand_available and right_hand_clear,
            left_hand_contact_force=(left_hand_force if left_hand_available else math.inf),
            right_hand_contact_force=(right_hand_force if right_hand_available else math.inf),
            left_owned=False,
            right_owned=False,
            left_release_frame=self._bc012_release_frame["left"],
            right_release_frame=self._bc012_release_frame["right"],
            control_hz=1.0 / max(float(self._control_dt), 1.0e-9),
        )
        metrics.update(
            {
                "left_support_force": left_support_force,
                "right_support_force": right_support_force,
                "left_support_available": left_support_available,
                "right_support_available": right_support_available,
                "left_hand_distance": left_hand_distance,
                "right_hand_distance": right_hand_distance,
                "left_hand_contact_available": left_hand_available,
                "right_hand_contact_available": right_hand_available,
            }
        )
        self._bc012_last_metrics = metrics
        return metrics

    def _compute_success_now(self) -> bool:
        return bool(self._bc012_metrics()["success_now"])

    def _compute_failure(self) -> bool:
        left = self._poses.get(LEFT_CAN_NAME, self._fallback_pose(LEFT_CAN_NAME))
        right = self._poses.get(RIGHT_CAN_NAME, self._fallback_pose(RIGHT_CAN_NAME))
        bin_pose = self._poses.get(BIN_NAME, self._fallback_pose(BIN_NAME))
        dropped = bool(
            float(left[2]) < TABLE_TOP_Z - 0.08
            or float(right[2]) < TABLE_TOP_Z - 0.08
        )
        workspace_exit = bool(
            np.any(left[:2] < WORKSPACE_LOW_XY - 0.10)
            or np.any(left[:2] > WORKSPACE_HIGH_XY + 0.10)
            or np.any(right[:2] < WORKSPACE_LOW_XY - 0.10)
            or np.any(right[:2] > WORKSPACE_HIGH_XY + 0.10)
        )
        bin_inverse = np.linalg.inv(pose_matrix(bin_pose))
        local_center = np.asarray(
            self._bc012_scene_manifest.get("can_local_center", [0.0, 0.0, 0.05]),
            dtype=np.float64,
        )
        left_local = (
            bin_inverse @ pose_matrix(left) @ np.asarray([*local_center, 1.0])
        )[:3]
        right_local = (
            bin_inverse @ pose_matrix(right) @ np.asarray([*local_center, 1.0])
        )[:3]
        velocities = (
            self._physics_body_velocity(LEFT_CAN_NAME),
            self._physics_body_velocity(RIGHT_CAN_NAME),
        )
        settled = all(
            velocity is not None
            and float(np.linalg.norm(velocity[:3])) <= 0.04
            and float(np.linalg.norm(velocity[3:])) <= 0.50
            for velocity in velocities
        )
        released = all(
            self._bc012_release_frame[side] is not None
            for side in ("left", "right")
        )
        same_side_placement = bool(
            released
            and settled
            and 0.044 <= float(left_local[1]) <= 0.086
            and -0.086 <= float(right_local[1]) <= -0.044
        )
        self._bc012_failed = bool(
            self._bc012_failed
            or dropped
            or workspace_exit
            or same_side_placement
        )
        return self._bc012_failed

    def _task_info(self) -> dict[str, Any]:
        metrics = self._bc012_last_metrics or self._bc012_metrics()
        release_delta_value = float(metrics["release_delta_frames"])
        release_delta = (
            int(release_delta_value)
            if math.isfinite(release_delta_value)
            else int(metrics["sync_window_frames"]) + 1
        )
        palm_distances = {
            side: (
                float(distance)
                if math.isfinite(float(distance))
                else 1.0e6
            )
            for side, distance in self._bc012_min_palm_to_can_center_m.items()
        }
        return {
            "bc012_success_now": bool(metrics["success_now"]),
            "bc012_synchronized": bool(metrics["synchronized"]),
            "bc012_release_delta_frames": release_delta,
            "bc012_sync_window_frames": int(metrics["sync_window_frames"]),
            "bc012_left_ready": bool(metrics["left"]["ready"]),
            "bc012_right_ready": bool(metrics["right"]["ready"]),
            "bc012_left_supported": bool(metrics["left"]["supported"]),
            "bc012_right_supported": bool(metrics["right"]["supported"]),
            "bc012_grasp_mode": "physical_contacts",
            "bc012_grasp_assist_enabled": False,
            "bc012_left_owned": False,
            "bc012_right_owned": False,
            "bc012_physical_grasp_seen": dict(
                self._bc012_physical_grasp_seen
            ),
            "bc012_grasp_contact_streak": dict(
                self._bc012_grasp_contact_streak
            ),
            "bc012_release_contact_streak": dict(
                self._bc012_release_contact_streak
            ),
            "bc012_grasp_frame_armed": dict(self._bc012_grasp_frame_armed),
            "bc012_release_pending": dict(self._bc012_release_pending),
            "bc012_joint_frame_alignment_error_m": dict(
                self._bc012_joint_frame_alignment_error_m
            ),
            "bc012_grasp_frame_staged_at": dict(
                self._bc012_grasp_frame_staged_at
            ),
            "bc012_release_frame_staged_at": dict(
                self._bc012_release_frame_staged_at
            ),
            "bc012_success_counter": int(self._success_counter),
            "bc012_success_steps_required": int(STABILITY_STEPS),
            "bc012_failed": bool(self._bc012_failed),
            "bc012_event_history": tuple(dict(event) for event in self._bc012_event_history),
            "bc012_grasp_attempt_count": dict(self._bc012_grasp_attempt_count),
            "bc012_grasp_errors": tuple(self._bc012_grasp_errors),
            "bc012_owner_contact_force": dict(self._bc012_owner_contact_force),
            "bc012_owner_contact_available": dict(
                self._bc012_owner_contact_available
            ),
            "bc012_joint_enabled_readback": dict(
                self._bc012_joint_enabled_readback
            ),
            "bc012_owner_continuity_ok": dict(self._bc012_owner_continuity_ok),
            "bc012_owner_continuity_position_error_m": {
                side: float(value) if math.isfinite(float(value)) else 1.0e6
                for side, value in self._bc012_owner_continuity_position_error_m.items()
            },
            "bc012_owner_continuity_angle_error_rad": {
                side: float(value) if math.isfinite(float(value)) else 1.0e6
                for side, value in self._bc012_owner_continuity_angle_error_rad.items()
            },
            "bc012_min_palm_to_can_center_m": palm_distances,
            "bc012_automatic_start_source": dict(
                self._bc012_automatic_start_source
            ),
        }

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._bc012_last_metrics or self._bc012_metrics()
        rows = [
            self._condition_row("bc012_left_ready", "左罐在指定分区、支撑稳定且已脱手", bool(metrics["left"]["ready"])),
            self._condition_row("bc012_right_ready", "右罐在指定分区、支撑稳定且已脱手", bool(metrics["right"]["ready"])),
            self._condition_row("bc012_synchronized", "双罐在同步窗口内释放", bool(metrics["synchronized"])),
        ]
        return self._append_success_hold_row(rows)

    def _task_status_text(self) -> str:
        metrics = self._bc012_last_metrics or self._bc012_metrics()
        ready = int(bool(metrics["left"]["ready"])) + int(
            bool(metrics["right"]["ready"])
        )
        return f"已放稳 {ready}/2  |  稳定 {self._success_counter}/{STABILITY_STEPS} 帧"

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        state = observation["state"]
        left_pose = self._poses.get(LEFT_CAN_NAME, self._fallback_pose(LEFT_CAN_NAME)).copy()
        right_pose = self._poses.get(RIGHT_CAN_NAME, self._fallback_pose(RIGHT_CAN_NAME)).copy()
        bin_pose = self._poses.get(BIN_NAME, self._fallback_pose(BIN_NAME)).copy()
        metrics = self._bc012_last_metrics or self._bc012_metrics()
        state.update(
            {
                "left_can_pose": left_pose,
                "right_can_pose": right_pose,
                "open_bin_pose": bin_pose,
                "partition_centers": partition_centers_world(bin_pose),
                "bin_load_metrics": np.asarray(metrics["metrics_vector"], dtype=np.float64),
                "bc012_task_state": np.asarray(
                    [
                        float(self._bc012_physical_grasp_seen["left"]),
                        float(self._bc012_physical_grasp_seen["right"]),
                        float(self._bc012_grasp_contact_streak["left"]),
                        float(self._bc012_grasp_contact_streak["right"]),
                        float(self._bc012_release_contact_streak["left"]),
                        float(self._bc012_release_contact_streak["right"]),
                        float(self._bc012_release_frame["left"] if self._bc012_release_frame["left"] is not None else -1),
                        float(self._bc012_release_frame["right"] if self._bc012_release_frame["right"] is not None else -1),
                        float(self._success_counter),
                        float(self._bc012_failed),
                        float(metrics["left"]["supported"]),
                        float(metrics["right"]["supported"]),
                        float(metrics["left"]["hand_contact_force"])
                        if bool(metrics["left_hand_contact_available"])
                        else -1.0,
                        float(metrics["right"]["hand_contact_force"])
                        if bool(metrics["right_hand_contact_available"])
                        else -1.0,
                    ],
                    dtype=np.float64,
                ),
            }
        )
        return observation

    @staticmethod
    def _bc012_action(
        right_pose: np.ndarray,
        left_pose: np.ndarray,
        right_hand: np.ndarray,
        left_hand: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [right_pose, left_pose, right_hand, left_hand]
        ).astype(np.float32)

    @staticmethod
    def _bc012_palm_pose_for_can_center(
        palm_template: np.ndarray,
        can_center_world: np.ndarray,
    ) -> np.ndarray:
        """Place a can among the fingers instead of against the wrist flange."""

        result = np.asarray(palm_template, dtype=np.float64).reshape(7).copy()
        palm_rotation = pose_matrix(result)[:3, :3]
        result[:3] = (
            np.asarray(can_center_world, dtype=np.float64).reshape(3)
            - palm_rotation @ _CAN_CENTER_IN_POWER_GRIP_PALM_M
        )
        return result

    @staticmethod
    def _interpolate_pose(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
        result = np.asarray(start, dtype=np.float64).copy()
        result[:3] = (1.0 - fraction) * start[:3] + fraction * target[:3]
        result[3:7] = target[3:7]
        return result

    def _differential_ik_command_type(self, side: str) -> str:
        """Keep full pose control available for the initial grasp."""

        del side
        return "pose"

    def _bc012_live_palm_action_pose(
        self,
        side: str,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Return the measured PhysX palm pose used to seed task waypoints."""

        matrix = self._robot_link_pose_matrix(side)
        if matrix is None:
            self._bc012_automatic_start_source[side] = "command_fallback"
            return np.asarray(fallback, dtype=np.float64).reshape(7).copy()
        pose = np.empty(7, dtype=np.float64)
        pose[:3] = np.asarray(matrix[:3, 3], dtype=np.float64)
        pose[3:7] = _matrix_to_quat_wxyz(matrix[:3, :3])
        self._bc012_automatic_start_source[side] = "physx_palm_link"
        return pose

    def automatic_success_actions(
        self,
        observation: Mapping[str, Any] | None = None,
    ) -> list[np.ndarray]:
        """Build a deterministic action-only bimanual success trajectory."""

        if observation is None:
            observation = self._compute_observation()
        state = observation["state"]
        tcp_command = np.asarray(state["tcp_pose"], dtype=np.float64).reshape(14)
        right_start = self._bc012_live_palm_action_pose(
            "right", tcp_command[:7]
        )
        left_start = self._bc012_live_palm_action_pose(
            "left", tcp_command[7:14]
        )
        right_can = np.asarray(state["right_can_pose"], dtype=np.float64)
        left_can = np.asarray(state["left_can_pose"], dtype=np.float64)
        bin_pose = np.asarray(state["open_bin_pose"], dtype=np.float64)
        local_center = np.asarray(
            self._bc012_scene_manifest.get("can_local_center", [0.0, 0.0, 0.05]),
            dtype=np.float64,
        )
        right_center = (pose_matrix(right_can) @ np.asarray([*local_center, 1.0]))[:3]
        left_center = (pose_matrix(left_can) @ np.asarray([*local_center, 1.0]))[:3]
        right_grasp = self._bc012_palm_pose_for_can_center(
            right_start, right_center
        )
        left_grasp = self._bc012_palm_pose_for_can_center(
            left_start, left_center
        )
        right_above = right_grasp.copy()
        right_above[2] += 0.075
        left_above = left_grasp.copy()
        left_above[2] += 0.075

        bin_matrix = pose_matrix(bin_pose)
        bin_inverse = np.linalg.inv(bin_matrix)

        def bin_point(local_xyz: Sequence[float]) -> np.ndarray:
            return (
                bin_matrix
                @ np.asarray([*np.asarray(local_xyz, dtype=np.float64), 1.0])
            )[:3]

        right_center_local = (
            bin_inverse @ np.asarray([*right_center, 1.0], dtype=np.float64)
        )[:3]
        left_center_local = (
            bin_inverse @ np.asarray([*left_center, 1.0], dtype=np.float64)
        )[:3]
        # First move outward of the bin's -X wall while still on each source
        # side.  Keep the can bottoms above the 0.15 m wall while crossing and
        # place the right can on a higher layer.  The vertical separation lets
        # the left can cross second without colliding with it, while remaining
        # below the shoulder-saturating route previously tried over the bin.
        right_lift_center = bin_point(
            [
                _OUTSIDE_CROSS_X_LOCAL_M,
                right_center_local[1],
                _RIGHT_OUTSIDE_CROSS_Z_LOCAL_M,
            ]
        )
        left_lift_center = bin_point(
            [
                _OUTSIDE_CROSS_X_LOCAL_M,
                left_center_local[1],
                _LEFT_OUTSIDE_CROSS_Z_LOCAL_M,
            ]
        )
        right_lift = self._bc012_palm_pose_for_can_center(
            right_grasp, right_lift_center
        )
        left_lift = self._bc012_palm_pose_for_can_center(
            left_grasp, left_lift_center
        )
        right_cross = self._bc012_palm_pose_for_can_center(
            right_lift,
            bin_point(
                [
                    _OUTSIDE_CROSS_X_LOCAL_M,
                    _OUTSIDE_CROSS_Y_LOCAL_M,
                    _RIGHT_OUTSIDE_CROSS_Z_LOCAL_M,
                ]
            ),
        )
        left_cross = self._bc012_palm_pose_for_can_center(
            left_lift,
            bin_point(
                [
                    _OUTSIDE_CROSS_X_LOCAL_M,
                    -_OUTSIDE_CROSS_Y_LOCAL_M,
                    _LEFT_OUTSIDE_CROSS_Z_LOCAL_M,
                ]
            ),
        )
        right_partition_local = PARTITION_CENTERS_LOCAL[
            TARGET_PARTITION_INDEX_BY_SIDE["right"]
        ]
        right_staged_over = self._bc012_palm_pose_for_can_center(
            right_cross,
            bin_point(
                [
                    _RIGHT_STAGED_CLEAR_X_LOCAL_M,
                    right_partition_local[1],
                    _RIGHT_OUTSIDE_CROSS_Z_LOCAL_M,
                ]
            ),
        )
        left_partition_local = PARTITION_CENTERS_LOCAL[
            TARGET_PARTITION_INDEX_BY_SIDE["left"]
        ]
        # The crossed arm chains have a repeatable outward reach bias near the
        # bin.  Command safe points nearer the divider and separate them in X;
        # success remains evaluated against the physical partition zones.
        right_goal = bin_point(
            [
                _PLACEMENT_COMMAND_X_LOCAL_M,
                _PLACEMENT_COMMAND_Y_LOCAL_M,
                right_partition_local[2],
            ]
        )
        left_goal = bin_point(
            [
                -_PLACEMENT_COMMAND_X_LOCAL_M,
                -_PLACEMENT_COMMAND_Y_LOCAL_M,
                left_partition_local[2],
            ]
        )
        # Translation-only IK deliberately releases wrist orientation during
        # the crossed reach.  Compensate the resulting measured palm-to-can
        # rotation at the action level so the physical can centers, rather
        # than the unconstrained palms, remain over their safe goal points.
        right_palm_compensation = (
            bin_matrix[:3, :3] @ _PLACEMENT_PALM_COMPENSATION_LOCAL_M["right"]
        )
        left_palm_compensation = (
            bin_matrix[:3, :3] @ _PLACEMENT_PALM_COMPENSATION_LOCAL_M["left"]
        )
        right_over = self._bc012_palm_pose_for_can_center(
            right_staged_over,
            right_goal + np.asarray([0.0, 0.0, _BIN_ENTRY_Z_LOCAL_M]),
        )
        right_over[:3] += right_palm_compensation
        left_over = self._bc012_palm_pose_for_can_center(
            left_cross,
            left_goal + np.asarray([0.0, 0.0, _BIN_ENTRY_Z_LOCAL_M]),
        )
        left_over[:3] += left_palm_compensation
        right_release = self._bc012_palm_pose_for_can_center(
            right_over,
            right_goal + np.asarray([0.0, 0.0, _RELEASE_DROP_HEIGHT_M]),
        )
        right_release[:3] += right_palm_compensation
        left_release = self._bc012_palm_pose_for_can_center(
            left_over,
            left_goal + np.asarray([0.0, 0.0, _RELEASE_DROP_HEIGHT_M]),
        )
        left_release[:3] += left_palm_compensation
        right_clear = right_release.copy()
        right_clear[:3] += np.asarray([0.0, 0.18, 0.18])
        left_clear = left_release.copy()
        left_clear[:3] += np.asarray([0.0, -0.18, 0.18])

        open_hand = np.zeros(20, dtype=np.float64)
        closed_hand = _C_TOOL_WUJI_POWER_GRIP.copy()
        pre_release_hand = closed_hand * (
            _PRE_RELEASE_HAND_MAX / float(np.max(np.abs(closed_hand)))
        )
        actions: list[np.ndarray] = []

        def hold(
            right_pose: np.ndarray,
            left_pose: np.ndarray,
            right_hand: np.ndarray,
            left_hand: np.ndarray,
            frames: int,
        ) -> None:
            action = self._bc012_action(right_pose, left_pose, right_hand, left_hand)
            actions.extend(action.copy() for _ in range(frames))

        def move(
            right_from: np.ndarray,
            right_to: np.ndarray,
            left_from: np.ndarray,
            left_to: np.ndarray,
            right_hand: np.ndarray,
            left_hand: np.ndarray,
            frames: int,
        ) -> None:
            for index in range(1, frames + 1):
                fraction = index / frames
                actions.append(
                    self._bc012_action(
                        self._interpolate_pose(right_from, right_to, fraction),
                        self._interpolate_pose(left_from, left_to, fraction),
                        right_hand,
                        left_hand,
                    )
                )

        hold(right_start, left_start, open_hand, open_hand, 15)
        move(right_start, right_above, left_start, left_above, open_hand, open_hand, 30)
        move(right_above, right_grasp, left_above, left_grasp, open_hand, open_hand, 20)
        hold(right_grasp, left_grasp, closed_hand, closed_hand, 24)
        move(right_grasp, right_lift, left_grasp, left_lift, closed_hand, closed_hand, 30)
        # Cross the right can first, then move it above its partition so the
        # outside lane is physically empty before the left hand enters it.
        move(right_lift, right_cross, left_lift, left_lift, closed_hand, closed_hand, 50)
        move(right_cross, right_staged_over, left_lift, left_lift, closed_hand, closed_hand, 30)
        move(right_staged_over, right_staged_over, left_lift, left_cross, closed_hand, closed_hand, 50)
        hold(right_staged_over, left_cross, closed_hand, closed_hand, 20)
        # Once both paths have crossed, settle one arm at a time.  Driving the
        # two crossed chains inward together leaves both wrists at their
        # kinematic envelope; each can remains owned until the shared release.
        move(right_staged_over, right_staged_over, left_cross, left_over, closed_hand, closed_hand, 60)
        hold(right_staged_over, left_over, closed_hand, closed_hand, 30)
        move(right_staged_over, right_staged_over, left_over, left_release, closed_hand, closed_hand, 50)
        hold(right_staged_over, left_release, closed_hand, closed_hand, 20)
        move(right_staged_over, right_over, left_release, left_release, closed_hand, closed_hand, 60)
        hold(right_over, left_release, closed_hand, closed_hand, 30)
        move(right_over, right_release, left_release, left_release, closed_hand, closed_hand, 50)
        hold(right_release, left_release, closed_hand, closed_hand, 20)
        # Relax both physical grips together before opening, then move both
        # hands clear so contact-loss release events share the same window.
        hold(right_release, left_release, pre_release_hand, pre_release_hand, 30)
        hold(right_release, left_release, open_hand, open_hand, 3)
        move(right_release, right_clear, left_release, left_clear, open_hand, open_hand, 20)
        hold(right_clear, left_clear, open_hand, open_hand, 90)
        return actions


__all__ = ["BC012IsaacLabEnv"]
