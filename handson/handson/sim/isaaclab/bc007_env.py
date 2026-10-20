"""Native Isaac Sim 5.1/PhysX runtime for B320-BC007."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .bc007_contract import (
    LongRodEvaluator,
    LongRodFrame,
    LongRodStage,
    decode_replay_assist_state,
    encode_replay_assist_state,
    rod_axis_yaw_deg,
    rod_endpoints,
    signed_yaw_progress_deg,
    wrap_degrees,
)
from .bc007_scene import (
    GOAL_NAME,
    HEIGHT_BAND_HIGH_M,
    HEIGHT_BAND_LOW_M,
    LEFT_ENDPOINT_LOCAL_SIGN,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_SPEED_MPS,
    MINIMUM_HAND_SEPARATION_M,
    RESET_XY_JITTER_M,
    RESET_YAW_JITTER_DEG,
    RIGHT_ENDPOINT_LOCAL_SIGN,
    ROD_HALF_LENGTH_M,
    ROD_NAME,
    ROD_RADIUS_M,
    ROD_START_CENTER,
    ROD_START_ORIENTATION_WXYZ,
    ROD_TARGET_CENTER_OFFSET,
    STABILITY_STEPS,
    TABLE_TOP_Z,
    TARGET_POSITION_TOLERANCE_M,
    TARGET_YAW_DELTA_DEG,
    TARGET_YAW_TOLERANCE_DEG,
    TASK_ID,
    TASK_NAME,
    bind_bc007_grasp_joint_owners,
    initial_orientation_wxyz,
    joint_local_frame_pair,
    matrix_to_quat_wxyz,
    set_goal_marker_pose,
    spawn_bc007_assets,
    target_orientation_wxyz,
)
from .priority_zero_robot_audit import initial_robot_state_report


_HAND_CONTACT_FORCE_N = 0.15
_END_REGION_INSET_M = 0.055
_END_REGION_DISTANCE_M = 0.090
_ROLE_WRONG_END_MIN_DISTANCE_M = 0.18
_STRONG_HAND_CONTACT_FORCE_N = 120.0
_TABLE_CONTACT_FORCE_N = 0.05
_TABLE_SUPPORT_FORCE_N = 0.01
_STRONG_TABLE_CONTACT_FORCE_N = 40.0
_GRASP_ENABLE_COMMAND = 0.55
_GRASP_RELEASE_COMMAND = 0.20
_GRASP_PALM_TO_END_MAX_M = 0.115
_OWNER_CONTINUITY_POSITION_TOLERANCE_M = 0.020
_OWNER_CONTINUITY_ANGLE_TOLERANCE_RAD = math.radians(12.0)
_ARM_TARGET_LIMIT_MARGIN_RAD = 0.140
_WORKSPACE_LOWER = np.asarray([-0.70, -0.48, 0.84], dtype=np.float64)
_WORKSPACE_UPPER = np.asarray([0.12, 0.48, 1.30], dtype=np.float64)


def _point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= np.finfo(np.float64).eps:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / denominator)
    closest = start + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - closest))


def _role_segment(
    center: np.ndarray, endpoint: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    axis = endpoint - center
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return endpoint.copy(), endpoint.copy()
    direction = axis / norm
    return endpoint - direction * _END_REGION_INSET_M, endpoint


class BC007IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Two assigned hands lift a long rod and yaw it by 90 degrees."""

    def __init__(self, **kwargs: Any):
        self._bc007_manifest: dict[str, Any] = {}
        self._bc007_evaluator = LongRodEvaluator()
        self._bc007_last_metrics: dict[str, Any] = {}
        self._bc007_hand_contact_views: dict[str, Any] = {}
        self._bc007_table_contact_view: Any | None = None
        self._bc007_contact_errors: set[str] = set()
        self._bc007_grasp_owner = {"left": False, "right": False}
        self._bc007_grasp_frame_armed = {"left": False, "right": False}
        self._bc007_release_pending = {"left": False, "right": False}
        self._bc007_joint_enabled_readback = {"left": False, "right": False}
        self._bc007_owner_relative_transform: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._bc007_owner_continuity_ok = {"left": False, "right": False}
        self._bc007_owner_continuity_position_error_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc007_owner_continuity_angle_error_rad = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc007_joint_frame_alignment_error_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc007_grasp_errors: list[str] = []
        self._bc007_grasp_events: list[dict[str, Any]] = []
        self._bc007_control_frame = 0
        # The production runtime leaves this unset.  The live P0 acceptance
        # executable installs a RobotStabilityAudit here so its samples are
        # taken after every real 120 Hz PhysX step, not four copies of the
        # final 30 Hz control-frame readback.
        self._bc007_per_physics_robot_audit: Any | None = None
        self._bc007_action_start_assist_state = encode_replay_assist_state(
            left_owner=False,
            right_owner=False,
            left_frame_armed=False,
            right_frame_armed=False,
            left_release_pending=False,
            right_release_pending=False,
        )
        self._bc007_initial_center = ROD_START_CENTER.copy()
        self._bc007_initial_orientation = ROD_START_ORIENTATION_WXYZ.copy()
        self._bc007_target_center = ROD_START_CENTER + ROD_TARGET_CENTER_OFFSET
        self._bc007_target_orientation = target_orientation_wxyz(
            ROD_START_ORIENTATION_WXYZ
        )
        self._bc007_initial_axis_yaw_deg = rod_axis_yaw_deg(
            [*ROD_START_CENTER, *ROD_START_ORIENTATION_WXYZ]
        )
        self._bc007_target_axis_yaw_deg = wrap_degrees(
            self._bc007_initial_axis_yaw_deg + TARGET_YAW_DELTA_DEG
        )
        kwargs.setdefault("physics_backend", "physx")
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / float(kwargs["hz"]))
        kwargs.setdefault("time_limit", 45.0)
        kwargs.setdefault("render_resolution", (480, 640))
        kwargs.setdefault("wrist_camera_output_size", (640, 480))
        super().__init__(TASK_NAME, **kwargs)
        # BC007 always uses three RTX recording cameras.  The regular Kit
        # cleanup path can block after capture; the shared skip-cleanup close
        # path is process-local and lets the official recorder exit promptly.
        self._fast_shutdown_on_close = True

    def _ready_state_settle_physics_steps(self) -> int:
        """Settle the awake rod for one real second before ready-state checks."""

        return max(5, int(round(1.0 / float(self._physics_dt))))

    def _compute_arm_joint_target(
        self,
        side: str,
        robot: Any,
        pose: np.ndarray,
        commanded: bool,
    ) -> tuple[Any, str]:
        """Keep BC007 Cartesian teleoperation away from arm soft limits."""

        arm_target, status = super()._compute_arm_joint_target(
            side, robot, pose, commanded
        )
        if arm_target is None:
            return arm_target, status
        limits = robot.data.soft_joint_pos_limits[:, self._arm_joint_ids[side], :]
        lower = limits[:, :, 0] + _ARM_TARGET_LIMIT_MARGIN_RAD
        upper = limits[:, :, 1] - _ARM_TARGET_LIMIT_MARGIN_RAD
        arm_target = self._torch.maximum(arm_target, lower)
        arm_target = self._torch.minimum(arm_target, upper)
        return arm_target, f"{status}_bc007_safe_margin"

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        del sim_utils
        if self._stage is None:
            raise RuntimeError("BC007 asset authoring requires a USD stage")
        self._bc007_manifest = spawn_bc007_assets(self._stage)

    def _configure_task_specific_pre_physx(self) -> bool:
        if self._stage is None:
            raise RuntimeError("BC007 grasp topology requires a USD stage")
        palm_paths = bind_bc007_grasp_joint_owners(
            self._stage,
            self._bc007_manifest.get("joint_paths", {}),
            str(self._bc007_manifest.get("rod_root") or ""),
        )
        self._bc007_manifest["palm_paths"] = palm_paths
        return bool(palm_paths)

    def _task_extra_pose_names(self) -> set[str]:
        extras = set(super()._task_extra_pose_names())
        extras.add(GOAL_NAME)
        return extras

    def _fallback_pose(self, name: str) -> np.ndarray:
        if name == ROD_NAME:
            return np.asarray(
                [*ROD_START_CENTER, *ROD_START_ORIENTATION_WXYZ], dtype=np.float64
            )
        if name == GOAL_NAME:
            return np.asarray(
                [*self._bc007_target_center, *self._bc007_target_orientation],
                dtype=np.float64,
            )
        return super()._fallback_pose(name)

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "rod_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "rod_target_pose": spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype=np.float64
                ),
                "rod_endpoints": spaces.Box(
                    -np.inf, np.inf, shape=(6,), dtype=np.float64
                ),
                "bc007_metrics": spaces.Box(
                    -np.inf, np.inf, shape=(16,), dtype=np.float64
                ),
                "bc007_task_state": spaces.Box(
                    -np.inf, np.inf, shape=(14,), dtype=np.float64
                ),
            }
        )
        return observation_space

    def _clear_bc007_assist_state(self, *, disable_joints: bool) -> None:
        """Clear every task-side variable that can affect future grasp physics."""

        disable_failures: list[str] = []
        if disable_joints:
            for side in ("left", "right"):
                if not self._set_bc007_grasp_joint(
                    side, False, record_event=False
                ):
                    disable_failures.append(side)
        self._bc007_grasp_owner = {"left": False, "right": False}
        self._bc007_grasp_frame_armed = {"left": False, "right": False}
        self._bc007_release_pending = {"left": False, "right": False}
        self._bc007_joint_enabled_readback = {"left": False, "right": False}
        self._bc007_owner_relative_transform = {"left": None, "right": None}
        self._bc007_owner_continuity_ok = {"left": False, "right": False}
        self._bc007_owner_continuity_position_error_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc007_owner_continuity_angle_error_rad = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc007_joint_frame_alignment_error_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._bc007_grasp_errors = []
        self._bc007_grasp_events = []
        self._bc007_control_frame = 0
        self._bc007_action_start_assist_state = encode_replay_assist_state(
            left_owner=False,
            right_owner=False,
            left_frame_armed=False,
            right_frame_armed=False,
            left_release_pending=False,
            right_release_pending=False,
        )
        if disable_failures:
            raise RuntimeError(
                "BC007 fixed grasp joints could not be disabled during reset: "
                + ",".join(disable_failures)
            )

    def _bc007_joint_enabled(self, side: str) -> bool:
        joint_path = str(self._bc007_manifest.get("joint_paths", {}).get(side) or "")
        if self._stage is None or not joint_path:
            raise RuntimeError(f"BC007 {side} fixed-joint readback is unavailable")
        try:
            from pxr import UsdPhysics

            joint = UsdPhysics.FixedJoint(self._stage.GetPrimAtPath(joint_path))
            attribute = joint.GetJointEnabledAttr() if joint else None
            if attribute is None or not attribute.IsValid():
                raise RuntimeError("joint-enabled attribute is unavailable")
            return bool(attribute.Get())
        except Exception as exc:
            raise RuntimeError(
                f"BC007 {side} fixed-joint readback is unavailable: {exc}"
            ) from exc

    def get_physical_replay_task_controller_state(self) -> np.ndarray:
        """Record the assist state latched immediately before the last action."""

        for side in ("left", "right"):
            readback = self._bc007_joint_enabled(side)
            self._bc007_joint_enabled_readback[side] = readback
            if readback != bool(self._bc007_grasp_owner[side]):
                raise RuntimeError(
                    f"BC007 {side} fixed-joint owner/readback state diverged"
                )
        state = np.asarray(self._bc007_action_start_assist_state)
        decode_replay_assist_state(state)
        return state.astype(np.uint8, copy=True)

    def _current_bc007_replay_assist_state(self) -> np.ndarray:
        return encode_replay_assist_state(
            left_owner=self._bc007_grasp_owner["left"],
            right_owner=self._bc007_grasp_owner["right"],
            left_frame_armed=self._bc007_grasp_frame_armed["left"],
            right_frame_armed=self._bc007_grasp_frame_armed["right"],
            left_release_pending=self._bc007_release_pending["left"],
            right_release_pending=self._bc007_release_pending["right"],
        )

    def set_physical_replay_task_controller_state(self, state: np.ndarray) -> None:
        """Restore recorded assist ownership before the matching raw command."""

        decoded = decode_replay_assist_state(state)
        for side in ("left", "right"):
            desired = decoded[side]
            if not desired["owner"]:
                if not self._set_bc007_grasp_joint(
                    side, False, record_event=False
                ):
                    raise RuntimeError(
                        f"BC007 replay could not disable the {side} fixed joint"
                    )
                if desired["frame_armed"]:
                    if not self._stage_bc007_grasp_joint_frame(side):
                        raise RuntimeError(
                            f"BC007 replay could not restore the {side} armed frame"
                        )
                else:
                    self._bc007_grasp_frame_armed[side] = False
                self._bc007_release_pending[side] = False
                continue

            if (
                bool(self._bc007_grasp_owner[side])
                and self._bc007_joint_enabled(side)
            ):
                self._bc007_grasp_frame_armed[side] = bool(
                    desired["frame_armed"]
                )
                self._bc007_release_pending[side] = bool(
                    desired["release_pending"]
                )
                continue
            if not self._stage_bc007_grasp_joint_frame(side):
                raise RuntimeError(
                    f"BC007 replay could not stage the {side} fixed-joint frame"
                )
            if not self._set_bc007_grasp_joint(
                side, True, record_event=False
            ):
                raise RuntimeError(
                    f"BC007 replay could not restore the {side} fixed-joint owner"
                )
            self._bc007_grasp_frame_armed[side] = bool(desired["frame_armed"])
            self._bc007_release_pending[side] = bool(desired["release_pending"])

    def get_recording_schema(self) -> dict[str, Any] | None:
        """Declare the shared controller field as restorable BC007 assist state."""

        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        raw_fields = result.get("fields")
        if not isinstance(raw_fields, dict):
            raise RuntimeError("BC007 recording schema has no PhysX field mapping")
        fields = {str(name): dict(spec) for name, spec in raw_fields.items()}
        controller = fields.get("physx_task_controller_state")
        if controller is None:
            raise RuntimeError(
                "BC007 recording schema omitted fixed-joint replay state"
            )
        controller.update(
            {
                "shape": [3],
                "dtype": "uint8",
                "layout": (
                    "left assist flags(owner,frame_armed,release_pending),"
                    "right assist flags(owner,frame_armed,release_pending),"
                    "schema version"
                ),
                "source": "bc007_action_aligned_fixed_joint_assist",
                "replay_restore_supported": True,
            }
        )
        fields["physx_task_controller_state"] = controller
        result["fields"] = fields
        result["task_controller_state"] = {
            "task_name": TASK_NAME,
            "field_name": "physx_task_controller_state",
            "capture_mode": "pre_action_latched_sampled_with_next_state",
            "replay_policy": "install_before_matching_raw_command",
            "ordinary_reset_state": "both_fixed_joints_disabled",
        }
        return result

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        self._clear_bc007_assist_state(disable_joints=True)
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        # The shared 5 cm table-height randomization can bring the complete
        # Wuji hand envelope into the tabletop at the canonical arm home.
        # BC007 already randomizes the rod in-plane, so keep the physical table
        # at its audited height on every reset.
        self.delta_h[:] = 0.0
        self._bc007_evaluator.reset()
        self._bc007_last_metrics = {}
        self._clear_bc007_assist_state(disable_joints=False)
        xy_jitter = (
            self._random.uniform(-RESET_XY_JITTER_M, RESET_XY_JITTER_M, size=2)
            if self.randomize
            else np.zeros(2, dtype=np.float64)
        )
        yaw_jitter = (
            float(self._random.uniform(-RESET_YAW_JITTER_DEG, RESET_YAW_JITTER_DEG))
            if self.randomize
            else 0.0
        )
        self._bc007_initial_center = ROD_START_CENTER.copy()
        self._bc007_initial_center[:2] += xy_jitter
        self._bc007_initial_orientation = initial_orientation_wxyz(yaw_jitter)
        self._bc007_target_center = (
            self._bc007_initial_center + ROD_TARGET_CENTER_OFFSET
        )
        self._bc007_target_orientation = target_orientation_wxyz(
            self._bc007_initial_orientation
        )
        initial_pose = np.asarray(
            [*self._bc007_initial_center, *self._bc007_initial_orientation],
            dtype=np.float64,
        )
        self._bc007_initial_axis_yaw_deg = rod_axis_yaw_deg(initial_pose)
        self._bc007_target_axis_yaw_deg = wrap_degrees(
            self._bc007_initial_axis_yaw_deg + TARGET_YAW_DELTA_DEG
        )

    def _write_rod_reset_pose(self) -> bool:
        if self._torch is None:
            return False
        rod_prim = self._find_rigid_body_prim(ROD_NAME)
        if rod_prim is None:
            self._add_task_state_warning("BC007 rod reset body is unavailable")
            return False
        path = str(rod_prim.GetPath())
        entry = next(
            (
                (view, indices, pose, velocity)
                for candidate_path, view, indices, pose, velocity
                in self._task_reset_body_states
                if candidate_path == path
            ),
            None,
        )
        if entry is None:
            self._add_task_state_warning("BC007 rod reset tensor is unavailable")
            return False
        view, indices, pose, velocity = entry
        reset_pose = pose.clone()
        reset_pose[:, :3] = self._torch.as_tensor(
            self._bc007_initial_center,
            dtype=reset_pose.dtype,
            device=reset_pose.device,
        )
        q = self._bc007_initial_orientation
        reset_pose[:, 3:7] = self._torch.as_tensor(
            [q[1], q[2], q[3], q[0]],
            dtype=reset_pose.dtype,
            device=reset_pose.device,
        )
        zero_velocity = self._torch.zeros_like(velocity)
        self._write_rigid_reset_state(path, view, indices, reset_pose, zero_velocity)
        return True

    def _wake_rod_after_reset(self) -> bool:
        """Keep the reset stability gate physical instead of masking it by sleep."""
        rod_prim = self._find_rigid_body_prim(ROD_NAME)
        if rod_prim is None:
            self._add_task_state_warning("BC007 rod wake body is unavailable")
            return False
        try:
            import omni.physx
            import omni.usd
            from pxr import PhysicsSchemaTools, Sdf

            simulation = omni.physx.get_physx_simulation_interface()
            stage_id = omni.usd.get_context().get_stage_id()
            simulation.wake_up(
                stage_id,
                PhysicsSchemaTools.sdfPathToInt(Sdf.Path(str(rod_prim.GetPath()))),
            )
            return True
        except Exception as exc:
            self._add_task_state_warning(f"BC007 rod reset wake failed: {exc}")
            return False

    def _rod_awake_readback(self) -> tuple[bool, bool, str]:
        """Read the long rod's real PhysX sleep flag without changing it."""

        rod_prim = self._find_rigid_body_prim(ROD_NAME)
        if rod_prim is None:
            return False, False, "rod rigid body is unavailable"
        try:
            import omni.physx
            import omni.usd
            from pxr import PhysicsSchemaTools, Sdf

            simulation = omni.physx.get_physx_simulation_interface()
            stage_id = omni.usd.get_context().get_stage_id()
            sleeping = simulation.is_sleeping(
                stage_id,
                PhysicsSchemaTools.sdfPathToInt(Sdf.Path(str(rod_prim.GetPath()))),
            )
            return not bool(sleeping), True, ""
        except Exception as exc:
            return False, False, f"{type(exc).__name__}: {exc}"

    def _apply_task_pose_randomization(self) -> None:
        if self._task_reset_body_states:
            self._write_rod_reset_pose()
        if self._stage is not None:
            set_goal_marker_pose(
                self._stage,
                self._bc007_target_center,
                self._bc007_target_orientation,
            )
        if not self._wake_rod_after_reset():
            raise RuntimeError("BC007 reset could not wake the long rod")

    def _hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (
                prim.GetName() == f"{side}_palm_link"
                or prim.GetName().startswith(f"{side}_finger")
            )
        ]

    def _init_task_specific_runtime(self) -> None:
        self._bc007_hand_contact_views = {}
        self._bc007_table_contact_view = None
        self._bc007_contact_errors = set()
        if not self._robots or self._stage is None:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        rod_prim = self._find_rigid_body_prim(ROD_NAME)
        table_prim = self._find_rigid_body_prim("table")
        if physics_view is None or rod_prim is None:
            self._add_task_state_warning("BC007 PhysX contact reporting is unavailable")
            return
        rod_path = str(rod_prim.GetPath())
        for side in ("left", "right"):
            hand_paths = self._hand_body_paths(side)
            if not hand_paths:
                self._bc007_contact_errors.add(f"hand:{side}")
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    rod_path,
                    filter_patterns=hand_paths,
                    max_contact_data_count=max(64, len(hand_paths) * 8),
                )
                if view is None or int(view.sensor_count) < 1 or int(view.filter_count) < 1:
                    self._bc007_contact_errors.add(f"hand:{side}")
                else:
                    self._bc007_hand_contact_views[side] = view
            except Exception as exc:
                self._bc007_contact_errors.add(f"hand:{side}")
                self._add_task_state_warning(
                    f"Could not bind BC007 {side}-hand/rod contact: {exc}"
                )
        if table_prim is not None:
            try:
                view = physics_view.create_rigid_contact_view(
                    rod_path,
                    filter_patterns=[str(table_prim.GetPath())],
                    max_contact_data_count=64,
                )
                if view is not None and int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                    self._bc007_table_contact_view = view
                else:
                    self._bc007_contact_errors.add("table")
            except Exception as exc:
                self._bc007_contact_errors.add("table")
                self._add_task_state_warning(f"Could not bind BC007 rod/table contact: {exc}")
        else:
            self._bc007_contact_errors.add("table")
        if self._bc007_contact_errors:
            self._add_task_state_warning(
                "BC007 contact reporting is incomplete and affected predicates fail closed: "
                + ",".join(sorted(self._bc007_contact_errors))
            )

    def _validate_task_initial_state_before_ready(self) -> Mapping[str, Any] | None:
        """Fail before collection if physical contact/support evidence is absent."""

        base_report = super()._validate_task_initial_state_before_ready()
        contact_force, contact_readable, missing = self._required_contact_snapshot()

        errors: list[str] = []
        if missing:
            errors.append(
                "required contact evidence unavailable: "
                + ",".join(sorted(missing))
            )
        pose = self._physics_body_pose(ROD_NAME)
        velocity = self._physics_body_velocity(ROD_NAME)
        if pose is None or not np.all(
            np.isfinite(np.asarray(pose, dtype=np.float64))
        ):
            errors.append("actual rod pose is unavailable or non-finite")
        if velocity is None or not np.all(
            np.isfinite(np.asarray(velocity, dtype=np.float64))
        ):
            errors.append("actual rod velocity is unavailable or non-finite")
        table_force = float(contact_force["table"])
        if not contact_readable["table"]:
            errors.append("rod/table support force readback is unavailable")
        elif table_force < _TABLE_SUPPORT_FORCE_N:
            errors.append(
                "awake rod has no measured tabletop support force: "
                f"{table_force:.6f}N < {_TABLE_SUPPORT_FORCE_N:.6f}N"
            )
        rod_awake, awake_readable, awake_error = self._rod_awake_readback()
        if not awake_readable:
            errors.append("rod awake readback is unavailable: " + awake_error)
        elif not rod_awake:
            errors.append("rod is sleeping at the collection-ready boundary")
        robot_pass, robot_report = initial_robot_state_report(self)
        if not robot_pass:
            errors.append("actual dual-ROKAE initial state failed the P0 audit")
        if errors:
            raise RuntimeError(
                "BC007 initial physical-state audit failed before ready: "
                + "; ".join(errors)
            )
        report = {
            "required_contact_evidence_available": True,
            "contact_force_n": dict(contact_force),
            "contact_readable": dict(contact_readable),
            "awake_table_support_force_n": float(table_force),
            "rod_awake_readback_available": bool(awake_readable),
            "rod_awake": bool(rod_awake),
            "actual_rod_pose": np.asarray(pose, dtype=np.float64).tolist(),
            "actual_rod_velocity": np.asarray(velocity, dtype=np.float64).tolist(),
            "actual_robot_state": robot_report,
        }
        self._bc007_initial_state_report = report
        if isinstance(base_report, Mapping):
            return {**base_report, **report}
        return report

    def _required_contact_snapshot(
        self,
    ) -> tuple[dict[str, float], dict[str, bool], set[str]]:
        """Read every contact channel whose absence must fail BC007 closed."""

        views = {
            "left_hand": self._bc007_hand_contact_views.get("left"),
            "right_hand": self._bc007_hand_contact_views.get("right"),
            "table": self._bc007_table_contact_view,
        }
        force: dict[str, float] = {}
        readable: dict[str, bool] = {}
        missing = set(self._bc007_contact_errors)
        for name, view in views.items():
            value, available = self._contact_force(view)
            force[name] = float(value)
            readable[name] = bool(available)
            if not available:
                missing.add(name)
        return force, readable, missing

    def stable_recording_origin_task_readiness(self) -> Mapping[str, Any]:
        """Continuously prove awake support and exact contact truth pre-action."""

        contact_force, contact_readable, missing = self._required_contact_snapshot()
        pose = self._physics_body_pose(ROD_NAME)
        velocity = self._physics_body_velocity(ROD_NAME)
        pose_values = (
            np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose is not None
            else np.zeros(0, dtype=np.float64)
        )
        velocity_values = (
            np.asarray(velocity, dtype=np.float64).reshape(-1)
            if velocity is not None
            else np.zeros(0, dtype=np.float64)
        )
        pose_readable = bool(
            pose_values.size == 7 and np.all(np.isfinite(pose_values))
        )
        velocity_readable = bool(
            velocity_values.size == 6 and np.all(np.isfinite(velocity_values))
        )
        rod_awake, awake_readable, awake_error = self._rod_awake_readback()
        support_force = float(contact_force["table"])
        support_active = bool(
            contact_readable["table"] and support_force >= _TABLE_SUPPORT_FORCE_N
        )
        contact_evidence_available = bool(
            not missing and all(contact_readable.values())
        )
        ready = bool(
            contact_evidence_available
            and pose_readable
            and velocity_readable
            and awake_readable
            and rod_awake
            and support_active
        )
        failure_reasons: list[str] = []
        if not contact_evidence_available:
            failure_reasons.append("required_contact_evidence_unavailable")
        if not pose_readable:
            failure_reasons.append("rod_pose_unavailable_or_nonfinite")
        if not velocity_readable:
            failure_reasons.append("rod_velocity_unavailable_or_nonfinite")
        if not awake_readable:
            failure_reasons.append("rod_awake_readback_unavailable")
        elif not rod_awake:
            failure_reasons.append("rod_sleeping")
        if not support_active:
            failure_reasons.append("table_support_force_below_minimum")
        return {
            "kind": "bc007_awake_table_support_and_three_contact_truth_v1",
            "ready": ready,
            "failure_reasons": failure_reasons,
            "required_contact_evidence_available": contact_evidence_available,
            "missing_contact_evidence": sorted(missing),
            "contact_readable": dict(contact_readable),
            "contact_force_n": dict(contact_force),
            "table_support_active": support_active,
            "table_support_force_n": support_force,
            "table_support_force_min_n": _TABLE_SUPPORT_FORCE_N,
            "rod_pose_readable": pose_readable,
            "rod_velocity_readable": velocity_readable,
            "rod_awake_readback_available": bool(awake_readable),
            "rod_awake_readback_error": awake_error,
            "rod_awake": bool(rod_awake),
        }

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Recheck the complete physical P0 boundary before action zero."""

        from .recording_origin import RecordingOriginStabilityError

        final_readiness = report.get("final_task_readiness")
        if (
            report.get("task_readiness_required") is not True
            or not isinstance(final_readiness, Mapping)
            or final_readiness.get("ready") is not True
        ):
            raise RecordingOriginStabilityError(
                "B320-BC007 reached action zero without continuous awake "
                "rod/support/contact evidence.",
                diagnostics={
                    "failure_phase": "bc007_origin_acceptance",
                    "task_readiness": final_readiness,
                },
            )
        if int(self.env_step) != 0:
            raise RecordingOriginStabilityError(
                "B320-BC007 origin acceptance ran after action zero.",
                diagnostics={
                    "failure_phase": "bc007_origin_acceptance",
                    "env_step": int(self.env_step),
                },
            )
        current_readiness = self.stable_recording_origin_task_readiness()
        if current_readiness.get("ready") is not True:
            raise RecordingOriginStabilityError(
                "B320-BC007 physical readiness changed at origin acceptance.",
                diagnostics={
                    "failure_phase": "bc007_origin_recheck",
                    "task_readiness": dict(current_readiness),
                },
            )
        try:
            initial_state = self._validate_task_initial_state_before_ready()
        except Exception as exc:
            raise RecordingOriginStabilityError(
                "B320-BC007 actual initial-state readback failed at origin "
                "acceptance.",
                diagnostics={
                    "failure_phase": "bc007_origin_actual_state",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        if not isinstance(initial_state, Mapping):
            raise RecordingOriginStabilityError(
                "B320-BC007 origin acceptance produced no actual-state evidence.",
                diagnostics={"failure_phase": "bc007_origin_actual_state"},
            )
        parent = super().accept_stable_recording_origin(report)
        payload = dict(parent) if isinstance(parent, Mapping) else {}
        payload.update(
            {
                "accepted": True,
                "kind": "bc007_awake_support_contact_and_actual_robot_boundary",
                "final_readiness": dict(current_readiness),
                "initial_state": dict(initial_state),
            }
        )
        return payload

    def _contact_force(self, view: Any | None) -> tuple[float, bool]:
        if view is None:
            return 0.0, False
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            forces = np.asarray(matrix, dtype=np.float64)
            if forces.size == 0 or not np.all(np.isfinite(forces)):
                return 0.0, False
            return float(np.linalg.norm(forces, axis=-1).sum()), True
        except Exception:
            return 0.0, False

    def _after_task_physics_substep(self) -> None:
        """Take optional live-acceptance robot evidence at the true solver rate."""

        super()._after_task_physics_substep()
        audit = self._bc007_per_physics_robot_audit
        if audit is not None:
            audit.sample(self)

    def _hand_closed(self, side: str) -> bool:
        command = self._left_hand if side == "left" else self._right_hand
        return bool(command.size and float(np.max(np.abs(command))) >= 0.08)

    def _hand_distance_to_segment(
        self, side: str, start: np.ndarray, end: np.ndarray
    ) -> float:
        return min(
            (
                _point_to_segment_distance(point, start, end)
                for point in self._a06_hand_points(side)
            ),
            default=math.inf,
        )

    def _palm_position(self, side: str) -> np.ndarray:
        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        matrix = self._native_end_effector_pose_matrix(side, fallback)
        return np.asarray(matrix[:3, 3], dtype=np.float64)

    def _bc007_palm_prim_path(self, side: str) -> str | None:
        if self._stage is None:
            return None
        palm_path = str(self._bc007_manifest.get("palm_paths", {}).get(side) or "")
        if not palm_path or not self._stage.GetPrimAtPath(palm_path).IsValid():
            return None
        return palm_path

    def _record_bc007_grasp_error(self, side: str, detail: str) -> None:
        message = f"{side}: {detail}"
        if message not in self._bc007_grasp_errors:
            self._bc007_grasp_errors.append(message)
            self._add_task_state_warning(f"BC007 grasp stabilizer unavailable: {message}")

    def _set_bc007_grasp_joint(
        self,
        side: str,
        enabled: bool,
        *,
        record_event: bool = True,
    ) -> bool:
        previous = bool(self._bc007_grasp_owner.get(side, False))
        if self._stage is None:
            self._bc007_grasp_owner[side] = False
            return not enabled
        joint_path = str(self._bc007_manifest.get("joint_paths", {}).get(side) or "")
        if not joint_path:
            if enabled:
                self._record_bc007_grasp_error(side, "joint path is unavailable")
            self._bc007_grasp_owner[side] = False
            return False
        try:
            from pxr import UsdPhysics

            joint = UsdPhysics.FixedJoint(self._stage.GetPrimAtPath(joint_path))
            if not joint:
                if enabled:
                    self._record_bc007_grasp_error(side, "fixed-joint schema is invalid")
                self._bc007_grasp_owner[side] = False
                return False
            enabled_attr = joint.GetJointEnabledAttr()
            if not enabled_attr.IsValid():
                enabled_attr = joint.CreateJointEnabledAttr()
            if enabled:
                force, available = self._contact_force(
                    self._bc007_hand_contact_views.get(side)
                )
                if not available or force < _HAND_CONTACT_FORCE_N:
                    return False
                palm_path = self._bc007_palm_prim_path(side)
                targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
                if palm_path is None or targets != [palm_path]:
                    raise RuntimeError(
                        f"preauthored palm owner changed: expected={palm_path}, "
                        f"observed={targets}"
                    )
                if not self._bc007_grasp_frame_armed[side]:
                    return False
                rod_matrix = self._current_body_matrix(ROD_NAME)
                palm_matrix = self._robot_link_pose_matrix(side)
                if rod_matrix is None or palm_matrix is None:
                    return False
                owner_relative = np.linalg.inv(palm_matrix) @ rod_matrix
                enabled_attr.Set(True)
                readback = bool(enabled_attr.Get())
                self._bc007_joint_enabled_readback[side] = readback
                if not readback:
                    enabled_attr.Set(False)
                    return False
                self._bc007_grasp_owner[side] = True
                self._bc007_owner_relative_transform[side] = owner_relative
                self._bc007_owner_continuity_ok[side] = True
                self._bc007_release_pending[side] = False
                if not previous and record_event:
                    self._bc007_grasp_events.append(
                        {
                            "frame": int(self._bc007_control_frame),
                            "event": "grasp",
                            "side": side,
                            "measured_contact_force_n": float(force),
                        }
                    )
                return True
            enabled_attr.Set(False)
            readback = bool(enabled_attr.Get())
            self._bc007_joint_enabled_readback[side] = readback
            self._bc007_grasp_owner[side] = readback
            if readback:
                self._record_bc007_grasp_error(side, "joint remained enabled")
                return False
            self._bc007_owner_relative_transform[side] = None
            self._bc007_owner_continuity_ok[side] = False
            self._bc007_grasp_frame_armed[side] = False
            self._bc007_release_pending[side] = False
            if previous and record_event:
                self._bc007_grasp_events.append(
                    {
                        "frame": int(self._bc007_control_frame),
                        "event": "release",
                        "side": side,
                    }
                )
            return True
        except Exception as exc:
            self._bc007_grasp_owner[side] = False
            if enabled:
                self._record_bc007_grasp_error(side, str(exc))
            return False

    @staticmethod
    def _bc007_relative_pose_error(
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

    def _bc007_owner_continuity(self, side: str) -> bool:
        reference = self._bc007_owner_relative_transform.get(side)
        rod_matrix = self._current_body_matrix(ROD_NAME)
        palm_matrix = self._robot_link_pose_matrix(side)
        joint_path = str(self._bc007_manifest.get("joint_paths", {}).get(side) or "")
        readback = False
        if self._stage is not None and joint_path:
            try:
                from pxr import UsdPhysics

                joint = UsdPhysics.FixedJoint(self._stage.GetPrimAtPath(joint_path))
                attribute = joint.GetJointEnabledAttr() if joint else None
                readback = bool(attribute and attribute.IsValid() and attribute.Get())
            except Exception:
                readback = False
        self._bc007_joint_enabled_readback[side] = readback
        if reference is None or rod_matrix is None or palm_matrix is None:
            self._bc007_owner_continuity_position_error_m[side] = math.inf
            self._bc007_owner_continuity_angle_error_rad[side] = math.inf
            self._bc007_owner_continuity_ok[side] = False
            return False
        current_relative = np.linalg.inv(palm_matrix) @ rod_matrix
        position_error, angle_error = self._bc007_relative_pose_error(
            reference, current_relative
        )
        continuity_ok = bool(
            readback
            and position_error <= _OWNER_CONTINUITY_POSITION_TOLERANCE_M
            and angle_error <= _OWNER_CONTINUITY_ANGLE_TOLERANCE_RAD
        )
        self._bc007_owner_continuity_position_error_m[side] = position_error
        self._bc007_owner_continuity_angle_error_rad[side] = angle_error
        self._bc007_owner_continuity_ok[side] = continuity_ok
        return continuity_ok

    def _stage_bc007_grasp_joint_frame(self, side: str) -> bool:
        if self._stage is None:
            return False
        joint_path = str(self._bc007_manifest.get("joint_paths", {}).get(side) or "")
        rod_path = str(self._bc007_manifest.get("rod_root") or "")
        palm_path = self._bc007_palm_prim_path(side)
        rod_matrix = self._current_body_matrix(ROD_NAME)
        palm_matrix = self._robot_link_pose_matrix(side)
        if (
            not joint_path
            or not rod_path
            or palm_path is None
            or rod_matrix is None
            or palm_matrix is None
        ):
            return False
        try:
            from pxr import Gf, Sdf, UsdPhysics

            joint = UsdPhysics.FixedJoint(self._stage.GetPrimAtPath(joint_path))
            if not joint:
                raise RuntimeError("fixed-joint schema is invalid")
            body0_targets = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            body1_targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            if body0_targets != [rod_path] or body1_targets != [palm_path]:
                raise RuntimeError(
                    f"preauthored topology changed: body0={body0_targets}, "
                    f"body1={body1_targets}"
                )
            body0_frame, body1_frame = joint_local_frame_pair(
                rod_matrix, palm_matrix
            )
            world0 = rod_matrix @ body0_frame
            world1 = palm_matrix @ body1_frame
            alignment_error = float(np.linalg.norm(world0[:3, 3] - world1[:3, 3]))
            if alignment_error > 1.0e-6:
                raise RuntimeError(
                    f"computed joint frames are not coincident: {alignment_error:.9f}m"
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
            self._bc007_grasp_frame_armed[side] = True
            self._bc007_joint_frame_alignment_error_m[side] = max(
                self._bc007_joint_frame_alignment_error_m[side], alignment_error
            )
            return True
        except Exception as exc:
            self._bc007_grasp_frame_armed[side] = False
            self._record_bc007_grasp_error(side, str(exc))
            return False

    def _postprocess_hand_action(self) -> None:
        # The shared physical trace samples this field after the action.  Latch
        # the state that actually governed this action before changing any
        # owner/release flag, so replay can install next_* before the matching
        # raw command without advancing the two-frame release state machine.
        self._bc007_action_start_assist_state = (
            self._current_bc007_replay_assist_state()
        )
        self._bc007_control_frame += 1
        rod = self._poses.get(ROD_NAME, self._fallback_pose(ROD_NAME))
        minus_endpoint, plus_endpoint = rod_endpoints(rod, ROD_HALF_LENGTH_M)
        endpoint_by_side = {
            "left": minus_endpoint if LEFT_ENDPOINT_LOCAL_SIGN < 0.0 else plus_endpoint,
            "right": minus_endpoint if RIGHT_ENDPOINT_LOCAL_SIGN < 0.0 else plus_endpoint,
        }
        opposite_by_side = {
            "left": endpoint_by_side["right"],
            "right": endpoint_by_side["left"],
        }
        for side in ("left", "right"):
            hand = self._left_hand if side == "left" else self._right_hand
            command = float(np.max(np.abs(hand))) if hand.size else 0.0
            owned = bool(self._bc007_grasp_owner[side])
            if owned and not self._bc007_owner_continuity(side):
                self._record_bc007_grasp_error(side, "live owner continuity failed")
                self._set_bc007_grasp_joint(side, False, record_event=False)
                continue
            if owned and command <= _GRASP_RELEASE_COMMAND:
                if not self._bc007_release_pending[side]:
                    if self._stage_bc007_grasp_joint_frame(side):
                        self._bc007_release_pending[side] = True
                    continue
                self._set_bc007_grasp_joint(side, False)
                continue
            if owned:
                self._bc007_release_pending[side] = False
                continue
            palm_matrix = self._robot_link_pose_matrix(side)
            if palm_matrix is None:
                continue
            palm_distance = float(
                np.linalg.norm(palm_matrix[:3, 3] - endpoint_by_side[side])
            )
            correct_start, correct_end = _role_segment(rod[:3], endpoint_by_side[side])
            wrong_start, wrong_end = _role_segment(rod[:3], opposite_by_side[side])
            correct_distance = self._hand_distance_to_segment(
                side, correct_start, correct_end
            )
            wrong_distance = self._hand_distance_to_segment(side, wrong_start, wrong_end)
            geometry_ready = bool(
                palm_distance <= _GRASP_PALM_TO_END_MAX_M
                and correct_distance <= _END_REGION_DISTANCE_M
                and wrong_distance >= _ROLE_WRONG_END_MIN_DISTANCE_M
            )
            force, available = self._contact_force(
                self._bc007_hand_contact_views.get(side)
            )
            contact_ready = bool(available and force >= _HAND_CONTACT_FORCE_N)
            if command < _GRASP_ENABLE_COMMAND:
                if geometry_ready:
                    self._stage_bc007_grasp_joint_frame(side)
                else:
                    self._bc007_grasp_frame_armed[side] = False
                continue
            if geometry_ready and contact_ready and self._bc007_grasp_frame_armed[side]:
                self._set_bc007_grasp_joint(side, True)
            elif geometry_ready:
                self._stage_bc007_grasp_joint_frame(side)
            else:
                self._bc007_grasp_frame_armed[side] = False

    def _bc007_metrics(self, *, update: bool) -> dict[str, Any]:
        rod = self._poses.get(ROD_NAME, self._fallback_pose(ROD_NAME))
        minus_endpoint, plus_endpoint = rod_endpoints(rod, ROD_HALF_LENGTH_M)
        endpoint_by_side = {
            "left": minus_endpoint
            if LEFT_ENDPOINT_LOCAL_SIGN < 0.0
            else plus_endpoint,
            "right": minus_endpoint
            if RIGHT_ENDPOINT_LOCAL_SIGN < 0.0
            else plus_endpoint,
        }
        opposite_by_side = {
            "left": endpoint_by_side["right"],
            "right": endpoint_by_side["left"],
        }
        role_owned: dict[str, bool] = {}
        role_distance: dict[str, float] = {}
        wrong_distance: dict[str, float] = {}
        contact_force: dict[str, float] = {}
        contact_available: dict[str, bool] = {}
        hand_closed: dict[str, bool] = {}
        for side in ("left", "right"):
            start, end = _role_segment(rod[:3], endpoint_by_side[side])
            other_start, other_end = _role_segment(rod[:3], opposite_by_side[side])
            role_distance[side] = self._hand_distance_to_segment(side, start, end)
            wrong_distance[side] = self._hand_distance_to_segment(
                side, other_start, other_end
            )
            contact_force[side], contact_available[side] = self._contact_force(
                self._bc007_hand_contact_views.get(side)
            )
            hand_closed[side] = self._hand_closed(side)
            owner_continuity = bool(
                self._bc007_grasp_owner[side]
                and self._bc007_owner_continuity(side)
            )
            measured_contact = bool(
                contact_available[side]
                and contact_force[side] >= _HAND_CONTACT_FORCE_N
            )
            role_owned[side] = bool(
                (measured_contact or owner_continuity)
                and hand_closed[side]
                and role_distance[side] <= _END_REGION_DISTANCE_M
                and wrong_distance[side] >= _ROLE_WRONG_END_MIN_DISTANCE_M
            )

        left_palm = self._palm_position("left")
        right_palm = self._palm_position("right")
        hand_separation = float(np.linalg.norm(left_palm - right_palm))
        current_yaw = rod_axis_yaw_deg(rod)
        yaw_progress = signed_yaw_progress_deg(
            current_yaw, self._bc007_initial_axis_yaw_deg
        )
        yaw_error = wrap_degrees(self._bc007_target_axis_yaw_deg - current_yaw)
        endpoint_height_difference = abs(
            float(minus_endpoint[2] - plus_endpoint[2])
        )
        position_error = float(np.linalg.norm(rod[:3] - self._bc007_target_center))
        velocity = self._physics_body_velocity(ROD_NAME)
        if velocity is None:
            linear_speed = math.inf
            angular_speed = math.inf
            velocity_finite = False
        else:
            velocity = np.asarray(velocity, dtype=np.float64).reshape(6)
            linear_speed = float(np.linalg.norm(velocity[:3]))
            angular_speed = float(np.linalg.norm(velocity[3:]))
            velocity_finite = bool(np.all(np.isfinite(velocity)))
        table_force, table_contact_available = self._contact_force(
            self._bc007_table_contact_view
        )
        table_contact = bool(
            table_contact_available and table_force >= _TABLE_CONTACT_FORCE_N
        )
        required_contact_evidence_available = bool(
            table_contact_available
            and contact_available["left"]
            and contact_available["right"]
        )
        strong_table_contact = bool(
            table_contact_available
            and table_force >= _STRONG_TABLE_CONTACT_FORCE_N
        )
        # The task manifest defines strong collision as robot/rod impact.  The
        # rod is intentionally tabletop-supported before lift, while any
        # rod/table contact after lift is already rejected by the stricter
        # table-contact predicate and grace window.
        strong_collision = bool(
            any(
                contact_available[side]
                and contact_force[side] >= _STRONG_HAND_CONTACT_FORCE_N
                for side in ("left", "right")
            )
        )
        finite_and_in_workspace = bool(
            np.all(np.isfinite(rod))
            and velocity_finite
            and np.all(rod[:3] >= _WORKSPACE_LOWER)
            and np.all(rod[:3] <= _WORKSPACE_UPPER)
            and np.all(minus_endpoint >= _WORKSPACE_LOWER)
            and np.all(minus_endpoint <= _WORKSPACE_UPPER)
            and np.all(plus_endpoint >= _WORKSPACE_LOWER)
            and np.all(plus_endpoint <= _WORKSPACE_UPPER)
        )
        dropped = bool(
            rod[2] < TABLE_TOP_Z + ROD_RADIUS_M - 0.010
            or minus_endpoint[2] < TABLE_TOP_Z - 0.010
            or plus_endpoint[2] < TABLE_TOP_Z - 0.010
        )
        frame = LongRodFrame(
            left_role_owned=role_owned["left"],
            right_role_owned=role_owned["right"],
            center_height_m=float(rod[2]),
            yaw_progress_deg=float(yaw_progress),
            yaw_error_deg=float(yaw_error),
            endpoint_height_difference_m=endpoint_height_difference,
            linear_speed_mps=linear_speed,
            angular_speed_radps=angular_speed,
            target_position_error_m=position_error,
            hand_separation_m=hand_separation,
            table_contact=table_contact,
            strong_collision=strong_collision,
            required_contact_evidence_available=(
                required_contact_evidence_available
            ),
            finite_and_in_workspace=finite_and_in_workspace,
            dropped=dropped,
        )
        history = (
            self._bc007_evaluator.update(frame)
            if update
            else self._bc007_evaluator.snapshot(success_now=False)
        )
        metrics: dict[str, Any] = {
            **history,
            "left_role_owned": role_owned["left"],
            "right_role_owned": role_owned["right"],
            "left_hand_closed": hand_closed["left"],
            "right_hand_closed": hand_closed["right"],
            "left_role_distance_m": role_distance["left"],
            "right_role_distance_m": role_distance["right"],
            "left_wrong_end_distance_m": wrong_distance["left"],
            "right_wrong_end_distance_m": wrong_distance["right"],
            "left_contact_force_n": contact_force["left"],
            "right_contact_force_n": contact_force["right"],
            "left_contact_available": contact_available["left"],
            "right_contact_available": contact_available["right"],
            "required_contact_evidence_available": (
                required_contact_evidence_available
            ),
            "left_grasp_owner": bool(self._bc007_grasp_owner["left"]),
            "right_grasp_owner": bool(self._bc007_grasp_owner["right"]),
            "left_owner_continuity_ok": bool(
                self._bc007_owner_continuity_ok["left"]
            ),
            "right_owner_continuity_ok": bool(
                self._bc007_owner_continuity_ok["right"]
            ),
            "left_owner_position_error_m": float(
                self._bc007_owner_continuity_position_error_m["left"]
            ),
            "right_owner_position_error_m": float(
                self._bc007_owner_continuity_position_error_m["right"]
            ),
            "left_owner_angle_error_rad": float(
                self._bc007_owner_continuity_angle_error_rad["left"]
            ),
            "right_owner_angle_error_rad": float(
                self._bc007_owner_continuity_angle_error_rad["right"]
            ),
            "table_contact_force_n": table_force,
            "table_contact_available": table_contact_available,
            "table_contact": table_contact,
            "strong_table_contact": strong_table_contact,
            "strong_collision": strong_collision,
            "current_yaw_deg": current_yaw,
            "initial_yaw_deg": self._bc007_initial_axis_yaw_deg,
            "target_yaw_deg": self._bc007_target_axis_yaw_deg,
            "yaw_progress_deg": yaw_progress,
            "yaw_error_deg": yaw_error,
            "center_height_m": float(rod[2]),
            "in_height_band": bool(
                HEIGHT_BAND_LOW_M <= rod[2] <= HEIGHT_BAND_HIGH_M
            ),
            "endpoint_height_difference_m": endpoint_height_difference,
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
            "target_position_error_m": position_error,
            "hand_separation_m": hand_separation,
            "finite_and_in_workspace": finite_and_in_workspace,
            "dropped": dropped,
            "minus_endpoint": minus_endpoint,
            "plus_endpoint": plus_endpoint,
            "left_palm_position": left_palm,
            "right_palm_position": right_palm,
        }
        if update:
            self._bc007_last_metrics = metrics
        return metrics

    def _compute_success_now(self) -> bool:
        return bool(self._bc007_metrics(update=True)["success_now"])

    def _compute_failure(self) -> bool:
        return bool(self._bc007_evaluator.failed)

    @staticmethod
    def _bc007_slerp(
        start_wxyz: np.ndarray,
        target_wxyz: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        start = np.asarray(start_wxyz, dtype=np.float64).reshape(4)
        target = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
        start /= max(float(np.linalg.norm(start)), 1.0e-12)
        target /= max(float(np.linalg.norm(target)), 1.0e-12)
        dot = float(np.dot(start, target))
        if dot < 0.0:
            target = -target
            dot = -dot
        if dot > 0.9995:
            result = (1.0 - fraction) * start + fraction * target
            return result / max(float(np.linalg.norm(result)), 1.0e-12)
        angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
        sine = math.sin(angle)
        return (
            math.sin((1.0 - fraction) * angle) / sine * start
            + math.sin(fraction * angle) / sine * target
        )

    @classmethod
    def _bc007_interpolate_pose(
        cls,
        start: np.ndarray,
        target: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        result = np.asarray(start, dtype=np.float64).copy()
        result[:3] = (1.0 - fraction) * start[:3] + fraction * target[:3]
        result[3:7] = cls._bc007_slerp(start[3:7], target[3:7], fraction)
        return result

    @staticmethod
    def _bc007_core_action(
        right_pose: np.ndarray,
        left_pose: np.ndarray,
        right_hand: np.ndarray,
        left_hand: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [right_pose, left_pose, right_hand, left_hand]
        ).astype(np.float32)

    def automatic_success_actions(
        self,
        observation: Mapping[str, Any],
    ) -> list[np.ndarray]:
        """Return a contact-first physical action proof for task validation."""

        state = observation["state"]
        endpoints = np.asarray(state["rod_endpoints"], dtype=np.float64).reshape(2, 3)
        rod_pose = np.asarray(state["rod_pose"], dtype=np.float64).reshape(7)
        target_pose = np.asarray(state["rod_target_pose"], dtype=np.float64).reshape(7)
        starts: dict[str, np.ndarray] = {}
        start_matrices: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            matrix = self._robot_link_pose_matrix(side)
            if matrix is None:
                raise RuntimeError(f"BC007 missing measured {side} palm pose")
            start_matrices[side] = matrix.copy()
            starts[side] = self._pose_matrix_to_vector(matrix)

        grasp_offsets = {
            # Keep both palms slightly inboard of the geometric endpoints.
            # The 4 cm vertical offset is required for early measured finger
            # contact; the task layout keeps the lift path away from the
            # folded-elbow singular region of the shared two-arm setup.
            "left": np.asarray([-0.04, 0.0, 0.040], dtype=np.float64),
            "right": np.asarray([-0.04, 0.030, 0.040], dtype=np.float64),
        }
        grasps = {side: starts[side].copy() for side in ("left", "right")}
        grasps["left"][:3] = endpoints[0] + grasp_offsets["left"]
        grasps["right"][:3] = endpoints[1] + grasp_offsets["right"]
        # Center each wrist excursion around a comfortable mirrored posture.
        # The round rod permits independent roll choices without changing the
        # assigned-end grasp geometry.
        pregrasp_angles = {
            "left": math.radians(-60.0),
            "right": math.radians(-70.0),
        }
        pregrasp_rotations: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            pregrasp_angle = pregrasp_angles[side]
            pregrasp_rotation = np.asarray(
                [
                    [math.cos(pregrasp_angle), -math.sin(pregrasp_angle), 0.0],
                    [math.sin(pregrasp_angle), math.cos(pregrasp_angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            matrix = start_matrices[side].copy()
            matrix[:3, :3] = pregrasp_rotation @ matrix[:3, :3]
            matrix[:3, 3] = grasps[side][:3]
            grasps[side][3:7] = self._pose_matrix_to_vector(matrix)[3:7]
            pregrasp_rotations[side] = matrix[:3, :3].copy()
        above = {side: pose.copy() for side, pose in grasps.items()}
        above["left"][2] += 0.080
        above["right"][2] += 0.080
        translation = target_pose[:3] - rod_pose[:3]
        # First establish table clearance, then blend the rest of the lift with
        # the beginning of the turn.  This follows the motion a teleoperator
        # would naturally use and avoids dwelling in the folded-shoulder pose
        # created by lifting both unrotated endpoints all the way to target.
        clearance_fraction = 0.70
        clearance = {side: pose.copy() for side, pose in grasps.items()}
        for side in ("left", "right"):
            clearance[side][:3] += clearance_fraction * translation

        open_hand = np.zeros(20, dtype=np.float64)
        closed_hand = self._tool_power_grip_pose(open_hand) * 0.70
        actions: list[np.ndarray] = []

        def hold(
            right_pose: np.ndarray,
            left_pose: np.ndarray,
            hand: np.ndarray,
            frames: int,
        ) -> None:
            action = self._bc007_core_action(right_pose, left_pose, hand, hand)
            actions.extend(action.copy() for _ in range(frames))

        def move(
            right_from: np.ndarray,
            right_to: np.ndarray,
            left_from: np.ndarray,
            left_to: np.ndarray,
            hand_from: np.ndarray,
            hand_to: np.ndarray,
            frames: int,
        ) -> None:
            for index in range(1, frames + 1):
                fraction = index / frames
                actions.append(
                    self._bc007_core_action(
                        self._bc007_interpolate_pose(right_from, right_to, fraction),
                        self._bc007_interpolate_pose(left_from, left_to, fraction),
                        (1.0 - fraction) * hand_from + fraction * hand_to,
                        (1.0 - fraction) * hand_from + fraction * hand_to,
                    )
                )

        hold(starts["right"], starts["left"], open_hand, 10)
        move(
            starts["right"], above["right"], starts["left"], above["left"],
            open_hand, open_hand, 30,
        )
        move(
            above["right"], grasps["right"], above["left"], grasps["left"],
            open_hand, open_hand, 25,
        )
        move(
            grasps["right"], grasps["right"], grasps["left"], grasps["left"],
            open_hand, closed_hand, 16,
        )
        hold(grasps["right"], grasps["left"], closed_hand, 8)
        move(
            grasps["right"], clearance["right"], grasps["left"], clearance["left"],
            closed_hand, closed_hand, 45,
        )
        hold(clearance["right"], clearance["left"], closed_hand, 8)

        grasp_matrices: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            matrix = start_matrices[side].copy()
            matrix[:3, :3] = pregrasp_rotations[side]
            matrix[:3, 3] = grasps[side][:3]
            grasp_matrices[side] = matrix
        start_pivot = rod_pose[:3]
        target_pivot = target_pose[:3]
        final: dict[str, np.ndarray] = {}
        proof_yaw_delta_deg = TARGET_YAW_DELTA_DEG

        def append_rigid_rod_motion(
            start_fraction: float,
            end_fraction: float,
            start_angle_deg: float,
            end_angle_deg: float,
            frames: int,
        ) -> None:
            for index in range(1, frames + 1):
                fraction = index / frames
                translation_fraction = (
                    (1.0 - fraction) * start_fraction + fraction * end_fraction
                )
                angle_deg = (
                    (1.0 - fraction) * start_angle_deg + fraction * end_angle_deg
                )
                angle = math.radians(angle_deg)
                cosine, sine = math.cos(angle), math.sin(angle)
                rotation = np.asarray(
                    [
                        [cosine, -sine, 0.0],
                        [sine, cosine, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                pivot = start_pivot + translation_fraction * (
                    target_pivot - start_pivot
                )
                poses: dict[str, np.ndarray] = {}
                for side in ("left", "right"):
                    matrix = np.eye(4, dtype=np.float64)
                    matrix[:3, :3] = rotation @ grasp_matrices[side][:3, :3]
                    matrix[:3, 3] = pivot + rotation @ (
                        grasp_matrices[side][:3, 3] - start_pivot
                    )
                    poses[side] = self._pose_matrix_to_vector(matrix)
                    if end_angle_deg == proof_yaw_delta_deg and index == frames:
                        final[side] = poses[side].copy()
                actions.append(
                    self._bc007_core_action(
                        poses["right"], poses["left"], closed_hand, closed_hand
                    )
                )

        append_rigid_rod_motion(
            clearance_fraction,
            1.0,
            0.0,
            35.0,
            90,
        )
        append_rigid_rod_motion(
            1.0,
            1.0,
            35.0,
            proof_yaw_delta_deg,
            145,
        )
        leveled_final = {side: pose.copy() for side, pose in final.items()}
        leveled_final["left"][2] += 0.015
        leveled_final["right"][2] -= 0.015
        move(
            final["right"], leveled_final["right"],
            final["left"], leveled_final["left"],
            closed_hand, closed_hand, 30,
        )
        hold(leveled_final["right"], leveled_final["left"], closed_hand, 60)
        return actions

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._bc007_last_metrics or self._bc007_metrics(update=False)
        stage = int(metrics["stage"])
        rows = [
            self._condition_row(
                "bc007_distinct_end_roles",
                "Left and right hands own different assigned rod ends",
                bool(metrics["left_role_owned"] and metrics["right_role_owned"]),
            ),
            self._condition_row(
                "bc007_lifted",
                "Rod was lifted into the prescribed height band",
                stage >= int(LongRodStage.LIFTED),
            ),
            self._condition_row(
                "bc007_rotation_proof",
                "Dual-hand airborne yaw rotation has started",
                bool(metrics["rotation_latched"]),
            ),
            self._condition_row(
                "bc007_target_yaw",
                "Rod yaw is within 8 degrees of the 90-degree target",
                abs(float(metrics["yaw_error_deg"])) <= TARGET_YAW_TOLERANCE_DEG,
            ),
            self._condition_row(
                "bc007_target_position",
                "Rod center is within 4 cm of the target center",
                float(metrics["target_position_error_m"])
                <= TARGET_POSITION_TOLERANCE_M,
            ),
            self._condition_row(
                "bc007_level",
                "Endpoint height difference is at most 3 cm",
                float(metrics["endpoint_height_difference_m"]) <= 0.03,
            ),
            self._condition_row(
                "bc007_hand_separation",
                "Hands remain at least 25 cm apart",
                float(metrics["hand_separation_m"]) >= MINIMUM_HAND_SEPARATION_M,
            ),
            self._condition_row(
                "bc007_stable",
                "Rod speed is inside the terminal stability limits",
                float(metrics["linear_speed_mps"]) <= MAX_LINEAR_SPEED_MPS
                and float(metrics["angular_speed_radps"])
                <= MAX_ANGULAR_SPEED_RADPS,
            ),
            self._condition_row(
                "bc007_contact_evidence",
                "Both hand/rod and rod/table contact evidence streams are live",
                bool(metrics["required_contact_evidence_available"]),
            ),
            self._condition_row(
                "bc007_no_table_assist",
                "Rod does not use the table during reorientation",
                not bool(metrics["table_contact"]),
            ),
            self._condition_row(
                "bc007_no_failure",
                "No drop, role loss, strong collision, or height-band failure",
                not bool(metrics["failed"]),
            ),
        ]
        return self._append_success_hold_row(rows)

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._bc007_last_metrics or self._bc007_metrics(update=False)
        rod = self._poses.get(ROD_NAME, self._fallback_pose(ROD_NAME))
        target = np.asarray(
            [*self._bc007_target_center, *self._bc007_target_orientation],
            dtype=np.float64,
        )
        observation["state"].update(
            {
                "rod_pose": rod.copy(),
                "rod_target_pose": target,
                "rod_endpoints": np.concatenate(
                    [metrics["minus_endpoint"], metrics["plus_endpoint"]]
                ).astype(np.float64),
                "bc007_metrics": np.asarray(
                    [
                        metrics["current_yaw_deg"],
                        metrics["target_yaw_deg"],
                        metrics["yaw_progress_deg"],
                        metrics["yaw_error_deg"],
                        metrics["center_height_m"],
                        metrics["endpoint_height_difference_m"],
                        metrics["target_position_error_m"],
                        metrics["linear_speed_mps"],
                        metrics["angular_speed_radps"],
                        metrics["hand_separation_m"],
                        metrics["left_role_distance_m"],
                        metrics["right_role_distance_m"],
                        metrics["left_contact_force_n"],
                        metrics["right_contact_force_n"],
                        metrics["table_contact_force_n"],
                        float(bool(metrics["finite_and_in_workspace"])),
                    ],
                    dtype=np.float64,
                ),
                "bc007_task_state": np.asarray(
                    [
                        metrics["stage"],
                        metrics["dual_grasp_streak"],
                        metrics["role_loss_streak"],
                        metrics["height_exit_streak"],
                        metrics["table_contact_streak"],
                        metrics["severe_tilt_streak"],
                        metrics["strong_collision_streak"],
                        float(bool(metrics["dual_grasp_latched"])),
                        float(bool(metrics["lift_latched"])),
                        float(bool(metrics["rotation_latched"])),
                        float(bool(metrics["left_role_owned"])),
                        float(bool(metrics["right_role_owned"])),
                        float(bool(metrics["failed"])),
                        float(bool(metrics["success_now"])),
                    ],
                    dtype=np.float64,
                ),
            }
        )
        return observation

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        metrics = self._bc007_last_metrics or self._bc007_metrics(update=False)
        info.update(
            {
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "task_stage": metrics["stage_name"],
                "task_stage_index": int(metrics["stage"]),
                "failure_reason": str(metrics["failure_reason"]),
                "success_steps_required": STABILITY_STEPS,
            }
        )
        for key, value in metrics.items():
            if isinstance(value, (bool, np.bool_)):
                info[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                info[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                info[key] = float(value)
            elif isinstance(value, str):
                info[key] = value
        return info

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        recorded_assist = state_parts.get("physx_task_controller_state")
        if recorded_assist is not None:
            decoded = decode_replay_assist_state(np.asarray(recorded_assist))
            if any(
                decoded[side][field]
                for side in ("left", "right")
                for field in ("owner", "frame_armed", "release_pending")
            ):
                raise ValueError(
                    "BC007 recorded initial state contains an active grasp assist"
                )
        self._clear_bc007_assist_state(disable_joints=True)
        super().restore_recorded_initial_state(state_parts)
        self._clear_bc007_assist_state(disable_joints=True)
        pose = np.asarray(state_parts.get("rod_pose", np.zeros(0)), dtype=np.float64).ravel()
        view = self._task_body_views.get(ROD_NAME)
        if pose.size == 7 and view is not None:
            self._set_task_body_world_pose(view, pose)
        if self._sim is not None:
            self._sim.forward()
        if not self._wake_rod_after_reset():
            raise RuntimeError("BC007 recorded restore could not wake the long rod")
        for robot in self._robots.values():
            robot.update(0.0)
        self._refresh_object_poses()
        self._bc007_evaluator.reset()
        self._bc007_last_metrics = {}
        return self._compute_observation()


__all__ = ["BC007IsaacLabEnv"]
