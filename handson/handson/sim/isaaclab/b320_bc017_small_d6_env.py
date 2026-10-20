"""Dedicated IsaacLab runtime for B320-BC017."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from gymnasium import spaces
import numpy as np

from .b320_bc017_small_d6_contract import (
    BC017Calibration,
    BC017_CONTRACT_SCHEMA_DIGEST_SHA256,
    BC017_CONTRACT_SCHEMA_NAME,
    BC017_CONTRACT_SCHEMA_VERSION,
    BC017Frame,
    BC017State,
    CONTACT_KEYS,
    BC017_OBSERVATION_DTYPE,
    BC017_PROPRIO_FLATTEN_ORDER,
    DrawerTravelCalibration,
    HIDDEN_OBSERVATION_FIELDS,
    HIDDEN_OBSERVATION_SHAPE,
    PHYSICS_DT_SECONDS,
    PHYSICS_HZ,
    PUBLIC_OBSERVATION_FIELDS,
    PUBLIC_OBSERVATION_GROUPS,
    PUBLIC_OBSERVATION_SHAPE,
    STABILITY_STEPS,
    SYNC_WINDOW_SECONDS,
    SYNC_WINDOW_STEPS,
    TASK_NAME,
    TASK_REQUIREMENT,
    bc017_observation_layout_metadata,
)
from .b320_bc017_small_d6_scene import (
    BC017_ASSET_ORIENTATION_WXYZ,
    BC017_ASSET_TRANSLATION_M,
    BC017_DRAWER_TRAVEL_M,
    BC017_GRASP_D6_AXES,
    BC017_GRASP_FREE_LIMIT_BY_AXIS,
    BC017_GRASP_LOCK_LIMIT_BY_AXIS,
    BC017_LEFT_JOINT_NAME,
    BC017_RIGHT_JOINT_NAME,
    SELECTED_ASSET_CREATOR,
    SELECTED_ASSET_LICENSE,
    SELECTED_ASSET_TITLE,
    SELECTED_ASSET_URL,
    author_bc017_fingertip_handle_joints,
    bc017_grasp_joint_path,
    bc017_handle_grip_bounds_world,
    spawn_bc017_task_assets,
)
from .b320_bimanual_env import (
    HandsOnB320BimanualIsaacLabEnv,
    _array_like,
    _indices_like,
    _tensor_to_numpy,
)
from .bimanual_assembly_env import (
    _matrix_to_quat_wxyz,
    _pose_isaaclab_to_wxyz_array,
    _quat_wxyz_to_matrix,
)


_DOF_NAMES = (BC017_LEFT_JOINT_NAME, BC017_RIGHT_JOINT_NAME)
_PUBLIC_GROUP_FIELDS = dict(PUBLIC_OBSERVATION_GROUPS)
_DRAWER_STATE_FIELDS = _PUBLIC_GROUP_FIELDS["bc017_drawer_state"]
_CONTACT_STATE_FIELDS = _PUBLIC_GROUP_FIELDS["bc017_contact_state"]
_STAGE_STATE_FIELDS = _PUBLIC_GROUP_FIELDS["bc017_stage_state"]
_TERMINAL_STATE_FIELDS = _PUBLIC_GROUP_FIELDS["bc017_terminal_state"]
_PUBLIC_INDEX = {
    name: index for index, name in enumerate(PUBLIC_OBSERVATION_FIELDS)
}
_OBSERVATION_KEYS = BC017_PROPRIO_FLATTEN_ORDER
_TASK_STATE_REPLAY_ATOL = {
    key: (0.0 if key == "bc017_hidden_state" else 1.0e-6)
    for key in _OBSERVATION_KEYS
}
_ORIGIN_Q_TOLERANCE_M = 1.0e-6
_ORIGIN_QD_TOLERANCE_M_S = 1.0e-8
# USD ``float`` attributes are exposed through PXR with their binary32
# representation (for example authored 0.16 becomes 0.1599999964237213).
# Keep the live/composed-stage comparison identical to the authoritative
# pre-mutation USD audit instead of applying a stricter binary64 tolerance.
_USD_LIMIT_ABS_TOLERANCE_M = 1.0e-7

# A BC017 attachment is deliberately narrower than the task's ordinary
# assigned-hand contact truth.  Only a distal link4 contact point inside the
# source handle's protruding pull tab may activate it, and the visible
# fingertip center must agree with that manifold point.  Once attached, the
# a pre-authored external D6 joint locks all six relative axes between that
# distal finger rigid body and the drawer.  Arm, palm, and remaining finger
# DOFs remain under normal operator control.
BC017_GRASP_RELEASE_PROGRESS = 0.60
_BC017_GRASP_CONTACT_MIN_FORCE_N = 0.01
_BC017_GRASP_CONTACT_ZONE_MARGIN_M = 0.0025
_BC017_GRASP_TIP_TO_CONTACT_MAX_M = 0.018

# Keep the neutral Wuji fingers clear of the desk while PhysX establishes the
# recording origin.  The shared home puts each palm at approximately
# (-0.3715, +/-0.2395, 0.9975) m; the long neutral fingers then reach the desk
# front and keep the hand joints moving.  This task-only home moves each palm
# 108.5 mm backward and 40.5 mm outward without changing its orientation.  It
# leaves roughly 40 mm between the neutral fingertip centers and the desk's
# front collision boundary while keeping both handles in comfortable reach.
_BC017_RESET_ARM_HOME = {
    "right": (
        -1.898050231,
        1.500932993,
        1.648349377,
        1.838314302,
        -1.482792491,
        0.056112780,
        -0.054552534,
    ),
    "left": (
        1.898050231,
        1.500932993,
        -1.648349377,
        1.838314302,
        1.482792491,
        0.056112780,
        0.054552534,
    ),
}

if BC017_OBSERVATION_DTYPE != np.dtype(np.float64).name:
    raise RuntimeError("BC017 contract/runtime observation dtype mismatch")


def _task_calibration() -> BC017Calibration:
    """Return the authored 0-160 mm travel used by the task."""

    travel = DrawerTravelCalibration(0.0, 0.16, formal=False)
    return BC017Calibration(
        left_travel=travel,
        right_travel=travel,
        formal=False,
    )


def _finite_pose(value: Any) -> tuple[np.ndarray, bool]:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(7)
    except Exception:
        return np.zeros(7, dtype=np.float64), False
    norm = float(np.linalg.norm(pose[3:7]))
    if not np.all(np.isfinite(pose)) or norm < 1.0e-8:
        return np.zeros(7, dtype=np.float64), False
    result = pose.copy()
    result[3:7] /= norm
    return result, True


def _pose_drift(current: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(current[:3] - reference[:3]))
    cosine_half = float(
        np.clip(abs(np.dot(current[3:7], reference[3:7])), 0.0, 1.0)
    )
    rotation = float(2.0 * math.acos(cosine_half))
    return translation, rotation


def bc017_grasp_release_reached(drawer_q_m: float) -> bool:
    """Return whether an attachment must permanently release this episode."""

    q_m = float(drawer_q_m)
    return bool(
        math.isfinite(q_m)
        and q_m / BC017_DRAWER_TRAVEL_M > BC017_GRASP_RELEASE_PROGRESS
    )


def bc017_fixed_joint_frame_pair(
    *,
    finger_pose_wxyz: np.ndarray,
    drawer_pose_wxyz: np.ndarray,
    contact_point_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build matching body-local joint frames at the live contact point."""

    finger_pose, finger_valid = _finite_pose(finger_pose_wxyz)
    drawer_pose, drawer_valid = _finite_pose(drawer_pose_wxyz)
    contact = np.asarray(contact_point_world_m, dtype=np.float64).reshape(3)
    if not finger_valid or not drawer_valid or not np.all(np.isfinite(contact)):
        raise ValueError("BC017 fixed-joint frame inputs must be finite poses")
    finger_rotation = _quat_wxyz_to_matrix(finger_pose[3:7])
    drawer_rotation = _quat_wxyz_to_matrix(drawer_pose[3:7])
    local_pos0 = finger_rotation.T @ (contact - finger_pose[:3])
    local_pos1 = drawer_rotation.T @ (contact - drawer_pose[:3])
    # Use the current finger frame as the common world joint orientation.
    # This makes body0's local rotation identity and preserves the complete
    # live finger/drawer relative orientation without a snap at attachment.
    local_rot0 = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    local_rot1 = _matrix_to_quat_wxyz(
        drawer_rotation.T @ finger_rotation
    )
    return local_pos0, local_rot0, local_pos1, local_rot1


class HandsOnB320BC017IsaacLabEnv(HandsOnB320BimanualIsaacLabEnv):
    """120 Hz physical runtime for synchronized drawers."""

    _bc017_dedicated_runtime = True
    _canonical_camera_postrender_supported = True
    # BC017's origin gate must distinguish an authoritative empty global
    # report from the narrower task-sensor tensor fallback.  The latter does
    # not cover robot/robot or robot/static-table pairs, so an all-zero tensor
    # payload is not sufficient evidence for this task.
    _physx_require_full_contact_report_capture = True

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"BC017 runtime does not support {task_name!r}")
        try:
            self._bc017_state = BC017State(_task_calibration())
            self._bc017_contact_views: dict[str, Any] = {}
            self._bc017_contact_bindings: dict[str, dict[str, Any]] = {}
            self._bc017_grasp_contact_views: dict[str, Any] = {}
            self._bc017_grasp_contact_bindings: dict[str, dict[str, Any]] = {}
            self._bc017_grasp_joint_paths: dict[str, tuple[str, ...]] = {}
            self._bc017_grasp_body_indices: dict[str, int] = {}
            self._bc017_grasp_attachment: dict[str, dict[str, Any] | None] = {
                "left": None,
                "right": None,
            }
            self._bc017_grasp_released = {"left": False, "right": False}
            self._bc017_grasp_metrics: dict[str, dict[str, Any]] = {
                side: {
                    "active": False,
                    "finger": "",
                    "reason": "initializing",
                    "release_progress": BC017_GRASP_RELEASE_PROGRESS,
                }
                for side in ("left", "right")
            }
            self._bc017_robot_contact_report_paths_preconfigured: (
                tuple[str, ...]
            ) = ()
            self._bc017_contact_force_n = {key: 0.0 for key in CONTACT_KEYS}
            self._bc017_contact_readable = {
                key: False for key in CONTACT_KEYS
            }
            self._bc017_q_m = np.zeros(2, dtype=np.float64)
            self._bc017_qd_m_s = np.zeros(2, dtype=np.float64)
            self._bc017_dof_readable = np.zeros(2, dtype=bool)
            self._bc017_fixture_pose_wxyz = np.zeros(7, dtype=np.float64)
            self._bc017_fixture_truth_readable = False
            self._bc017_fixture_origin_candidate: np.ndarray | None = None
            self._bc017_recording_origin_fixture_pose: np.ndarray | None = None
            self._bc017_recording_origin_q_m = np.zeros(2, dtype=np.float64)
            self._bc017_recording_origin_accepted = False
            self._bc017_recording_origin_sha256: str | None = None
            self._bc017_runtime_integrity_failure: str | None = None
            self._bc017_physics_step = 0
            self._bc017_expected_substep_index = 0
            self._bc017_reset_write_count = 0
            super().__init__(task_name=task_name, **kwargs)
        except BaseException:
            try:
                # A partially initialized base may already own USD/Hydra
                # references.  Give it the first chance to release them; the
                # private asset bytes must outlive that teardown.
                super().close()
            except BaseException:
                pass
            raise

    def physical_replay_post_substep_observer_is_safe(
        self,
        callback: Any,
    ) -> bool:
        return bool(
            getattr(callback, "__self__", None) is self
            and getattr(callback, "__func__", None)
            is HandsOnB320BC017IsaacLabEnv._handson_post_physics_substep_callback
        )

    def _configure_task_robot_cfg(self, robot_cfg: Any, side: str) -> Any:
        """Install BC017's collision-clear reset home before USD creation."""

        robot_cfg = super()._configure_task_robot_cfg(robot_cfg, side)
        try:
            arm_home = _BC017_RESET_ARM_HOME[side]
        except KeyError as exc:
            raise ValueError(f"unsupported BC017 robot side: {side!r}") from exc
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None)
        if not isinstance(joint_pos, dict):
            raise RuntimeError(
                f"BC017 {side} robot config has no writable joint home"
            )
        for index, value in enumerate(arm_home, start=1):
            joint_pos[f"joint{index}_{side}(_1)?"] = float(value)
        return robot_cfg

    def _task_articulation_root_is_writable(self, path: str) -> bool:
        """The manifest fixture is world-fixed; only its two DOFs may reset."""

        if self.task_name == TASK_NAME:
            return False
        return super()._task_articulation_root_is_writable(path)

    def _task_articulation_dofs_are_writable(self, path: str) -> bool:
        del path
        return True

    def _configure_task_specific_contact_reporting(self) -> int:
        """Attach BC017 ground-truth reporters before the first PhysX reset.

        The live acceptance validator treats every enabled robot rigid body as
        a contact-view sensor when checking the complete robot/fixture and
        robot/robot envelopes.  PhysX requires ``PhysxContactReportAPI`` on
        those exact sensor links before tensor creation; adding it after the
        first reset can invalidate existing tensor views.  Ordinary previews
        do not pay for the extra reporters because they do not enable the
        recording ground-truth contract.
        """

        configured = super()._configure_task_specific_contact_reporting()
        self._bc017_robot_contact_report_paths_preconfigured = ()
        if not self._physx_ground_truth_enabled:
            return configured
        if self._stage is None:
            raise RuntimeError(
                "BC017 robot contact reporting requires the constructed USD stage"
            )

        from pxr import PhysxSchema, Usd, UsdPhysics

        report_paths: list[str] = []
        for side in ("Left", "Right"):
            root_path = f"/World/HandsOnRobots/{side}Robot"
            root = self._stage.GetPrimAtPath(root_path)
            if root is None or not root.IsValid():
                raise RuntimeError(
                    f"BC017 robot contact-report root is missing: {root_path}"
                )
            side_paths: list[str] = []
            for prim in Usd.PrimRange(root):
                if not prim.IsActive() or not prim.HasAPI(
                    UsdPhysics.RigidBodyAPI
                ):
                    continue
                rigid = UsdPhysics.RigidBodyAPI(prim)
                enabled = rigid.GetRigidBodyEnabledAttr()
                if (
                    enabled
                    and enabled.IsValid()
                    and enabled.HasAuthoredValueOpinion()
                    and not bool(enabled.Get())
                ):
                    continue
                report = (
                    PhysxSchema.PhysxContactReportAPI.Get(
                        self._stage, prim.GetPath()
                    )
                    if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                    else PhysxSchema.PhysxContactReportAPI.Apply(prim)
                )
                if not report:
                    raise RuntimeError(
                        "BC017 could not attach a pre-reset robot contact "
                        f"reporter: {prim.GetPath()}"
                    )
                threshold = report.CreateThresholdAttr()
                if not threshold.Set(0.0):
                    raise RuntimeError(
                        "BC017 could not author a zero contact-report threshold: "
                        f"{prim.GetPath()}"
                    )
                threshold_value = threshold.Get()
                if threshold_value is None or not math.isclose(
                    float(threshold_value), 0.0, rel_tol=0.0, abs_tol=0.0
                ):
                    raise RuntimeError(
                        "BC017 contact-report threshold did not round-trip as zero: "
                        f"{prim.GetPath()}"
                    )
                path = str(prim.GetPath())
                side_paths.append(path)
                report_paths.append(path)
            if len(side_paths) < 2:
                raise RuntimeError(
                    f"BC017 {side.lower()} robot exposes fewer than two enabled "
                    "rigid bodies"
                )

        self._bc017_robot_contact_report_paths_preconfigured = tuple(
            sorted(set(report_paths))
        )
        return configured + len(
            self._bc017_robot_contact_report_paths_preconfigured
        )

    def _configure_ground_truth_contact_reports(self, task_mjcf: Path) -> None:
        """Publish the exact pre-reset BC017 full-report coverage.

        The shared setup owns task-body reporter selection.  BC017 additionally
        authors one reporter on every enabled robot rigid body in
        ``_configure_task_specific_contact_reporting``.  Merge those exact
        paths into the trace contract after the shared task reporters have
        been configured so the recorder can prove that a source-1 empty frame
        covers robot/fixture, robot/robot, drawer/fixture, and drawer/table
        pairs.  No USD API is added here; this method only publishes reporters
        that were already authored before the first ``sim.reset``.
        """

        super()._configure_ground_truth_contact_reports(task_mjcf)
        robot_paths = tuple(
            sorted(
                set(
                    map(
                        str,
                        self._bc017_robot_contact_report_paths_preconfigured,
                    )
                )
            )
        )
        if self._physx_ground_truth_enabled and len(robot_paths) < 4:
            raise RuntimeError(
                "BC017 full contact report lacks pre-reset robot coverage"
            )
        shared_task_paths = tuple(
            map(
                str,
                getattr(
                    self,
                    "_physx_contact_report_paths_preconfigured",
                    (),
                ),
            )
        )
        task_body_paths = tuple(
            sorted(
                set(
                    map(
                        str,
                        self._b320_spawn_result.body_paths.values(),
                    )
                )
            )
        )
        if self._physx_ground_truth_enabled and len(task_body_paths) != 3:
            raise RuntimeError(
                "BC017 full contact report lacks exact three-body task coverage"
            )
        self._physx_contact_report_paths_preconfigured = tuple(
            sorted(
                set(shared_task_paths)
                .union(task_body_paths)
                .union(robot_paths)
            )
        )

    def _pre_simulation_context_task_audit(self) -> None:
        """BC017 has no task-specific preflight or acceptance audit."""

        return None

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("BC017 stage is unavailable before asset spawn")
        self._b320_spawn_result = spawn_bc017_task_assets(
            self._stage,
            sim_utils,
        )
        self._bc017_state = BC017State(_task_calibration())
        return

    def _configure_task_specific_pre_physx(self) -> bool:
        """Pre-author free external D6 joints after robot prims exist."""

        if self._stage is None:
            raise RuntimeError("BC017 fingertip joints require a USD stage")
        drawer_paths = {
            "left": self._b320_spawn_result.body_paths.get("left_drawer", ""),
            "right": self._b320_spawn_result.body_paths.get("right_drawer", ""),
        }
        self._bc017_grasp_joint_paths = author_bc017_fingertip_handle_joints(
            self._stage,
            drawer_paths=drawer_paths,
        )
        if any(
            len(self._bc017_grasp_joint_paths.get(side, ())) != 5
            for side in ("left", "right")
        ):
            raise RuntimeError(
                "BC017 did not pre-author all ten fingertip D6 joints"
            )
        root = self._stage.GetPrimAtPath(
            self._b320_spawn_result.body_paths["fixture"]
        ).GetParent()
        if root.IsValid():
            import json

            root.SetCustomDataByKey(
                "handson:fingertipHandleGraspJoints",
                json.dumps(
                    self._bc017_grasp_joint_paths,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        return True

    def _bc017_fingertip_body_paths(self, side: str) -> tuple[str, ...]:
        """Resolve only the five distal link4 rigid bodies for one hand."""

        if self._stage is None:
            return ()
        from pxr import Usd, UsdPhysics

        normalized_side = str(side).lower()
        if normalized_side not in {"left", "right"}:
            raise ValueError(f"unsupported BC017 hand side: {side!r}")
        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{normalized_side.capitalize()}Robot"
        )
        if root is None or not root.IsValid():
            return ()
        expected = {
            f"{normalized_side}_finger{finger}_link4": finger
            for finger in range(1, 6)
        }
        paths = {
            prim.GetName(): str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and prim.GetName() in expected
        }
        return tuple(
            paths[name]
            for name, _finger in sorted(
                expected.items(),
                key=lambda item: item[1],
            )
            if name in paths
        )

    def _init_task_specific_runtime(self) -> None:
        self._bc017_contact_views = {}
        self._bc017_contact_bindings = {}
        self._bc017_grasp_contact_views = {}
        self._bc017_grasp_contact_bindings = {}
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            self._add_task_state_warning("BC017 PhysX contact view is unavailable")
            return
        bindings = (
            ("LL", "left_drawer", "left"),
            ("LR", "right_drawer", "left"),
            ("RL", "left_drawer", "right"),
            ("RR", "right_drawer", "right"),
        )
        for key, drawer_role, hand_side in bindings:
            sensor = self._b320_spawn_result.body_paths.get(drawer_role)
            filters = tuple(self._hand_body_paths(hand_side))
            binding = {
                "drawer_role": drawer_role,
                "hand_side": hand_side,
                "sensor_path": str(sensor or ""),
                "filter_paths": list(filters),
                "sensor_count": 0,
                "filter_count": 0,
                "bound": False,
            }
            self._bc017_contact_bindings[key] = binding
            if not sensor or not filters:
                self._add_task_state_warning(
                    f"BC017 {key} contact binding is incomplete"
                )
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    str(sensor),
                    filter_patterns=list(filters),
                    max_contact_data_count=max(128, len(filters) * 16),
                )
                sensor_count = int(getattr(view, "sensor_count", 0))
                filter_count = int(getattr(view, "filter_count", 0))
                bound = bool(
                    sensor_count == 1 and filter_count == len(filters)
                )
                binding.update(
                    {
                        "sensor_count": sensor_count,
                        "filter_count": filter_count,
                        "bound": bound,
                    }
                )
                if bound:
                    self._bc017_contact_views[key] = view
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not initialize BC017 {key} contact view: {exc}"
                )
        missing = sorted(set(CONTACT_KEYS) - set(self._bc017_contact_views))
        if missing:
            self._add_task_state_warning(
                "BC017 four-channel contact truth is incomplete; success "
                "fails closed: " + ",".join(missing)
            )

        # Separate narrow views retain filter-index identity for the five
        # distal fingers.  The broad four-channel views above continue to own
        # task-success semantics; these views only gate the requested handle
        # attachment and cannot be triggered by palms or proximal phalanges.
        for side, drawer_role in (
            ("left", "left_drawer"),
            ("right", "right_drawer"),
        ):
            sensor = self._b320_spawn_result.body_paths.get(drawer_role)
            filters = self._bc017_fingertip_body_paths(side)
            binding = {
                "drawer_role": drawer_role,
                "hand_side": side,
                "sensor_path": str(sensor or ""),
                "filter_paths": list(filters),
                "sensor_count": 0,
                "filter_count": 0,
                "bound": False,
                "strict_distal_link4_only": True,
            }
            self._bc017_grasp_contact_bindings[side] = binding
            if not sensor or len(filters) != 5:
                self._add_task_state_warning(
                    f"BC017 {side} fingertip/handle binding is incomplete"
                )
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    str(sensor),
                    filter_patterns=list(filters),
                    max_contact_data_count=160,
                )
                sensor_count = int(getattr(view, "sensor_count", 0))
                filter_count = int(getattr(view, "filter_count", 0))
                bound = bool(sensor_count == 1 and filter_count == len(filters))
                binding.update(
                    {
                        "sensor_count": sensor_count,
                        "filter_count": filter_count,
                        "bound": bound,
                    }
                )
                if bound:
                    self._bc017_grasp_contact_views[side] = view
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not initialize BC017 {side} fingertip/handle view: {exc}"
                )
        missing_grasp = sorted(
            {"left", "right"} - set(self._bc017_grasp_contact_views)
        )
        if missing_grasp:
            self._add_task_state_warning(
                "BC017 strict fingertip/handle attachment is incomplete: "
                + ",".join(missing_grasp)
            )

    def _reset_bc017_runtime_state(self) -> None:
        self._free_all_bc017_grasp_joints(record=False)
        self._bc017_state.reset()
        self._bc017_contact_force_n = {key: 0.0 for key in CONTACT_KEYS}
        self._bc017_contact_readable = {key: False for key in CONTACT_KEYS}
        self._bc017_q_m = np.zeros(2, dtype=np.float64)
        self._bc017_qd_m_s = np.zeros(2, dtype=np.float64)
        self._bc017_dof_readable = np.zeros(2, dtype=bool)
        self._bc017_fixture_pose_wxyz = np.zeros(7, dtype=np.float64)
        self._bc017_fixture_truth_readable = False
        self._bc017_fixture_origin_candidate = None
        self._bc017_recording_origin_fixture_pose = None
        self._bc017_recording_origin_q_m = np.zeros(2, dtype=np.float64)
        self._bc017_recording_origin_accepted = False
        self._bc017_recording_origin_sha256 = None
        self._bc017_runtime_integrity_failure = None
        self._bc017_physics_step = 0
        self._bc017_expected_substep_index = 0
        self._bc017_grasp_attachment = {"left": None, "right": None}
        self._bc017_grasp_released = {"left": False, "right": False}
        self._bc017_grasp_metrics = {
            side: {
                "active": False,
                "finger": "",
                "reason": "reset_waiting_for_pull_tab_fingertip_contact",
                "release_progress": BC017_GRASP_RELEASE_PROGRESS,
            }
            for side in ("left", "right")
        }

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._reset_bc017_runtime_state()

    def _apply_task_pose_randomization(self) -> None:
        # Shared B320 restores captured articulation state while protecting the
        # fixed root.  BC017 then overwrites only the two writable DOFs with
        # manifest-authored closed positions and zero velocities.
        super()._apply_task_pose_randomization()
        if self._launch_isaac:
            self._write_bc017_closed_dofs()

    def _write_bc017_closed_dofs(self) -> None:
        if self._torch is None:
            raise RuntimeError("BC017 cannot reset without the tensor runtime")
        exact_topology, topology = self._bc017_exact_dof_topology()
        if not exact_topology:
            raise RuntimeError(
                "BC017 cannot reset a non-canonical DOF topology: "
                f"{topology!r}"
            )
        entries = [self._task_dof_views.get(name) for name in _DOF_NAMES]
        if any(entry is None for entry in entries):
            missing = [name for name, entry in zip(_DOF_NAMES, entries) if entry is None]
            raise RuntimeError("BC017 exact reset DOFs are unavailable: " + ",".join(missing))
        left_entry, right_entry = entries
        assert left_entry is not None and right_entry is not None
        if left_entry[0] is not right_entry[0]:
            raise RuntimeError("BC017 drawer DOFs are not in one fixed articulation")
        view = left_entry[0]
        position_ref = view.get_dof_positions()
        velocity_ref = view.get_dof_velocities()
        positions = _tensor_to_numpy(position_ref)
        velocities = np.zeros_like(_tensor_to_numpy(velocity_ref))
        if positions.ndim != 2 or velocities.shape != positions.shape:
            raise RuntimeError("BC017 articulation tensors have invalid shapes")
        indices_by_name = (int(left_entry[1]), int(right_entry[1]))
        if len(set(indices_by_name)) != 2 or any(
            index < 0 or index >= positions.shape[1] for index in indices_by_name
        ):
            raise RuntimeError("BC017 exact DOF indices are invalid or aliased")
        closed = (
            self._bc017_state.calibration.left_travel.closed_q_m,
            self._bc017_state.calibration.right_travel.closed_q_m,
        )
        for index, value in zip(indices_by_name, closed):
            positions[:, index] = float(value)
        indices = _indices_like(position_ref, int(view.count), self._torch)
        velocity_payload = _array_like(velocity_ref, velocities, self._torch)
        position_payload = _array_like(position_ref, positions, self._torch)
        view.set_dof_velocities(velocity_payload, indices)
        view.set_dof_positions(position_payload, indices)
        reset_path = next(
            (
                str(path)
                for path, candidate, *_state in self._task_reset_articulation_states
                if candidate is view
            ),
            None,
        )
        if reset_path is None:
            raise RuntimeError("BC017 reset articulation is absent from the journal")
        self._record_authored_task_reset_write(
            reset_path,
            dof_position=positions,
            dof_velocity=velocities,
        )
        self._bc017_reset_write_count += 1

    def _bc017_exact_dof_topology(self) -> tuple[bool, dict[str, Any]]:
        """Require one articulation with exactly the two canonical DOFs."""

        names_exact = set(self._task_dof_views) == set(_DOF_NAMES)
        entries = [self._task_dof_views.get(name) for name in _DOF_NAMES]
        same_view = bool(
            all(entry is not None for entry in entries)
            and entries[0][0] is entries[1][0]
        )
        distinct_indices = bool(
            same_view
            and {int(entries[0][1]), int(entries[1][1])} == {0, 1}
        )
        tensor_dof_count = -1
        metatype_names: set[str] = set()
        if same_view:
            view = entries[0][0]
            try:
                tensor_dof_count = int(
                    _tensor_to_numpy(view.get_dof_positions()).shape[-1]
                )
            except Exception:
                tensor_dof_count = -1
            try:
                metatype = view.get_metatype(0)
                metatype_names = set(getattr(metatype, "dof_indices", {}))
            except Exception:
                metatype_names = set()
        metatype_exact = metatype_names == set(_DOF_NAMES)
        ready = bool(
            names_exact
            and same_view
            and distinct_indices
            and tensor_dof_count == 2
            and metatype_exact
        )
        return ready, {
            "mapping_names_exact": names_exact,
            "same_articulation_view": same_view,
            "distinct_indices_0_1": distinct_indices,
            "tensor_dof_count": tensor_dof_count,
            "metatype_dof_names": sorted(metatype_names),
            "metatype_names_exact": metatype_exact,
        }

    def _bc017_physics_dof_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read only the two exact manifest joint names; never fuzzy-match."""

        q = np.zeros(2, dtype=np.float64)
        qd = np.zeros(2, dtype=np.float64)
        readable = np.zeros(2, dtype=bool)
        for output_index, name in enumerate(_DOF_NAMES):
            entry = self._task_dof_views.get(name)
            if entry is None:
                continue
            view, dof_index, _initial = entry
            try:
                positions = _tensor_to_numpy(view.get_dof_positions())
                velocities = _tensor_to_numpy(view.get_dof_velocities())
                index = int(dof_index)
                value = float(positions[0, index])
                velocity = float(velocities[0, index])
            except Exception:
                continue
            # Successful tensor readback is "readable" even when its value is
            # malformed.  The contract must attribute non-finite truth to its
            # dedicated first-cause code rather than downgrading it to missing.
            q[output_index] = value
            qd[output_index] = velocity
            readable[output_index] = True
        return q, qd, readable

    def _bc017_fixture_pose(self) -> tuple[np.ndarray, bool]:
        entry = self._task_articulation_views.get("fixture")
        if entry is not None:
            try:
                transform = _tensor_to_numpy(entry[0].get_root_transforms())[0]
                transform = np.asarray(transform, dtype=np.float64).reshape(-1)
                if transform.shape != (7,):
                    return np.full(7, np.nan, dtype=np.float64), True
                return (
                    np.asarray(
                        [
                            *transform[:3],
                            transform[6],
                            transform[3],
                            transform[4],
                            transform[5],
                        ],
                        dtype=np.float64,
                    ),
                    True,
                )
            except Exception:
                pass
        body_view = self._task_body_views.get("fixture")
        if body_view is None:
            return np.zeros(7, dtype=np.float64), False
        try:
            transform = _tensor_to_numpy(body_view.get_transforms())[0]
            transform = np.asarray(transform, dtype=np.float64).reshape(-1)
            if transform.shape != (7,):
                return np.full(7, np.nan, dtype=np.float64), True
            return np.asarray(
                [
                    *transform[:3],
                    transform[6],
                    transform[3],
                    transform[4],
                    transform[5],
                ],
                dtype=np.float64,
            ), True
        except Exception:
            return np.zeros(7, dtype=np.float64), False

    @staticmethod
    def _read_bc017_contact_force(view: Any | None, dt: float) -> tuple[float, bool]:
        """Preserve invalid readback as invalid, not as unavailable truth."""

        if view is None:
            return 0.0, False
        try:
            raw = _tensor_to_numpy(view.get_contact_force_matrix(dt)).astype(
                np.float64
            )
        except Exception:
            return 0.0, False
        if raw.size == 0:
            return 0.0, False
        if raw.ndim == 0:
            return float(raw), True
        if not np.all(np.isfinite(raw)):
            return float("nan"), True
        return float(np.linalg.norm(raw, axis=-1).sum()), True

    def _bc017_grasp_contact_samples(
        self,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read world contact points, forces, and exact distal-filter indices."""

        view = self._bc017_grasp_contact_views.get(side)
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
            forces_np = np.abs(
                _tensor_to_numpy(normal_forces).astype(np.float64).reshape(-1)
            )
            points_np = (
                _tensor_to_numpy(points).astype(np.float64).reshape(-1, 3)
            )
            shape = (int(view.sensor_count), int(view.filter_count))
            counts_np = (
                _tensor_to_numpy(counts).astype(np.int64).reshape(shape)
            )
            starts_np = (
                _tensor_to_numpy(starts).astype(np.int64).reshape(shape)
            )
        except Exception as exc:
            self._add_task_state_warning(
                f"BC017 {side} fingertip contact readback failed: {exc}"
            )
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )

        selected_points: list[np.ndarray] = []
        selected_forces: list[np.ndarray] = []
        selected_filters: list[np.ndarray] = []
        filter_count = int(view.filter_count)
        for flat_index, (count, start) in enumerate(
            zip(counts_np.ravel(), starts_np.ravel(), strict=True)
        ):
            count_int = int(count)
            start_int = int(start)
            if count_int <= 0 or start_int < 0:
                continue
            stop = start_int + count_int
            if stop > len(points_np) or stop > len(forces_np):
                continue
            selected_points.append(points_np[start_int:stop])
            selected_forces.append(forces_np[start_int:stop])
            selected_filters.append(
                np.full(
                    count_int,
                    flat_index % filter_count,
                    dtype=np.int64,
                )
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
            np.concatenate(selected_filters),
        )

    @staticmethod
    def _bc017_point_inside_bounds(
        point: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        *,
        margin_m: float,
    ) -> bool:
        value = np.asarray(point, dtype=np.float64).reshape(3)
        return bool(
            np.all(np.isfinite(value))
            and np.all(value >= minimum - margin_m)
            and np.all(value <= maximum + margin_m)
        )

    def _bc017_grasp_candidate(
        self,
        side: str,
        drawer_q_m: float,
    ) -> dict[str, Any] | None:
        """Return a real fingertip contact on the protruding pull tab only."""

        points, forces, filter_indices = self._bc017_grasp_contact_samples(side)
        binding = self._bc017_grasp_contact_bindings.get(side, {})
        filter_paths = tuple(map(str, binding.get("filter_paths", ())))
        if points.size == 0 or not filter_paths:
            return None
        try:
            tips = self._wuji_fingertip_positions(side)
        except Exception:
            return None
        # PhysX can report a few float32 ulps outside an authored hard stop.
        # Clamp only for moving the geometric query bounds; task truth keeps
        # the unmodified DOF value and its existing limit checks.
        bounds_q_m = float(
            np.clip(drawer_q_m, 0.0, BC017_DRAWER_TRAVEL_M)
        )
        bounds = bc017_handle_grip_bounds_world(side, bounds_q_m)
        minimum = np.asarray(bounds[0], dtype=np.float64)
        maximum = np.asarray(bounds[1], dtype=np.float64)
        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for point, force, filter_index in zip(
            points,
            forces,
            filter_indices,
            strict=True,
        ):
            if (
                not math.isfinite(float(force))
                or float(force) < _BC017_GRASP_CONTACT_MIN_FORCE_N
                or not self._bc017_point_inside_bounds(
                    point,
                    minimum,
                    maximum,
                    margin_m=_BC017_GRASP_CONTACT_ZONE_MARGIN_M,
                )
            ):
                continue
            index = int(filter_index)
            if index < 0 or index >= len(filter_paths):
                continue
            body_path = filter_paths[index]
            body_name = body_path.rsplit("/", 1)[-1]
            if not body_name.startswith(f"{side}_finger") or not body_name.endswith(
                "_link4"
            ):
                continue
            tip_name = body_name.removesuffix("_link4") + "_tip"
            tip_position = tips.get(tip_name)
            if tip_position is None:
                continue
            tip_position = np.asarray(tip_position, dtype=np.float64).reshape(3)
            tip_contact_distance = float(np.linalg.norm(tip_position - point))
            if (
                not math.isfinite(tip_contact_distance)
                or tip_contact_distance > _BC017_GRASP_TIP_TO_CONTACT_MAX_M
            ):
                continue
            candidates.append(
                (
                    tip_contact_distance,
                    -float(force),
                    {
                        "finger": tip_name,
                        "finger_body_path": body_path,
                        "tip_position_m": tip_position.copy(),
                        "contact_point_m": np.asarray(
                            point, dtype=np.float64
                        ).copy(),
                        "contact_force_n": float(force),
                        "tip_contact_distance_m": tip_contact_distance,
                    },
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    def _bc017_finger_body_pose_wxyz(
        self,
        side: str,
        body_path: str,
    ) -> np.ndarray | None:
        robot = self._robots.get(side)
        if robot is None:
            return None
        body_name = str(body_path).rsplit("/", 1)[-1]
        cache_key = f"{side}:{body_name}"
        body_index = self._bc017_grasp_body_indices.get(cache_key)
        if body_index is None:
            try:
                body_index = next(
                    index
                    for index, name in enumerate(robot.body_names)
                    if str(name) == body_name or body_name in str(name)
                )
            except (StopIteration, TypeError):
                return None
            self._bc017_grasp_body_indices[cache_key] = int(body_index)
        try:
            pose = robot.data.body_pose_w[0, int(body_index), :7]
            if hasattr(pose, "detach"):
                pose = pose.detach().cpu().numpy()
            pose_wxyz = _pose_isaaclab_to_wxyz_array(
                np.asarray(pose, dtype=np.float64).reshape(7),
                uses_xyzw=self._isaaclab_uses_xyzw,
            )
        except Exception:
            return None
        normalized, valid = _finite_pose(pose_wxyz)
        return normalized if valid else None

    def _bc017_drawer_body_pose_wxyz(
        self,
        drawer_q_m: float,
    ) -> np.ndarray | None:
        fixture_pose, readable = self._bc017_fixture_pose()
        fixture_pose, valid = _finite_pose(fixture_pose)
        if not readable or not valid:
            fixture_pose = np.asarray(
                [*BC017_ASSET_TRANSLATION_M, *BC017_ASSET_ORIENTATION_WXYZ],
                dtype=np.float64,
            )
            fixture_pose, valid = _finite_pose(fixture_pose)
        if not valid or not math.isfinite(float(drawer_q_m)):
            return None
        rotation = _quat_wxyz_to_matrix(fixture_pose[3:7])
        position = fixture_pose[:3] + rotation @ np.asarray(
            [float(drawer_q_m), 0.0, 0.0], dtype=np.float64
        )
        return np.asarray([*position, *fixture_pose[3:7]], dtype=np.float64)

    def _set_bc017_grasp_joint_locked(
        self,
        side: str,
        candidate: dict[str, Any],
        drawer_q_m: float,
    ) -> str | None:
        """Lock a pre-authored external D6 joint at the live contact pose."""

        if self._stage is None:
            return None
        from pxr import Gf, Sdf, UsdPhysics

        body_path = str(candidate["finger_body_path"])
        body_name = body_path.rsplit("/", 1)[-1]
        joint_path = bc017_grasp_joint_path(side, body_name)
        if joint_path not in self._bc017_grasp_joint_paths.get(side, ()):
            self._add_task_state_warning(
                f"BC017 {side} contact resolved an undeclared grasp joint"
            )
            return None
        finger_pose = self._bc017_finger_body_pose_wxyz(side, body_path)
        drawer_pose = self._bc017_drawer_body_pose_wxyz(drawer_q_m)
        if finger_pose is None or drawer_pose is None:
            self._add_task_state_warning(
                f"BC017 {side} grasp pose readback is unavailable"
            )
            return None
        contact_point = np.asarray(
            candidate["contact_point_m"], dtype=np.float64
        ).reshape(3)
        try:
            local_pos0, local_rot0, local_pos1, local_rot1 = (
                bc017_fixed_joint_frame_pair(
                    finger_pose_wxyz=finger_pose,
                    drawer_pose_wxyz=drawer_pose,
                    contact_point_world_m=contact_point,
                )
            )
            prim = self._stage.GetPrimAtPath(joint_path)
            joint = UsdPhysics.Joint(prim)
            if not prim.IsValid() or not joint:
                raise RuntimeError("pre-authored D6 joint is missing")
            # PhysX observes USD change notices.  Publish the complete live
            # frame and all six lock limits in one notice so the solver never
            # sees a half-updated cross-articulation joint.
            with Sdf.ChangeBlock():
                joint.GetLocalPos0Attr().Set(
                    Gf.Vec3f(*(float(value) for value in local_pos0))
                )
                joint.GetLocalPos1Attr().Set(
                    Gf.Vec3f(*(float(value) for value in local_pos1))
                )
                joint.GetLocalRot0Attr().Set(
                    Gf.Quatf(
                        float(local_rot0[0]),
                        Gf.Vec3f(
                            *(float(value) for value in local_rot0[1:4])
                        ),
                    )
                )
                joint.GetLocalRot1Attr().Set(
                    Gf.Quatf(
                        float(local_rot1[0]),
                        Gf.Vec3f(
                            *(float(value) for value in local_rot1[1:4])
                        ),
                    )
                )
                for axis in BC017_GRASP_D6_AXES:
                    limit = UsdPhysics.LimitAPI.Get(prim, axis)
                    if not limit:
                        raise RuntimeError(
                            f"missing pre-authored {axis} limit"
                        )
                    lock_limit = BC017_GRASP_LOCK_LIMIT_BY_AXIS[axis]
                    limit.GetLowAttr().Set(-lock_limit)
                    limit.GetHighAttr().Set(lock_limit)
            prim.SetCustomDataByKey("handson:runtimeLocked", True)
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not lock BC017 {side} fingertip/handle D6 joint: {exc}"
            )
            return None
        self._record_trajectory_state_write(
            "_lock_bc017_fingertip_handle_d6_joint",
            subject=joint_path,
        )
        return joint_path

    def _set_bc017_grasp_joint_free(
        self,
        joint_path: str,
        *,
        record: bool,
    ) -> bool:
        """Free all six axes without changing the pre-authored topology."""

        if self._stage is None or not joint_path:
            return False
        from pxr import Sdf, UsdPhysics

        try:
            prim = self._stage.GetPrimAtPath(joint_path)
            if not prim.IsValid():
                return False
            limits = [
                UsdPhysics.LimitAPI.Get(prim, axis)
                for axis in BC017_GRASP_D6_AXES
            ]
            if not all(limits):
                return False
            already_free = all(
                math.isclose(
                    float(limit.GetLowAttr().Get()),
                    -BC017_GRASP_FREE_LIMIT_BY_AXIS[axis],
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                and math.isclose(
                    float(limit.GetHighAttr().Get()),
                    BC017_GRASP_FREE_LIMIT_BY_AXIS[axis],
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                for axis, limit in zip(BC017_GRASP_D6_AXES, limits)
            )
            if not already_free:
                with Sdf.ChangeBlock():
                    for axis, limit in zip(BC017_GRASP_D6_AXES, limits):
                        free_limit = BC017_GRASP_FREE_LIMIT_BY_AXIS[axis]
                        limit.GetLowAttr().Set(-free_limit)
                        limit.GetHighAttr().Set(free_limit)
            prim.SetCustomDataByKey("handson:runtimeLocked", False)
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not free BC017 fingertip/handle D6 joint: {exc}"
            )
            return False
        if record:
            self._record_trajectory_state_write(
                "_release_bc017_fingertip_handle_d6_joint",
                subject=joint_path,
            )
        return True

    def _free_all_bc017_grasp_joints(self, *, record: bool) -> None:
        for paths in self._bc017_grasp_joint_paths.values():
            for path in paths:
                self._set_bc017_grasp_joint_free(path, record=record)

    def _clear_bc017_grasp_attachment(
        self,
        side: str,
        *,
        reason: str,
        released: bool = False,
    ) -> None:
        previous = self._bc017_grasp_attachment.get(side)
        joint_path = str((previous or {}).get("joint_path", ""))
        if joint_path:
            self._set_bc017_grasp_joint_free(joint_path, record=True)
        self._bc017_grasp_attachment[side] = None
        if released:
            self._bc017_grasp_released[side] = True
        previous_finger = str((previous or {}).get("finger", ""))
        self._bc017_grasp_metrics[side] = {
            "active": False,
            "finger": previous_finger,
            "joint_path": joint_path,
            "constraint_kind": "physx_d6_fixed_all_six_axes",
            "reason": str(reason),
            "released_for_episode": bool(self._bc017_grasp_released[side]),
            "release_progress": BC017_GRASP_RELEASE_PROGRESS,
        }
        if previous is not None:
            print(
                "[HandsOn/IsaacLab] BC017 fingertip/handle attachment released: "
                f"side={side}, finger={previous_finger}, reason={reason}",
                flush=True,
            )

    def _update_bc017_grasp_attachment(self) -> None:
        """Attach on strict pull-tab contact and release permanently above 60%."""

        if not self._launch_isaac:
            return
        q_m, _qd_m_s, readable = self._bc017_physics_dof_state()
        for index, side in enumerate(("left", "right")):
            if not bool(readable[index]) or not math.isfinite(float(q_m[index])):
                continue
            q_value = float(q_m[index])
            progress = float(np.clip(q_value / BC017_DRAWER_TRAVEL_M, 0.0, 1.0))
            if bc017_grasp_release_reached(q_value):
                self._clear_bc017_grasp_attachment(
                    side,
                    reason="drawer_progress_above_60_percent",
                    released=True,
                )
                self._bc017_grasp_metrics[side]["progress"] = progress
                continue
            state = self._bc017_grasp_attachment.get(side)
            if state is not None:
                self._bc017_grasp_metrics[side].update(
                    {
                        "active": True,
                        "progress": progress,
                        "reason": "attached_until_above_60_percent",
                    }
                )
                continue
            if self._bc017_grasp_released[side]:
                continue
            candidate = self._bc017_grasp_candidate(side, q_value)
            if candidate is None:
                self._bc017_grasp_metrics[side] = {
                    "active": False,
                    "finger": "",
                    "reason": "waiting_for_distal_fingertip_on_pull_tab",
                    "released_for_episode": False,
                    "release_progress": BC017_GRASP_RELEASE_PROGRESS,
                    "progress": progress,
                }
                continue
            joint_path = self._set_bc017_grasp_joint_locked(
                side,
                candidate,
                q_value,
            )
            if joint_path is None:
                self._bc017_grasp_metrics[side]["reason"] = (
                    "pull_tab_contact_but_d6_lock_failed"
                )
                continue
            state = {
                **candidate,
                "joint_path": joint_path,
            }
            self._bc017_grasp_attachment[side] = state
            self._bc017_grasp_metrics[side] = {
                "active": True,
                "finger": str(candidate["finger"]),
                "finger_body_path": str(candidate["finger_body_path"]),
                "joint_path": joint_path,
                "constraint_kind": "physx_d6_fixed_all_six_axes",
                "contact_point_m": np.asarray(
                    candidate["contact_point_m"], dtype=np.float64
                ).tolist(),
                "contact_force_n": float(candidate["contact_force_n"]),
                "tip_contact_distance_m": float(
                    candidate["tip_contact_distance_m"]
                ),
                "reason": "attached_on_distal_fingertip_pull_tab_contact",
                "released_for_episode": False,
                "release_progress": BC017_GRASP_RELEASE_PROGRESS,
                "progress": progress,
            }
            print(
                "[HandsOn/IsaacLab] BC017 fingertip/handle attachment active: "
                f"side={side}, finger={candidate['finger']}, "
                f"force_n={candidate['contact_force_n']:.3f}",
                flush=True,
            )

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        self._update_bc017_grasp_attachment()

    def _read_bc017_physics_truth(
        self,
        *,
        physics_step: int,
        physics_time_s: float,
        allow_fixture_candidate: bool,
    ) -> BC017Frame:
        q, qd, dof_readable = self._bc017_physics_dof_state()
        self._bc017_q_m = np.where(np.isfinite(q), q, 0.0)
        self._bc017_qd_m_s = np.where(np.isfinite(qd), qd, 0.0)
        self._bc017_dof_readable = dof_readable.copy()
        raw_forces: dict[str, float] = {}
        for key in CONTACT_KEYS:
            force, readable = self._read_bc017_contact_force(
                self._bc017_contact_views.get(key),
                self._sim_dt,
            )
            raw_forces[key] = float(force)
            self._bc017_contact_force_n[key] = (
                float(force) if math.isfinite(force) else 0.0
            )
            self._bc017_contact_readable[key] = bool(readable)
        for side, key in (("left", "LL"), ("right", "RR")):
            self._b320_contact_available[side] = key in self._bc017_contact_views
            self._b320_contact_readable[side] = self._bc017_contact_readable[key]
            self._b320_contact_force[side] = self._bc017_contact_force_n[key]

        fixture_pose, fixture_getter_readable = self._bc017_fixture_pose()
        normalized_fixture, fixture_pose_valid = _finite_pose(fixture_pose)
        self._bc017_fixture_pose_wxyz = normalized_fixture.copy()
        self._bc017_fixture_truth_readable = bool(fixture_getter_readable)
        reference = self._bc017_recording_origin_fixture_pose
        if reference is None:
            reference = self._bc017_fixture_origin_candidate
        if (
            fixture_getter_readable
            and fixture_pose_valid
            and reference is None
            and allow_fixture_candidate
        ):
            self._bc017_fixture_origin_candidate = normalized_fixture.copy()
            reference = self._bc017_fixture_origin_candidate
        if fixture_getter_readable and fixture_pose_valid and reference is not None:
            translation_drift, rotation_drift = _pose_drift(
                normalized_fixture,
                reference,
            )
            fixture_frame_readable = True
        elif fixture_getter_readable:
            # Getter success plus malformed/non-finite truth must reach
            # FIXTURE_TRUTH_INVALID in BC017State.
            translation_drift, rotation_drift = float("nan"), float("nan")
            fixture_frame_readable = True
        else:
            translation_drift, rotation_drift = 0.0, 0.0
            fixture_frame_readable = False
        return BC017Frame(
            physics_step=int(physics_step),
            physics_time_s=float(physics_time_s),
            left_q_m=float(q[0]),
            right_q_m=float(q[1]),
            left_qd_m_s=float(qd[0]),
            right_qd_m_s=float(qd[1]),
            left_dof_readable=bool(dof_readable[0]),
            right_dof_readable=bool(dof_readable[1]),
            ll_force_n=raw_forces["LL"],
            lr_force_n=raw_forces["LR"],
            rl_force_n=raw_forces["RL"],
            rr_force_n=raw_forces["RR"],
            ll_readable=self._bc017_contact_readable["LL"],
            lr_readable=self._bc017_contact_readable["LR"],
            rl_readable=self._bc017_contact_readable["RL"],
            rr_readable=self._bc017_contact_readable["RR"],
            fixture_translation_drift_m=translation_drift,
            fixture_rotation_drift_rad=rotation_drift,
            fixture_truth_readable=fixture_frame_readable,
        )

    def _handson_post_physics_substep_callback(
        self,
        substep_index: int,
        substep_count: int,
    ) -> None:
        if not self._bc017_recording_origin_accepted or self._bc017_state.terminated:
            return
        substep_continuous = bool(
            int(substep_count) == 4
            and int(substep_index) == self._bc017_expected_substep_index
        )
        next_step = self._bc017_physics_step + (1 if substep_continuous else 2)
        frame = self._read_bc017_physics_truth(
            physics_step=next_step,
            physics_time_s=next_step * PHYSICS_DT_SECONDS,
            allow_fixture_candidate=False,
        )
        self._bc017_state.update(frame)
        # Publish the integer clock only after the frame was consumed.  The
        # terminal frame remains frozen forever after this assignment.
        self._bc017_physics_step = next_step
        if substep_continuous:
            self._bc017_expected_substep_index = (
                self._bc017_expected_substep_index + 1
            ) % 4

    def _condition_flags(self) -> tuple[bool, ...]:
        snapshot = self._bc017_state.snapshot()
        left_event, right_event = snapshot.threshold_event_step
        synchronized = bool(
            left_event >= 0
            and right_event >= 0
            and abs(left_event - right_event) <= SYNC_WINDOW_STEPS
        )
        return (
            bool(
                snapshot.truth_valid
                and snapshot.semantic_valid
            ),
            bool(left_event >= 0),
            bool(right_event >= 0),
            synchronized,
            bool(snapshot.success and snapshot.stable_steps >= STABILITY_STEPS),
        )

    def _metrics(self, *, update: bool) -> dict[str, Any]:
        del update
        snapshot = self._bc017_state.snapshot()
        self._b320_condition_flags = self._condition_flags()
        values = {
            "left_progress": float(snapshot.drawer_progress[0]),
            "right_progress": float(snapshot.drawer_progress[1]),
            "left_event_step": int(snapshot.threshold_event_step[0]),
            "right_event_step": int(snapshot.threshold_event_step[1]),
            "sync_delta_steps": int(snapshot.sync_delta_steps),
            "sync_delta_seconds": float(snapshot.sync_delta_seconds),
            "stable_substeps": int(snapshot.stable_steps),
            "truth_valid": bool(snapshot.truth_valid),
            "semantic_valid": bool(snapshot.semantic_valid),
            "calibration_formal": bool(self._bc017_state.calibration.formal),
            "success_now": bool(
                snapshot.success
                and self._bc017_runtime_integrity_failure is None
            ),
        }
        self._b320_last_metrics = dict(values)
        return values

    def _success_steps_required(self) -> int:
        # BC017State owns the exact two-sample 120 Hz deglitch latch.
        return 1

    def _compute_failure(self) -> bool:
        if self._bc017_state.failure:
            evidence = self._bc017_state.terminal_evidence
            self._b320_failure_reason = (
                evidence.failure_reason.name.lower() if evidence is not None else None
            )
            self._success_counter = 0
            return True
        return bool(super()._compute_failure())

    def _build_observation_space(self) -> spaces.Dict:
        result = super()._build_observation_space()
        state_space = result.spaces["state"]
        if not isinstance(state_space, spaces.Dict):
            raise RuntimeError("BC017 base state observation is not a Dict")
        state_space.spaces.update(
            {
                "bc017_drawer_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(len(_DRAWER_STATE_FIELDS),),
                    dtype=np.float64,
                ),
                "bc017_contact_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(len(_CONTACT_STATE_FIELDS),),
                    dtype=np.float64,
                ),
                "bc017_stage_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(len(_STAGE_STATE_FIELDS),),
                    dtype=np.float64,
                ),
                "bc017_terminal_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(len(_TERMINAL_STATE_FIELDS),),
                    dtype=np.float64,
                ),
                "bc017_hidden_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=HIDDEN_OBSERVATION_SHAPE,
                    dtype=np.float64,
                ),
            }
        )
        # Named task-state groups are duplicated at the top level strictly for
        # recording.  The policy still consumes the task-declared flattened
        # state, while HandsOnObsAdapter preserves these names for aligned
        # current/next replay evidence.
        result.spaces.update(
            {key: state_space.spaces[key] for key in _OBSERVATION_KEYS}
        )
        return result

    @staticmethod
    def _public_slice(public: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
        result = np.asarray(
            [public[_PUBLIC_INDEX[name]] for name in fields],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("BC017 public observation contains non-finite values")
        return result

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        state = observation.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("BC017 observation is missing state mapping")
        public = self._bc017_state.public_observation()
        hidden = self._bc017_state.hidden_observation()
        state.update(
            {
                "bc017_drawer_state": self._public_slice(
                    public, _DRAWER_STATE_FIELDS
                ),
                "bc017_contact_state": self._public_slice(
                    public, _CONTACT_STATE_FIELDS
                ),
                "bc017_stage_state": self._public_slice(
                    public, _STAGE_STATE_FIELDS
                ),
                "bc017_terminal_state": self._public_slice(
                    public, _TERMINAL_STATE_FIELDS
                ),
                "bc017_hidden_state": hidden,
            }
        )
        observation.update(
            {key: np.asarray(state[key]).copy() for key in _OBSERVATION_KEYS}
        )
        for key in _OBSERVATION_KEYS:
            value = state[key]
            if value.dtype != np.float64 or not np.all(np.isfinite(value)):
                raise RuntimeError(f"BC017 observation {key} is not finite float64")
        return observation

    def get_recording_schema(self) -> dict[str, Any] | None:
        """Extend PhysX truth with read-only BC017 task-state replay fields."""

        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        raw_fields = result.get("fields")
        if not isinstance(raw_fields, dict):
            raise RuntimeError("BC017 recording schema has no PhysX field mapping")
        fields = {str(name): dict(spec) for name, spec in raw_fields.items()}
        group_fields = dict(PUBLIC_OBSERVATION_GROUPS)
        group_fields["bc017_hidden_state"] = HIDDEN_OBSERVATION_FIELDS
        for key in _OBSERVATION_KEYS:
            observation_space = self.observation_space.spaces.get(key)
            if not isinstance(observation_space, spaces.Box):
                raise RuntimeError(
                    f"BC017 recording observation space is missing {key}"
                )
            fields[key] = {
                "shape": list(observation_space.shape),
                "dtype": observation_space.dtype.name,
                "source": "bc017_action_aligned_task_state",
                "layout": list(group_fields[key]),
                "visibility": (
                    "hidden_task_logic"
                    if key == "bc017_hidden_state"
                    else "public_task_state"
                ),
                "replay_domain": "task_state",
                "physical_replay": True,
                "replay_restore_supported": False,
                "replay_atol": _TASK_STATE_REPLAY_ATOL[key],
            }
        result["fields"] = fields
        result["task_state"] = {
            "task_name": TASK_NAME,
            "field_names": list(_OBSERVATION_KEYS),
            "hidden_field_names": ["bc017_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return result

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        self._b320_condition_flags = self._condition_flags()
        return [
            self._condition_row(
                f"b320_bc017_{index + 1}",
                label,
                self._b320_condition_flags[index],
            )
            for index, label in enumerate(self._b320_contract.success_labels)
        ]

    def stable_recording_origin_task_readiness(
        self,
    ) -> dict[str, Any] | None:
        frame = self._read_bc017_physics_truth(
            physics_step=0,
            physics_time_s=0.0,
            allow_fixture_candidate=True,
        )
        calibration = self._bc017_state.calibration
        closed = np.asarray(
            [
                calibration.left_travel.closed_q_m,
                calibration.right_travel.closed_q_m,
            ],
            dtype=np.float64,
        )
        raw_q = np.asarray(frame.q_m, dtype=np.float64)
        raw_qd = np.asarray(frame.qd_m_s, dtype=np.float64)
        q = np.where(np.isfinite(raw_q), raw_q, 0.0)
        qd = np.where(np.isfinite(raw_qd), raw_qd, 0.0)
        exact_views, topology = self._bc017_exact_dof_topology()
        contact_views = set(self._bc017_contact_views) == set(CONTACT_KEYS)
        grasp_contact_views = set(self._bc017_grasp_contact_views) == {
            "left",
            "right",
        }
        grasp_joint_topology = bool(
            self._stage is not None
            and all(
                len(self._bc017_grasp_joint_paths.get(side, ())) == 5
                and all(
                    self._stage.GetPrimAtPath(path).IsValid()
                    for path in self._bc017_grasp_joint_paths.get(side, ())
                )
                for side in ("left", "right")
            )
        )
        dof_truth = bool(frame.left_dof_readable and frame.right_dof_readable)
        dof_finite = bool(np.all(np.isfinite(raw_q)) and np.all(np.isfinite(raw_qd)))
        raw_contact = np.asarray(
            [frame.contact_force_n[key] for key in CONTACT_KEYS],
            dtype=np.float64,
        )
        contact_truth = bool(
            all(frame.contact_readable.values())
            and np.all(np.isfinite(raw_contact))
            and np.all(raw_contact >= 0.0)
        )
        closed_ok = bool(
            dof_truth
            and dof_finite
            and np.all(np.abs(q - closed) <= _ORIGIN_Q_TOLERANCE_M)
        )
        stationary = bool(
            dof_truth
            and dof_finite
            and np.all(np.abs(qd) <= _ORIGIN_QD_TOLERANCE_M_S)
        )
        fixture_translation_drift = float(frame.fixture_translation_drift_m)
        fixture_rotation_drift = float(frame.fixture_rotation_drift_rad)
        fixture_stable = bool(
            frame.fixture_truth_readable
            and math.isfinite(fixture_translation_drift)
            and math.isfinite(fixture_rotation_drift)
            and fixture_translation_drift
            <= calibration.fixture_translation_limit_m
            and fixture_rotation_drift <= calibration.fixture_rotation_limit_rad
        )
        ready = bool(
            exact_views
            and contact_views
            and grasp_contact_views
            and grasp_joint_topology
            and dof_truth
            and dof_finite
            and contact_truth
            and fixture_stable
            and closed_ok
            and stationary
        )
        return {
            "kind": "bc017_two_dof_four_contact_fixed_fixture_truth",
            "ready": ready,
            "dof_names": list(_DOF_NAMES),
            "exact_dof_views": exact_views,
            "exact_dof_topology": topology,
            "dof_values_finite": dof_finite,
            "q_m": q.tolist(),
            "qd_m_s": qd.tolist(),
            "closed_q_m": closed.tolist(),
            "closed_position_ok": closed_ok,
            "stationary": stationary,
            "contact_keys": list(CONTACT_KEYS),
            "contact_view_present": {
                key: key in self._bc017_contact_views for key in CONTACT_KEYS
            },
            "strict_fingertip_handle_view_present": {
                side: side in self._bc017_grasp_contact_views
                for side in ("left", "right")
            },
            "fingertip_handle_d6_joint_topology_ready": (
                grasp_joint_topology
            ),
            "contact_readable": dict(self._bc017_contact_readable),
            "fixture_truth_readable": bool(frame.fixture_truth_readable),
            "fixture_translation_drift_m": (
                fixture_translation_drift
                if math.isfinite(fixture_translation_drift)
                else 0.0
            ),
            "fixture_rotation_drift_rad": (
                fixture_rotation_drift
                if math.isfinite(fixture_rotation_drift)
                else 0.0
            ),
            "fixture_stable": fixture_stable,
            "collection_ready": ready,
        }

    def accept_stable_recording_origin(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any] | None:
        from .recording_origin import RecordingOriginStabilityError

        final_readiness = report.get("final_task_readiness")
        if (
            report.get("task_readiness_required") is not True
            or not isinstance(final_readiness, dict)
            or final_readiness.get("ready") is not True
        ):
            raise RecordingOriginStabilityError(
                "B320-BC017 reached action zero without continuous exact "
                "two-DOF/four-contact/fixed-fixture truth.",
                diagnostics={
                    "failure_phase": "bc017_origin_acceptance",
                    "task_readiness": final_readiness,
                },
            )
        if int(self.env_step) != 0:
            raise RecordingOriginStabilityError(
                "B320-BC017 origin acceptance ran after action zero.",
                diagnostics={
                    "failure_phase": "bc017_origin_acceptance",
                    "env_step": int(self.env_step),
                },
            )
        readiness = self.stable_recording_origin_task_readiness()
        if readiness is None or not readiness["ready"]:
            raise RecordingOriginStabilityError(
                "B320-BC017 physical truth changed after the exact-zero boundary.",
                diagnostics={
                    "failure_phase": "bc017_origin_acceptance",
                    "readiness": readiness,
                },
            )
        fixture_pose, fixture_readable = self._bc017_fixture_pose()
        fixture_pose, fixture_valid = _finite_pose(fixture_pose)
        if not fixture_readable or not fixture_valid:
            raise RecordingOriginStabilityError(
                "B320-BC017 fixed-fixture pose vanished at origin acceptance.",
                diagnostics={"failure_phase": "bc017_origin_fixture"},
            )
        accepted_q = self._bc017_q_m.copy()
        self._bc017_recording_origin_fixture_pose = fixture_pose.copy()
        self._bc017_recording_origin_q_m = accepted_q.copy()
        payload = {
            "q_m": accepted_q.tolist(),
            "fixture_pose_wxyz": fixture_pose.tolist(),
        }
        self._bc017_recording_origin_sha256 = None
        self._bc017_state.reset()
        self._bc017_physics_step = 0
        self._bc017_expected_substep_index = 0
        prime = self._read_bc017_physics_truth(
            physics_step=0,
            physics_time_s=0.0,
            allow_fixture_candidate=False,
        )
        snapshot = self._bc017_state.prime(prime)
        if snapshot.terminated or not snapshot.truth_valid or not snapshot.semantic_valid:
            raise RecordingOriginStabilityError(
                "B320-BC017 action-zero prime did not preserve physical truth.",
                diagnostics={
                    "failure_phase": "bc017_origin_prime",
                    "failure_reason_code": snapshot.failure_reason_code,
                },
            )
        self._bc017_recording_origin_accepted = True
        self._b320_failure_reason = None
        return {
            "accepted": True,
            "kind": "bc017_stable_physical_boundary_prime_v1",
            "physics_state_writes": False,
            "origin": payload,
            "physics_step": 0,
            "physics_time_s": 0.0,
            "dof_truth_available": True,
            "contact_truth_available": True,
            "fixture_truth_available": True,
            "collection_ready": True,
        }

    def _task_info(self) -> dict[str, Any]:
        info = dict(super()._task_info())
        snapshot = self._bc017_state.snapshot()
        evidence = snapshot.terminal_evidence
        info.update(
            {
                "bc017_task_requirement": TASK_REQUIREMENT,
                "bc017_asset_title": SELECTED_ASSET_TITLE,
                "bc017_asset_creator": SELECTED_ASSET_CREATOR,
                "bc017_asset_url": SELECTED_ASSET_URL,
                "bc017_asset_license": SELECTED_ASSET_LICENSE,
                "bc017_calibration": self._bc017_state.calibration.as_dict(),
                "bc017_contact_bindings": {
                    key: dict(value)
                    for key, value in self._bc017_contact_bindings.items()
                },
                "bc017_grasp_contact_bindings": {
                    key: dict(value)
                    for key, value in self._bc017_grasp_contact_bindings.items()
                },
                "bc017_fingertip_handle_attachment": {
                    side: dict(metrics)
                    for side, metrics in self._bc017_grasp_metrics.items()
                },
                "bc017_fingertip_handle_released": dict(
                    self._bc017_grasp_released
                ),
                "bc017_grasp_release_progress": BC017_GRASP_RELEASE_PROGRESS,
                "bc017_contact_force_n": dict(self._bc017_contact_force_n),
                "bc017_contact_readable": dict(self._bc017_contact_readable),
                "bc017_dof_names": list(_DOF_NAMES),
                "bc017_q_m": self._bc017_q_m.tolist(),
                "bc017_qd_m_s": self._bc017_qd_m_s.tolist(),
                "bc017_dof_readable": self._bc017_dof_readable.tolist(),
                "bc017_physics_step": int(snapshot.physics_step),
                "bc017_physics_time_s": float(snapshot.physics_time_s),
                "bc017_contract_schema_name": BC017_CONTRACT_SCHEMA_NAME,
                "bc017_contract_schema_version": (
                    BC017_CONTRACT_SCHEMA_VERSION
                ),
                "bc017_contract_schema_digest_sha256": (
                    BC017_CONTRACT_SCHEMA_DIGEST_SHA256
                ),
                "bc017_public_observation_fields": list(
                    PUBLIC_OBSERVATION_FIELDS
                ),
                "bc017_public_observation_shape": list(
                    PUBLIC_OBSERVATION_SHAPE
                ),
                "bc017_observation_layout": (
                    bc017_observation_layout_metadata()
                ),
                "bc017_proprio_flatten_order": list(
                    BC017_PROPRIO_FLATTEN_ORDER
                ),
                "bc017_hidden_state_fields": list(HIDDEN_OBSERVATION_FIELDS),
                "bc017_hidden_state_shape": list(HIDDEN_OBSERVATION_SHAPE),
                "bc017_hidden_state": self._bc017_state.hidden_observation().tolist(),
                "bc017_terminal_failure_reason_code": (
                    evidence.reason_code if evidence is not None else 0
                ),
                "bc017_terminal_failure_reason": (
                    evidence.failure_reason.name.lower()
                    if evidence is not None and evidence.failure
                    else None
                ),
                "bc017_terminal_metric": (
                    evidence.metric if evidence is not None else None
                ),
                "bc017_terminal_metric_code": (
                    int(evidence.metric_code) if evidence is not None else 0
                ),
                "bc017_recording_origin_accepted": bool(
                    self._bc017_recording_origin_accepted
                ),
                "b320_physical_replay_certified": False,
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        """Return the ordinary physical reset contract without release gates."""

        contract = dict(super().get_physical_replay_reset_contract())
        terminal_evidence = self._bc017_state.terminal_evidence
        origin_payload = None
        if self._bc017_recording_origin_fixture_pose is not None:
            origin_payload = {
                "q_m": self._bc017_recording_origin_q_m.tolist(),
                "fixture_pose_wxyz": (
                    self._bc017_recording_origin_fixture_pose.tolist()
                ),
            }
        contract.update(
            {
                "bc017_task_requirement": TASK_REQUIREMENT,
                "bc017_dof_names": list(_DOF_NAMES),
                "bc017_contact_keys": list(CONTACT_KEYS),
                "bc017_physics_hz": PHYSICS_HZ,
                "bc017_sync_window_seconds": SYNC_WINDOW_SECONDS,
                "bc017_sync_window_substeps": SYNC_WINDOW_STEPS,
                "bc017_stability_substeps": STABILITY_STEPS,
                "bc017_fingertip_handle_attachment": {
                    "trigger": (
                        "assigned_distal_link4_contact_on_protruding_pull_tab"
                    ),
                    "constraint": (
                        "preauthored_external_d6_joint_all_six_axes_locked_"
                        "at_live_contact_pose"
                    ),
                    "palm_or_proximal_contact_allowed": False,
                    "release_progress_strictly_above": (
                        BC017_GRASP_RELEASE_PROGRESS
                    ),
                    "reattach_after_release_in_same_episode": False,
                },
                "bc017_calibration": self._bc017_state.calibration.as_dict(),
                "bc017_state_machine": (
                    "exact_crossing_contact_attribution_120hz_two_sample_v1"
                ),
                "bc017_contract_schema_name": BC017_CONTRACT_SCHEMA_NAME,
                "bc017_contract_schema_version": BC017_CONTRACT_SCHEMA_VERSION,
                "bc017_public_observation_fields": list(
                    PUBLIC_OBSERVATION_FIELDS
                ),
                "bc017_public_observation_shape": list(
                    PUBLIC_OBSERVATION_SHAPE
                ),
                "bc017_observation_layout": (
                    bc017_observation_layout_metadata()
                ),
                "bc017_proprio_flatten_order": list(
                    BC017_PROPRIO_FLATTEN_ORDER
                ),
                "bc017_hidden_state_fields": list(HIDDEN_OBSERVATION_FIELDS),
                "bc017_hidden_state_shape": list(HIDDEN_OBSERVATION_SHAPE),
                "bc017_hidden_state": (
                    self._bc017_state.hidden_observation().tolist()
                ),
                "bc017_terminal_failure_reason_code": (
                    terminal_evidence.reason_code
                    if terminal_evidence is not None
                    else 0
                ),
                "bc017_terminal_failure_reason": (
                    terminal_evidence.failure_reason.name.lower()
                    if terminal_evidence is not None
                    and terminal_evidence.failure
                    else None
                ),
                "bc017_recording_origin_accepted": bool(
                    self._bc017_recording_origin_accepted
                ),
                "bc017_recording_origin": origin_payload,
                "b320_physical_replay_certified": False,
            }
        )
        return contract

    def close(self) -> None:
        """Close the ordinary IsaacLab environment."""

        self._free_all_bc017_grasp_joints(record=False)
        self._bc017_grasp_attachment = {"left": None, "right": None}
        super().close()


__all__ = [
    "HandsOnB320BC017IsaacLabEnv",
    "BC017_GRASP_RELEASE_PROGRESS",
    "_OBSERVATION_KEYS",
    "bc017_fixed_joint_frame_pair",
    "bc017_grasp_release_reached",
]
