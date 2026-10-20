"""Dedicated IsaacLab runtime for B320-BC017."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from gymnasium import spaces
import numpy as np

from .b320_bc017_contract import (
    BC017Calibration,
    BC017_CONTRACT_SCHEMA_DIGEST_SHA256,
    BC017_CONTRACT_SCHEMA_NAME,
    BC017_CONTRACT_SCHEMA_VERSION,
    BC017Frame,
    BC017_OBSERVATION_DTYPE,
    BC017_PROPRIO_FLATTEN_ORDER,
    BC017State,
    CONTACT_KEYS,
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
from .b320_bc017_scene import (
    BC017_ASSET_ORIENTATION_WXYZ,
    BC017_ASSET_TRANSLATION_M,
    BC017_DOF_NAMES,
    BC017_DRAWER_JOINT_NAMES,
    BC017_DRAWER_ROLES_BY_SIDE,
    BC017_DRAWER_SLOT_NAMES,
    BC017_DRAWER_TRAVEL_M,
    SELECTED_ASSET_CREATOR,
    SELECTED_ASSET_LICENSE,
    SELECTED_ASSET_TITLE,
    SELECTED_ASSET_URL,
    _spawn_bc017_study_room,
    bc017_drawer_role,
    bc017_handle_grip_bounds_world,
    bc017_point_on_handle_surface_world,
    set_bc017_target_edge_highlight_visibility,
    spawn_bc017_task_assets,
)
from .b320_bimanual_env import (
    HandsOnB320BimanualIsaacLabEnv,
    _array_like,
    _indices_like,
    _tensor_to_numpy,
)

_DOF_NAMES = tuple(BC017_DOF_NAMES)
_DRAWER_ROLES = tuple(
    role
    for side in ("left", "right")
    for role in BC017_DRAWER_ROLES_BY_SIDE[side]
)
_PHYSICAL_CONTACT_KEYS = tuple(
    (drawer_role, hand_side)
    for drawer_role in _DRAWER_ROLES
    for hand_side in ("left", "right")
)
_TARGET_SELECTION_ALGORITHM = (
    "numpy_randomstate_left_then_right_independent_randint_0_3_v1"
)
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
# representation (for example authored 0.248 becomes 0.24800000339746475).
# Keep the live/composed-stage comparison identical to the authoritative
# pre-mutation USD audit instead of applying a stricter binary64 tolerance.
_USD_LIMIT_ABS_TOLERANCE_M = 1.0e-7

# Success contact is accepted only when the live PhysX manifold lies on the
# source-derived U-pull.  The old single-fingertip six-axis D6 weld is not
# authored or updated by the runtime; the enlarged handle must be pulled using
# ordinary collision, friction, and the hand's commanded articulation.
_BC017_HANDLE_CONTACT_ZONE_MARGIN_M = 0.0025
_BC017_HANDLE_SURFACE_TOLERANCE_M = 0.0015

# The 1.55x desk exceeds the square camera_pose_v6 horizontal envelope.  Keep
# the calibrated physical pose and lens, but give BC017 a 2:1 main stream so
# both handles and hands remain visible.  Wrist-camera calibration is intact.
_BC017_MAIN_RENDER_RESOLUTION_HW = (640, 1280)
_BC017_HEAD_RECORDING_CAMERA_PROFILE_VERSION = "bc017-study-camera-v2"

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
if len(BC017_DRAWER_SLOT_NAMES) != 3 or len(_DRAWER_ROLES) != 6:
    raise RuntimeError("BC017 runtime requires exactly three drawers per side")
if len(_DOF_NAMES) != 6 or set(BC017_DRAWER_JOINT_NAMES) != set(_DRAWER_ROLES):
    raise RuntimeError("BC017 runtime requires one canonical DOF per drawer role")
if tuple(BC017_DRAWER_JOINT_NAMES[role] for role in _DRAWER_ROLES) != _DOF_NAMES:
    raise RuntimeError("BC017 drawer-role and canonical-DOF order must agree")


def _task_calibration() -> BC017Calibration:
    """Return the source-scaled authored drawer travel used by the task."""

    travel = DrawerTravelCalibration(0.0, BC017_DRAWER_TRAVEL_M, formal=False)
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
            self._bc017_target_slots = (0, 0)
            self._bc017_physical_contact_views: dict[tuple[str, str], Any] = {}
            self._bc017_physical_contact_bindings: dict[
                tuple[str, str], dict[str, Any]
            ] = {}
            self._bc017_contact_views: dict[str, Any] = {}
            self._bc017_contact_bindings: dict[str, dict[str, Any]] = {}
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
            self._bc017_all_q_m = np.zeros(len(_DOF_NAMES), dtype=np.float64)
            self._bc017_all_qd_m_s = np.zeros(len(_DOF_NAMES), dtype=np.float64)
            self._bc017_all_dof_readable = np.zeros(len(_DOF_NAMES), dtype=bool)
            self._bc017_fixture_pose_wxyz = np.zeros(7, dtype=np.float64)
            self._bc017_fixture_truth_readable = False
            self._bc017_fixture_origin_candidate: np.ndarray | None = None
            self._bc017_recording_origin_fixture_pose: np.ndarray | None = None
            self._bc017_recording_origin_q_m = np.zeros(
                len(_DOF_NAMES), dtype=np.float64
            )
            self._bc017_recording_origin_accepted = False
            self._bc017_recording_origin_sha256: str | None = None
            self._bc017_runtime_integrity_failure: str | None = None
            self._bc017_physics_step = 0
            self._bc017_expected_substep_index = 0
            self._bc017_reset_write_count = 0
            self._bc017_study_room_paths: tuple[str, ...] = ()
            self._bc017_highlight_visibility: dict[str, bool] = {}
            kwargs.setdefault(
                "render_resolution",
                _BC017_MAIN_RENDER_RESOLUTION_HW,
            )
            super().__init__(task_name=task_name, **kwargs)
            self.recording_camera_profile["head"]["output_resolution"] = [
                int(self._render_width),
                int(self._render_height),
            ]
            self.recording_camera_profile["task_override"] = {
                "task_name": TASK_NAME,
                "version": _BC017_HEAD_RECORDING_CAMERA_PROFILE_VERSION,
                "reason": "2:1 framing for the source-scaled 1.55x dual-drawer desk",
            }
            if self._launch_isaac:
                # Compose render-only dressing only after the physical scene,
                # PhysX context, and tensor views are fully initialized.  This
                # mirrors BC014's visual-environment isolation contract.
                import isaaclab.sim as sim_utils

                self._bc017_study_room_paths = _spawn_bc017_study_room(
                    self._stage,
                    sim_utils,
                )
                # The canonical floor remains the collision authority.  Hide
                # only its gray renderable so the offset oak floor/rug layer
                # is visible; this mirrors BC014's visual-scene composition.
                from pxr import UsdGeom

                floor = self._find_prim("floor")
                if floor is None or not floor.IsValid():
                    raise RuntimeError(
                        "BC017 study composition cannot find the canonical floor"
                    )
                UsdGeom.Imageable(floor).MakeInvisible()
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
        """The manifest fixture is world-fixed; only its six DOFs may reset."""

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
        if self._physx_ground_truth_enabled and len(task_body_paths) != 7:
            raise RuntimeError(
                "BC017 full contact report lacks exact seven-body task coverage"
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
        self._bc017_state.reset(target_slots=self._bc017_target_slots)
        return

    def _configure_task_specific_pre_physx(self) -> bool:
        """Keep BC017 free of cross-articulation fingertip constraints."""

        return False

    def _init_task_specific_runtime(self) -> None:
        self._bc017_physical_contact_views = {}
        self._bc017_physical_contact_bindings = {}
        self._bc017_contact_views = {}
        self._bc017_contact_bindings = {}
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            self._add_task_state_warning("BC017 PhysX contact view is unavailable")
            return
        filters_by_hand = {
            side: tuple(self._hand_body_paths(side))
            for side in ("left", "right")
        }
        for drawer_role, hand_side in _PHYSICAL_CONTACT_KEYS:
            physical_key = (drawer_role, hand_side)
            sensor = self._b320_spawn_result.body_paths.get(drawer_role)
            filters = filters_by_hand[hand_side]
            binding = {
                "drawer_role": drawer_role,
                "hand_side": hand_side,
                "sensor_path": str(sensor or ""),
                "filter_paths": list(filters),
                "sensor_count": 0,
                "filter_count": 0,
                "bound": False,
            }
            self._bc017_physical_contact_bindings[physical_key] = binding
            if not sensor or not filters:
                self._add_task_state_warning(
                    "BC017 physical contact binding is incomplete: "
                    f"{drawer_role}/{hand_side}"
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
                    self._bc017_physical_contact_views[physical_key] = view
            except Exception as exc:
                self._add_task_state_warning(
                    "Could not initialize BC017 physical contact view "
                    f"{drawer_role}/{hand_side}: {exc}"
                )
        self._map_bc017_semantic_contact_views()
        missing = sorted(
            f"{role}/{hand}"
            for role, hand in (
                set(_PHYSICAL_CONTACT_KEYS)
                - set(self._bc017_physical_contact_views)
            )
        )
        if missing:
            self._add_task_state_warning(
                "BC017 twelve-view contact truth is incomplete; success "
                "fails closed: " + ",".join(missing)
            )

    def _bc017_selected_roles(self) -> tuple[str, str]:
        left_slot, right_slot = self._bc017_target_slots
        return (
            bc017_drawer_role("left", left_slot),
            bc017_drawer_role("right", right_slot),
        )

    def _bc017_selected_dof_names(self) -> tuple[str, str]:
        left_role, right_role = self._bc017_selected_roles()
        return (
            BC017_DRAWER_JOINT_NAMES[left_role],
            BC017_DRAWER_JOINT_NAMES[right_role],
        )

    def _map_bc017_semantic_contact_views(self) -> None:
        left_role, right_role = self._bc017_selected_roles()
        semantic_bindings = (
            ("LL", left_role, "left", self._bc017_target_slots[0]),
            ("LR", right_role, "left", self._bc017_target_slots[1]),
            ("RL", left_role, "right", self._bc017_target_slots[0]),
            ("RR", right_role, "right", self._bc017_target_slots[1]),
        )
        self._bc017_contact_views = {}
        self._bc017_contact_bindings = {}
        for semantic_key, drawer_role, hand_side, slot in semantic_bindings:
            physical_key = (drawer_role, hand_side)
            binding = dict(
                self._bc017_physical_contact_bindings.get(physical_key, {})
            )
            binding.update(
                {
                    "semantic_key": semantic_key,
                    "drawer_role": drawer_role,
                    "hand_side": hand_side,
                    "target_slot": int(slot),
                    "physical_key": f"{drawer_role}:{hand_side}",
                }
            )
            self._bc017_contact_bindings[semantic_key] = binding
            view = self._bc017_physical_contact_views.get(physical_key)
            if view is not None:
                self._bc017_contact_views[semantic_key] = view

    def _reset_bc017_runtime_state(
        self,
        *,
        target_slots: tuple[int, int],
    ) -> None:
        self._bc017_state.reset(target_slots=target_slots)
        self._bc017_target_slots = self._bc017_state.target_slots
        self._map_bc017_semantic_contact_views()
        self._bc017_contact_force_n = {key: 0.0 for key in CONTACT_KEYS}
        self._bc017_contact_readable = {key: False for key in CONTACT_KEYS}
        self._bc017_q_m = np.zeros(2, dtype=np.float64)
        self._bc017_qd_m_s = np.zeros(2, dtype=np.float64)
        self._bc017_dof_readable = np.zeros(2, dtype=bool)
        self._bc017_all_q_m = np.zeros(len(_DOF_NAMES), dtype=np.float64)
        self._bc017_all_qd_m_s = np.zeros(len(_DOF_NAMES), dtype=np.float64)
        self._bc017_all_dof_readable = np.zeros(len(_DOF_NAMES), dtype=bool)
        self._bc017_fixture_pose_wxyz = np.zeros(7, dtype=np.float64)
        self._bc017_fixture_truth_readable = False
        self._bc017_fixture_origin_candidate = None
        self._bc017_recording_origin_fixture_pose = None
        self._bc017_recording_origin_q_m = np.zeros(
            len(_DOF_NAMES), dtype=np.float64
        )
        self._bc017_recording_origin_accepted = False
        self._bc017_recording_origin_sha256 = None
        self._bc017_runtime_integrity_failure = None
        self._bc017_physics_step = 0
        self._bc017_expected_substep_index = 0

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        target_slots = (
            int(self._random.randint(0, len(BC017_DRAWER_SLOT_NAMES))),
            int(self._random.randint(0, len(BC017_DRAWER_SLOT_NAMES))),
        )
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._reset_bc017_runtime_state(target_slots=target_slots)
        if self._stage is not None:
            self._bc017_highlight_visibility = (
                set_bc017_target_edge_highlight_visibility(
                    self._stage,
                    left_slot=target_slots[0],
                    right_slot=target_slots[1],
                )
            )

    def _apply_task_pose_randomization(self) -> None:
        # Shared B320 restores captured articulation state while protecting the
        # fixed root.  BC017 then overwrites all six writable DOFs with
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
            missing = [
                name
                for name, entry in zip(_DOF_NAMES, entries, strict=True)
                if entry is None
            ]
            raise RuntimeError(
                "BC017 exact reset DOFs are unavailable: " + ",".join(missing)
            )
        resolved_entries = [entry for entry in entries if entry is not None]
        view = resolved_entries[0][0]
        if any(entry[0] is not view for entry in resolved_entries):
            raise RuntimeError("BC017 drawer DOFs are not in one fixed articulation")
        position_ref = view.get_dof_positions()
        velocity_ref = view.get_dof_velocities()
        positions = _tensor_to_numpy(position_ref)
        velocities = np.zeros_like(_tensor_to_numpy(velocity_ref))
        if (
            positions.ndim != 2
            or velocities.shape != positions.shape
            or positions.shape[1] != len(_DOF_NAMES)
        ):
            raise RuntimeError("BC017 articulation tensors have invalid shapes")
        indices_by_name = tuple(int(entry[1]) for entry in resolved_entries)
        expected_indices = tuple(range(len(_DOF_NAMES)))
        if indices_by_name != expected_indices:
            raise RuntimeError(
                "BC017 exact DOF indices are invalid, aliased, or out of order"
            )
        # Every drawer remains a free 0..travel prismatic body.  Reset closes
        # all six; target selection changes semantics/highlights only and never
        # locks, teleports, or removes inactive drawers from physical truth.
        positions[:, list(indices_by_name)] = 0.0
        if not math.isclose(
            self._bc017_state.calibration.left_travel.closed_q_m,
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            self._bc017_state.calibration.right_travel.closed_q_m,
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("BC017 canonical closed position must be exact zero")
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
        """Require one articulation with exactly six ordered canonical DOFs."""

        names_exact = set(self._task_dof_views) == set(_DOF_NAMES)
        entries = [self._task_dof_views.get(name) for name in _DOF_NAMES]
        same_view = bool(
            all(entry is not None for entry in entries)
            and all(entry[0] is entries[0][0] for entry in entries[1:])
        )
        indices_by_name = tuple(
            int(entry[1]) if entry is not None else -1 for entry in entries
        )
        indices_exact = bool(
            same_view and indices_by_name == tuple(range(len(_DOF_NAMES)))
        )
        tensor_dof_count = -1
        metatype_names: set[str] = set()
        metatype_indices: tuple[int, ...] = ()
        if same_view:
            assert entries[0] is not None
            view = entries[0][0]
            try:
                tensor_dof_count = int(
                    _tensor_to_numpy(view.get_dof_positions()).shape[-1]
                )
            except Exception:
                tensor_dof_count = -1
            try:
                metatype = view.get_metatype(0)
                raw_indices = getattr(metatype, "dof_indices", {})
                metatype_names = set(raw_indices)
                metatype_indices = tuple(
                    int(raw_indices[name]) for name in _DOF_NAMES
                )
            except Exception:
                metatype_names = set()
                metatype_indices = ()
        metatype_exact = metatype_names == set(_DOF_NAMES)
        metatype_indices_exact = (
            metatype_indices == tuple(range(len(_DOF_NAMES)))
        )
        ready = bool(
            names_exact
            and same_view
            and indices_exact
            and tensor_dof_count == len(_DOF_NAMES)
            and metatype_exact
            and metatype_indices_exact
        )
        return ready, {
            "mapping_names_exact": names_exact,
            "same_articulation_view": same_view,
            "indices_by_name": list(indices_by_name),
            "indices_0_through_5_exact": indices_exact,
            "tensor_dof_count": tensor_dof_count,
            "metatype_dof_names": sorted(metatype_names),
            "metatype_names_exact": metatype_exact,
            "metatype_indices_by_name": list(metatype_indices),
            "metatype_indices_0_through_5_exact": metatype_indices_exact,
        }

    def _bc017_physics_dof_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read all six exact manifest joint names; never fuzzy-match."""

        q = np.zeros(len(_DOF_NAMES), dtype=np.float64)
        qd = np.zeros(len(_DOF_NAMES), dtype=np.float64)
        readable = np.zeros(len(_DOF_NAMES), dtype=bool)
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

    def _bc017_selected_dof_indices(self) -> tuple[int, int]:
        left_name, right_name = self._bc017_selected_dof_names()
        return (_DOF_NAMES.index(left_name), _DOF_NAMES.index(right_name))

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

    def _read_bc017_handle_contact_force(
        self,
        key: str,
        drawer_q_m: float,
    ) -> tuple[float, bool]:
        """Sum only real contact manifolds on the moving visible U-pull."""

        view = self._bc017_contact_views.get(key)
        binding = self._bc017_contact_bindings.get(key, {})
        drawer_role = str(binding.get("drawer_role", ""))
        if view is None or drawer_role not in _DRAWER_ROLES:
            return 0.0, False
        if not math.isfinite(float(drawer_q_m)):
            # Preserve the contract's DOF_NON_FINITE first cause instead of
            # crashing while transforming the contact surface.
            return 0.0, False
        try:
            normal_forces, points, _normals, _separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )
            forces_np = np.abs(
                _tensor_to_numpy(normal_forces).astype(np.float64).reshape(-1)
            )
            points_np = _tensor_to_numpy(points).astype(np.float64).reshape(-1, 3)
            shape = (int(view.sensor_count), int(view.filter_count))
            counts_np = _tensor_to_numpy(counts).astype(np.int64).reshape(shape)
            starts_np = _tensor_to_numpy(starts).astype(np.int64).reshape(shape)
        except Exception:
            return 0.0, False
        if not np.all(np.isfinite(forces_np)) or not np.all(
            np.isfinite(points_np)
        ):
            return float("nan"), True
        q_m = float(np.clip(drawer_q_m, 0.0, BC017_DRAWER_TRAVEL_M))
        minimum, maximum = bc017_handle_grip_bounds_world(drawer_role, q_m)
        lower = np.asarray(minimum, dtype=np.float64)
        upper = np.asarray(maximum, dtype=np.float64)
        total_force_n = 0.0
        for count, start in zip(
            counts_np.ravel(), starts_np.ravel(), strict=True
        ):
            count_int = int(count)
            start_int = int(start)
            if count_int <= 0 or start_int < 0:
                continue
            stop = start_int + count_int
            if stop > len(points_np) or stop > len(forces_np):
                return 0.0, False
            for point, force in zip(
                points_np[start_int:stop],
                forces_np[start_int:stop],
                strict=True,
            ):
                if (
                    self._bc017_point_inside_bounds(
                        point,
                        lower,
                        upper,
                        margin_m=_BC017_HANDLE_CONTACT_ZONE_MARGIN_M,
                    )
                    and bc017_point_on_handle_surface_world(
                        drawer_role,
                        q_m,
                        point,
                        tolerance_m=_BC017_HANDLE_SURFACE_TOLERANCE_M,
                    )
                ):
                    total_force_n += float(force)
        return total_force_n, True

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

    def _read_bc017_physics_truth(
        self,
        *,
        physics_step: int,
        physics_time_s: float,
        allow_fixture_candidate: bool,
    ) -> BC017Frame:
        all_q, all_qd, all_dof_readable = self._bc017_physics_dof_state()
        self._bc017_all_q_m = np.where(np.isfinite(all_q), all_q, 0.0)
        self._bc017_all_qd_m_s = np.where(np.isfinite(all_qd), all_qd, 0.0)
        self._bc017_all_dof_readable = all_dof_readable.copy()
        selected_indices = self._bc017_selected_dof_indices()
        q = all_q[list(selected_indices)]
        qd = all_qd[list(selected_indices)]
        dof_readable = all_dof_readable[list(selected_indices)]
        self._bc017_q_m = np.where(np.isfinite(q), q, 0.0)
        self._bc017_qd_m_s = np.where(np.isfinite(qd), qd, 0.0)
        self._bc017_dof_readable = dof_readable.copy()
        raw_forces: dict[str, float] = {}
        for key in CONTACT_KEYS:
            drawer_index = 0 if key in {"LL", "RL"} else 1
            force, readable = self._read_bc017_handle_contact_force(
                key,
                float(q[drawer_index]),
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

    def _bc017_target_selection_payload(self) -> dict[str, Any]:
        if self._bc017_state.target_slots != self._bc017_target_slots:
            raise RuntimeError("BC017 environment/state target selection drifted")
        left_role, right_role = self._bc017_selected_roles()
        return {
            "algorithm": _TARGET_SELECTION_ALGORITHM,
            "rng": "environment_self_random",
            "seed_source": "episode_reset_seed",
            "draw_order": ["left", "right"],
            "sampling": "independent_uniform_with_replacement",
            "slot_names": list(BC017_DRAWER_SLOT_NAMES),
            "target_slots": list(self._bc017_target_slots),
            "target_roles": {"left": left_role, "right": right_role},
            "active_dof_names": list(self._bc017_selected_dof_names()),
            "all_dof_names": list(_DOF_NAMES),
            "roles_by_side": {
                side: list(BC017_DRAWER_ROLES_BY_SIDE[side])
                for side in ("left", "right")
            },
        }

    def _refresh_object_poses(self) -> None:
        super()._refresh_object_poses()
        left_role, right_role = self._bc017_selected_roles()
        self._poses["left_drawer"] = self._pose(left_role).copy()
        self._poses["right_drawer"] = self._pose(right_role).copy()

    def _articulation_vector(self) -> np.ndarray:
        values: list[float] = []
        for name in self._bc017_selected_dof_names():
            fraction, available = self._dof_fraction(name)
            values.append(float(fraction) if available else 0.0)
        return np.asarray([*values, 0.0, 0.0], dtype=np.float64)

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
        closed = np.zeros(len(_DOF_NAMES), dtype=np.float64)
        raw_q, raw_qd, all_dof_readable = self._bc017_physics_dof_state()
        q = np.where(np.isfinite(raw_q), raw_q, 0.0)
        qd = np.where(np.isfinite(raw_qd), raw_qd, 0.0)
        exact_views, topology = self._bc017_exact_dof_topology()
        semantic_contact_views = set(self._bc017_contact_views) == set(CONTACT_KEYS)
        physical_contact_views = (
            set(self._bc017_physical_contact_views)
            == set(_PHYSICAL_CONTACT_KEYS)
        )
        dof_truth = bool(np.all(all_dof_readable))
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
            and semantic_contact_views
            and physical_contact_views
            and dof_truth
            and dof_finite
            and contact_truth
            and fixture_stable
            and closed_ok
            and stationary
        )
        selection = self._bc017_target_selection_payload()
        return {
            "kind": "bc017_six_dof_twelve_view_selected_four_contact_truth_v2",
            "ready": ready,
            "dof_names": list(_DOF_NAMES),
            "active_dof_names": list(self._bc017_selected_dof_names()),
            "target_selection": selection,
            "exact_dof_views": exact_views,
            "exact_dof_topology": topology,
            "dof_values_finite": dof_finite,
            "q_m": q.tolist(),
            "qd_m_s": qd.tolist(),
            "closed_q_m": closed.tolist(),
            "closed_position_ok": closed_ok,
            "stationary": stationary,
            "contact_keys": list(CONTACT_KEYS),
            "semantic_contact_view_present": {
                key: key in self._bc017_contact_views for key in CONTACT_KEYS
            },
            "physical_contact_view_present": {
                f"{role}:{hand}": (role, hand)
                in self._bc017_physical_contact_views
                for role, hand in _PHYSICAL_CONTACT_KEYS
            },
            "contact_geometry": "source_u_pull_three_hull_surface",
            "grasp_constraint_kind": "none_native_physx_contact_only",
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
                "six-DOF/twelve-view/fixed-fixture truth.",
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
        accepted_q = self._bc017_all_q_m.copy()
        self._bc017_recording_origin_fixture_pose = fixture_pose.copy()
        self._bc017_recording_origin_q_m = accepted_q.copy()
        selection = self._bc017_target_selection_payload()
        payload = {
            "dof_names": list(_DOF_NAMES),
            "q_m": accepted_q.tolist(),
            "fixture_pose_wxyz": fixture_pose.tolist(),
            "target_selection": selection,
        }
        self._bc017_recording_origin_sha256 = None
        self._bc017_state.reset(target_slots=self._bc017_target_slots)
        self._bc017_physics_step = 0
        self._bc017_expected_substep_index = 0
        prime = self._read_bc017_physics_truth(
            physics_step=0,
            physics_time_s=0.0,
            allow_fixture_candidate=False,
        )
        snapshot = self._bc017_state.prime(prime)
        if (
            snapshot.terminated
            or not snapshot.truth_valid
            or not snapshot.semantic_valid
        ):
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
            "kind": "bc017_stable_physical_boundary_prime_v2",
            "physics_state_writes": False,
            "origin": payload,
            "target_selection": selection,
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
                "bc017_physical_contact_bindings": {
                    f"{role}:{hand}": dict(value)
                    for (role, hand), value in (
                        self._bc017_physical_contact_bindings.items()
                    )
                },
                "bc017_target_selection": (
                    self._bc017_target_selection_payload()
                ),
                "bc017_target_highlight_visibility": dict(
                    self._bc017_highlight_visibility
                ),
                "bc017_handle_contact_geometry": (
                    "source_u_pull_three_hull_surface"
                ),
                "bc017_grasp_constraint_kind": (
                    "none_native_physx_contact_only"
                ),
                "bc017_study_room_root": (
                    "/World/VisualEnvironment/BC017Study"
                ),
                "bc017_study_room_render_paths": list(
                    self._bc017_study_room_paths
                ),
                "bc017_contact_force_n": dict(self._bc017_contact_force_n),
                "bc017_contact_readable": dict(self._bc017_contact_readable),
                "bc017_dof_names": list(_DOF_NAMES),
                "bc017_active_dof_names": list(
                    self._bc017_selected_dof_names()
                ),
                "bc017_q_m": self._bc017_all_q_m.tolist(),
                "bc017_qd_m_s": self._bc017_all_qd_m_s.tolist(),
                "bc017_dof_readable": (
                    self._bc017_all_dof_readable.tolist()
                ),
                "bc017_active_q_m": self._bc017_q_m.tolist(),
                "bc017_active_qd_m_s": self._bc017_qd_m_s.tolist(),
                "bc017_active_dof_readable": (
                    self._bc017_dof_readable.tolist()
                ),
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
                "dof_names": list(_DOF_NAMES),
                "q_m": self._bc017_recording_origin_q_m.tolist(),
                "fixture_pose_wxyz": (
                    self._bc017_recording_origin_fixture_pose.tolist()
                ),
                "target_selection": self._bc017_target_selection_payload(),
            }
        contract.update(
            {
                "bc017_task_requirement": TASK_REQUIREMENT,
                "bc017_dof_names": list(_DOF_NAMES),
                "bc017_active_dof_names": list(
                    self._bc017_selected_dof_names()
                ),
                "bc017_contact_keys": list(CONTACT_KEYS),
                "bc017_physical_contact_keys": [
                    f"{role}:{hand}" for role, hand in _PHYSICAL_CONTACT_KEYS
                ],
                "bc017_target_selection": (
                    self._bc017_target_selection_payload()
                ),
                "bc017_physics_hz": PHYSICS_HZ,
                "bc017_sync_window_seconds": SYNC_WINDOW_SECONDS,
                "bc017_sync_window_substeps": SYNC_WINDOW_STEPS,
                "bc017_stability_substeps": STABILITY_STEPS,
                "bc017_handle_contact": {
                    "trigger": (
                        "live_assigned_hand_manifold_on_source_derived_"
                        "three_hull_u_pull_surface"
                    ),
                    "constraint": "none_native_physx_contact_only",
                    "whole_drawer_face_contact_allowed": False,
                    "wrong_hand_contact_allowed": False,
                    "contact_persists_only_while_manifold_exists": True,
                },
                "bc017_study_room": {
                    "root": "/World/VisualEnvironment/BC017Study",
                    "render_only": True,
                    "paths": list(self._bc017_study_room_paths),
                },
                "bc017_calibration": self._bc017_state.calibration.as_dict(),
                "bc017_state_machine": (
                    "selected_pair_exact_crossing_contact_120hz_two_sample_v2"
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

        super().close()


__all__ = [
    "HandsOnB320BC017IsaacLabEnv",
    "_OBSERVATION_KEYS",
]
