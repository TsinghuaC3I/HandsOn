"""Native Isaac Sim 5.1 collection environment for YCB03."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from ...tasks.t35_rotate_propeller.semantics import (
    MAX_FINAL_ANGULAR_SPEED,
    MAX_FINAL_FOURTH_WHEEL_GAP_M,
    MAX_FINAL_LINEAR_SPEED,
    MAX_FINAL_PROPELLER_SPEED,
    MAX_FINAL_TRANSLATION_M,
    MAX_FINAL_YAW_DEG,
    MAX_OPERATION_LIFT_M,
    MAX_OPERATION_TILT_DEG,
    MAX_OPERATION_TRANSLATION_M,
    MAX_SUCCESS_REVERSE_RAD,
    MIN_LEFT_HOLD_SECONDS,
    REQUIRED_FORWARD_TRAVEL_RAD,
    YCB03_HISTORY_SIZE,
    event_history_vector,
    event_state_from_history,
    YCB03EventState,
    YCB03PhysicsSample,
)
from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _invert_pose_matrix,
    _pose_vector_to_matrix,
)

TASK_ID = "YCB03"
TASK_NAME = "t35_rotate_propeller"
AIRPLANE_BODY = "ycb03_airplane"
PROPELLER_BODY = "ycb03_airplane_propeller"
PROPELLER_JOINT = "a_toy_airplane_Propeller_joint"
WHEEL_BODIES = (
    "ycb03_airplane_fl_wheel",
    "ycb03_airplane_fr_wheel",
    "ycb03_airplane_rl_wheel",
    "ycb03_airplane_rr_wheel",
)
AIRPLANE_LINK_BODIES = (AIRPLANE_BODY, PROPELLER_BODY, *WHEEL_BODIES)

_NOMINAL_ROOT_POSITION = np.asarray([-0.24, 0.0, 0.9775755], dtype=np.float64)
_ROOT_XY_RANDOMIZATION_M = np.asarray([0.080, 0.090], dtype=np.float64)
_ROOT_YAW_RANDOMIZATION_DEG = 25.0
_LEFT_FUSELAGE_CONTACT_FORCE_N = 0.15
_RIGHT_PROPELLER_CONTACT_FORCE_N = 0.03
_LEFT_PROPELLER_CONTACT_FORCE_N = 0.03
_ANY_AIRPLANE_CONTACT_FORCE_N = 0.02
_WHEEL_TABLE_CONTACT_FORCE_N = 1e-4
_TABLE_X_BOUNDS = (-0.55, 0.25)
_TABLE_Y_BOUNDS = (-0.80, 0.80)
_TABLE_TOP_Z = 0.89
_WHEEL_PIVOTS = (
    np.asarray([0.04835150, -0.08226950, -0.05589450]),
    np.asarray([-0.05244850, -0.08226950, -0.05589450]),
    np.asarray([0.01805150, 0.09423050, -0.06699450]),
    np.asarray([-0.02214850, 0.09423050, -0.06699450]),
)
_WHEEL_RADII = np.asarray([0.0314255, 0.0314255, 0.0205810, 0.0205810])


@lru_cache(maxsize=1)
def _airplane_reset_collision_vertices():
    """Original source hulls; separate boxes preserve the empty wing gaps."""
    directory = (
        Path(__file__).resolve().parents[1]
        / "envs/xmls/lightwheel_ycb/a_toy_airplane/collision"
    )
    hulls = []
    for path in sorted(directory.glob("*.obj")):
        vertices = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in path.read_text().splitlines()
                if line.startswith("v ")
            ]
        )
        if vertices.size:
            hulls.append(("Propeller" in path.name, vertices))
    if not hulls:
        raise RuntimeError("YCB03 original collision hulls are unavailable")
    return tuple(hulls)


def _airplane_reset_boxes(propeller_angle):
    from ...tasks.t35_rotate_propeller.asset_audit import _rotation_about_axis

    pivot = np.array([-0.00244850, -0.12516950, 0.03930550])
    rotation = _rotation_about_axis(
        np.array([-0.0012613, 0.983399, -0.181454]), propeller_angle
    )
    for propeller, vertices in _airplane_reset_collision_vertices():
        points = (vertices - pivot) @ rotation.T + pivot if propeller else vertices
        lower, upper = points.min(axis=0), points.max(axis=0)
        yield (upper - lower) / 2, (upper + lower) / 2


class HandsOnYCB03AirplaneIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Dual-arm physical task with no grasp, body, or propeller assistance."""

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"Expected {TASK_NAME!r}, got {task_name!r}")
        # Direct runtime users and TaskConfig share the same default cadence.
        # The base environment's 20 ms default is not a 30 Hz control period.
        # hz controls wall-clock throttling; hz=0 keeps the physical period.
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        self._ycb03_events = YCB03EventState()
        self._ycb03_sampled_root_xy = _NOMINAL_ROOT_POSITION[:2].copy()
        self._ycb03_sampled_yaw_deg = 0.0
        self._ycb03_sampled_propeller_angle = 0.0
        self._ycb03_initial_pose = np.asarray(
            [*_NOMINAL_ROOT_POSITION, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._ycb03_contact_views: dict[str, Any] = {}
        self._ycb03_any_contact_views: dict[str, list[Any]] = {
            "left": [],
            "right": [],
        }
        self._ycb03_wheel_contact_views: dict[str, Any] = {}
        self._ycb03_contact_forces: dict[str, float] = {
            "left_fuselage": 0.0,
            "right_propeller": 0.0,
            "left_propeller": 0.0,
            "left_any": 0.0,
            "right_any": 0.0,
        }
        self._ycb03_last_sample: YCB03PhysicsSample | None = None
        self._ycb03_contact_validity = np.zeros(9, dtype=np.float64)
        self._ycb03_last_transition_reason = ""
        super().__init__(task_name=task_name, **kwargs)

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        for key, shape in {
            "airplane_pose": (7,),
            "airplane_linear_velocity": (3,),
            "airplane_angular_velocity": (3,),
            "propeller_joint_state": (6,),
            "contact_state": (10,),
            "contact_validity": (9,),
            "flick_state": (10,),
            "wheel_support_state": (10,),
            "airplane_displacement_state": (5,),
            "release_stability_state": (8,),
            "event_flags": (8,),
            "event_latch_state": (10,),
            "reset_state": (9,),
            "ycb03_replay_history": (YCB03_HISTORY_SIZE + 1,),
        }.items():
            state_spaces[key] = spaces.Box(
                -np.inf,
                np.inf,
                shape=shape,
                dtype=np.float64,
            )
        return observation_space

    def _task_extra_pose_names(self) -> set[str]:
        return {
            PROPELLER_BODY,
            *WHEEL_BODIES,
        }

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        super()._spawn_task_specific_assets(sim_utils)
        if self._stage is None:
            return
        from pxr import PhysxSchema

        for name in (*AIRPLANE_LINK_BODIES, "table"):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"YCB03 contact-report body is unavailable: {name}"
                )
                continue
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    prim.GetPath(),
                )
            else:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)

    def _init_task_specific_runtime(self) -> None:
        super()._init_task_specific_runtime()
        self._init_ycb03_contact_views()

    def _hand_body_paths(self, side: str, *, fingertips_only: bool) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if fingertips_only:
                accepted = name.startswith(f"{side}_finger") and name.endswith("_link4")
            else:
                accepted = bool(
                    name == f"{side}_palm_link" or name.startswith(f"{side}_finger")
                )
            if accepted:
                paths.append(str(prim.GetPath()))
        return paths

    def _create_contact_view(
        self,
        sensor_name: str,
        filter_paths: list[str],
        *,
        max_contacts: int,
    ) -> Any | None:
        if not self._robots or not filter_paths:
            return None
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        sensor_prim = self._find_rigid_body_prim(sensor_name)
        if physics_sim_view is None or sensor_prim is None:
            return None
        view = physics_sim_view.create_rigid_contact_view(
            str(sensor_prim.GetPath()),
            filter_patterns=filter_paths,
            max_contact_data_count=max_contacts,
        )
        if (
            view is None
            or int(view.sensor_count) != 1
            or int(view.filter_count) != len(filter_paths)
        ):
            return None
        return view

    def _init_ycb03_contact_views(self) -> None:
        self._ycb03_contact_views = {}
        self._ycb03_any_contact_views = {"left": [], "right": []}
        self._ycb03_wheel_contact_views = {}
        if not self._robots:
            return
        left_hand = self._hand_body_paths("left", fingertips_only=False)
        right_hand = self._hand_body_paths("right", fingertips_only=False)
        left_tips = self._hand_body_paths("left", fingertips_only=True)
        right_tips = self._hand_body_paths("right", fingertips_only=True)
        table_prim = self._find_rigid_body_prim("table")
        table_paths = [] if table_prim is None else [str(table_prim.GetPath())]
        try:
            candidates = {
                "left_fuselage": self._create_contact_view(
                    AIRPLANE_BODY,
                    left_hand,
                    max_contacts=max(64, len(left_hand) * 8),
                ),
                "right_propeller": self._create_contact_view(
                    PROPELLER_BODY,
                    right_tips,
                    max_contacts=max(64, len(right_tips) * 12),
                ),
                "left_propeller": self._create_contact_view(
                    PROPELLER_BODY,
                    left_tips,
                    max_contacts=max(64, len(left_tips) * 12),
                ),
            }
            self._ycb03_contact_views = {
                name: view for name, view in candidates.items() if view is not None
            }
            for side, hand_paths in (
                ("left", left_hand),
                ("right", right_hand),
            ):
                for body_name in AIRPLANE_LINK_BODIES:
                    view = self._create_contact_view(
                        body_name,
                        hand_paths,
                        max_contacts=max(64, len(hand_paths) * 8),
                    )
                    if view is not None:
                        self._ycb03_any_contact_views[side].append(view)
            for wheel_name in WHEEL_BODIES:
                view = self._create_contact_view(
                    wheel_name,
                    table_paths,
                    max_contacts=32,
                )
                if view is not None:
                    self._ycb03_wheel_contact_views[wheel_name] = view
        except Exception as exc:
            self._ycb03_contact_views = {}
            self._ycb03_any_contact_views = {"left": [], "right": []}
            self._ycb03_wheel_contact_views = {}
            self._add_task_state_warning(
                f"Could not initialize YCB03 physical contact views: {exc}"
            )
            return

        required = {"left_fuselage", "right_propeller", "left_propeller"}
        if set(self._ycb03_contact_views) != required:
            self._add_task_state_warning(
                "YCB03 hand contact views are incomplete; success fails closed"
            )
        if any(
            len(self._ycb03_any_contact_views[side]) != len(AIRPLANE_LINK_BODIES)
            for side in ("left", "right")
        ):
            self._add_task_state_warning(
                "YCB03 release contact views are incomplete; success fails closed"
            )
        if len(self._ycb03_wheel_contact_views) != len(WHEEL_BODIES):
            self._add_task_state_warning(
                "YCB03 wheel/table contact views are incomplete; success fails closed"
            )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        if self.randomize:
            xy_offset = self._random.uniform(
                -_ROOT_XY_RANDOMIZATION_M,
                _ROOT_XY_RANDOMIZATION_M,
            )
            self._ycb03_sampled_root_xy = (
                _NOMINAL_ROOT_POSITION[:2] + xy_offset
            ).astype(np.float64)
            self._ycb03_sampled_yaw_deg = float(
                self._random.uniform(
                    -_ROOT_YAW_RANDOMIZATION_DEG,
                    _ROOT_YAW_RANDOMIZATION_DEG,
                )
            )
            self._ycb03_sampled_propeller_angle = float(
                self._random.uniform(-math.pi, math.pi)
            )
        else:
            self._ycb03_sampled_root_xy = _NOMINAL_ROOT_POSITION[:2].copy()
            self._ycb03_sampled_yaw_deg = 0.0
            self._ycb03_sampled_propeller_angle = 0.0

        yaw = math.radians(self._ycb03_sampled_yaw_deg)
        self._ycb03_initial_pose = np.asarray(
            [
                *self._ycb03_sampled_root_xy,
                _NOMINAL_ROOT_POSITION[2],
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            ],
            dtype=np.float64,
        )
        self._ycb03_events = YCB03EventState(
            previous_propeller_angle_rad=self._ycb03_sampled_propeller_angle
        )
        self._ycb03_last_sample = None
        self._ycb03_last_transition_reason = ""
        self._ycb03_contact_validity[:] = 0.0
        for key in self._ycb03_contact_forces:
            self._ycb03_contact_forces[key] = 0.0

    def _apply_task_pose_randomization(self) -> None:
        articulation = self._task_articulation_views.get(AIRPLANE_BODY)
        if articulation is None or self._torch is None:
            self._add_task_state_warning(
                "YCB03 airplane articulation is unavailable during reset"
            )
            return
        view, indices = articulation
        cached = next(
            (entry for entry in self._task_reset_articulation_states if entry[1] is view),
            None,
        )
        if cached is None:
            self._add_task_state_warning(
                "YCB03 airplane articulation reset snapshot is unavailable"
            )
            return
        (
            path,
            _view,
            _indices,
            root_pose,
            root_velocity,
            joint_positions,
            joint_velocities,
        ) = cached
        reset_root_pose = root_pose.clone()
        reset_root_velocity = self._torch.zeros_like(root_velocity)
        reset_joint_positions = joint_positions.clone()
        reset_joint_velocities = self._torch.zeros_like(joint_velocities)
        clearance_boxes = tuple(_airplane_reset_boxes(self._ycb03_sampled_propeller_angle))
        for attempt in range(128):
            yaw = math.radians(self._ycb03_sampled_yaw_deg)
            pose = np.asarray(
                [
                    *self._ycb03_sampled_root_xy,
                    _NOMINAL_ROOT_POSITION[2],
                    math.cos(yaw / 2),
                    0.0,
                    0.0,
                    math.sin(yaw / 2),
                ]
            )
            if all(
                self._reset_box_clears_hands(pose, extent, center)
                for extent, center in clearance_boxes
            ):
                break
            if not self.randomize or attempt == 127:
                raise RuntimeError(
                    "YCB03 cannot sample an airplane clear of the reset hands"
                )
            self._ycb03_sampled_root_xy = _NOMINAL_ROOT_POSITION[:2] + self._random.uniform(
                -_ROOT_XY_RANDOMIZATION_M, _ROOT_XY_RANDOMIZATION_M
            )
            self._ycb03_sampled_yaw_deg = float(
                self._random.uniform(
                    -_ROOT_YAW_RANDOMIZATION_DEG, _ROOT_YAW_RANDOMIZATION_DEG
                )
            )
        self._ycb03_initial_pose = pose.copy()
        reset_root_pose[:, :3] = self._torch.as_tensor(
            [
                *self._ycb03_sampled_root_xy,
                _NOMINAL_ROOT_POSITION[2],
            ],
            dtype=reset_root_pose.dtype,
            device=reset_root_pose.device,
        )
        reset_root_pose[:, 3:7] = self._torch.as_tensor(
            [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
            dtype=reset_root_pose.dtype,
            device=reset_root_pose.device,
        )
        propeller_entry = self._task_dof_views.get(PROPELLER_JOINT)
        if propeller_entry is None or propeller_entry[0] is not view:
            self._add_task_state_warning("YCB03 propeller DOF is unavailable during reset")
            return
        propeller_index = int(propeller_entry[1])
        reset_joint_positions[:, propeller_index] = self._ycb03_sampled_propeller_angle
        view.set_root_transforms(reset_root_pose, indices)
        view.set_root_velocities(reset_root_velocity, indices)
        view.set_dof_positions(reset_joint_positions, indices)
        view.set_dof_velocities(reset_joint_velocities, indices)
        self._task_dof_views[PROPELLER_JOINT] = (
            view,
            propeller_index,
            self._ycb03_sampled_propeller_angle,
        )
        self._record_authored_task_reset_write(
            path,
            transform_xyzw=reset_root_pose,
            velocity=reset_root_velocity,
            dof_position=reset_joint_positions,
            dof_velocity=reset_joint_velocities,
        )

    def _dof_velocity(self, name: str) -> float | None:
        entry = self._task_dof_views.get(name)
        if entry is None:
            return None
        view, index, _initial = entry
        try:
            value = view.get_dof_velocities()[0, int(index)]
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            return float(value)
        except Exception:
            return None

    def _contact_force(self, view: Any | None) -> float:
        # Unknown contact is not evidence of release. Keep the invalid value
        # until the sample has checked every required channel.
        if view is None:
            return math.nan
        try:
            forces = view.get_contact_force_matrix(float(self._sim_dt))
            if hasattr(forces, "detach"):
                forces = forces.detach().cpu().numpy()
            forces = np.asarray(forces, dtype=np.float64)
            if forces.size == 0 or forces.shape[-1] != 3 or not np.all(np.isfinite(forces)):
                return math.nan
            return float(np.linalg.norm(forces.reshape(-1, 3), axis=1).sum())
        except Exception:
            return math.nan

    def _summed_contact_force(self, views: list[Any]) -> float:
        if len(views) != len(AIRPLANE_LINK_BODIES):
            return math.nan
        return float(sum(self._contact_force(view) for view in views))

    @staticmethod
    def _relative_rpy_degrees(
        initial_pose: np.ndarray,
        current_pose: np.ndarray,
    ) -> tuple[float, float, float]:
        relative = _invert_pose_matrix(
            _pose_vector_to_matrix(initial_pose)
        ) @ _pose_vector_to_matrix(current_pose)
        rotation = relative[:3, :3]
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        return tuple(math.degrees(value) for value in (roll, pitch, yaw))

    def _wheel_metrics(
        self,
        root_pose: np.ndarray,
    ) -> tuple[tuple[bool, ...], tuple[float, ...]]:
        root_matrix = _pose_vector_to_matrix(root_pose)
        table_top = self._current_table_top_z()
        gaps: list[float] = []
        supports: list[bool] = []
        contact_views_complete = len(self._ycb03_wheel_contact_views) == len(
            WHEEL_BODIES
        )
        for name, pivot, radius in zip(
            WHEEL_BODIES,
            _WHEEL_PIVOTS,
            _WHEEL_RADII,
        ):
            pivot_world = root_matrix @ np.asarray([*pivot, 1.0])
            gap = max(0.0, float(pivot_world[2] - radius - table_top))
            gaps.append(gap)
            force = self._contact_force(self._ycb03_wheel_contact_views.get(name))
            supports.append(
                bool(contact_views_complete and force >= _WHEEL_TABLE_CONTACT_FORCE_N)
            )
        return tuple(supports), tuple(gaps)

    def _read_ycb03_sample(self) -> YCB03PhysicsSample:
        root_pose = self._physics_body_pose(AIRPLANE_BODY)
        root_pose_available = root_pose is not None
        root_velocity = self._physics_body_velocity(AIRPLANE_BODY)
        joint = self._physics_dof_position(PROPELLER_JOINT)
        joint_velocity = self._dof_velocity(PROPELLER_JOINT)
        if root_pose is None:
            root_pose = self._ycb03_initial_pose.copy()
        if root_velocity is None:
            root_velocity = np.full(6, np.nan, dtype=np.float64)
        propeller_angle = math.nan if joint is None else float(joint[0])
        propeller_speed = math.nan if joint_velocity is None else float(joint_velocity)
        roll_deg, pitch_deg, yaw_deg = self._relative_rpy_degrees(
            self._ycb03_initial_pose,
            root_pose,
        )
        translation = float(
            np.linalg.norm(
                np.asarray(root_pose[:2]) - np.asarray(self._ycb03_initial_pose[:2])
            )
        )
        lift = float(root_pose[2] - self._ycb03_initial_pose[2])
        wheel_support, wheel_gaps = self._wheel_metrics(root_pose)

        self._ycb03_contact_forces = {
            "left_fuselage": self._contact_force(
                self._ycb03_contact_views.get("left_fuselage")
            ),
            "right_propeller": self._contact_force(
                self._ycb03_contact_views.get("right_propeller")
            ),
            "left_propeller": self._contact_force(
                self._ycb03_contact_views.get("left_propeller")
            ),
            "left_any": self._summed_contact_force(
                self._ycb03_any_contact_views["left"]
            ),
            "right_any": self._summed_contact_force(
                self._ycb03_any_contact_views["right"]
            ),
        }
        self._ycb03_contact_validity = np.asarray(
            [
                *(math.isfinite(force) for force in self._ycb03_contact_forces.values()),
                *(math.isfinite(self._contact_force(self._ycb03_wheel_contact_views.get(name)))
                  for name in WHEEL_BODIES),
            ],
            dtype=np.float64,
        )
        finite_values = np.concatenate(
            [
                np.asarray(root_pose, dtype=np.float64),
                np.asarray(root_velocity, dtype=np.float64),
                np.asarray([propeller_angle, propeller_speed]),
            ]
        )
        off_table = bool(
            root_pose[0] < _TABLE_X_BOUNDS[0]
            or root_pose[0] > _TABLE_X_BOUNDS[1]
            or root_pose[1] < _TABLE_Y_BOUNDS[0]
            or root_pose[1] > _TABLE_Y_BOUNDS[1]
            or root_pose[2] < _TABLE_TOP_Z - 0.05
        )
        return YCB03PhysicsSample(
            propeller_angle_rad=propeller_angle,
            propeller_speed_rad_s=propeller_speed,
            left_fuselage_contact=(
                self._ycb03_contact_forces["left_fuselage"]
                >= _LEFT_FUSELAGE_CONTACT_FORCE_N
            ),
            right_propeller_contact=(
                self._ycb03_contact_forces["right_propeller"]
                >= _RIGHT_PROPELLER_CONTACT_FORCE_N
            ),
            left_propeller_contact=(
                self._ycb03_contact_forces["left_propeller"]
                >= _LEFT_PROPELLER_CONTACT_FORCE_N
            ),
            left_any_airplane_contact=(
                self._ycb03_contact_forces["left_any"] >= _ANY_AIRPLANE_CONTACT_FORCE_N
                or self._ycb03_contact_forces["left_fuselage"] >= _LEFT_FUSELAGE_CONTACT_FORCE_N
                or self._ycb03_contact_forces["left_propeller"] >= _LEFT_PROPELLER_CONTACT_FORCE_N
            ),
            right_any_airplane_contact=(
                self._ycb03_contact_forces["right_any"] >= _ANY_AIRPLANE_CONTACT_FORCE_N
                or self._ycb03_contact_forces["right_propeller"] >= _RIGHT_PROPELLER_CONTACT_FORCE_N
            ),
            translation_m=translation,
            lift_m=lift,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_delta_deg=yaw_deg,
            linear_speed_m_s=float(np.linalg.norm(root_velocity[:3])),
            angular_speed_rad_s=float(np.linalg.norm(root_velocity[3:])),
            wheel_support_count=int(sum(wheel_support)),
            maximum_wheel_gap_m=float(max(wheel_gaps, default=math.inf)),
            airplane_off_table_or_workspace=off_table,
            finite=bool(root_pose_available and np.all(np.isfinite(finite_values))),
            contacts_valid=bool(np.all(self._ycb03_contact_validity)),
        )

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        sample = self._read_ycb03_sample()
        self._ycb03_events.update(sample, self._sim_dt)
        self._ycb03_last_sample = sample

    def _success_steps_required(self) -> int:
        return 1

    def _compute_success_now(self) -> bool:
        return bool(self._ycb03_events.final_ready())

    def _compute_failure(self) -> bool:
        return bool(self._ycb03_events.first_terminal_reason)

    def _task_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "terminal_reason": self._ycb03_events.first_terminal_reason,
            "transition_reason": self._ycb03_last_transition_reason,
            "ycb03_reset_xy": self._ycb03_sampled_root_xy.copy(),
            "ycb03_reset_yaw_deg": float(self._ycb03_sampled_yaw_deg),
            "ycb03_reset_propeller_angle_rad": float(
                self._ycb03_sampled_propeller_angle
            ),
            "ycb03_success_counter": int(self._success_counter),
            "ycb03_success_steps_required": self._success_steps_required(),
            "ycb03_contact_validity": self._ycb03_contact_validity.copy(),
            "ycb03_left_fuselage_contact_force_n": float(
                self._ycb03_contact_forces["left_fuselage"]
            ),
            "ycb03_right_propeller_contact_force_n": float(
                self._ycb03_contact_forces["right_propeller"]
            ),
            "ycb03_left_propeller_contact_force_n": float(
                self._ycb03_contact_forces["left_propeller"]
            ),
            "ycb03_left_any_contact_force_n": float(
                self._ycb03_contact_forces["left_any"]
            ),
            "ycb03_right_any_contact_force_n": float(
                self._ycb03_contact_forces["right_any"]
            ),
        }
        info.update(self._ycb03_events.metrics())
        return info

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        state = observation["state"]
        root_pose = self._poses.get(
            AIRPLANE_BODY,
            self._ycb03_initial_pose,
        ).astype(np.float64, copy=True)
        root_velocity = self._physics_body_velocity(AIRPLANE_BODY)
        if root_velocity is None:
            root_velocity = np.zeros(6, dtype=np.float64)
        joint = self._physics_dof_position(PROPELLER_JOINT)
        propeller_angle = (
            self._ycb03_sampled_propeller_angle if joint is None else float(joint[0])
        )
        propeller_speed = self._dof_velocity(PROPELLER_JOINT) or 0.0
        sample = self._ycb03_last_sample
        if sample is None:
            sample = YCB03PhysicsSample(
                propeller_angle_rad=propeller_angle,
                propeller_speed_rad_s=propeller_speed,
                left_fuselage_contact=False,
                right_propeller_contact=False,
                left_propeller_contact=False,
                left_any_airplane_contact=False,
                right_any_airplane_contact=False,
                translation_m=0.0,
                lift_m=0.0,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_delta_deg=0.0,
                linear_speed_m_s=0.0,
                angular_speed_rad_s=0.0,
                wheel_support_count=0,
                maximum_wheel_gap_m=math.inf,
                finite=False,
                contacts_valid=False,
            )
        root_matrix = _pose_vector_to_matrix(root_pose)
        table_top = self._current_table_top_z()
        wheel_gaps = tuple(
            max(
                0.0,
                float(
                    (root_matrix @ np.asarray([*pivot, 1.0]))[2] - radius - table_top
                ),
            )
            for pivot, radius in zip(_WHEEL_PIVOTS, _WHEEL_RADII)
        )
        wheel_support = tuple(
            float(
                self._contact_force(self._ycb03_wheel_contact_views.get(name))
                >= _WHEEL_TABLE_CONTACT_FORCE_N
            )
            for name in WHEEL_BODIES
        )
        metrics = self._ycb03_events.metrics()
        state.update(
            {
                "airplane_pose": root_pose,
                "airplane_linear_velocity": np.asarray(
                    root_velocity[:3], dtype=np.float64
                ),
                "airplane_angular_velocity": np.asarray(
                    root_velocity[3:], dtype=np.float64
                ),
                "propeller_joint_state": np.asarray(
                    [
                        propeller_angle,
                        propeller_speed,
                        self._ycb03_events.unwrapped_signed_travel_rad,
                        self._ycb03_events.direction,
                        self._ycb03_events.forward_travel_rad,
                        self._ycb03_events.reverse_travel_rad,
                    ],
                    dtype=np.float64,
                ),
                "contact_state": np.asarray(
                    [
                        self._ycb03_contact_forces["left_fuselage"],
                        self._ycb03_contact_forces["right_propeller"],
                        self._ycb03_contact_forces["left_propeller"],
                        self._ycb03_contact_forces["left_any"],
                        self._ycb03_contact_forces["right_any"],
                        float(sample.left_fuselage_contact),
                        float(sample.right_propeller_contact),
                        float(sample.left_propeller_contact),
                        float(sample.left_any_airplane_contact),
                        float(sample.right_any_airplane_contact),
                    ],
                    dtype=np.float64,
                ),
                "contact_validity": self._ycb03_contact_validity.copy(),
                "flick_state": np.asarray(
                    [
                        self._ycb03_events.direction,
                        self._ycb03_events.forward_travel_rad,
                        self._ycb03_events.reverse_travel_rad,
                        self._ycb03_events.valid_flick_count,
                        float(self._ycb03_events.active_flick),
                        self._ycb03_events.left_hold_seconds,
                        self._ycb03_events.active_flick_coast_seconds,
                        float(self._ycb03_events.right_separation_armed),
                        self._ycb03_events.hard_reverse_travel_rad,
                        self._ycb03_events.unwrapped_signed_travel_rad,
                    ],
                    dtype=np.float64,
                ),
                "wheel_support_state": np.asarray(
                    [
                        *wheel_support,
                        *wheel_gaps,
                        sample.wheel_support_count,
                        sample.maximum_wheel_gap_m,
                    ],
                    dtype=np.float64,
                ),
                "airplane_displacement_state": np.asarray(
                    [
                        sample.translation_m,
                        sample.lift_m,
                        sample.roll_deg,
                        sample.pitch_deg,
                        sample.yaw_delta_deg,
                    ],
                    dtype=np.float64,
                ),
                "release_stability_state": np.asarray(
                    [
                        float(metrics["ycb03_hands_released"]),
                        float(
                            abs(sample.propeller_speed_rad_s)
                            < MAX_FINAL_PROPELLER_SPEED
                        ),
                        float(sample.linear_speed_m_s < MAX_FINAL_LINEAR_SPEED),
                        float(sample.angular_speed_rad_s < MAX_FINAL_ANGULAR_SPEED),
                        float(sample.translation_m <= MAX_FINAL_TRANSLATION_M),
                        float(abs(sample.yaw_delta_deg) <= MAX_FINAL_YAW_DEG),
                        float(
                            sample.wheel_support_count >= 3
                            and sample.maximum_wheel_gap_m
                            <= MAX_FINAL_FOURTH_WHEEL_GAP_M
                        ),
                        float(metrics["ycb03_final_ready"]),
                    ],
                    dtype=np.float64,
                ),
                "event_flags": np.asarray(
                    [
                        float(metrics["ycb03_final_ready"]),
                        float(metrics["ycb03_operation_valid"]),
                        float(bool(self._ycb03_events.first_terminal_reason)),
                        float(
                            self._ycb03_events.absolute_travel_rad
                            >= REQUIRED_FORWARD_TRAVEL_RAD - 1e-10
                        ),
                        float(metrics["ycb03_final_ready"]),
                        float(
                            self._ycb03_events.reverse_travel_rad
                            <= MAX_SUCCESS_REVERSE_RAD
                        ),
                        float(metrics["ycb03_hands_released"]),
                        float(sample.wheel_support_count >= 3),
                    ],
                    dtype=np.float64,
                ),
                "ycb03_replay_history": np.asarray(
                    [*event_history_vector(self._ycb03_events), float(self._success_counter)],
                    dtype=np.float64,
                ),
                "event_latch_state": np.asarray(
                    [
                        float(self._ycb03_events.active_flick_valid),
                        self._ycb03_events.active_flick_positive_travel_rad,
                        self._ycb03_events.active_flick_negative_travel_rad,
                        float(self._ycb03_events.right_contact_previous),
                        float(self._ycb03_events.left_propeller_active),
                        self._ycb03_events.left_propeller_travel_rad,
                        self._ycb03_events.lift_violation_seconds,
                        self._ycb03_events.tilt_violation_seconds,
                        float(metrics["ycb03_terminal_reason_code"]),
                        self._ycb03_events.active_flick_coast_seconds,
                    ],
                    dtype=np.float64,
                ),
                "reset_state": np.asarray(
                    [
                        *self._ycb03_initial_pose,
                        self._ycb03_sampled_propeller_angle,
                        float(self._seed),
                    ],
                    dtype=np.float64,
                ),
            }
        )
        return observation

    def _task_status_text(self) -> str:
        metrics = self._ycb03_events.metrics()
        feedback = {
            "waiting_for_joint_state": "等待桨叶角度读数",
            "rotate_propeller": "转动桨叶即可累计，指尖离开后继续计数",
            "complete": "已满足完成条件",
            "terminal_failure": f"失败: {self._ycb03_events.first_terminal_reason}",
        }[metrics["ycb03_count_blocker"]]
        return (
            f"累计旋转 {float(metrics['ycb03_absolute_turns']):.2f}/3.00 圈"
            f"  |  完整圈数 {int(metrics['ycb03_completed_turns'])}/3"
            f"\n累计绝对转角 {float(metrics['ycb03_absolute_angle_deg']):.1f}° / 1080°"
            f"\n{feedback}"
        )

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        return [self._condition_row(
            "ycb03_three_turns",
            "桨叶累计绝对旋转达到 1080°（360° 算一圈）",
            self._ycb03_events.final_ready(),
        )]

    def step_with_current_joint_targets(
        self,
        *,
        start_time: float | None = None,
    ):
        observation, reward, terminated, truncated, info = (
            super().step_with_current_joint_targets(start_time=start_time)
        )
        timed_out = bool(
            self.env_step * self._control_dt >= self._time_limit
            and not info.get("succeed", False)
            and not info.get("failed", False)
        )
        if timed_out:
            self._ycb03_last_transition_reason = "timeout"
            info["transition_reason"] = "timeout"
            info["truncated_reason"] = "time_limit"
            return observation, reward, False, True, info
        if info.get("failed", False):
            self._ycb03_last_transition_reason = (
                self._ycb03_events.first_terminal_reason
            )
            info["transition_reason"] = self._ycb03_last_transition_reason
        elif info.get("succeed", False):
            self._ycb03_last_transition_reason = "success"
            info["transition_reason"] = "success"
        return observation, reward, terminated, truncated, info

    def restore_recorded_initial_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        # Validate history before changing the scene. Older incomplete histories
        # cannot silently start a fresh semantic state in an independent replay.
        history = np.asarray(state_parts.get("ycb03_replay_history", []), dtype=np.float64)
        if history.shape != (YCB03_HISTORY_SIZE + 1,):
            raise ValueError("Missing or invalid YCB03 replay history")
        restored_events = event_state_from_history(history[:-1])
        hold = float(history[-1])
        if not np.isfinite(hold) or hold < 0 or hold != int(hold):
            raise ValueError("Invalid YCB03 recorded success hold")
        reset = np.asarray(state_parts.get("reset_state", []), dtype=np.float64)
        validity = np.asarray(state_parts.get("contact_validity", []), dtype=np.float64)
        contact = np.asarray(state_parts.get("contact_state", []), dtype=np.float64)
        if (reset.shape != (9,) or not np.all(np.isfinite(reset))
                or np.linalg.norm(reset[3:7]) <= 1e-8
                or validity.shape != (9,) or not np.all(np.isin(validity, [0., 1.]))
                or contact.shape != (10,) or not np.all(np.isfinite(contact))):
            raise ValueError("Invalid YCB03 recorded reset/contact state")
        observation = super().restore_recorded_initial_state(state_parts)
        self._ycb03_events = restored_events
        self._ycb03_last_sample = restored_events.last_sample
        self._success_counter = int(hold)
        self._ycb03_last_transition_reason = restored_events.first_terminal_reason
        self._ycb03_initial_pose = reset[:7].copy()
        self._ycb03_sampled_root_xy = reset[:2].copy()
        self._ycb03_sampled_yaw_deg = math.degrees(2.0 * math.atan2(reset[6], reset[3]))
        self._ycb03_sampled_propeller_angle = float(reset[7])
        self._ycb03_contact_validity = validity.copy()
        self._ycb03_contact_forces = dict(zip(self._ycb03_contact_forces, contact[:5]))
        joint_state = np.asarray(
            state_parts.get("propeller_joint_state", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        entry = self._task_dof_views.get(PROPELLER_JOINT)
        if joint_state.size >= 2 and entry is not None and self._torch is not None:
            view, index, _initial = entry
            positions = view.get_dof_positions().clone()
            velocities = view.get_dof_velocities().clone()
            positions[:, int(index)] = float(joint_state[0])
            velocities[:, int(index)] = float(joint_state[1])
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(velocities, indices)
        self._refresh_object_poses()
        return self._compute_observation() if observation is not None else observation


__all__ = ["HandsOnYCB03AirplaneIsaacLabEnv"]
