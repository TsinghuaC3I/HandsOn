"""Isaac Sim collection environment for the pizza long-horizon task."""

from __future__ import annotations

import math
from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from ...runtime.gpu import PIZZA_CPU_PHYSX_GROUND_TRUTH_EXCEPTION


_PIZZA_TASK_NAME = "user_make_pizza_long_horizon"
_PIZZA_SLICES = tuple(f"pizza_slice{index}" for index in range(1, 9))
_PIZZA_INGREDIENTS = (
    "pizza_ketchup_sauce",
    "pizza_cheese_topping",
    "pizza_tomato_topping",
    "pizza_pepper_topping",
)
_PIZZA_CONTACT_OBJECTS = (
    "pizza_raw_dough",
    "pizza_rolling_pin",
    *_PIZZA_INGREDIENTS,
    "pizza_cutter",
)
_PIZZA_STAGE_NAMES = (
    "kneading",
    "shaping",
    "toppings",
    "cooking",
    "cutting",
    "plating",
    "complete",
)
_PIZZA_ASSEMBLY_XY = np.asarray([0.02, 0.05], dtype=np.float64)
_PIZZA_CONTACT_EPSILON = 1.0e-6
_PIZZA_KNEAD_MIN_SECONDS = 1.0
_PIZZA_KNEAD_MIN_PATH_M = 0.015
_PIZZA_SHAPE_MIN_SECONDS = 0.35
_PIZZA_SHAPE_MIN_PATH_M = 0.045
_PIZZA_COOK_MIN_SECONDS = 2.0
_PIZZA_STABLE_LINEAR_SPEED = 0.50
_PIZZA_STABLE_ANGULAR_SPEED = 4.0
_PIZZA_SLICE_PLATE_RADIUS_M = 0.18
_PIZZA_SLICE_PLATE_Z_RANGE_M = (-0.04, 0.18)
_PIZZA_CUT_PLANE_TOLERANCE_M = 0.10
_PIZZA_CUT_CENTER_TOLERANCE_M = 0.065
_PIZZA_CUT_MIN_ENDPOINT_RADIUS_M = 0.06
_PIZZA_CUT_MIN_PATH_M = 0.08
_PIZZA_CUT_MIN_ORIENTATION_SEPARATION_RAD = math.radians(40.0)


def _pizza_pose_distance(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return math.inf
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3 or not np.all(np.isfinite(a[:3])) or not np.all(np.isfinite(b[:3])):
        return math.inf
    return float(np.linalg.norm(a[:3] - b[:3]))


def _pizza_angle_distance(first: float, second: float) -> float:
    """Return the unoriented (mod-pi) difference between two cut headings."""

    delta = abs(float(first) - float(second)) % math.pi
    return float(min(delta, math.pi - delta))


class UserMakePizzaLongHorizonIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Native bimanual pizza environment with an auditable physical state machine.

    The scene still comes from the exact downloaded assets.  This class owns
    only the task truth: stage progression is driven by live PhysX contacts,
    relative poses, measured motion, and the source pizza slice nodes.  No
    action path writes an object pose or velocity.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # The host's GPU PhysX allocator is unstable for this imported asset
        # set. Keep the operator-test CPU path explicit and auditable so the
        # normal PhysX ground-truth recorder can still initialize.
        self._physx_cpu_ground_truth_exception = (
            PIZZA_CPU_PHYSX_GROUND_TRUTH_EXCEPTION
        )
        self._reset_pizza_episode_state()
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Scene/body ownership
    # ------------------------------------------------------------------

    def _pizza_source_slice_prim(self, name: str):
        if self._stage is None:
            return None
        from pxr import Usd, UsdPhysics

        target = str(name)
        if target == "pizza_assembled_slices":
            target = "slice1"
        elif target.startswith("pizza_slice"):
            target = target.replace("pizza_", "", 1)
        root = self._stage.GetPrimAtPath(
            "/World/Task/PizzaLongHorizon/pizza_assembled_slices"
        )
        if root is None or not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim.GetName() == target and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
        return None

    def _find_rigid_body_prim(self, name: str):
        if self.task_name == _PIZZA_TASK_NAME and (
            name == "pizza_assembled_slices" or name in _PIZZA_SLICES
        ):
            return self._pizza_source_slice_prim(name)
        return super()._find_rigid_body_prim(name)

    def _task_extra_pose_names(self) -> set[str]:
        extras = set(super()._task_extra_pose_names())
        if self.task_name == _PIZZA_TASK_NAME:
            extras.update(_PIZZA_SLICES)
        return extras

    def _task_specific_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        if self.task_name != _PIZZA_TASK_NAME or not self.randomize:
            return super()._task_specific_reset_xy_positions()
        # The source slice roots must retain identical XY at reset; otherwise
        # the authored fixed joints start with a stretched pizza assembly.
        return {
            "pizza_assembled_slices": _PIZZA_ASSEMBLY_XY.copy(),
            **{name: _PIZZA_ASSEMBLY_XY.copy() for name in _PIZZA_SLICES},
        }

    def _prim_pose(self, name: str, fallback: np.ndarray) -> np.ndarray:
        if self.task_name == _PIZZA_TASK_NAME and name == "pizza_assembled_slices":
            pose = self._pizza_assembly_pose()
            if pose is not None:
                return pose
        return super()._prim_pose(name, fallback)

    def _pizza_assembly_pose(self) -> np.ndarray | None:
        poses = [self._current_body_pose(name) for name in _PIZZA_SLICES]
        poses = [pose for pose in poses if pose is not None]
        if not poses:
            return None
        position = np.mean(np.stack([pose[:3] for pose in poses]), axis=0)
        quaternion = np.asarray(poses[0][3:7], dtype=np.float64).copy()
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-8:
            quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            quaternion /= norm
        return np.concatenate([position, quaternion]).astype(np.float64)

    def _current_body_pose(self, name: str) -> np.ndarray | None:
        if self.task_name == _PIZZA_TASK_NAME and name == "pizza_assembled_slices":
            pose = self._pizza_assembly_pose()
            if pose is not None:
                return pose
        return super()._current_body_pose(name)

    # ------------------------------------------------------------------
    # Runtime contact channels and reset state
    # ------------------------------------------------------------------

    def _configure_task_specific_contact_reporting(self) -> int:
        configured = super()._configure_task_specific_contact_reporting()
        if self.task_name != _PIZZA_TASK_NAME or self._stage is None:
            return configured
        from pxr import PhysxSchema

        names = (*_PIZZA_CONTACT_OBJECTS, "pizza_frying_pan", *_PIZZA_SLICES)
        missing: list[str] = []
        for name in names:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                missing.append(name)
                continue
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    prim.GetPath(),
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)
            configured += 1
        if missing:
            message = "Pizza contact reporting is incomplete: " + ",".join(missing)
            self._add_task_state_warning(message)
            raise RuntimeError(message)
        return configured

    def _init_task_specific_runtime(self) -> None:
        super()._init_task_specific_runtime()
        if self.task_name == _PIZZA_TASK_NAME:
            self._init_pizza_contact_views()

    def _init_pizza_contact_views(self) -> None:
        self._pizza_contact_views = {}
        self._pizza_contact_available = {}
        self._pizza_physical_truth_available = False
        if not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        hand_paths = self._c02_hand_body_paths()
        if physics_sim_view is None or not hand_paths:
            self._add_task_state_warning(
                "Pizza hand contact reporting is unavailable: no hand paths"
            )
            return

        def create_view(key: str, sensor_name: str, filters: list[str]) -> None:
            prim = self._find_rigid_body_prim(sensor_name)
            if prim is None:
                self._pizza_contact_available[key] = False
                return
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(prim.GetPath()),
                    filter_patterns=filters,
                    max_contact_data_count=max(64, 8 * len(filters)),
                )
                valid = bool(
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                )
                self._pizza_contact_available[key] = valid
                if valid:
                    self._pizza_contact_views[key] = view
            except Exception as exc:
                self._pizza_contact_available[key] = False
                self._add_task_state_warning(
                    f"Pizza contact view {key} unavailable: {exc}"
                )

        for name in _PIZZA_CONTACT_OBJECTS:
            create_view(f"hand:{name}", name, hand_paths)

        roller_prim = self._find_rigid_body_prim("pizza_rolling_pin")
        dough_prim = self._find_rigid_body_prim("pizza_raw_dough")
        if roller_prim is not None and dough_prim is not None:
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(dough_prim.GetPath()),
                    filter_patterns=[str(roller_prim.GetPath())],
                    max_contact_data_count=64,
                )
                self._pizza_contact_available["pizza:roller_dough"] = bool(
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                )
                if self._pizza_contact_available["pizza:roller_dough"]:
                    self._pizza_contact_views["pizza:roller_dough"] = view
            except Exception as exc:
                self._pizza_contact_available["pizza:roller_dough"] = False
                self._add_task_state_warning(
                    f"Pizza roller/dough contact view unavailable: {exc}"
                )
        else:
            self._pizza_contact_available["pizza:roller_dough"] = False

        pan_prim = self._find_rigid_body_prim("pizza_frying_pan")
        slice_paths = [
            str(prim.GetPath())
            for prim in (self._find_rigid_body_prim(name) for name in _PIZZA_SLICES)
            if prim is not None
        ]
        if pan_prim is not None and slice_paths:
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(pan_prim.GetPath()),
                    filter_patterns=slice_paths,
                    max_contact_data_count=64,
                )
                self._pizza_contact_available["pizza:pan"] = bool(
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                )
                if self._pizza_contact_available["pizza:pan"]:
                    self._pizza_contact_views["pizza:pan"] = view
            except Exception as exc:
                self._pizza_contact_available["pizza:pan"] = False
                self._add_task_state_warning(
                    f"Pizza pan/slice contact view unavailable: {exc}"
                )

        cutter_prim = self._find_rigid_body_prim("pizza_cutter")
        if cutter_prim is not None and slice_paths:
            for index, slice_name in enumerate(_PIZZA_SLICES, start=1):
                create_view(
                    f"cutter:{slice_name}",
                    slice_name,
                    [str(cutter_prim.GetPath())],
                )
        required = [
            *(f"hand:{name}" for name in _PIZZA_CONTACT_OBJECTS),
            "pizza:roller_dough",
            "pizza:pan",
            *(f"cutter:{name}" for name in _PIZZA_SLICES),
        ]
        self._pizza_physical_truth_available = all(
            self._pizza_contact_available.get(key, False) for key in required
        )

    def _reset_pizza_episode_state(self) -> None:
        self._pizza_stage_index = 0
        self._pizza_episode_started = False
        self._pizza_contact_views = getattr(self, "_pizza_contact_views", {})
        self._pizza_contact_available = getattr(self, "_pizza_contact_available", {})
        self._pizza_physical_truth_available = bool(
            getattr(self, "_pizza_physical_truth_available", False)
        )
        self._pizza_knead_hold_s = 0.0
        self._pizza_knead_path_m = 0.0
        self._pizza_shape_hold_s = 0.0
        self._pizza_shape_path_m = 0.0
        self._pizza_cook_hold_s = 0.0
        self._pizza_plating_hold_steps = 0
        self._pizza_ingredient_progress = 0
        self._pizza_ingredient_contact_seen = {
            name: False for name in _PIZZA_INGREDIENTS
        }
        self._pizza_ingredient_placed = {
            name: False for name in _PIZZA_INGREDIENTS
        }
        self._pizza_initial_poses: dict[str, np.ndarray] = {}
        self._pizza_previous_poses: dict[str, np.ndarray] = {}
        self._pizza_cut_paths: list[dict[str, float]] = []
        self._pizza_cut_track_start: np.ndarray | None = None
        self._pizza_cut_track_last: np.ndarray | None = None
        self._pizza_cut_track_length_m = 0.0
        self._pizza_slice_joints_released = False
        self._pizza_last_contacts: dict[str, float] = {}
        self._pizza_failure_reason = ""
        self._pizza_success_now = False
        if getattr(self, "task_name", None) == _PIZZA_TASK_NAME:
            for name in (
                "pizza_raw_dough",
                "pizza_kneading_bowl",
                "pizza_rolling_pin",
                "pizza_cutting_board",
                *_PIZZA_INGREDIENTS,
                "pizza_frying_pan",
                "pizza_assembled_slices",
                "pizza_cutter",
                "pizza_serving_plate",
                *_PIZZA_SLICES,
            ):
                pose = self._current_body_pose(name)
                if pose is not None:
                    self._pizza_initial_poses[name] = pose.copy()
                    self._pizza_previous_poses[name] = pose.copy()

    def _reset_scene_state(self) -> None:
        super()._reset_scene_state()
        if self.task_name == _PIZZA_TASK_NAME:
            self._reset_pizza_episode_state()
            self._set_pizza_slice_joints_enabled(True)

    def _before_control_step(self) -> None:
        super()._before_control_step()
        if self.task_name == _PIZZA_TASK_NAME:
            self._pizza_episode_started = True

    # ------------------------------------------------------------------
    # Physical state machine
    # ------------------------------------------------------------------

    def _pizza_contact_force(self, key: str) -> float:
        view = self._pizza_contact_views.get(key)
        if view is not None:
            try:
                matrix = view.get_contact_force_matrix(
                    float(getattr(self, "_sim_dt", 1.0 / 120.0))
                )
                if hasattr(matrix, "detach"):
                    matrix = matrix.detach().cpu().numpy()
                values = np.asarray(matrix, dtype=np.float64)
                if values.size and np.all(np.isfinite(values)):
                    return float(np.linalg.norm(values, axis=-1).sum())
            except Exception:
                pass
        # Interface-only tests and offline replay may inject a force sample;
        # live environments remain fail-closed when the channel is absent.
        return float(getattr(self, "_pizza_test_contact_forces", {}).get(key, 0.0))

    def _pizza_contact_now(self, key: str) -> bool:
        force = self._pizza_contact_force(key)
        self._pizza_last_contacts[key] = float(force)
        return bool(force > _PIZZA_CONTACT_EPSILON)

    def _pizza_body_speed_is_stable(self, name: str) -> bool:
        velocity = self._physics_body_velocity(name)
        if velocity is None:
            return False
        values = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if values.size < 6 or not np.all(np.isfinite(values)):
            return False
        return bool(
            np.linalg.norm(values[:3]) <= _PIZZA_STABLE_LINEAR_SPEED
            and np.linalg.norm(values[3:6]) <= _PIZZA_STABLE_ANGULAR_SPEED
        )

    @staticmethod
    def _pizza_in_zone(
        object_pose: np.ndarray | None,
        zone_pose: np.ndarray | None,
        *,
        radius: float,
        z_range: tuple[float, float] | None = None,
    ) -> bool:
        if object_pose is None or zone_pose is None:
            return False
        object_pose = np.asarray(object_pose, dtype=np.float64).reshape(7)
        zone_pose = np.asarray(zone_pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(object_pose)) or not np.all(np.isfinite(zone_pose)):
            return False
        if float(np.linalg.norm(object_pose[:2] - zone_pose[:2])) > float(radius):
            return False
        if z_range is not None:
            relative_z = float(object_pose[2] - zone_pose[2])
            if not (float(z_range[0]) <= relative_z <= float(z_range[1])):
                return False
        return True

    def _pizza_record_cut_path(self, previous: np.ndarray | None, current: np.ndarray | None) -> None:
        if previous is None or current is None or len(self._pizza_cut_paths) >= 2:
            return
        cutter_contact = any(
            self._pizza_contact_now(f"cutter:{name}") for name in _PIZZA_SLICES
        )
        if not cutter_contact:
            self._pizza_cut_track_start = None
            self._pizza_cut_track_last = None
            self._pizza_cut_track_length_m = 0.0
            return
        pizza = self._pizza_assembly_pose()
        if pizza is None:
            return
        previous = np.asarray(previous, dtype=np.float64).reshape(7)
        current = np.asarray(current, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
            return
        delta = current[:2] - previous[:2]
        length = float(np.linalg.norm(delta))

        def opposite_sides(start: np.ndarray, end: np.ndarray) -> bool:
            center = pizza[:2]
            return bool(
                float(np.dot(start[:2] - center, end[:2] - center)) <= 0.0
            )

        def try_accept(start: np.ndarray, end: np.ndarray, path_length: float) -> bool:
            delta_path = end[:2] - start[:2]
            chord_length = float(np.linalg.norm(delta_path))
            if chord_length < _PIZZA_CUT_MIN_PATH_M:
                return False
            if abs(float(end[2] - pizza[2])) > _PIZZA_CUT_PLANE_TOLERANCE_M:
                return False
            center = pizza[:2]
            start_segment = start[:2] - center
            end_segment = end[:2] - center
            # Use the scalar 2-D cross product explicitly; np.cross on two
            # vectors is deprecated in newer NumPy releases.
            cross = (
                float(delta_path[0]) * float(start_segment[1])
                - float(delta_path[1]) * float(start_segment[0])
            )
            cross_distance = abs(cross) / max(chord_length, 1.0e-8)
            if cross_distance > _PIZZA_CUT_CENTER_TOLERANCE_M:
                return False
            if not opposite_sides(start, end):
                return False
            if min(
                float(np.linalg.norm(start_segment)),
                float(np.linalg.norm(end_segment)),
            ) < _PIZZA_CUT_MIN_ENDPOINT_RADIUS_M:
                return False
            angle = float(math.atan2(delta_path[1], delta_path[0]) % math.pi)
            if any(
                _pizza_angle_distance(angle, item["angle_rad"])
                < _PIZZA_CUT_MIN_ORIENTATION_SEPARATION_RAD
                for item in self._pizza_cut_paths
            ):
                return False
            force = max(
                self._pizza_contact_force(f"cutter:{name}")
                for name in _PIZZA_SLICES
            )
            self._pizza_cut_paths.append(
                {
                    "angle_rad": angle,
                    "length_m": float(max(path_length, chord_length)),
                    "contact_force_n": float(force),
                }
            )
            return True

        # Prefer the current substep chord when it is large enough (this also
        # preserves a clean event for low-rate interface tests). For ordinary
        # 120 Hz PhysX motion, accumulate the cutter trajectory below.
        if length >= _PIZZA_CUT_MIN_PATH_M:
            accepted = try_accept(previous, current, length)
            endpoint_clear = min(
                float(np.linalg.norm(previous[:2] - pizza[:2])),
                float(np.linalg.norm(current[:2] - pizza[:2])),
            ) >= _PIZZA_CUT_MIN_ENDPOINT_RADIUS_M
            if accepted or (opposite_sides(previous, current) and endpoint_clear):
                self._pizza_cut_track_start = None
                self._pizza_cut_track_last = None
                self._pizza_cut_track_length_m = 0.0
            return

        if self._pizza_cut_track_start is None:
            self._pizza_cut_track_start = previous.copy()
            self._pizza_cut_track_length_m = 0.0
        track_last = self._pizza_cut_track_last
        if track_last is not None:
            self._pizza_cut_track_length_m += float(
                np.linalg.norm(current[:2] - track_last[:2])
            )
        else:
            self._pizza_cut_track_length_m += length
        self._pizza_cut_track_last = current.copy()
        start = self._pizza_cut_track_start
        if start is None:
            return
        if self._pizza_cut_track_length_m >= _PIZZA_CUT_MIN_PATH_M:
            accepted = try_accept(start, current, self._pizza_cut_track_length_m)
            endpoint_clear = min(
                float(np.linalg.norm(start[:2] - pizza[:2])),
                float(np.linalg.norm(current[:2] - pizza[:2])),
            ) >= _PIZZA_CUT_MIN_ENDPOINT_RADIUS_M
            if accepted or (opposite_sides(start, current) and endpoint_clear):
                self._pizza_cut_track_start = None
                self._pizza_cut_track_last = None
                self._pizza_cut_track_length_m = 0.0

    def _set_pizza_slice_joints_enabled(self, enabled: bool) -> None:
        if self._stage is None:
            return
        from pxr import Usd

        root = self._stage.GetPrimAtPath(
            "/World/Task/PizzaLongHorizon/pizza_assembled_slices/source_slice_joints"
        )
        if root is None or not root.IsValid():
            return
        for prim in Usd.PrimRange(root):
            if prim.IsValid() and prim.IsActive() != bool(enabled):
                prim.SetActive(bool(enabled))

    def _release_pizza_slice_joints(self) -> None:
        if self._pizza_slice_joints_released:
            return
        self._set_pizza_slice_joints_enabled(False)
        self._pizza_slice_joints_released = True

    def _update_pizza_physical_events(self) -> None:
        if self.task_name != _PIZZA_TASK_NAME or not self._pizza_episode_started:
            return
        dt = float(getattr(self, "_sim_dt", 0.0) or getattr(self, "_control_dt", 1.0 / 30.0))
        poses = {
            name: self._current_body_pose(name)
            for name in (
                "pizza_raw_dough",
                "pizza_kneading_bowl",
                "pizza_rolling_pin",
                "pizza_cutting_board",
                *_PIZZA_INGREDIENTS,
                "pizza_frying_pan",
                "pizza_assembled_slices",
                "pizza_cutter",
                "pizza_serving_plate",
                *_PIZZA_SLICES,
            )
        }
        previous_dough = self._pizza_previous_poses.get("pizza_raw_dough")
        previous_roller = self._pizza_previous_poses.get("pizza_rolling_pin")
        previous_cutter = self._pizza_previous_poses.get("pizza_cutter")
        dough = poses.get("pizza_raw_dough")
        bowl = poses.get("pizza_kneading_bowl")
        board = poses.get("pizza_cutting_board")
        pizza = poses.get("pizza_assembled_slices")
        pan = poses.get("pizza_frying_pan")
        plate = poses.get("pizza_serving_plate")

        if self._pizza_stage_index == 0:
            in_bowl = self._pizza_in_zone(dough, bowl, radius=0.14, z_range=(-0.04, 0.18))
            contacting = self._pizza_contact_now("hand:pizza_raw_dough")
            if in_bowl and contacting:
                self._pizza_knead_hold_s += dt
                self._pizza_knead_path_m += _pizza_pose_distance(previous_dough, dough)
            else:
                self._pizza_knead_hold_s = 0.0
                self._pizza_knead_path_m = 0.0
            if (
                self._pizza_knead_hold_s >= _PIZZA_KNEAD_MIN_SECONDS
                and self._pizza_knead_path_m >= _PIZZA_KNEAD_MIN_PATH_M
            ):
                self._pizza_stage_index = 1

        if self._pizza_stage_index == 1:
            on_board = self._pizza_in_zone(dough, board, radius=0.18, z_range=(-0.08, 0.16))
            roller_contact = self._pizza_contact_now("hand:pizza_rolling_pin")
            roller_dough_contact = self._pizza_contact_now("pizza:roller_dough")
            roller_near = _pizza_pose_distance(poses.get("pizza_rolling_pin"), dough) <= 0.16
            if on_board and roller_contact and roller_dough_contact and roller_near:
                self._pizza_shape_hold_s += dt
                self._pizza_shape_path_m += _pizza_pose_distance(previous_roller, poses.get("pizza_rolling_pin"))
            else:
                self._pizza_shape_hold_s = 0.0
                self._pizza_shape_path_m = 0.0
            if (
                self._pizza_shape_hold_s >= _PIZZA_SHAPE_MIN_SECONDS
                and self._pizza_shape_path_m >= _PIZZA_SHAPE_MIN_PATH_M
            ):
                self._pizza_stage_index = 2

        if self._pizza_stage_index == 2:
            for name in _PIZZA_INGREDIENTS:
                if self._pizza_contact_now(f"hand:{name}"):
                    self._pizza_ingredient_contact_seen[name] = True
            if self._pizza_ingredient_progress < len(_PIZZA_INGREDIENTS):
                name = _PIZZA_INGREDIENTS[self._pizza_ingredient_progress]
                initial = self._pizza_initial_poses.get(name)
                moved = _pizza_pose_distance(initial, poses.get(name)) >= 0.025
                placed = self._pizza_in_zone(
                    poses.get(name),
                    dough,
                    radius=0.14,
                    z_range=(-0.08, 0.18),
                )
                stable = self._pizza_body_speed_is_stable(name)
                if self._pizza_ingredient_contact_seen[name] and moved and placed and stable:
                    self._pizza_ingredient_placed[name] = True
                    self._pizza_ingredient_progress += 1
            if self._pizza_ingredient_progress >= len(_PIZZA_INGREDIENTS):
                self._pizza_stage_index = 3

        if self._pizza_stage_index == 3:
            in_pan = self._pizza_in_zone(pizza, pan, radius=0.18, z_range=(-0.08, 0.20))
            pan_contact = self._pizza_contact_now("pizza:pan")
            if in_pan and pan_contact and self._pizza_body_speed_is_stable("pizza_assembled_slices"):
                self._pizza_cook_hold_s += dt
            else:
                self._pizza_cook_hold_s = 0.0
            if self._pizza_cook_hold_s >= _PIZZA_COOK_MIN_SECONDS:
                self._pizza_stage_index = 4

        if self._pizza_stage_index == 4:
            self._pizza_record_cut_path(previous_cutter, poses.get("pizza_cutter"))
            if len(self._pizza_cut_paths) >= 2:
                self._release_pizza_slice_joints()
                self._pizza_stage_index = 5

        if self._pizza_stage_index == 5:
            plated = 0
            for name in _PIZZA_SLICES:
                pose = poses.get(name)
                initial = self._pizza_initial_poses.get(name)
                if (
                    self._pizza_in_zone(
                        pose,
                        plate,
                        radius=_PIZZA_SLICE_PLATE_RADIUS_M,
                        z_range=_PIZZA_SLICE_PLATE_Z_RANGE_M,
                    )
                    and _pizza_pose_distance(initial, pose) >= 0.025
                    and self._pizza_body_speed_is_stable(name)
                ):
                    plated += 1
            if plated == len(_PIZZA_SLICES):
                self._pizza_plating_hold_steps += 1
            else:
                self._pizza_plating_hold_steps = 0
            required = max(1, int(self._success_steps_required()))
            if self._pizza_plating_hold_steps >= required:
                self._pizza_stage_index = 6

        for name, pose in poses.items():
            if pose is not None:
                self._pizza_previous_poses[name] = np.asarray(pose, dtype=np.float64).copy()

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        if self.task_name == _PIZZA_TASK_NAME:
            self._update_pizza_physical_events()

    def _compute_success_now(self) -> bool:
        if self.task_name != _PIZZA_TASK_NAME:
            return super()._compute_success_now()
        if not self._pizza_episode_started:
            self._pizza_episode_started = True
            self._update_pizza_physical_events()
        self._pizza_success_now = bool(self._pizza_stage_index >= 6)
        return self._pizza_success_now

    def _compute_failure(self) -> bool:
        if self.task_name != _PIZZA_TASK_NAME:
            return super()._compute_failure()
        # Missing contact channels make a live episode uncertifiable, but do
        # not force-reset the operator preview before it has had a chance to
        # report its startup warning.  Once a physical episode begins, expose
        # the reason in info and fail closed only on non-finite task state.
        if self._pizza_episode_started:
            for name in (*_PIZZA_INGREDIENTS, *_PIZZA_SLICES):
                pose = self._current_body_pose(name)
                if pose is not None and not np.all(np.isfinite(pose)):
                    self._pizza_failure_reason = f"non_finite_pose:{name}"
                    return True
        return False

    # ------------------------------------------------------------------
    # Collection-facing observations and metadata
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        base = super()._build_observation_space()
        if self.task_name != _PIZZA_TASK_NAME:
            return base
        state_spaces = dict(base["state"].spaces)
        pizza_spaces = {
            "pizza_task_stage": spaces.Box(
                0.0, 1.0, shape=(len(_PIZZA_STAGE_NAMES),), dtype=np.float64
            ),
            "pizza_task_stage_index": spaces.Box(
                0.0, len(_PIZZA_STAGE_NAMES) - 1,
                shape=(1,), dtype=np.float64,
            ),
            "pizza_ingredient_order": spaces.Box(
                0.0, 1.0, shape=(len(_PIZZA_INGREDIENTS),), dtype=np.float64
            ),
            "pizza_cut_paths": spaces.Box(
                -np.inf, np.inf, shape=(2, 3), dtype=np.float64
            ),
            "pizza_slice_poses": spaces.Box(
                -np.inf, np.inf,
                shape=(len(_PIZZA_SLICES), 7), dtype=np.float64,
            ),
            "pizza_contact_state": spaces.Box(
                0.0, np.inf,
                shape=(len(_PIZZA_CONTACT_OBJECTS) + 1,), dtype=np.float64,
            ),
        }
        state_spaces.update(pizza_spaces)
        result: dict[str, Any] = {"state": spaces.Dict(state_spaces)}
        if "images" in base.spaces:
            result["images"] = base["images"]
        # These fields are duplicated outside ``state`` so the Zarr recorder
        # can retain fixed-shape semantic streams without parsing the policy
        # flattening.  The adapter exposes them through its pizza_ passthrough
        # prefix while the policy still receives the same values in state.
        result.update(pizza_spaces)
        return spaces.Dict(result)

    def _pizza_observation_state(self) -> dict[str, np.ndarray]:
        stage = np.zeros(len(_PIZZA_STAGE_NAMES), dtype=np.float64)
        stage[int(np.clip(self._pizza_stage_index, 0, len(stage) - 1))] = 1.0
        ingredient_order = np.asarray(
            [float(self._pizza_ingredient_placed[name]) for name in _PIZZA_INGREDIENTS],
            dtype=np.float64,
        )
        cut_paths = np.zeros((2, 3), dtype=np.float64)
        for index, item in enumerate(self._pizza_cut_paths[:2]):
            cut_paths[index] = [
                float(item.get("angle_rad", 0.0)),
                float(item.get("length_m", 0.0)),
                float(item.get("contact_force_n", 0.0)),
            ]
        slice_poses = np.stack(
            [
                self._current_body_pose(name)
                if self._current_body_pose(name) is not None
                else np.zeros(7, dtype=np.float64)
                for name in _PIZZA_SLICES
            ],
            axis=0,
        )
        contacts = np.asarray(
            [
                self._pizza_last_contacts.get(f"hand:{name}", 0.0)
                for name in _PIZZA_CONTACT_OBJECTS
            ]
            + [self._pizza_last_contacts.get("pizza:pan", 0.0)],
            dtype=np.float64,
        )
        return {
            "pizza_task_stage": stage,
            "pizza_task_stage_index": np.asarray(
                [float(self._pizza_stage_index)], dtype=np.float64
            ),
            "pizza_ingredient_order": ingredient_order,
            "pizza_cut_paths": cut_paths,
            "pizza_slice_poses": np.nan_to_num(slice_poses, nan=0.0, posinf=0.0, neginf=0.0),
            "pizza_contact_state": np.nan_to_num(contacts, nan=0.0, posinf=0.0, neginf=0.0),
        }

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        if self.task_name == _PIZZA_TASK_NAME:
            pizza_state = self._pizza_observation_state()
            observation["state"].update(pizza_state)
            observation.update(pizza_state)
        return observation

    def get_recording_schema(self) -> dict[str, Any] | None:
        base_schema = super().get_recording_schema()
        if self.task_name != _PIZZA_TASK_NAME:
            return base_schema
        schema = dict(base_schema or {})
        schema.setdefault("name", "handson.user_make_pizza_long_horizon")
        schema.setdefault("version", 1)
        fields = dict(schema.get("fields", {}))
        fields.update(
            {
                "pizza_task_stage": {
                    "shape": [len(_PIZZA_STAGE_NAMES)],
                    "dtype": "float64",
                    "source": "live_task_stage_state_machine",
                },
                "pizza_task_stage_index": {
                    "shape": [1],
                    "dtype": "float64",
                    "source": "live_task_stage_state_machine",
                },
                "pizza_ingredient_order": {
                    "shape": [len(_PIZZA_INGREDIENTS)],
                    "dtype": "float64",
                    "source": "ordered_physical_placement_latches",
                },
                "pizza_cut_paths": {
                    "shape": [2, 3],
                    "dtype": "float64",
                    "source": "cutter_slice_contact_and_path_geometry",
                },
                "pizza_slice_poses": {
                    "shape": [len(_PIZZA_SLICES), 7],
                    "dtype": "float64",
                    "source": "source_slice_rigid_body_readback",
                },
                "pizza_contact_state": {
                    "shape": [len(_PIZZA_CONTACT_OBJECTS) + 1],
                    "dtype": "float64",
                    "source": "physx_contact_force_readback",
                },
            }
        )
        schema["fields"] = fields
        return schema

    def _task_info(self) -> dict[str, Any]:
        info = dict(super()._task_info())
        if self.task_name != _PIZZA_TASK_NAME:
            return info
        slice_poses = {
            name: (
                self._current_body_pose(name).tolist()
                if self._current_body_pose(name) is not None
                else None
            )
            for name in _PIZZA_SLICES
        }
        info.update(
            {
                "task_stage": _PIZZA_STAGE_NAMES[
                    int(np.clip(self._pizza_stage_index, 0, len(_PIZZA_STAGE_NAMES) - 1))
                ],
                "task_stage_index": int(self._pizza_stage_index),
                "pizza_task_stage": _PIZZA_STAGE_NAMES[
                    int(np.clip(self._pizza_stage_index, 0, len(_PIZZA_STAGE_NAMES) - 1))
                ],
                "pizza_task_stage_index": int(self._pizza_stage_index),
                "ingredient_order": tuple(_PIZZA_INGREDIENTS),
                "pizza_ingredient_progress": int(self._pizza_ingredient_progress),
                "pizza_ingredient_placed": dict(self._pizza_ingredient_placed),
                "pizza_cut_paths": tuple(dict(item) for item in self._pizza_cut_paths),
                "pizza_cut_path_count": int(len(self._pizza_cut_paths)),
                "cut_paths": tuple(dict(item) for item in self._pizza_cut_paths),
                "slice_poses": slice_poses,
                "pizza_slice_poses": slice_poses,
                "pizza_slice_plate_count": int(self._pizza_plated_slice_count()),
                "pizza_physical_truth_available": bool(self._pizza_physical_truth_available),
                "pizza_success_now": bool(self._pizza_success_now),
                "pizza_success_counter": int(self._success_counter),
                "pizza_success_steps_required": int(self._success_steps_required()),
                "pizza_failure_reason": str(self._pizza_failure_reason),
                "pizza_formal_collection_ready": False,
            }
        )
        return info

    def _pizza_plated_slice_count(self) -> int:
        plate = self._current_body_pose("pizza_serving_plate")
        if plate is None:
            return 0
        count = 0
        for name in _PIZZA_SLICES:
            pose = self._current_body_pose(name)
            initial = self._pizza_initial_poses.get(name)
            if (
                self._pizza_in_zone(
                    pose,
                    plate,
                    radius=_PIZZA_SLICE_PLATE_RADIUS_M,
                    z_range=_PIZZA_SLICE_PLATE_Z_RANGE_M,
                )
                and _pizza_pose_distance(initial, pose) >= 0.025
            ):
                count += 1
        return count

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        from .user_make_pizza_long_horizon_scene import (
            hide_default_assembly_clutter,
            spawn_assets,
        )

        self._pizza_hidden_bootstrap_props = hide_default_assembly_clutter(
            self._stage
        )
        self._pizza_scene_report = spawn_assets(self._stage, sim_utils)
