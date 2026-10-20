"""Action-aligned PhysX ground-truth snapshots for demonstration recording.

The policy observation intentionally contains compact task state and commanded
robot targets.  That is the right contract for training, but it is not enough
to reconstruct the path PhysX actually simulated.  This module adds a separate
recording-only contract containing simulator readback after every control step.

No Isaac Lab modules are imported here.  The recorder operates on the public
tensor attributes already owned by an initialized environment and, when Kit is
running, copies PhysX's one-step contact-report buffer after every simulation
substep.  This keeps the layouts unit-testable without launching Kit.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from handson.runtime.gpu import (
    is_canonical_cuda_device,
    is_physx_recording_device_allowed,
)


PHYSX_GROUND_TRUTH_ENV = "HANDSON_RECORD_PHYSX_GROUND_TRUTH"
PHYSX_MAX_CONTACT_HEADERS_ENV = "HANDSON_PHYSX_MAX_CONTACT_HEADERS"
PHYSX_MAX_CONTACT_POINTS_ENV = "HANDSON_PHYSX_MAX_CONTACT_POINTS"
PHYSX_MAX_FRICTION_ANCHORS_ENV = "HANDSON_PHYSX_MAX_FRICTION_ANCHORS"
PHYSX_STATE_LAYOUT = (
    "position_xyz",
    "quaternion_wxyz",
    "linear_velocity_xyz",
    "angular_velocity_xyz",
)
ROBOT_SIDES = ("right", "left")
CONTACT_HEADER_ID_LAYOUT = (
    "actor0_path_id",
    "actor1_path_id",
    "collider0_path_id",
    "collider1_path_id",
    "stage_id",
)
CONTACT_HEADER_META_LAYOUT = (
    "event_type",
    "contact_data_offset",
    "contact_data_count",
    "friction_anchor_offset",
    "friction_anchor_count",
    "proto_index0",
    "proto_index1",
)
CONTACT_POINT_LAYOUT = (
    "position_xyz",
    "normal_xyz",
    "solved_impulse_xyz",
    "separation",
)
FRICTION_ANCHOR_LAYOUT = (
    "position_xyz",
    "solved_friction_impulse_xyz",
)
CONTACT_SIGNATURE_PAIR_PATH_LAYOUT = (
    "actor0_canonical_path_index",
    "collider0_canonical_path_index",
    "actor1_canonical_path_index",
    "collider1_canonical_path_index",
)
CONTACT_SIGNATURE_PAIR_COUNT_LAYOUT = (
    "contact_point_count",
    "friction_point_count",
)
CONTACT_SIGNATURE_PAIR_METRIC_LAYOUT = (
    "maximum_penetration_depth_m",
    "total_normal_impulse_Ns",
    "total_friction_impulse_Ns",
)
TENSOR_CONTACT_EVENT_TYPE = np.uint32(0xFFFFFFFE)


def _canonical_vector_norm3(values: Any) -> float:
    """Return a layout-independent Euclidean norm for a three-vector."""

    components = tuple(float(value) for value in values)
    if len(components) != 3:
        raise ValueError(
            "A canonical contact vector must contain exactly three values"
        )
    return math.sqrt(math.fsum(value * value for value in components))


def _canonical_normal_impulse3(
    impulse: Any,
    normal: Any,
    normal_norm: float,
) -> float:
    """Project a three-vector without NumPy layout-dependent rounding."""

    impulse_values = tuple(float(value) for value in impulse)
    normal_values = tuple(float(value) for value in normal)
    if len(impulse_values) != 3 or len(normal_values) != 3:
        raise ValueError(
            "Canonical contact impulse and normal must be three-vectors"
        )
    return abs(
        math.fsum(
            impulse_value * normal_value
            for impulse_value, normal_value in zip(
                impulse_values,
                normal_values,
                strict=True,
            )
        )
        / float(normal_norm)
    )


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return int(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def physx_ground_truth_enabled() -> bool:
    """Return whether the isolated PhysX-trace recorder is enabled."""

    return os.environ.get(PHYSX_GROUND_TRUTH_ENV, "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def configure_physx_contact_reports_for_recording(
    stage: Any,
    prim_paths: Any,
) -> tuple[str, ...]:
    """Author contact-report APIs before PhysX tensor views are created.

    Adding a contact-report API after the first simulation reset can rebuild a
    collision shape and invalidate Isaac Lab tensor views.  Environment setup
    therefore calls this helper after USD construction but before ``sim.reset``.
    """

    if not physx_ground_truth_enabled():
        return ()
    from pxr import PhysxSchema, UsdPhysics

    report_prims: dict[str, Any] = {}
    for raw_path in prim_paths:
        path = str(raw_path)
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                "Could not enable PhysX contact recording because the task "
                f"prim does not exist: {path}"
            )

        articulation_root = None
        ancestor = prim
        while ancestor and ancestor.IsValid():
            ancestor_path = str(ancestor.GetPath())
            if (
                ancestor.HasAPI(UsdPhysics.ArticulationRootAPI)
                and not ancestor_path.endswith("/worldBody")
            ):
                articulation_root = ancestor
                break
            ancestor = ancestor.GetParent()
        report_prim = articulation_root
        if report_prim is None:
            ancestor = prim
            while ancestor and ancestor.IsValid():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    report_prim = ancestor
                    break
                ancestor = ancestor.GetParent()
        if report_prim is None:
            continue
        report_prims[str(report_prim.GetPath())] = report_prim

    for path, prim in sorted(report_prims.items()):
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            report_api = PhysxSchema.PhysxContactReportAPI.Get(stage, path)
        else:
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
    return tuple(sorted(report_prims))


def _numpy(value: Any, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    elif hasattr(value, "warp"):
        value = value.warp
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _first_row(value: Any, *, dtype=np.float64) -> np.ndarray:
    array = _numpy(value, dtype=dtype)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    return array.reshape(-1, array.shape[-1])[0]


def _physx_transform_state(transform: Any, velocity: Any) -> np.ndarray:
    """Return one PhysX tensor-view state as xyz+wxyz+linear+angular.

    PhysX tensor views expose transforms as ``xyz+xyzw``.  Isaac Lab asset
    buffers instead expose root/body state quaternions as ``wxyz``.  All trace
    fields use the latter so robot, rigid-body, and articulation rows share one
    documented layout.
    """

    pose_xyzw = _first_row(transform).reshape(7)
    velocity_vector = _first_row(velocity).reshape(6)
    pose_wxyz = np.concatenate(
        [pose_xyzw[:3], pose_xyzw[6:7], pose_xyzw[3:6]]
    )
    return np.concatenate([pose_wxyz, velocity_vector]).astype(
        np.float64,
        copy=False,
    )


def _physx_transform_states(transform: Any, velocity: Any) -> np.ndarray:
    """Vectorized ``_physx_transform_state`` for one multi-body view."""

    pose_xyzw = _numpy(transform).reshape(-1, 7)
    velocity_vector = _numpy(velocity).reshape(-1, 6)
    if pose_xyzw.shape[0] != velocity_vector.shape[0]:
        raise RuntimeError(
            "PhysX rigid-body transform/velocity batch sizes differ: "
            f"pose={pose_xyzw.shape}, velocity={velocity_vector.shape}"
        )
    return np.concatenate(
        (
            pose_xyzw[:, :3],
            pose_xyzw[:, 6:7],
            pose_xyzw[:, 3:6],
            velocity_vector,
        ),
        axis=1,
    ).astype(np.float64, copy=False)


def _isaaclab_state(value: Any) -> np.ndarray:
    """Return one Isaac Lab root/body state already laid out as xyz+wxyz+v+w."""

    return _first_row(value).reshape(-1)[:13].astype(np.float64, copy=True)


@dataclass(frozen=True)
class _RobotEntry:
    side: str
    robot: Any | None
    joint_names: tuple[str, ...]
    joint_count: int
    tcp_body_index: int | None


@dataclass(frozen=True)
class _RigidBodyEntry:
    name: str
    source: str
    handle: Any
    indices: Any | None
    kinematic: bool


@dataclass(frozen=True)
class _ArticulationEntry:
    name: str
    view: Any
    indices: Any | None
    dof_names: tuple[str, ...]
    dof_count: int


@dataclass(frozen=True)
class _DeformableEntry:
    name: str
    position_attribute: str
    velocity_attribute: str
    invalid_attribute: str | None
    shape: tuple[int, int]


class PhysxGroundTruthTrace:
    """Describe and sample the fixed-shape PhysX trace for one environment."""

    _SUBSTEP_CAPTURE_FIELDS = (
        "physx_sim_time",
        "physx_robot_joint_position",
        "physx_robot_joint_velocity",
        "physx_robot_joint_target",
        "physx_robot_joint_effort",
        "physx_robot_joint_valid",
        "physx_robot_joint_target_valid",
        "physx_robot_joint_effort_valid",
        "physx_command_tcp_pose",
        "physx_command_hand",
        "physx_command_hand_valid",
        "physx_pose_commanded",
        "physx_tcp_state",
        "physx_tcp_valid",
        "physx_task_rigid_body_state",
        "physx_task_rigid_body_valid",
        "physx_task_articulation_root_state",
        "physx_task_articulation_root_valid",
        "physx_task_articulation_dof_position",
        "physx_task_articulation_dof_velocity",
        "physx_task_articulation_dof_valid",
        "physx_contact_report_valid",
        "physx_contact_capture_source",
        "physx_contact_header_count",
        "physx_contact_header_ids",
        "physx_contact_header_meta",
        "physx_contact_point_count",
        "physx_contact_point_data",
        "physx_contact_point_face_indices",
        "physx_contact_point_material_ids",
        "physx_friction_anchor_count",
        "physx_friction_anchor_data",
        "physx_contact_signature_valid",
        "physx_contact_signature_pair_count",
        "physx_contact_signature_pair_path_indices",
        "physx_contact_signature_pair_counts",
        "physx_contact_signature_pair_metrics",
    )

    def __init__(self, env: Any):
        self.env = env
        self._substep_snapshots: list[dict[str, np.ndarray]] = []
        # The final live substep is already the action-aligned state consumed
        # by the ordinary observation snapshot.  Retain that read-only sample
        # so the recorder does not synchronize and read every robot/body/
        # contact tensor a second time at the end of the same control step.
        self._latest_substep_sample: dict[str, np.ndarray] | None = None
        self.has_task_controller_state = callable(
            getattr(
                self.env,
                "get_physical_replay_task_controller_state",
                None,
            )
        )
        self.robots = self._build_robot_layout()
        self.hand_command_dim = max(
            1,
            int(getattr(self.env, "_hand_action_dim", 20)),
        )
        self.has_rng_state = callable(
            getattr(getattr(self.env, "_random", None), "get_state", None)
        ) and hasattr(self.env, "_seed")
        self.max_robot_joints = max(
            1,
            *(entry.joint_count for entry in self.robots),
        )
        self.rigid_bodies = self._build_rigid_body_layout()
        self._batched_rigid_body_view: Any | None = None
        self._batched_rigid_body_rows: tuple[tuple[int, int], ...] = ()
        self._initialize_batched_rigid_body_capture()
        self.articulations = self._build_articulation_layout()
        self.max_articulation_dofs = max(
            (entry.dof_count for entry in self.articulations),
            default=0,
        )
        self.deformable = self._deformable_layout()
        self.deformable_shape = (
            self.deformable.shape if self.deformable is not None else None
        )
        default_contact_headers = (
            256
            if getattr(self.env, "task_name", "")
            == "t42_sort_objects"
            else 128
        )
        self.max_contact_headers = _positive_env_int(
            PHYSX_MAX_CONTACT_HEADERS_ENV,
            default_contact_headers,
        )
        self.max_contact_points = _positive_env_int(
            PHYSX_MAX_CONTACT_POINTS_ENV,
            512,
        )
        self.max_friction_anchors = _positive_env_int(
            PHYSX_MAX_FRICTION_ANCHORS_ENV,
            512,
        )
        self._contact_report_interface: Any | None = None
        self._contact_report_subscription: Any | None = None
        self._pending_contact_fields: dict[str, np.ndarray] | None = None
        self._contact_report_callback_error: Exception | None = None
        self._contact_report_paths: tuple[str, ...] = ()
        self._contact_report_path_table: tuple[dict[str, Any], ...] = ()
        self._contact_stage_id = 0
        self._contact_tensor_dt = float(
            getattr(
                self.env,
                "_sim_dt",
                getattr(self.env, "_physics_dt", 1.0),
            )
        )
        if not np.isfinite(self._contact_tensor_dt) or self._contact_tensor_dt <= 0:
            self._contact_tensor_dt = 1.0
        self._tensor_contact_views: tuple[Any, ...] = ()
        self._tensor_contact_view_paths: tuple[
            tuple[tuple[str, ...], tuple[str, ...]], ...
        ] = ()
        self._tensor_contact_sensor_paths: tuple[str, ...] = ()
        self._tensor_contact_filter_paths: tuple[str, ...] = ()
        self._initialize_contact_report_capture()
        self._initialize_tensor_contact_capture()

    def _initialize_contact_report_capture(self) -> None:
        """Attach to preconfigured zero-threshold public contact reports."""

        injected_interface = getattr(
            self.env,
            "_physx_contact_report_interface",
            None,
        )
        stage = getattr(self.env, "_stage", None)
        if stage is None:
            self._contact_report_interface = injected_interface
            subscribe = getattr(
                injected_interface,
                "subscribe_contact_report_events",
                None,
            )
            if callable(subscribe):
                self._contact_report_subscription = subscribe(
                    self._on_contact_report_event
                )
            return

        configured_paths = getattr(
            self.env,
            "_physx_contact_report_paths_preconfigured",
            None,
        )
        if configured_paths is None:
            raise RuntimeError(
                "PhysX contact recording was not configured before sim.reset(); "
                "refusing to modify USD after tensor views were created"
            )
        try:
            import omni.physx
            from pxr import PhysicsSchemaTools, UsdPhysics
        except Exception as exc:
            raise RuntimeError(
                "PhysX ground-truth contact recording is enabled, but the "
                "contact-report APIs could not be imported"
            ) from exc

        interface = injected_interface or omni.physx.get_physx_simulation_interface()
        if interface is None:
            raise RuntimeError(
                "PhysX ground-truth contact recording could not acquire the "
                "simulation interface"
            )
        self._contact_report_interface = interface
        self._contact_report_paths = tuple(str(path) for path in configured_paths)
        try:
            import omni.usd

            self._contact_stage_id = int(
                omni.usd.get_context().get_stage_id()
            )
        except Exception:
            get_attached_stage = getattr(interface, "get_attached_stage", None)
            if callable(get_attached_stage):
                self._contact_stage_id = int(get_attached_stage())

        path_table = []
        for prim in stage.Traverse():
            if not prim.IsActive():
                continue
            if not (
                prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(UsdPhysics.MaterialAPI)
                or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ):
                continue
            prim_path = prim.GetPath()
            path_table.append(
                {
                    "id": int(PhysicsSchemaTools.sdfPathToInt(prim_path)),
                    "path": str(prim_path),
                }
            )
        self._contact_report_path_table = tuple(
            sorted(path_table, key=lambda value: value["path"])
        )
        subscribe = getattr(interface, "subscribe_contact_report_events", None)
        if callable(subscribe):
            self._contact_report_subscription = subscribe(
                self._on_contact_report_event
            )

    def _initialize_tensor_contact_capture(self) -> None:
        """Create detailed public contact views without modifying live USD."""

        stage = getattr(self.env, "_stage", None)
        if stage is None:
            return
        sensor_paths = tuple(
            str(path)
            for path in getattr(
                self.env,
                "_physx_contact_sensor_paths_preconfigured",
                (),
            )
        )
        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )
        sensor_paths = tuple(
            path for path in sensor_paths if path not in kinematic_paths
        )
        if not sensor_paths:
            return
        # P0-03 composes its exact Ruler035 articulation after the generic
        # scene graph and owns a fresh tensor simulation view which is already
        # proven by the task's force/penetration contract.  Reuse that same
        # public PhysX frontend for the formal detailed contact views.  Other
        # tasks retain the long-lived robot simulation view below.
        physics_sim_view = getattr(
            self.env,
            "_p003_contact_physics_view",
            None,
        )
        for robot in self._robot_mapping().values():
            if physics_sim_view is not None:
                break
            physics_sim_view = getattr(robot, "_physics_sim_view", None)
            if physics_sim_view is not None:
                break
        if physics_sim_view is None:
            raise RuntimeError(
                "PhysX contact recording could not acquire the tensor "
                "simulation view"
            )

        from pxr import UsdPhysics

        filter_paths = []
        for prim in stage.Traverse():
            if not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
            if enabled_attr.IsValid() and enabled_attr.HasAuthoredValueOpinion():
                if not bool(enabled_attr.Get()):
                    continue
            filter_paths.append(str(prim.GetPath()))
        filter_paths = sorted(dict.fromkeys(filter_paths))
        # One view accepts an explicit list of sensors and one filter list per
        # sensor.  Keeping all sensors in one view changes neither the public
        # pairwise contact payload nor its fixed recording capacity, but it
        # turns eight device-to-host tensor reads *per task body* into eight
        # reads for the complete task.  D04 otherwise performs nearly one
        # hundred synchronizing reads after every 120 Hz physics substep.
        sensor_filter_patterns = [list(filter_paths) for _ in sensor_paths]
        try:
            view = physics_sim_view.create_rigid_contact_view(
                list(sensor_paths),
                filter_patterns=sensor_filter_patterns,
                max_contact_data_count=max(
                    self.max_contact_points,
                    self.max_friction_anchors,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not create the batched detailed PhysX contact view for "
                f"{len(sensor_paths)} task bodies"
            ) from exc
        if (
            view is None
            or int(view.sensor_count) != len(sensor_paths)
            or int(view.filter_count) != len(filter_paths)
        ):
            raise RuntimeError(
                "Batched detailed PhysX contact view layout does not match "
                "its task/filter paths: "
                f"sensors={getattr(view, 'sensor_count', 0)}, "
                f"filters={getattr(view, 'filter_count', 0)}, "
                f"expected_sensors={len(sensor_paths)}, "
                f"expected_filters={len(filter_paths)}"
            )
        actual_sensor_paths = tuple(
            str(path) for path in getattr(view, "sensor_paths", sensor_paths)
        )
        raw_filter_paths = getattr(view, "filter_paths", filter_paths)
        if raw_filter_paths and isinstance(raw_filter_paths[0], (list, tuple)):
            filter_paths_by_sensor = tuple(
                tuple(str(path) for path in paths)
                for paths in raw_filter_paths
            )
            if (
                len(filter_paths_by_sensor) != len(actual_sensor_paths)
                or any(
                    paths != filter_paths_by_sensor[0]
                    for paths in filter_paths_by_sensor[1:]
                )
            ):
                raise RuntimeError(
                    "Batched detailed PhysX contact view reordered filters "
                    "differently for individual sensors"
                )
            actual_filter_paths = filter_paths_by_sensor[0]
        else:
            actual_filter_paths = tuple(
                str(path) for path in raw_filter_paths
            )
        if (
            len(actual_sensor_paths) != len(sensor_paths)
            or set(actual_sensor_paths) != set(sensor_paths)
            or len(actual_filter_paths) != len(filter_paths)
            or set(actual_filter_paths) != set(filter_paths)
        ):
            raise RuntimeError(
                "Batched detailed PhysX contact view returned a different "
                "sensor/filter path layout: "
                f"expected_sensors={sensor_paths}, "
                f"actual_sensors={actual_sensor_paths}, "
                f"expected_filters={tuple(filter_paths)}, "
                f"actual_filters={actual_filter_paths}"
            )
        self._tensor_contact_views = (view,)
        self._tensor_contact_view_paths = (
            (actual_sensor_paths, actual_filter_paths),
        )
        self._tensor_contact_sensor_paths = tuple(sensor_paths)
        self._tensor_contact_filter_paths = tuple(filter_paths)

    def _robot_mapping(self) -> dict[str, Any]:
        robots = getattr(self.env, "_robots", None)
        if isinstance(robots, Mapping) and robots:
            return {str(key): value for key, value in robots.items()}

        scene = getattr(self.env, "_scene", None)
        if scene is None:
            return {}
        result = {}
        for side in ROBOT_SIDES:
            try:
                result[side] = scene[f"{side}_robot"]
            except Exception:
                continue
        return result

    def _build_robot_layout(self) -> tuple[_RobotEntry, ...]:
        robot_mapping = self._robot_mapping()
        tcp_indices = getattr(self.env, "_ee_body_indices", {})
        fallback_count = max(
            1,
            7 + int(getattr(self.env, "_hand_action_dim", 20)),
        )
        entries = []
        for side in ROBOT_SIDES:
            robot = robot_mapping.get(side)
            names = tuple(str(name) for name in getattr(robot, "joint_names", ()))
            joint_count = len(names)
            if robot is not None and joint_count == 0:
                try:
                    joint_count = int(
                        _numpy(robot.data.joint_pos).shape[-1]
                    )
                except Exception:
                    joint_count = 0
            if joint_count <= 0:
                joint_count = fallback_count
                names = tuple(
                    f"{side}_joint_{index}"
                    for index in range(joint_count)
                )
            elif len(names) != joint_count:
                names = tuple(
                    f"{side}_joint_{index}"
                    for index in range(joint_count)
                )
            tcp_index = None
            if isinstance(tcp_indices, Mapping) and side in tcp_indices:
                try:
                    tcp_index = int(tcp_indices[side])
                except (TypeError, ValueError):
                    tcp_index = None
            entries.append(
                _RobotEntry(
                    side=side,
                    robot=robot,
                    joint_names=names,
                    joint_count=joint_count,
                    tcp_body_index=tcp_index,
                )
            )
        return tuple(entries)

    def _build_rigid_body_layout(self) -> tuple[_RigidBodyEntry, ...]:
        entries: list[_RigidBodyEntry] = []
        seen: set[str] = set()
        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )

        for state in getattr(self.env, "_task_reset_body_states", ()):
            if len(state) < 2:
                continue
            name = str(state[0])
            if name in seen:
                continue
            seen.add(name)
            entries.append(
                _RigidBodyEntry(
                    name=name,
                    source="physx_tensor_view",
                    handle=state[1],
                    indices=state[2] if len(state) > 2 else None,
                    kinematic=name in kinematic_paths,
                )
            )

        scene = getattr(self.env, "_scene", None)
        rigid_objects = getattr(scene, "rigid_objects", None)
        if isinstance(rigid_objects, Mapping):
            for raw_name, rigid_object in sorted(
                rigid_objects.items(),
                key=lambda item: str(item[0]),
            ):
                name = str(raw_name)
                if name in seen:
                    continue
                seen.add(name)
                cfg = getattr(rigid_object, "cfg", None)
                spawn = getattr(cfg, "spawn", None)
                rigid_props = getattr(spawn, "rigid_props", None)
                kinematic = bool(
                    getattr(rigid_props, "kinematic_enabled", False)
                )
                entries.append(
                    _RigidBodyEntry(
                        name=name,
                        source="isaaclab_root_state_w",
                        handle=rigid_object,
                        indices=None,
                        kinematic=kinematic,
                    )
                )
        return tuple(entries)

    def _initialize_batched_rigid_body_capture(self) -> None:
        """Create one read-only tensor view for all standalone task bodies."""

        indexed_entries = tuple(
            (index, entry)
            for index, entry in enumerate(self.rigid_bodies)
            if entry.source == "physx_tensor_view"
        )
        if len(indexed_entries) < 2:
            return
        physics_sim_view = None
        for robot in self._robot_mapping().values():
            physics_sim_view = getattr(robot, "_physics_sim_view", None)
            if physics_sim_view is not None:
                break
        if physics_sim_view is None:
            return
        expected_paths = tuple(entry.name for _index, entry in indexed_entries)
        try:
            view = physics_sim_view.create_rigid_body_view(list(expected_paths))
        except Exception as exc:
            raise RuntimeError(
                "Could not create the batched PhysX task-body trace view"
            ) from exc
        actual_paths = tuple(
            str(path) for path in getattr(view, "prim_paths", ())
        )
        if (
            view is None
            or int(getattr(view, "count", 0)) != len(expected_paths)
            or len(actual_paths) != len(expected_paths)
            or set(actual_paths) != set(expected_paths)
        ):
            raise RuntimeError(
                "Batched PhysX task-body trace view returned a different "
                f"layout: expected={expected_paths}, actual={actual_paths}"
            )
        batch_index_by_path = {
            path: index for index, path in enumerate(actual_paths)
        }
        self._batched_rigid_body_view = view
        self._batched_rigid_body_rows = tuple(
            (entry_index, batch_index_by_path[entry.name])
            for entry_index, entry in indexed_entries
        )

    @staticmethod
    def _articulation_dof_names(view: Any, dof_count: int) -> tuple[str, ...]:
        try:
            metatype = view.get_metatype(0)
            indices = getattr(metatype, "dof_indices", {})
            if isinstance(indices, Mapping):
                ordered = sorted(
                    ((int(index), str(name)) for name, index in indices.items()),
                    key=lambda item: item[0],
                )
                if len(ordered) == dof_count:
                    return tuple(name for _index, name in ordered)
        except Exception:
            pass
        return tuple(f"dof_{index}" for index in range(dof_count))

    def _articulation_root_is_writable(self, name: str) -> bool:
        predicate = getattr(
            self.env,
            "_task_articulation_root_is_writable",
            None,
        )
        return True if not callable(predicate) else bool(predicate(name))

    def _articulation_dofs_are_writable(self, name: str) -> bool:
        predicate = getattr(
            self.env,
            "_task_articulation_dofs_are_writable",
            None,
        )
        return True if not callable(predicate) else bool(predicate(name))

    def _articulation_root_velocity_is_writable(self, name: str) -> bool:
        predicate = getattr(
            self.env,
            "_task_articulation_root_velocity_is_writable",
            None,
        )
        if callable(predicate):
            return bool(predicate(name))
        return self._articulation_root_is_writable(name)

    def _articulation_dof_velocity_is_writable(self, name: str) -> bool:
        predicate = getattr(
            self.env,
            "_task_articulation_dof_velocity_is_writable",
            None,
        )
        if callable(predicate):
            return bool(predicate(name))
        return self._articulation_dofs_are_writable(name)

    def _build_articulation_layout(self) -> tuple[_ArticulationEntry, ...]:
        entries = []
        seen: set[str] = set()
        rigid_body_names = {entry.name for entry in self.rigid_bodies}
        for state in getattr(self.env, "_task_reset_articulation_states", ()):
            if len(state) < 2:
                continue
            name = str(state[0])
            if name in seen:
                continue
            view = state[1]
            try:
                dof_count = int(_numpy(view.get_dof_positions()).shape[-1])
            except Exception:
                dof_count = 0
            # MJCF free bodies may carry ArticulationRootAPI even though the
            # resulting PhysX articulation has no reduced-coordinate DOFs.
            # When the same actor is already available through a rigid-body
            # tensor view, recording/restoring both views writes the same root
            # twice and also prevents the dynamic rigid body from being put to
            # sleep.  Prefer the rigid-body representation for that case.
            if dof_count == 0 and name in rigid_body_names:
                continue
            seen.add(name)
            entries.append(
                _ArticulationEntry(
                    name=name,
                    view=view,
                    indices=state[2] if len(state) > 2 else None,
                    dof_names=self._articulation_dof_names(view, dof_count),
                    dof_count=dof_count,
                )
            )
        return tuple(entries)

    def _deformable_layout(self) -> _DeformableEntry | None:
        candidates = (
            (
                "surface",
                "_e_nodes_np",
                "_e_velocities_np",
                "_e_readback_failed",
            ),
            (
                "rope",
                "_e02_nodes_np",
                "_e02_nodal_velocities_np",
                None,
            ),
        )
        for name, position_attribute, velocity_attribute, invalid_attribute in (
            candidates
        ):
            positions = getattr(self.env, position_attribute, None)
            velocities = getattr(self.env, velocity_attribute, None)
            if positions is None or velocities is None:
                continue
            position_array = np.asarray(positions)
            velocity_array = np.asarray(velocities)
            if (
                position_array.ndim == 2
                and position_array.shape[-1] == 3
                and position_array.shape == velocity_array.shape
                and position_array.size > 0
            ):
                return _DeformableEntry(
                    name=name,
                    position_attribute=position_attribute,
                    velocity_attribute=velocity_attribute,
                    invalid_attribute=invalid_attribute,
                    shape=tuple(int(value) for value in position_array.shape),
                )
        return None

    @property
    def field_specs(self) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {
            "physx_robot_joint_position": {
                "shape": [2, self.max_robot_joints],
                "dtype": "float64",
                "layout": "right_then_left; padded to max joint count",
                "source": "ArticulationData.joint_pos",
                "units": "rad_or_m",
            },
            "physx_robot_joint_velocity": {
                "shape": [2, self.max_robot_joints],
                "dtype": "float64",
                "layout": "right_then_left; padded to max joint count",
                "source": "ArticulationData.joint_vel",
                "units": "rad_or_m_per_second",
            },
            "physx_robot_joint_target": {
                "shape": [2, self.max_robot_joints],
                "dtype": "float64",
                "layout": "right_then_left; padded to max joint count",
                "source": "position_controller_target",
                "units": "rad_or_m",
            },
            "physx_robot_joint_target_valid": {
                "shape": [2, self.max_robot_joints],
                "dtype": "uint8",
                "layout": "1=controller target is available",
            },
            "physx_robot_joint_valid": {
                "shape": [2, self.max_robot_joints],
                "dtype": "uint8",
                "layout": "1=joint exists; 0=padding_or_unavailable",
            },
            "physx_robot_root_state": {
                "shape": [2, 13],
                "dtype": "float64",
                "layout": list(PHYSX_STATE_LAYOUT),
                "source": "ArticulationData.root_state_w",
            },
            "physx_robot_root_valid": {
                "shape": [2],
                "dtype": "uint8",
                "layout": "right_then_left; 1=robot root readback available",
            },
            "physx_robot_joint_effort": {
                "shape": [2, self.max_robot_joints],
                "dtype": "float64",
                "layout": "right_then_left; padded to max joint count",
                "source": "ArticulationData.applied_or_computed_torque",
                "units": "N_or_Nm",
            },
            "physx_robot_joint_effort_valid": {
                "shape": [2, self.max_robot_joints],
                "dtype": "uint8",
                "layout": "1=joint effort readback available",
            },
            "physx_tcp_state": {
                "shape": [2, 13],
                "dtype": "float64",
                "layout": list(PHYSX_STATE_LAYOUT),
                "source": "palm_link body_state_w",
            },
            "physx_tcp_valid": {
                "shape": [2],
                "dtype": "uint8",
                "layout": "right_then_left; 1=PhysX readback available",
            },
            "physx_sim_time": {
                "shape": [1],
                "dtype": "float64",
                "source": "SimulationContext.current_time_or_control_clock",
                "units": "s",
            },
            "physx_control_step": {
                "shape": [1],
                "dtype": "float64",
                "source": "environment_control_step",
                "units": "step",
            },
            "physx_command_tcp_pose": {
                "shape": [2, 7],
                "dtype": "float64",
                "layout": "right_then_left; xyz+quaternion_wxyz",
                "source": "environment_tcp_command",
            },
            "physx_command_hand": {
                "shape": [2, self.hand_command_dim],
                "dtype": "float64",
                "layout": "right_then_left; complete hand command",
                "source": "environment_hand_command",
            },
            "physx_command_hand_valid": {
                "shape": [2, self.hand_command_dim],
                "dtype": "uint8",
                "layout": "1=command element is available",
            },
            "physx_pose_commanded": {
                "shape": [2],
                "dtype": "uint8",
                "layout": "right_then_left boolean",
            },
            "physx_pose_filter_state": {
                "shape": [2, 14],
                "dtype": "float64",
                "layout": (
                    "side_then_raw_xyz_raw_qwxyz_filtered_xyz_filtered_qwxyz"
                ),
                "source": "environment_arm_pose_filters",
            },
            "physx_pose_filter_valid": {
                "shape": [2],
                "dtype": "uint8",
                "layout": "right_then_left boolean",
            },
            "physx_contact_report_valid": {
                "shape": [1],
                "dtype": "uint8",
                "layout": "1=public PhysX contact buffers were read",
                "source": "PhysX contact report or rigid contact tensor",
            },
            "physx_contact_capture_source": {
                "shape": [1],
                "dtype": "uint8",
                "layout": "0=unavailable; 1=contact_report; 2=rigid_contact_tensor",
                "source": "recorder capture-path selector",
            },
            "physx_contact_header_count": {
                "shape": [1],
                "dtype": "uint32",
                "layout": "valid rows in physx_contact_header_*",
                "source": "ContactEventHeaderVector length",
            },
            "physx_contact_header_ids": {
                "shape": [self.max_contact_headers, 5],
                "dtype": "uint64",
                "layout": list(CONTACT_HEADER_ID_LAYOUT),
                "source": "ContactEventHeader",
            },
            "physx_contact_header_meta": {
                "shape": [self.max_contact_headers, 7],
                "dtype": "uint32",
                "layout": list(CONTACT_HEADER_META_LAYOUT),
                "source": "ContactEventHeader",
            },
            "physx_contact_point_count": {
                "shape": [1],
                "dtype": "uint32",
                "layout": "valid rows in physx_contact_point_*",
                "source": "ContactDataVector length",
            },
            "physx_contact_point_data": {
                "shape": [self.max_contact_points, 10],
                "dtype": "float32",
                "layout": list(CONTACT_POINT_LAYOUT),
                "source": "ContactData public solved-contact readback",
                "units": "m; unit_normal; N_s; m",
            },
            "physx_contact_point_face_indices": {
                "shape": [self.max_contact_points, 2],
                "dtype": "uint32",
                "layout": ["face_index0", "face_index1"],
                "source": "ContactData",
            },
            "physx_contact_point_material_ids": {
                "shape": [self.max_contact_points, 2],
                "dtype": "uint64",
                "layout": ["material0_path_id", "material1_path_id"],
                "source": "ContactData",
            },
            "physx_friction_anchor_count": {
                "shape": [1],
                "dtype": "uint32",
                "layout": "valid rows in physx_friction_anchor_data",
                "source": "FrictionAnchorsDataVector length",
            },
            "physx_friction_anchor_data": {
                "shape": [self.max_friction_anchors, 6],
                "dtype": "float32",
                "layout": list(FRICTION_ANCHOR_LAYOUT),
                "source": (
                    "FrictionAnchor or RigidContactView friction-point public "
                    "solved-friction readback"
                ),
                "units": "m; N_s",
            },
            "physx_contact_signature_valid": {
                "shape": [1],
                "dtype": "uint8",
                "layout": (
                    "1=canonical path-resolved contact signature is complete"
                ),
                "source": "canonicalized public PhysX contact readback",
            },
            "physx_contact_signature_pair_count": {
                "shape": [1],
                "dtype": "uint32",
                "layout": "valid canonical pair rows",
                "source": "canonicalized public PhysX contact readback",
            },
            "physx_contact_signature_pair_path_indices": {
                "shape": [self.max_contact_headers, 4],
                "dtype": "uint32",
                "layout": list(CONTACT_SIGNATURE_PAIR_PATH_LAYOUT),
                "source": (
                    "1-based index in contact_report.path_table sorted by path"
                ),
            },
            "physx_contact_signature_pair_counts": {
                "shape": [self.max_contact_headers, 2],
                "dtype": "uint32",
                "layout": list(CONTACT_SIGNATURE_PAIR_COUNT_LAYOUT),
                "source": "order-independent active pair aggregation",
            },
            "physx_contact_signature_pair_metrics": {
                "shape": [self.max_contact_headers, 3],
                "dtype": "float64",
                "layout": list(CONTACT_SIGNATURE_PAIR_METRIC_LAYOUT),
                "source": "order-independent active pair aggregation",
                "units": "m; N_s; N_s",
            },
        }
        if self.has_rng_state:
            fields.update(
                {
                    "physx_rng_seed": {
                        "shape": [1],
                        "dtype": "int64",
                        "source": "environment_seed",
                    },
                    "physx_rng_mt19937_keys": {
                        "shape": [624],
                        "dtype": "uint32",
                        "source": "numpy_random_state.get_state",
                    },
                    "physx_rng_mt19937_position": {
                        "shape": [1],
                        "dtype": "int64",
                        "source": "numpy_random_state.get_state",
                    },
                    "physx_rng_mt19937_has_gauss": {
                        "shape": [1],
                        "dtype": "uint8",
                        "source": "numpy_random_state.get_state",
                    },
                    "physx_rng_mt19937_cached_gaussian": {
                        "shape": [1],
                        "dtype": "float64",
                        "source": "numpy_random_state.get_state",
                    },
                }
            )
        if self.rigid_bodies:
            fields.update(
                {
                    "physx_task_rigid_body_state": {
                        "shape": [len(self.rigid_bodies), 13],
                        "dtype": "float64",
                        "layout": list(PHYSX_STATE_LAYOUT),
                        "source": "PhysX rigid body readback",
                    },
                    "physx_task_rigid_body_valid": {
                        "shape": [len(self.rigid_bodies)],
                        "dtype": "uint8",
                        "layout": "1=readback valid",
                    },
                }
            )
        if self.articulations:
            fields.update(
                {
                    "physx_task_articulation_root_state": {
                        "shape": [len(self.articulations), 13],
                        "dtype": "float64",
                        "layout": list(PHYSX_STATE_LAYOUT),
                        "source": "PhysX articulation root readback",
                    },
                    "physx_task_articulation_root_valid": {
                        "shape": [len(self.articulations)],
                        "dtype": "uint8",
                        "layout": "1=readback valid",
                    },
                }
            )
        if self.max_articulation_dofs > 0:
            fields.update(
                {
                    "physx_task_articulation_dof_position": {
                        "shape": [
                            len(self.articulations),
                            self.max_articulation_dofs,
                        ],
                        "dtype": "float64",
                        "source": "PhysX articulation DOF readback",
                        "units": "rad_or_m",
                    },
                    "physx_task_articulation_dof_velocity": {
                        "shape": [
                            len(self.articulations),
                            self.max_articulation_dofs,
                        ],
                        "dtype": "float64",
                        "source": "PhysX articulation DOF readback",
                        "units": "rad_or_m_per_second",
                    },
                    "physx_task_articulation_dof_valid": {
                        "shape": [
                            len(self.articulations),
                            self.max_articulation_dofs,
                        ],
                        "dtype": "uint8",
                        "layout": "1=DOF exists and readback is available",
                    },
                }
            )
        if self.deformable_shape is not None:
            fields.update(
                {
                    "physx_deformable_position": {
                        "shape": list(self.deformable_shape),
                        "dtype": "float64",
                        "source": "PhysX nodal position readback",
                        "units": "m",
                    },
                    "physx_deformable_velocity": {
                        "shape": list(self.deformable_shape),
                        "dtype": "float64",
                        "source": "PhysX nodal velocity readback",
                        "units": "m_per_second",
                    },
                    "physx_deformable_valid": {
                        "shape": [1],
                        "dtype": "uint8",
                        "layout": "1=finite accepted PhysX nodal readback",
                    },
                }
            )
        if self.has_task_controller_state:
            fields["physx_task_controller_state"] = {
                "shape": [3],
                "dtype": "uint8",
                "layout": (
                    "task_defined_phase_code,configure_preparing,"
                    "workspace_preparing"
                ),
                "source": (
                    "environment action-aligned task controller; installed "
                    "before raw action during physical replay"
                ),
                "replay_restore_supported": False,
            }
        substep_count = max(1, int(getattr(self.env, "_n_substeps", 1)))
        for source_name in self._SUBSTEP_CAPTURE_FIELDS:
            source_spec = fields.get(source_name)
            if source_spec is None:
                continue
            substep_name = source_name.replace(
                "physx_",
                "physx_substep_",
                1,
            )
            substep_spec = dict(source_spec)
            substep_spec["shape"] = [
                substep_count,
                *source_spec["shape"],
            ]
            substep_spec["source"] = (
                "per_physics_substep:"
                + str(source_spec.get("source", source_name))
            )
            fields[substep_name] = substep_spec
        fields["physx_substep_count"] = {
            "shape": [1],
            "dtype": "int64",
            "layout": "number of valid physics substeps in this control step",
            "source": "environment_step_substep_counter",
        }
        for name, spec in fields.items():
            if np.dtype(spec["dtype"]).kind in "biu":
                tolerance = 0.0
            elif name == "physx_sim_time":
                tolerance = 1.0e-12
            elif "velocity" in name:
                tolerance = (
                    1.0e-4 if "deformable" in name else 1.0e-5
                )
            elif "effort" in name:
                tolerance = 1.0e-4
            elif "deformable_position" in name:
                tolerance = 1.0e-5
            elif "command_" in name or "filter_state" in name:
                tolerance = 1.0e-9
            else:
                tolerance = 1.0e-6
            is_canonical_contact_signature = name.startswith(
                "physx_contact_signature_"
            ) or name.startswith("physx_substep_contact_signature_")
            is_contact_report = (
                name.startswith("physx_contact_")
                or name.startswith("physx_friction_anchor_")
                or name.startswith("physx_substep_contact_")
                or name.startswith("physx_substep_friction_anchor_")
            )
            is_non_restorable_solver_output = (
                name == "physx_robot_joint_effort"
            )
            spec["physical_replay"] = (
                is_canonical_contact_signature or not is_contact_report
            )
            if is_contact_report or is_non_restorable_solver_output:
                spec["replay_restore_supported"] = False
            spec["replay_atol"] = tolerance

        override_provider = getattr(
            self.env,
            "_physx_ground_truth_replay_atol_overrides",
            None,
        )
        overrides = override_provider() if callable(override_provider) else {}
        if not isinstance(overrides, Mapping):
            raise ValueError(
                "PhysX replay tolerance overrides must be a mapping"
            )
        supported_override_fields = set(fields)
        supported_override_fields.update(self._SUBSTEP_CAPTURE_FIELDS)
        supported_override_fields.update(
            name.replace("physx_", "physx_substep_", 1)
            for name in self._SUBSTEP_CAPTURE_FIELDS
        )
        unknown_fields = sorted(set(overrides) - supported_override_fields)
        if unknown_fields:
            raise ValueError(
                "PhysX replay tolerance overrides contain unknown fields: "
                + ", ".join(str(name) for name in unknown_fields)
            )
        for name, raw_tolerance in overrides.items():
            if (
                not isinstance(name, str)
                or type(raw_tolerance) not in {int, float}
                or isinstance(raw_tolerance, bool)
                or not math.isfinite(float(raw_tolerance))
                or float(raw_tolerance) < 0.0
            ):
                raise ValueError(
                    f"Invalid PhysX replay tolerance override: "
                    f"{name!r}={raw_tolerance!r}"
                )
            if name not in fields:
                # Task bodies and articulations are not present in the trace
                # created during early environment construction.  The same
                # provider is validated and applied when the post-composition
                # trace exposes those optional fields.
                continue
            if np.dtype(fields[name]["dtype"]).kind in "biu" and float(
                raw_tolerance
            ) != 0.0:
                raise ValueError(
                    f"Integer PhysX field {name} cannot use a nonzero tolerance"
                )
            fields[name]["replay_atol"] = float(raw_tolerance)
        return fields

    @property
    def observation_space_specs(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        return {
            name: (
                tuple(int(value) for value in spec["shape"]),
                np.dtype(spec["dtype"]),
            )
            for name, spec in self.field_specs.items()
        }

    def _physics_config(self) -> dict[str, Any]:
        sim = getattr(self.env, "_sim", None)
        sim_cfg = getattr(sim, "cfg", None)
        physx_cfg = getattr(sim_cfg, "physx", None)
        solver_names = (
            "solver_type",
            "enable_ccd",
            "enable_enhanced_determinism",
            "max_position_iteration_count",
            "max_velocity_iteration_count",
            "bounce_threshold_velocity",
            "friction_offset_threshold",
            "friction_correlation_distance",
            "gpu_max_rigid_contact_count",
            "gpu_max_rigid_patch_count",
        )
        solver = {}
        if physx_cfg is not None:
            for name in solver_names:
                value = getattr(physx_cfg, name, None)
                if isinstance(value, (str, bool, int, float)):
                    solver[name] = value
        device = getattr(self.env, "_isaac_device", "")
        cpu_exception = getattr(
            self.env,
            "_physx_cpu_ground_truth_exception",
            "",
        )
        if not is_physx_recording_device_allowed(
            device,
            cpu_exception=cpu_exception,
        ):
            raise RuntimeError(
                "PhysX ground-truth schema requires a canonical CUDA device "
                "or a schema-bound exact-asset CPU contract; "
                f"got {device!r}."
            )
        return {
            "device": device,
            "physics_dt": float(getattr(self.env, "_physics_dt", 0.0)),
            "live_physics_dt": float(
                getattr(
                    self.env,
                    "_sim_dt",
                    getattr(self.env, "_physics_dt", 0.0),
                )
            ),
            "control_dt": float(getattr(self.env, "_control_dt", 0.0)),
            "n_substeps": int(getattr(self.env, "_n_substeps", 0)),
            "simulation_dt": (
                float(getattr(sim_cfg, "dt"))
                if isinstance(getattr(sim_cfg, "dt", None), (int, float))
                else None
            ),
            "render_interval": (
                int(getattr(sim_cfg, "render_interval"))
                if isinstance(
                    getattr(sim_cfg, "render_interval", None),
                    (int, float),
                )
                else None
            ),
            "solver": solver,
        }

    def schema(self) -> dict[str, Any]:
        physics_config = self._physics_config()
        deterministic_kinematic_hand_inputs = bool(
            getattr(self.env, "_simplified_post_ik", False)
        )
        cpu_exception = getattr(
            self.env,
            "_physx_cpu_ground_truth_exception",
            "",
        )
        return {
            "name": "handson.physx_ground_truth",
            "version": 6,
            "description": (
                "Action-aligned simulator readback. Current fields describe the "
                "state before action[t]; next_* fields describe the state after "
                "action[t]. physx_substep_* fields preserve every live PhysX "
                "substep inside that control step, including public contact and "
                "friction-anchor reports."
            ),
            "fields": self.field_specs,
            "physx": {
                "bitwise_checkpoint": False,
                "requires_fresh_process_certification": True,
                "requires_cuda_backend": is_canonical_cuda_device(
                    physics_config["device"]
                ),
                "cpu_backend_exception": cpu_exception,
                "certification_forbids_state_writes": False,
                "allows_single_recorded_t0_restore": True,
                "certification_forbids_state_writes_after_action_start": (
                    not deterministic_kinematic_hand_inputs
                ),
                "deterministic_kinematic_hand_inputs": (
                    deterministic_kinematic_hand_inputs
                ),
                "capture_mode": (
                    "read_only_live_public_tensor_and_contact_report_snapshot"
                ),
                "unavailable_public_state": [
                    "persistent_contact_manifold_feature_matching_cache",
                    "solver_friction_patch_cache_beyond_reported_anchors",
                    "solver_warm_start_impulses_and_lagrange_multipliers",
                    "solver_island_sleep_ccd_broadphase_and_pair_caches",
                ],
                "contact_report": {
                    "available_in_this_runtime": (
                        self._contact_report_interface is not None
                    ),
                    "capture_api": (
                        "IPhysxSimulation.subscribe_contact_report_events+"
                        "get_full_contact_report, with "
                        "RigidContactView.get_contact_data/get_friction_data "
                        "fallback"
                    ),
                    "buffer_lifetime": "one_simulation_step",
                    "post_step_callback_copy": True,
                    "capture_frequency": "every_physics_substep",
                    "overflow_policy": "raise_without_writing_partial_data",
                    "max_headers": self.max_contact_headers,
                    "max_contact_points": self.max_contact_points,
                    "max_friction_anchors": self.max_friction_anchors,
                    "report_enabled_task_body_paths": list(
                        self._contact_report_paths
                    ),
                    "path_id_encoding": "PhysicsSchemaTools.sdfPathToInt",
                    "path_table": list(self._contact_report_path_table),
                    "capture_source_encoding": {
                        "0": "unavailable",
                        "1": "full_contact_report",
                        "2": "rigid_contact_tensor",
                    },
                    "tensor_fallback_available_in_this_runtime": bool(
                        self._tensor_contact_views
                    ),
                    "tensor_sensor_paths": list(
                        self._tensor_contact_sensor_paths
                    ),
                    "tensor_filter_paths": list(
                        self._tensor_contact_filter_paths
                    ),
                    "tensor_event_type_sentinel": int(
                        TENSOR_CONTACT_EVENT_TYPE
                    ),
                    "tensor_source_actor_and_collider_ids_are_body_ids": True,
                    "tensor_source_face_and_material_ids_available": False,
                    "tensor_source_friction_rows_are_points_not_cache_anchors": True,
                    "reported_contact_points_are_manifold_snapshot": True,
                    "reported_friction_anchors_are_internal_cache": False,
                    "reported_impulses_are_current_step_solver_output": True,
                    "solver_warm_start_state_available": False,
                    "replay_restore_supported": False,
                },
                "state_layout": list(PHYSX_STATE_LAYOUT),
                "robot_sides": list(ROBOT_SIDES),
                "robot_joint_names": {
                    entry.side: list(entry.joint_names)
                    for entry in self.robots
                },
                "rigid_body_names": [
                    entry.name for entry in self.rigid_bodies
                ],
                "rigid_body_sources": [
                    entry.source for entry in self.rigid_bodies
                ],
                "articulation_names": [
                    entry.name for entry in self.articulations
                ],
                "articulation_dof_names": {
                    entry.name: list(entry.dof_names)
                    for entry in self.articulations
                },
                "deformable_name": (
                    self.deformable.name
                    if self.deformable is not None
                    else None
                ),
                "deformable_position_source": (
                    self.deformable.position_attribute
                    if self.deformable is not None
                    else None
                ),
                "deformable_velocity_source": (
                    self.deformable.velocity_attribute
                    if self.deformable is not None
                    else None
                ),
                "physics_config": physics_config,
                "physics_dt": physics_config["physics_dt"],
                "control_dt": physics_config["control_dt"],
                "n_substeps": physics_config["n_substeps"],
                "enhanced_determinism": bool(
                    physics_config["solver"].get(
                        "enable_enhanced_determinism",
                        False,
                    )
                ),
            },
        }

    @staticmethod
    def _schema_physx(recorded_schema: Mapping[str, Any]) -> Mapping[str, Any]:
        schema: Any = recorded_schema
        environment_schema = schema.get("environment_schema")
        if isinstance(environment_schema, Mapping):
            schema = environment_schema
        physx = schema.get("physx") if isinstance(schema, Mapping) else None
        if not isinstance(physx, Mapping):
            raise ValueError("Recording schema does not contain environment_schema.physx")
        return physx

    @staticmethod
    def _require_unique_names(
        values: Any,
        *,
        label: str,
    ) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Recording schema {label} must be a list")
        names = tuple(str(value) for value in values)
        if len(set(names)) != len(names):
            raise ValueError(f"Recording schema {label} contains duplicate names")
        return names

    def validate_restore_schema(
        self,
        recorded_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Validate that a recorded entity layout can be written to this env.

        Replay maps by side, joint, and task-entity name instead of assuming
        the current implementation happens to enumerate assets in the same
        order as the recording.
        """

        physx = dict(self._schema_physx(recorded_schema))
        structured_robots = physx.get("robots")
        if not isinstance(structured_robots, (list, tuple)):
            structured_robots = ()

        def _structured_entity_aliases(values: Any) -> tuple[str, ...]:
            if not isinstance(values, (list, tuple)):
                return ()
            aliases: list[str] = []
            for value in values:
                if not isinstance(value, Mapping):
                    continue
                explicit_name = value.get("name")
                if explicit_name:
                    aliases.append(str(explicit_name))
                path = value.get("path")
                if path:
                    path_text = str(path)
                    aliases.append(path_text)
                    basename = path_text.rstrip("/").rsplit("/", 1)[-1]
                    aliases.append(basename)
                    aliases.append(basename.lstrip("_"))
            return tuple(dict.fromkeys(aliases))

        recorded_sides = physx.get("robot_sides")
        if not isinstance(recorded_sides, (list, tuple)):
            recorded_sides = physx.get("sides")
        if not isinstance(recorded_sides, (list, tuple)):
            recorded_sides = [
                entry.get("side")
                for entry in structured_robots
                if isinstance(entry, Mapping) and entry.get("side")
            ]
        sides = self._require_unique_names(
            recorded_sides,
            label="robot_sides",
        )
        physx["robot_sides"] = list(sides)
        joint_names = physx.get("robot_joint_names")
        if not isinstance(joint_names, Mapping):
            joint_names = {
                str(entry["side"]): entry.get("joint_names", ())
                for entry in structured_robots
                if isinstance(entry, Mapping) and entry.get("side")
            }
        if not isinstance(joint_names, Mapping):
            raise ValueError("Recording schema robot_joint_names must be an object")
        physx["robot_joint_names"] = dict(joint_names)
        for entry in self.robots:
            if entry.side not in sides:
                raise ValueError(
                    f"Recording schema is missing robot side {entry.side!r}"
                )
            recorded = self._require_unique_names(
                joint_names.get(entry.side),
                label=f"robot_joint_names[{entry.side!r}]",
            )
            missing = sorted(set(entry.joint_names) - set(recorded))
            if missing:
                raise ValueError(
                    f"Recording schema is missing {entry.side} joints: {missing}"
                )

        recorded_rigid_names = physx.get("rigid_body_names")
        if not isinstance(recorded_rigid_names, (list, tuple)):
            current_rigid_names = {entry.name for entry in self.rigid_bodies}
            recorded_rigid_names = []
            for structured_entry in physx.get("rigid_bodies", ()):
                aliases = _structured_entity_aliases((structured_entry,))
                matched = next(
                    (alias for alias in aliases if alias in current_rigid_names),
                    aliases[0] if aliases else "",
                )
                recorded_rigid_names.append(matched)
        recorded_rigid = self._require_unique_names(
            recorded_rigid_names,
            label="rigid_body_names",
        )
        physx["rigid_body_names"] = list(recorded_rigid)
        missing_rigid = sorted(
            {entry.name for entry in self.rigid_bodies} - set(recorded_rigid)
        )
        if missing_rigid:
            raise ValueError(
                f"Recording schema is missing rigid bodies: {missing_rigid}"
            )

        structured_articulations = physx.get("articulations")
        recorded_articulation_names = physx.get("articulation_names")
        if not isinstance(recorded_articulation_names, (list, tuple)):
            current_articulation_names = {
                entry.name for entry in self.articulations
            }
            recorded_articulation_names = []
            for structured_entry in structured_articulations or ():
                aliases = _structured_entity_aliases((structured_entry,))
                matched = next(
                    (
                        alias
                        for alias in aliases
                        if alias in current_articulation_names
                    ),
                    aliases[0] if aliases else "",
                )
                recorded_articulation_names.append(matched)
        recorded_articulations = self._require_unique_names(
            recorded_articulation_names,
            label="articulation_names",
        )
        physx["articulation_names"] = list(recorded_articulations)
        missing_articulations = sorted(
            {entry.name for entry in self.articulations}
            - set(recorded_articulations)
        )
        if missing_articulations:
            raise ValueError(
                "Recording schema is missing articulations: "
                f"{missing_articulations}"
            )
        articulation_dofs = physx.get("articulation_dof_names")
        if not isinstance(articulation_dofs, Mapping) and isinstance(
            structured_articulations, (list, tuple)
        ):
            articulation_dofs = {}
            for structured_entry in structured_articulations:
                if not isinstance(structured_entry, Mapping):
                    continue
                dof_names = structured_entry.get("joint_names", ())
                for alias in _structured_entity_aliases((structured_entry,)):
                    articulation_dofs[alias] = dof_names
        if not isinstance(articulation_dofs, Mapping):
            raise ValueError(
                "Recording schema articulation_dof_names must be an object"
            )
        physx["articulation_dof_names"] = dict(articulation_dofs)
        for entry in self.articulations:
            recorded = self._require_unique_names(
                articulation_dofs.get(entry.name),
                label=f"articulation_dof_names[{entry.name!r}]",
            )
            missing = sorted(set(entry.dof_names) - set(recorded))
            if missing:
                raise ValueError(
                    f"Recording schema is missing DOFs for {entry.name}: {missing}"
                )

        recorded_deformable = physx.get("deformable_name")
        if self.deformable is not None and (
            str(recorded_deformable) != self.deformable.name
        ):
            raise ValueError(
                "Recording deformable does not match runtime: "
                f"recorded={recorded_deformable!r}, "
                f"runtime={self.deformable.name!r}"
            )
        return physx

    @staticmethod
    def _clone(value: Any) -> Any:
        clone = getattr(value, "clone", None)
        if callable(clone):
            return clone()
        return np.array(value, copy=True)

    def _array_like(self, template: Any, values: Any) -> Any:
        # Isaac Lab 3 exposes several PhysX buffers as ``ProxyArray`` objects.
        # The proxy itself advertises a Warp vector dtype (for example
        # ``wp.vec13f``), while its ``.torch`` view has the scalar dtype/device
        # expected by ``torch.as_tensor``.  Passing the outer dtype to Torch
        # raises at runtime, so always choose the concrete backend first.
        if hasattr(template, "torch"):
            template = template.torch
        elif hasattr(template, "warp"):
            template = template.warp
        torch_module = getattr(self.env, "_torch", None)
        if (
            torch_module is not None
            and hasattr(template, "detach")
            and hasattr(template, "dtype")
            and hasattr(template, "device")
        ):
            return torch_module.as_tensor(
                values,
                dtype=template.dtype,
                device=template.device,
            )
        if (
            hasattr(template, "dtype")
            and hasattr(template, "device")
            and not isinstance(template, np.ndarray)
        ):
            import warp as wp

            return wp.array(
                np.asarray(values, dtype=np.float32),
                dtype=template.dtype,
                device=template.device,
            )
        return np.asarray(values, dtype=_numpy(template).dtype)

    @staticmethod
    def _set_indexed(
        setter: Any,
        values: Any,
        indices: Any | None,
    ) -> None:
        if indices is None:
            setter(values)
        else:
            setter(values, indices)

    @staticmethod
    def _require_snapshot_array(
        snapshot: Mapping[str, Any],
        name: str,
        *,
        ndim: int,
    ) -> np.ndarray:
        value = snapshot.get(name)
        direct_aliases = {
            "physx_robot_joint_position": "physx_robot_joint_q",
            "physx_robot_joint_velocity": "physx_robot_joint_qd",
        }
        if value is None and name in direct_aliases:
            value = snapshot.get(direct_aliases[name])

        split_state_aliases = {
            "physx_task_rigid_body_state": (
                "physx_task_rigid_pose",
                "physx_task_rigid_velocity",
            ),
            "physx_task_articulation_root_state": (
                "physx_task_articulation_root_pose",
                "physx_task_articulation_root_velocity",
            ),
            "physx_robot_root_state": (
                "physx_robot_root_pose",
                "physx_robot_root_velocity",
            ),
        }
        if value is None and name in split_state_aliases:
            pose_name, velocity_name = split_state_aliases[name]
            pose = snapshot.get(pose_name)
            velocity = snapshot.get(velocity_name)
            if pose is not None and velocity is not None:
                value = np.concatenate(
                    (np.asarray(pose), np.asarray(velocity)),
                    axis=-1,
                )

        valid_references = {
            "physx_robot_joint_valid": "physx_robot_joint_q",
            "physx_robot_joint_target_valid": "physx_robot_joint_target",
            "physx_task_rigid_body_valid": "physx_task_rigid_pose",
            "physx_task_articulation_root_valid": (
                "physx_task_articulation_root_pose"
            ),
            "physx_task_articulation_dof_valid": (
                "physx_task_articulation_dof_position"
            ),
        }
        if value is None and name in valid_references:
            reference = snapshot.get(valid_references[name])
            if reference is not None:
                reference_shape = np.asarray(reference).shape
                if len(reference_shape) == ndim:
                    valid_shape = reference_shape
                elif (
                    name.startswith("physx_robot_joint_")
                    and ndim == 2
                    and len(reference_shape) == 1
                    and reference_shape[0] % 27 == 0
                ):
                    valid_shape = (reference_shape[0] // 27, 27)
                elif len(reference_shape) == ndim + 1:
                    valid_shape = reference_shape[:-1]
                else:
                    valid_shape = ()
                if valid_shape:
                    value = np.ones(valid_shape, dtype=bool)

        if value is None:
            raise ValueError(f"PhysX replay snapshot is missing {name}")
        value = np.asarray(value)
        if (
            value.ndim == 1
            and ndim == 2
            and name.startswith("physx_robot_joint_")
            and value.size % 27 == 0
        ):
            value = value.reshape(-1, 27)
        if value.ndim != ndim:
            raise ValueError(
                f"PhysX replay field {name} has shape {value.shape}; "
                f"expected {ndim} dimensions"
            )
        return value

    def _restore_controller_and_rng_fields(
        self,
        snapshot: Mapping[str, Any],
    ) -> None:
        """Restore action-controller memory at the pre-action boundary."""

        tcp = snapshot.get("physx_command_tcp_pose")
        hand = snapshot.get("physx_command_hand")
        hand_valid = snapshot.get("physx_command_hand_valid")
        pose_commanded = snapshot.get("physx_pose_commanded")
        filter_state = snapshot.get("physx_pose_filter_state")
        filter_valid = snapshot.get("physx_pose_filter_valid")

        if tcp is not None:
            tcp = np.asarray(tcp, dtype=np.float64)
        if hand is not None:
            hand = np.asarray(hand, dtype=np.float64)
        if hand_valid is not None:
            hand_valid = np.asarray(hand_valid, dtype=bool)
        if pose_commanded is not None:
            pose_commanded = np.asarray(pose_commanded, dtype=bool).reshape(-1)
        if filter_state is not None:
            filter_state = np.asarray(filter_state, dtype=np.float64)
        if filter_valid is not None:
            filter_valid = np.asarray(filter_valid, dtype=bool).reshape(-1)

        filters = getattr(self.env, "_arm_pose_filters", None)
        if not isinstance(filters, dict):
            filters = {"right": None, "left": None}
            setattr(self.env, "_arm_pose_filters", filters)

        for side_index, side in enumerate(ROBOT_SIDES):
            if (
                tcp is not None
                and tcp.ndim == 2
                and side_index < tcp.shape[0]
                and tcp.shape[1] >= 7
                and np.all(np.isfinite(tcp[side_index, :7]))
            ):
                setattr(
                    self.env,
                    f"_{side}_tcp_pose",
                    tcp[side_index, :7].copy(),
                )

            if (
                hand is not None
                and hand.ndim == 2
                and side_index < hand.shape[0]
            ):
                current_hand = getattr(
                    self.env,
                    f"_{side}_hand",
                    np.zeros(hand.shape[1], dtype=np.float64),
                )
                current = np.asarray(
                    current_hand,
                    dtype=np.float64,
                ).copy()
                count = min(current.size, hand.shape[1])
                valid = np.isfinite(hand[side_index, :count])
                if (
                    hand_valid is not None
                    and hand_valid.ndim == 2
                    and side_index < hand_valid.shape[0]
                ):
                    valid &= hand_valid[side_index, :count]
                current[:count][valid] = hand[side_index, :count][valid]
                setattr(self.env, f"_{side}_hand", current)

            if pose_commanded is not None and side_index < pose_commanded.size:
                setattr(
                    self.env,
                    f"_{side}_pose_commanded",
                    bool(pose_commanded[side_index]),
                )

            filters[side] = None
            if (
                filter_state is not None
                and filter_state.ndim == 2
                and side_index < filter_state.shape[0]
                and filter_state.shape[1] >= 14
                and np.all(np.isfinite(filter_state[side_index, :14]))
                and (
                    filter_valid is None
                    or (
                        side_index < filter_valid.size
                        and bool(filter_valid[side_index])
                    )
                )
            ):
                state = filter_state[side_index]
                filters[side] = {
                    "raw_pos": state[0:3].copy(),
                    "raw_quat": state[3:7].copy(),
                    "pos": state[7:10].copy(),
                    "quat": state[10:14].copy(),
                }

        rng_fields = (
            "physx_rng_mt19937_keys",
            "physx_rng_mt19937_position",
            "physx_rng_mt19937_has_gauss",
            "physx_rng_mt19937_cached_gaussian",
        )
        if all(snapshot.get(name) is not None for name in rng_fields):
            random_state = getattr(self.env, "_random", None)
            set_state = getattr(random_state, "set_state", None)
            keys = np.asarray(
                snapshot["physx_rng_mt19937_keys"],
                dtype=np.uint32,
            ).reshape(-1)
            position = int(
                np.asarray(
                    snapshot["physx_rng_mt19937_position"],
                ).reshape(-1)[0]
            )
            has_gauss = int(
                np.asarray(
                    snapshot["physx_rng_mt19937_has_gauss"],
                ).reshape(-1)[0]
            )
            cached_gaussian = float(
                np.asarray(
                    snapshot["physx_rng_mt19937_cached_gaussian"],
                ).reshape(-1)[0]
            )
            if callable(set_state) and keys.shape == (624,):
                set_state(
                    (
                        "MT19937",
                        keys.copy(),
                        position,
                        has_gauss,
                        cached_gaussian,
                    )
                )
            seed = snapshot.get("physx_rng_seed")
            if seed is not None:
                setattr(
                    self.env,
                    "_seed",
                    int(np.asarray(seed).reshape(-1)[0]),
                )

    def _restore_robot_fields(
        self,
        snapshot: Mapping[str, Any],
        physx: Mapping[str, Any],
    ) -> int:
        positions = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_position",
            ndim=2,
        )
        velocities = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_velocity",
            ndim=2,
        )
        valid = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_valid",
            ndim=2,
        ).astype(bool)
        if "physx_robot_joint_target" in snapshot:
            targets = self._require_snapshot_array(
                snapshot,
                "physx_robot_joint_target",
                ndim=2,
            )
            target_valid = self._require_snapshot_array(
                snapshot,
                "physx_robot_joint_target_valid",
                ndim=2,
            ).astype(bool)
        else:
            targets = positions
            target_valid = valid
        sides = tuple(str(side) for side in physx["robot_sides"])
        joint_names = physx["robot_joint_names"]
        restored = 0

        for entry in self.robots:
            if entry.robot is None:
                continue
            side_index = sides.index(entry.side)
            recorded_names = tuple(
                str(name) for name in joint_names[entry.side]
            )
            recorded_index = {
                name: index for index, name in enumerate(recorded_names)
            }
            robot = entry.robot
            position_template = robot.data.joint_pos
            velocity_template = robot.data.joint_vel
            position_values = _numpy(position_template).copy()
            velocity_values = _numpy(velocity_template).copy()
            position_rows = position_values.reshape(-1, position_values.shape[-1])
            velocity_rows = velocity_values.reshape(-1, velocity_values.shape[-1])

            for runtime_index, name in enumerate(entry.joint_names):
                source_index = recorded_index[name]
                if (
                    source_index >= positions.shape[1]
                    or side_index >= positions.shape[0]
                    or not valid[side_index, source_index]
                ):
                    continue
                position_rows[0, runtime_index] = positions[
                    side_index, source_index
                ]
                velocity_rows[0, runtime_index] = velocities[
                    side_index, source_index
                ]
                restored += 1

            joint_position = self._array_like(
                position_template,
                position_values,
            )
            joint_velocity = self._array_like(
                velocity_template,
                velocity_values,
            )
            robot.write_joint_state_to_sim(joint_position, joint_velocity)

        self._apply_robot_target_fields(
            targets,
            target_valid,
            physx,
        )
        self._restore_controller_and_rng_fields(snapshot)
        return restored

    def _apply_robot_target_fields(
        self,
        targets: np.ndarray,
        target_valid: np.ndarray,
        physx: Mapping[str, Any],
    ) -> int:
        sides = tuple(str(side) for side in physx["robot_sides"])
        joint_names = physx["robot_joint_names"]
        applied = 0
        for entry in self.robots:
            if entry.robot is None:
                continue
            side_index = sides.index(entry.side)
            recorded_names = tuple(
                str(name) for name in joint_names[entry.side]
            )
            recorded_index = {
                name: index for index, name in enumerate(recorded_names)
            }
            robot = entry.robot
            target_template = getattr(self.env, "_joint_targets", {}).get(
                entry.side,
                robot.data.joint_pos,
            )
            target_values = _numpy(target_template).copy()
            target_rows = target_values.reshape(-1, target_values.shape[-1])
            for runtime_index, name in enumerate(entry.joint_names):
                source_index = recorded_index[name]
                if (
                    source_index >= targets.shape[1]
                    or side_index >= targets.shape[0]
                    or not target_valid[side_index, source_index]
                ):
                    continue
                target_rows[0, runtime_index] = targets[
                    side_index, source_index
                ]
                applied += 1
            joint_target = self._array_like(target_template, target_values)
            robot.set_joint_position_target(joint_target)
            joint_targets = getattr(self.env, "_joint_targets", None)
            if isinstance(joint_targets, dict):
                joint_targets[entry.side] = joint_target
            write_data = getattr(robot, "write_data_to_sim", None)
            if callable(write_data):
                write_data()
        return applied

    def apply_robot_joint_targets(
        self,
        snapshot: Mapping[str, Any],
        recorded_schema: Mapping[str, Any],
    ) -> int:
        """Apply only recorded controller targets without teleporting state."""

        physx = self.validate_restore_schema(recorded_schema)
        targets = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_target",
            ndim=2,
        )
        target_valid = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_target_valid",
            ndim=2,
        ).astype(bool)
        return self._apply_robot_target_fields(
            targets,
            target_valid,
            physx,
        )

    def restore_robot_state_only(
        self,
        snapshot: Mapping[str, Any],
        recorded_schema: Mapping[str, Any],
    ) -> int:
        """Restore robot joints and drives without writing any task object."""

        physx = self.validate_restore_schema(recorded_schema)
        return self._restore_robot_fields(snapshot, physx)

    @staticmethod
    def _quaternion_error_rotvec_wxyz(
        target: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        """Return the shortest world-frame rotation from current to target."""

        target = np.asarray(target, dtype=np.float64).reshape(4)
        current = np.asarray(current, dtype=np.float64).reshape(4)
        target_norm = float(np.linalg.norm(target))
        current_norm = float(np.linalg.norm(current))
        if target_norm <= 1.0e-12 or current_norm <= 1.0e-12:
            return np.zeros(3, dtype=np.float64)
        tw, tx, ty, tz = target / target_norm
        cw, cx, cy, cz = current / current_norm
        # q_error = q_target * conjugate(q_current)
        error = np.asarray(
            [
                tw * cw + tx * cx + ty * cy + tz * cz,
                -tw * cx + tx * cw - ty * cz + tz * cy,
                -tw * cy + tx * cz + ty * cw - tz * cx,
                -tw * cz - tx * cy + ty * cx + tz * cw,
            ],
            dtype=np.float64,
        )
        if error[0] < 0.0:
            error *= -1.0
        vector_norm = float(np.linalg.norm(error[1:]))
        if vector_norm <= 1.0e-12:
            return 2.0 * error[1:]
        angle = 2.0 * np.arctan2(vector_norm, max(error[0], 0.0))
        return error[1:] * (angle / vector_norm)

    @staticmethod
    def _limit_vector_norm(values: np.ndarray, limit: float) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        norm = float(np.linalg.norm(result))
        if limit > 0.0 and norm > limit:
            result = result * (limit / norm)
        return result

    @staticmethod
    def _advance_quaternion_wxyz(
        quaternion: np.ndarray,
        angular_velocity_world: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
        angular_velocity_world = np.asarray(
            angular_velocity_world,
            dtype=np.float64,
        ).reshape(3)
        speed = float(np.linalg.norm(angular_velocity_world))
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1.0e-12:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quaternion = quaternion / quaternion_norm
        if speed <= 1.0e-12 or dt <= 0.0:
            return quaternion
        half_angle = 0.5 * speed * float(dt)
        delta = np.concatenate(
            (
                np.asarray([np.cos(half_angle)], dtype=np.float64),
                angular_velocity_world
                * (np.sin(half_angle) / speed),
            )
        )
        dw, dx, dy, dz = delta
        qw, qx, qy, qz = quaternion
        predicted = np.asarray(
            [
                dw * qw - dx * qx - dy * qy - dz * qz,
                dw * qx + dx * qw + dy * qz - dz * qy,
                dw * qy - dx * qz + dy * qw + dz * qx,
                dw * qz + dx * qy - dy * qx + dz * qw,
            ],
            dtype=np.float64,
        )
        return predicted / max(float(np.linalg.norm(predicted)), 1.0e-12)

    def apply_task_tracking_wrenches(
        self,
        target_snapshot: Mapping[str, Any],
        recorded_schema: Mapping[str, Any],
        *,
        linear_position_gain: float,
        linear_velocity_gain: float,
        angular_position_gain: float,
        angular_velocity_gain: float,
        max_linear_acceleration: float,
        max_angular_acceleration: float,
        physics_dt: float,
    ) -> dict[str, float | int]:
        """Guide recorded rigid bodies with forces, never with state writes.

        This is deliberately a separate, non-certifying replay path.  The
        recorded object state is a feedback reference, while PhysX still
        integrates every body and resolves every contact.
        """

        if not self.rigid_bodies:
            return {
                "body_count": 0,
                "max_position_error": 0.0,
                "max_angle_error": 0.0,
            }
        physx = self.validate_restore_schema(recorded_schema)
        states = self._require_snapshot_array(
            target_snapshot,
            "physx_task_rigid_body_state",
            ndim=2,
        )
        valid = self._require_snapshot_array(
            target_snapshot,
            "physx_task_rigid_body_valid",
            ndim=1,
        ).astype(bool)
        recorded_names = tuple(
            str(name) for name in physx["rigid_body_names"]
        )
        recorded_index = {
            name: index for index, name in enumerate(recorded_names)
        }
        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )
        applied = 0
        max_position_error = 0.0
        max_angle_error = 0.0
        for entry in self.rigid_bodies:
            if (
                entry.source != "physx_tensor_view"
                or entry.name in kinematic_paths
            ):
                continue
            source_index = recorded_index[entry.name]
            if source_index >= states.shape[0] or not valid[source_index]:
                continue
            target = np.asarray(states[source_index], dtype=np.float64)
            if target.shape != (13,) or not np.all(np.isfinite(target)):
                continue
            try:
                transform_template = entry.handle.get_transforms()
                velocity_template = entry.handle.get_velocities()
                current = _physx_transform_state(
                    transform_template,
                    velocity_template,
                )
                position_error = target[:3] - current[:3]
                rotation_error = self._quaternion_error_rotvec_wxyz(
                    target[3:7],
                    current[3:7],
                )
                predicted_position = (
                    current[:3] + target[7:10] * float(physics_dt)
                )
                predicted_orientation = self._advance_quaternion_wxyz(
                    current[3:7],
                    target[10:13],
                    float(physics_dt),
                )
                position_residual = target[:3] - predicted_position
                rotation_residual = self._quaternion_error_rotvec_wxyz(
                    target[3:7],
                    predicted_orientation,
                )
                linear_acceleration = (
                    float(linear_position_gain) * position_residual
                    + float(linear_velocity_gain) * (target[7:10] - current[7:10])
                )
                angular_acceleration = (
                    float(angular_position_gain) * rotation_residual
                    + float(angular_velocity_gain) * (target[10:13] - current[10:13])
                )
                linear_acceleration = self._limit_vector_norm(
                    linear_acceleration,
                    float(max_linear_acceleration),
                )
                angular_acceleration = self._limit_vector_norm(
                    angular_acceleration,
                    float(max_angular_acceleration),
                )

                masses = _numpy(entry.handle.get_masses()).reshape(-1)
                mass = float(masses[0])
                if not np.isfinite(mass) or mass <= 0.0:
                    continue
                inertia_scale = mass * 1.0e-3
                get_inertias = getattr(entry.handle, "get_inertias", None)
                if callable(get_inertias):
                    inertias = _numpy(get_inertias()).reshape(-1, 9)[0]
                    diagonal_mean = float(
                        np.mean(np.abs(inertias[[0, 4, 8]]))
                    )
                    if np.isfinite(diagonal_mean) and diagonal_mean > 1.0e-12:
                        inertia_scale = diagonal_mean

                row_count = int(_numpy(velocity_template).reshape(-1, 6).shape[0])
                forces = np.zeros((row_count, 3), dtype=np.float64)
                torques = np.zeros((row_count, 3), dtype=np.float64)
                forces[0] = mass * linear_acceleration
                torques[0] = inertia_scale * angular_acceleration
                entry.handle.apply_forces_and_torques_at_position(
                    self._array_like(velocity_template, forces),
                    self._array_like(velocity_template, torques),
                    None,
                    entry.indices,
                    True,
                )
                applied += 1
                max_position_error = max(
                    max_position_error,
                    float(np.linalg.norm(position_error)),
                )
                max_angle_error = max(
                    max_angle_error,
                    float(np.linalg.norm(rotation_error)),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not apply guided replay wrench to {entry.name}: {exc}"
                ) from exc
        return {
            "body_count": applied,
            "max_position_error": max_position_error,
            "max_angle_error": max_angle_error,
        }

    def _restore_rigid_body_fields(
        self,
        snapshot: Mapping[str, Any],
        physx: Mapping[str, Any],
    ) -> int:
        if not self.rigid_bodies:
            return 0
        states = self._require_snapshot_array(
            snapshot,
            "physx_task_rigid_body_state",
            ndim=2,
        )
        valid = self._require_snapshot_array(
            snapshot,
            "physx_task_rigid_body_valid",
            ndim=1,
        ).astype(bool)
        recorded_names = tuple(
            str(name) for name in physx["rigid_body_names"]
        )
        recorded_index = {
            name: index for index, name in enumerate(recorded_names)
        }
        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )
        restored = 0
        for entry in self.rigid_bodies:
            source_index = recorded_index[entry.name]
            if source_index >= states.shape[0] or not valid[source_index]:
                continue
            state = np.asarray(states[source_index], dtype=np.float64)
            if state.shape != (13,) or not np.all(np.isfinite(state)):
                continue
            if entry.source == "physx_tensor_view":
                transform_template = entry.handle.get_transforms()
                velocity_template = entry.handle.get_velocities()
                transforms = _numpy(transform_template).copy()
                velocities = _numpy(velocity_template).copy()
                transform_row = transforms.reshape(-1, transforms.shape[-1])[0]
                velocity_row = velocities.reshape(-1, velocities.shape[-1])[0]
                transform_row[:3] = state[:3]
                transform_row[3:7] = state[[4, 5, 6, 3]]
                velocity_row[:6] = state[7:13]
                self._set_indexed(
                    entry.handle.set_transforms,
                    self._array_like(transform_template, transforms),
                    entry.indices,
                )
                if entry.name not in kinematic_paths:
                    self._set_indexed(
                        entry.handle.set_velocities,
                        self._array_like(velocity_template, velocities),
                        entry.indices,
                    )
            else:
                root_template = entry.handle.data.root_state_w
                roots = _numpy(root_template).copy()
                roots.reshape(-1, roots.shape[-1])[0, :13] = state
                entry.handle.write_root_state_to_sim(
                    self._array_like(root_template, roots)
                )
                write_data = getattr(entry.handle, "write_data_to_sim", None)
                if callable(write_data):
                    write_data()
            restored += 1
        return restored

    def _restore_articulation_fields(
        self,
        snapshot: Mapping[str, Any],
        physx: Mapping[str, Any],
    ) -> tuple[int, int]:
        if not self.articulations:
            return 0, 0
        roots = self._require_snapshot_array(
            snapshot,
            "physx_task_articulation_root_state",
            ndim=2,
        )
        root_valid = self._require_snapshot_array(
            snapshot,
            "physx_task_articulation_root_valid",
            ndim=1,
        ).astype(bool)
        recorded_names = tuple(
            str(name) for name in physx["articulation_names"]
        )
        recorded_index = {
            name: index for index, name in enumerate(recorded_names)
        }
        recorded_dofs = physx["articulation_dof_names"]
        if any(entry.dof_count > 0 for entry in self.articulations):
            dof_positions = self._require_snapshot_array(
                snapshot,
                "physx_task_articulation_dof_position",
                ndim=2,
            )
            dof_velocities = self._require_snapshot_array(
                snapshot,
                "physx_task_articulation_dof_velocity",
                ndim=2,
            )
            dof_valid = self._require_snapshot_array(
                snapshot,
                "physx_task_articulation_dof_valid",
                ndim=2,
            ).astype(bool)
        restored_roots = 0
        restored_dofs = 0

        for entry in self.articulations:
            source_index = recorded_index[entry.name]
            if (
                self._articulation_root_is_writable(entry.name)
                and source_index < roots.shape[0]
                and root_valid[source_index]
            ):
                state = np.asarray(roots[source_index], dtype=np.float64)
                if state.shape == (13,) and np.all(np.isfinite(state)):
                    transform_template = entry.view.get_root_transforms()
                    velocity_template = entry.view.get_root_velocities()
                    transforms = _numpy(transform_template).copy()
                    velocities = _numpy(velocity_template).copy()
                    transform_row = transforms.reshape(
                        -1, transforms.shape[-1]
                    )[0]
                    velocity_row = velocities.reshape(
                        -1, velocities.shape[-1]
                    )[0]
                    transform_row[:3] = state[:3]
                    transform_row[3:7] = state[[4, 5, 6, 3]]
                    velocity_row[:6] = state[7:13]
                    self._set_indexed(
                        entry.view.set_root_transforms,
                        self._array_like(transform_template, transforms),
                        entry.indices,
                    )
                    self._set_indexed(
                        entry.view.set_root_velocities,
                        self._array_like(velocity_template, velocities),
                        entry.indices,
                    )
                    restored_roots += 1

            if (
                entry.dof_count <= 0
                or not self._articulation_dofs_are_writable(entry.name)
            ):
                continue
            position_template = entry.view.get_dof_positions()
            velocity_template = entry.view.get_dof_velocities()
            position_values = _numpy(position_template).copy()
            velocity_values = _numpy(velocity_template).copy()
            position_row = position_values.reshape(
                -1, position_values.shape[-1]
            )[0]
            velocity_row = velocity_values.reshape(
                -1, velocity_values.shape[-1]
            )[0]
            source_dofs = {
                str(name): index
                for index, name in enumerate(recorded_dofs[entry.name])
            }
            any_valid = False
            for runtime_index, name in enumerate(entry.dof_names):
                source_dof_index = source_dofs[name]
                if (
                    source_index >= dof_valid.shape[0]
                    or source_dof_index >= dof_valid.shape[1]
                    or not dof_valid[source_index, source_dof_index]
                ):
                    continue
                position_row[runtime_index] = dof_positions[
                    source_index, source_dof_index
                ]
                velocity_row[runtime_index] = dof_velocities[
                    source_index, source_dof_index
                ]
                restored_dofs += 1
                any_valid = True
            if any_valid:
                self._set_indexed(
                    entry.view.set_dof_positions,
                    self._array_like(position_template, position_values),
                    entry.indices,
                )
                self._set_indexed(
                    entry.view.set_dof_velocities,
                    self._array_like(velocity_template, velocity_values),
                    entry.indices,
                )
        return restored_roots, restored_dofs

    def _restore_deformable_fields(
        self,
        snapshot: Mapping[str, Any],
    ) -> int:
        if self.deformable is None:
            return 0
        valid = self._require_snapshot_array(
            snapshot,
            "physx_deformable_valid",
            ndim=1,
        ).astype(bool)
        if not valid.size or not valid[0]:
            return 0
        positions = self._require_snapshot_array(
            snapshot,
            "physx_deformable_position",
            ndim=2,
        )
        velocities = self._require_snapshot_array(
            snapshot,
            "physx_deformable_velocity",
            ndim=2,
        )
        if positions.shape != self.deformable_shape:
            raise ValueError(
                "PhysX deformable replay shape changed: "
                f"recorded={positions.shape}, runtime={self.deformable_shape}"
            )
        if self.deformable.name == "surface":
            view = getattr(self.env, "_e_garment_view", None)
            indices = getattr(self.env, "_e_garment_indices", None)
        else:
            view = getattr(self.env, "_e02_rope_view", None)
            indices = getattr(self.env, "_e02_rope_indices", None)
        if view is None:
            raise RuntimeError(
                f"PhysX deformable view for {self.deformable.name!r} is unavailable"
            )
        position_template = view.get_simulation_nodal_positions()
        velocity_template = view.get_simulation_nodal_velocities()
        position_values = _numpy(position_template).copy()
        velocity_values = _numpy(velocity_template).copy()
        position_values.reshape(-1, *positions.shape)[0] = positions
        velocity_values.reshape(-1, *velocities.shape)[0] = velocities
        self._set_indexed(
            view.set_simulation_nodal_positions,
            self._array_like(position_template, position_values),
            indices,
        )
        self._set_indexed(
            view.set_simulation_nodal_velocities,
            self._array_like(velocity_template, velocity_values),
            indices,
        )
        setattr(
            self.env,
            self.deformable.position_attribute,
            np.asarray(positions, dtype=np.float64).copy(),
        )
        setattr(
            self.env,
            self.deformable.velocity_attribute,
            np.asarray(velocities, dtype=np.float64).copy(),
        )
        return int(positions.shape[0])

    def _sync_after_restore(self) -> None:
        sim = getattr(self.env, "_sim", None)
        if sim is not None:
            sim.forward()
        scene = getattr(self.env, "_scene", None)
        if scene is not None:
            try:
                scene.update(0.0)
            except Exception:
                pass
        for entry in self.robots:
            if entry.robot is None:
                continue
            try:
                entry.robot.update(0.0)
            except Exception:
                pass
        refresh = getattr(self.env, "_refresh_object_poses", None)
        if callable(refresh):
            refresh()

    def restore(
        self,
        snapshot: Mapping[str, Any],
        recorded_schema: Mapping[str, Any],
        *,
        sync: bool = True,
    ) -> dict[str, int]:
        """Write one named PhysX snapshot back into the live simulator."""

        physx = self.validate_restore_schema(recorded_schema)
        restored_articulation_roots, restored_articulation_dofs = (
            self._restore_articulation_fields(snapshot, physx)
        )
        result = {
            "robot_joints": self._restore_robot_fields(snapshot, physx),
            "rigid_bodies": self._restore_rigid_body_fields(snapshot, physx),
            "articulation_roots": restored_articulation_roots,
            "articulation_dofs": restored_articulation_dofs,
            "deformable_nodes": self._restore_deformable_fields(snapshot),
        }
        control_step = np.asarray(
            snapshot.get("physx_control_step", np.zeros(0))
        ).reshape(-1)
        if control_step.size and np.isfinite(control_step[0]):
            self.env.env_step = int(round(float(control_step[0])))
        if sync:
            self._sync_after_restore()
        return result

    def put_task_bodies_to_sleep(self) -> dict[str, Any]:
        """Put reset task bodies into PhysX's wakeable sleeping state."""
        import omni.physx
        import omni.usd
        from pxr import PhysicsSchemaTools, Sdf

        simulation = omni.physx.get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        slept: list[str] = []
        skipped: list[str] = []
        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )
        articulation_paths = tuple(
            str(entry.name) for entry in self.articulations
        )
        for raw_name in dict.fromkeys(entry.name for entry in self.rigid_bodies):
            name = str(raw_name)
            belongs_to_articulation = any(
                name == root_path or name.startswith(f"{root_path}/")
                for root_path in articulation_paths
            )
            if (
                not name.startswith("/")
                or name in kinematic_paths
                or belongs_to_articulation
            ):
                skipped.append(name)
                continue
            path = Sdf.Path(name)
            simulation.put_to_sleep(
                stage_id,
                PhysicsSchemaTools.sdfPathToInt(path),
            )
            slept.append(name)
        return {
            "rigid_bodies": len(slept),
            "body_paths": slept,
            "skipped_non_dynamic_or_non_path_names": skipped,
        }

    def _sample_robot_fields(self) -> dict[str, np.ndarray]:
        batched = self._sample_batched_robot_fields()
        if batched is not None:
            return batched

        shape = (len(ROBOT_SIDES), self.max_robot_joints)
        position = np.zeros(shape, dtype=np.float64)
        velocity = np.zeros(shape, dtype=np.float64)
        target = np.zeros(shape, dtype=np.float64)
        effort = np.zeros(shape, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.uint8)
        target_valid = np.zeros(shape, dtype=np.uint8)
        effort_valid = np.zeros(shape, dtype=np.uint8)
        root_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        root_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        tcp_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        tcp_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        joint_targets = getattr(self.env, "_joint_targets", {})

        for side_index, entry in enumerate(self.robots):
            robot = entry.robot
            if robot is None:
                continue
            try:
                actual_position = _first_row(robot.data.joint_pos).reshape(-1)
                actual_velocity = _first_row(robot.data.joint_vel).reshape(-1)
                count = min(
                    entry.joint_count,
                    actual_position.size,
                    actual_velocity.size,
                    self.max_robot_joints,
                )
                finite = (
                    np.isfinite(actual_position[:count])
                    & np.isfinite(actual_velocity[:count])
                )
                position[side_index, :count][finite] = actual_position[:count][
                    finite
                ]
                velocity[side_index, :count][finite] = actual_velocity[:count][
                    finite
                ]
                valid[side_index, :count] = finite.astype(np.uint8)
            except Exception:
                count = 0

            for source_name in ("applied_torque", "computed_torque"):
                try:
                    effort_values = _first_row(
                        getattr(robot.data, source_name)
                    ).reshape(-1)
                except Exception:
                    continue
                effort_count = min(
                    effort_values.size,
                    self.max_robot_joints,
                )
                finite = np.isfinite(effort_values[:effort_count])
                effort[side_index, :effort_count][finite] = effort_values[
                    :effort_count
                ][finite]
                effort_valid[side_index, :effort_count] = finite.astype(
                    np.uint8
                )
                break

            for source_name in ("root_state_w", "root_link_state_w"):
                try:
                    root = _first_row(
                        getattr(robot.data, source_name)
                    ).reshape(-1)[:13]
                except Exception:
                    continue
                if root.shape == (13,) and np.all(np.isfinite(root)):
                    root_state[side_index] = root
                    root_valid[side_index] = 1
                    break

            if isinstance(joint_targets, Mapping) and entry.side in joint_targets:
                try:
                    target_values = _first_row(
                        joint_targets[entry.side]
                    ).reshape(-1)
                    target_count = min(target_values.size, self.max_robot_joints)
                    finite = np.isfinite(target_values[:target_count])
                    target[side_index, :target_count][finite] = target_values[
                        :target_count
                    ][finite]
                    target_valid[side_index, :target_count] = finite.astype(
                        np.uint8
                    )
                except Exception:
                    pass

            if entry.tcp_body_index is not None:
                try:
                    body_states = _numpy(robot.data.body_state_w)
                    tcp = body_states[0, entry.tcp_body_index, :13]
                    if tcp.shape == (13,) and np.all(np.isfinite(tcp)):
                        tcp_state[side_index] = tcp
                        tcp_valid[side_index] = 1
                except Exception:
                    pass

        return {
            "physx_robot_joint_position": position,
            "physx_robot_joint_velocity": velocity,
            "physx_robot_joint_target": target,
            "physx_robot_joint_target_valid": target_valid,
            "physx_robot_joint_valid": valid,
            "physx_robot_root_state": root_state,
            "physx_robot_root_valid": root_valid,
            "physx_robot_joint_effort": effort,
            "physx_robot_joint_effort_valid": effort_valid,
            "physx_tcp_state": tcp_state,
            "physx_tcp_valid": tcp_valid,
        }

    def _sample_batched_robot_fields(self) -> dict[str, np.ndarray] | None:
        """Read both CUDA robot buffers with one synchronized host transfer.

        Isaac Lab publishes each robot quantity as a separate Torch view.
        Calling ``.cpu()`` on position, velocity, target, effort, root, and TCP
        independently serializes the CUDA stream twelve times per physics
        substep.  Concatenating those small live views on-device preserves the
        exact values while reducing the readback boundary to one operation.
        Non-Torch test/CPU backends keep using the generic sampler above.
        """

        torch = getattr(self.env, "_torch", None)
        if torch is None or not callable(getattr(torch, "cat", None)):
            return None
        joint_targets = getattr(self.env, "_joint_targets", {})
        if not isinstance(joint_targets, Mapping):
            return None

        pieces: list[Any] = []
        segments: dict[tuple[int, str], tuple[int, int]] = {}
        cursor = 0

        def append(
            side_index: int,
            name: str,
            value: Any,
        ) -> bool:
            nonlocal cursor
            if hasattr(value, "torch"):
                value = value.torch
            if not (
                hasattr(value, "detach")
                and callable(getattr(value, "reshape", None))
                and callable(getattr(value, "numel", None))
            ):
                return False
            flat = value.detach().reshape(-1)
            count = int(flat.numel())
            pieces.append(flat)
            segments[(side_index, name)] = (cursor, cursor + count)
            cursor += count
            return True

        for side_index, entry in enumerate(self.robots):
            robot = entry.robot
            if robot is None:
                return None
            try:
                effort = next(
                    getattr(robot.data, name)
                    for name in ("applied_torque", "computed_torque")
                    if getattr(robot.data, name, None) is not None
                )
                root = next(
                    getattr(robot.data, name)
                    for name in ("root_state_w", "root_link_state_w")
                    if getattr(robot.data, name, None) is not None
                )
                target = joint_targets[entry.side]
                values = (
                    ("position", robot.data.joint_pos),
                    ("velocity", robot.data.joint_vel),
                    ("target", target),
                    ("effort", effort),
                    ("root", root),
                )
            except (AttributeError, KeyError, StopIteration):
                return None
            if not all(
                append(side_index, name, value)
                for name, value in values
            ):
                return None
            if entry.tcp_body_index is not None:
                try:
                    tcp = robot.data.body_state_w[
                        0,
                        entry.tcp_body_index,
                        :13,
                    ]
                except Exception:
                    return None
                if not append(side_index, "tcp", tcp):
                    return None

        if not pieces:
            return None
        try:
            packed = (
                torch.cat(pieces)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        except Exception:
            return None

        shape = (len(ROBOT_SIDES), self.max_robot_joints)
        position = np.zeros(shape, dtype=np.float64)
        velocity = np.zeros(shape, dtype=np.float64)
        target = np.zeros(shape, dtype=np.float64)
        effort = np.zeros(shape, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.uint8)
        target_valid = np.zeros(shape, dtype=np.uint8)
        effort_valid = np.zeros(shape, dtype=np.uint8)
        root_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        root_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        tcp_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        tcp_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)

        def values(side_index: int, name: str) -> np.ndarray:
            start, stop = segments[(side_index, name)]
            return packed[start:stop]

        for side_index, entry in enumerate(self.robots):
            actual_position = values(side_index, "position")
            actual_velocity = values(side_index, "velocity")
            count = min(
                entry.joint_count,
                actual_position.size,
                actual_velocity.size,
                self.max_robot_joints,
            )
            finite = (
                np.isfinite(actual_position[:count])
                & np.isfinite(actual_velocity[:count])
            )
            position[side_index, :count][finite] = actual_position[:count][finite]
            velocity[side_index, :count][finite] = actual_velocity[:count][finite]
            valid[side_index, :count] = finite.astype(np.uint8)

            target_values = values(side_index, "target")
            count = min(target_values.size, self.max_robot_joints)
            finite = np.isfinite(target_values[:count])
            target[side_index, :count][finite] = target_values[:count][finite]
            target_valid[side_index, :count] = finite.astype(np.uint8)

            effort_values = values(side_index, "effort")
            count = min(effort_values.size, self.max_robot_joints)
            finite = np.isfinite(effort_values[:count])
            effort[side_index, :count][finite] = effort_values[:count][finite]
            effort_valid[side_index, :count] = finite.astype(np.uint8)

            root = values(side_index, "root")[:13]
            if root.shape == (13,) and np.all(np.isfinite(root)):
                root_state[side_index] = root
                root_valid[side_index] = 1
            if (side_index, "tcp") in segments:
                tcp = values(side_index, "tcp")[:13]
                if tcp.shape == (13,) and np.all(np.isfinite(tcp)):
                    tcp_state[side_index] = tcp
                    tcp_valid[side_index] = 1

        return {
            "physx_robot_joint_position": position,
            "physx_robot_joint_velocity": velocity,
            "physx_robot_joint_target": target,
            "physx_robot_joint_target_valid": target_valid,
            "physx_robot_joint_valid": valid,
            "physx_robot_root_state": root_state,
            "physx_robot_root_valid": root_valid,
            "physx_robot_joint_effort": effort,
            "physx_robot_joint_effort_valid": effort_valid,
            "physx_tcp_state": tcp_state,
            "physx_tcp_valid": tcp_valid,
        }

    def _sample_command_fields(self) -> dict[str, np.ndarray]:
        tcp = np.zeros((len(ROBOT_SIDES), 7), dtype=np.float64)
        hand = np.zeros(
            (len(ROBOT_SIDES), self.hand_command_dim),
            dtype=np.float64,
        )
        hand_valid = np.zeros(hand.shape, dtype=np.uint8)
        pose_commanded = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        filter_state = np.zeros((len(ROBOT_SIDES), 14), dtype=np.float64)
        filter_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        filters = getattr(self.env, "_arm_pose_filters", {})

        for side_index, side in enumerate(ROBOT_SIDES):
            try:
                value = np.asarray(
                    getattr(self.env, f"_{side}_tcp_pose"),
                    dtype=np.float64,
                ).reshape(7)
                if np.all(np.isfinite(value)):
                    tcp[side_index] = value
            except Exception:
                pass
            try:
                value = np.asarray(
                    getattr(self.env, f"_{side}_hand"),
                    dtype=np.float64,
                ).reshape(-1)
                count = min(value.size, self.hand_command_dim)
                finite = np.isfinite(value[:count])
                hand[side_index, :count][finite] = value[:count][finite]
                hand_valid[side_index, :count] = finite.astype(np.uint8)
            except Exception:
                pass
            pose_commanded[side_index] = int(
                bool(getattr(self.env, f"_{side}_pose_commanded", False))
            )

            state = filters.get(side) if isinstance(filters, Mapping) else None
            if not isinstance(state, Mapping):
                continue
            components = (
                ("raw_pos", 3),
                ("raw_quat", 4),
                ("pos", 3),
                ("quat", 4),
            )
            values = []
            try:
                for name, width in components:
                    value = np.asarray(
                        state[name],
                        dtype=np.float64,
                    ).reshape(width)
                    if not np.all(np.isfinite(value)):
                        raise ValueError(name)
                    values.append(value)
            except (KeyError, TypeError, ValueError):
                continue
            filter_state[side_index] = np.concatenate(values)
            filter_valid[side_index] = 1

        return {
            "physx_command_tcp_pose": tcp,
            "physx_command_hand": hand,
            "physx_command_hand_valid": hand_valid,
            "physx_pose_commanded": pose_commanded,
            "physx_pose_filter_state": filter_state,
            "physx_pose_filter_valid": filter_valid,
        }

    def _sample_rng_fields(self) -> dict[str, np.ndarray]:
        random_state = getattr(self.env, "_random", None)
        if not callable(getattr(random_state, "get_state", None)):
            raise RuntimeError(
                "PhysX replay certification requires environment._random "
                "to expose NumPy RandomState.get_state()."
            )
        algorithm, keys, position, has_gauss, cached = random_state.get_state()
        keys = np.asarray(keys, dtype=np.uint32)
        if algorithm != "MT19937" or keys.shape != (624,):
            raise RuntimeError(
                "PhysX replay certification requires an MT19937 RandomState."
            )
        return {
            "physx_rng_seed": np.asarray(
                [int(getattr(self.env, "_seed"))],
                dtype=np.int64,
            ),
            "physx_rng_mt19937_keys": keys.copy(),
            "physx_rng_mt19937_position": np.asarray(
                [int(position)],
                dtype=np.int64,
            ),
            "physx_rng_mt19937_has_gauss": np.asarray(
                [int(has_gauss)],
                dtype=np.uint8,
            ),
            "physx_rng_mt19937_cached_gaussian": np.asarray(
                [float(cached)],
                dtype=np.float64,
            ),
        }

    def _sample_rigid_body_fields(self) -> dict[str, np.ndarray]:
        if not self.rigid_bodies:
            return {}
        states = np.zeros((len(self.rigid_bodies), 13), dtype=np.float64)
        valid = np.zeros((len(self.rigid_bodies),), dtype=np.uint8)
        body_is_active = getattr(
            self.env,
            "_physx_task_rigid_body_is_active",
            None,
        )
        batched_rows = dict(self._batched_rigid_body_rows)
        batched_states = None
        if self._batched_rigid_body_view is not None:
            try:
                batched_states = _physx_transform_states(
                    self._batched_rigid_body_view.get_transforms(),
                    self._batched_rigid_body_view.get_velocities(),
                )
            except Exception:
                # Preserve the prior fail-soft validity behavior.  The exact
                # per-body views remain available if a batched read is ever
                # invalidated by an out-of-band stage mutation.
                batched_states = None
        for index, entry in enumerate(self.rigid_bodies):
            # Keep the trace layout and row indices fixed across randomized
            # variants, but do not let a deliberately parked actor contribute
            # to the physical-origin gate or certified replay comparisons.
            if callable(body_is_active) and not bool(body_is_active(entry.name)):
                continue
            try:
                batch_index = batched_rows.get(index)
                if batched_states is not None and batch_index is not None:
                    state = batched_states[batch_index]
                elif entry.source == "physx_tensor_view":
                    state = _physx_transform_state(
                        entry.handle.get_transforms(),
                        entry.handle.get_velocities(),
                    )
                else:
                    state = _isaaclab_state(entry.handle.data.root_state_w)
                if state.shape == (13,) and np.all(np.isfinite(state)):
                    states[index] = state
                    valid[index] = 1
            except Exception:
                continue
        return {
            "physx_task_rigid_body_state": states,
            "physx_task_rigid_body_valid": valid,
        }

    def _sample_articulation_fields(self) -> dict[str, np.ndarray]:
        if not self.articulations:
            return {}
        count = len(self.articulations)
        roots = np.zeros((count, 13), dtype=np.float64)
        root_valid = np.zeros((count,), dtype=np.uint8)
        has_dofs = self.max_articulation_dofs > 0
        if has_dofs:
            dof_shape = (count, self.max_articulation_dofs)
            dof_position = np.zeros(dof_shape, dtype=np.float64)
            dof_velocity = np.zeros(dof_shape, dtype=np.float64)
            dof_valid = np.zeros(dof_shape, dtype=np.uint8)

        for index, entry in enumerate(self.articulations):
            try:
                root = _physx_transform_state(
                    entry.view.get_root_transforms(),
                    entry.view.get_root_velocities(),
                )
                if root.shape == (13,) and np.all(np.isfinite(root)):
                    roots[index] = root
                    root_valid[index] = 1
            except Exception:
                pass
            if not has_dofs:
                continue
            try:
                position = _first_row(
                    entry.view.get_dof_positions()
                ).reshape(-1)
                velocity = _first_row(
                    entry.view.get_dof_velocities()
                ).reshape(-1)
                dof_count = min(
                    entry.dof_count,
                    position.size,
                    velocity.size,
                    self.max_articulation_dofs,
                )
                finite = (
                    np.isfinite(position[:dof_count])
                    & np.isfinite(velocity[:dof_count])
                )
                dof_position[index, :dof_count][finite] = position[:dof_count][
                    finite
                ]
                dof_velocity[index, :dof_count][finite] = velocity[:dof_count][
                    finite
                ]
                dof_valid[index, :dof_count] = finite.astype(np.uint8)
            except Exception:
                pass
        result = {
            "physx_task_articulation_root_state": roots,
            "physx_task_articulation_root_valid": root_valid,
        }
        if has_dofs:
            result.update(
                {
                    "physx_task_articulation_dof_position": dof_position,
                    "physx_task_articulation_dof_velocity": dof_velocity,
                    "physx_task_articulation_dof_valid": dof_valid,
                }
            )
        return result

    @staticmethod
    def _contact_uint32(value: Any) -> np.uint32:
        raw_value = getattr(value, "value", value)
        return np.uint32(int(raw_value) & 0xFFFFFFFF)

    @staticmethod
    def _contact_uint64(value: Any) -> np.uint64:
        return np.uint64(int(value) & 0xFFFFFFFFFFFFFFFF)

    @staticmethod
    def _contact_vector3(value: Any, *, label: str) -> np.ndarray:
        try:
            result = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            result = np.asarray(
                [value.x, value.y, value.z],
                dtype=np.float32,
            )
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"PhysX contact report returned invalid {label}: {result!r}"
            )
        return result

    def _empty_contact_fields(self) -> dict[str, np.ndarray]:
        return {
            "physx_contact_report_valid": np.zeros(1, dtype=np.uint8),
            "physx_contact_capture_source": np.zeros(1, dtype=np.uint8),
            "physx_contact_header_count": np.zeros(1, dtype=np.uint32),
            "physx_contact_header_ids": np.zeros(
                (self.max_contact_headers, 5),
                dtype=np.uint64,
            ),
            "physx_contact_header_meta": np.zeros(
                (self.max_contact_headers, 7),
                dtype=np.uint32,
            ),
            "physx_contact_point_count": np.zeros(1, dtype=np.uint32),
            "physx_contact_point_data": np.zeros(
                (self.max_contact_points, 10),
                dtype=np.float32,
            ),
            "physx_contact_point_face_indices": np.zeros(
                (self.max_contact_points, 2),
                dtype=np.uint32,
            ),
            "physx_contact_point_material_ids": np.zeros(
                (self.max_contact_points, 2),
                dtype=np.uint64,
            ),
            "physx_friction_anchor_count": np.zeros(1, dtype=np.uint32),
            "physx_friction_anchor_data": np.zeros(
                (self.max_friction_anchors, 6),
                dtype=np.float32,
            ),
            "physx_contact_signature_valid": np.zeros(1, dtype=np.uint8),
            "physx_contact_signature_pair_count": np.zeros(
                1,
                dtype=np.uint32,
            ),
            "physx_contact_signature_pair_path_indices": np.zeros(
                (self.max_contact_headers, 4),
                dtype=np.uint32,
            ),
            "physx_contact_signature_pair_counts": np.zeros(
                (self.max_contact_headers, 2),
                dtype=np.uint32,
            ),
            "physx_contact_signature_pair_metrics": np.zeros(
                (self.max_contact_headers, 3),
                dtype=np.float64,
            ),
        }

    def _canonicalize_contact_fields(
        self,
        fields: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Add a process- and ordering-independent contact signature.

        Raw PhysX actor/collider ids and buffer offsets are intentionally kept
        as diagnostic fields.  Certification instead compares this aggregate,
        whose identities are 1-based indices into the lexically sorted USD
        path table and whose rows/endpoints are sorted canonically.

        Invalid reports, incomplete id-to-path resolution, and empty reporter
        coverage leave ``signature_valid=0``.  The certifier rejects such a
        recording instead of treating an all-zero signature as evidence that
        the physical contact path reproduced.
        """

        fields["physx_contact_signature_valid"].fill(0)
        fields["physx_contact_signature_pair_count"].fill(0)
        fields["physx_contact_signature_pair_path_indices"].fill(0)
        fields["physx_contact_signature_pair_counts"].fill(0)
        fields["physx_contact_signature_pair_metrics"].fill(0)

        valid = np.asarray(fields["physx_contact_report_valid"]).reshape(-1)
        source_values = np.asarray(
            fields["physx_contact_capture_source"]
        ).reshape(-1)
        if not valid.size or int(valid[0]) != 1 or not source_values.size:
            return fields
        source = int(source_values[0])
        report_paths = tuple(getattr(self, "_contact_report_paths", ()))
        tensor_sensors = tuple(
            getattr(self, "_tensor_contact_sensor_paths", ())
        )
        tensor_filters = tuple(
            getattr(self, "_tensor_contact_filter_paths", ())
        )
        if source == 1:
            coverage_paths = report_paths
        elif source == 2 and tensor_sensors and tensor_filters:
            coverage_paths = (*tensor_sensors, *tensor_filters)
        else:
            return fields
        if not coverage_paths:
            return fields

        id_to_path: dict[int, str] = {}
        path_to_id: dict[str, int] = {}
        for entry in getattr(self, "_contact_report_path_table", ()):
            if not isinstance(entry, Mapping):
                return fields
            try:
                process_id = int(entry["id"]) & 0xFFFFFFFFFFFFFFFF
                path = str(entry["path"])
            except (KeyError, TypeError, ValueError):
                return fields
            if not path or (
                process_id in id_to_path and id_to_path[process_id] != path
            ) or (path in path_to_id and path_to_id[path] != process_id):
                return fields
            id_to_path[process_id] = path
            path_to_id[path] = process_id
        canonical_paths = tuple(sorted(path_to_id))
        if not canonical_paths or any(
            str(path) not in path_to_id for path in coverage_paths
        ):
            return fields
        path_indices = {
            path: index + 1 for index, path in enumerate(canonical_paths)
        }

        header_count = int(
            np.asarray(fields["physx_contact_header_count"]).reshape(-1)[0]
        )
        point_count = int(
            np.asarray(fields["physx_contact_point_count"]).reshape(-1)[0]
        )
        anchor_count = int(
            np.asarray(fields["physx_friction_anchor_count"]).reshape(-1)[0]
        )
        header_ids = np.asarray(fields["physx_contact_header_ids"])
        header_meta = np.asarray(fields["physx_contact_header_meta"])
        point_data = np.asarray(fields["physx_contact_point_data"])
        anchor_data = np.asarray(fields["physx_friction_anchor_data"])
        if (
            header_count < 0
            or header_count > header_ids.shape[0]
            or header_count > header_meta.shape[0]
            or point_count < 0
            or point_count > point_data.shape[0]
            or anchor_count < 0
            or anchor_count > anchor_data.shape[0]
        ):
            return fields

        # A canonical aggregate is valid only when it accounts for the whole
        # public contact payload exactly once.  PhysX headers contain slices
        # into shared point/anchor buffers; overlapping slices would double
        # count solver output, while gaps would silently omit part of the
        # manifold.  Check both conditions before accepting the signature.
        point_coverage = np.zeros(point_count, dtype=np.uint32)
        anchor_coverage = np.zeros(anchor_count, dtype=np.uint32)

        aggregates: dict[
            tuple[tuple[str, str], tuple[str, str]],
            dict[str, Any],
        ] = {}
        for header_index in range(header_count):
            contact_offset = int(header_meta[header_index, 1])
            contact_rows = int(header_meta[header_index, 2])
            friction_offset = int(header_meta[header_index, 3])
            friction_rows = int(header_meta[header_index, 4])
            if contact_rows <= 0 and friction_rows <= 0:
                continue
            if (
                contact_offset < 0
                or contact_rows < 0
                or contact_offset + contact_rows > point_count
                or friction_offset < 0
                or friction_rows < 0
                or friction_offset + friction_rows > anchor_count
            ):
                return fields
            point_coverage[
                contact_offset : contact_offset + contact_rows
            ] += 1
            anchor_coverage[
                friction_offset : friction_offset + friction_rows
            ] += 1
            contact_payload = point_data[
                contact_offset : contact_offset + contact_rows
            ]
            if source == 2:
                # RigidContactView can include an all-zero padding row inside
                # its declared active slice.  Its public tensor payload is the
                # ten point-data values (position, normal, impulse, and
                # separation); face/material ids are unavailable sentinels for
                # this source.  Ignore only an exact all-zero payload here so
                # raw diagnostics remain untouched and every partially
                # populated/malformed row still fails closed below.
                padding_rows = np.all(contact_payload == 0.0, axis=1)
                contact_payload = contact_payload[~padding_rows]
            try:
                actor0, actor1, collider0, collider1 = (
                    id_to_path[int(value)]
                    for value in header_ids[header_index, :4]
                )
            except (KeyError, TypeError, ValueError):
                return fields
            endpoints = sorted(
                ((actor0, collider0), (actor1, collider1))
            )
            pair = (endpoints[0], endpoints[1])
            if not contact_payload.shape[0] and friction_rows <= 0:
                continue
            aggregate = aggregates.setdefault(
                pair,
                {
                    "contact_count": 0,
                    "friction_count": 0,
                    "penetrations": [],
                    "normal_impulses": [],
                    "friction_impulses": [],
                },
            )
            aggregate["contact_count"] += int(contact_payload.shape[0])
            aggregate["friction_count"] += friction_rows
            for row in contact_payload:
                position = np.asarray(row[:3], dtype=np.float64)
                normal = np.asarray(row[3:6], dtype=np.float64)
                impulse = np.asarray(row[6:9], dtype=np.float64)
                separation = float(row[9])
                normal_norm = _canonical_vector_norm3(normal)
                if (
                    not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(normal))
                    or not np.all(np.isfinite(impulse))
                    or not math.isfinite(separation)
                    or normal_norm <= 1.0e-12
                ):
                    return fields
                aggregate["penetrations"].append(max(0.0, -separation))
                aggregate["normal_impulses"].append(
                    _canonical_normal_impulse3(
                        impulse,
                        normal,
                        normal_norm,
                    )
                )
            for row in anchor_data[
                friction_offset : friction_offset + friction_rows
            ]:
                impulse = np.asarray(row[3:6], dtype=np.float64)
                if not np.all(np.isfinite(impulse)):
                    return fields
                aggregate["friction_impulses"].append(
                    _canonical_vector_norm3(impulse)
                )

        if (
            (point_count and not np.all(point_coverage == 1))
            or (anchor_count and not np.all(anchor_coverage == 1))
        ):
            return fields

        ordered_pairs = sorted(aggregates)
        if len(ordered_pairs) > self.max_contact_headers:
            return fields
        pair_paths = fields["physx_contact_signature_pair_path_indices"]
        pair_counts = fields["physx_contact_signature_pair_counts"]
        pair_metrics = fields["physx_contact_signature_pair_metrics"]
        for row_index, pair in enumerate(ordered_pairs):
            (actor0, collider0), (actor1, collider1) = pair
            aggregate = aggregates[pair]
            pair_paths[row_index] = (
                path_indices[actor0],
                path_indices[collider0],
                path_indices[actor1],
                path_indices[collider1],
            )
            pair_counts[row_index] = (
                aggregate["contact_count"],
                aggregate["friction_count"],
            )
            pair_metrics[row_index] = (
                max(aggregate["penetrations"], default=0.0),
                math.fsum(sorted(aggregate["normal_impulses"])),
                math.fsum(sorted(aggregate["friction_impulses"])),
            )
        fields["physx_contact_signature_pair_count"][0] = len(ordered_pairs)
        fields["physx_contact_signature_valid"][0] = 1
        return fields

    def _require_contact_capacity(
        self,
        *,
        label: str,
        count: int,
        capacity: int,
        environment_variable: str,
    ) -> None:
        if count <= capacity:
            return
        raise RuntimeError(
            f"PhysX produced {count} {label}, exceeding the recording capacity "
            f"of {capacity}. Increase {environment_variable}; refusing to write "
            "a partial contact trace."
        )

    def _contact_fields_from_report(self, report: Any) -> dict[str, np.ndarray]:
        """Encode one public contact-report payload as fixed-shape arrays."""

        result = self._empty_contact_fields()
        if not isinstance(report, (tuple, list)) or len(report) != 3:
            raise RuntimeError(
                "PhysX get_full_contact_report() returned an unexpected payload"
            )

        headers, contact_points, friction_anchors = report
        header_count = len(headers)
        point_count = len(contact_points)
        anchor_count = len(friction_anchors)
        self._require_contact_capacity(
            label="contact headers",
            count=header_count,
            capacity=self.max_contact_headers,
            environment_variable=PHYSX_MAX_CONTACT_HEADERS_ENV,
        )
        self._require_contact_capacity(
            label="contact points",
            count=point_count,
            capacity=self.max_contact_points,
            environment_variable=PHYSX_MAX_CONTACT_POINTS_ENV,
        )
        self._require_contact_capacity(
            label="friction anchors",
            count=anchor_count,
            capacity=self.max_friction_anchors,
            environment_variable=PHYSX_MAX_FRICTION_ANCHORS_ENV,
        )

        result["physx_contact_report_valid"][0] = 1
        result["physx_contact_capture_source"][0] = 1
        result["physx_contact_header_count"][0] = header_count
        result["physx_contact_point_count"][0] = point_count
        result["physx_friction_anchor_count"][0] = anchor_count

        header_ids = result["physx_contact_header_ids"]
        header_meta = result["physx_contact_header_meta"]
        for index, header in enumerate(headers):
            header_ids[index] = (
                self._contact_uint64(header.actor0),
                self._contact_uint64(header.actor1),
                self._contact_uint64(header.collider0),
                self._contact_uint64(header.collider1),
                self._contact_uint64(header.stage_id),
            )
            contact_offset = int(header.contact_data_offset)
            contact_count = int(header.num_contact_data)
            friction_offset = int(header.friction_anchors_offset)
            friction_count = int(header.num_friction_anchors_data)
            if (
                contact_offset < 0
                or contact_count < 0
                or contact_offset + contact_count > point_count
            ):
                raise RuntimeError(
                    "PhysX contact header references contact data outside the "
                    f"reported buffer: offset={contact_offset}, "
                    f"count={contact_count}, buffer={point_count}"
                )
            if (
                friction_offset < 0
                or friction_count < 0
                or friction_offset + friction_count > anchor_count
            ):
                raise RuntimeError(
                    "PhysX contact header references friction anchors outside "
                    f"the reported buffer: offset={friction_offset}, "
                    f"count={friction_count}, buffer={anchor_count}"
                )
            header_meta[index] = (
                self._contact_uint32(header.type),
                self._contact_uint32(contact_offset),
                self._contact_uint32(contact_count),
                self._contact_uint32(friction_offset),
                self._contact_uint32(friction_count),
                self._contact_uint32(header.proto_index0),
                self._contact_uint32(header.proto_index1),
            )

        point_data = result["physx_contact_point_data"]
        face_indices = result["physx_contact_point_face_indices"]
        material_ids = result["physx_contact_point_material_ids"]
        for index, point in enumerate(contact_points):
            point_data[index, :3] = self._contact_vector3(
                point.position,
                label="contact position",
            )
            point_data[index, 3:6] = self._contact_vector3(
                point.normal,
                label="contact normal",
            )
            point_data[index, 6:9] = self._contact_vector3(
                point.impulse,
                label="contact impulse",
            )
            separation = float(point.separation)
            if not np.isfinite(separation):
                raise RuntimeError(
                    "PhysX contact report returned a non-finite separation"
                )
            point_data[index, 9] = separation
            face_indices[index] = (
                self._contact_uint32(point.face_index0),
                self._contact_uint32(point.face_index1),
            )
            material_ids[index] = (
                self._contact_uint64(point.material0),
                self._contact_uint64(point.material1),
            )

        anchor_data = result["physx_friction_anchor_data"]
        for index, anchor in enumerate(friction_anchors):
            anchor_data[index, :3] = self._contact_vector3(
                anchor.position,
                label="friction-anchor position",
            )
            anchor_data[index, 3:6] = self._contact_vector3(
                anchor.impulse,
                label="friction-anchor impulse",
            )
        return result

    def _sample_tensor_contact_fields(self) -> dict[str, np.ndarray] | None:
        """Read detailed contact patches and friction points from tensor views."""

        if not self._tensor_contact_views:
            return None
        from pxr import PhysicsSchemaTools, Sdf

        result = self._empty_contact_fields()
        result["physx_contact_report_valid"][0] = 1
        result["physx_contact_capture_source"][0] = 2
        header_count = 0
        point_count = 0
        anchor_count = 0
        seen_pairs: set[tuple[str, str]] = set()
        stage_id = int(self._contact_stage_id)

        for view, (sensor_paths, filter_paths) in zip(
            self._tensor_contact_views,
            self._tensor_contact_view_paths,
            strict=True,
        ):
            tensor_dt = self._contact_tensor_dt
            normal_forces, points, normals, separations, counts, starts = (
                view.get_contact_data(tensor_dt)
            )
            friction_forces, friction_points, friction_counts, friction_starts = (
                view.get_friction_data(tensor_dt)
            )
            normal_forces = _numpy(normal_forces, dtype=np.float32).reshape(-1)
            points = _numpy(points, dtype=np.float32).reshape(-1, 3)
            normals = _numpy(normals, dtype=np.float32).reshape(-1, 3)
            separations = _numpy(separations, dtype=np.float32).reshape(-1)
            counts = _numpy(counts, dtype=np.uint32).reshape(
                int(view.sensor_count),
                int(view.filter_count),
            )
            starts = _numpy(starts, dtype=np.uint32).reshape(counts.shape)
            friction_forces = _numpy(
                friction_forces,
                dtype=np.float32,
            ).reshape(-1, 3)
            friction_points = _numpy(
                friction_points,
                dtype=np.float32,
            ).reshape(-1, 3)
            friction_counts = _numpy(
                friction_counts,
                dtype=np.uint32,
            ).reshape(counts.shape)
            friction_starts = _numpy(
                friction_starts,
                dtype=np.uint32,
            ).reshape(counts.shape)
            if int(np.sum(counts, dtype=np.uint64)) >= int(
                view.max_contact_data_count
            ):
                raise RuntimeError(
                    "PhysX rigid contact tensor filled its complete contact "
                    f"buffer ({view.max_contact_data_count}); increase "
                    f"{PHYSX_MAX_CONTACT_POINTS_ENV} to prove no contacts "
                    "were truncated"
                )
            if int(np.sum(friction_counts, dtype=np.uint64)) >= int(
                view.max_contact_data_count
            ):
                raise RuntimeError(
                    "PhysX rigid contact tensor filled its complete friction "
                    f"buffer ({view.max_contact_data_count}); increase "
                    f"{PHYSX_MAX_FRICTION_ANCHORS_ENV} to prove no friction "
                    "points were truncated"
                )

            for sensor_index, sensor_path in enumerate(sensor_paths):
                for filter_index, filter_path in enumerate(filter_paths):
                    pair = tuple(sorted((sensor_path, filter_path)))
                    if pair in seen_pairs:
                        continue
                    contact_count = int(counts[sensor_index, filter_index])
                    friction_count = int(
                        friction_counts[sensor_index, filter_index]
                    )
                    if contact_count <= 0 and friction_count <= 0:
                        continue
                    seen_pairs.add(pair)
                    self._require_contact_capacity(
                        label="contact headers",
                        count=header_count + 1,
                        capacity=self.max_contact_headers,
                        environment_variable=PHYSX_MAX_CONTACT_HEADERS_ENV,
                    )
                    self._require_contact_capacity(
                        label="contact points",
                        count=point_count + contact_count,
                        capacity=self.max_contact_points,
                        environment_variable=PHYSX_MAX_CONTACT_POINTS_ENV,
                    )
                    self._require_contact_capacity(
                        label="friction points",
                        count=anchor_count + friction_count,
                        capacity=self.max_friction_anchors,
                        environment_variable=PHYSX_MAX_FRICTION_ANCHORS_ENV,
                    )

                    sensor_id = self._contact_uint64(
                        PhysicsSchemaTools.sdfPathToInt(Sdf.Path(sensor_path))
                    )
                    filter_id = self._contact_uint64(
                        PhysicsSchemaTools.sdfPathToInt(Sdf.Path(filter_path))
                    )
                    result["physx_contact_header_ids"][header_count] = (
                        sensor_id,
                        filter_id,
                        sensor_id,
                        filter_id,
                        self._contact_uint64(stage_id),
                    )
                    result["physx_contact_header_meta"][header_count] = (
                        TENSOR_CONTACT_EVENT_TYPE,
                        self._contact_uint32(point_count),
                        self._contact_uint32(contact_count),
                        self._contact_uint32(anchor_count),
                        self._contact_uint32(friction_count),
                        self._contact_uint32(-1),
                        self._contact_uint32(-1),
                    )

                    source_start = int(starts[sensor_index, filter_index])
                    source_stop = source_start + contact_count
                    if source_stop > points.shape[0]:
                        raise RuntimeError(
                            "PhysX rigid contact tensor returned an invalid "
                            "contact offset/count"
                        )
                    destination = slice(
                        point_count,
                        point_count + contact_count,
                    )
                    source = slice(source_start, source_stop)
                    result["physx_contact_point_data"][destination, :3] = points[
                        source
                    ]
                    result["physx_contact_point_data"][destination, 3:6] = normals[
                        source
                    ]
                    result["physx_contact_point_data"][destination, 6:9] = (
                        normals[source]
                        * normal_forces[source, None]
                        * tensor_dt
                    )
                    result["physx_contact_point_data"][destination, 9] = (
                        separations[source]
                    )
                    result["physx_contact_point_face_indices"][destination] = (
                        np.iinfo(np.uint32).max
                    )

                    friction_start = int(
                        friction_starts[sensor_index, filter_index]
                    )
                    friction_stop = friction_start + friction_count
                    if friction_stop > friction_points.shape[0]:
                        raise RuntimeError(
                            "PhysX rigid contact tensor returned an invalid "
                            "friction offset/count"
                        )
                    friction_destination = slice(
                        anchor_count,
                        anchor_count + friction_count,
                    )
                    friction_source = slice(friction_start, friction_stop)
                    result["physx_friction_anchor_data"][
                        friction_destination,
                        :3,
                    ] = friction_points[friction_source]
                    result["physx_friction_anchor_data"][
                        friction_destination,
                        3:6,
                    ] = friction_forces[friction_source] * tensor_dt
                    point_count += contact_count
                    anchor_count += friction_count
                    header_count += 1

        result["physx_contact_header_count"][0] = header_count
        result["physx_contact_point_count"][0] = point_count
        result["physx_friction_anchor_count"][0] = anchor_count
        return result

    def _on_contact_report_event(
        self,
        contact_headers: Any,
        contact_points: Any,
    ) -> None:
        """Copy contact buffers while PhysX guarantees they are still alive."""

        try:
            interface = self._contact_report_interface
            full_report = interface.get_full_contact_report()
            if not isinstance(full_report, (tuple, list)) or len(full_report) != 3:
                raise RuntimeError(
                    "PhysX full contact report is unavailable inside its "
                    "post-step contact callback"
                )
            friction_anchors = full_report[2]
            self._pending_contact_fields = self._contact_fields_from_report(
                (contact_headers, contact_points, friction_anchors)
            )
            self._contact_report_callback_error = None
        except Exception as exc:
            self._contact_report_callback_error = exc

    def _sample_contact_fields(self) -> dict[str, np.ndarray]:
        """Return the callback copy, or read an empty one-step report directly."""

        if bool(
            getattr(self.env, "_physx_prefer_tensor_contact_capture", False)
        ):
            # P0-03's exact articulated asset exposes stable USD body paths
            # through its preconfigured RigidContactViews.  The CPU callback
            # report contains transient collider ids for this articulation,
            # so it cannot produce a process-independent path signature even
            # though the contact payload itself is populated.  Honor the
            # task's explicit preference and use the public tensor payload,
            # whose sensor/filter paths are schema-bound and canonicalizable.
            self._pending_contact_fields = None
            self._contact_report_callback_error = None
            tensor_fields = self._sample_tensor_contact_fields()
            if tensor_fields is None:
                raise RuntimeError(
                    "PhysX tensor contact capture was explicitly required but "
                    "no detailed contact views are available"
                )
            return self._canonicalize_contact_fields(tensor_fields)

        callback_error = self._contact_report_callback_error
        if callback_error is not None:
            self._contact_report_callback_error = None
            raise RuntimeError(
                "Could not copy PhysX contact data in its post-step callback"
            ) from callback_error
        pending = self._pending_contact_fields
        if pending is not None:
            self._pending_contact_fields = None
            pending_copy = {
                name: np.asarray(value).copy()
                for name, value in pending.items()
            }
            if int(pending_copy["physx_contact_header_count"][0]) > 0:
                return self._canonicalize_contact_fields(pending_copy)
        interface = self._contact_report_interface
        if interface is None:
            return self._canonicalize_contact_fields(
                self._empty_contact_fields()
            )
        try:
            report = interface.get_full_contact_report()
        except Exception as exc:
            raise RuntimeError(
                "Could not read PhysX's full contact report after a physics step"
            ) from exc
        report_fields = self._contact_fields_from_report(report)
        if int(report_fields["physx_contact_header_count"][0]) > 0 or bool(
            getattr(
                self.env,
                "_physx_require_full_contact_report_capture",
                False,
            )
        ):
            return self._canonicalize_contact_fields(report_fields)
        tensor_fields = self._sample_tensor_contact_fields()
        return self._canonicalize_contact_fields(
            tensor_fields if tensor_fields is not None else report_fields
        )

    def _sample_deformable_fields(self) -> dict[str, np.ndarray]:
        if self.deformable is None:
            return {}
        positions = np.asarray(
            getattr(self.env, self.deformable.position_attribute),
            dtype=np.float64,
        )
        velocities = np.asarray(
            getattr(self.env, self.deformable.velocity_attribute),
            dtype=np.float64,
        )
        if (
            positions.shape != self.deformable_shape
            or velocities.shape != self.deformable_shape
        ):
            raise RuntimeError(
                "PhysX deformable trace shape changed during an episode: "
                f"position={positions.shape}, velocity={velocities.shape}, "
                f"expected={self.deformable_shape}."
            )
        invalid = False
        if self.deformable.invalid_attribute is not None:
            invalid = bool(
                getattr(self.env, self.deformable.invalid_attribute, False)
            )
        finite = bool(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
        )
        return {
            "physx_deformable_position": np.nan_to_num(
                positions,
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "physx_deformable_velocity": np.nan_to_num(
                velocities,
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "physx_deformable_valid": np.asarray(
                [int(not invalid and finite)],
                dtype=np.uint8,
            ),
        }

    def _sim_time(self) -> float:
        sim = getattr(self.env, "_sim", None)
        if sim is not None:
            for name in ("current_time", "get_current_time"):
                value = getattr(sim, name, None)
                try:
                    value = value() if callable(value) else value
                    if value is not None and np.isfinite(float(value)):
                        return float(value)
                except Exception:
                    continue
        return float(getattr(self.env, "env_step", 0)) * float(
            getattr(self.env, "_control_dt", 0.0)
        )

    def begin_substep_capture(self) -> None:
        """Start a fresh fixed-rate trace for one environment control step."""

        self._substep_snapshots = []
        self._latest_substep_sample = None

    def capture_substep(self) -> None:
        """Capture commands and task readback after one live PhysX substep."""

        max_count = max(1, int(getattr(self.env, "_n_substeps", 1)))
        if len(self._substep_snapshots) >= max_count:
            raise RuntimeError(
                "PhysX executed more substeps than the recording schema "
                f"declares ({max_count}); refusing to truncate the physical "
                "replay trajectory"
            )
        sample: dict[str, np.ndarray] = {}
        sample.update(self._sample_robot_fields())
        sample.update(self._sample_command_fields())
        sample.update(self._sample_rigid_body_fields())
        sample.update(self._sample_articulation_fields())
        sample.update(self._sample_contact_fields())
        sample["physx_sim_time"] = np.asarray(
            [self._sim_time()],
            dtype=np.float64,
        )
        self._latest_substep_sample = sample
        self._substep_snapshots.append(
            {
                name: np.asarray(sample[name]).copy()
                for name in self._SUBSTEP_CAPTURE_FIELDS
                if name in sample
            }
        )

    def latest_substep_fields(
        self,
        names: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        """Return small read-only copies from the just-captured substep.

        Task safety checks use this seam to reuse the recorder's one batched
        GPU readback instead of synchronizing a second set of rigid-body
        tensor views.  Missing fields stay missing so callers can fail closed
        or disable a diagnostic without receiving stale data.
        """

        sample = self._latest_substep_sample
        if sample is None:
            return {}
        return {
            name: np.asarray(sample[name]).copy()
            for name in names
            if name in sample
        }

    def substep_snapshot(
        self,
        *,
        consume: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return the most recent substeps padded to the schema shape."""

        snapshots = self._substep_snapshots
        result: dict[str, np.ndarray] = {}
        specs = self.field_specs
        for source_name in self._SUBSTEP_CAPTURE_FIELDS:
            substep_name = source_name.replace(
                "physx_",
                "physx_substep_",
                1,
            )
            spec = specs.get(substep_name)
            if spec is None:
                continue
            value = np.zeros(
                tuple(int(size) for size in spec["shape"]),
                dtype=np.dtype(spec["dtype"]),
            )
            for index, snapshot in enumerate(snapshots[: value.shape[0]]):
                if source_name in snapshot:
                    value[index] = snapshot[source_name]
            result[substep_name] = value
        result["physx_substep_count"] = np.asarray(
            [len(snapshots)],
            dtype=np.int64,
        )
        if consume:
            self._substep_snapshots = []
        return result

    def snapshot_from_latest_substep(self) -> dict[str, np.ndarray]:
        """Reuse the final substep readback for the same control step.

        Deformable, RNG, and task-controller fields are not part of the
        fixed-rate rigid-body substep schema, so they remain sampled at the
        action boundary.  If no current substep exists (for example during
        reset), fall back to a full live snapshot instead of reusing stale
        data.
        """

        sample = self._latest_substep_sample
        if sample is None:
            return self.snapshot()
        result = {
            name: np.asarray(value).copy()
            for name, value in sample.items()
        }
        result.update(self._sample_deformable_fields())
        if self.has_rng_state:
            result.update(self._sample_rng_fields())
        result.update(self._sample_task_controller_state())
        result["physx_control_step"] = np.asarray(
            [float(getattr(self.env, "env_step", 0))],
            dtype=np.float64,
        )
        return result

    def _sample_task_controller_state(self) -> dict[str, np.ndarray]:
        """Sample optional action-aligned task state at either snapshot path."""

        if not self.has_task_controller_state:
            return {}
        provider = getattr(
            self.env,
            "get_physical_replay_task_controller_state",
        )
        controller_state = np.asarray(provider())
        if controller_state.shape != (3,) or (
            controller_state.dtype != np.dtype(np.uint8)
        ):
            raise RuntimeError(
                "Physical-replay task controller state must have uint8 "
                f"shape (3,), got {controller_state.dtype} "
                f"{controller_state.shape}"
            )
        return {"physx_task_controller_state": controller_state.copy()}

    def snapshot(self) -> dict[str, np.ndarray]:
        result = self._sample_robot_fields()
        result.update(self._sample_command_fields())
        result.update(self._sample_rigid_body_fields())
        result.update(self._sample_articulation_fields())
        result.update(self._sample_contact_fields())
        result.update(self._sample_deformable_fields())
        if self.has_rng_state:
            result.update(self._sample_rng_fields())
        result.update(self._sample_task_controller_state())
        result["physx_sim_time"] = np.asarray(
            [self._sim_time()],
            dtype=np.float64,
        )
        result["physx_control_step"] = np.asarray(
            [float(getattr(self.env, "env_step", 0))],
            dtype=np.float64,
        )
        return result

    @staticmethod
    def _require_binary_valid_mask(
        value: np.ndarray,
        *,
        name: str,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Validate one replay validity mask before any boundary write."""

        mask = np.asarray(value)
        if mask.shape != shape:
            raise ValueError(
                f"PhysX replay field {name} has shape {mask.shape}; "
                f"expected {shape}"
            )
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                f"PhysX replay field {name} is not a binary validity mask"
            )
        return mask.astype(bool)

    def write_zero_velocity_boundary(
        self,
        snapshot: Mapping[str, Any],
        recorded_schema: Mapping[str, Any],
        *,
        sync: bool = True,
    ) -> dict[str, int]:
        """Write an exact-zero live boundary without teleporting coordinates.

        This transaction is intentionally narrower than :meth:`restore`.
        It validates the complete named layout first, aligns robot drive
        targets to the accepted live joint coordinates, and writes only root,
        joint, rigid-body, articulation, and deformable velocities.  In
        particular, it never calls a pose, joint-position, or nodal-position
        setter, so a settled contact manifold remains the one validated by the
        preceding physical gate.
        """

        physx = self.validate_restore_schema(recorded_schema)

        # Validate every array and named live row before the first simulator
        # write.  A malformed/partial mask must fail closed instead of leaving
        # an only-partly zeroed physical boundary.
        robot_position = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_position",
            ndim=2,
        )
        robot_velocity = self._require_snapshot_array(
            snapshot,
            "physx_robot_joint_velocity",
            ndim=2,
        )
        if robot_velocity.shape != robot_position.shape:
            raise ValueError(
                "PhysX robot joint position/velocity boundary shapes differ"
            )
        robot_valid = self._require_binary_valid_mask(
            self._require_snapshot_array(
                snapshot,
                "physx_robot_joint_valid",
                ndim=2,
            ),
            name="physx_robot_joint_valid",
            shape=robot_position.shape,
        )
        robot_roots = self._require_snapshot_array(
            snapshot,
            "physx_robot_root_state",
            ndim=2,
        )
        if robot_roots.shape[-1:] != (13,):
            raise ValueError(
                "PhysX robot root boundary state must have 13 columns"
            )
        robot_root_valid = self._require_binary_valid_mask(
            self._require_snapshot_array(
                snapshot,
                "physx_robot_root_valid",
                ndim=1,
            ),
            name="physx_robot_root_valid",
            shape=(robot_roots.shape[0],),
        )

        if self.rigid_bodies:
            rigid_states = self._require_snapshot_array(
                snapshot,
                "physx_task_rigid_body_state",
                ndim=2,
            )
            if rigid_states.shape != (len(physx["rigid_body_names"]), 13):
                raise ValueError(
                    "PhysX rigid-body boundary state does not match its schema"
                )
            rigid_valid = self._require_binary_valid_mask(
                self._require_snapshot_array(
                    snapshot,
                    "physx_task_rigid_body_valid",
                    ndim=1,
                ),
                name="physx_task_rigid_body_valid",
                shape=(rigid_states.shape[0],),
            )
        else:
            rigid_states = np.zeros((0, 13), dtype=np.float64)
            rigid_valid = np.zeros((0,), dtype=bool)

        if self.articulations:
            articulation_roots = self._require_snapshot_array(
                snapshot,
                "physx_task_articulation_root_state",
                ndim=2,
            )
            if articulation_roots.shape != (
                len(physx["articulation_names"]),
                13,
            ):
                raise ValueError(
                    "PhysX articulation-root boundary state does not match "
                    "its schema"
                )
            articulation_root_valid = self._require_binary_valid_mask(
                self._require_snapshot_array(
                    snapshot,
                    "physx_task_articulation_root_valid",
                    ndim=1,
                ),
                name="physx_task_articulation_root_valid",
                shape=(articulation_roots.shape[0],),
            )
        else:
            articulation_roots = np.zeros((0, 13), dtype=np.float64)
            articulation_root_valid = np.zeros((0,), dtype=bool)
        has_articulation_dofs = any(
            entry.dof_count > 0 for entry in self.articulations
        )
        if has_articulation_dofs:
            articulation_dof_position = self._require_snapshot_array(
                snapshot,
                "physx_task_articulation_dof_position",
                ndim=2,
            )
            articulation_dof_velocity = self._require_snapshot_array(
                snapshot,
                "physx_task_articulation_dof_velocity",
                ndim=2,
            )
            if articulation_dof_velocity.shape != articulation_dof_position.shape:
                raise ValueError(
                    "PhysX articulation DOF position/velocity boundary shapes differ"
                )
            articulation_dof_valid = self._require_binary_valid_mask(
                self._require_snapshot_array(
                    snapshot,
                    "physx_task_articulation_dof_valid",
                    ndim=2,
                ),
                name="physx_task_articulation_dof_valid",
                shape=articulation_dof_position.shape,
            )
        else:
            articulation_dof_position = np.zeros((len(self.articulations), 0))
            articulation_dof_valid = np.zeros_like(
                articulation_dof_position,
                dtype=bool,
            )

        deformable_valid = False
        if self.deformable is not None:
            deformable_position = self._require_snapshot_array(
                snapshot,
                "physx_deformable_position",
                ndim=2,
            )
            deformable_velocity = self._require_snapshot_array(
                snapshot,
                "physx_deformable_velocity",
                ndim=2,
            )
            if (
                deformable_position.shape != self.deformable_shape
                or deformable_velocity.shape != self.deformable_shape
            ):
                raise ValueError(
                    "PhysX deformable boundary state does not match its schema"
                )
            deformable_mask = self._require_binary_valid_mask(
                self._require_snapshot_array(
                    snapshot,
                    "physx_deformable_valid",
                    ndim=1,
                ),
                name="physx_deformable_valid",
                shape=(1,),
            )
            deformable_valid = bool(deformable_mask[0])
            if deformable_valid and not np.all(np.isfinite(deformable_position)):
                raise ValueError(
                    "PhysX zero-velocity boundary has non-finite deformable positions"
                )

        sides = tuple(str(side) for side in physx["robot_sides"])
        joint_names = physx["robot_joint_names"]
        robot_joint_count = 0
        robot_write_plan: list[tuple[Any, np.ndarray, bool]] = []
        for entry in self.robots:
            if entry.robot is None:
                continue
            side_index = sides.index(entry.side)
            if side_index >= robot_position.shape[0]:
                raise ValueError(
                    f"PhysX robot boundary is missing side {entry.side!r}"
                )
            recorded_names = tuple(
                str(name) for name in joint_names[entry.side]
            )
            recorded_index = {
                name: index for index, name in enumerate(recorded_names)
            }
            live_targets = np.zeros(entry.joint_count, dtype=np.float64)
            for runtime_index, name in enumerate(entry.joint_names):
                source_index = recorded_index[name]
                if (
                    source_index >= robot_position.shape[1]
                    or not robot_valid[side_index, source_index]
                    or not np.isfinite(robot_position[side_index, source_index])
                ):
                    raise ValueError(
                        "PhysX zero-velocity boundary is missing valid robot "
                        f"joint {entry.side}/{name}"
                    )
                live_targets[runtime_index] = robot_position[
                    side_index,
                    source_index,
                ]
                robot_joint_count += 1
            root_is_valid = bool(
                side_index < robot_root_valid.shape[0]
                and robot_root_valid[side_index]
            )
            if root_is_valid and not np.all(
                np.isfinite(robot_roots[side_index])
            ):
                raise ValueError(
                    f"PhysX zero-velocity boundary has non-finite {entry.side} root"
                )
            robot_write_plan.append(
                (entry, live_targets, root_is_valid)
            )

        rigid_names = tuple(str(name) for name in physx["rigid_body_names"])
        rigid_index = {name: index for index, name in enumerate(rigid_names)}
        body_is_active = getattr(
            self.env,
            "_physx_task_rigid_body_is_active",
            None,
        )
        active_rigid_count = 0
        for entry in self.rigid_bodies:
            source_index = rigid_index[entry.name]
            active = not callable(body_is_active) or bool(
                body_is_active(entry.name)
            )
            if active:
                if not rigid_valid[source_index] or not np.all(
                    np.isfinite(rigid_states[source_index])
                ):
                    raise ValueError(
                        "PhysX zero-velocity boundary is missing valid rigid "
                        f"body {entry.name}"
                    )
                active_rigid_count += 1

        articulation_names = tuple(
            str(name) for name in physx["articulation_names"]
        )
        articulation_index = {
            name: index for index, name in enumerate(articulation_names)
        }
        active_articulation_roots = 0
        active_articulation_dofs = 0
        for entry in self.articulations:
            source_index = articulation_index[entry.name]
            if self._articulation_root_velocity_is_writable(entry.name):
                if not articulation_root_valid[source_index] or not np.all(
                    np.isfinite(articulation_roots[source_index])
                ):
                    raise ValueError(
                        "PhysX zero-velocity boundary is missing valid articulation "
                        f"root {entry.name}"
                    )
                active_articulation_roots += 1
            if self._articulation_dof_velocity_is_writable(entry.name):
                source_dofs = {
                    str(name): index
                    for index, name in enumerate(
                        physx["articulation_dof_names"][entry.name]
                    )
                }
                for name in entry.dof_names:
                    source_dof_index = source_dofs[name]
                    if (
                        source_dof_index >= articulation_dof_valid.shape[1]
                        or not articulation_dof_valid[
                            source_index,
                            source_dof_index,
                        ]
                        or not np.isfinite(
                            articulation_dof_position[
                                source_index,
                                source_dof_index,
                            ]
                        )
                    ):
                        raise ValueError(
                            "PhysX zero-velocity boundary is missing valid "
                            f"articulation DOF {entry.name}/{name}"
                        )
                    active_articulation_dofs += 1

        if self.deformable is not None and not deformable_valid:
            raise ValueError(
                "PhysX zero-velocity boundary is missing its deformable body"
            )

        # Resolve every required writer before mutating the simulator.  This
        # keeps missing backend capabilities in the fail-closed preflight
        # phase instead of discovering them after a different actor was
        # already changed.
        for entry, _live_targets, root_is_valid in robot_write_plan:
            if not callable(
                getattr(entry.robot, "write_joint_velocity_to_sim", None)
            ):
                raise RuntimeError(
                    "PhysX zero-velocity boundary requires the robot joint-"
                    "velocity writer"
                )
            if root_is_valid and not callable(
                getattr(entry.robot, "write_root_velocity_to_sim", None)
            ):
                raise RuntimeError(
                    "PhysX zero-velocity boundary requires the robot root-"
                    "velocity writer"
                )
        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )
        rigid_name_set = {entry.name for entry in self.rigid_bodies}
        for entry in self.rigid_bodies:
            if entry.kinematic or entry.name in kinematic_paths:
                continue
            writer = (
                getattr(entry.handle, "set_velocities", None)
                if entry.source == "physx_tensor_view"
                else getattr(entry.handle, "write_root_velocity_to_sim", None)
            )
            if not callable(writer):
                raise RuntimeError(
                    "PhysX zero-velocity boundary is missing a rigid-body "
                    f"velocity writer for {entry.name}"
                )
        for entry in self.articulations:
            if (
                self._articulation_root_velocity_is_writable(entry.name)
                and entry.name not in rigid_name_set
                and not callable(
                getattr(entry.view, "set_root_velocities", None)
                )
            ):
                raise RuntimeError(
                    "PhysX zero-velocity boundary is missing an articulation-"
                    f"root velocity writer for {entry.name}"
                )
            if (
                self._articulation_dof_velocity_is_writable(entry.name)
                and entry.dof_count
                and not callable(
                getattr(entry.view, "set_dof_velocities", None)
                )
            ):
                raise RuntimeError(
                    "PhysX zero-velocity boundary is missing an articulation-"
                    f"DOF velocity writer for {entry.name}"
                )
        if self.deformable is not None:
            deformable_view = getattr(
                self.env,
                (
                    "_e_garment_view"
                    if self.deformable.name == "surface"
                    else "_e02_rope_view"
                ),
                None,
            )
            if deformable_view is None or not callable(
                getattr(
                    deformable_view,
                    "set_simulation_nodal_velocities",
                    None,
                )
            ):
                raise RuntimeError(
                    "PhysX zero-velocity boundary is missing its deformable "
                    "velocity writer"
                )

        # All coverage is valid.  From this point onward only velocity and
        # actuator-target APIs are permitted.
        for entry, live_targets, root_is_valid in robot_write_plan:
            robot = entry.robot
            velocity_template = robot.data.joint_vel
            zero_joint_velocity = np.zeros_like(_numpy(velocity_template))
            write_joint_velocity = robot.write_joint_velocity_to_sim
            write_joint_velocity(
                self._array_like(velocity_template, zero_joint_velocity)
            )
            if root_is_valid:
                root_template = robot.data.root_state_w
                zero_root_velocity = np.zeros(
                    (_numpy(root_template).shape[0], 6),
                    dtype=np.float64,
                )
                write_root_velocity = robot.write_root_velocity_to_sim
                write_root_velocity(
                    self._array_like(root_template, zero_root_velocity)
                )
            target_template = getattr(self.env, "_joint_targets", {}).get(
                entry.side,
                robot.data.joint_pos,
            )
            target_values = _numpy(target_template).copy()
            target_values.reshape(-1, target_values.shape[-1])[0, : entry.joint_count] = (
                live_targets
            )
            target = self._array_like(target_template, target_values)
            robot.set_joint_position_target(target)
            joint_targets = getattr(self.env, "_joint_targets", None)
            if isinstance(joint_targets, dict):
                joint_targets[entry.side] = target
            write_data = getattr(robot, "write_data_to_sim", None)
            if callable(write_data):
                write_data()

        self.zero_task_velocities()
        if sync:
            self._sync_after_restore()
        return {
            "robot_joints": int(robot_joint_count),
            "rigid_bodies": int(active_rigid_count),
            "articulation_roots": int(active_articulation_roots),
            "articulation_dofs": int(active_articulation_dofs),
            "deformable_nodes": (
                int(self.deformable_shape[0])
                if self.deformable is not None and deformable_valid
                else 0
            ),
        }

    def zero_task_velocities(self) -> dict[str, int]:
        """Zero task velocities without rewriting poses or robot state."""

        kinematic_paths = set(
            getattr(self.env, "_task_kinematic_body_paths", ())
        )
        rigid_names: set[str] = set()
        rigid_bodies = 0
        articulation_roots = 0
        articulation_dofs = 0
        deformable_nodes = 0
        for entry in self.rigid_bodies:
            rigid_names.add(entry.name)
            if entry.kinematic or entry.name in kinematic_paths:
                continue
            if entry.source == "physx_tensor_view":
                template = entry.handle.get_velocities()
                values = np.zeros_like(_numpy(template))
                self._set_indexed(
                    entry.handle.set_velocities,
                    self._array_like(template, values),
                    entry.indices,
                )
            else:
                template = entry.handle.data.root_state_w
                row_count = max(1, int(_numpy(template).shape[0]))
                velocities = np.zeros((row_count, 6), dtype=np.float64)
                entry.handle.write_root_velocity_to_sim(
                    self._array_like(template, velocities)
                )
            rigid_bodies += 1

        for entry in self.articulations:
            if (
                self._articulation_root_velocity_is_writable(entry.name)
                and entry.name not in rigid_names
            ):
                template = entry.view.get_root_velocities()
                values = np.zeros_like(_numpy(template))
                self._set_indexed(
                    entry.view.set_root_velocities,
                    self._array_like(template, values),
                    entry.indices,
                )
                articulation_roots += 1
            if (
                self._articulation_dof_velocity_is_writable(entry.name)
                and entry.dof_count
            ):
                template = entry.view.get_dof_velocities()
                values = np.zeros_like(_numpy(template))
                self._set_indexed(
                    entry.view.set_dof_velocities,
                    self._array_like(template, values),
                    entry.indices,
                )
                articulation_dofs += entry.dof_count

        if self.deformable is not None:
            if self.deformable.name == "surface":
                view = getattr(self.env, "_e_garment_view", None)
                indices = getattr(self.env, "_e_garment_indices", None)
            else:
                view = getattr(self.env, "_e02_rope_view", None)
                indices = getattr(self.env, "_e02_rope_indices", None)
            if view is None:
                raise RuntimeError(
                    "PhysX deformable view for "
                    f"{self.deformable.name!r} is unavailable"
                )
            position_template = view.get_simulation_nodal_positions()
            velocity_template = view.get_simulation_nodal_velocities()
            position_values = _numpy(position_template)
            position_rows = position_values.reshape(
                -1,
                *self.deformable_shape,
            )
            body_index = 0
            if indices is not None:
                index_values = _numpy(indices, dtype=np.int64).reshape(-1)
                if index_values.size != 1:
                    raise RuntimeError(
                        "Stable-origin deformable clamp requires exactly one "
                        f"task body; got indices={index_values.tolist()}."
                    )
                body_index = int(index_values[0])
            if body_index < 0 or body_index >= position_rows.shape[0]:
                raise RuntimeError(
                    "Stable-origin deformable task index is outside the live "
                    f"PhysX tensor: index={body_index}, "
                    f"bodies={position_rows.shape[0]}."
                )
            live_positions = np.asarray(
                position_rows[body_index],
                dtype=np.float64,
            ).copy()
            zero_velocities = np.zeros(
                self.deformable_shape,
                dtype=np.float64,
            )
            if self.deformable.name == "surface":
                validate_nodal_state = getattr(
                    self.env,
                    "_basic_nodal_state_issue",
                    None,
                )
                if callable(validate_nodal_state):
                    issue, diagnostics = validate_nodal_state(
                        live_positions,
                        zero_velocities,
                        expected_node_count=getattr(
                            self.env,
                            "_e_expected_node_count",
                            self.deformable_shape[0],
                        ),
                    )
                    if issue is not None:
                        reject_readback = getattr(
                            self.env,
                            "_reject_garment_readback",
                            None,
                        )
                        if callable(reject_readback):
                            reject_readback(issue, diagnostics)
                        raise RuntimeError(
                            "Stable-origin velocity clamp rejected invalid "
                            f"surface readback: {issue}."
                        )
            if not np.all(np.isfinite(live_positions)):
                raise RuntimeError(
                    "Stable-origin velocity clamp read non-finite deformable "
                    "positions."
                )
            velocity_values = np.zeros_like(_numpy(velocity_template))
            velocity_payload = None
            if self.deformable.name == "surface":
                surface_array_like = getattr(
                    self.env,
                    "_surface_array_like",
                    None,
                )
                if callable(surface_array_like):
                    velocity_payload = surface_array_like(
                        velocity_template,
                        velocity_values,
                    )
            if velocity_payload is None:
                velocity_payload = self._array_like(
                    velocity_template,
                    velocity_values,
                )
            self._set_indexed(
                view.set_simulation_nodal_velocities,
                velocity_payload,
                indices,
            )
            # The collection environments cache deformable readback for trace
            # sampling.  Keep the live position reached by this physics step;
            # only velocity is overwritten.  Otherwise a later canonical
            # boundary write could accidentally restore stale reset nodes.
            accept_garment_readback = getattr(
                self.env,
                "_accept_garment_readback",
                None,
            )
            if (
                self.deformable.name == "surface"
                and callable(accept_garment_readback)
            ):
                # Keep every garment guard/cache on the exact live state.  A
                # direct update of only ``_e_nodes_np`` leaves the 15 mm
                # control-frame displacement baseline stale and can make the
                # next otherwise-valid readback latch a false terminal error.
                accept_garment_readback(
                    live_positions,
                    zero_velocities,
                    {
                        "max_control_node_displacement": 0.0,
                        "max_nodal_speed": 0.0,
                    },
                )
            else:
                setattr(
                    self.env,
                    self.deformable.position_attribute,
                    live_positions,
                )
                setattr(
                    self.env,
                    self.deformable.velocity_attribute,
                    zero_velocities,
                )
            if self.deformable.name == "rope":
                centerline_state = getattr(
                    self.env,
                    "_centerline_state_from_nodes",
                    None,
                )
                if callable(centerline_state):
                    centers, center_velocities = centerline_state(
                        live_positions,
                        zero_velocities,
                    )
                    self.env._e02_centers_np = np.asarray(
                        centers,
                        dtype=np.float64,
                    ).copy()
                    self.env._e02_velocities_np = np.asarray(
                        center_velocities,
                        dtype=np.float64,
                    ).copy()
            deformable_nodes = int(self.deformable_shape[0])
        return {
            "rigid_bodies": rigid_bodies,
            "articulation_roots": articulation_roots,
            "articulation_dofs": articulation_dofs,
            "deformable_nodes": deformable_nodes,
        }
