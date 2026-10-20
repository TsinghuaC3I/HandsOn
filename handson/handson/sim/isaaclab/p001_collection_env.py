"""Native Isaac Sim 5.1 / PhysX runtime for P0-01."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .p001_contract import (
    ACTIVE_JOINT_NAMES,
    ASSET_FILE_SHA256,
    ASSET_RELATIVE_PATH,
    CENTER_BODY_PATH,
    CONTACT_POLICY_VERSION,
    FAN_ROOT_PATH,
    FORBIDDEN_CONTACT_FORCE_N,
    LEFT_END_BODY_PATH,
    LEFT_RIGHT_BALANCE_MAX_DEG,
    P001PhysicalReadback,
    P001ResetState,
    P001_TASK_STATE_DIM,
    P001_TASK_STATE_SCHEMA_VERSION,
    PLANARITY_MAX_M,
    RENDER_RESOLUTION_HW,
    REQUIRED_BILATERAL_EXPANSION_FRACTION,
    RIGHT_END_BODY_PATH,
    ROOT_Z_M,
    STABLE_JOINT_SPEED_RADPS,
    STABLE_LINEAR_SPEED_MPS,
    SUCCESS_HOLD_STEPS,
    TARGET_OPEN_CHOICES_DEG,
    TARGET_TOLERANCE_DEG,
    TASK_NAME,
    TIME_LIMIT_SECONDS,
    bilateral_contact_credit_allowed,
    evaluate_physical_readback,
    hard_failure_reason,
    sample_reset_state,
    span_components_deg,
)
from .physx_ground_truth import configure_physx_contact_reports_for_recording


_MATERIAL_PATH = "/World/PhysicsMaterials/P001FoldingFan"
_LIGHT_ROOT = "/World/Task/P001Lights"
_PLANAR_JOINT_PATH = "/World/Task/P001FanPlanarSupportJoint"
_SAFE_X_RANGE_M = (-0.40, 0.40)
_SAFE_Y_RANGE_M = (-0.40, 0.40)
_SAFE_Z_RANGE_M = (0.84, 1.15)
_BODY_PATTERN = f"{FAN_ROOT_PATH}/Foldingfan001*"
_FAN_BODY_PATHS = (
    CENTER_BODY_PATH,
    *(f"{FAN_ROOT_PATH}/Foldingfan001_Prop{index:03d}" for index in range(1, 20)),
)
_FAN_COLLIDER_COUNT = 21
_FAN_CONTACT_OFFSET_M = 0.001
_FAN_EFFECTIVE_RADIUS_M = 0.1955
_P001_PHYSICAL_STATE_REPLAY_ATOL = 1.0e-4
_P001_PHYSICAL_STATE_REPLAY_FIELDS = frozenset(
    {
        "physx_task_rigid_body_state",
        "physx_task_articulation_root_state",
        "physx_task_articulation_dof_position",
        "physx_substep_task_rigid_body_state",
        "physx_substep_task_articulation_root_state",
        "physx_substep_task_articulation_dof_position",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_fan_quaternion_xyzw(yaw_deg: float) -> np.ndarray:
    """Rotate source XZ fan plane onto XY, then point it by table yaw."""

    roll = math.radians(90.0)
    yaw = math.radians(float(yaw_deg))
    qx = np.asarray([math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0)])
    qz = np.asarray([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _authored_rigid_state_from_transform(transform_xyzw: Any) -> np.ndarray:
    """Convert one synchronized public PhysX pose to trace state layout."""

    if hasattr(transform_xyzw, "detach"):
        transform_xyzw = transform_xyzw.detach()
    if hasattr(transform_xyzw, "cpu"):
        transform_xyzw = transform_xyzw.cpu()
    if hasattr(transform_xyzw, "numpy"):
        transform_xyzw = transform_xyzw.numpy()
    transform = np.asarray(transform_xyzw, dtype=np.float64).reshape(-1, 7)[0]
    if not np.all(np.isfinite(transform)):
        raise ValueError("P0-01 authored rigid transform is non-finite")
    state = np.zeros(13, dtype=np.float64)
    state[:3] = transform[:3]
    state[3:7] = transform[[6, 3, 4, 5]]
    return state


class HandsOnP001IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual physical folding-fan opening with exact endpoint ownership."""

    def __init__(self, task_name: str = TASK_NAME, *args: Any, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"P0-01 runtime requires {TASK_NAME!r}, got {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).strip().lower()
        if physics_backend not in {"physx", "isaacsim", "isaac"}:
            raise ValueError("P0-01 is Isaac Sim 5.1 PhysX-only")
        kwargs["physics_backend"] = "physx"
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("time_limit", TIME_LIMIT_SECONDS)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)

        requested_seed = 0 if kwargs.get("seed") is None else int(kwargs["seed"])
        requested_target = kwargs.pop("target_open_deg", None)
        self._p001_forced_target_deg = (
            None if requested_target is None else float(requested_target)
        )
        self._p001_reset_serial = -1
        self._p001_reset = sample_reset_state(
            requested_seed, target_open_deg=self._p001_forced_target_deg
        )
        self._p001_articulation_view: Any | None = None
        self._p001_center_view: Any | None = None
        self._p001_left_end_view: Any | None = None
        self._p001_right_end_view: Any | None = None
        self._p001_all_body_view: Any | None = None
        self._p001_contact_views: dict[str, Any] = {}
        self._p001_contact_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._p001_active_dof_indices: tuple[int, ...] = ()
        self._p001_dof_names: tuple[str, ...] = ()
        self._p001_reset_root_position = np.asarray(
            [self._p001_reset.root_x_m, self._p001_reset.root_y_m, ROOT_Z_M],
            dtype=np.float64,
        )
        self._p001_positive_span_increase_deg = 0.0
        self._p001_bilateral_positive_span_increase_deg = 0.0
        self._p001_bilateral_frame_start_deg = 0.0
        self._p001_previous_span_deg = float(self._p001_reset.initial_span_deg)
        self._p001_last_readback: P001PhysicalReadback | None = None
        self._p001_last_metrics: dict[str, float | bool] = {}
        self._p001_failure_latched = False
        self._p001_failure_reported = False
        self._p001_failure_reason = ""
        self._p001_contact_seen_this_control_frame = False
        # Kept as zero/false compatibility slots in the 23-value task state.
        # Robot-table force is diagnostic-only under CONTACT_POLICY_VERSION.
        self._p001_robot_table_hard_contact_substeps = 0
        self._p001_robot_table_hard_failure_latched = False
        self._p001_contact_contract_ready = False
        self._p001_substep_count = 0
        self._p001_mass_attrs: list[tuple[Any, float]] = []
        self._p001_material_api: Any | None = None
        self._physx_prefer_tensor_contact_capture = True
        self._fast_shutdown_on_close = True
        super().__init__(task_name, *args, **kwargs)

    def _configure_ground_truth_contact_reports(self, task_mjcf: Path) -> None:
        """Register the project-owned fan bodies in every replay process."""

        del task_mjcf
        if self._stage is None:
            raise RuntimeError(
                "P0-01 PhysX contact recording requires the constructed USD stage"
            )
        from pxr import UsdPhysics

        prim_paths: list[str] = []
        for path in _FAN_BODY_PATHS:
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsActive():
                raise RuntimeError(
                    f"P0-01 PhysX contact-report body is missing: {path}"
                )
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(
                    f"P0-01 PhysX contact-report prim is not rigid: {path}"
                )
            prim_paths.append(path)
        self._physx_contact_sensor_paths_preconfigured = tuple(prim_paths)
        configured = configure_physx_contact_reports_for_recording(
            self._stage,
            prim_paths,
        )
        self._physx_contact_report_paths_preconfigured = configured
        if not configured:
            raise RuntimeError(
                "P0-01 failed to configure the fan articulation for PhysX "
                "contact recording"
            )

    def _physx_ground_truth_replay_atol_overrides(self) -> dict[str, float]:
        """Keep the articulated fan pose budget local to P0-01."""

        return {
            name: _P001_PHYSICAL_STATE_REPLAY_ATOL
            for name in _P001_PHYSICAL_STATE_REPLAY_FIELDS
        }

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "fan_center_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "fan_left_end_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "fan_right_end_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "fan_joint_positions": spaces.Box(-np.inf, np.inf, shape=(18,)),
                "fan_joint_velocities": spaces.Box(-np.inf, np.inf, shape=(18,)),
                "fan_target_angle": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "fan_task_state": spaces.Box(
                    -np.inf, np.inf, shape=(P001_TASK_STATE_DIM,)
                ),
            }
        )
        return observation_space

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("P0-01 stage is unavailable before asset composition")
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

        package_root = Path(__file__).resolve().parents[2]
        asset_path = package_root / ASSET_RELATIVE_PATH
        bundle_root = asset_path.parent
        for relative_path, expected_sha in ASSET_FILE_SHA256.items():
            path = bundle_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"P0-01 exact asset file is missing: {path}")
            actual_sha = _sha256(path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"P0-01 asset SHA mismatch for {relative_path}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )

        root = UsdGeom.Xform.Define(self._stage, FAN_ROOT_PATH).GetPrim()
        root.GetReferences().ClearReferences()
        root.GetReferences().AddReference(str(asset_path.resolve()))
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(
                float(self._p001_reset.root_x_m),
                float(self._p001_reset.root_y_m),
                ROOT_Z_M,
            )
        )
        quaternion = _flat_fan_quaternion_xyzw(self._p001_reset.root_yaw_deg)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(
                float(quaternion[3]),
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
            )
        )

        required_prims = [
            self._stage.GetPrimAtPath(CENTER_BODY_PATH),
            self._stage.GetPrimAtPath(LEFT_END_BODY_PATH),
            self._stage.GetPrimAtPath(RIGHT_END_BODY_PATH),
        ]
        if not all(prim.IsValid() for prim in required_prims):
            raise RuntimeError("P0-01 composed center and endpoint bodies are incomplete")

        sim_utils.make_uninstanceable(FAN_ROOT_PATH, stage=self._stage)
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self._p001_reset.friction),
            dynamic_friction=float(self._p001_reset.friction),
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(_MATERIAL_PATH, material_cfg)
        sim_utils.bind_physics_material(
            FAN_ROOT_PATH,
            _MATERIAL_PATH,
            stage=self._stage,
            stronger_than_descendants=True,
        )
        self._p001_material_api = UsdPhysics.MaterialAPI(
            self._stage.GetPrimAtPath(_MATERIAL_PATH)
        )

        self._p001_mass_attrs.clear()
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
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
                else 0.005
            )
            mass_attr.Set(nominal_mass * float(self._p001_reset.mass_scale))
            self._p001_mass_attrs.append((mass_attr, nominal_mass))
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api.CreateDisableGravityAttr(False)
            rigid_api.CreateAngularDampingAttr(3.0)
            report = (
                PhysxSchema.PhysxContactReportAPI.Get(self._stage, prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(prim)
            )
            report.CreateThresholdAttr().Set(0.0)

        # The ribs are only about 1.8 mm thick.  PhysX's generic contact shell
        # is too large for this asset and lifts/slides the pivot on first
        # contact.  Match the benchmark's established thin-object shell while
        # preserving every source mesh and collider unchanged.
        collider_count = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collider_count += 1
            collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not collision_api:
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api.CreateContactOffsetAttr().Set(_FAN_CONTACT_OFFSET_M)
            collision_api.CreateRestOffsetAttr().Set(0.0)
        if collider_count != _FAN_COLLIDER_COUNT:
            raise RuntimeError(
                "P0-01 exact asset collider count changed: "
                f"expected {_FAN_COLLIDER_COUNT}, got {collider_count}"
            )
        for endpoint_body_path, collider_name in (
            (LEFT_END_BODY_PATH, "Foldingfan001_Collider1"),
            (RIGHT_END_BODY_PATH, "Foldingfan001_Collider19"),
        ):
            endpoint_collider = self._stage.GetPrimAtPath(
                f"{endpoint_body_path}/Collisions/{collider_name}"
            )
            endpoint_mesh_api = UsdPhysics.MeshCollisionAPI(endpoint_collider)
            endpoint_mesh_api.CreateApproximationAttr().Set("sdf")
            endpoint_sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI(
                endpoint_collider
            )
            if not endpoint_sdf_api:
                endpoint_sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(
                    endpoint_collider
                )
            endpoint_sdf_api.CreateSdfResolutionAttr().Set(512)
            endpoint_sdf_api.CreateSdfMarginAttr().Set(0.001)
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(root)
        articulation.CreateEnabledSelfCollisionsAttr(False)
        articulation.CreateSolverPositionIterationCountAttr(12)
        articulation.CreateSolverVelocityIterationCountAttr(4)

        # The rigid sectors do not include a deformable paper sheet, so the
        # authored 44 mm pivot stack has no broad support and the otherwise
        # horizontal fan rolls onto its edge.  Use a transparent planar root
        # joint: only table-normal tipping (rotX/rotY) is locked.  All XYZ
        # translations, table yaw, and all 19 source revolute joints stay free.
        planar_joint = UsdPhysics.Joint.Define(
            self._stage, _PLANAR_JOINT_PATH
        )
        planar_joint.CreateExcludeFromArticulationAttr().Set(True)
        planar_joint.CreateBody1Rel().SetTargets([Sdf.Path(CENTER_BODY_PATH)])
        planar_joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(
                float(self._p001_reset_root_position[0]),
                float(self._p001_reset_root_position[1]),
                float(self._p001_reset_root_position[2]),
            )
        )
        planar_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        # Express the D6 angular axes in the table/world frame.  The source
        # fan's local +Y becomes world +Z after reset, so using the body frame
        # here would leave a horizontal tipping axis free instead of table yaw.
        planar_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        planar_joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(
                float(quaternion[3]),
                Gf.Vec3f(
                    -float(quaternion[0]),
                    -float(quaternion[1]),
                    -float(quaternion[2]),
                ),
            )
        )
        # In this Isaac/PhysX build an undeclared D6 axis is free and the
        # inverted limit below locks that axis.  Keep XYZ and rotZ undeclared;
        # only table-normal tipping is constrained.
        for axis in ("rotX", "rotY"):
            limit = UsdPhysics.LimitAPI.Apply(planar_joint.GetPrim(), axis)
            limit.CreateLowAttr().Set(1.0)
            limit.CreateHighAttr().Set(-1.0)

        # The common table is authored before this hook but does not carry a
        # contact reporter on the canonical baseline.  The robot-table
        # forbidden-contact view is created after the first reset, so enable
        # its exact PhysX report here while stage mutation is still legal.
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            raise RuntimeError("P0-01 table rigid body is unavailable at spawn")
        table_report = (
            PhysxSchema.PhysxContactReportAPI.Get(
                self._stage, table_prim.GetPath()
            )
            if table_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            else PhysxSchema.PhysxContactReportAPI.Apply(table_prim)
        )
        table_report.CreateThresholdAttr().Set(0.0)

        dome = UsdLux.DomeLight.Define(self._stage, f"{_LIGHT_ROOT}/DomeLight")
        dome.CreateIntensityAttr(450.0)
        dome.CreateColorAttr(Gf.Vec3f(0.88, 0.92, 1.0))
        distant = UsdLux.DistantLight.Define(
            self._stage, f"{_LIGHT_ROOT}/DistantLight"
        )
        distant.CreateIntensityAttr(1800.0)
        distant.CreateAngleAttr(2.0)
        distant.AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 18.0, 24.0))

    def _configure_task_collision_safety(self, sim_utils: Any) -> tuple[int, int]:
        tuned = super()._configure_task_collision_safety(sim_utils)
        if self._stage is None:
            return tuned
        from pxr import PhysxSchema

        # Robots are authored after _spawn_task_specific_assets.  This hook is
        # the last collision-configuration phase before SimulationContext
        # reset, so hand reporters become part of the parsed PhysX scene.
        reporter_count = 0
        for side in ("left", "right"):
            for body_path in self._robot_rigid_body_paths(side, hand_only=True):
                body_prim = self._stage.GetPrimAtPath(body_path)
                report = (
                    PhysxSchema.PhysxContactReportAPI.Get(
                        self._stage, body_prim.GetPath()
                    )
                    if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                    else PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
                )
                report.CreateThresholdAttr().Set(0.0)
                reporter_count += 1
        if reporter_count:
            sim_utils.update_stage()
        return tuned

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

    def _create_contact_view(
        self,
        physics_view: Any,
        key: str,
        sensor_path: str,
        filter_paths: list[str],
    ) -> None:
        if not filter_paths:
            raise RuntimeError(f"P0-01 contact filter {key} resolved no bodies")
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
            raise RuntimeError(f"P0-01 contact view {key} is incomplete")
        self._p001_contact_views[key] = view
        self._p001_contact_paths[key] = (sensor_path, tuple(filter_paths))

    def _create_net_contact_view(
        self, physics_view: Any, key: str, sensor_path: str
    ) -> None:
        view = physics_view.create_rigid_contact_view(sensor_path)
        if view is None or int(view.sensor_count) <= 0:
            raise RuntimeError(f"P0-01 net contact view {key} is incomplete")
        self._p001_contact_views[key] = view
        self._p001_contact_paths[key] = (sensor_path, ())

    @staticmethod
    def _resolve_dof_names(view: Any) -> tuple[str, ...]:
        for owner in (view, getattr(view, "shared_metatype", None)):
            if owner is None:
                continue
            for attribute in ("dof_names", "joint_names"):
                names = getattr(owner, attribute, None)
                if names:
                    return tuple(str(name) for name in names)
        raise RuntimeError("P0-01 articulation did not expose DOF names")

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            raise RuntimeError("P0-01 PhysX tensor simulation view is unavailable")
        self._p001_articulation_view = physics_view.create_articulation_view(
            FAN_ROOT_PATH
        )
        self._p001_center_view = physics_view.create_rigid_body_view(CENTER_BODY_PATH)
        self._p001_left_end_view = physics_view.create_rigid_body_view(
            LEFT_END_BODY_PATH
        )
        self._p001_right_end_view = physics_view.create_rigid_body_view(
            RIGHT_END_BODY_PATH
        )
        self._p001_all_body_view = physics_view.create_rigid_body_view(_BODY_PATTERN)
        views = (
            self._p001_articulation_view,
            self._p001_center_view,
            self._p001_left_end_view,
            self._p001_right_end_view,
            self._p001_all_body_view,
        )
        if any(view is None or int(view.count) <= 0 for view in views):
            raise RuntimeError("P0-01 physical articulation/body views are incomplete")
        if int(self._p001_all_body_view.count) != 20:
            raise RuntimeError(
                "P0-01 exact asset must expose 20 rigid bodies, got "
                f"{int(self._p001_all_body_view.count)}"
            )
        self._p001_dof_names = self._resolve_dof_names(
            self._p001_articulation_view
        )
        missing = [name for name in ACTIVE_JOINT_NAMES if name not in self._p001_dof_names]
        if missing:
            raise RuntimeError(f"P0-01 active joints are missing: {missing}")
        self._p001_active_dof_indices = tuple(
            self._p001_dof_names.index(name) for name in ACTIVE_JOINT_NAMES
        )

        left_hand = self._robot_rigid_body_paths("left", hand_only=True)
        right_hand = self._robot_rigid_body_paths("right", hand_only=True)
        all_robot = self._robot_rigid_body_paths("left", hand_only=False)
        all_robot.extend(self._robot_rigid_body_paths("right", hand_only=False))
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            raise RuntimeError("P0-01 table rigid body is unavailable")
        self._create_contact_view(
            physics_view, "left_end", LEFT_END_BODY_PATH, left_hand
        )
        self._create_contact_view(
            physics_view, "right_end", RIGHT_END_BODY_PATH, right_hand
        )
        self._create_contact_view(
            physics_view, "left_wrong_end", RIGHT_END_BODY_PATH, left_hand
        )
        self._create_contact_view(
            physics_view, "right_wrong_end", LEFT_END_BODY_PATH, right_hand
        )
        self._create_contact_view(
            physics_view,
            "left_end_reverse",
            "/World/HandsOnRobots/LeftRobot/base_left/left_finger*",
            [LEFT_END_BODY_PATH],
        )
        self._create_contact_view(
            physics_view,
            "right_end_reverse",
            "/World/HandsOnRobots/RightRobot/base_right/right_finger*",
            [RIGHT_END_BODY_PATH],
        )
        self._create_contact_view(
            physics_view,
            "left_wrong_end_reverse",
            "/World/HandsOnRobots/LeftRobot/base_left/left_finger*",
            [RIGHT_END_BODY_PATH],
        )
        self._create_contact_view(
            physics_view,
            "right_wrong_end_reverse",
            "/World/HandsOnRobots/RightRobot/base_right/right_finger*",
            [LEFT_END_BODY_PATH],
        )
        self._create_contact_view(
            physics_view, "robot_table", str(table_prim.GetPath()), all_robot
        )
        self._create_contact_view(
            physics_view, "fan_table", _BODY_PATTERN, [str(table_prim.GetPath())]
        )
        self._create_contact_view(
            physics_view, "fan_any_right", _BODY_PATTERN, right_hand
        )
        self._create_contact_view(
            physics_view,
            "fan_any_right_reverse",
            "/World/HandsOnRobots/RightRobot/base_right/right_finger*",
            list(_FAN_BODY_PATHS),
        )
        self._create_net_contact_view(
            physics_view, "right_end_net", RIGHT_END_BODY_PATH
        )
        self._p001_contact_contract_ready = True
        print(
            "[HandsOn/P001] Exact contract ready: "
            f"bodies=20, active_joints=18, left_hand={len(left_hand)}, "
            f"right_hand={len(right_hand)}",
            flush=True,
        )

    def reset(
        self,
        seed: int | None = None,
        *,
        target_label: str | None = None,
        target_open_deg: float | None = None,
        **kwargs: Any,
    ):
        requested_target = target_open_deg
        if requested_target is None and target_label:
            requested_target = float(str(target_label).replace("deg", "").strip())
        if target_open_deg is not None or target_label is not None:
            self._p001_forced_target_deg = (
                None if requested_target is None else float(requested_target)
            )
        if seed is not None:
            self._p001_reset_serial = -1
        return super().reset(seed=seed, **kwargs)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._p001_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._p001_reset_serial
        self._p001_reset = sample_reset_state(
            episode_seed, target_open_deg=self._p001_forced_target_deg
        )
        self._p001_positive_span_increase_deg = 0.0
        self._p001_bilateral_positive_span_increase_deg = 0.0
        self._p001_bilateral_frame_start_deg = 0.0
        self._p001_previous_span_deg = float(self._p001_reset.initial_span_deg)
        self._p001_last_readback = None
        self._p001_last_metrics = {}
        self._p001_failure_latched = False
        self._p001_failure_reported = False
        self._p001_failure_reason = ""
        self._p001_contact_seen_this_control_frame = False
        # Legacy task-state compatibility slots; table contact no longer gates.
        self._p001_robot_table_hard_contact_substeps = 0
        self._p001_robot_table_hard_failure_latched = False
        self._p001_substep_count = 0

    def _apply_randomized_dynamics(self) -> None:
        material_api = self._p001_material_api
        if material_api:
            static_attr = material_api.GetStaticFrictionAttr()
            dynamic_attr = material_api.GetDynamicFrictionAttr()
            if not static_attr or not static_attr.IsValid():
                static_attr = material_api.CreateStaticFrictionAttr()
            if not dynamic_attr or not dynamic_attr.IsValid():
                dynamic_attr = material_api.CreateDynamicFrictionAttr()
            static_attr.Set(float(self._p001_reset.friction))
            dynamic_attr.Set(float(self._p001_reset.friction))
        for mass_attr, nominal_mass in self._p001_mass_attrs:
            mass_attr.Set(float(nominal_mass) * float(self._p001_reset.mass_scale))

    def _apply_task_pose_randomization(self) -> None:
        view = self._p001_articulation_view
        if view is None or self._torch is None:
            return
        indices = self._torch.arange(
            int(view.count), dtype=self._torch.long, device=self._isaac_device
        )
        transforms = view.get_root_transforms().clone()
        root_velocities = self._torch.zeros_like(view.get_root_velocities())
        dof_positions = view.get_dof_positions().clone()
        dof_velocities = self._torch.zeros_like(view.get_dof_velocities())
        quaternion = _flat_fan_quaternion_xyzw(self._p001_reset.root_yaw_deg)
        root_position = np.asarray(
            [self._p001_reset.root_x_m, self._p001_reset.root_y_m, ROOT_Z_M],
            dtype=np.float64,
        )
        transforms[:, :3] = self._torch.as_tensor(
            root_position, dtype=transforms.dtype, device=transforms.device
        )
        transforms[:, 3:7] = self._torch.as_tensor(
            quaternion, dtype=transforms.dtype, device=transforms.device
        )
        active_degrees = self._p001_reset.active_joint_positions_deg
        for dof_index, angle_deg in zip(self._p001_active_dof_indices, active_degrees):
            dof_positions[:, int(dof_index)] = math.radians(float(angle_deg))
        view.set_root_transforms(transforms, indices)
        view.set_root_velocities(root_velocities, indices)
        view.set_dof_positions(dof_positions, indices)
        view.set_dof_velocities(dof_velocities, indices)
        # The low-level PhysX setters are asynchronous.  Refresh articulation
        # kinematics without advancing the solver so the pre-step authored
        # snapshot contains child-link poses consistent with the DOFs just
        # written.  Otherwise an outer rib is compared against the
        # previous cached transform and appears to rotate by about 0.117 rad
        # on frame zero, even though that change is only setter propagation.
        if getattr(self, "_physx_ground_truth_enabled", False):
            physics_view = next(iter(self._robots.values()))._physics_sim_view
            update_kinematics = getattr(
                physics_view,
                "update_articulations_kinematic",
                None,
            )
            if not callable(update_kinematics):
                raise RuntimeError(
                    "P0-01 reset requires no-step articulation kinematic sync"
                )
            update_kinematics()
        self._record_authored_task_reset_write(
            FAN_ROOT_PATH,
            transform_xyzw=transforms,
            velocity=root_velocities,
            dof_position=dof_positions,
            dof_velocity=dof_velocities,
        )
        self._p001_reset_root_position = root_position.copy()
        self._apply_randomized_dynamics()
        print(
            "[HandsOn/P001] Seeded reset: "
            + json.dumps(self._p001_reset.to_dict(), sort_keys=True),
            flush=True,
        )

    def _capture_recording_origin_authored_reset_snapshot(
        self,
        *,
        require_authored_task_writes: bool = True,
    ) -> None:
        """Replace stale trace-view link caches at the authored boundary."""

        super()._capture_recording_origin_authored_reset_snapshot(
            require_authored_task_writes=require_authored_task_writes
        )
        if not getattr(self, "_physx_ground_truth_enabled", False):
            return
        snapshot = getattr(
            self,
            "_recording_origin_authored_reset_snapshot",
            None,
        )
        trace = getattr(self, "_physx_ground_truth_trace", None)
        if not isinstance(snapshot, dict) or trace is None:
            raise RuntimeError("P0-01 authored ground-truth snapshot is unavailable")
        view_paths = {
            CENTER_BODY_PATH,
            LEFT_END_BODY_PATH,
            RIGHT_END_BODY_PATH,
        }
        expected = {
            str(entry.name)
            for entry in trace.rigid_bodies
            if str(entry.name) in view_paths
        }
        if not expected:
            articulation_names = {
                str(entry.name)
                for entry in getattr(trace, "articulations", ())
            }
            if FAN_ROOT_PATH not in articulation_names:
                raise RuntimeError(
                    "P0-01 authored ground-truth coverage is missing the fan "
                    f"articulation: expected={FAN_ROOT_PATH!r}, "
                    f"actual={sorted(articulation_names)}"
                )
            # Shared articulation capture owns the fan root and all of its
            # moving links.  Older layouts also exposed selected child links
            # as rigid rows, whose stale transforms still need the patch below.
            # An articulation-only layout has no independent rigid row to patch.
            return
        views = {
            CENTER_BODY_PATH: self._p001_center_view,
            LEFT_END_BODY_PATH: self._p001_left_end_view,
            RIGHT_END_BODY_PATH: self._p001_right_end_view,
        }
        states = np.asarray(snapshot["physx_task_rigid_body_state"])
        valid = np.asarray(snapshot["physx_task_rigid_body_valid"])
        patched: set[str] = set()
        for row_index, entry in enumerate(trace.rigid_bodies):
            path = str(entry.name)
            view = views.get(path)
            if view is None:
                continue
            states[row_index] = _authored_rigid_state_from_transform(
                view.get_transforms()
            )
            valid[row_index] = 1
            patched.add(path)
        if patched != expected:
            raise RuntimeError(
                "P0-01 authored rigid-view coverage is incomplete: "
                f"expected={sorted(expected)}, patched={sorted(patched)}"
            )

    @staticmethod
    def _view_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    def _body_pose(self, view: Any | None, fallback_position: np.ndarray) -> np.ndarray:
        if view is None:
            return np.concatenate(
                [fallback_position, np.asarray([0.0, 0.0, 0.0, 1.0])]
            )
        try:
            transform = self._view_numpy(view.get_transforms())[0].reshape(7)
            return transform.copy()
        except Exception:
            return np.concatenate(
                [fallback_position, np.asarray([0.0, 0.0, 0.0, 1.0])]
            )

    def _active_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        view = self._p001_articulation_view
        if view is None or not self._p001_active_dof_indices:
            positions = np.deg2rad(
                np.asarray(self._p001_reset.active_joint_positions_deg)
            )
            return positions, np.zeros(18, dtype=np.float64)
        positions = self._view_numpy(view.get_dof_positions())[0]
        velocities = self._view_numpy(view.get_dof_velocities())[0]
        indices = np.asarray(self._p001_active_dof_indices, dtype=np.int64)
        return positions[indices].copy(), velocities[indices].copy()

    def _contact_force(self, key: str) -> float:
        view = self._p001_contact_views.get(key)
        if view is None:
            return 0.0
        try:
            normal_forces, _points, _normals, _separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )
            forces = np.abs(self._view_numpy(normal_forces).reshape(-1))
            counts_np = self._view_numpy(counts).astype(np.int64).reshape(-1)
            starts_np = self._view_numpy(starts).astype(np.int64).reshape(-1)
            total = 0.0
            for count, start in zip(counts_np, starts_np):
                if count > 0:
                    total += float(np.sum(forces[int(start) : int(start + count)]))
            return total
        except Exception as exc:
            self._add_task_state_warning(f"P0-01 contact readback {key} failed: {exc}")
            return math.nan

    def _bidirectional_contact_force(self, key: str) -> float:
        forward = self._contact_force(key)
        reverse = self._contact_force(f"{key}_reverse")
        return max(float(forward), float(reverse))

    def _body_metrics(self) -> tuple[float, float, bool, float]:
        view = self._p001_all_body_view
        if view is None:
            return 0.0, 0.0, True, ROOT_Z_M
        try:
            transforms = self._view_numpy(view.get_transforms()).reshape(-1, 7)
            velocities = self._view_numpy(view.get_velocities()).reshape(-1, 6)
            positions = transforms[:, :3]
            max_speed = float(np.max(np.linalg.norm(velocities[:, :3], axis=1)))
            # All source joints share the pivot, so body-origin Z spread alone
            # cannot detect the whole fan tipping.  Source local +Y is the fan
            # plane normal; rotate it by each PhysX xyzw quaternion and convert
            # the worst tilt to an outer-rib height error.
            quaternions = transforms[:, 3:7]
            normals = np.repeat(
                np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
                repeats=len(quaternions),
                axis=0,
            )
            q_vectors = quaternions[:, :3]
            twice_cross = 2.0 * np.cross(q_vectors, normals)
            world_normals = (
                normals
                + quaternions[:, 3:4] * twice_cross
                + np.cross(q_vectors, twice_cross)
            )
            tilt_error = _FAN_EFFECTIVE_RADIUS_M * float(
                np.max(np.linalg.norm(world_normals[:, :2], axis=1))
            )
            planarity = max(float(np.ptp(positions[:, 2])), tilt_error)
            safe = bool(
                np.all((positions[:, 0] >= _SAFE_X_RANGE_M[0]) & (positions[:, 0] <= _SAFE_X_RANGE_M[1]))
                and np.all((positions[:, 1] >= _SAFE_Y_RANGE_M[0]) & (positions[:, 1] <= _SAFE_Y_RANGE_M[1]))
                and np.all((positions[:, 2] >= _SAFE_Z_RANGE_M[0]) & (positions[:, 2] <= _SAFE_Z_RANGE_M[1]))
            )
            return max_speed, planarity, safe, float(np.min(positions[:, 2]))
        except Exception:
            return math.nan, math.nan, False, math.nan

    def _physical_readback(self) -> P001PhysicalReadback:
        positions, velocities = self._active_joint_state()
        center_pose = self._body_pose(
            self._p001_center_view, self._p001_reset_root_position
        )
        max_speed, planarity, safe, min_height = self._body_metrics()
        return P001PhysicalReadback(
            active_joint_angles_rad=tuple(float(value) for value in positions),
            active_joint_velocities_radps=tuple(float(value) for value in velocities),
            root_position_m=tuple(float(value) for value in center_pose[:3]),
            reset_root_position_m=tuple(
                float(value) for value in self._p001_reset_root_position
            ),
            max_body_linear_speed_mps=max_speed,
            planarity_error_m=planarity,
            left_end_contact_force_n=self._bidirectional_contact_force("left_end"),
            right_end_contact_force_n=self._bidirectional_contact_force("right_end"),
            left_wrong_end_contact_force_n=self._bidirectional_contact_force(
                "left_wrong_end"
            ),
            right_wrong_end_contact_force_n=self._bidirectional_contact_force(
                "right_wrong_end"
            ),
            robot_table_contact_force_n=self._contact_force("robot_table"),
            positive_span_increase_deg=self._p001_positive_span_increase_deg,
            bilateral_positive_span_increase_deg=(
                self._p001_bilateral_positive_span_increase_deg
            ),
            all_bodies_inside_safe_zone=safe,
            min_body_height_m=min_height,
        )

    def _refresh_metrics(self) -> dict[str, float | bool]:
        self._p001_last_readback = self._physical_readback()
        self._p001_last_metrics = evaluate_physical_readback(
            self._p001_last_readback, self._p001_reset
        )
        return self._p001_last_metrics

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        if self._p001_articulation_view is None:
            return
        self._p001_substep_count += 1
        positions, _velocities = self._active_joint_state()
        span_deg, _left_deg, _right_deg = span_components_deg(positions)
        increase = max(0.0, span_deg - self._p001_previous_span_deg)
        left_force = self._bidirectional_contact_force("left_end")
        right_force = self._bidirectional_contact_force("right_end")
        left_wrong_force = self._bidirectional_contact_force("left_wrong_end")
        right_wrong_force = self._bidirectional_contact_force("right_wrong_end")
        action_phase_active = bool(
            getattr(self, "_trajectory_action_phase_active", False)
        )
        substep_index = int(
            getattr(self, "_trajectory_action_phase_substep_index", -1)
        )
        if action_phase_active and substep_index == 0:
            self._p001_contact_seen_this_control_frame = False
            self._p001_bilateral_frame_start_deg = float(
                self._p001_bilateral_positive_span_increase_deg
            )
        wrong_end_clear = bool(
            left_wrong_force <= FORBIDDEN_CONTACT_FORCE_N
            and right_wrong_force <= FORBIDDEN_CONTACT_FORCE_N
        )
        if action_phase_active:
            self._p001_positive_span_increase_deg += increase
            if not wrong_end_clear:
                if not self._p001_contact_seen_this_control_frame:
                    # A crossed-rib touch invalidates bilateral credit for the
                    # whole control frame, including earlier clear substeps.
                    self._p001_bilateral_positive_span_increase_deg = float(
                        self._p001_bilateral_frame_start_deg
                    )
                self._p001_contact_seen_this_control_frame = True
        # These fields remain in fan_task_state for recording compatibility,
        # but robot-table contact is deliberately diagnostic-only.
        self._p001_robot_table_hard_contact_substeps = 0
        self._p001_robot_table_hard_failure_latched = False
        if (
            action_phase_active
            and not self._p001_contact_seen_this_control_frame
            and bilateral_contact_credit_allowed(
                left_end_force_n=left_force,
                right_end_force_n=right_force,
                left_wrong_end_force_n=left_wrong_force,
                right_wrong_end_force_n=right_wrong_force,
            )
        ):
            self._p001_bilateral_positive_span_increase_deg += increase
        self._p001_previous_span_deg = span_deg
        metrics = self._refresh_metrics()
        self._update_p001_failure_latch(metrics)

    def _update_p001_failure_latch(
        self, metrics: Mapping[str, float | bool]
    ) -> bool:
        reason = hard_failure_reason(metrics)
        if reason is not None and not self._p001_failure_latched:
            self._p001_failure_latched = True
            self._p001_failure_reason = reason
        if self._p001_failure_latched and not self._p001_failure_reported:
            self._p001_failure_reported = True
            print(
                "[HandsOn/P001] Failure latched: "
                + json.dumps(
                    {
                        "env_step": int(self.env_step),
                        "physics_substep": int(self._p001_substep_count),
                        "reason": self._p001_failure_reason,
                        "contact_policy_version": CONTACT_POLICY_VERSION,
                        "robot_table_gating_enabled": False,
                        "metrics": metrics,
                        "readback": asdict(self._p001_last_readback),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return self._p001_failure_latched

    def _compute_success_now(self) -> bool:
        metrics = self._refresh_metrics()
        self._update_p001_failure_latch(metrics)
        return bool(
            metrics["success_now"]
            and not self._p001_contact_seen_this_control_frame
            and not self._p001_failure_latched
        )

    def _compute_failure(self) -> bool:
        metrics = self._refresh_metrics()
        return self._update_p001_failure_latch(metrics)

    def _success_steps_required(self) -> int:
        return SUCCESS_HOLD_STEPS

    def _task_info(self) -> dict[str, Any]:
        metrics = self._p001_last_metrics or self._refresh_metrics()
        readback = self._p001_last_readback
        robot_table_force_n = float(
            0.0 if readback is None else readback.robot_table_contact_force_n
        )
        robot_table_diagnostic_valid = math.isfinite(robot_table_force_n)
        contact_warnings: list[str] = []
        if not bool(metrics["wrong_end_clear"]):
            contact_warnings.append("crossed_rib_contact")
        if self._p001_contact_seen_this_control_frame and not contact_warnings:
            contact_warnings.append("transient_crossed_rib_contact")
        return {
            **{f"p001_{key}": value for key, value in metrics.items()},
            "p001_current_hard_failure": bool(metrics["failure"]),
            "p001_failure": bool(self._p001_failure_latched),
            "p001_success_counter": int(self._success_counter),
            "p001_success_steps_required": SUCCESS_HOLD_STEPS,
            "p001_failure_latched": bool(self._p001_failure_latched),
            "p001_failure_reason": self._p001_failure_reason or None,
            "p001_contact_policy_version": CONTACT_POLICY_VERSION,
            "p001_task_state_schema_version": P001_TASK_STATE_SCHEMA_VERSION,
            "p001_task_state_dim": P001_TASK_STATE_DIM,
            "p001_recoverable_contact_warnings": contact_warnings,
            "p001_contact_seen_this_control_frame": bool(
                self._p001_contact_seen_this_control_frame
            ),
            "p001_left_wrong_end_contact_force_n": float(
                0.0
                if readback is None
                else readback.left_wrong_end_contact_force_n
            ),
            "p001_right_wrong_end_contact_force_n": float(
                0.0
                if readback is None
                else readback.right_wrong_end_contact_force_n
            ),
            "p001_robot_table_contact_force_n": robot_table_force_n,
            "p001_robot_table_diagnostic_only": True,
            "p001_robot_table_diagnostic_valid": robot_table_diagnostic_valid,
            "p001_robot_table_contact_observed": bool(
                robot_table_diagnostic_valid
                and robot_table_force_n > FORBIDDEN_CONTACT_FORCE_N
            ),
            "p001_robot_table_hard_contact_force_n": None,
            "p001_robot_table_hard_contact_substeps": 0,
            "p001_robot_table_hard_contact_substeps_required": 0,
            "p001_robot_table_hard_failure_latched": False,
            "p001_time_limit_seconds": float(
                getattr(self, "_time_limit", TIME_LIMIT_SECONDS)
            ),
            "p001_reset": self._p001_reset.to_dict(),
            "p001_target_choices_deg": list(TARGET_OPEN_CHOICES_DEG),
            "p001_contact_contract_ready": bool(self._p001_contact_contract_ready),
            "p001_contact_view_paths": self._p001_contact_paths,
            "p001_object_state_writes_after_action_zero": 0,
        }

    def _task_status_text(self) -> str:
        metrics = self._p001_last_metrics or self._refresh_metrics()
        warnings: list[str] = []
        if not bool(metrics["wrong_end_clear"]):
            warnings.append("crossed rib")
        if (
            getattr(self, "_p001_contact_seen_this_control_frame", False)
            and not warnings
        ):
            warnings.append("transient crossed rib")
        warning_text = (
            "  |  Contact warning: " + ", ".join(warnings) if warnings else ""
        )
        return (
            f"Fan {float(metrics.get('span_deg', 0.0)):.1f}/"
            f"{self._p001_reset.target_open_deg:.0f} deg"
            f"  |  L/R diff {float(metrics.get('balance_error_deg', 0.0)):.1f} deg"
            f"  |  Hold {self._success_counter}/{SUCCESS_HOLD_STEPS}"
            f"{warning_text}"
        )

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._p001_last_metrics or self._refresh_metrics()
        rows = [
            self._condition_row(
                "p001_target",
                f"目标张角 {self._p001_reset.target_open_deg:.0f}°±{TARGET_TOLERANCE_DEG:.0f}°",
                bool(metrics["target_ok"]),
            ),
            self._condition_row(
                "p001_balance",
                f"左右展开差≤{LEFT_RIGHT_BALANCE_MAX_DEG:.0f}°",
                bool(metrics["balance_ok"]),
            ),
            self._condition_row(
                "p001_contacts",
                "左右手分别接触两根最外侧扇骨",
                bool(metrics["left_end_contact"] and metrics["right_end_contact"]),
            ),
            self._condition_row(
                "p001_expansion",
                "净展开达到70%，且至少"
                f"{REQUIRED_BILATERAL_EXPANSION_FRACTION * 100:.0f}%"
                "展开增量发生在双手有效接触期间",
                bool(metrics["net_increase_ok"] and metrics["bilateral_expansion_ok"]),
            ),
            self._condition_row(
                "p001_support",
                f"扇体平放且平面误差≤{PLANARITY_MAX_M * 1000:.0f} mm",
                bool(metrics["root_stable"] and metrics["planarity_ok"]),
            ),
            self._condition_row(
                "p001_velocity",
                f"末态速度≤{STABLE_LINEAR_SPEED_MPS:.2f} m/s、关节≤{STABLE_JOINT_SPEED_RADPS:.2f} rad/s",
                bool(metrics["velocity_stable"]),
            ),
            self._condition_row(
                "p001_forbidden",
                "当前无错侧接触且折扇未越界",
                bool(
                    metrics["wrong_end_clear"]
                    and metrics["safe_zone"]
                    and not getattr(
                        self, "_p001_contact_seen_this_control_frame", False
                    )
                ),
            ),
        ]
        return self._append_success_hold_row(rows)

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._p001_last_metrics or self._refresh_metrics()
        positions, velocities = self._active_joint_state()
        center = self._body_pose(self._p001_center_view, self._p001_reset_root_position)
        left = self._body_pose(self._p001_left_end_view, center[:3])
        right = self._body_pose(self._p001_right_end_view, center[:3])
        state = observation["state"]
        state["fan_center_pose"] = center
        state["fan_left_end_pose"] = left
        state["fan_right_end_pose"] = right
        state["fan_joint_positions"] = positions
        state["fan_joint_velocities"] = velocities
        state["fan_target_angle"] = np.asarray(
            [math.radians(self._p001_reset.target_open_deg)], dtype=np.float64
        )
        task_state = np.asarray(
            [
                float(self._p001_reset.seed),
                float(self._p001_reset.root_x_m),
                float(self._p001_reset.root_y_m),
                float(self._p001_reset.root_yaw_deg),
                float(self._p001_reset.initial_span_deg),
                float(self._p001_reset.target_open_deg),
                float(self._p001_reset.friction),
                float(self._p001_reset.mass_scale),
                float(metrics["span_deg"]),
                float(metrics["left_span_deg"]),
                float(metrics["right_span_deg"]),
                float(metrics["balance_error_deg"]),
                float(metrics["bilateral_expansion_fraction"]),
                float(metrics["root_displacement_m"]),
                float(metrics["planarity_error_m"]),
                float(metrics["left_end_contact"]),
                float(metrics["right_end_contact"]),
                float(metrics["velocity_stable"]),
                float(self._success_counter),
                float(self._p001_failure_latched),
                float(self._p001_contact_seen_this_control_frame),
                float(self._p001_robot_table_hard_contact_substeps),
                float(self._p001_robot_table_hard_failure_latched),
            ],
            dtype=np.float64,
        )
        if task_state.shape != (P001_TASK_STATE_DIM,):
            raise RuntimeError(
                "P0-01 fan_task_state schema mismatch: "
                f"expected {(P001_TASK_STATE_DIM,)}, got {task_state.shape}"
            )
        state["fan_task_state"] = task_state
        return observation


__all__ = ["HandsOnP001IsaacLabEnv"]
