"""Native Isaac Sim 5.1 / PhysX runtime for P0-05."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .priority_zero_robot_audit import initial_robot_state_report, pose_error_wxyz
from .p005_contract import (
    CONTROL_HZ,
    EXPECTED_DOF_NAMES,
    FAILURE_DWELL_SUBSTEPS,
    KEY_AXIS_ERROR_DEG,
    KEY_AXIS_LOCAL_M,
    KEY_BODY_NAME,
    KEY_INSERTION_RELEASE_DEPTH_M,
    KEY_INSERTION_DEPTH_M,
    KEY_LATERAL_ERROR_M,
    KEY_TIP_LOCAL_M,
    KEY_TURN_RELEASE_DEG,
    KEY_TURN_TARGET_DEG,
    KEY_DROP_MARGIN_M,
    LOCK_CYLINDER_DAMPING,
    LOCK_CYLINDER_DOF,
    LOCK_CYLINDER_BODY_NAME,
    LOCK_CYLINDER_MAX_FORCE,
    LOCK_CYLINDER_STIFFNESS,
    LOCKED_SHACKLE_DAMPING,
    LOCKED_SHACKLE_HINGE_STIFFNESS,
    LOCKED_SHACKLE_MAX_FORCE,
    LOCKED_SHACKLE_SLIDE_STIFFNESS,
    PADLOCK_BODY_NAME,
    PADLOCK_KEYHOLE_LOCAL_M,
    PADLOCK_KEYWAY_INWARD_LOCAL_M,
    PADLOCK_ROOT_QUATERNION_WXYZ,
    PADLOCK_LIFT_MIN_M,
    PHYSICS_HZ,
    P005PhysicalReadback,
    P005StageTracker,
    RENDER_RESOLUTION_HW,
    REQUIRED_CONTACT_FORCE_N,
    SHACKLE_BODY_NAME,
    SHACKLE_HINGE_DOF,
    SHACKLE_SLIDE_DOF,
    SUCCESS_HOLD_STEPS,
    TABLE_TOP_Z_M,
    TASK_NAME,
    TIMEOUT_STEPS,
    UNLOCKED_SHACKLE_DAMPING,
    UNLOCKED_SHACKLE_HINGE_STIFFNESS,
    UNLOCKED_SHACKLE_MAX_FORCE,
    UNLOCKED_SHACKLE_SLIDE_STIFFNESS,
    UNLOCKED_SHACKLE_TARGET_M,
    angle_between_degrees,
    cylinder_coupling_geometry_ok,
    evaluate_physical_readback,
    insertion_geometry_ok,
    key_reset_quaternion_wxyz,
    sample_reset_state,
)


_TASK_ROOT = "/World/Task"
_KEY_MATERIAL_PATH = "/World/PhysicsMaterials/P005Key"
_KEY_TIP_LOCAL_M = KEY_TIP_LOCAL_M
_KEYHOLE_LOCAL_M = PADLOCK_KEYHOLE_LOCAL_M
_KEY_AXIS_LOCAL = KEY_AXIS_LOCAL_M
_KEYWAY_INWARD_LOCAL = PADLOCK_KEYWAY_INWARD_LOCAL_M
_KEY_THICKNESS_HALF_M = 0.00672
_TASK_CONTACT_OFFSET_M = 0.001
_AUTHORED_RESET_TRANSLATION_LIMIT_M = 0.005
_AUTHORED_RESET_ROTATION_LIMIT_RAD = math.radians(3.0)


def _rotation_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class HandsOnP005IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Left-hand padlock lift, right-hand key turn, and auto shackle opening."""

    def __init__(self, task_name: str = TASK_NAME, *args: Any, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"P0-05 runtime requires {TASK_NAME!r}, got {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).lower()
        if physics_backend not in {"physx", "isaacsim", "isaac"}:
            raise ValueError("P0-05 requires Isaac Sim 5.1 PhysX")
        kwargs["physics_backend"] = "physx"
        kwargs.setdefault("physics_dt", 1.0 / PHYSICS_HZ)
        kwargs.setdefault("control_dt", 1.0 / CONTROL_HZ)
        kwargs.setdefault("hz", CONTROL_HZ)
        kwargs.setdefault("time_limit", TIMEOUT_STEPS / CONTROL_HZ)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)
        requested_timeout_steps = max(
            1,
            int(round(float(kwargs["time_limit"]) / float(kwargs["control_dt"]))),
        )

        requested_seed = 0 if kwargs.get("seed") is None else int(kwargs["seed"])
        self._p005_timeout_steps = requested_timeout_steps
        self._p005_reset_serial = -1
        self._p005_reset = sample_reset_state(requested_seed)
        self._p005_tracker = P005StageTracker()
        self._p005_body_views: dict[str, Any] = {}
        self._p005_dof_views: dict[str, tuple[Any, int]] = {}
        self._p005_contact_views: dict[str, Any] = {}
        self._p005_contact_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._p005_contact_contract_ready = False
        self._p005_last_readback: P005PhysicalReadback | None = None
        self._p005_last_metrics: dict[str, Any] = {}
        self._p005_failure_latched = False
        self._p005_failure_streak_substeps = 0
        self._p005_failure_snapshot: dict[str, Any] | None = None
        self._p005_shackle_locked: bool | None = None
        self._p005_lock_target_rad = 0.0
        self._p005_last_drive_target_rad = -math.inf
        self._p005_initial_state_report: dict[str, Any] = {}
        super().__init__(task_name, *args, **kwargs)
        # The shared runtime snaps the requested control interval to an integer
        # number of available PhysX substeps.  Preserve the public episode
        # budget in control steps after that snap instead of silently reducing
        # a nominal 1500-step P0-05 episode to roughly 1040 actions.
        self._time_limit = requested_timeout_steps * float(self._control_dt)

    def _ready_state_settle_physics_steps(self) -> int:
        """Settle free task bodies for one real second before ready-state checks."""

        return max(5, int(round(1.0 / float(self._physics_dt))))

    def _authored_reset_pose_report(self) -> dict[str, Any]:
        key_quaternion = key_reset_quaternion_wxyz(
            self._p005_reset.key_yaw_deg
        )
        targets = {
            "key": np.asarray(
                [
                    self._p005_reset.key_x_m,
                    self._p005_reset.key_y_m,
                    self._p005_reset.key_z_m,
                    *key_quaternion,
                ],
                dtype=np.float64,
            ),
            "padlock": np.concatenate(
                [
                    np.asarray(
                        [
                            self._p005_reset.padlock_x_m,
                            self._p005_reset.padlock_y_m,
                            self._p005_reset.padlock_z_m,
                        ],
                        dtype=np.float64,
                    ),
                    PADLOCK_ROOT_QUATERNION_WXYZ,
                ]
            ).astype(np.float64),
        }
        actuals = {
            "key": self._body_pose_wxyz(KEY_BODY_NAME),
            "padlock": self._body_pose_wxyz(PADLOCK_BODY_NAME),
        }
        rows: dict[str, Any] = {}
        for label in ("key", "padlock"):
            finite, translation_error, rotation_error = pose_error_wxyz(
                actuals[label], targets[label]
            )
            rows[label] = {
                "passed": bool(
                    finite
                    and translation_error <= _AUTHORED_RESET_TRANSLATION_LIMIT_M
                    and rotation_error <= _AUTHORED_RESET_ROTATION_LIMIT_RAD
                ),
                "target_pose_wxyz": targets[label].tolist(),
                "actual_pose_wxyz": actuals[label].tolist(),
                "translation_error_m": float(translation_error),
                "rotation_error_deg": math.degrees(rotation_error),
            }
        return {
            "passed": all(bool(row["passed"]) for row in rows.values()),
            "bodies": rows,
        }

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "p005_key_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "p005_padlock_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "p005_shackle_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "p005_joint_state": spaces.Box(-np.inf, np.inf, shape=(6,)),
                "p005_role_contacts": spaces.Box(-np.inf, np.inf, shape=(5,)),
                "p005_task_state": spaces.Box(-np.inf, np.inf, shape=(10,)),
            }
        )
        return observation_space

    def _task_extra_pose_names(self) -> set[str]:
        extras = super()._task_extra_pose_names()
        extras.update(
            {KEY_BODY_NAME, PADLOCK_BODY_NAME, SHACKLE_BODY_NAME,
             LOCK_CYLINDER_BODY_NAME, "p005_shackle_slide_body"}
        )
        return extras

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        """Add CCD and narrow contact shells to the key/lock mechanism."""

        rigid_count, collider_count = super()._configure_task_collision_safety(
            sim_utils
        )
        if self._stage is None:
            return rigid_count, collider_count

        from pxr import PhysxSchema, Usd, UsdPhysics

        task_bodies = []
        for name in (KEY_BODY_NAME, PADLOCK_BODY_NAME, SHACKLE_BODY_NAME, LOCK_CYLINDER_BODY_NAME):
            body = self._find_rigid_body_prim(name)
            if body is None:
                self._add_task_state_warning(
                    f"P0-05 collision body {name!r} is unavailable"
                )
                continue
            rigid = PhysxSchema.PhysxRigidBodyAPI(body)
            if not rigid:
                rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            for getter, creator, value in (
                ("GetEnableCCDAttr", "CreateEnableCCDAttr", True),
                (
                    "GetSolverPositionIterationCountAttr",
                    "CreateSolverPositionIterationCountAttr",
                    32,
                ),
                (
                    "GetSolverVelocityIterationCountAttr",
                    "CreateSolverVelocityIterationCountAttr",
                    8,
                ),
                (
                    "GetMaxDepenetrationVelocityAttr",
                    "CreateMaxDepenetrationVelocityAttr",
                    0.08,
                ),
                (
                    "GetLinearDampingAttr",
                    "CreateLinearDampingAttr",
                    0.35,
                ),
                (
                    "GetAngularDampingAttr",
                    "CreateAngularDampingAttr",
                    0.35,
                ),
            ):
                attribute = getattr(rigid, getter)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid, creator)()
                attribute.Set(value)
            task_bodies.append(body)

        collider_paths: set[str] = set()
        for name in (KEY_BODY_NAME, PADLOCK_BODY_NAME, LOCK_CYLINDER_BODY_NAME):
            root = self._find_rigid_body_prim(name)
            if root is None:
                continue
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                path = str(prim.GetPath())
                if path in collider_paths:
                    continue
                collision = PhysxSchema.PhysxCollisionAPI(prim)
                if not collision:
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact = collision.GetContactOffsetAttr()
                if not contact or not contact.IsValid():
                    contact = collision.CreateContactOffsetAttr()
                contact.Set(_TASK_CONTACT_OFFSET_M)
                rest = collision.GetRestOffsetAttr()
                if not rest or not rest.IsValid():
                    rest = collision.CreateRestOffsetAttr()
                rest.Set(0.0)
                collider_paths.add(path)

        return (
            rigid_count + len(task_bodies),
            collider_count + len(collider_paths),
        )

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("P0-05 stage is unavailable")
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        required: dict[str, Any] = {}
        for name in (KEY_BODY_NAME, PADLOCK_BODY_NAME, SHACKLE_BODY_NAME, LOCK_CYLINDER_BODY_NAME):
            prim = self._find_rigid_body_prim(name)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"P0-05 rigid body {name!r} is missing")
            required[name] = prim
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid.CreateMaxLinearVelocityAttr(3.0)
            rigid.CreateMaxAngularVelocityAttr(12.0)

        table = self._find_rigid_body_prim("table")
        if table is None:
            raise RuntimeError("P0-05 table rigid body is missing")
        PhysxSchema.PhysxContactReportAPI.Apply(table).CreateThresholdAttr(0.0)

        # Collision meshes are only physical shells; hide their debug render
        # geometry so the operator sees the YCB visuals and not duplicate gray
        # meshes around the key, body, guide rails, or shackle.
        collision_prefixes = (
            "p005_key_collision",
            "p005_padlock_rear_collision",
            "p005_keyway_guide_",
            "p005_shackle_arc",
            "p005_shackle_left_leg",
            "p005_shackle_right_leg",
        )
        for prim in Usd.PrimRange(self._stage.GetPrimAtPath(_TASK_ROOT)):
            if any(str(prim.GetName()).startswith(prefix) for prefix in collision_prefixes):
                UsdGeom.Imageable(prim).MakeInvisible()

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self._p005_reset.key_friction),
            dynamic_friction=float(max(0.45, self._p005_reset.key_friction - 0.10)),
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(_KEY_MATERIAL_PATH, material_cfg)
        key_material = UsdShade.Material(
            self._stage.GetPrimAtPath(_KEY_MATERIAL_PATH)
        )
        UsdShade.MaterialBindingAPI.Apply(required[KEY_BODY_NAME]).Bind(
            key_material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )

        # Keep the free lock body dynamic; the shackle and cylinder remain live
        # reduced-coordinate child joints while the left hand carries it.
        padlock_rigid = UsdPhysics.RigidBodyAPI(required[PADLOCK_BODY_NAME])
        kinematic = padlock_rigid.GetKinematicEnabledAttr()
        if kinematic.IsValid() and bool(kinematic.Get()):
            kinematic.Set(False)

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
            raise RuntimeError(f"P0-05 contact filter {key!r} resolved no bodies")
        view = physics_view.create_rigid_contact_view(
            sensor_path,
            filter_patterns=filter_paths,
            max_contact_data_count=max(128, len(filter_paths) * 16),
        )
        try:
            ready = bool(
                view is not None
                and int(view.sensor_count) == 1
                and int(view.filter_count) == len(filter_paths)
            )
        except (AttributeError, RuntimeError):
            ready = False
        if not ready:
            raise RuntimeError(f"P0-05 contact view {key!r} is empty or incomplete")
        self._p005_contact_views[key] = view
        self._p005_contact_paths[key] = (sensor_path, tuple(filter_paths))

    def _init_task_specific_runtime(self) -> None:
        if not self._robots or self._stage is None:
            return
        from pxr import Usd, UsdPhysics

        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            raise RuntimeError("P0-05 PhysX tensor simulation view is unavailable")

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        for prim in Usd.PrimRange(task_root):
            if not prim.IsActive() or not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            path = str(prim.GetPath())
            if path in {f"{_TASK_ROOT}/worldBody", f"{_TASK_ROOT}/Geometry"}:
                continue
            if "/base_left" in path or "/base_right" in path:
                continue
            try:
                view = physics_view.create_articulation_view(path)
                if int(view.count) != 1 or int(view.max_dofs) < 1:
                    continue
                names = {
                    str(name): int(index)
                    for name, index in view.get_metatype(0).dof_indices.items()
                }
                for name in EXPECTED_DOF_NAMES.intersection(names):
                    self._p005_dof_views[name] = (view, names[name])
            except Exception:
                continue
        if set(self._p005_dof_views) != set(EXPECTED_DOF_NAMES):
            raise RuntimeError(
                "P0-05 expected DOFs are unavailable: "
                f"found={sorted(self._p005_dof_views)}"
            )

        body_paths: dict[str, str] = {}
        for name in (KEY_BODY_NAME, PADLOCK_BODY_NAME, SHACKLE_BODY_NAME, LOCK_CYLINDER_BODY_NAME):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                raise RuntimeError(f"P0-05 body {name!r} did not import")
            path = str(prim.GetPath())
            view = physics_view.create_rigid_body_view(path)
            if int(view.count) != 1:
                raise RuntimeError(f"P0-05 rigid-body view {name!r} is empty")
            self._p005_body_views[name] = view
            body_paths[name] = path

        left_hand = self._robot_rigid_body_paths("left", hand_only=True)
        right_hand = self._robot_rigid_body_paths("right", hand_only=True)
        table = self._find_rigid_body_prim("table")
        if table is None:
            raise RuntimeError("P0-05 table body is unavailable")
        self._create_contact_view(
            physics_view, "right_key", body_paths[KEY_BODY_NAME], right_hand
        )
        self._create_contact_view(
            physics_view, "left_padlock", body_paths[PADLOCK_BODY_NAME], left_hand
        )
        # The front keyway guides used to belong to the shell. Keep physical
        # left-hand support on these surfaces counted after they move to the
        # cylinder; never count the shackle as a substitute for holding the lock.
        self._create_contact_view(
            physics_view, "left_cylinder", body_paths[LOCK_CYLINDER_BODY_NAME], left_hand
        )
        self._create_contact_view(
            physics_view,
            "key_support",
            body_paths[KEY_BODY_NAME],
            [str(table.GetPath())],
        )
        lock_filters = [
            body_paths[PADLOCK_BODY_NAME], body_paths[SHACKLE_BODY_NAME],
            body_paths[LOCK_CYLINDER_BODY_NAME],
        ]
        self._create_contact_view(
            physics_view, "key_lock", body_paths[KEY_BODY_NAME], lock_filters
        )
        self._create_contact_view(
            physics_view,
            "shackle_lock",
            body_paths[SHACKLE_BODY_NAME],
            [body_paths[PADLOCK_BODY_NAME], body_paths[LOCK_CYLINDER_BODY_NAME]],
        )
        self._p005_contact_contract_ready = True
        self._set_task_drives(shackle_locked=True, force=True)
        print(
            "[HandsOn/P005] physical contract ready: "
            f"dofs={sorted(self._p005_dof_views)}, "
            f"left_hand={len(left_hand)}, right_hand={len(right_hand)}",
            flush=True,
        )

    def _validate_task_initial_state_before_ready(self) -> Mapping[str, Any] | None:
        base_report = super()._validate_task_initial_state_before_ready()
        errors: list[str] = []
        readback = self._physical_readback()
        raw_cylinder_rad, _ = self._dof_reading(LOCK_CYLINDER_DOF)
        raw_slide_m, _ = self._dof_reading(SHACKLE_SLIDE_DOF)
        raw_hinge_rad, _ = self._dof_reading(SHACKLE_HINGE_DOF)
        raw_locked_dofs_finite = bool(
            np.isfinite([raw_cylinder_rad, raw_slide_m, raw_hinge_rad]).all()
        )
        authored_pose_report = self._authored_reset_pose_report()
        support_force, support_penetration = self._contact_reading("key_support")
        robot_pass, robot_report = initial_robot_state_report(self)
        if not readback.finite_state:
            errors.append("actual key/lock state is unavailable or non-finite")
        if support_force < 0.01:
            errors.append(
                f"key has no measured table support force: {support_force:.6f}N"
            )
        if not np.isfinite(support_penetration) or support_penetration > 0.0005:
            errors.append(
                "key/table support penetration is unavailable or exceeds 0.5 mm: "
                f"{support_penetration:.6f}m"
            )
        if readback.key_linear_speed_mps > 0.05:
            errors.append(
                "key initial linear speed exceeds 0.05 m/s: "
                f"{readback.key_linear_speed_mps:.6f}m/s"
            )
        if readback.key_angular_speed_radps > 0.75:
            errors.append(
                "key initial angular speed exceeds 0.75 rad/s: "
                f"{readback.key_angular_speed_radps:.6f}rad/s"
            )
        if not raw_locked_dofs_finite:
            errors.append("actual lock cylinder/slide/hinge state is unavailable")
        elif abs(math.degrees(raw_cylinder_rad)) > 1.0:
            errors.append(
                "lock cylinder must start at its locked zero: "
                f"{math.degrees(raw_cylinder_rad):.6f}deg"
            )
        if raw_locked_dofs_finite and abs(raw_slide_m) > 0.001:
            errors.append(
                "shackle slide must start closed within 1 mm: "
                f"{raw_slide_m:.6f}m"
            )
        if raw_locked_dofs_finite and abs(math.degrees(raw_hinge_rad)) > 1.0:
            errors.append(
                "shackle hinge must start at its locked zero: "
                f"{math.degrees(raw_hinge_rad):.6f}deg"
            )
        if self._p005_failure_latched:
            errors.append("task failure latched during pre-ready settling")
        if self._p005_tracker.success:
            errors.append("task reached success during pre-ready settling")
        if not bool(authored_pose_report["passed"]):
            failed_bodies = []
            for name, row in dict(authored_pose_report.get("bodies", {})).items():
                if not bool(row.get("passed", False)):
                    failed_bodies.append(
                        f"{name}(translation={float(row.get('translation_error_m', math.nan)):.6f}m,"
                        f"rotation={float(row.get('rotation_error_deg', math.nan)):.3f}deg)"
                    )
            detail = ", ".join(failed_bodies) or "unknown bodies"
            errors.append(
                "actual key/padlock poses differ from the authored reset: " + detail
            )
        if not robot_pass:
            errors.append("actual dual-ROKAE initial state failed the P0 audit")
        if errors:
            raise RuntimeError(
                "P0-05 initial physical-state audit failed before ready: "
                + "; ".join(errors)
            )
        report = {
            "key_support_force_n": float(support_force),
            "key_support_penetration_m": float(support_penetration),
            "key_linear_speed_mps": float(readback.key_linear_speed_mps),
            "key_angular_speed_radps": float(readback.key_angular_speed_radps),
            "lock_cylinder_angle_deg": float(math.degrees(raw_cylinder_rad)),
            "shackle_slide_m": float(raw_slide_m),
            "shackle_hinge_deg": float(math.degrees(raw_hinge_rad)),
            "actual_robot_state": robot_report,
            "authored_reset_pose": authored_pose_report,
        }
        self._p005_initial_state_report = report
        if isinstance(base_report, Mapping):
            return {**base_report, **report}
        return report

    def stable_recording_origin_task_readiness(self) -> Mapping[str, Any]:
        readback = self._physical_readback()
        raw_cylinder_rad, _ = self._dof_reading(LOCK_CYLINDER_DOF)
        raw_slide_m, _ = self._dof_reading(SHACKLE_SLIDE_DOF)
        raw_hinge_rad, _ = self._dof_reading(SHACKLE_HINGE_DOF)
        raw_locked_dofs_finite = bool(
            np.isfinite([raw_cylinder_rad, raw_slide_m, raw_hinge_rad]).all()
        )
        authored_pose_report = self._authored_reset_pose_report()
        support_force, support_penetration = self._contact_reading("key_support")
        ready = bool(
            readback.finite_state
            and raw_locked_dofs_finite
            and bool(authored_pose_report["passed"])
            and support_force >= 0.01
            and np.isfinite(support_penetration)
            and support_penetration <= 0.0005
            and readback.key_bottom_z_m >= TABLE_TOP_Z_M - KEY_DROP_MARGIN_M
            and abs(math.degrees(raw_cylinder_rad)) <= 1.0
            and abs(raw_slide_m) <= 0.001
            and abs(math.degrees(raw_hinge_rad)) <= 1.0
            and not self._p005_failure_latched
            and not self._p005_tracker.success
        )
        return {
            "ready": ready,
            "finite": bool(readback.finite_state),
            "key_support_force_n": float(support_force),
            "key_support_penetration_m": float(support_penetration),
            "key_bottom_z_m": float(readback.key_bottom_z_m),
            "lock_cylinder_angle_deg": float(math.degrees(raw_cylinder_rad)),
            "shackle_slide_m": float(raw_slide_m),
            "shackle_hinge_deg": float(math.degrees(raw_hinge_rad)),
            "failure_latched": bool(self._p005_failure_latched),
            "authored_reset_pose": authored_pose_report,
        }

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        parent = super().accept_stable_recording_origin(report)
        payload = dict(parent) if isinstance(parent, Mapping) else {}
        actual = self._validate_task_initial_state_before_ready()
        if actual is None:
            raise RuntimeError("P0-05 stable origin lacks actual-state evidence")
        payload.update(
            {
                "accepted": True,
                "kind": "p005_awake_key_support_and_actual_robot_boundary",
                "initial_state": dict(actual),
            }
        )
        return payload

    @staticmethod
    def _view_numpy(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(shape)

    def _grouped_dofs(self) -> dict[int, tuple[Any, dict[str, int]]]:
        grouped: dict[int, tuple[Any, dict[str, int]]] = {}
        for name, (view, index) in self._p005_dof_views.items():
            key = id(view)
            if key not in grouped:
                grouped[key] = (view, {})
            grouped[key][1][name] = int(index)
        return grouped

    def _set_task_drives(self, *, shackle_locked: bool, force: bool = False) -> None:
        if self._torch is None:
            self._p005_shackle_locked = bool(shackle_locked)
            return
        if (
            not force
            and self._p005_shackle_locked == bool(shackle_locked)
            and abs(self._p005_last_drive_target_rad - self._p005_lock_target_rad) < 1e-4
        ):
            return
        for view, indices_by_name in self._grouped_dofs().values():
            stiffness = view.get_dof_stiffnesses().clone()
            damping = view.get_dof_dampings().clone()
            max_force = view.get_dof_max_forces().clone()
            targets = view.get_dof_position_targets().clone()
            cylinder = indices_by_name.get(LOCK_CYLINDER_DOF)
            if cylinder is not None:
                stiffness[:, cylinder] = LOCK_CYLINDER_STIFFNESS
                damping[:, cylinder] = LOCK_CYLINDER_DAMPING
                max_force[:, cylinder] = LOCK_CYLINDER_MAX_FORCE
                targets[:, cylinder] = float(self._p005_lock_target_rad)
            for name, locked_stiffness in (
                (SHACKLE_SLIDE_DOF, LOCKED_SHACKLE_SLIDE_STIFFNESS),
                (SHACKLE_HINGE_DOF, LOCKED_SHACKLE_HINGE_STIFFNESS),
            ):
                index = indices_by_name.get(name)
                if index is None:
                    continue
                if shackle_locked:
                    stiffness[:, index] = locked_stiffness
                    damping[:, index] = LOCKED_SHACKLE_DAMPING
                    max_force[:, index] = LOCKED_SHACKLE_MAX_FORCE
                    targets[:, index] = 0.0
                else:
                    stiffness[:, index] = (
                        UNLOCKED_SHACKLE_SLIDE_STIFFNESS
                        if name == SHACKLE_SLIDE_DOF
                        else UNLOCKED_SHACKLE_HINGE_STIFFNESS
                    )
                    damping[:, index] = UNLOCKED_SHACKLE_DAMPING
                    max_force[:, index] = UNLOCKED_SHACKLE_MAX_FORCE
                    targets[:, index] = (
                        UNLOCKED_SHACKLE_TARGET_M
                        if name == SHACKLE_SLIDE_DOF
                        else 0.0
                    )

            def _indices_like(value: Any):
                return self._torch.arange(
                    int(view.count), dtype=self._torch.long, device=value.device
                )

            view.set_dof_stiffnesses(stiffness, _indices_like(stiffness))
            view.set_dof_dampings(damping, _indices_like(damping))
            view.set_dof_max_forces(max_force, _indices_like(max_force))
            view.set_dof_position_targets(targets, _indices_like(targets))
        self._p005_shackle_locked = bool(shackle_locked)
        self._p005_last_drive_target_rad = float(self._p005_lock_target_rad)

    def reset(self, seed: int | None = None, **kwargs: Any):
        if seed is not None:
            self._p005_reset_serial = -1
        result = super().reset(seed=seed, **kwargs)
        # Scene initialization may finalize a different PhysX-compatible
        # control interval.  Reapply the action-count contract afterwards.
        self._time_limit = self._p005_timeout_steps * float(self._control_dt)
        return result

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._p005_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._p005_reset_serial
        hand_centers = (
            np.asarray(self._right_tcp_pose[:3], dtype=np.float64),
            np.asarray(self._left_tcp_pose[:3], dtype=np.float64),
        )
        self._p005_reset = sample_reset_state(
            episode_seed,
            forbidden_centers=hand_centers,
        )
        self._p005_tracker.reset()
        self._p005_last_readback = None
        self._p005_last_metrics = {}
        self._p005_failure_latched = False
        self._p005_failure_streak_substeps = 0
        self._p005_failure_snapshot = None
        self._p005_lock_target_rad = 0.0
        self._p005_last_drive_target_rad = -math.inf
        self._p005_shackle_locked = None

    def _set_key_material_friction(self) -> None:
        if self._stage is None:
            return
        try:
            from pxr import UsdPhysics

            material = self._stage.GetPrimAtPath(_KEY_MATERIAL_PATH)
            api = UsdPhysics.MaterialAPI(material)
            api.GetStaticFrictionAttr().Set(float(self._p005_reset.key_friction))
            api.GetDynamicFrictionAttr().Set(
                float(max(0.45, self._p005_reset.key_friction - 0.10))
            )
        except Exception as exc:
            self._add_task_state_warning(f"P0-05 key friction update failed: {exc}")

    def _apply_task_pose_randomization(self) -> None:
        if self._torch is None:
            return
        key_entry = next(
            (
                value
                for value in self._task_reset_body_states
                if KEY_BODY_NAME in str(value[0])
            ),
            None,
        )
        padlock_entry = next(
            (
                value
                for value in self._task_reset_articulation_states
                if PADLOCK_BODY_NAME in str(value[0])
            ),
            None,
        )
        if key_entry is None or padlock_entry is None:
            missing = []
            if key_entry is None:
                missing.append("key")
            if padlock_entry is None:
                missing.append("padlock articulation")
            self._add_task_state_warning(
                "P0-05 reset state is unavailable: " + ", ".join(missing)
            )
            return

        path, view, indices, root_pose, root_velocity = key_entry
        pose = root_pose.clone()
        pose[:, 0] = float(self._p005_reset.key_x_m)
        pose[:, 1] = float(self._p005_reset.key_y_m)
        pose[:, 2] = float(self._p005_reset.key_z_m)
        key_quaternion = key_reset_quaternion_wxyz(
            self._p005_reset.key_yaw_deg
        )
        # Contract quaternions are wxyz; PhysX root poses use xyzw.
        pose[:, 3] = float(key_quaternion[1])
        pose[:, 4] = float(key_quaternion[2])
        pose[:, 5] = float(key_quaternion[3])
        pose[:, 6] = float(key_quaternion[0])
        velocity = self._torch.zeros_like(root_velocity)
        view.set_transforms(pose, indices)
        view.set_velocities(velocity, indices)
        self._record_authored_task_reset_write(
            path, transform_xyzw=pose, velocity=velocity
        )

        (
            padlock_path,
            padlock_view,
            padlock_indices,
            padlock_root_pose,
            padlock_root_velocity,
            padlock_joint_pos,
            padlock_joint_velocity,
        ) = padlock_entry
        padlock_pose_np = self._view_numpy(
            padlock_root_pose, (int(padlock_view.count), 7)
        ).copy()
        padlock_pose_np[:, 0] = float(self._p005_reset.padlock_x_m)
        padlock_pose_np[:, 1] = float(self._p005_reset.padlock_y_m)
        padlock_pose_np[:, 2] = float(self._p005_reset.padlock_z_m)
        padlock_pose_np[:, 3:7] = np.asarray(
            [0.0, 0.0, 0.0, 1.0], dtype=padlock_pose_np.dtype
        )
        padlock_pose = self._torch.as_tensor(
            padlock_pose_np,
            dtype=padlock_root_pose.dtype,
            device=padlock_root_pose.device,
        )
        padlock_root_velocity = self._torch.zeros_like(padlock_root_velocity)
        padlock_joint_pos = self._torch.zeros_like(padlock_joint_pos)
        padlock_joint_velocity = self._torch.zeros_like(padlock_joint_velocity)
        padlock_view.set_root_transforms(padlock_pose, padlock_indices)
        padlock_view.set_root_velocities(
            padlock_root_velocity, padlock_indices
        )
        padlock_view.set_dof_positions(padlock_joint_pos, padlock_indices)
        padlock_view.set_dof_velocities(
            padlock_joint_velocity, padlock_indices
        )
        self._record_authored_task_reset_write(
            padlock_path,
            transform_xyzw=padlock_pose,
            velocity=padlock_root_velocity,
            dof_position=padlock_joint_pos,
            dof_velocity=padlock_joint_velocity,
        )
        self._set_key_material_friction()
        self._set_task_drives(shackle_locked=True, force=True)
        # Tensor setters enqueue root/DOF writes.  Flush them before the first
        # physics step so a newly converted or rebuilt USD cannot retain the
        # previous cache's authored key/padlock pose during the ready audit.
        if self._sim is not None:
            self._sim.forward()
        hand_centers = (
            np.asarray(self._right_tcp_pose[:2], dtype=np.float64),
            np.asarray(self._left_tcp_pose[:2], dtype=np.float64),
        )
        key_xy = np.asarray(
            [self._p005_reset.key_x_m, self._p005_reset.key_y_m],
            dtype=np.float64,
        )
        padlock_xy = np.asarray(
            [self._p005_reset.padlock_x_m, self._p005_reset.padlock_y_m],
            dtype=np.float64,
        )
        key_hand_clearance = min(
            float(np.linalg.norm(key_xy - center)) for center in hand_centers
        )
        padlock_hand_clearance = min(
            float(np.linalg.norm(padlock_xy - center))
            for center in hand_centers
        )
        print(
            "[HandsOn/P005] seeded reset: "
            f"seed={self._p005_reset.seed}, "
            f"key_xy=({self._p005_reset.key_x_m:+.4f},"
            f"{self._p005_reset.key_y_m:+.4f})m, "
            f"padlock_xy=({self._p005_reset.padlock_x_m:+.4f},"
            f"{self._p005_reset.padlock_y_m:+.4f})m, "
            f"yaw={self._p005_reset.key_yaw_deg:+.2f}deg, "
            f"mu={self._p005_reset.key_friction:.3f}, "
            f"hand_clearance=({key_hand_clearance:.3f},"
            f"{padlock_hand_clearance:.3f})m",
            flush=True,
        )

    def _dof_reading(self, name: str) -> tuple[float, float]:
        entry = self._p005_dof_views.get(name)
        if entry is None:
            return math.nan, math.nan
        view, index = entry
        positions = self._view_numpy(view.get_dof_positions(), (int(view.count), -1))
        velocities = self._view_numpy(view.get_dof_velocities(), (int(view.count), -1))
        return float(positions[0, index]), float(velocities[0, index])

    def _body_pose_wxyz(self, name: str) -> np.ndarray:
        view = self._p005_body_views.get(name)
        if view is None:
            return np.zeros(7, dtype=np.float64)
        transform = self._view_numpy(view.get_transforms()[0], (7,))
        return np.concatenate([transform[:3], transform[6:7], transform[3:6]])

    def _body_velocity(self, name: str) -> np.ndarray:
        view = self._p005_body_views.get(name)
        if view is None:
            return np.zeros(6, dtype=np.float64)
        return self._view_numpy(view.get_velocities()[0], (6,))

    def _contact_reading(self, key: str) -> tuple[float, float]:
        view = self._p005_contact_views.get(key)
        if view is None:
            return 0.0, 0.0
        try:
            normal_forces, _points, _normals, separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )
            forces = self._view_numpy(normal_forces, (-1,))
            separations = self._view_numpy(separations, (-1,))
            counts = self._view_numpy(
                counts, (int(view.sensor_count), int(view.filter_count))
            ).astype(np.int64)
            starts = self._view_numpy(starts, counts.shape).astype(np.int64)
            total_force = 0.0
            max_penetration = 0.0
            for count, start in zip(counts.ravel(), starts.ravel()):
                if count <= 0:
                    continue
                stop = int(start + count)
                total_force += float(np.sum(np.abs(forces[int(start):stop])))
                max_penetration = max(
                    max_penetration,
                    float(np.max(np.maximum(0.0, -separations[int(start):stop]))),
                )
            return total_force, max_penetration
        except Exception as exc:
            self._add_task_state_warning(f"P0-05 contact readback {key} failed: {exc}")
            return 0.0, math.inf

    def _key_geometry(self) -> tuple[float, float, float, float]:
        key_pose = self._body_pose_wxyz(KEY_BODY_NAME)
        padlock_pose = self._body_pose_wxyz(PADLOCK_BODY_NAME)
        if not np.isfinite(key_pose).all() or not np.isfinite(padlock_pose).all():
            return -math.inf, math.inf, math.inf, 0.0
        key_rotation = _rotation_from_quat_wxyz(key_pose[3:7])
        padlock_rotation = _rotation_from_quat_wxyz(padlock_pose[3:7])
        key_tip = key_pose[:3] + key_rotation @ _KEY_TIP_LOCAL_M
        key_axis = key_rotation @ _KEY_AXIS_LOCAL
        keyhole = padlock_pose[:3] + padlock_rotation @ _KEYHOLE_LOCAL_M
        # The cylinder is on the padlock's local +X side face.  The key enters
        # toward the body along local -X; its local -Y shaft axis is aligned
        # with that inward normal.
        inward = padlock_rotation @ _KEYWAY_INWARD_LOCAL
        displacement = key_tip - keyhole
        depth = float(np.dot(displacement, inward))
        lateral = float(np.linalg.norm(displacement - depth * inward))
        axis_error = float(angle_between_degrees(key_axis, inward))
        key_reference = key_rotation @ np.asarray([0.0, 0.0, 1.0])
        lock_reference = padlock_rotation @ np.asarray([0.0, 0.0, 1.0])
        # Project onto the plane normal to the keyway so small insertion-axis
        # errors cannot bias the measured roll. Positive about inward -X is
        # clockwise when looking into the visible +X side face.
        key_reference -= np.dot(key_reference, inward) * inward
        lock_reference -= np.dot(lock_reference, inward) * inward
        signed_turn_deg = math.degrees(
            math.atan2(
                float(np.dot(np.cross(lock_reference, key_reference), inward)),
                float(np.dot(lock_reference, key_reference)),
            )
        )
        # Positive key rotation is clockwise as viewed from the side face.
        return depth, lateral, axis_error, float(max(0.0, signed_turn_deg))

    def _physical_readback(self) -> P005PhysicalReadback:
        depth, lateral, axis_error, key_turn_deg = self._key_geometry()
        cylinder, cylinder_speed = self._dof_reading(LOCK_CYLINDER_DOF)
        slide, slide_speed = self._dof_reading(SHACKLE_SLIDE_DOF)
        hinge, hinge_speed = self._dof_reading(SHACKLE_HINGE_DOF)
        key_velocity = self._body_velocity(KEY_BODY_NAME)
        padlock_velocity = self._body_velocity(PADLOCK_BODY_NAME)
        shackle_velocity = self._body_velocity(SHACKLE_BODY_NAME)
        key_pose = self._body_pose_wxyz(KEY_BODY_NAME)
        padlock_pose = self._body_pose_wxyz(PADLOCK_BODY_NAME)
        readings = {
            key: self._contact_reading(key)
            for key in (
                "right_key",
                "left_padlock",
                "left_cylinder",
                "key_lock",
                "shackle_lock",
            )
        }
        finite = bool(
            np.isfinite(
                [
                    *key_pose,
                    *key_velocity,
                    *padlock_pose,
                    *padlock_velocity,
                    *shackle_velocity,
                    cylinder,
                    cylinder_speed,
                    slide,
                    slide_speed,
                    hinge,
                    hinge_speed,
                    key_turn_deg,
                ]
            ).all()
        )
        return P005PhysicalReadback(
            key_insertion_depth_m=depth,
            key_lateral_error_m=lateral,
            key_axis_error_deg=axis_error,
            lock_cylinder_angle_deg=math.degrees(max(0.0, cylinder)),
            shackle_slide_m=max(0.0, slide),
            shackle_hinge_deg=math.degrees(max(0.0, hinge)),
            shackle_linear_speed_mps=float(np.linalg.norm(shackle_velocity[:3])),
            shackle_angular_speed_radps=float(np.linalg.norm(shackle_velocity[3:])),
            key_linear_speed_mps=float(np.linalg.norm(key_velocity[:3])),
            key_angular_speed_radps=float(np.linalg.norm(key_velocity[3:])),
            key_bottom_z_m=float(key_pose[2] - _KEY_THICKNESS_HALF_M),
            padlock_lift_m=float(
                max(0.0, padlock_pose[2] - float(self._p005_reset.padlock_z_m))
            ),
            right_key_contact_force_n=readings["right_key"][0],
            left_padlock_contact_force_n=(
                readings["left_padlock"][0] + readings["left_cylinder"][0]
            ),
            left_shackle_contact_force_n=0.0,
            right_shackle_contact_force_n=0.0,
            # Retain the recorded-state slot for compatibility, but do not
            # create or read the former all-robot/table contact view. Table
            # contact is recoverable operator motion, not a task failure.
            robot_table_contact_force_n=0.0,
            # Hand contact is an expected part of the task. Do not turn a
            # fingertip/key or fingertip/padlock shell overlap into a reset;
            # only task-object pairs are safety-failure penetration gates.
            max_penetration_m=max(
                readings["key_lock"][1], readings["shackle_lock"][1]
            ),
            finite_state=finite,
        )

    @staticmethod
    def _readback_metrics(readback: P005PhysicalReadback) -> dict[str, float]:
        return {
            "key_insertion_depth_m": readback.key_insertion_depth_m,
            "key_lateral_error_m": readback.key_lateral_error_m,
            "key_axis_error_deg": readback.key_axis_error_deg,
            "lock_cylinder_angle_deg": readback.lock_cylinder_angle_deg,
            "shackle_slide_m": readback.shackle_slide_m,
            "shackle_hinge_deg": readback.shackle_hinge_deg,
            "shackle_linear_speed_mps": readback.shackle_linear_speed_mps,
            "right_key_contact_force_n": readback.right_key_contact_force_n,
            "left_padlock_contact_force_n": readback.left_padlock_contact_force_n,
            "padlock_lift_m": readback.padlock_lift_m,
            "left_shackle_contact_force_n": readback.left_shackle_contact_force_n,
            "right_shackle_contact_force_n": readback.right_shackle_contact_force_n,
            "max_penetration_m": readback.max_penetration_m,
        }

    def _capture_first_failure(
        self, readback: P005PhysicalReadback, metrics: dict[str, Any]
    ) -> None:
        if self._p005_failure_snapshot is not None or not metrics.get("failed"):
            return
        self._p005_failure_snapshot = {
            "control_step": int(getattr(self, "env_step", -1)),
            "physics_substep": int(
                getattr(self, "_trajectory_action_phase_substep_index", -1)
            ),
            "key_linear_speed_mps": float(readback.key_linear_speed_mps),
            "key_angular_speed_radps": float(readback.key_angular_speed_radps),
            "shackle_linear_speed_mps": float(
                readback.shackle_linear_speed_mps
            ),
            "shackle_angular_speed_radps": float(
                readback.shackle_angular_speed_radps
            ),
            "key_bottom_z_m": float(readback.key_bottom_z_m),
            "max_penetration_m": float(readback.max_penetration_m),
            "right_key_contact_force_n": float(
                readback.right_key_contact_force_n
            ),
            "left_padlock_contact_force_n": float(
                readback.left_padlock_contact_force_n
            ),
            "preunlock_shackle_motion": bool(
                metrics.get("preunlock_shackle_motion", False)
            ),
            "preunlock_shackle_motion_steps": int(
                metrics.get("preunlock_shackle_motion_steps", 0)
            ),
            "finite": bool(metrics.get("finite", False)),
            "exploded": bool(metrics.get("exploded", False)),
            "penetration_ok": bool(metrics.get("penetration_ok", False)),
            "key_not_dropped": bool(metrics.get("key_not_dropped", False)),
        }

    def _update_failure_latch(
        self, readback: P005PhysicalReadback, metrics: dict[str, Any]
    ) -> None:
        """Debounce recoverable contact spikes before ending the episode."""

        raw_failure = bool(metrics.get("failed", False))
        if not bool(metrics.get("finite", False)):
            # Non-finite state is not recoverable and remains an immediate
            # hard failure.
            self._p005_failure_streak_substeps = FAILURE_DWELL_SUBSTEPS
        elif raw_failure:
            self._p005_failure_streak_substeps += 1
        else:
            self._p005_failure_streak_substeps = 0

        should_latch = bool(
            self._p005_failure_streak_substeps >= FAILURE_DWELL_SUBSTEPS
        )
        if should_latch and not self._p005_failure_latched:
            self._capture_first_failure(readback, metrics)
        self._p005_failure_latched |= should_latch
        metrics["failure_detected_raw"] = raw_failure
        metrics["failure_streak_substeps"] = int(
            self._p005_failure_streak_substeps
        )
        metrics["failed"] = bool(self._p005_failure_latched)

    def _refresh_metrics(self, *, update_tracker: bool) -> dict[str, Any]:
        self._p005_last_readback = self._physical_readback()
        previous_unlocked = self._p005_tracker.shackle_unlock_allowed
        if update_tracker:
            self._p005_tracker.update(self._p005_last_readback)
        if self._p005_tracker.shackle_unlock_allowed != previous_unlocked:
            self._set_task_drives(
                shackle_locked=not self._p005_tracker.shackle_unlock_allowed,
                force=True,
            )
        self._p005_last_metrics = evaluate_physical_readback(
            self._p005_last_readback, self._p005_tracker
        )
        self._p005_last_metrics.update(
            self._readback_metrics(self._p005_last_readback)
        )
        self._p005_last_metrics["key_physical_turn_deg"] = self._key_geometry()[3]
        self._p005_last_metrics["failure_detected_raw"] = bool(
            self._p005_last_metrics.get("failed", False)
        )
        self._p005_last_metrics["failure_streak_substeps"] = int(
            self._p005_failure_streak_substeps
        )
        self._p005_last_metrics["failed"] = bool(self._p005_failure_latched)
        return self._p005_last_metrics

    def _after_task_physics_substep(self) -> None:
        if not self._p005_contact_contract_ready:
            return
        readback = self._physical_readback()
        # A low-force servo mirrors only a physically owned, inserted key's
        # roll into the cylinder.  It cannot progress from endpoint pose alone.
        if (
            readback.right_key_contact_force_n >= REQUIRED_CONTACT_FORCE_N
            and readback.left_padlock_contact_force_n >= REQUIRED_CONTACT_FORCE_N
            and readback.padlock_lift_m >= PADLOCK_LIFT_MIN_M
            and cylinder_coupling_geometry_ok(readback)
        ):
            key_turn_deg = self._key_geometry()[3]
            self._p005_lock_target_rad = math.radians(
                float(np.clip(key_turn_deg, 0.0, 80.0))
            )
            self._set_task_drives(
                shackle_locked=not self._p005_tracker.shackle_unlock_allowed
            )
        self._p005_last_readback = readback
        self._p005_last_metrics = evaluate_physical_readback(
            readback, self._p005_tracker
        )
        self._p005_last_metrics.update(self._readback_metrics(readback))
        self._p005_last_metrics["key_physical_turn_deg"] = self._key_geometry()[3]
        self._update_failure_latch(readback, self._p005_last_metrics)

    def _compute_success_now(self) -> bool:
        metrics = self._refresh_metrics(update_tracker=True)
        return bool(metrics["success_now"] and not self._p005_failure_latched)

    def _compute_failure(self) -> bool:
        return bool(self._p005_failure_latched)

    def _success_steps_required(self) -> int:
        return 1

    def _task_info(self) -> dict[str, Any]:
        metrics = dict(self._p005_last_metrics)
        metrics.update(
            {
                "p005_task_id": "P0-05",
                "p005_progress_blockers": self._progress_blockers(),
                "p005_reset_seed": int(self._p005_reset.seed),
                "p005_reset_friction": float(self._p005_reset.key_friction),
                "p005_shackle_locked": bool(self._p005_shackle_locked),
                "p005_internal_success_hold_steps": SUCCESS_HOLD_STEPS,
                "p005_robot_table_contact_gating_enabled": False,
                "p005_failure_snapshot": self._p005_failure_snapshot,
                "p005_initial_state_report": dict(
                    self._p005_initial_state_report
                ),
            }
        )
        return metrics

    def _progress_blockers(self) -> list[str]:
        """Operator diagnostics only; never replace the physical stage tracker."""
        m = self._p005_last_metrics
        if m.get("failed", False):
            return ["failed - reset required"]
        if m.get("success", False):
            return []
        blockers = []
        for name, label in (("left_padlock_contact_force_n", "left lock contact"),
                            ("right_key_contact_force_n", "right key contact")):
            if float(m.get(name, 0.0)) < REQUIRED_CONTACT_FORCE_N:
                blockers.append(label)
        if float(m.get("padlock_lift_m", 0.0)) < PADLOCK_LIFT_MIN_M:
            blockers.append("lift lock >= 35 mm")
        depth_limit = (KEY_INSERTION_RELEASE_DEPTH_M if m.get("key_inserted_seen")
                       else KEY_INSERTION_DEPTH_M)
        if float(m.get("key_insertion_depth_m", -math.inf)) < depth_limit:
            blockers.append(f"insert >= {depth_limit * 1000:.0f} mm")
        if float(m.get("key_lateral_error_m", math.inf)) > KEY_LATERAL_ERROR_M:
            blockers.append("center key in slot")
        if float(m.get("key_axis_error_deg", math.inf)) > KEY_AXIS_ERROR_DEG:
            blockers.append("align shaft <= 12 deg")
        if blockers:
            return blockers
        if not m.get("key_inserted_seen"):
            return ["hold insertion"]
        if not m.get("key_turned_seen"):
            return ["turn clockwise >= 55 deg (looking into keyhole)"]
        return ["keep holding until shackle is open and stable"]

    def _task_status_text(self) -> str:
        m = self._p005_last_metrics
        return (
            f"Stage {m.get('phase', 'seek_key')} | "
            f"Lift {float(m.get('padlock_lift_m', 0.0)) * 1000:.1f}/35 mm | "
            f"L/R contact {float(m.get('left_padlock_contact_force_n', 0.0)):.2f}/"
            f"{float(m.get('right_key_contact_force_n', 0.0)):.2f} N\n"
            f"Insert {float(m.get('key_insertion_depth_m', 0.0)) * 1000:.1f} mm | "
            f"Offset {float(m.get('key_lateral_error_m', 0.0)) * 1000:.1f}/8.0 mm | "
            f"Axis {float(m.get('key_axis_error_deg', 0.0)):.1f}/12 deg\n"
            f"Key/Cylinder {float(m.get('key_physical_turn_deg', 0.0)):.1f}/"
            f"{float(m.get('lock_cylinder_angle_deg', 0.0)):.1f} deg | "
            f"Shackle {float(m.get('shackle_slide_m', 0.0)) * 1000:.1f} mm\n"
            f"Need: {', '.join(self._progress_blockers()) or 'complete'}"
        )

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._p005_last_metrics
        return [
            self._condition_row(
                "p005_lift_insert",
                "Left hand holds/lifts lock >= 35 mm; right key inserts >= 12 mm",
                bool(metrics.get("key_inserted_seen", False)),
            ),
            self._condition_row(
                "p005_turn",
                "Keep both objects held; turn key clockwise >= 55 degrees",
                bool(metrics.get("key_turned_seen", False)),
            ),
            self._condition_row(
                "p005_open",
                "Unlocked U-shaped shackle pops up >= 16 mm and stabilizes",
                bool(metrics.get("success", False)),
            ),
            self._condition_row(
                "p005_safety",
                "No early pull, key drop, excessive penetration, or instability",
                not bool(metrics.get("failed", False)),
            ),
        ]

    @staticmethod
    def _zero_readback() -> P005PhysicalReadback:
        return P005PhysicalReadback(
            key_insertion_depth_m=-math.inf,
            key_lateral_error_m=math.inf,
            key_axis_error_deg=math.inf,
            lock_cylinder_angle_deg=0.0,
            shackle_slide_m=0.0,
            shackle_hinge_deg=0.0,
            shackle_linear_speed_mps=0.0,
            shackle_angular_speed_radps=0.0,
            key_linear_speed_mps=0.0,
            key_angular_speed_radps=0.0,
            key_bottom_z_m=TABLE_TOP_Z_M,
            padlock_lift_m=0.0,
            right_key_contact_force_n=0.0,
            left_padlock_contact_force_n=0.0,
            left_shackle_contact_force_n=0.0,
            right_shackle_contact_force_n=0.0,
            robot_table_contact_force_n=0.0,
            max_penetration_m=0.0,
            finite_state=True,
        )

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        if self._p005_last_readback is None and self._p005_contact_contract_ready:
            self._refresh_metrics(update_tracker=False)
        readback = self._p005_last_readback or self._zero_readback()
        metrics = self._p005_last_metrics
        state = observation["state"]
        state.update(
            {
                "p005_key_pose": self._body_pose_wxyz(KEY_BODY_NAME),
                "p005_padlock_pose": self._body_pose_wxyz(PADLOCK_BODY_NAME),
                "p005_shackle_pose": self._body_pose_wxyz(SHACKLE_BODY_NAME),
                "p005_joint_state": np.asarray(
                    [
                        math.radians(readback.lock_cylinder_angle_deg),
                        readback.shackle_slide_m,
                        math.radians(readback.shackle_hinge_deg),
                        readback.key_insertion_depth_m,
                        readback.key_lateral_error_m,
                        readback.key_axis_error_deg,
                    ],
                    dtype=np.float64,
                ),
                "p005_role_contacts": np.asarray(
                    [
                        readback.right_key_contact_force_n,
                        readback.left_padlock_contact_force_n,
                        readback.padlock_lift_m,
                        # Reserved compatibility slot. Robot/table contact is
                        # intentionally not sensed or failure-gated for P005.
                        readback.robot_table_contact_force_n,
                        readback.max_penetration_m,
                    ],
                    dtype=np.float64,
                ),
                "p005_task_state": np.asarray(
                    [
                        float(self._p005_tracker.phase),
                        float(self._p005_tracker.insertion_steps),
                        float(self._p005_tracker.turn_steps),
                        float(self._p005_tracker.success_steps),
                        float(self._p005_tracker.right_key_contact_seen),
                        float(self._p005_tracker.left_padlock_contact_seen),
                        float(self._p005_tracker.key_inserted_seen),
                        float(self._p005_tracker.key_turned_seen),
                        float(self._p005_tracker.shackle_open_seen),
                        float(metrics.get("failed", False)),
                    ],
                    dtype=np.float64,
                ),
            }
        )
        return observation

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        observation = super().restore_recorded_initial_state(state_parts)
        joint_state = np.asarray(
            state_parts.get("p005_joint_state", np.zeros(0)), dtype=np.float64
        ).ravel()
        if self._torch is None or joint_state.size < 3:
            return observation
        for view, indices_by_name in self._grouped_dofs().values():
            positions = view.get_dof_positions().clone()
            velocities = self._torch.zeros_like(view.get_dof_velocities())
            for offset, name in enumerate(
                (LOCK_CYLINDER_DOF, SHACKLE_SLIDE_DOF, SHACKLE_HINGE_DOF)
            ):
                index = indices_by_name.get(name)
                if index is not None:
                    positions[:, index] = float(joint_state[offset])
            indices = self._torch.arange(
                int(view.count), dtype=self._torch.long, device=positions.device
            )
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(velocities, indices)
        self._p005_tracker.reset()
        self._p005_lock_target_rad = float(max(0.0, joint_state[0]))
        self._set_task_drives(shackle_locked=True, force=True)
        if self._sim is not None:
            self._sim.forward()
        self._refresh_metrics(update_tracker=False)
        return self._compute_observation()


__all__ = ["HandsOnP005IsaacLabEnv"]
