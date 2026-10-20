"""Isaac Sim 5.1 collection environment for YCB08."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _matrix_to_quat_wxyz,
    _pose_vector_to_matrix,
)
from .ycb08_semantics import (
    YCB08_BODY_NAMES,
    YCB08_BOWL_CENTER_XY_BOUNDS_M,
    YCB08_CONTACT_FORCE_FLOOR_N,
    YCB08_DIRECT_HAND_DEBOUNCE_SECONDS,
    YCB08_EVENT_FLAG_NAMES,
    YCB08_EXCESSIVE_VELOCITY_DEBOUNCE_SECONDS,
    YCB08_FAILURE_CODES,
    YCB08_FINAL_FLIP_ANGLE_DEG,
    YCB08_FINAL_STABILITY_SECONDS,
    YCB08_LEFT_PAN_LOCK_DEBOUNCE_SECONDS,
    YCB08_LEFT_PAN_LOCK_MAX_ANGULAR_SPEED_RADPS,
    YCB08_LEFT_PAN_LOCK_MAX_LINEAR_SPEED_MPS,
    YCB08_LEFT_PAN_LOCK_MAX_TILT_DEG,
    YCB08_LEFT_PAN_LOCK_MIN_LIFT_M,
    YCB08_LEFT_PAN_UNLOCK_DEBOUNCE_SECONDS,
    YCB08_MAX_FINAL_ANGULAR_SPEED_RADPS,
    YCB08_MAX_FINAL_LINEAR_SPEED_MPS,
    YCB08_MAX_FINAL_PAN_TILT_DEG,
    YCB08_MAX_PAN_TILT_DURING_FLIP_DEG,
    YCB08_MIN_BLADE_OVERLAP,
    YCB08_MIN_SPONGE_LIFT_M,
    YCB08_OUTSIDE_SUPPORT_DEBOUNCE_SECONDS,
    YCB08_PAN_OVERTURN_ANGLE_DEG,
    YCB08_PAN_OVERTURN_DEBOUNCE_SECONDS,
    YCB08_SETTLE_SECONDS,
    YCB08_SKILLET_NAME,
    YCB08_SPATULA_CENTER_OF_MASS_LOCAL_M,
    YCB08_SPATULA_DIAGONAL_INERTIA_KG_M2,
    YCB08_SPATULA_FACE_UP_PROBABILITY,
    YCB08_SPATULA_MASS_KG,
    YCB08_SPATULA_CENTER_XY_BOUNDS_M,
    YCB08_SPATULA_NAME,
    YCB08_SPATULA_YAW_RANGE_DEG,
    YCB08_SPONGE_NAME,
    YCB08_TASK_ID,
    YCB08_TASK_NAME,
    YCB08_TOOL_MOVE_DISTANCE_M,
    YCB08_TOOL_RELEASE_SECONDS,
    YCB08_PAN_YAW_RANGE_DEG,
    YCB08_WORKSPACE_XY_BOUNDS,
    YCB08_WORKSPACE_Z_RANGE_M,
    initial_ycb08_left_pan_lock_state,
    initial_ycb08_state,
    quaternion_wxyz_to_matrix,
    sample_ycb08_layout,
    update_ycb08_left_pan_lock_state,
    update_ycb08_state,
    ycb08_blade_overlap_metrics,
    ycb08_event_flags,
    ycb08_final_checks,
    ycb08_flip_angle_degrees,
    ycb08_hidden_state,
    ycb08_locked_teleop_action,
    ycb08_pan_containment_metrics,
    ycb08_pan_tilt_degrees,
    ycb08_spatula_is_face_up,
)


_YCB08_RESET_LINEAR_DAMPING = 0.12
_YCB08_RESET_ANGULAR_DAMPING = 0.80
_YCB08_RESET_SLEEP_THRESHOLD = 0.005
# The inverted ready pose deliberately exposes the thin leading edge. Give
# just the spatula realistic rotational drag and cap overlap correction so a
# first fingertip contact cannot solver-snap it onto the other face.
_YCB08_SPATULA_ANGULAR_DAMPING = 1.50
_YCB08_SPATULA_MAX_DEPENETRATION_VELOCITY_MPS = 1.00
# Keep the exact imported collision meshes, but give their support contacts a
# narrow positive rest shell.  The skillet's thin faceted floor otherwise
# lets the sponge settle about 0.8 mm into its nominal surface, which exceeds
# the common 0.5 mm recording-origin penetration gate.
_YCB08_CONTACT_OFFSET_M = 0.0015
_YCB08_REST_OFFSET_M = 0.0005
# Only the four source blade hulls slide against the pan and sponge. The
# handle keeps its existing grip; no collision mesh or body pose is replaced.
_YCB08_BLADE_COLLIDER_NAMES = frozenset(
    f"ycb08_spatula_collision_{index:02d}" for index in range(36, 40)
)
_YCB08_BLADE_STATIC_FRICTION = 0.25
_YCB08_BLADE_DYNAMIC_FRICTION = 0.18
_YCB08_BLADE_CONTACT_OFFSET_M = 0.001
_YCB08_BLADE_REST_OFFSET_M = 0.0
_YCB08_BLADE_MATERIAL_PATH = "/World/PhysicsMaterials/YCB08Blade"
_YCB08_RIGHT_WRIST_CAMERA_LOCAL_ROTATION_DEG = (-30.0, 45.0, 0.0)
_YCB08_CORE_CONTACT_KEYS = (
    "sponge_spatula",
    "sponge_pan",
    "sponge_table",
    "sponge_hands",
    "spatula_right_hand",
)
_YCB08_LEFT_PAN_CONTACT_KEY = "skillet_left_hand"


def _axis_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if axis == "x":
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    if axis == "y":
        return np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    if axis == "z":
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported rotation axis: {axis!r}")


def _ycb08_wrist_camera_mount_pose(
    base_pose: np.ndarray,
    *,
    side: str,
    legacy: bool,
) -> np.ndarray:
    """Aim only YCB08's current-profile right wrist camera at the task area."""
    result = np.asarray(base_pose, dtype=np.float64).copy()
    if side != "right" or legacy:
        return result
    x_deg, y_deg, z_deg = _YCB08_RIGHT_WRIST_CAMERA_LOCAL_ROTATION_DEG
    delta = (
        _axis_rotation_matrix("x", x_deg)
        @ _axis_rotation_matrix("y", y_deg)
        @ _axis_rotation_matrix("z", z_deg)
    )
    matrix = _pose_vector_to_matrix(result)
    matrix[:3, :3] = matrix[:3, :3] @ delta
    result[3:7] = _matrix_to_quat_wxyz(matrix[:3, :3])
    return result


class YCB08SpatulaFlipSpongeIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Dual-hand tool-use task with a physical-state-gated left-pan hold."""

    def __init__(self, task_name: str = YCB08_TASK_NAME, **kwargs: Any) -> None:
        if task_name != YCB08_TASK_NAME:
            raise ValueError(f"YCB08 environment requires {YCB08_TASK_NAME!r}")
        self._ycb08_state: dict[str, Any] = initial_ycb08_state()
        self._ycb08_contact_views: dict[str, Any] = {}
        self._ycb08_contact_available: dict[str, bool] = {}
        self._ycb08_contact_forces: dict[str, float] = {}
        self._ycb08_sampled_reset_poses: dict[str, np.ndarray] = {}
        self._ycb08_settled_reset_poses: dict[str, np.ndarray] = {}
        self._ycb08_reset_face_normal_world = np.asarray(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        self._ycb08_reference_captured = False
        self._ycb08_last_metrics: dict[str, Any] = {}
        self._ycb08_reset_ordinal = 0
        self._ycb08_left_pan_lock_state = initial_ycb08_left_pan_lock_state()
        self._ycb08_left_pan_locked_hand: np.ndarray | None = None
        self._ycb08_left_pan_rebase_pending = False
        self._ycb08_left_pan_live_offset: tuple[np.ndarray, np.ndarray] | None = None
        super().__init__(task_name, **kwargs)

    def _recording_wrist_camera_mount_pose(self, side: str) -> np.ndarray:
        return _ycb08_wrist_camera_mount_pose(
            super()._recording_wrist_camera_mount_pose(side),
            side=side,
            legacy=bool(self._legacy_wrist_cameras),
        )

    def _build_observation_space(self) -> spaces.Dict:
        base = super()._build_observation_space()
        state_spaces = dict(base.spaces["state"].spaces)
        vector3 = lambda: spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float64
        )
        pose7 = lambda: spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float64
        )
        state_spaces.update(
            {
                "skillet_pose": pose7(),
                "skillet_linear_velocity": vector3(),
                "skillet_angular_velocity": vector3(),
                "spatula_pose": pose7(),
                "spatula_linear_velocity": vector3(),
                "spatula_angular_velocity": vector3(),
                "sponge_pose": pose7(),
                "sponge_linear_velocity": vector3(),
                "sponge_angular_velocity": vector3(),
                "sponge_in_pan_position": vector3(),
                "ycb08_geometry_metrics": spaces.Box(
                    -np.inf, np.inf, shape=(10,), dtype=np.float64
                ),
                "ycb08_contact_state": spaces.Box(
                    0.0, np.inf, shape=(6,), dtype=np.float64
                ),
                "ycb08_left_pan_lock_state": spaces.Box(
                    -np.inf, np.inf, shape=(10,), dtype=np.float64
                ),
                "ycb08_event_flags": spaces.Box(
                    0.0,
                    1.0,
                    shape=(len(YCB08_EVENT_FLAG_NAMES),),
                    dtype=np.float64,
                ),
                "ycb08_hidden_state": spaces.Box(
                    -np.inf, np.inf, shape=(12,), dtype=np.float64
                ),
                "ycb08_sampled_reset_poses": spaces.Box(
                    -np.inf, np.inf, shape=(3, 7), dtype=np.float64
                ),
            }
        )
        observation_spaces = dict(base.spaces)
        observation_spaces["state"] = spaces.Dict(state_spaces)
        return spaces.Dict(observation_spaces)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        # Keep the support plane fixed.  Every free YCB body still receives
        # independent, seeded, nonzero XY variation from the task sampler.
        self.delta_h[:] = 0.0
        self._right_tcp_pose[1] = -0.36
        self._left_tcp_pose[1] = 0.36
        self._ycb08_sampled_reset_poses = sample_ycb08_layout(
            self._random,
            randomize=self.randomize,
            table_top_z=0.89,
        )
        self._ycb08_state = initial_ycb08_state()
        self._ycb08_settled_reset_poses = {}
        self._ycb08_reference_captured = False
        self._ycb08_last_metrics = {}
        self._ycb08_left_pan_lock_state = initial_ycb08_left_pan_lock_state()
        self._ycb08_left_pan_locked_hand = None
        self._ycb08_left_pan_rebase_pending = False
        self._ycb08_left_pan_live_offset = None
        self._ycb08_contact_forces = {
            key: 0.0 for key in self._ycb08_contact_views
        }
        self._ycb08_reset_ordinal += 1

    def _apply_task_pose_randomization(self) -> None:
        if not self._task_reset_body_states and not self._task_reset_articulation_states:
            return
        if set(self._ycb08_sampled_reset_poses) != set(YCB08_BODY_NAMES):
            raise RuntimeError("YCB08 complete reset layout was not sampled")

        reset_state_by_path = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity in self._task_reset_body_states
        }
        reset_articulation_by_path = {
            path: (view, indices, root_pose, root_velocity)
            for (
                path,
                view,
                indices,
                root_pose,
                root_velocity,
                _joint_pos,
                _joint_vel,
            ) in self._task_reset_articulation_states
        }
        for name in YCB08_BODY_NAMES:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                raise RuntimeError(f"YCB08 reset body is unavailable: {name}")
            path = str(prim.GetPath())
            reset_state = self._reset_transform_state_for_path(
                path,
                reset_state_by_path,
                reset_articulation_by_path,
            )
            if reset_state is None:
                raise RuntimeError(f"YCB08 reset tensor is unavailable: {name}")
            state_kind, view, indices, root_pose, root_velocity = reset_state
            pose_wxyz = self._ycb08_sampled_reset_poses[name]
            pose_xyzw = np.concatenate(
                [pose_wxyz[:3], pose_wxyz[4:7], pose_wxyz[3:4]]
            ).reshape(1, 7)
            pose_xyzw = np.repeat(pose_xyzw, int(root_pose.shape[0]), axis=0)
            pose_tensor = self._torch.as_tensor(
                pose_xyzw,
                dtype=root_pose.dtype,
                device=root_pose.device,
            )
            zero_velocity = self._torch.zeros_like(root_velocity)
            if state_kind == "articulation":
                view.set_root_transforms(pose_tensor, indices)
                view.set_root_velocities(zero_velocity, indices)
                self._record_authored_task_reset_write(
                    path,
                    transform_xyzw=pose_tensor,
                    velocity=zero_velocity,
                )
            else:
                self._write_rigid_reset_state(
                    path,
                    view,
                    indices,
                    pose_tensor,
                    zero_velocity,
                )
        print(
            "[HandsOn/IsaacLab] YCB08 reset layout: "
            + ", ".join(
                f"{name}={pose[:2].round(4).tolist()}"
                for name, pose in self._ycb08_sampled_reset_poses.items()
            )
            + ", spatula_face="
            + (
                "up"
                if ycb08_spatula_is_face_up(
                    self._ycb08_sampled_reset_poses[YCB08_SPATULA_NAME]
                )
                else "down"
            ),
            flush=True,
        )

    def _configure_task_reset_stability(self, sim_utils: Any) -> int:
        configured = super()._configure_task_reset_stability(sim_utils)
        if self.task_name != YCB08_TASK_NAME or self._stage is None:
            return configured
        configured += self._configure_named_reset_stability(
            sim_utils,
            YCB08_BODY_NAMES,
            linear_damping=_YCB08_RESET_LINEAR_DAMPING,
            angular_damping=_YCB08_RESET_ANGULAR_DAMPING,
            sleep_threshold=_YCB08_RESET_SLEEP_THRESHOLD,
            warning_label="YCB08 source-scale YCB bodies",
            strip_redundant_articulation_root=True,
        )
        from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade

        blade_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=_YCB08_BLADE_STATIC_FRICTION,
            dynamic_friction=_YCB08_BLADE_DYNAMIC_FRICTION,
            friction_combine_mode="min",
            restitution=0.0,
            restitution_combine_mode="min",
        )
        blade_material_cfg.func(_YCB08_BLADE_MATERIAL_PATH, blade_material_cfg)
        blade_material = UsdShade.Material(
            self._stage.GetPrimAtPath(_YCB08_BLADE_MATERIAL_PATH)
        )

        for name in YCB08_BODY_NAMES:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            ccd = rigid_api.GetEnableCCDAttr()
            if not ccd or not ccd.IsValid():
                ccd = rigid_api.CreateEnableCCDAttr()
            ccd.Set(name in {YCB08_SPATULA_NAME, YCB08_SPONGE_NAME})
            if name == YCB08_SPATULA_NAME:
                # Always re-author these values on the live prim. Existing
                # MJCF-to-USD caches may still contain the old 51.5 g inertial
                # even after the canonical XML has been corrected.
                mass_api = UsdPhysics.MassAPI(prim)
                if not mass_api:
                    mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateMassAttr().Set(YCB08_SPATULA_MASS_KG)
                mass_api.CreateCenterOfMassAttr().Set(
                    Gf.Vec3f(
                        *[
                            float(value)
                            for value in YCB08_SPATULA_CENTER_OF_MASS_LOCAL_M
                        ]
                    )
                )
                mass_api.CreateDiagonalInertiaAttr().Set(
                    Gf.Vec3f(
                        *[
                            float(value)
                            for value in YCB08_SPATULA_DIAGONAL_INERTIA_KG_M2
                        ]
                    )
                )
                mass_api.CreatePrincipalAxesAttr().Set(
                    Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
                )
                rigid_api.CreateAngularDampingAttr().Set(
                    _YCB08_SPATULA_ANGULAR_DAMPING
                )
                rigid_api.CreateMaxDepenetrationVelocityAttr().Set(
                    _YCB08_SPATULA_MAX_DEPENETRATION_VELOCITY_MPS
                )
            for descendant in Usd.PrimRange(prim):
                if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
                contact = collision_api.GetContactOffsetAttr()
                if not contact or not contact.IsValid():
                    contact = collision_api.CreateContactOffsetAttr()
                blade = (
                    name == YCB08_SPATULA_NAME
                    and descendant.GetName() in _YCB08_BLADE_COLLIDER_NAMES
                )
                contact.Set(
                    _YCB08_BLADE_CONTACT_OFFSET_M if blade else _YCB08_CONTACT_OFFSET_M
                )
                rest = collision_api.GetRestOffsetAttr()
                if not rest or not rest.IsValid():
                    rest = collision_api.CreateRestOffsetAttr()
                rest.Set(_YCB08_BLADE_REST_OFFSET_M if blade else _YCB08_REST_OFFSET_M)
                if blade:
                    UsdShade.MaterialBindingAPI.Apply(descendant).Bind(
                        blade_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        materialPurpose="physics",
                    )
        return configured

    def _configure_task_specific_contact_reporting(self) -> int:
        """Author YCB08 task and left-pan-assist contact reporting."""
        configured = super()._configure_task_specific_contact_reporting()
        if self.task_name != YCB08_TASK_NAME or self._stage is None:
            return configured
        from pxr import PhysxSchema

        for name in (*YCB08_BODY_NAMES, "table"):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not configure YCB08 contact reporting for {name}"
                )
                continue
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage, prim.GetPath()
                )
            else:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)
            configured += 1
        return configured

    # Keep the pre-rename hook as a compatibility shim for focused callers
    # and older task harnesses.  The shared ABC startup invokes the canonical
    # ``_configure_task_specific_contact_reporting`` hook above.
    def _configure_task_contact_reporting(self) -> int:
        return self._configure_task_specific_contact_reporting()

    def _ycb08_hand_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        prefix = str(side).lower()
        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{prefix.capitalize()}Robot"
        )
        if root is None or not root.IsValid():
            return []
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (
                prim.GetName() == f"{prefix}_palm_link"
                or prim.GetName().startswith(f"{prefix}_finger")
            )
        ]

    def _init_task_contact_views(self) -> None:
        self._ycb08_contact_views = {}
        self._ycb08_contact_available = {}
        if self.task_name != YCB08_TASK_NAME or not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        prims = {
            name: self._find_rigid_body_prim(name)
            for name in (*YCB08_BODY_NAMES, "table")
        }
        left_paths = self._ycb08_hand_paths("left")
        right_paths = self._ycb08_hand_paths("right")
        if physics_view is None or any(prim is None for prim in prims.values()):
            self._add_task_state_warning(
                "YCB08 contact reporting is unavailable: task rigid bodies missing"
            )
            return

        def create(key: str, sensor_name: str, filters: list[str]) -> None:
            try:
                view = physics_view.create_rigid_contact_view(
                    str(prims[sensor_name].GetPath()),
                    filter_patterns=filters,
                    max_contact_data_count=max(96, len(filters) * 12),
                )
                available = bool(
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                )
                self._ycb08_contact_available[key] = available
                if available:
                    self._ycb08_contact_views[key] = view
                else:
                    self._add_task_state_warning(
                        f"YCB08 contact view {key} has no sensors or filters"
                    )
            except Exception as exc:
                self._ycb08_contact_available[key] = False
                self._add_task_state_warning(
                    f"Could not initialize YCB08 contact view {key}: {exc}"
                )

        create(
            "sponge_spatula",
            YCB08_SPONGE_NAME,
            [str(prims[YCB08_SPATULA_NAME].GetPath())],
        )
        create(
            "sponge_pan",
            YCB08_SPONGE_NAME,
            [str(prims[YCB08_SKILLET_NAME].GetPath())],
        )
        create(
            "sponge_table",
            YCB08_SPONGE_NAME,
            [str(prims["table"].GetPath())],
        )
        create("sponge_hands", YCB08_SPONGE_NAME, left_paths + right_paths)
        create("spatula_right_hand", YCB08_SPATULA_NAME, right_paths)
        create(_YCB08_LEFT_PAN_CONTACT_KEY, YCB08_SKILLET_NAME, left_paths)

    def _init_task_specific_runtime(self) -> None:
        """Create YCB08 contact views after the first PhysX reset."""
        super()._init_task_specific_runtime()
        self._init_task_contact_views()

    def _ycb08_contact_force(self, key: str) -> float:
        force = float(self._c04_contact_force(self._ycb08_contact_views.get(key)))
        self._ycb08_contact_forces[key] = force
        return force

    def _ycb08_pose(self, name: str) -> np.ndarray:
        if self._task_body_views.get(name) is None:
            sampled = self._ycb08_sampled_reset_poses.get(name)
            if sampled is not None:
                return sampled.copy()
        return self._poses.get(name, self._fallback_pose(name)).copy()

    def _ycb08_velocity(self, name: str) -> np.ndarray:
        velocity = self._physics_body_velocity(name)
        if velocity is None:
            return np.zeros(6, dtype=np.float64)
        return np.asarray(velocity, dtype=np.float64).reshape(6)

    def _ycb08_metrics(self, *, update_contacts: bool) -> dict[str, Any]:
        self._refresh_object_poses()
        pan = self._ycb08_pose(YCB08_SKILLET_NAME)
        spatula = self._ycb08_pose(YCB08_SPATULA_NAME)
        sponge = self._ycb08_pose(YCB08_SPONGE_NAME)
        velocities = {
            name: self._ycb08_velocity(name) for name in YCB08_BODY_NAMES
        }
        containment = ycb08_pan_containment_metrics(sponge, pan)
        blade = ycb08_blade_overlap_metrics(sponge, spatula)
        reset_sponge = self._ycb08_settled_reset_poses.get(
            YCB08_SPONGE_NAME,
            self._ycb08_sampled_reset_poses.get(YCB08_SPONGE_NAME, sponge),
        )
        reset_pan = self._ycb08_settled_reset_poses.get(
            YCB08_SKILLET_NAME,
            self._ycb08_sampled_reset_poses.get(YCB08_SKILLET_NAME, pan),
        )
        reset_spatula = self._ycb08_settled_reset_poses.get(
            YCB08_SPATULA_NAME,
            self._ycb08_sampled_reset_poses.get(YCB08_SPATULA_NAME, spatula),
        )
        if update_contacts:
            forces = {
                key: self._ycb08_contact_force(key)
                for key in (*_YCB08_CORE_CONTACT_KEYS, _YCB08_LEFT_PAN_CONTACT_KEY)
            }
        else:
            forces = {
                key: float(self._ycb08_contact_forces.get(key, 0.0))
                for key in (*_YCB08_CORE_CONTACT_KEYS, _YCB08_LEFT_PAN_CONTACT_KEY)
            }
        contact = {
            key: bool(
                self._ycb08_contact_available.get(key, False)
                and force >= YCB08_CONTACT_FORCE_FLOOR_N
            )
            for key, force in forces.items()
        }
        all_contact_views = bool(
            all(self._ycb08_contact_available.get(key, False) for key in _YCB08_CORE_CONTACT_KEYS)
        )
        left_pan_contact_available = bool(
            self._ycb08_contact_available.get(_YCB08_LEFT_PAN_CONTACT_KEY, False)
        )
        pan_velocity = velocities[YCB08_SKILLET_NAME]
        pan_linear_speed = float(np.linalg.norm(pan_velocity[:3]))
        pan_angular_speed = float(np.linalg.norm(pan_velocity[3:]))
        linear_speeds = {
            name: float(np.linalg.norm(value[:3]))
            for name, value in velocities.items()
        }
        angular_speeds = {
            name: float(np.linalg.norm(value[3:]))
            for name, value in velocities.items()
        }
        max_linear_speed_body = max(linear_speeds, key=linear_speeds.get)
        max_angular_speed_body = max(angular_speeds, key=angular_speeds.get)
        sponge_velocity = velocities[YCB08_SPONGE_NAME]
        sponge_linear_speed = float(np.linalg.norm(sponge_velocity[:3]))
        sponge_angular_speed = float(np.linalg.norm(sponge_velocity[3:]))
        pan_support_ok = bool(
            all_contact_views
            and contact["sponge_pan"]
            and not contact["sponge_table"]
            and not contact["sponge_spatula"]
            and not contact["sponge_hands"]
            and containment["rim_clear"]
            and containment["com_below_rim"]
            and -0.003 <= float(containment["floor_gap_m"]) <= 0.012
        )
        lower_xy, upper_xy = YCB08_WORKSPACE_XY_BOUNDS
        workspace_ok = all(
            np.all(pose[:2] >= lower_xy)
            and np.all(pose[:2] <= upper_xy)
            and YCB08_WORKSPACE_Z_RANGE_M[0] <= float(pose[2]) <= YCB08_WORKSPACE_Z_RANGE_M[1]
            for pose in (pan, spatula, sponge)
        )
        finite = bool(
            all(np.all(np.isfinite(value)) for value in (pan, spatula, sponge))
            and all(np.all(np.isfinite(value)) for value in velocities.values())
        )
        metrics = {
            **containment,
            **blade,
            "pan_pose": pan,
            "spatula_pose": spatula,
            "sponge_pose": sponge,
            "pan_velocity": pan_velocity,
            "spatula_velocity": velocities[YCB08_SPATULA_NAME],
            "sponge_velocity": sponge_velocity,
            "pan_tilt_deg": ycb08_pan_tilt_degrees(pan),
            "flip_angle_deg": ycb08_flip_angle_degrees(
                sponge, self._ycb08_reset_face_normal_world
            ),
            "sponge_lift_m": float(sponge[2] - reset_sponge[2]),
            "pan_lift_m": float(pan[2] - reset_pan[2]),
            "pan_linear_speed_mps": pan_linear_speed,
            "pan_angular_speed_radps": pan_angular_speed,
            "spatula_move_distance_m": float(
                np.linalg.norm(spatula[:3] - reset_spatula[:3])
            ),
            "sponge_linear_speed_mps": sponge_linear_speed,
            "sponge_angular_speed_radps": sponge_angular_speed,
            "max_linear_speed_mps": max(
                linear_speeds.values()
            ),
            "max_angular_speed_radps": max(
                angular_speeds.values()
            ),
            "max_linear_speed_body": max_linear_speed_body,
            "max_angular_speed_body": max_angular_speed_body,
            "spatula_sponge_contact": contact["sponge_spatula"],
            "pan_sponge_contact": contact["sponge_pan"],
            "sponge_table_contact": contact["sponge_table"],
            "direct_hand_sponge_contact": contact["sponge_hands"],
            "spatula_right_hand_contact": contact["spatula_right_hand"],
            "left_hand_pan_contact": contact[_YCB08_LEFT_PAN_CONTACT_KEY],
            "left_pan_contact_available": left_pan_contact_available,
            "pan_support_ok": pan_support_ok,
            "contact_views_available": all_contact_views,
            "contact_forces": forces,
            "workspace_ok": workspace_ok,
            "finite": finite,
        }
        return metrics

    def _capture_ycb08_settled_reference(self) -> None:
        self._refresh_object_poses()
        self._ycb08_settled_reset_poses = {
            name: self._ycb08_pose(name) for name in YCB08_BODY_NAMES
        }
        sponge = self._ycb08_settled_reset_poses[YCB08_SPONGE_NAME]
        self._ycb08_reset_face_normal_world = quaternion_wxyz_to_matrix(
            sponge[3:7]
        )[:, 2]
        self._ycb08_reference_captured = True

    def _update_ycb08_events(self) -> None:
        if (
            not self._ycb08_reference_captured
            and float(self._ycb08_state.get("settle_s", 0.0)) + self._sim_dt
            >= YCB08_SETTLE_SECONDS
        ):
            self._capture_ycb08_settled_reference()
        metrics = self._ycb08_metrics(update_contacts=True)
        failure_before = self._ycb08_state.get("failure_reason")
        update_ycb08_state(self._ycb08_state, metrics, self._sim_dt)
        failure_after = self._ycb08_state.get("failure_reason")
        if failure_before is None and failure_after is not None:
            print(
                "[HandsOn/IsaacLab] YCB08 terminal failure: "
                f"reason={failure_after}, "
                f"max_linear={float(metrics.get('max_linear_speed_mps', 0.0)):.3f}m/s"
                f"({metrics.get('max_linear_speed_body')}), "
                f"max_angular={float(metrics.get('max_angular_speed_radps', 0.0)):.3f}rad/s"
                f"({metrics.get('max_angular_speed_body')}), "
                f"pan_tilt={float(metrics.get('pan_tilt_deg', 0.0)):.1f}deg, "
                f"containment={float(metrics.get('containment_fraction', 0.0)):.2f}, "
                f"direct_hand_sponge={bool(metrics.get('direct_hand_sponge_contact'))}, "
                f"sponge_table={bool(metrics.get('sponge_table_contact'))}.",
                flush=True,
            )
        self._update_ycb08_left_pan_hold(metrics)
        metrics["pan_support_ok"] = bool(metrics["pan_support_ok"])
        metrics["failure_reason"] = self._ycb08_state.get("failure_reason")
        metrics["success"] = bool(self._ycb08_state.get("success"))
        self._ycb08_last_metrics = metrics

    def _update_ycb08_left_pan_hold(self, metrics: dict[str, Any]) -> None:
        lock_was_active = bool(self._ycb08_left_pan_lock_state.get("active"))
        update_ycb08_left_pan_lock_state(
            self._ycb08_left_pan_lock_state,
            metrics,
            self._sim_dt,
        )
        if (
            not lock_was_active
            and bool(self._ycb08_left_pan_lock_state.get("active"))
        ):
            self._ycb08_left_pan_locked_hand = np.asarray(
                self._left_hand, dtype=np.float64
            ).copy()
            print(
                "[HandsOn/IsaacLab] YCB08 left-pan hold locked: left arm "
                "pose actions will be recorded as zero while the hold stays stable.",
                flush=True,
            )
        elif lock_was_active and not self._ycb08_left_pan_lock_state.get("active"):
            self._ycb08_left_pan_locked_hand = None
            self._ycb08_left_pan_rebase_pending = True
            print(
                "[HandsOn/IsaacLab] YCB08 left-pan hold released: regrip and "
                "level the pan to lock again.",
                flush=True,
            )
        metrics["left_pan_lock_active"] = bool(
            self._ycb08_left_pan_lock_state.get("active")
        )
        metrics["left_pan_lock_ready_s"] = float(
            self._ycb08_left_pan_lock_state.get("ready_s", 0.0)
        )

    def on_live_teleop_pose_rebase(self) -> None:
        """Discard the old live offset after a tracker hold or Z-key rebase."""
        self._ycb08_left_pan_live_offset = None
        self._ycb08_left_pan_rebase_pending = True

    def condition_live_teleop_action(
        self,
        action: Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        """Freeze the latched left target and expose that hold in recorded actions."""

        if not bool(self._ycb08_left_pan_lock_state.get("active", False)):
            # Tracker targets continue moving while the hand is locked. Rebase
            # the live mapping on release so that old motion cannot snap the
            # arm away from the pan. Only the resulting action is recorded.
            left = np.asarray(action.get("left", []), dtype=np.float64).reshape(-1)
            if left.size < 7 or np.allclose(left[:7], 0.0):
                return action
            if self._ycb08_left_pan_rebase_pending:
                self._ycb08_left_pan_live_offset = (
                    self._left_tcp_pose[:3] - left[:3],
                    quaternion_wxyz_to_matrix(left[3:7]).T
                    @ quaternion_wxyz_to_matrix(self._left_tcp_pose[3:7]),
                )
                self._ycb08_left_pan_rebase_pending = False
            if self._ycb08_left_pan_live_offset is not None:
                translation, rotation = self._ycb08_left_pan_live_offset
                conditioned = dict(action)
                left = left.copy()
                left[:3] += translation
                left[3:7] = _matrix_to_quat_wxyz(
                    quaternion_wxyz_to_matrix(left[3:7]) @ rotation
                )
                conditioned["left"] = left
                live_hints = getattr(self, "_live_teleop_velocity_hints", None)
                if isinstance(live_hints, dict):
                    live_hints.pop("left", None)
                return conditioned
            return action
        if "left" not in action:
            return action

        conditioned, frozen_hand = ycb08_locked_teleop_action(
            action,
            self._ycb08_left_pan_locked_hand,
        )
        self._ycb08_left_pan_locked_hand = frozen_hand.copy()

        # The pose action is now a replayable no-op.  Remove the live-only
        # velocity hint as well, otherwise H01 velocity assistance could keep
        # moving the arm despite the zero action saved to the episode.
        live_hints = getattr(self, "_live_teleop_velocity_hints", None)
        if isinstance(live_hints, dict):
            live_hints.pop("left", None)
        return conditioned

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        self._update_ycb08_events()

    def _success_steps_required(self) -> int:
        return 1

    def _compute_success_now(self) -> bool:
        return bool(
            self._ycb08_state.get("success")
            and self._ycb08_state.get("failure_reason") is None
        )

    def _compute_failure(self) -> bool:
        return bool(self._ycb08_state.get("failure_reason") is not None)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._ycb08_last_metrics or self._ycb08_metrics(
            update_contacts=False
        )
        state = self._ycb08_state
        checks = ycb08_final_checks(state, metrics)
        containment_percent = 100.0 * float(
            metrics.get("containment_fraction", 0.0)
        )
        floor_gap_mm = 1000.0 * float(metrics.get("floor_gap_m", 0.0))
        contact = {
            "pan": bool(metrics.get("pan_sponge_contact", False)),
            "table": bool(metrics.get("sponge_table_contact", False)),
            "spatula": bool(metrics.get("spatula_sponge_contact", False)),
            "hands": bool(metrics.get("direct_hand_sponge_contact", False)),
        }

        def yes_no(value: Any) -> str:
            return "yes" if bool(value) else "no"

        return [
            self._condition_row(
                "ycb08_tool_grasp_move",
                "Right hand grasped the spatula and moved it at least 20 mm",
                state.get("tool_grasp_seen") and state.get("tool_moved_seen"),
            ),
            self._condition_row(
                "ycb08_tool_under",
                "Blade contacts the sponge with at least 20% bottom overlap",
                state.get("tool_contact_seen") and state.get("tool_under_seen"),
            ),
            self._condition_row(
                "ycb08_lift",
                "Sponge was lifted at least 20 mm",
                state.get("lift_seen", False),
            ),
            self._condition_row(
                "ycb08_flip",
                "Valid flip crossed 90 deg and reached 155 deg; pan stayed within 25 deg",
                state.get("valid_flip_seen", False),
            ),
            self._condition_row(
                "ycb08_final_containment",
                f"In-pan area {containment_percent:.0f}% / min 80%; "
                f"rim clear={yes_no(metrics.get('rim_clear'))}, "
                f"COM below rim={yes_no(metrics.get('com_below_rim'))}",
                checks["containment_ok"]
                and bool(metrics.get("rim_clear", False))
                and bool(metrics.get("com_below_rim", False)),
            ),
            self._condition_row(
                "ycb08_final_support",
                f"Pan-only support: pan contact={yes_no(contact['pan'])}; "
                f"table/spatula/hand contact={yes_no(contact['table'])}/"
                f"{yes_no(contact['spatula'])}/{yes_no(contact['hands'])}; "
                f"floor gap={floor_gap_mm:.1f} mm",
                checks["pan_support_ok"],
            ),
            self._condition_row(
                "ycb08_no_hand_touch",
                "Neither hand directly touched the sponge",
                not state.get("direct_hand_violation", False),
            ),
            self._condition_row(
                "ycb08_release",
                f"Spatula clear of sponge {float(state.get('tool_release_s', 0.0)):.2f}"
                f" / {YCB08_TOOL_RELEASE_SECONDS:.2f} s",
                checks["tool_released"],
            ),
            self._condition_row(
                "ycb08_final_face_now",
                f"Sponge remains flipped {float(metrics.get('flip_angle_deg', 0.0)):.0f} deg"
                f" / min {YCB08_FINAL_FLIP_ANGLE_DEG:.0f} deg",
                checks["opposite_face_now"],
            ),
            self._condition_row(
                "ycb08_final_pan_level",
                f"Pan tilt {float(metrics.get('pan_tilt_deg', 0.0)):.1f} deg"
                f" / max {YCB08_MAX_FINAL_PAN_TILT_DEG:.0f} deg",
                checks["pan_level_now"],
            ),
            self._condition_row(
                "ycb08_final_linear_speed",
                f"Sponge linear speed {float(metrics.get('sponge_linear_speed_mps', 0.0)):.3f}"
                f" / max {YCB08_MAX_FINAL_LINEAR_SPEED_MPS:.3f} m/s",
                checks["sponge_linear_speed_ok"],
            ),
            self._condition_row(
                "ycb08_final_angular_speed",
                f"Sponge angular speed {float(metrics.get('sponge_angular_speed_radps', 0.0)):.2f}"
                f" / max {YCB08_MAX_FINAL_ANGULAR_SPEED_RADPS:.2f} rad/s",
                checks["sponge_angular_speed_ok"],
            ),
            self._condition_row(
                "ycb08_stable",
                f"All final checks held for "
                f"{float(state.get('final_stability_s', 0.0)):.2f}"
                f" / {YCB08_FINAL_STABILITY_SECONDS:.2f} s (any failed check resets timer)",
                float(state.get("final_stability_s", 0.0))
                >= YCB08_FINAL_STABILITY_SECONDS,
            ),
        ]

    def _task_status_text(self) -> str:
        metrics = self._ycb08_last_metrics
        lock_text = (
            "left hand locked"
            if bool(self._ycb08_left_pan_lock_state.get("active"))
            else "left hand free: grip and level pan to lock"
        )
        return (
            f"flip {float(metrics.get('flip_angle_deg', 0.0)):.0f}/155 deg"
            f" | pan tilt {float(metrics.get('pan_tilt_deg', 0.0)):.1f}/12 deg"
            f" | in pan {100.0 * float(metrics.get('containment_fraction', 0.0)):.0f}/80%"
            f" | sponge lin/ang speed "
            f"{float(metrics.get('sponge_linear_speed_mps', 0.0)):.3f}/"
            f"{float(metrics.get('sponge_angular_speed_radps', 0.0)):.2f}"
            f" | stable {float(self._ycb08_state.get('final_stability_s', 0.0)):.2f}/1.00 s"
            f" | {lock_text}"
        )

    def _task_info(self) -> dict[str, Any]:
        metrics = self._ycb08_last_metrics or self._ycb08_metrics(
            update_contacts=False
        )
        final_checks = ycb08_final_checks(self._ycb08_state, metrics)
        sampled_spatula = self._ycb08_sampled_reset_poses.get(
            YCB08_SPATULA_NAME
        )
        if sampled_spatula is None:
            sampled_spatula = np.asarray(
                metrics.get(
                    "spatula_pose",
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ),
                dtype=np.float64,
            )
        return {
            **metrics,
            "task_id": YCB08_TASK_ID,
            "task_name": YCB08_TASK_NAME,
            "assistance": "left_pan_hold_after_physical_lift_and_level",
            "left_pan_lock_active": bool(
                self._ycb08_left_pan_lock_state.get("active", False)
            ),
            "left_pan_lock_ready_s": float(
                self._ycb08_left_pan_lock_state.get("ready_s", 0.0)
            ),
            "left_pan_unlock_unstable_s": float(
                self._ycb08_left_pan_lock_state.get("unstable_s", 0.0)
            ),
            "left_pan_unlock_debounce_s": YCB08_LEFT_PAN_UNLOCK_DEBOUNCE_SECONDS,
            "reset_ordinal": int(self._ycb08_reset_ordinal),
            "sampled_reset_poses": {
                name: pose.copy()
                for name, pose in self._ycb08_sampled_reset_poses.items()
            },
            "ycb08_state": dict(self._ycb08_state),
            "failure_reason": self._ycb08_state.get("failure_reason"),
            "terminal_reason": self._ycb08_state.get("failure_reason"),
            "final_stability_checks": final_checks,
            "final_stability_blockers": [
                name for name, passed in final_checks.items() if not passed
            ],
            "sampled_spatula_face_up": ycb08_spatula_is_face_up(
                sampled_spatula
            ),
            "pan_bowl_center_xy_bounds_m": (
                YCB08_BOWL_CENTER_XY_BOUNDS_M.copy()
            ),
            "spatula_center_xy_bounds_m": (
                YCB08_SPATULA_CENTER_XY_BOUNDS_M.copy()
            ),
            "pan_handle_yaw_range_deg": tuple(YCB08_PAN_YAW_RANGE_DEG),
            "spatula_handle_yaw_range_deg": tuple(
                YCB08_SPATULA_YAW_RANGE_DEG
            ),
            "spatula_face_up_probability": float(
                YCB08_SPATULA_FACE_UP_PROBABILITY
            ),
            "contact_views_available": bool(
                all(
                    self._ycb08_contact_available.get(key, False)
                    for key in _YCB08_CORE_CONTACT_KEYS
                )
            ),
            "left_pan_lock_min_lift_m": float(YCB08_LEFT_PAN_LOCK_MIN_LIFT_M),
            "left_pan_lock_max_tilt_deg": float(
                YCB08_LEFT_PAN_LOCK_MAX_TILT_DEG
            ),
            "left_pan_lock_max_linear_speed_mps": float(
                YCB08_LEFT_PAN_LOCK_MAX_LINEAR_SPEED_MPS
            ),
            "left_pan_lock_max_angular_speed_radps": float(
                YCB08_LEFT_PAN_LOCK_MAX_ANGULAR_SPEED_RADPS
            ),
            "left_pan_lock_debounce_s": float(
                YCB08_LEFT_PAN_LOCK_DEBOUNCE_SECONDS
            ),
            "excessive_velocity_debounce_s": float(
                YCB08_EXCESSIVE_VELOCITY_DEBOUNCE_SECONDS
            ),
            "spatula_mass_kg": float(YCB08_SPATULA_MASS_KG),
            "min_blade_overlap": float(YCB08_MIN_BLADE_OVERLAP),
            "min_lift_m": float(YCB08_MIN_SPONGE_LIFT_M),
            "final_flip_angle_deg": float(YCB08_FINAL_FLIP_ANGLE_DEG),
            "max_flip_pan_tilt_deg": float(
                YCB08_MAX_PAN_TILT_DURING_FLIP_DEG
            ),
            "max_final_pan_tilt_deg": float(YCB08_MAX_FINAL_PAN_TILT_DEG),
            "direct_hand_debounce_s": float(
                YCB08_DIRECT_HAND_DEBOUNCE_SECONDS
            ),
            "outside_support_debounce_s": float(
                YCB08_OUTSIDE_SUPPORT_DEBOUNCE_SECONDS
            ),
            "pan_overturn_angle_deg": float(YCB08_PAN_OVERTURN_ANGLE_DEG),
            "pan_overturn_debounce_s": float(YCB08_PAN_OVERTURN_DEBOUNCE_SECONDS),
        }

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._ycb08_last_metrics or self._ycb08_metrics(
            update_contacts=False
        )
        state = observation["state"]
        pan_velocity = np.asarray(metrics["pan_velocity"], dtype=np.float64)
        spatula_velocity = np.asarray(metrics["spatula_velocity"], dtype=np.float64)
        sponge_velocity = np.asarray(metrics["sponge_velocity"], dtype=np.float64)
        forces = metrics.get("contact_forces", {})
        state.update(
            {
                "skillet_pose": np.asarray(metrics["pan_pose"], dtype=np.float64),
                "skillet_linear_velocity": pan_velocity[:3],
                "skillet_angular_velocity": pan_velocity[3:],
                "spatula_pose": np.asarray(metrics["spatula_pose"], dtype=np.float64),
                "spatula_linear_velocity": spatula_velocity[:3],
                "spatula_angular_velocity": spatula_velocity[3:],
                "sponge_pose": np.asarray(metrics["sponge_pose"], dtype=np.float64),
                "sponge_linear_velocity": sponge_velocity[:3],
                "sponge_angular_velocity": sponge_velocity[3:],
                "sponge_in_pan_position": np.asarray(
                    metrics["sponge_in_pan_position"], dtype=np.float64
                ),
                "ycb08_geometry_metrics": np.asarray(
                    [
                        metrics.get("flip_angle_deg", 0.0),
                        metrics.get("pan_tilt_deg", 0.0),
                        metrics.get("containment_fraction", 0.0),
                        metrics.get("blade_overlap_fraction", 0.0),
                        metrics.get("sponge_lift_m", 0.0),
                        metrics.get("spatula_move_distance_m", 0.0),
                        metrics.get("floor_gap_m", 0.0),
                        metrics.get("max_footprint_radius_m", 0.0),
                        metrics.get("sponge_linear_speed_mps", 0.0),
                        metrics.get("sponge_angular_speed_radps", 0.0),
                    ],
                    dtype=np.float64,
                ),
                "ycb08_contact_state": np.asarray(
                    [
                        forces.get("sponge_spatula", 0.0),
                        forces.get("sponge_pan", 0.0),
                        forces.get("sponge_table", 0.0),
                        forces.get("sponge_hands", 0.0),
                        forces.get("spatula_right_hand", 0.0),
                        float(bool(metrics.get("contact_views_available", False))),
                    ],
                    dtype=np.float64,
                ),
                "ycb08_left_pan_lock_state": np.asarray(
                    [
                        float(
                            bool(
                                self._ycb08_left_pan_lock_state.get(
                                    "active", False
                                )
                            )
                        ),
                        float(
                            self._ycb08_left_pan_lock_state.get("ready_s", 0.0)
                        ),
                        forces.get(_YCB08_LEFT_PAN_CONTACT_KEY, 0.0),
                        float(
                            bool(metrics.get("left_pan_contact_available", False))
                        ),
                        float(bool(metrics.get("left_hand_pan_contact", False))),
                        metrics.get("pan_lift_m", 0.0),
                        metrics.get("pan_tilt_deg", 0.0),
                        metrics.get("pan_linear_speed_mps", 0.0),
                        metrics.get("pan_angular_speed_radps", 0.0),
                        float(self._ycb08_left_pan_lock_state.get("unstable_s", 0.0)),
                    ],
                    dtype=np.float64,
                ),
                "ycb08_event_flags": ycb08_event_flags(
                    self._ycb08_state, metrics
                ),
                "ycb08_hidden_state": ycb08_hidden_state(self._ycb08_state),
                "ycb08_sampled_reset_poses": np.stack(
                    [
                        self._ycb08_sampled_reset_poses.get(
                            name, self._ycb08_pose(name)
                        )
                        for name in YCB08_BODY_NAMES
                    ],
                    axis=0,
                ).astype(np.float64),
            }
        )
        return observation

    def restore_recorded_initial_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        report = super().restore_recorded_initial_state(state_parts)
        self._ycb08_left_pan_rebase_pending = False
        self._ycb08_left_pan_live_offset = None
        lock_values = np.asarray(
            state_parts.get("ycb08_left_pan_lock_state", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        self._ycb08_left_pan_lock_state = initial_ycb08_left_pan_lock_state()
        if lock_values.size >= 1:
            self._ycb08_left_pan_lock_state["active"] = bool(
                lock_values[0] > 0.5
            )
        if lock_values.size >= 2:
            self._ycb08_left_pan_lock_state["ready_s"] = float(lock_values[1])
        # Nine-element recordings predate recoverable holds.
        if lock_values.size >= 10:
            self._ycb08_left_pan_lock_state["unstable_s"] = float(lock_values[9])
        self._ycb08_left_pan_locked_hand = (
            np.asarray(self._left_hand, dtype=np.float64).copy()
            if self._ycb08_left_pan_lock_state["active"]
            else None
        )
        sampled = np.asarray(
            state_parts.get("ycb08_sampled_reset_poses", np.zeros((0, 7))),
            dtype=np.float64,
        ).reshape(-1, 7)
        if sampled.shape[0] >= len(YCB08_BODY_NAMES):
            self._ycb08_sampled_reset_poses = {
                name: sampled[index].copy()
                for index, name in enumerate(YCB08_BODY_NAMES)
            }
        flags = np.asarray(
            state_parts.get("ycb08_event_flags", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        hidden = np.asarray(
            state_parts.get("ycb08_hidden_state", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        restored = initial_ycb08_state()
        for index, name in enumerate(YCB08_EVENT_FLAG_NAMES):
            if index >= flags.size:
                break
            if name in restored:
                restored[name] = bool(flags[index] > 0.5)
        timer_names = (
            "settle_s",
            "direct_hand_contact_s",
            "outside_support_s",
            "pan_overturn_s",
            "tool_release_s",
            "final_stability_s",
            "flip_attempt_max_pan_tilt_deg",
            "max_flip_angle_deg",
        )
        for index, name in enumerate(timer_names):
            if index < hidden.size:
                restored[name] = float(hidden[index])
        if hidden.size >= 10:
            restored["flip_attempt_active"] = bool(hidden[8] > 0.5)
            restored["flip_attempt_crossed_90"] = bool(hidden[9] > 0.5)
        if hidden.size >= 11:
            reverse_codes = {value: key for key, value in YCB08_FAILURE_CODES.items()}
            restored["failure_reason"] = reverse_codes.get(
                int(round(float(hidden[10]))),
                "unknown_recorded_failure",
            )
        if hidden.size >= 12:
            restored["excessive_velocity_s"] = float(hidden[11])
        self._ycb08_state = restored
        self._capture_ycb08_settled_reference()
        self._ycb08_last_metrics = self._ycb08_metrics(update_contacts=False)
        report["ycb08_hidden_state_restored"] = True
        report["ycb08_left_pan_lock_state_restored"] = True
        return report

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        timed_out = bool(
            self.env_step * self._control_dt >= self._time_limit
            and not info.get("succeed", False)
            and not info.get("failed", False)
            and not info.get("manual_reset", False)
        )
        if timed_out:
            terminated = False
            truncated = True
            info["truncated_reason"] = "time_limit"
            info["terminal_reason"] = "time_limit"
        return observation, reward, terminated, truncated, info


__all__ = ["YCB08SpatulaFlipSpongeIsaacLabEnv"]
