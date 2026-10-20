"""Native Isaac Sim 5.1 / IsaacLab 2.3.2 collection environment for YCB11."""

from __future__ import annotations

import math
import json
from dataclasses import replace
from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv, _quat_wxyz_to_matrix
from .ycb11_scene import (
    YCB11_SPONGE_COLLISION_MESH_PATH,
    YCB11_MUG_BODY_NAME,
    YCB11_MUG_CONTACT_OFFSET_M,
    YCB11_MUG_NOMINAL_POSITION,
    YCB11_MUG_RESET_ANGULAR_DAMPING,
    YCB11_MUG_RESET_LINEAR_DAMPING,
    YCB11_MUG_RESET_SLEEP_THRESHOLD,
    YCB11_MUG_REST_OFFSET_M,
    YCB11_SPONGE_NOMINAL_POSITION,
    YCB11_SPONGE_ROOT_PATH,
    YCB11_SPONGE_SOLVER_ITERATIONS,
    YCB11_SUCCESS_CONDITION_LABELS,
    YCB11_TABLE_TOP_Z,
    YCB11_TASK_NAME,
    spawn_ycb11_volume_deformable_sponge,
)
from .ycb11_semantics import (
    DEFAULT_THRESHOLDS,
    YCB11_CONTROL_HZ,
    YCB11_HAND_MAX_DEPENETRATION_VELOCITY_MPS,
    YCB11_PHYSICS_HZ,
    YCB11Signals,
    YCB11Stage,
    YCB11StateMachine,
    compression_ratio,
    containment_metrics,
    mug_tilt_deg,
    principal_span,
)

_YCB11_MUG_CONTACT_FORCE_N = 0.08
_YCB11_SPONGE_CONTACT_SURFACE_TOLERANCE_M = 0.0035
_YCB11_SPONGE_CONTACT_MIN_NODES = 2
_YCB11_GRASP_DEFORMATION_RATIO = 0.98
_YCB11_MUG_BOUNDING_RADIUS_M = 0.056
_YCB11_SPONGE_WORKSPACE_X = (-0.58, 0.28)
_YCB11_SPONGE_WORKSPACE_Y = (-0.82, 0.82)
_YCB11_MAX_WORLD_Z = 1.55


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_nonzero_uniform(random_state: Any, low: float, high: float) -> float:
    value = float(random_state.uniform(low, high))
    if abs(value) < 1e-4:
        value = math.copysign(1e-4, value if value != 0.0 else high)
    return value


class _YCB11CurrentVolumeData:
    """Small data facade for Isaac Sim's successor volume-body tensor view.

    IsaacLab's ``DeformableObject`` in the checked-out 2.3 tree still binds
    the deprecated ``PhysxDeformableBodyAPI``/``SoftBodyView`` pair.  YCB11 is
    authored with the current ``OmniPhysicsDeformableBodyAPI`` and therefore
    uses ``DeformableBodyView`` directly.  Keeping this facade's attribute
    names aligned with IsaacLab's data object lets the task logic stay the
    same while ensuring no legacy schema is applied to the YCB asset.
    """

    def __init__(self, view: Any):
        import torch

        self._view = view
        self._timestamp = 0.0
        initial_positions = view.get_simulation_nodal_positions()
        initial_velocities = view.get_simulation_nodal_velocities()
        self.default_nodal_state_w: Any | None = torch.cat(
            (initial_positions, initial_velocities), dim=-1
        ).clone()
        self._rest_nodes = np.asarray(
            _tensor_numpy(view.get_rest_nodal_positions()), dtype=np.float64
        )
        self._element_indices = np.asarray(
            _tensor_numpy(view.get_simulation_element_indices()), dtype=np.int64
        )

    def update(self, dt: float) -> None:
        self._timestamp += float(dt)

    @property
    def nodal_pos_w(self) -> Any:
        return self._view.get_simulation_nodal_positions()

    @property
    def nodal_vel_w(self) -> Any:
        return self._view.get_simulation_nodal_velocities()

    @property
    def nodal_state_w(self) -> Any:
        # The current view intentionally exposes positions and velocities as
        # separate tensors, just like the IsaacLab data object does lazily.
        import torch

        return torch.cat((self.nodal_pos_w, self.nodal_vel_w), dim=-1)

    @property
    def sim_element_deform_gradient_w(self) -> np.ndarray:
        """Return CPU deformation gradients derived from current tet nodes.

        ``DeformableBodyView`` does not expose the deprecated gradient getter.
        The rest-shape and element-index accessors are part of the current API,
        so compute the same \(F = D_s D_m^{-1}\) quantity locally.  Keeping
        this diagnostic on CPU also avoids the CUDA/Vulkan small-matrix kernel
        issue documented by the task's readback guard.
        """

        current = np.asarray(_tensor_numpy(self.nodal_pos_w), dtype=np.float64)
        rest = self._rest_nodes
        indices = self._element_indices
        if current.ndim != 3 or rest.ndim != 3 or indices.ndim != 3:
            raise RuntimeError(
                "YCB11 current volume topology has an invalid tensor layout"
            )
        node_count = current.shape[1]
        # The view is single-instance for this task.  Filter any optional
        # padded element slots defensively before indexing the node buffer.
        body_indices = indices[0].reshape(-1, indices.shape[-1])
        if body_indices.shape[1] != 4:
            raise RuntimeError(
                "YCB11 current volume simulation elements are not tetrahedral"
            )
        valid = np.all(
            (body_indices >= 0) & (body_indices < min(node_count, rest.shape[1])),
            axis=1,
        )
        body_indices = body_indices[valid]
        if body_indices.size == 0:
            raise RuntimeError("YCB11 current volume has no valid simulation tets")
        now = current[0, body_indices]
        ref = rest[0, body_indices]
        dm = np.stack(
            (ref[:, 1] - ref[:, 0], ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 0]),
            axis=-1,
        )
        ds = np.stack(
            (now[:, 1] - now[:, 0], now[:, 2] - now[:, 0], now[:, 3] - now[:, 0]),
            axis=-1,
        )
        # Tetrahedra generated by PhysX are non-degenerate.  Pinv is used as
        # a defensive fallback for a malformed/custom asset rather than
        # emitting NaNs into the health guard.
        try:
            inverse = np.linalg.inv(dm)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(dm)
        gradients = np.matmul(ds, inverse)
        return gradients[None, ...]


class _YCB11CurrentVolumeBody:
    """IsaacLab-shaped wrapper around a current ``DeformableBodyView``."""

    def __init__(self, view: Any):
        self.root_physx_view = view
        self.data = _YCB11CurrentVolumeData(view)
        self._is_initialized = bool(
            int(getattr(view, "count", 0)) == 1
            and (not hasattr(view, "check") or bool(view.check()))
        )

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def write_nodal_state_to_sim(self, nodal_state: Any, env_ids: Any = None) -> None:
        if env_ids is None:
            indices = self._indices_for(nodal_state, count=int(self.root_physx_view.count))
        else:
            indices = env_ids
        self.root_physx_view.set_simulation_nodal_positions(
            nodal_state[..., :3], indices
        )
        self.root_physx_view.set_simulation_nodal_velocities(
            nodal_state[..., 3:], indices
        )

    def reset(self) -> None:
        # Current PhysX views apply writes immediately; the ABC reset path has
        # already written the complete nodal state before calling reset.
        return None

    def update(self, dt: float) -> None:
        self.data.update(dt)

    @staticmethod
    def _indices_for(state: Any, *, count: int) -> Any:
        # Keep the index tensor on the same device/frontend as the state.  The
        # current tensor API converts it to uint32 internally.
        import torch

        return torch.arange(
            count,
            dtype=torch.int32,
            device=getattr(state, "device", None),
        )


def _capsule_signed_distances(
    points: np.ndarray,
    pose_wxyz: np.ndarray,
    *,
    radius_m: float,
    half_length_m: float,
) -> np.ndarray:
    """Signed distances from world points to a body-local Z capsule."""

    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    rotation = _quat_wxyz_to_matrix(pose[3:7])
    axis = rotation[:, 2]
    segment_start = pose[:3] - float(half_length_m) * axis
    segment_end = pose[:3] + float(half_length_m) * axis
    segment = segment_end - segment_start
    denominator = float(np.dot(segment, segment))
    relative = np.asarray(points, dtype=np.float64) - segment_start[None, :]
    if denominator <= 1e-16:
        closest = np.broadcast_to(segment_start, relative.shape)
    else:
        alpha = np.clip((relative @ segment) / denominator, 0.0, 1.0)
        closest = segment_start[None, :] + alpha[:, None] * segment[None, :]
    return np.linalg.norm(np.asarray(points) - closest, axis=1) - float(radius_m)


def _sphere_signed_distances(
    points: np.ndarray,
    pose_wxyz: np.ndarray,
    *,
    center_local_m: tuple[float, float, float],
    radius_m: float,
) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    rotation = _quat_wxyz_to_matrix(pose[3:7])
    center = pose[:3] + rotation @ np.asarray(center_local_m, dtype=np.float64)
    return np.linalg.norm(np.asarray(points) - center[None, :], axis=1) - float(
        radius_m
    )


def _box_signed_distances(
    points: np.ndarray,
    pose_wxyz: np.ndarray,
    *,
    center_local_m: tuple[float, float, float],
    half_extents_m: tuple[float, float, float],
) -> np.ndarray:
    """Exact signed distance to an oriented box (negative inside)."""

    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    rotation = _quat_wxyz_to_matrix(pose[3:7])
    local = (np.asarray(points, dtype=np.float64) - pose[:3]) @ rotation
    q = np.abs(local - np.asarray(center_local_m, dtype=np.float64)) - np.asarray(
        half_extents_m, dtype=np.float64
    )
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


class _YCB11DeformableViewAdapter:
    """Expose IsaacLab's volume object through the recorder's nodal-view API."""

    def __init__(self, deformable: Any):
        self._deformable = deformable

    def get_simulation_nodal_positions(self):
        return self._deformable.data.nodal_pos_w

    def get_simulation_nodal_velocities(self):
        return self._deformable.data.nodal_vel_w

    def set_simulation_nodal_positions(self, values: Any, indices: Any = None):
        view = self._deformable.root_physx_view
        if indices is None:
            indices = self._deformable._indices_for(values, count=int(view.count))
        view.set_simulation_nodal_positions(values, indices)

    def set_simulation_nodal_velocities(self, values: Any, indices: Any = None):
        # A velocity-only canonical boundary must never rewrite positions:
        # doing so rebuilds the soft-body state/contact history after the
        # strict origin proof. The current native API has separate setters.
        view = self._deformable.root_physx_view
        if indices is None:
            indices = self._deformable._indices_for(values, count=int(view.count))
        view.set_simulation_nodal_velocities(values, indices)


class HandsOnYCB11CompressSpongeIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual physical sponge compression and insertion using only YCB assets."""

    _TASK_NAME = YCB11_TASK_NAME
    _TASK_ID = "YCB11"

    def __init__(self, task_name: str = YCB11_TASK_NAME, **kwargs: Any):
        self._ycb11_sponge: Any | None = None
        self._ycb11_deformable_view: Any | None = None
        self._ycb11_deformable_indices: Any | None = None
        self._ycb11_default_nodal_state: Any | None = None
        self._ycb11_rest_nodes_np = np.zeros((4, 3), dtype=np.float64)
        self._ycb11_rest_collision_nodes_np = self._ycb11_rest_nodes_np.copy()
        self._ycb11_nodes_np = self._ycb11_rest_nodes_np.copy()
        self._ycb11_velocities_np = np.zeros_like(self._ycb11_nodes_np)
        self._ycb11_expected_node_count = 0
        self._ycb11_rest_principal_span_m = float(
            np.max(np.ptp(self._ycb11_rest_nodes_np, axis=0))
        )
        self._ycb11_keypoint_indices = np.zeros(8, dtype=np.int64)
        self._ycb11_readback_failed = False
        self._ycb11_readback_failure_reason: str | None = None
        self._ycb11_initial_state = np.zeros(7, dtype=np.float64)
        self._ycb11_initial_mug_pose = np.asarray(
            [*YCB11_MUG_NOMINAL_POSITION, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._ycb11_initial_sponge_com_z = float(
            YCB11_SPONGE_NOMINAL_POSITION[2]
        )
        self._ycb11_state_machine = YCB11StateMachine()
        self._ycb11_last_signals = YCB11Signals()
        self._ycb11_last_metrics: dict[str, Any] = {}
        self._ycb11_hand_collider_audit: list[dict[str, Any]] = []
        self._ycb11_hand_hulls: dict[str, list[tuple]] | None = None
        self._ycb11_left_mug_views: tuple[Any, ...] = ()
        self._ycb11_mug_table_view: Any | None = None
        # Full YCB11 diagnostics synchronize the native deformable node and
        # deformation-gradient tensors.  Accumulate only the cheap rigid
        # contact facts at each PhysX substep; the full readback is consumed
        # once, at the control-frame boundary, by ``_refresh_object_poses``.
        self._ycb11_substep_count = 0
        self._ycb11_substep_left_mug_contact_all = True
        self._ycb11_substep_left_mug_contact_any = False
        self._ycb11_substep_mug_table_contact_all = True
        self._ycb11_substep_mug_table_contact_any = False
        self._ycb11_substep_contact_readback_failed = False
        self._ycb11_contact_readback_failed = False
        self._ycb11_substep_last_left_mug_forces: tuple[float, ...] = ()
        self._ycb11_substep_last_mug_table_force = 0.0
        super().__init__(task_name=task_name, **kwargs)
        # Keep raw 30 Hz actions intact. Bound only physical drive targets;
        # contact geometry and actual readback remain authoritative.
        self._smooth_grasp_hand_target_interpolation_enabled = True

    def _configure_task_robot_cfg(self, robot_cfg: Any, side: str) -> Any:
        from .abc_collection_env import _smooth_grasp_hand_drive_maps

        robot_cfg = super()._configure_task_robot_cfg(robot_cfg, side)
        # Use the existing Wuji source gains/torques (distal joints about
        # 0.3--0.5 Nm), rather than the generic 6 Nm, 8 Nm/rad hand drive.
        # Fingers must yield when foam or the mug blocks their commanded pose.
        hand = robot_cfg.actuators["hand"]
        hand.stiffness, hand.damping, hand.effort_limit_sim = (
            _smooth_grasp_hand_drive_maps(side, self.hand_type)
        )
        hand.friction = 0.0
        hand.velocity_limit_sim = 2.0
        robot_cfg.actuators["arm"].velocity_limit_sim = 2.5
        return robot_cfg

    def _physx_contact_arm_target_limits_enabled(self) -> bool:
        return True

    def _limit_arm_pose_target(self, side: str, robot: Any, pose: np.ndarray) -> np.ndarray:
        from .ycb11_control_safety import limit_pose_motion

        actual = self._pose_matrix_to_vector(
            self._robot_body_pose_matrix(robot, self._ee_body_indices[side])
        )
        # An unbounded DLS correction at the extended reset posture can rotate
        # the wrist through objects. These are servo-error bounds, not edits to
        # the recorded operator target or task-object state.
        limited = limit_pose_motion(actual, pose, translation_m=0.040, rotation_deg=8.0)
        if self._stage is not None and self._ycb11_hand_collider_audit:
            # The arm may press the sponge but cannot request its complete
            # hand below the supporting table. Use the authored hand surface,
            # not the TCP or the obsolete capsule approximation. This bounds
            # a drive target only; no robot/body/node state is overwritten.
            current_rotation = _quat_wxyz_to_matrix(actual[3:])
            target_rotation = _quat_wxyz_to_matrix(limited[3:])
            poses = _tensor_numpy(robot.data.body_pose_w)[0]
            lowest = math.inf
            for index, points, _planes, _group in self._hand_contact_hulls()[side]:
                body = poses[index]
                world = points @ _quat_wxyz_to_matrix(body[3:7]).T + body[:3]
                relative = (world - actual[:3]) @ current_rotation
                lowest = min(lowest, float((relative @ target_rotation.T)[:, 2].min()))
            floor = YCB11_TABLE_TOP_Z + float(self._ycb11_initial_state[6]) + 0.0005
            limited[2] = max(limited[2], floor - lowest)
            delta = limited[:3] - actual[:3]
            if abs(delta[2]) <= 0.040:
                xy_budget = math.sqrt(max(0.0, 0.040**2 - delta[2]**2))
                delta[:2] *= min(1.0, xy_budget / max(float(np.linalg.norm(delta[:2])), 1e-12))
                limited[:3] = actual[:3] + delta
            else:
                # A displaced hand recovers without an unbounded corrective
                # step; defer wrist rotation until that recovery completes.
                limited = limit_pose_motion(actual, np.r_[limited[:3], actual[3:]], 0.040, 8.0)
        return limited

    def _limit_arm_joint_target(self, side: str, robot: Any, arm_target: Any) -> Any:
        current = robot.data.joint_pos[:, self._arm_joint_ids[side]]
        delta = arm_target - current
        peak = self._torch.amax(self._torch.abs(delta), dim=-1, keepdim=True)
        return current + delta * self._torch.clamp(0.12 / peak.clamp_min(1e-12), max=1.0)

    def _prepare_physics_substep_joint_targets(
        self, substep_index: int, substep_count: int,
    ) -> str:
        mode = super()._prepare_physics_substep_joint_targets(
            substep_index, substep_count,
        )
        if mode == "exact_pre_substep_callback" or not self._robots:
            return mode
        # Keep the source PD fingers compliant under a blocked close command.
        # Otherwise a full glove closure preloads even the source gains enough
        # to overload the foam. Raw goals remain available to the next
        # substep and recorder; exact accepted-drive replay bypasses this pass.
        for side, robot in self._robots.items():
            ids = self._hand_joint_ids[side]
            current = robot.data.joint_pos[:, ids]
            target = self._joint_targets[side].clone()
            target[:, ids] = current + self._torch.clamp(
                target[:, ids] - current, -0.04, 0.04,
            )
            robot.set_joint_position_target(target)
            self._joint_targets[side] = target
        return mode

    def _reset_scene_state(self) -> None:
        super()._reset_scene_state()
        if not self._robots or self._direct_hand_control:
            return
        # The shared source home is expressed in the controller frame, while
        # native IK consumes the actual palm frame. Rebase the command origin
        # after reset so a stationary operator cannot drive a 90-degree turn
        # and sweep the mug before the first intentional action.
        for side, robot in self._robots.items():
            robot.update(0.0)
            matrix = self._native_end_effector_pose_matrix(
                side, getattr(self, f"_{side}_tcp_pose"),
            )
            pose = self._pose_matrix_to_vector(matrix)
            if not np.isfinite(pose).all():
                raise RuntimeError(f"YCB11 reset returned a nonfinite {side} palm pose")
            setattr(self, f"_{side}_tcp_pose", pose.copy())

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("YCB11 stage is unavailable before sponge spawn")
        spawn_ycb11_volume_deformable_sponge(self._stage, sim_utils)
        # The current volume body is initialized after the shared ABC scene
        # has created its Torch-backed task simulation view.  Do not construct
        # IsaacLab's legacy DeformableObject here: that class searches for
        # PhysxSchema.PhysxDeformableBodyAPI and would either return zero
        # bodies or apply a conflicting deprecated schema.

    def _hand_rigid_body_paths(self, side: str) -> tuple[tuple[str, str], ...]:
        if self._stage is None:
            return ()
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return ()
        result: list[tuple[str, str]] = []
        for prim in Usd.PrimRange(root):
            normalized = prim.GetName().lower()
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if "palm" not in normalized and "finger" not in normalized:
                continue
            result.append((normalized, str(prim.GetPath())))
        return tuple(result)

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        del sim_utils
        if self._stage is None:
            return 0, 0
        from pxr import PhysxSchema, Usd, UsdPhysics
        from .flexible_hand_colliders import author_hand_contact_hulls

        # The imported capsules leave up to 25 mm of the visible finger links
        # uncovered, and the fingertip spheres miss about 3 mm of the pads.
        # Expand only the instance branches that own those old collision shapes
        # before replacing them with GPU-compatible hulls of the actual CAD.
        for prim in list(self._stage.Traverse()):
            path = str(prim.GetPath()).lower()
            if (prim.IsInstance() and path.startswith("/world/handsonrobots/")
                    and "collisions" in path
                    and ("finger" in path or "palm" in path)):
                prim.SetInstanceable(False)
        self._ycb11_hand_collider_audit = author_hand_contact_hulls(
            self._stage, contact_offset=0.003, rest_offset=0.0005,
        )
        self._ycb11_hand_hulls = None

        body_count = 0
        for side in ("right", "left"):
            for _name, path in self._hand_rigid_body_paths(side):
                prim = self._stage.GetPrimAtPath(path)
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
                rigid_api.CreateSolverPositionIterationCountAttr().Set(64)
                rigid_api.CreateSolverVelocityIterationCountAttr().Set(8)
                rigid_api.CreateMaxDepenetrationVelocityAttr().Set(
                    YCB11_HAND_MAX_DEPENETRATION_VELOCITY_MPS
                )
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report_api.CreateThresholdAttr().Set(0.0)
                body_count += 1

        for name in (YCB11_MUG_BODY_NAME, "table"):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
            rigid_api.CreateSolverPositionIterationCountAttr().Set(48)
            rigid_api.CreateSolverVelocityIterationCountAttr().Set(8)
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)
            body_count += 1

        root_prim = self._stage.GetPrimAtPath(YCB11_SPONGE_ROOT_PATH)
        simulation_prim = self._stage.GetPrimAtPath(
            f"{YCB11_SPONGE_ROOT_PATH}/simulation_mesh"
        )
        collision_prim = self._stage.GetPrimAtPath(
            YCB11_SPONGE_COLLISION_MESH_PATH
        )
        if (
            not root_prim.IsValid()
            or not root_prim.HasAPI("OmniPhysicsDeformableBodyAPI")
            or not simulation_prim.IsValid()
            or not simulation_prim.HasAPI("OmniPhysicsVolumeDeformableSimAPI")
            or not collision_prim.IsValid()
            or not collision_prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            raise RuntimeError(
                "YCB11 sponge is missing the current PhysX volume-body hierarchy"
            )
        for prim in self._stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                configured_hz = int(round(1.0 / float(self._physics_dt)))
                if configured_hz != YCB11_PHYSICS_HZ:
                    raise RuntimeError(
                        "YCB11 volume-deformable contact requires "
                        f"{YCB11_PHYSICS_HZ} Hz physics, got {configured_hz} Hz"
                    )
                scene_api.GetTimeStepsPerSecondAttr().Set(configured_hz)
                scene_api.GetMaxPositionIterationCountAttr().Set(YCB11_SPONGE_SOLVER_ITERATIONS)
                break
        return body_count, 1 + len(self._ycb11_hand_collider_audit)

    @staticmethod
    def _validated_contact_view(
        physics_sim_view: Any,
        sensor_path: str | tuple[str, ...],
        filter_path: str,
        *,
        label: str,
    ) -> Any:
        sensor_paths = (sensor_path,) if isinstance(sensor_path, str) else sensor_path
        if not sensor_paths or len(set(sensor_paths)) != len(sensor_paths):
            raise RuntimeError(f"YCB11 physical {label} requires distinct sensors")
        try:
            view = physics_sim_view.create_rigid_contact_view(
                list(sensor_paths),
                filter_patterns=[[filter_path] for _ in sensor_paths],
                # These task sensors only consume pair forces. The recorder
                # owns the separate full contact report and its strict audit.
                max_contact_data_count=0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"YCB11 could not create required physical {label} contact view"
            ) from exc
        if (
            view is None
            or int(getattr(view, "sensor_count", 0)) != len(sensor_paths)
            or int(getattr(view, "filter_count", 0)) != 1
            or set(view.sensor_paths) != set(sensor_paths)
        ):
            raise RuntimeError(
                f"YCB11 physical {label} contact view has an invalid layout"
            )
        return view

    def _init_task_specific_runtime(self) -> None:
        # This hook runs immediately after the first reset and *before* the
        # shared ABC captures its auxiliary Torch task-state view.  The
        # IsaacLab robot owns a live PhysX tensor view at this point; it uses
        # the same backend and is the correct bootstrap source for the current
        # volume body view.
        physics_sim_view = (
            next(iter(self._robots.values()))._physics_sim_view
            if self._robots
            else None
        )
        if physics_sim_view is None:
            physics_sim_view = self._task_physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("YCB11 could not acquire a PhysX simulation view")
        try:
            volume_view = physics_sim_view.create_volume_deformable_body_view(
                YCB11_SPONGE_ROOT_PATH
            )
        except Exception as exc:
            raise RuntimeError(
                "YCB11 could not create the current volume-deformable body view"
            ) from exc
        if volume_view is None or int(getattr(volume_view, "count", 0)) != 1:
            raise RuntimeError(
                "YCB11 current volume-deformable body view has an invalid count: "
                f"{0 if volume_view is None else int(getattr(volume_view, 'count', 0))}"
            )
        self._ycb11_sponge = _YCB11CurrentVolumeBody(volume_view)
        if not self._ycb11_sponge.is_initialized:
            raise RuntimeError("YCB11 native volume-deformable sponge did not initialize")
        nodes = _tensor_numpy(self._ycb11_sponge.data.nodal_pos_w)[0]
        velocities = _tensor_numpy(self._ycb11_sponge.data.nodal_vel_w)[0]
        if (
            nodes.ndim != 2
            or nodes.shape[1:] != (3,)
            or nodes.shape[0] < 32
            or velocities.shape != nodes.shape
            or not np.all(np.isfinite(nodes))
        ):
            raise RuntimeError(
                f"YCB11 PhysX volume nodal state is invalid: {nodes.shape}"
            )
        self._ycb11_default_nodal_state = (
            self._ycb11_sponge.data.default_nodal_state_w.clone()
        )
        self._ycb11_deformable_view = _YCB11DeformableViewAdapter(
            self._ycb11_sponge
        )
        self._ycb11_deformable_indices = self._torch.arange(
            1,
            dtype=self._torch.long,
            device=self._isaac_device,
        )
        self._ycb11_rest_nodes_np = np.asarray(nodes, dtype=np.float64).copy()
        self._ycb11_nodes_np = self._ycb11_rest_nodes_np.copy()
        self._ycb11_velocities_np = np.asarray(velocities, dtype=np.float64).copy()
        self._ycb11_expected_node_count = int(nodes.shape[0])
        self._ycb11_rest_collision_nodes_np = np.asarray(_tensor_numpy(
            self._ycb11_sponge.root_physx_view.get_collision_nodal_positions()
        )[0], dtype=np.float64).copy()
        self._ycb11_rest_principal_span_m = principal_span(
            self._ycb11_rest_collision_nodes_np,
            rest_nodes=self._ycb11_rest_collision_nodes_np,
        )
        self._ycb11_initial_sponge_com_z = float(np.mean(nodes[:, 2]))
        self._ycb11_keypoint_indices = self._select_keypoints(nodes)

        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("YCB11 could not acquire the PhysX simulation view")
        mug_prim = self._find_rigid_body_prim(YCB11_MUG_BODY_NAME)
        table_prim = self._find_rigid_body_prim("table")
        if mug_prim is None or table_prim is None:
            raise RuntimeError("YCB11 mug/table rigid bodies are unavailable")
        mug_path = str(mug_prim.GetPath())
        table_path = str(table_prim.GetPath())
        left_paths = tuple(path for _name, path in self._hand_rigid_body_paths("left"))
        # One batched readback retains all 21 hand-body/mug pairs. Taking the
        # maximum of their vector norms is identical to the former 21 views,
        # without 21 GPU synchronizations on every 120 Hz substep.
        self._ycb11_left_mug_views = (
            self._validated_contact_view(
                physics_sim_view,
                left_paths,
                mug_path,
                label="left-hand-mug",
            ),
        )
        self._ycb11_mug_table_view = self._validated_contact_view(
            physics_sim_view,
            mug_path,
            table_path,
            label="mug-table-support",
        )
        if not self._ycb11_left_mug_views:
            raise RuntimeError("YCB11 hand contact sensors are incomplete")
        print(
            "[HandsOn/IsaacLab] YCB11 native volume sponge ready: "
            f"nodes={self._ycb11_expected_node_count}, "
            f"rest_span={self._ycb11_rest_principal_span_m:.6f}m, "
            f"left_mug_sensors={len(left_paths)} (batched), "
            "right_sponge_contact_source=native nodes + exact hand colliders",
            flush=True,
        )

    @staticmethod
    def _select_keypoints(nodes: np.ndarray) -> np.ndarray:
        nodes = np.asarray(nodes, dtype=np.float64)
        centered = nodes - np.mean(nodes, axis=0)
        _u, _s, axes_t = np.linalg.svd(centered, full_matrices=False)
        directions = (
            axes_t[0],
            -axes_t[0],
            axes_t[1],
            -axes_t[1],
            axes_t[2],
            -axes_t[2],
            axes_t[0] + axes_t[1],
            axes_t[0] - axes_t[1],
        )
        selected: list[int] = []
        for direction in directions:
            index = int(np.argmax(centered @ direction))
            if index not in selected:
                selected.append(index)
        radial = np.argsort(np.linalg.norm(centered, axis=1))[::-1]
        for index in radial:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= 8:
                break
        return np.asarray((selected + [0] * 8)[:8], dtype=np.int64)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = (
            float(self._random.uniform(0.0, 0.015)) if self.randomize else 0.0
        )
        if self.randomize:
            self._ycb11_initial_state = np.asarray(
                [
                    _finite_nonzero_uniform(self._random, -0.070, 0.070),
                    _finite_nonzero_uniform(self._random, -0.070, 0.090),
                    math.radians(
                        _finite_nonzero_uniform(self._random, -45.0, 45.0)
                    ),
                    _finite_nonzero_uniform(self._random, -0.085, 0.085),
                    _finite_nonzero_uniform(self._random, -0.075, 0.035),
                    math.radians(
                        _finite_nonzero_uniform(self._random, -60.0, 60.0)
                    ),
                    float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
        else:
            self._ycb11_initial_state = np.zeros(7, dtype=np.float64)
        self._ycb11_initial_state[6] = float(self.delta_h[0])
        self._ycb11_state_machine.reset()
        self._ycb11_joint_limit_excess_rad = 0.0
        self._ycb11_last_signals = YCB11Signals()
        self._ycb11_last_metrics = {}
        self._ycb11_readback_failed = False
        self._ycb11_readback_failure_reason = None
        self._reset_ycb11_substep_latches()

    def _apply_task_pose_randomization(self) -> None:
        if self._ycb11_sponge is None or self._ycb11_default_nodal_state is None:
            return
        self._clear_mug_reset_from_hands()
        mug_dx, mug_dy, mug_yaw, sponge_dx, sponge_dy, sponge_yaw, delta_h = (
            self._ycb11_initial_state
        )

        table_view = self._task_body_views.get("table")
        if table_view is not None and abs(float(delta_h)) > 0.0:
            table_pose = self._physics_body_pose("table")
            if table_pose is not None:
                table_pose = table_pose.copy()
                table_pose[2] += float(delta_h)
                self._set_task_body_world_pose(table_view, table_pose)

        mug_pose = np.asarray(
            [
                YCB11_MUG_NOMINAL_POSITION[0] + mug_dx,
                YCB11_MUG_NOMINAL_POSITION[1] + mug_dy,
                YCB11_MUG_NOMINAL_POSITION[2] + delta_h,
                math.cos(0.5 * mug_yaw),
                0.0,
                0.0,
                math.sin(0.5 * mug_yaw),
            ],
            dtype=np.float64,
        )
        mug_view = self._task_body_views.get(YCB11_MUG_BODY_NAME)
        if mug_view is None or not self._set_task_body_world_pose(mug_view, mug_pose):
            raise RuntimeError("YCB11 could not write the randomized YCB mug pose")
        self._ycb11_initial_mug_pose = mug_pose.copy()

        nodes = self._ycb11_rest_nodes_np.copy()
        center = np.mean(nodes, axis=0)
        cosine, sine = math.cos(float(sponge_yaw)), math.sin(float(sponge_yaw))
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        nodes = (nodes - center) @ rotation.T + center
        nodes += np.asarray([sponge_dx, sponge_dy, delta_h], dtype=np.float64)
        nodal_state = self._ycb11_default_nodal_state.clone()
        nodal_state[0, :, :3] = self._torch.as_tensor(
            nodes,
            dtype=nodal_state.dtype,
            device=nodal_state.device,
        )
        nodal_state[0, :, 3:] = 0.0
        self._ycb11_sponge.write_nodal_state_to_sim(nodal_state)
        self._record_authored_deformable_reset_write(
            "ycb11_sponge",
            position=nodal_state[0, :, :3],
            velocity=nodal_state[0, :, 3:],
        )
        self._ycb11_sponge.reset()
        self._ycb11_nodes_np = nodes.copy()
        self._ycb11_velocities_np = np.zeros_like(nodes)
        self._ycb11_initial_sponge_com_z = float(np.mean(nodes[:, 2]))
        self._ycb11_state_machine.reset()
        self._reset_ycb11_substep_latches()
        print(
            "[HandsOn/IsaacLab] YCB11 reset: "
            f"mug_delta=({mug_dx:+.4f},{mug_dy:+.4f})m/{math.degrees(mug_yaw):+.1f}deg, "
            f"sponge_delta=({sponge_dx:+.4f},{sponge_dy:+.4f})m/{math.degrees(sponge_yaw):+.1f}deg, "
            f"table_delta={delta_h:+.4f}m",
            flush=True,
        )

    def _clear_mug_reset_from_hands(self) -> None:
        """Sample a clear mug pose before settling, using reset-home hands.

        Negative-X samples can put the rim inside the left thumb. The drive
        then keeps the mug moving throughout the strict origin proof. Reject
        such layouts before writing the initial pose; the physical gate still
        runs unchanged afterwards. The accepted XY is part of initial_state.
        """
        if not self._robots:
            return
        centers, radii = [], []
        for side, hulls in self._hand_contact_hulls().items():
            poses = _tensor_numpy(self._robots[side].data.body_pose_w)[0]
            for index, points, _planes, _group in hulls:
                pose = poses[index]
                rotation = _quat_wxyz_to_matrix(pose[3:7])
                center = (points.min(0) + points.max(0)) / 2
                centers.append(pose[:3] + rotation @ center)
                # A circumsphere covers the entire authored convex component,
                # including its faces, for both object reset-clearance tests.
                radii.append(float(np.linalg.norm(points - center, axis=1).max()))
        if not centers:
            raise RuntimeError("YCB11 reset cannot read the physical hand layout")
        centers_np = np.asarray(centers)
        radii_np = np.asarray(radii)
        yaw = float(self._ycb11_initial_state[2])
        pose = np.array([
            *YCB11_MUG_NOMINAL_POSITION,
            math.cos(yaw / 2), 0., 0., math.sin(yaw / 2),
        ])
        pose[2] += float(self._ycb11_initial_state[6])
        for attempt in range(65):
            pose[:2] = YCB11_MUG_NOMINAL_POSITION[:2] + self._ycb11_initial_state[:2]
            # A conservative box enclosing the original YCB mug, including
            # its handle; this box is only a reset-layout test, never a collider.
            clearance = np.min(_box_signed_distances(
                centers_np, pose, center_local_m=(0., .0013, -.0004),
                half_extents_m=(.041, .054, .043),
            ) - radii_np)
            if np.isfinite(clearance) and clearance >= .003:
                # Sample the sponge independently, keeping the rotated soft
                # body's complete envelope clear of both hands and the mug.
                for sponge_attempt in range(128):
                    delta = self._ycb11_initial_state[3:5]
                    sponge_center = self._ycb11_rest_nodes_np.mean(axis=0) + np.array(
                        [*delta, self._ycb11_initial_state[6]])
                    sponge_yaw = self._ycb11_initial_state[5]
                    sponge_pose = np.array([*sponge_center, math.cos(sponge_yaw/2),
                                            0., 0., math.sin(sponge_yaw/2)])
                    # The node mean is the rotation pivot, which need not be
                    # the bounds midpoint for a nonuniform tetrahedral mesh.
                    # Cover every rest node about that actual pivot.
                    half_extents = np.max(np.abs(
                        self._ycb11_rest_nodes_np - self._ycb11_rest_nodes_np.mean(axis=0)), axis=0)
                    separated = np.linalg.norm(pose[:2] - sponge_center[:2]) >= (
                        .069 + np.linalg.norm(half_extents[:2]) + .010)
                    sponge_clearance = np.min(_box_signed_distances(
                        centers_np, sponge_pose, center_local_m=(0., 0., 0.),
                        half_extents_m=half_extents,
                    ) - radii_np)
                    if separated and np.isfinite(sponge_clearance) and sponge_clearance >= .003:
                        return
                    if not self.randomize:
                        break
                    self._ycb11_initial_state[3:5] = self._random.uniform(
                        [-.085, -.075], [.085, .035])
                    self._ycb11_initial_state[5] = math.radians(self._random.uniform(-60., 60.))
            if not self.randomize or attempt == 64:
                raise RuntimeError("YCB11 could not sample a mug pose clear of the reset hands")
            self._ycb11_initial_state[:2] = self._random.uniform(
                [-.070, -.070], [.070, .090],
            )

    def _configure_task_reset_stability(self, sim_utils: Any) -> int:
        """Give the flat-bottomed mug a narrow, wakeable table contact."""

        configured = super()._configure_task_reset_stability(sim_utils)
        if self.task_name != YCB11_TASK_NAME or self._stage is None:
            return configured
        configured += self._configure_named_reset_stability(
            sim_utils,
            (YCB11_MUG_BODY_NAME,),
            linear_damping=YCB11_MUG_RESET_LINEAR_DAMPING,
            angular_damping=YCB11_MUG_RESET_ANGULAR_DAMPING,
            sleep_threshold=YCB11_MUG_RESET_SLEEP_THRESHOLD,
            warning_label="YCB11 source-scale YCB mug",
            strip_redundant_articulation_root=True,
        )

        from pxr import PhysxSchema, Usd, UsdPhysics

        mug_prim = self._find_rigid_body_prim(YCB11_MUG_BODY_NAME)
        if mug_prim is None:
            return configured
        for descendant in Usd.PrimRange(mug_prim):
            if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
            if not collision_api:
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
            contact = collision_api.GetContactOffsetAttr()
            if not contact or not contact.IsValid():
                contact = collision_api.CreateContactOffsetAttr()
            contact.Set(YCB11_MUG_CONTACT_OFFSET_M)
            rest = collision_api.GetRestOffsetAttr()
            if not rest or not rest.IsValid():
                rest = collision_api.CreateRestOffsetAttr()
            rest.Set(YCB11_MUG_REST_OFFSET_M)
        return configured

    def _contact_force(self, view: Any, dt: float) -> float:
        if view is None:
            return 0.0
        try:
            matrix = _tensor_numpy(view.get_contact_force_matrix(float(dt)))
            return float(np.max(np.linalg.norm(matrix, axis=-1)))
        except Exception:
            # Contact tensors are an auxiliary semantic signal.  Isaac Sim
            # 5.1 can briefly invalidate a rigid contact view while a
            # deformable body enters the narrowphase.  Let the frame-level
            # health/readback path report that condition instead of turning a
            # recoverable contact sample into an uncaught process exit.
            self._ycb11_contact_readback_failed = True
            return 0.0

    def _reset_ycb11_substep_latches(self) -> None:
        """Clear the low-cost contact summary for the next control frame."""

        self._ycb11_substep_count = 0
        self._ycb11_substep_left_mug_contact_all = True
        self._ycb11_substep_left_mug_contact_any = False
        self._ycb11_substep_mug_table_contact_all = True
        self._ycb11_substep_mug_table_contact_any = False
        self._ycb11_substep_contact_readback_failed = False
        self._ycb11_contact_readback_failed = False
        self._ycb11_substep_last_left_mug_forces = ()
        self._ycb11_substep_last_mug_table_force = 0.0

    def _latch_ycb11_substep_contacts(self) -> None:
        """Sample only rigid contact tensors at 120 Hz.

        The expensive native-volume node/velocity readback and CPU
        deformation SVD are intentionally absent here.  ``all`` preserves
        the state machine's continuous-contact semantics when the summary is
        consumed at the control boundary; ``any`` is retained for diagnostics
        so a brief contact is still visible even if it does not satisfy a
        hold condition.
        """

        self._ycb11_substep_count += 1
        self._ycb11_contact_readback_failed = False
        try:
            left_forces = [
                self._contact_force(view, self._sim_dt)
                for view in self._ycb11_left_mug_views
            ]
            left_contact = bool(
                left_forces
                and max(left_forces) >= _YCB11_MUG_CONTACT_FORCE_N
            )
            mug_table_force = self._contact_force(
                self._ycb11_mug_table_view, self._sim_dt
            )
            mug_table_contact = bool(
                mug_table_force >= _YCB11_MUG_CONTACT_FORCE_N
            )
        except Exception:
            # The full control-frame readback remains the authoritative error
            # path.  Keep the latch conservative and let that path report the
            # exact failure without interrupting a physics substep.
            self._ycb11_substep_contact_readback_failed = True
            return

        if self._ycb11_contact_readback_failed:
            self._ycb11_substep_contact_readback_failed = True
            return

        if self._ycb11_substep_count == 1:
            self._ycb11_substep_left_mug_contact_all = left_contact
            self._ycb11_substep_mug_table_contact_all = mug_table_contact
        else:
            self._ycb11_substep_left_mug_contact_all &= left_contact
            self._ycb11_substep_mug_table_contact_all &= mug_table_contact
        self._ycb11_substep_left_mug_contact_any |= left_contact
        self._ycb11_substep_mug_table_contact_any |= mug_table_contact
        self._ycb11_substep_last_left_mug_forces = tuple(left_forces)
        self._ycb11_substep_last_mug_table_force = float(mug_table_force)

    def _robot_hard_limit_failure(self) -> bool:
        self._ycb11_joint_limit_excess_rad = 0.0
        for robot in self._robots.values():
            try:
                positions = _tensor_numpy(robot.data.joint_pos)[0]
                limits = _tensor_numpy(robot.data.joint_pos_limits)[0]
                if not np.all(np.isfinite(positions)):
                    self._ycb11_joint_limit_excess_rad = math.inf
                    return True
                self._ycb11_joint_limit_excess_rad = max(
                    self._ycb11_joint_limit_excess_rad,
                    float(np.max(np.maximum(limits[:, 0] - positions, positions - limits[:, 1]))),
                )
            except Exception:
                continue
        return self._ycb11_joint_limit_excess_rad > DEFAULT_THRESHOLDS.failure_joint_limit_hard_excess_rad

    def _hand_contact_hulls(self) -> dict[str, list[tuple]]:
        """Cache the very same local convex geometry authored for PhysX."""
        if self._ycb11_hand_hulls is not None:
            return self._ycb11_hand_hulls
        from pxr import UsdGeom
        from scipy.spatial import ConvexHull

        hulls: dict[str, list[tuple]] = {"right": [], "left": []}
        for row in self._ycb11_hand_collider_audit:
            side = "right" if "/RightRobot/" in row["collider"] else "left"
            prim = self._stage.GetPrimAtPath(row["collider"])
            name = prim.GetParent().GetName()
            index = self._robots[side].body_names.index(name)
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            group = (
                "thumb" if "finger1" in name
                else "opposing" if "finger" in name else None
            )
            hulls[side].append((index, points, ConvexHull(points).equations, group))
        self._ycb11_hand_hulls = hulls
        return hulls

    def _hand_sponge_geometry(self, nodes: np.ndarray, side: str) -> dict[str, float]:
        """Measure live nodes against the actual CAD-derived contact hulls.

        Rigid/deformable forces are unavailable through the 5.1 rigid-contact
        tensor. Convex supporting-plane gaps supply geometry only; measured
        deformation, lift and the ordered task events still qualify a grasp.
        """
        result = {
            "thumb_clearance_m": math.inf,
            "opposing_clearance_m": math.inf,
            "hand_clearance_m": 0.0,
            "thumb_near_nodes": 0.0,
            "opposing_near_nodes": 0.0,
        }
        robot = self._robots.get(side)
        if robot is None:
            return result
        body_poses = _tensor_numpy(robot.data.body_pose_w)[0]
        group_distances: dict[str, list[np.ndarray]] = {
            "thumb": [], "opposing": [], "all": [],
        }
        for index, _points, planes, group in self._hand_contact_hulls()[side]:
            pose = body_poses[index]
            local = (nodes - pose[:3]) @ _quat_wxyz_to_matrix(pose[3:7])
            distances = np.max(local @ planes[:, :3].T + planes[:, 3], axis=1)
            group_distances["all"].append(distances)
            if group is not None:
                group_distances[group].append(distances)

        for group in ("thumb", "opposing"):
            if not group_distances[group]:
                continue
            closest_per_node = np.min(np.stack(group_distances[group]), axis=0)
            result[f"{group}_clearance_m"] = max(
                0.0, float(np.min(closest_per_node))
            )
            result[f"{group}_near_nodes"] = float(
                np.count_nonzero(
                    closest_per_node
                    <= _YCB11_SPONGE_CONTACT_SURFACE_TOLERANCE_M
                )
            )
        if group_distances["all"]:
            result["hand_clearance_m"] = max(
                0.0,
                float(np.min(np.stack(group_distances["all"]))),
            )
        return result

    def _minimum_hand_distance_to_mug(
        self, side: str, mug_pose: np.ndarray
    ) -> float:
        points = self._a06_hand_points(side)
        if not points:
            return 0.0
        center_distance = min(
            float(np.linalg.norm(point - mug_pose[:3])) for point in points
        )
        return max(0.0, center_distance - _YCB11_MUG_BOUNDING_RADIUS_M)

    def _deformation_diagnostics(self) -> tuple[float, float]:
        if self._ycb11_sponge is None:
            return -math.inf, math.inf
        # Torch's batched CUDA determinant/SVD kernels fail with
        # CUDA_ERROR_INVALID_VALUE under the production Isaac Sim 5.1
        # CUDA/Vulkan interop configuration.  PhysX's public gradients are
        # small 3x3 matrices; copying them to CPU keeps the truth source the
        # same while making the health check deterministic and portable.
        gradient = np.asarray(
            _tensor_numpy(
                self._ycb11_sponge.data.sim_element_deform_gradient_w
            )[0],
            dtype=np.float64,
        )
        if gradient.ndim != 3 or gradient.shape[1:] != (3, 3):
            raise RuntimeError(
                f"YCB11 deformation-gradient layout is invalid: {gradient.shape}"
            )
        determinants = np.linalg.det(gradient)
        stretches = np.linalg.svd(gradient, compute_uv=False)
        if not np.all(np.isfinite(determinants)) or not np.all(
            np.isfinite(stretches)
        ):
            raise RuntimeError("YCB11 deformation gradients are non-finite")
        return (
            float(np.min(determinants)),
            float(np.max(stretches)),
        )

    def _refresh_ycb11_metrics(self, *, update_state_machine: bool) -> None:
        if self._ycb11_sponge is None:
            return
        substep_count = int(self._ycb11_substep_count)
        substep_contact_readback_failed = bool(
            self._ycb11_substep_contact_readback_failed
        )
        try:
            nodes = np.asarray(
                _tensor_numpy(self._ycb11_sponge.data.nodal_pos_w)[0],
                dtype=np.float64,
            )
            velocities = np.asarray(
                _tensor_numpy(self._ycb11_sponge.data.nodal_vel_w)[0],
                dtype=np.float64,
            )
            valid = bool(
                nodes.shape == (self._ycb11_expected_node_count, 3)
                and velocities.shape == nodes.shape
                and np.all(np.isfinite(nodes))
                and np.all(np.isfinite(velocities))
            )
            if not valid:
                raise ValueError(
                    f"node/velocity shape or finiteness changed: {nodes.shape}/{velocities.shape}"
                )
            self._ycb11_nodes_np = nodes
            self._ycb11_velocities_np = velocities
            self._ycb11_readback_failed = False
            self._ycb11_readback_failure_reason = None
        except Exception as exc:
            self._ycb11_readback_failed = True
            self._ycb11_readback_failure_reason = str(exc)

        nodes = self._ycb11_nodes_np
        velocities = self._ycb11_velocities_np
        mug_pose = self._physics_body_pose(YCB11_MUG_BODY_NAME)
        if mug_pose is None:
            mug_pose = self._ycb11_initial_mug_pose.copy()
        mug_velocity = self._physics_body_velocity(YCB11_MUG_BODY_NAME)
        if mug_velocity is None:
            mug_velocity = np.full(6, math.inf, dtype=np.float64)

        # Geometry follows the source-derived collision volume. Read it
        # directly so containment and fingertip contact use the actual shape.
        surface_nodes = nodes
        try:
            surface_nodes = np.asarray(_tensor_numpy(
                self._ycb11_sponge.root_physx_view.get_collision_nodal_positions()
            )[0], dtype=np.float64)
            if (surface_nodes.ndim != 2 or surface_nodes.shape[1:] != (3,)
                    or len(surface_nodes) < 8 or not np.all(np.isfinite(surface_nodes))):
                raise ValueError("YCB11 collision surface readback is invalid")
        except Exception as exc:
            self._ycb11_readback_failed = True
            self._ycb11_readback_failure_reason = str(exc)
            surface_nodes = nodes
        containment = containment_metrics(surface_nodes, mug_pose)
        rest_surface = self._ycb11_rest_collision_nodes_np
        if surface_nodes.shape != rest_surface.shape:
            # A failed collision readback is already latched above. Keep the
            # failure metrics finite until the structured terminal is emitted.
            rest_surface = self._ycb11_rest_nodes_np
        ratio = compression_ratio(
            surface_nodes, self._ycb11_rest_principal_span_m,
            rest_nodes=rest_surface,
        )
        nodal_speed = np.linalg.norm(velocities, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(nodal_speed))))
        max_speed = float(np.max(nodal_speed))
        try:
            min_det, max_stretch = self._deformation_diagnostics()
        except Exception as exc:
            min_det, max_stretch = -math.inf, math.inf
            self._ycb11_readback_failed = True
            self._ycb11_readback_failure_reason = str(exc)

        if (
            substep_count > 0
            and not substep_contact_readback_failed
            and len(self._ycb11_substep_last_left_mug_forces)
            == len(self._ycb11_left_mug_views)
        ):
            # Reuse the final substep's contact tensor readback.  Besides
            # avoiding a duplicate synchronization, this keeps the metrics'
            # force values and the latch decision on the same PhysX sample.
            left_mug_forces = list(self._ycb11_substep_last_left_mug_forces)
            mug_table_force = float(self._ycb11_substep_last_mug_table_force)
        else:
            # The substep latch is deliberately conservative when contact
            # readback is unavailable.  Reusing the safe zero-force fallback
            # here keeps the control frame alive while the deformable node
            # readback remains authoritative for task health.
            left_mug_forces = [
                self._contact_force(view, self._sim_dt)
                for view in self._ycb11_left_mug_views
            ]
            mug_table_force = self._contact_force(
                self._ycb11_mug_table_view, self._sim_dt
            )
            substep_contact_readback_failed = bool(
                substep_contact_readback_failed
                or self._ycb11_contact_readback_failed
            )
        left_contact = bool(
            left_mug_forces
            and max(left_mug_forces) >= _YCB11_MUG_CONTACT_FORCE_N
        )
        hand_geometry = self._hand_sponge_geometry(surface_nodes, "right")
        grasp_deformation = ratio <= _YCB11_GRASP_DEFORMATION_RATIO
        thumb_contact = bool(
            grasp_deformation
            and hand_geometry["thumb_near_nodes"]
            >= _YCB11_SPONGE_CONTACT_MIN_NODES
        )
        opposing_contact = bool(
            grasp_deformation
            and hand_geometry["opposing_near_nodes"]
            >= _YCB11_SPONGE_CONTACT_MIN_NODES
        )
        mug_table_contact = mug_table_force >= _YCB11_MUG_CONTACT_FORCE_N
        right_distance = hand_geometry["hand_clearance_m"]
        left_distance = self._minimum_hand_distance_to_mug("left", mug_pose)
        right_mug_distance = self._minimum_hand_distance_to_mug("right", mug_pose)
        left_sponge_distance = self._hand_sponge_geometry(surface_nodes, "left")["hand_clearance_m"]
        tilt = mug_tilt_deg(mug_pose)
        mug_displacement = float(
            np.linalg.norm(mug_pose[:2] - self._ycb11_initial_mug_pose[:2])
        )
        table_top = YCB11_TABLE_TOP_Z + float(self.delta_h[0])
        sponge_com = np.mean(nodes, axis=0)
        sponge_out = bool(
            sponge_com[0] < _YCB11_SPONGE_WORKSPACE_X[0]
            or sponge_com[0] > _YCB11_SPONGE_WORKSPACE_X[1]
            or sponge_com[1] < _YCB11_SPONGE_WORKSPACE_Y[0]
            or sponge_com[1] > _YCB11_SPONGE_WORKSPACE_Y[1]
            or float(np.min(nodes[:, 2])) < table_top - 0.05
            or float(np.max(nodes[:, 2])) > _YCB11_MAX_WORLD_Z
        )
        mug_out = bool(
            mug_pose[0] < _YCB11_SPONGE_WORKSPACE_X[0]
            or mug_pose[0] > _YCB11_SPONGE_WORKSPACE_X[1]
            or mug_pose[1] < _YCB11_SPONGE_WORKSPACE_Y[0]
            or mug_pose[1] > _YCB11_SPONGE_WORKSPACE_Y[1]
            or mug_pose[2] < table_top - 0.10
            or mug_pose[2] > _YCB11_MAX_WORLD_Z
        )
        ejection = bool(
            self._ycb11_state_machine.stage >= YCB11Stage.PACKED
            and float(containment["inside_fraction"]) < 0.70
        )
        signals = YCB11Signals(
            left_mug_contact=left_contact,
            right_thumb_sponge_contact=thumb_contact,
            right_opposing_sponge_contact=opposing_contact,
            sponge_lift_m=float(sponge_com[2] - self._ycb11_initial_sponge_com_z),
            compression_ratio=float(ratio),
            rim_crossed=bool(containment["rim_crossed"]),
            packed=bool(containment["packed"]),
            mug_table_contact=mug_table_contact,
            mug_tilt_deg=float(tilt),
            mug_displacement_m=mug_displacement,
            right_released_sponge=(
                right_distance >= DEFAULT_THRESHOLDS.release_hand_clearance_m
            ),
            left_released_mug=not left_contact,
            right_hand_sponge_distance_m=right_distance,
            left_hand_mug_distance_m=left_distance,
            right_hand_mug_distance_m=right_mug_distance,
            left_hand_sponge_distance_m=left_sponge_distance,
            mug_linear_speed_mps=float(np.linalg.norm(mug_velocity[:3])),
            mug_angular_speed_radps=float(np.linalg.norm(mug_velocity[3:])),
            sponge_rms_speed_mps=rms_speed,
            ejection_detected=ejection,
            mug_out_of_workspace=mug_out,
            sponge_out_of_workspace=sponge_out,
            deformable_state_invalid=self._ycb11_readback_failed,
            max_nodal_speed_mps=max_speed,
            min_deformation_determinant=min_det,
            max_principal_stretch=max_stretch,
            robot_safety_failure=self._robot_hard_limit_failure(),
            robot_joint_limit_excess_rad=self._ycb11_joint_limit_excess_rad,
        )
        if update_state_machine and substep_count > 0:
            # A contact that was present for every substep is the equivalent
            # of the old per-substep state-machine updates.  Using ``all``
            # avoids turning a one-substep touch into a full control-frame
            # hold, while the ``any`` counters below retain short events in
            # the diagnostic stream.
            if not substep_contact_readback_failed:
                signals = replace(
                    signals,
                    left_mug_contact=self._ycb11_substep_left_mug_contact_all,
                    left_released_mug=(
                        not self._ycb11_substep_left_mug_contact_all
                    ),
                    mug_table_contact=(
                        self._ycb11_substep_mug_table_contact_all
                    ),
                )
        self._ycb11_last_signals = signals
        self._ycb11_last_metrics = {
            "mug_pose": mug_pose.copy(),
            "mug_velocity": mug_velocity.copy(),
            "sponge_com": sponge_com.copy(),
            "sponge_aabb": np.concatenate(
                [np.min(nodes, axis=0), np.max(nodes, axis=0)]
            ),
            "containment": dict(containment),
            "compression_ratio": float(ratio),
            "principal_span_m": principal_span(
                surface_nodes, rest_nodes=rest_surface,
            ),
            "rest_principal_span_m": self._ycb11_rest_principal_span_m,
            "rms_nodal_speed_mps": rms_speed,
            "max_nodal_speed_mps": max_speed,
            "min_deformation_determinant": min_det,
            "max_principal_stretch": max_stretch,
            "left_mug_contact_force_n": max(left_mug_forces, default=0.0),
            "right_thumb_collider_clearance_m": hand_geometry[
                "thumb_clearance_m"
            ],
            "right_opposing_collider_clearance_m": hand_geometry[
                "opposing_clearance_m"
            ],
            "right_thumb_near_node_count": hand_geometry["thumb_near_nodes"],
            "right_opposing_near_node_count": hand_geometry[
                "opposing_near_nodes"
            ],
            "grasp_deformation_ratio_threshold": _YCB11_GRASP_DEFORMATION_RATIO,
            "mug_table_contact_force_n": mug_table_force,
            "right_hand_sponge_distance_m": right_distance,
            "left_hand_mug_distance_m": left_distance,
            "right_hand_mug_distance_m": right_mug_distance,
            "left_hand_sponge_distance_m": left_sponge_distance,
            "mug_tilt_deg": tilt,
            "mug_displacement_m": mug_displacement,
            "ejection_detected": bool(ejection),
            "readback_failed": bool(self._ycb11_readback_failed),
            "readback_failure_reason": self._ycb11_readback_failure_reason,
            "substep_count": substep_count,
            "substep_left_mug_contact_any": bool(
                self._ycb11_substep_left_mug_contact_any
            ),
            "substep_left_mug_contact_all": bool(
                self._ycb11_substep_left_mug_contact_all
            ),
            "substep_mug_table_contact_any": bool(
                self._ycb11_substep_mug_table_contact_any
            ),
            "substep_mug_table_contact_all": bool(
                self._ycb11_substep_mug_table_contact_all
            ),
            "substep_contact_readback_failed": substep_contact_readback_failed,
            "contact_readback_failed": bool(self._ycb11_contact_readback_failed),
        }
        if update_state_machine and substep_count > 0:
            self._ycb11_state_machine.update(
                signals,
                self._sim_dt * substep_count,
            )
            self._reset_ycb11_substep_latches()
        state_snapshot = self._ycb11_state_machine.snapshot()
        self._ycb11_last_metrics.update(
            {
                "stage": state_snapshot["stage"],
                "stage_name": state_snapshot["stage_name"],
                "failure_reason": state_snapshot["failure_reason"],
                "left_mug_contact_s": state_snapshot["left_contact_s"],
                "stable_s": state_snapshot["stable_s"],
                "success": bool(state_snapshot["success"]),
                "failed": bool(state_snapshot["failed"]),
            }
        )

    def _after_task_physics_substep(self) -> None:
        if self._ycb11_sponge is None:
            return
        self._ycb11_sponge.update(self._sim_dt)
        self._latch_ycb11_substep_contacts()

    def _refresh_object_poses(self) -> None:
        super()._refresh_object_poses()
        if self._ycb11_sponge is not None:
            # This is called once after all fixed-rate PhysX substeps for a
            # control action (and once after reset settling).  Keep the
            # complete node/velocity/gradient/geometry readback here so the
            # observation and replay contract still expose the final physical
            # state without synchronizing eight times per action.
            self._refresh_ycb11_metrics(update_state_machine=True)

    def _compute_success_now(self) -> bool:
        return self._ycb11_state_machine.success

    def _compute_failure(self) -> bool:
        return self._ycb11_state_machine.failed

    def _success_steps_required(self) -> int:
        """The state machine owns the one-second physical stability hold."""

        return 1

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        """Keep the live YCB11 panel labels aligned with the task contract."""

        key = str(row.get("key") or "")
        if key.startswith("ycb11_") and row.get("label"):
            return str(row["label"])
        return HandsOnABCIsaacLabEnv._success_condition_display_label(row)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        """Expose the latched physical YCB11 milestones to the Isaac overlay."""

        snapshot = self._ycb11_state_machine.snapshot()
        metrics = self._ycb11_last_metrics or {}
        containment = metrics.get("containment", {})
        stage = int(snapshot.get("stage", int(YCB11Stage.APPROACH)))
        packed_now = bool(containment.get("packed", False))
        return [
            self._condition_row(
                "ycb11_left_mug_stabilized",
                YCB11_SUCCESS_CONDITION_LABELS[0],
                stage >= int(YCB11Stage.MUG_STABILIZED),
            ),
            self._condition_row(
                "ycb11_right_sponge_grasped",
                YCB11_SUCCESS_CONDITION_LABELS[1],
                stage >= int(YCB11Stage.SPONGE_GRASPED),
            ),
            self._condition_row(
                "ycb11_compressed_before_rim",
                YCB11_SUCCESS_CONDITION_LABELS[2],
                stage >= int(YCB11Stage.SPONGE_COMPRESSED),
            ),
            self._condition_row(
                "ycb11_packed_inside_mug",
                YCB11_SUCCESS_CONDITION_LABELS[3],
                stage >= int(YCB11Stage.PACKED) and packed_now,
            ),
            self._condition_row(
                "ycb11_released_stable",
                YCB11_SUCCESS_CONDITION_LABELS[4],
                bool(snapshot.get("success", False)),
            ),
        ]

    def _task_status_text(self) -> str:
        """Return compact live progress/failure text for the operator panel."""

        snapshot = self._ycb11_state_machine.snapshot()
        metrics = self._ycb11_last_metrics or {}
        containment = metrics.get("containment", {})

        def _fmt(value: Any, digits: int = 2) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "--"
            return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "--"

        stage_name = str(snapshot.get("stage_name", "approach")).replace("_", " ")
        failure = snapshot.get("failure_reason") or "none"
        return (
            f"Stage {stage_name} | mug contact "
            f"{_fmt(snapshot.get('left_contact_s', 0.0))}/"
            f"{DEFAULT_THRESHOLDS.mug_contact_hold_s:.2f}s | compression "
            f"{_fmt(metrics.get('compression_ratio', math.inf))}/"
            f"{DEFAULT_THRESHOLDS.max_compression_ratio:.2f} | inside "
            f"{100.0 * float(containment.get('inside_fraction', 0.0)):.0f}% | "
            f"stable {_fmt(snapshot.get('stable_s', 0.0))}/"
            f"{DEFAULT_THRESHOLDS.stable_hold_s:.2f}s | failure {failure}"
        )

    def _task_info(self) -> dict[str, Any]:
        snapshot = self._ycb11_state_machine.snapshot()
        return {
            "task_id": self._TASK_ID,
            "task_name": YCB11_TASK_NAME,
            "ycb11_stage": snapshot["stage"],
            "ycb11_stage_name": snapshot["stage_name"],
            "ycb11_event_times_s": snapshot["event_times_s"],
            "ycb11_failure_reason": snapshot["failure_reason"],
            "ycb11_state": snapshot,
            "ycb11_metrics": dict(self._ycb11_last_metrics),
            "ycb11_ycb_only": False,
            "ycb11_sponge_source": "plaggy_sponge3_cc_by_4_0",
            "ycb11_source_scale": 1.0,
            "ycb11_physics_hz": int(round(1.0 / float(self._sim_dt))),
            "ycb11_control_hz": int(round(1.0 / float(self._control_dt))),
            "failure_reason": snapshot["failure_reason"],
            "terminal_reason": snapshot["failure_reason"],
            "robot_joint_limit_excess_rad": getattr(self, "_ycb11_joint_limit_excess_rad", 0.0),
        }

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces: dict[str, spaces.Space] = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "mug_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "mug_velocity": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "sponge_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "sponge_keypoints": spaces.Box(
                -np.inf, np.inf, shape=(24,), dtype=np.float64
            ),
            "sponge_aabb": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "sponge_nodal_velocity_summary": spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float64
            ),
            "compression_metrics": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "containment_metrics": spaces.Box(
                -np.inf, np.inf, shape=(8,), dtype=np.float64
            ),
            "contact_metrics": spaces.Box(
                -np.inf, np.inf, shape=(8,), dtype=np.float64
            ),
            "release_metrics": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "task_phase": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float64),
            "robot_limit_state": spaces.Box(0.0, np.inf, shape=(2,), dtype=np.float64),
            "task_events": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float64),
            "task_initial_state": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "table_delta_height": spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float64
            ),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces)
        }
        if self.image_obs:
            wrist_width, wrist_height = (640, 480)
            try:
                from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE

                wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            except Exception:
                pass
            wrist_shape = (wrist_height, wrist_width, 3)
            observation_spaces["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(
                        0, 255, shape=wrist_shape, dtype=np.uint8
                    ),
                    "wrist_right": spaces.Box(
                        0, 255, shape=wrist_shape, dtype=np.uint8
                    ),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0, 255, shape=self._image_shape, dtype=np.uint8
                    ),
                }
            )
        return spaces.Dict(observation_spaces)

    def _compute_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        if self.image_obs:
            ego, wrist_left, wrist_right = self.render()
            obs["images"] = {
                "random_camera" if self.randomize else "ego": ego,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        gripper_pose = np.zeros(32, dtype=np.float64)
        gripper_pose[: min(16, self._right_hand.size)] = self._right_hand[
            : min(16, self._right_hand.size)
        ]
        gripper_pose[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[
            : min(16, self._left_hand.size)
        ]
        metrics = self._ycb11_last_metrics
        nodes = self._ycb11_nodes_np
        velocities = self._ycb11_velocities_np
        mug_pose = np.asarray(
            metrics.get("mug_pose", self._ycb11_initial_mug_pose),
            dtype=np.float64,
        )
        mug_velocity = np.asarray(
            metrics.get("mug_velocity", np.zeros(6)), dtype=np.float64
        )
        sponge_com = np.mean(nodes, axis=0)
        sponge_pose = np.asarray(
            [*sponge_com, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        containment = metrics.get("containment", {})
        snapshot = self._ycb11_state_machine.snapshot()
        stage = int(snapshot["stage"])
        phase = np.zeros(8, dtype=np.float64)
        phase[min(max(stage, 0), 7)] = 1.0
        events = np.asarray([float(stage >= index) for index in range(8)], dtype=np.float64)
        speed = np.linalg.norm(velocities, axis=1)
        state = {
            "tcp_pose": np.concatenate(
                [self._right_tcp_pose, self._left_tcp_pose]
            ).astype(np.float64),
            "gripper_pose": gripper_pose,
            "mug_pose": mug_pose,
            "mug_velocity": mug_velocity,
            "sponge_pose": sponge_pose,
            "sponge_keypoints": nodes[self._ycb11_keypoint_indices]
            .reshape(-1)
            .astype(np.float64),
            "sponge_aabb": np.asarray(
                metrics.get(
                    "sponge_aabb",
                    np.concatenate([np.min(nodes, axis=0), np.max(nodes, axis=0)]),
                ),
                dtype=np.float64,
            ),
            "sponge_nodal_velocity_summary": np.asarray(
                [
                    float(np.sqrt(np.mean(np.square(speed)))),
                    float(np.max(speed)),
                    float(np.mean(speed)),
                    float(np.percentile(speed, 95.0)),
                ],
                dtype=np.float64,
            ),
            "compression_metrics": np.asarray(
                [
                    metrics.get("rest_principal_span_m", 0.0),
                    metrics.get("principal_span_m", 0.0),
                    metrics.get("compression_ratio", math.inf),
                    DEFAULT_THRESHOLDS.max_compression_ratio,
                    float(stage >= int(YCB11Stage.SPONGE_COMPRESSED)),
                    metrics.get("max_principal_stretch", math.inf),
                ],
                dtype=np.float64,
            ),
            "containment_metrics": np.asarray(
                [
                    containment.get("inside_fraction", 0.0),
                    containment.get("below_rim_fraction", 0.0),
                    containment.get("com_local_z_m", math.inf),
                    containment.get("com_depth_below_rim_m", -math.inf),
                    containment.get("min_local_z_m", math.inf),
                    containment.get("max_local_z_m", math.inf),
                    float(containment.get("rim_crossed", False)),
                    float(containment.get("packed", False)),
                ],
                dtype=np.float64,
            ),
            "contact_metrics": np.asarray(
                [
                    metrics.get("left_mug_contact_force_n", 0.0),
                    metrics.get("right_thumb_collider_clearance_m", math.inf),
                    metrics.get("right_opposing_collider_clearance_m", math.inf),
                    metrics.get("mug_table_contact_force_n", 0.0),
                    float(self._ycb11_last_signals.left_mug_contact),
                    float(self._ycb11_last_signals.right_thumb_sponge_contact),
                    float(self._ycb11_last_signals.right_opposing_sponge_contact),
                    float(self._ycb11_last_signals.mug_table_contact),
                ],
                dtype=np.float64,
            ),
            "release_metrics": np.asarray(
                [
                    metrics.get("right_hand_sponge_distance_m", 0.0),
                    metrics.get("left_hand_mug_distance_m", 0.0),
                    float(self._ycb11_last_signals.right_released_sponge),
                    float(self._ycb11_last_signals.left_released_mug),
                    metrics.get("mug_tilt_deg", math.inf),
                    metrics.get("mug_displacement_m", math.inf),
                ],
                dtype=np.float64,
            ),
            "task_phase": phase,
            "task_events": events,
            "task_initial_state": self._ycb11_initial_state.copy(),
            "robot_limit_state": np.asarray([
                getattr(self, "_ycb11_joint_limit_excess_rad", 0.0),
                self._ycb11_state_machine.joint_limit_excursion_s,
            ], dtype=np.float64),
            "table_delta_height": self.delta_h.copy(),
        }
        obs["state"] = state
        return obs

    def step(self, action: Any):
        was_failed = self._ycb11_state_machine.failed
        observation, reward, terminated, truncated, info = super().step(action)
        if self._ycb11_state_machine.failed and not was_failed:
            print("[YCB11 terminal] " + json.dumps({
                "reason": self._ycb11_state_machine.failure_reason,
                "joint_limit_excess_rad": getattr(self, "_ycb11_joint_limit_excess_rad", 0.0),
                "state": self._ycb11_state_machine.snapshot(),
                "metrics": self._ycb11_last_metrics,
            }, default=str), flush=True)
        timed_out = bool(
            terminated
            and not bool(info.get("succeed", False))
            and not bool(info.get("failed", False))
            and self.env_step * self._control_dt >= self._time_limit
        )
        if timed_out:
            terminated = False
            truncated = True
            info["time_limit_reached"] = True
        return observation, reward, terminated, truncated, info


__all__ = ["HandsOnYCB11CompressSpongeIsaacLabEnv"]
