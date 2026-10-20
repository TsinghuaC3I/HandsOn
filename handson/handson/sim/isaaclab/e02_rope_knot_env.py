"""Native IsaacLab collection environment for E02 rope knot tying."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv, _randomized_physx_pose
from .e02_rope_knot_scene import (
    E02_ROPE_CENTERLINE_LENGTH,
    E02_ROPE_CONTACT_OFFSET,
    E02_ROPE_DIAMETER,
    E02_ROPE_NODE_COUNT,
    E02_ROPE_RADIAL_VERTEX_COUNT,
    E02_ROPE_RADIUS,
    E02_ROPE_REST_OFFSET,
    E02_ROPE_RING_COUNT,
    E02_ROPE_ROOT_PATH,
    E02_ROPE_SEGMENT_COUNT,
    E02_TABLE_TOP_Z,
    E02_TASK_NAME,
    e02_initial_rope_centers,
    e02_initial_rope_surface_mesh,
    spawn_e02_surface_deformable_rope,
)


_E02_KEYPOINT_COUNT = 16
_E02_MIN_ARC_GAP = 6
_E02_CONTACT_DISTANCE = 1.35 * E02_ROPE_DIAMETER
_E02_MAX_CROSSING_Z_GAP = 2.4 * E02_ROPE_DIAMETER
_E02_MAX_CORE_RADIUS = 0.105
_E02_TIGHT_CORE_RADIUS = 0.068
_E02_MIN_TAIL_DISTANCE = 0.105
_E02_MIN_END_SEPARATION = 0.30
_E02_MIN_PULL_PROGRESS = 0.035
_E02_MIN_NONLOCAL_DISTANCE = 0.40 * E02_ROPE_DIAMETER
_E02_STABLE_RMS_SPEED = 0.045
_E02_STABLE_MAX_SPEED = 0.24
_E02_TABLE_X_BOUNDS = (-0.55, 0.25)
_E02_TABLE_Y_BOUNDS = (-0.80, 0.80)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64).copy()


def _segment_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> float:
    """Shortest 3-D distance between two finite line segments."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    small = 1e-12

    s_num = 0.0
    s_den = denominator
    t_num = 0.0
    t_den = denominator
    if denominator < small:
        s_num = 0.0
        s_den = 1.0
        t_num = e
        t_den = c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        if s_num < 0.0:
            s_num = 0.0
            t_num = e
            t_den = c
        elif s_num > s_den:
            s_num = s_den
            t_num = e + b
            t_den = c

    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num = -d
            s_den = a
    elif t_num > t_den:
        t_num = t_den
        if (-d + b) < 0.0:
            s_num = 0.0
        elif (-d + b) > a:
            s_num = s_den
        else:
            s_num = -d + b
            s_den = a

    s = 0.0 if abs(s_num) < small else s_num / max(s_den, small)
    t = 0.0 if abs(t_num) < small else t_num / max(t_den, small)
    return float(np.linalg.norm(w + s * u - t * v))


def _projected_crossing(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> tuple[bool, float]:
    """Return whether two links cross in XY and their interpolated Z gap."""
    r = p1[:2] - p0[:2]
    s = q1[:2] - q0[:2]
    denominator = float(r[0] * s[1] - r[1] * s[0])
    if abs(denominator) < 1e-10:
        return False, math.inf
    delta = q0[:2] - p0[:2]
    t = float((delta[0] * s[1] - delta[1] * s[0]) / denominator)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / denominator)
    if not (0.02 <= t <= 0.98 and 0.02 <= u <= 0.98):
        return False, math.inf
    z_p = float(p0[2] + t * (p1[2] - p0[2]))
    z_q = float(q0[2] + u * (q1[2] - q0[2]))
    return True, abs(z_p - z_q)


class HandsOnE02RopeKnotIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual E02 task backed by a self-colliding Surface Deformable rope."""

    _TASK_NAME = E02_TASK_NAME
    _TASK_ID = "E02"
    _ROPE_LOGICAL_NAME = "e02_rope"
    _ROPE_MODEL_LABEL = "PhysX Surface Deformable braided rope"
    _ROPE_ROOT_PATH = E02_ROPE_ROOT_PATH
    _ROPE_NODE_COUNT = E02_ROPE_NODE_COUNT
    _ROPE_RING_COUNT = E02_ROPE_RING_COUNT
    _ROPE_RADIAL_VERTEX_COUNT = E02_ROPE_RADIAL_VERTEX_COUNT
    _ROPE_SEGMENT_COUNT = E02_ROPE_SEGMENT_COUNT
    _ROPE_RADIUS = E02_ROPE_RADIUS
    _ROPE_DIAMETER = E02_ROPE_DIAMETER
    _ROPE_CENTERLINE_LENGTH = E02_ROPE_CENTERLINE_LENGTH
    _ROPE_CONTACT_OFFSET = E02_ROPE_CONTACT_OFFSET
    _ROPE_REST_OFFSET = E02_ROPE_REST_OFFSET

    def __init__(self, task_name: str = E02_TASK_NAME, **kwargs: Any):
        if task_name != self._TASK_NAME:
            raise ValueError(
                f"Unsupported rope task {task_name!r}; expected {self._TASK_NAME!r}"
            )

        fallback = self._initial_rope_centers()
        fallback_nodes, _face_counts, _face_indices = self._initial_surface_mesh()
        self._e02_rope_sim_view: Any | None = None
        self._e02_rope_view: Any | None = None
        self._e02_rope_indices: Any | None = None
        self._e02_base_nodes_np = fallback_nodes.copy()
        self._e02_nodes_np = fallback_nodes.copy()
        self._e02_nodal_velocities_np = np.zeros_like(fallback_nodes)
        self._e02_reset_nodes_np = fallback_nodes.copy()
        self._e02_centers_np = fallback.copy()
        self._e02_velocities_np = np.zeros_like(fallback)
        self._e02_reset_centers_np = fallback.copy()
        self._e02_initial_state = np.zeros(6, dtype=np.float64)
        self._e02_keypoint_indices = np.linspace(
            0,
            self._ROPE_SEGMENT_COUNT - 1,
            _E02_KEYPOINT_COUNT,
        ).round().astype(np.int64)
        self._e02_last_metrics: dict[str, Any] = {}
        self._e02_knot_seen = False
        self._e02_knot_first_end_distance = 0.0
        self._e02_peak_end_distance = 0.0
        super().__init__(task_name=task_name, **kwargs)

    def _initial_rope_centers(self) -> np.ndarray:
        return e02_initial_rope_centers()

    def _initial_surface_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return e02_initial_rope_surface_mesh()

    def _spawn_surface_deformable_rope(self) -> None:
        if self._stage is None:
            raise RuntimeError(f"{self._TASK_ID} stage is unavailable before rope spawn")
        spawn_e02_surface_deformable_rope(self._stage)

    # ------------------------------------------------------------------
    # Native rope setup, collision safety, and tensor reset
    # ------------------------------------------------------------------

    def _task_specific_isaac_kit_args(self) -> str | None:
        return "--/persistent/physics/enableDeformableBeta=true"

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        del sim_utils
        self._spawn_surface_deformable_rope()
        print(
            f"[HandsOn/IsaacLab] {self._TASK_ID} authored as a Surface "
            "Deformable rope: "
            f"segments={self._ROPE_SEGMENT_COUNT}, "
            f"nodes={self._ROPE_NODE_COUNT}, "
            f"diameter={self._ROPE_DIAMETER * 1000.0:.1f}mm, "
            f"length={self._ROPE_CENTERLINE_LENGTH:.3f}m",
            flush=True,
        )

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        del sim_utils
        if self._stage is None:
            return 0, 0

        from pxr import PhysxSchema, Usd, UsdPhysics

        tuned_bodies = 0
        tuned_colliders = 0
        roots = (
            self._stage.GetPrimAtPath(self._ROPE_ROOT_PATH),
            self._stage.GetPrimAtPath("/World/HandsOnRobots"),
        )
        for root in roots:
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                path_lower = str(prim.GetPath()).lower()
                is_rope = path_lower.startswith(self._ROPE_ROOT_PATH.lower())
                is_hand = "palm" in path_lower or "finger" in path_lower
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (is_rope or is_hand):
                    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
                    rigid_api.CreateSolverPositionIterationCountAttr().Set(64)
                    rigid_api.CreateSolverVelocityIterationCountAttr().Set(8)
                    rigid_api.CreateMaxDepenetrationVelocityAttr().Set(
                        0.35 if is_rope else 0.75
                    )
                    tuned_bodies += 1
                if (
                    is_rope
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ):
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision_api.CreateContactOffsetAttr().Set(
                        self._ROPE_CONTACT_OFFSET
                    )
                    collision_api.CreateRestOffsetAttr().Set(self._ROPE_REST_OFFSET)
                    tuned_colliders += 1
        return tuned_bodies, tuned_colliders

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        import omni.physics.tensors as physics_tensors

        physics_sim_view = physics_tensors.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        view = physics_sim_view.create_surface_deformable_body_view(
            self._ROPE_ROOT_PATH
        )
        view_count = int(view.count)
        view_valid = bool(view.check()) if view_count else False
        if view_count != 1:
            raise RuntimeError(
                f"{self._TASK_ID} Surface Deformable tensor view is invalid: "
                f"count={view_count}, check={view_valid}, "
                f"root={self._ROPE_ROOT_PATH}"
            )
        try:
            positions = view.get_simulation_nodal_positions().clone()
            velocities = view.get_simulation_nodal_velocities().clone()
        except Exception as exc:
            raise RuntimeError(
                f"{self._TASK_ID} surface tensor exists but PhysX did not initialize its "
                f"nodal state (check={view_valid})"
            ) from exc
        if int(positions.shape[1]) != self._ROPE_NODE_COUNT:
            raise RuntimeError(
                f"{self._TASK_ID} cooked rope topology changed unexpectedly: "
                f"nodes={int(positions.shape[1])}, expected={self._ROPE_NODE_COUNT}"
            )
        self._e02_rope_sim_view = physics_sim_view
        self._e02_rope_view = view
        self._e02_rope_indices = self._torch.arange(
            int(view.count),
            dtype=self._torch.long,
            device=self._isaac_device,
        )
        self._e02_nodes_np = _tensor_to_numpy(positions)[0]
        self._e02_nodal_velocities_np = _tensor_to_numpy(velocities)[0]
        self._e02_base_nodes_np = self._e02_nodes_np.copy()
        self._e02_reset_nodes_np = self._e02_nodes_np.copy()
        self._refresh_rope_state()
        self._e02_reset_centers_np = self._e02_centers_np.copy()
        print(
            f"[HandsOn/IsaacLab] {self._TASK_ID} surface tensor ready: "
            f"nodes={self._e02_nodes_np.shape[0]}, "
            f"triangles={int(view.max_simulation_elements_per_body)}",
            flush=True,
        )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        if self.randomize:
            self.delta_h[:] = self._random.uniform(0.0, 0.025)
            self._e02_initial_state = np.asarray(
                [
                    self._random.uniform(-0.035, 0.035),
                    self._random.uniform(-0.055, 0.055),
                    math.radians(self._random.uniform(-15.0, 15.0)),
                    float(self._random.randint(0, 2)),
                    self._random.uniform(0.0, 0.003),
                    float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
        else:
            self.delta_h[:] = 0.0
            self._e02_initial_state = np.zeros(6, dtype=np.float64)
        self._e02_last_metrics = {}
        self._e02_knot_seen = False
        self._e02_knot_first_end_distance = 0.0
        self._e02_peak_end_distance = 0.0

    def _write_table_height(self) -> None:
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            return
        table_path = str(table_prim.GetPath())
        for path, view, indices, root_pose, root_velocity in self._task_reset_body_states:
            if path != table_path:
                continue
            randomized = _randomized_physx_pose(
                root_pose.detach().cpu().numpy(),
                xy_offset=np.zeros(2, dtype=np.float64),
                z_offset=float(self.delta_h[0]),
                yaw_degrees=0.0,
            )
            view.set_transforms(
                self._torch.as_tensor(
                    randomized,
                    dtype=root_pose.dtype,
                    device=root_pose.device,
                ),
                indices,
            )
            view.set_velocities(self._torch.zeros_like(root_velocity), indices)
            return

    def _transformed_interface_centers(self) -> np.ndarray:
        centers = e02_initial_rope_centers()
        return self._transform_rope_points(centers)

    def _transform_rope_points(self, points: np.ndarray) -> np.ndarray:
        transformed = np.asarray(points, dtype=np.float64).copy()
        dx, dy, yaw, flip, clearance, delta_h = self._e02_initial_state
        total_yaw = float(yaw) + (math.pi if int(round(float(flip))) else 0.0)
        cosine, sine = math.cos(total_yaw), math.sin(total_yaw)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        anchor = np.mean(transformed, axis=0)
        transformed[:, :2] = (
            (transformed[:, :2] - anchor[:2]) @ rotation.T + anchor[:2]
        )
        transformed[:, 0] += float(dx)
        transformed[:, 1] += float(dy)
        transformed[:, 2] += float(clearance + delta_h)
        return transformed

    def _centerline_state_from_nodes(
        self,
        nodes: np.ndarray,
        nodal_velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(nodes, dtype=np.float64).reshape(
            self._ROPE_RING_COUNT,
            self._ROPE_RADIAL_VERTEX_COUNT,
            3,
        )
        nodal_velocities = np.asarray(nodal_velocities, dtype=np.float64).reshape(
            self._ROPE_RING_COUNT,
            self._ROPE_RADIAL_VERTEX_COUNT,
            3,
        )
        ring_centers = np.mean(nodes, axis=1)
        ring_velocities = np.mean(nodal_velocities, axis=1)
        return (
            0.5 * (ring_centers[:-1] + ring_centers[1:]),
            0.5 * (ring_velocities[:-1] + ring_velocities[1:]),
        )

    def _reset_scene_state(self) -> None:
        interface_only = not self._robots
        super()._reset_scene_state()
        if interface_only:
            self._apply_task_pose_randomization()

    def _apply_task_pose_randomization(self) -> None:
        self._write_table_height()
        interface_centers = self._transformed_interface_centers()
        self._e02_reset_centers_np = interface_centers.copy()
        self._e02_centers_np = interface_centers.copy()
        self._e02_velocities_np = np.zeros_like(interface_centers)

        reset_nodes = self._transform_rope_points(self._e02_base_nodes_np)
        self._e02_reset_nodes_np = reset_nodes.copy()
        self._e02_nodes_np = reset_nodes.copy()
        self._e02_nodal_velocities_np = np.zeros_like(reset_nodes)
        if self._e02_rope_view is not None and self._e02_rope_indices is not None:
            positions = self._e02_rope_view.get_simulation_nodal_positions()
            position_tensor = self._torch.as_tensor(
                reset_nodes.reshape(positions.shape),
                dtype=positions.dtype,
                device=positions.device,
            )
            self._e02_rope_view.set_simulation_nodal_positions(
                position_tensor,
                self._e02_rope_indices,
            )
            self._e02_rope_view.set_simulation_nodal_velocities(
                self._torch.zeros_like(position_tensor),
                self._e02_rope_indices,
            )
            readback = _tensor_to_numpy(
                self._e02_rope_view.get_simulation_nodal_positions()
            )[0]
            max_error = float(
                np.max(np.linalg.norm(readback - reset_nodes, axis=1))
            )
            if not np.all(np.isfinite(readback)) or max_error > 1e-4:
                raise RuntimeError(
                    "E02 tensor reset write did not reach PhysX: "
                    f"max_node_error={max_error:.6f}m"
                )
        print(
            "[HandsOn/IsaacLab] Reset E02 rope through its Surface Deformable tensor: "
            f"table_delta_h={self.delta_h[0]:.4f}m, "
            f"xy=({self._e02_initial_state[0]:+.4f},"
            f"{self._e02_initial_state[1]:+.4f})m, "
            f"yaw={math.degrees(self._e02_initial_state[2]):+.1f}deg, "
            f"end_swap={bool(round(float(self._e02_initial_state[3])))}, "
            f"clearance={self._e02_initial_state[4]:.4f}m",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Live rope tensors, topology-aware success, and observations
    # ------------------------------------------------------------------

    def _refresh_rope_state(self) -> None:
        if self._e02_rope_view is not None:
            self._e02_nodes_np = _tensor_to_numpy(
                self._e02_rope_view.get_simulation_nodal_positions()
            )[0]
            self._e02_nodal_velocities_np = _tensor_to_numpy(
                self._e02_rope_view.get_simulation_nodal_velocities()
            )[0]
            self._e02_centers_np, self._e02_velocities_np = (
                self._centerline_state_from_nodes(
                    self._e02_nodes_np,
                    self._e02_nodal_velocities_np,
                )
            )
        self._e02_last_metrics = self._rope_metrics(update_history=True)

    def _refresh_object_poses(self) -> None:
        super()._refresh_object_poses()
        self._refresh_rope_state()
        self._poses[self._ROPE_LOGICAL_NAME] = self._rope_pose()

    def _rope_pose(self) -> np.ndarray:
        center = np.mean(self._e02_centers_np, axis=0)
        direction = self._e02_centers_np[-1, :2] - self._e02_centers_np[0, :2]
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        return np.asarray(
            [
                *center,
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            ],
            dtype=np.float64,
        )

    def _rope_contact_geometry(self) -> dict[str, Any]:
        points = self._e02_centers_np
        contacts: list[tuple[int, int, float]] = []
        crossings: list[tuple[int, int, float]] = []
        min_distance = math.inf
        for first in range(2, points.shape[0] - 3):
            for second in range(
                first + _E02_MIN_ARC_GAP,
                points.shape[0] - 3,
            ):
                distance = _segment_distance(
                    points[first],
                    points[first + 1],
                    points[second],
                    points[second + 1],
                )
                min_distance = min(min_distance, distance)
                if distance <= _E02_CONTACT_DISTANCE:
                    contacts.append((first, second, distance))
                intersects, z_gap = _projected_crossing(
                    points[first],
                    points[first + 1],
                    points[second],
                    points[second + 1],
                )
                if intersects and z_gap <= _E02_MAX_CROSSING_Z_GAP:
                    crossings.append((first, second, z_gap))

        involved: set[int] = set()
        source_pairs = crossings if crossings else contacts
        for first, second, _value in source_pairs:
            involved.update((first, first + 1, second, second + 1))
        expanded: set[int] = set()
        for index in involved:
            expanded.update(
                range(max(0, index - 1), min(points.shape[0], index + 2))
            )
        core_indices = np.asarray(sorted(expanded), dtype=np.int64)
        if core_indices.size:
            core_points = points[core_indices]
            core_center = np.mean(core_points, axis=0)
            core_radius = float(
                np.max(np.linalg.norm(core_points[:, :2] - core_center[:2], axis=1))
            )
        else:
            core_center = np.mean(points, axis=0)
            core_radius = math.inf
        return {
            "nonlocal_contact_count": len(contacts),
            "projected_crossing_count": len(crossings),
            "min_nonlocal_distance": float(min_distance),
            "core_indices": core_indices,
            "core_center": core_center,
            "core_radius": core_radius,
        }

    def _rope_metrics(self, *, update_history: bool) -> dict[str, Any]:
        points = self._e02_centers_np
        velocities = self._e02_velocities_np
        geometry = self._rope_contact_geometry()
        core_center = geometry["core_center"]
        end_distance = float(np.linalg.norm(points[-1] - points[0]))
        first_tail_distance = float(np.linalg.norm(points[0] - core_center))
        second_tail_distance = float(np.linalg.norm(points[-1] - core_center))
        bounds_min = np.min(points, axis=0) - E02_ROPE_RADIUS
        bounds_max = np.max(points, axis=0) + E02_ROPE_RADIUS
        table_top = E02_TABLE_TOP_Z + float(self.delta_h[0])
        speed = np.linalg.norm(velocities, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(speed))))
        max_speed = float(np.max(speed))

        localized_core = bool(geometry["core_radius"] <= _E02_MAX_CORE_RADIUS)
        nonpenetrating = bool(
            geometry["min_nonlocal_distance"] >= _E02_MIN_NONLOCAL_DISTANCE
        )
        knot_formed = bool(
            geometry["projected_crossing_count"] >= 2
            and geometry["nonlocal_contact_count"] >= 2
            and localized_core
            and nonpenetrating
        )
        if update_history and knot_formed:
            if not self._e02_knot_seen:
                self._e02_knot_seen = True
                self._e02_knot_first_end_distance = end_distance
                self._e02_peak_end_distance = end_distance
            self._e02_peak_end_distance = max(
                self._e02_peak_end_distance,
                end_distance,
            )
        pull_reference = (
            self._e02_knot_first_end_distance
            if self._e02_knot_seen
            else end_distance
        )
        pull_progress = max(0.0, self._e02_peak_end_distance - pull_reference)
        tails_extended = bool(
            first_tail_distance >= _E02_MIN_TAIL_DISTANCE
            and second_tail_distance >= _E02_MIN_TAIL_DISTANCE
        )
        ends_pulled = bool(
            self._e02_knot_seen
            and self._e02_peak_end_distance >= _E02_MIN_END_SEPARATION
            and pull_progress >= _E02_MIN_PULL_PROGRESS
        )
        knot_tightened = bool(
            knot_formed
            and geometry["core_radius"] <= _E02_TIGHT_CORE_RADIUS
            and ends_pulled
        )
        on_table = bool(
            bounds_min[2] >= table_top - 0.025
            and bounds_max[2] <= table_top + 0.30
        )
        in_workspace = bool(
            bounds_min[0] >= _E02_TABLE_X_BOUNDS[0] - 0.03
            and bounds_max[0] <= _E02_TABLE_X_BOUNDS[1] + 0.03
            and bounds_min[1] >= _E02_TABLE_Y_BOUNDS[0] - 0.03
            and bounds_max[1] <= _E02_TABLE_Y_BOUNDS[1] + 0.03
        )
        stable = bool(
            rms_speed <= _E02_STABLE_RMS_SPEED
            and max_speed <= _E02_STABLE_MAX_SPEED
        )
        return {
            **geometry,
            "endpoint_distance": end_distance,
            "first_tail_distance": first_tail_distance,
            "second_tail_distance": second_tail_distance,
            "pull_progress": float(pull_progress),
            "rms_link_speed": rms_speed,
            "max_link_speed": max_speed,
            "knot_formed": knot_formed,
            "tails_extended": tails_extended,
            "ends_pulled": ends_pulled,
            "knot_tightened": knot_tightened,
            "nonpenetrating": nonpenetrating,
            "on_table": on_table,
            "in_workspace": in_workspace,
            "stable": stable,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "success_now": bool(
                knot_tightened
                and tails_extended
                and on_table
                and in_workspace
                and stable
            ),
        }

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "rope_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "rope_keypoints": spaces.Box(
                -np.inf,
                np.inf,
                shape=(_E02_KEYPOINT_COUNT * 3,),
                dtype=np.float64,
            ),
            "rope_endpoints": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "knot_metrics": spaces.Box(
                -np.inf, np.inf, shape=(16,), dtype=np.float64
            ),
            "rope_initial_state": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "table_delta_height": spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float64
            ),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces)
        }
        if self.image_obs:
            image_space = lambda shape=self._image_shape: spaces.Box(
                0, 255, shape=shape, dtype=np.uint8
            )
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": image_space(),
                    "wrist_left": image_space((*reversed(self._wrist_recording_camera_output_size), 3)),
                    "wrist_right": image_space((*reversed(self._wrist_recording_camera_output_size), 3)),
                }
            )
        return spaces.Dict(observation_spaces)

    def _robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        tcp = np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(
            np.float64
        )
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[:16]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[:16]
        return tcp, gripper

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._e02_last_metrics or self._rope_metrics(update_history=False)
        tcp, gripper = self._robot_state()
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "rope_pose": self._rope_pose(),
            "rope_keypoints": self._e02_centers_np[
                self._e02_keypoint_indices
            ].reshape(-1).astype(np.float64, copy=True),
            "rope_endpoints": np.concatenate(
                [self._e02_centers_np[0], self._e02_centers_np[-1]]
            ).astype(np.float64),
            "knot_metrics": np.asarray(
                [
                    metrics["endpoint_distance"],
                    metrics["first_tail_distance"],
                    metrics["second_tail_distance"],
                    metrics["pull_progress"],
                    metrics["projected_crossing_count"],
                    metrics["nonlocal_contact_count"],
                    metrics["core_radius"],
                    metrics["min_nonlocal_distance"],
                    metrics["rms_link_speed"],
                    metrics["max_link_speed"],
                    metrics["knot_formed"],
                    metrics["tails_extended"],
                    metrics["ends_pulled"],
                    metrics["knot_tightened"],
                    metrics["nonpenetrating"],
                    self._success_counter / max(1, self._success_steps_required()),
                ],
                dtype=np.float64,
            ),
            "rope_initial_state": self._e02_initial_state.astype(
                np.float64, copy=True
            ),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        observation: dict[str, Any] = {"state": state}
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        self._e02_last_metrics = self._rope_metrics(update_history=True)
        return bool(self._e02_last_metrics["success_now"])

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        return str(row.get("label") or row.get("key") or "Condition")

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._e02_last_metrics or self._rope_metrics(update_history=False)
        return [
            self._condition_row(
                "e02_knot_formed",
                "A stable overhand knot is formed",
                metrics["knot_formed"],
            ),
            self._condition_row(
                "e02_tails_extended",
                "Both free ends extend from the knot core",
                metrics["tails_extended"],
            ),
            self._condition_row(
                "e02_knot_tightened",
                "The knot is tightened by pulling both ends",
                metrics["knot_tightened"],
            ),
            self._condition_row(
                "e02_nonpenetrating",
                "Rope contacts remain collision-safe",
                metrics["nonpenetrating"] and metrics["on_table"] and metrics["in_workspace"],
            ),
            self._condition_row(
                "success_hold",
                f"The tightened knot remains stable for {self._success_steps_required()} frames",
                self._success_counter >= self._success_steps_required(),
            ),
        ]

    def _compute_failure(self) -> bool:
        metrics = self._e02_last_metrics or self._rope_metrics(update_history=False)
        center = np.mean(self._e02_centers_np, axis=0)
        return bool(
            metrics["bounds_min"][2] < 0.55
            or center[0] < -0.90
            or center[0] > 0.60
            or abs(float(center[1])) > 1.10
        )

    def _task_info(self) -> dict[str, Any]:
        metrics = self._e02_last_metrics or self._rope_metrics(update_history=False)
        return {
            "task_id": self._TASK_ID,
            "task_name": self.task_name,
            "rope_model": self._ROPE_MODEL_LABEL,
            "rope_segment_count": self._ROPE_SEGMENT_COUNT,
            "rope_diameter_m": self._ROPE_DIAMETER,
            "rope_length_m": self._ROPE_CENTERLINE_LENGTH,
            "success_counter": int(self._success_counter),
            "success_steps_required": int(self._success_steps_required()),
            **{
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "core_indices",
                    "core_center",
                    "bounds_min",
                    "bounds_max",
                }
            },
        }

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        delta = np.asarray(
            state_parts.get("table_delta_height", self.delta_h),
            dtype=np.float64,
        ).ravel()
        if delta.size:
            self.delta_h[:] = float(delta[0])
        initial = np.asarray(
            state_parts.get("rope_initial_state", self._e02_initial_state),
            dtype=np.float64,
        ).ravel()
        if initial.size >= 5:
            self._e02_initial_state[:5] = initial[:5]
        self._e02_initial_state[5] = float(self.delta_h[0])
        if self._robots:
            self._restore_task_scene_state()
        self._apply_task_pose_randomization()
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()

    def close(self) -> None:
        self._e02_rope_view = None
        self._e02_rope_indices = None
        self._e02_rope_sim_view = None
        super().close()


__all__ = ["HandsOnE02RopeKnotIsaacLabEnv"]
