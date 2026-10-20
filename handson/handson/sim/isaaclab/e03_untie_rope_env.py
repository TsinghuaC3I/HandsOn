"""Native IsaacLab environment for E03 randomized rope-knot untying."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv, _randomized_physx_pose
from .e02_rope_knot_env import HandsOnE02RopeKnotIsaacLabEnv, _tensor_to_numpy
from .e03_untie_rope_scene import (
    E03_KNOT_NAMES,
    E03_ROPE_CENTERLINE_LENGTH,
    E03_ROPE_CONTACT_OFFSET,
    E03_ROPE_DIAMETER,
    E03_ROPE_NODE_COUNT,
    E03_ROPE_RADIAL_VERTEX_COUNT,
    E03_ROPE_RADIUS,
    E03_ROPE_REST_OFFSET,
    E03_ROPE_RING_COUNT,
    E03_ROPE_ROOT_PATH,
    E03_ROPE_SEGMENT_COUNT,
    E03_TABLE_TOP_Z,
    E03_TASK_NAME,
    e03_knot_centerline_centers,
    e03_knot_surface_mesh,
    spawn_e03_surface_deformable_rope,
)


_E03_KEYPOINT_COUNT = 16
_E03_MAX_REMAINING_CROSSINGS = 1
_E03_MAX_REMAINING_CONTACTS = 1
_E03_MIN_ENDPOINT_DISTANCE = 0.54
_E03_MIN_STRAIGHTNESS = 0.68
_E03_MIN_NONLOCAL_DISTANCE = 0.40 * E03_ROPE_DIAMETER
_E03_HAND_RELEASE_DISTANCE = 0.060
_E03_STABLE_RMS_SPEED = 0.045
_E03_STABLE_MAX_SPEED = 0.24
_E03_TABLE_X_BOUNDS = (-0.55, 0.25)
_E03_TABLE_Y_BOUNDS = (-0.80, 0.80)


class HandsOnE03UntieRopeIsaacLabEnv(HandsOnE02RopeKnotIsaacLabEnv):
    """Untie one of three seeded, self-colliding continuous-rope knots."""

    _TASK_NAME = E03_TASK_NAME
    _TASK_ID = "E03"
    _ROPE_LOGICAL_NAME = "e03_rope"
    _ROPE_MODEL_LABEL = "PhysX Surface Deformable randomized knotted rope"
    _ROPE_ROOT_PATH = E03_ROPE_ROOT_PATH
    _ROPE_NODE_COUNT = E03_ROPE_NODE_COUNT
    _ROPE_RING_COUNT = E03_ROPE_RING_COUNT
    _ROPE_RADIAL_VERTEX_COUNT = E03_ROPE_RADIAL_VERTEX_COUNT
    _ROPE_SEGMENT_COUNT = E03_ROPE_SEGMENT_COUNT
    _ROPE_RADIUS = E03_ROPE_RADIUS
    _ROPE_DIAMETER = E03_ROPE_DIAMETER
    _ROPE_CENTERLINE_LENGTH = E03_ROPE_CENTERLINE_LENGTH
    _ROPE_CONTACT_OFFSET = E03_ROPE_CONTACT_OFFSET
    _ROPE_REST_OFFSET = E03_ROPE_REST_OFFSET

    def __init__(self, task_name: str = E03_TASK_NAME, **kwargs: Any):
        self._e03_initial_state = np.zeros(8, dtype=np.float64)
        self._e03_initial_knot_verified = False
        self._e03_last_metrics: dict[str, Any] = {}
        super().__init__(task_name=task_name, **kwargs)
        self._e02_keypoint_indices = np.linspace(
            0,
            E03_ROPE_SEGMENT_COUNT - 1,
            _E03_KEYPOINT_COUNT,
        ).round().astype(np.int64)

    def _initial_rope_centers(self) -> np.ndarray:
        return e03_knot_centerline_centers(0)

    def _initial_surface_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return e03_knot_surface_mesh(0)

    def _spawn_surface_deformable_rope(self) -> None:
        if self._stage is None:
            raise RuntimeError("E03 stage is unavailable before rope spawn")
        spawn_e03_surface_deformable_rope(self._stage)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        HandsOnABCIsaacLabEnv._sample_reset_state(
            self,
            clear_manual_request=clear_manual_request,
        )
        if self.randomize:
            self.delta_h[:] = self._random.uniform(0.0, 0.025)
            self._e03_initial_state = np.asarray(
                [
                    self._random.uniform(-0.035, 0.035),
                    self._random.uniform(-0.070, 0.070),
                    math.radians(self._random.uniform(-20.0, 20.0)),
                    float(self._random.randint(0, len(E03_KNOT_NAMES))),
                    self._random.uniform(0.90, 1.10),
                    self._random.uniform(0.0, 0.0025),
                    float(self._random.randint(0, 2)),
                    float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
        else:
            self.delta_h[:] = 0.0
            self._e03_initial_state = np.asarray(
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
        self._e03_initial_knot_verified = False
        self._e03_last_metrics = {}
        self._e02_last_metrics = {}

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

    def _selected_template(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        knot_index = int(round(float(self._e03_initial_state[3]))) % len(
            E03_KNOT_NAMES
        )
        tightness = float(self._e03_initial_state[4])
        centers = e03_knot_centerline_centers(knot_index, tightness=tightness)
        nodes, _face_counts, _face_indices = e03_knot_surface_mesh(
            knot_index,
            tightness=tightness,
        )
        return centers, nodes

    def _transform_selected_points(self, points: np.ndarray) -> np.ndarray:
        transformed = np.asarray(points, dtype=np.float64).copy()
        dx, dy, yaw, _knot, _tightness, clearance, end_swap, delta_h = (
            self._e03_initial_state
        )
        total_yaw = float(yaw) + (
            math.pi if int(round(float(end_swap))) else 0.0
        )
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

    def _apply_task_pose_randomization(self) -> None:
        self._write_table_height()
        centers, nodes = self._selected_template()
        reset_centers = self._transform_selected_points(centers)
        reset_nodes = self._transform_selected_points(nodes)
        self._e02_reset_centers_np = reset_centers.copy()
        self._e02_centers_np = reset_centers.copy()
        self._e02_velocities_np = np.zeros_like(reset_centers)
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
                    "E03 tensor reset write did not reach PhysX: "
                    f"max_node_error={max_error:.6f}m"
                )

        geometry = self._rope_contact_geometry()
        self._e03_initial_knot_verified = bool(
            geometry["projected_crossing_count"] >= 2
        )
        self._e03_last_metrics = self._rope_metrics(update_history=False)
        knot_index = int(round(float(self._e03_initial_state[3]))) % len(
            E03_KNOT_NAMES
        )
        print(
            "[HandsOn/IsaacLab] Reset E03 randomized knot through the Surface "
            "Deformable tensor: "
            f"template={E03_KNOT_NAMES[knot_index]}, "
            f"verified={self._e03_initial_knot_verified}, "
            f"xy=({self._e03_initial_state[0]:+.4f},"
            f"{self._e03_initial_state[1]:+.4f})m, "
            f"yaw={math.degrees(self._e03_initial_state[2]):+.1f}deg, "
            f"tightness={self._e03_initial_state[4]:.3f}, "
            f"end_swap={bool(round(float(self._e03_initial_state[6])))}, "
            f"table_delta_h={self.delta_h[0]:.4f}m",
            flush=True,
        )

    def _minimum_hand_distance(self) -> float:
        points = self._e02_centers_np
        minimum = math.inf
        for side in ("right", "left"):
            for hand_point in self._a06_hand_points(side):
                minimum = min(
                    minimum,
                    float(np.min(np.linalg.norm(points - hand_point, axis=1))),
                )
        return minimum

    def _rope_metrics(self, *, update_history: bool) -> dict[str, Any]:
        del update_history
        points = self._e02_centers_np
        velocities = self._e02_velocities_np
        geometry = self._rope_contact_geometry()
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        contour_length = float(np.sum(segment_lengths))
        endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
        straightness = endpoint_distance / max(contour_length, 1e-9)
        bounds_min = np.min(points, axis=0) - E03_ROPE_RADIUS
        bounds_max = np.max(points, axis=0) + E03_ROPE_RADIUS
        table_top = E03_TABLE_TOP_Z + float(self.delta_h[0])
        speed = np.linalg.norm(velocities, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(speed))))
        max_speed = float(np.max(speed))
        hand_distance = self._minimum_hand_distance()

        knot_open = bool(
            geometry["projected_crossing_count"] <= _E03_MAX_REMAINING_CROSSINGS
            and geometry["nonlocal_contact_count"] <= _E03_MAX_REMAINING_CONTACTS
        )
        ends_separated = bool(endpoint_distance >= _E03_MIN_ENDPOINT_DISTANCE)
        extended = bool(ends_separated and straightness >= _E03_MIN_STRAIGHTNESS)
        nonpenetrating = bool(
            geometry["min_nonlocal_distance"] >= _E03_MIN_NONLOCAL_DISTANCE
        )
        on_table = bool(
            bounds_min[2] >= table_top - 0.025
            and bounds_max[2] <= table_top + 0.32
        )
        in_workspace = bool(
            bounds_min[0] >= _E03_TABLE_X_BOUNDS[0] - 0.03
            and bounds_max[0] <= _E03_TABLE_X_BOUNDS[1] + 0.03
            and bounds_min[1] >= _E03_TABLE_Y_BOUNDS[0] - 0.03
            and bounds_max[1] <= _E03_TABLE_Y_BOUNDS[1] + 0.03
        )
        hands_released = bool(hand_distance >= _E03_HAND_RELEASE_DISTANCE)
        stable = bool(
            rms_speed <= _E03_STABLE_RMS_SPEED
            and max_speed <= _E03_STABLE_MAX_SPEED
        )
        success_now = bool(
            self._e03_initial_knot_verified
            and knot_open
            and extended
            and nonpenetrating
            and on_table
            and in_workspace
            and hands_released
            and stable
        )
        return {
            **geometry,
            "contour_length": contour_length,
            "endpoint_distance": endpoint_distance,
            "straightness": float(straightness),
            "rms_link_speed": rms_speed,
            "max_link_speed": max_speed,
            "hand_distance": float(hand_distance),
            "initial_knot_verified": bool(self._e03_initial_knot_verified),
            "knot_open": knot_open,
            "ends_separated": ends_separated,
            "extended": extended,
            "nonpenetrating": nonpenetrating,
            "on_table": on_table,
            "in_workspace": in_workspace,
            "hands_released": hands_released,
            "stable": stable,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "success_now": success_now,
        }

    def _refresh_rope_state(self) -> None:
        super()._refresh_rope_state()
        self._e03_last_metrics = dict(self._e02_last_metrics)

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
                shape=(_E03_KEYPOINT_COUNT * 3,),
                dtype=np.float64,
            ),
            "rope_endpoints": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "untie_metrics": spaces.Box(
                -np.inf, np.inf, shape=(18,), dtype=np.float64
            ),
            "knot_template": spaces.Box(
                0.0, 1.0, shape=(len(E03_KNOT_NAMES),), dtype=np.float64
            ),
            "rope_initial_state": spaces.Box(
                -np.inf, np.inf, shape=(8,), dtype=np.float64
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

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._e03_last_metrics or self._rope_metrics(update_history=False)
        tcp, gripper = self._robot_state()
        knot_index = int(round(float(self._e03_initial_state[3]))) % len(
            E03_KNOT_NAMES
        )
        knot_selector = np.zeros(len(E03_KNOT_NAMES), dtype=np.float64)
        knot_selector[knot_index] = 1.0
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
            "untie_metrics": np.asarray(
                [
                    metrics["endpoint_distance"],
                    metrics["straightness"],
                    metrics["projected_crossing_count"],
                    metrics["nonlocal_contact_count"],
                    metrics["min_nonlocal_distance"],
                    metrics["core_radius"],
                    metrics["rms_link_speed"],
                    metrics["max_link_speed"],
                    metrics["hand_distance"],
                    metrics["initial_knot_verified"],
                    metrics["knot_open"],
                    metrics["ends_separated"],
                    metrics["extended"],
                    metrics["nonpenetrating"],
                    metrics["on_table"],
                    metrics["in_workspace"],
                    metrics["hands_released"],
                    self._success_counter / max(1, self._success_steps_required()),
                ],
                dtype=np.float64,
            ),
            "knot_template": knot_selector,
            "rope_initial_state": self._e03_initial_state.astype(
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
        self._e03_last_metrics = self._rope_metrics(update_history=False)
        self._e02_last_metrics = dict(self._e03_last_metrics)
        return bool(self._e03_last_metrics["success_now"])

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._e03_last_metrics or self._rope_metrics(update_history=False)
        return [
            self._condition_row(
                "e03_knot_open",
                "The sampled knot core has been completely opened",
                metrics["initial_knot_verified"] and metrics["knot_open"],
            ),
            self._condition_row(
                "e03_rope_extended",
                "Both free ends are separated and the rope is extended",
                metrics["extended"],
            ),
            self._condition_row(
                "e03_collision_safe",
                "The rope is collision-safe and inside the reachable workspace",
                metrics["nonpenetrating"]
                and metrics["on_table"]
                and metrics["in_workspace"],
            ),
            self._condition_row(
                "e03_hands_released",
                "Both hands have released the untied rope",
                metrics["hands_released"],
            ),
            self._condition_row(
                "success_hold",
                f"The untied rope remains stable for {self._success_steps_required()} frames",
                self._success_counter >= self._success_steps_required(),
            ),
        ]

    def _compute_failure(self) -> bool:
        metrics = self._e03_last_metrics or self._rope_metrics(update_history=False)
        center = np.mean(self._e02_centers_np, axis=0)
        return bool(
            metrics["bounds_min"][2] < 0.55
            or center[0] < -0.90
            or center[0] > 0.60
            or abs(float(center[1])) > 1.10
        )

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        knot_index = int(round(float(self._e03_initial_state[3]))) % len(
            E03_KNOT_NAMES
        )
        info.update(
            {
                "task_id": "E03",
                "knot_template": E03_KNOT_NAMES[knot_index],
                "knot_template_index": knot_index,
                "knot_template_count": len(E03_KNOT_NAMES),
                "initial_knot_verified": bool(self._e03_initial_knot_verified),
            }
        )
        return info

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
            state_parts.get("rope_initial_state", self._e03_initial_state),
            dtype=np.float64,
        ).ravel()
        if initial.size >= 7:
            self._e03_initial_state[:7] = initial[:7]
        self._e03_initial_state[7] = float(self.delta_h[0])
        if self._robots:
            self._restore_task_scene_state()
        self._apply_task_pose_randomization()
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()


__all__ = ["HandsOnE03UntieRopeIsaacLabEnv"]
