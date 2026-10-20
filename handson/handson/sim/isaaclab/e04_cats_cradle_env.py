"""Native IsaacLab environment for randomized Cat's Cradle transitions."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .e02_rope_knot_env import (
    HandsOnE02RopeKnotIsaacLabEnv,
    _projected_crossing,
    _segment_distance,
    _tensor_to_numpy,
)
from .e04_cats_cradle_scene import (
    E04_CYCLE_TRANSITIONS,
    E04_FIGURE_INDEX,
    E04_FIGURE_KEYS,
    E04_FIGURE_LABELS,
    E04_RECORDED_FIGURE_STEPS,
    E04_STRING_CENTERLINE_LENGTH,
    E04_STRING_CONTACT_OFFSET,
    E04_STRING_DIAMETER,
    E04_STRING_NODE_COUNT,
    E04_STRING_RADIAL_VERTEX_COUNT,
    E04_STRING_RADIUS,
    E04_STRING_REST_OFFSET,
    E04_STRING_RING_COUNT,
    E04_STRING_ROOT_PATH,
    E04_STRING_SEGMENT_COUNT,
    E04_TABLE_TOP_Z,
    E04_TASK_NAME,
    e04_figure_centerline,
    e04_string_surface_mesh,
    spawn_e04_surface_deformable_string,
)


_E04_KEYPOINT_COUNT = 24
_E04_MIN_ARC_GAP = 6
_E04_CONTACT_DISTANCE = 1.5 * E04_STRING_DIAMETER
_E04_MAX_CROSSING_Z_GAP = 8.0 * E04_STRING_DIAMETER
_E04_MIN_NONLOCAL_DISTANCE = 0.75 * E04_STRING_DIAMETER
_E04_TARGET_RMS_TOLERANCE = 0.022
_E04_SOURCE_DEPARTURE_RMS = 0.032
_E04_MAX_CENTROID_DRIFT = 0.080
_E04_HAND_SUPPORT_DISTANCE = 0.065
_E04_STABLE_RMS_SPEED = 0.060
_E04_STABLE_MAX_SPEED = 0.30
_E04_TABLE_X_BOUNDS = (-0.55, 0.25)
_E04_TABLE_Y_BOUNDS = (-0.80, 0.80)


def _cyclic_segment_pairs(count: int, minimum_gap: int) -> tuple[np.ndarray, np.ndarray]:
    first: list[int] = []
    second: list[int] = []
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            gap = min(
                second_index - first_index,
                count - (second_index - first_index),
            )
            if gap >= minimum_gap:
                first.append(first_index)
                second.append(second_index)
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


_E04_PAIR_FIRST, _E04_PAIR_SECOND = _cyclic_segment_pairs(
    E04_STRING_SEGMENT_COUNT,
    _E04_MIN_ARC_GAP,
)
_E04_SEGMENT_SAMPLES = np.linspace(0.0, 1.0, 5, dtype=np.float64)


class HandsOnE04CatsCradleIsaacLabEnv(HandsOnE02RopeKnotIsaacLabEnv):
    """One random legal transition in the recorded Cat's Cradle cycle."""

    _TASK_NAME = E04_TASK_NAME
    _TASK_ID = "E04"
    _ROPE_LOGICAL_NAME = "e04_string_loop"
    _ROPE_MODEL_LABEL = "PhysX Surface Deformable closed Cat's Cradle loop"
    _ROPE_ROOT_PATH = E04_STRING_ROOT_PATH
    _ROPE_NODE_COUNT = E04_STRING_NODE_COUNT
    _ROPE_RING_COUNT = E04_STRING_RING_COUNT
    _ROPE_RADIAL_VERTEX_COUNT = E04_STRING_RADIAL_VERTEX_COUNT
    _ROPE_SEGMENT_COUNT = E04_STRING_SEGMENT_COUNT
    _ROPE_RADIUS = E04_STRING_RADIUS
    _ROPE_DIAMETER = E04_STRING_DIAMETER
    _ROPE_CENTERLINE_LENGTH = E04_STRING_CENTERLINE_LENGTH
    _ROPE_CONTACT_OFFSET = E04_STRING_CONTACT_OFFSET
    _ROPE_REST_OFFSET = E04_STRING_REST_OFFSET

    def __init__(self, task_name: str = E04_TASK_NAME, **kwargs: Any):
        self._e04_initial_state = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._e04_source_centers_np = e04_figure_centerline(0)
        self._e04_target_centers_np = e04_figure_centerline(1)
        self._e04_last_metrics: dict[str, Any] = {}
        self._e04_source_verified = False
        self._e04_target_crossing_count = 0
        super().__init__(task_name=task_name, **kwargs)
        self._e02_keypoint_indices = np.linspace(
            0,
            E04_STRING_SEGMENT_COUNT - 1,
            _E04_KEYPOINT_COUNT,
        ).round().astype(np.int64)

    def _initial_rope_centers(self) -> np.ndarray:
        return e04_figure_centerline(0)

    def _initial_surface_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return e04_string_surface_mesh(0)

    def _spawn_surface_deformable_rope(self) -> None:
        if self._stage is None:
            raise RuntimeError("E04 stage is unavailable before string-loop spawn")
        spawn_e04_surface_deformable_string(self._stage)

    def _centerline_state_from_nodes(
        self,
        nodes: np.ndarray,
        nodal_velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(nodes, dtype=np.float64).reshape(
            E04_STRING_RING_COUNT,
            E04_STRING_RADIAL_VERTEX_COUNT,
            3,
        )
        velocities = np.asarray(nodal_velocities, dtype=np.float64).reshape(
            E04_STRING_RING_COUNT,
            E04_STRING_RADIAL_VERTEX_COUNT,
            3,
        )
        return np.mean(nodes, axis=1), np.mean(velocities, axis=1)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        HandsOnABCIsaacLabEnv._sample_reset_state(
            self,
            clear_manual_request=clear_manual_request,
        )
        if self.randomize:
            self.delta_h[:] = self._random.uniform(0.0, 0.025)
            self._e04_initial_state = np.asarray(
                [
                    self._random.uniform(-0.030, 0.030),
                    self._random.uniform(-0.060, 0.060),
                    math.radians(self._random.uniform(-15.0, 15.0)),
                    float(self._random.randint(0, len(E04_CYCLE_TRANSITIONS))),
                    float(self._random.randint(0, E04_STRING_RING_COUNT)),
                    float(self._random.randint(0, 2)),
                    self._random.uniform(0.0, 0.002),
                    float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
        else:
            self.delta_h[:] = 0.0
            self._e04_initial_state = np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
        self._e04_last_metrics = {}
        self._e02_last_metrics = {}
        self._e04_source_verified = False

    def _transition(self):
        index = int(round(float(self._e04_initial_state[3]))) % len(
            E04_CYCLE_TRANSITIONS
        )
        return index, E04_CYCLE_TRANSITIONS[index]

    def _transform_figure_points(self, points: np.ndarray) -> np.ndarray:
        transformed = np.asarray(points, dtype=np.float64).copy()
        dx, dy, yaw, _transition, _phase, mirror, clearance, delta_h = (
            self._e04_initial_state
        )
        anchor = np.mean(transformed, axis=0)
        if int(round(float(mirror))):
            transformed[:, 1] = 2.0 * anchor[1] - transformed[:, 1]
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        transformed[:, :2] = (
            (transformed[:, :2] - anchor[:2]) @ rotation.T + anchor[:2]
        )
        transformed[:, 0] += float(dx)
        transformed[:, 1] += float(dy)
        transformed[:, 2] += float(clearance + delta_h)
        return transformed

    def _rolled_centers(self, figure_index: int) -> np.ndarray:
        phase_roll = int(round(float(self._e04_initial_state[4]))) % (
            E04_STRING_RING_COUNT
        )
        centers = np.roll(e04_figure_centerline(figure_index), phase_roll, axis=0)
        return self._transform_figure_points(centers)

    def _rolled_nodes(self, figure_index: int) -> np.ndarray:
        phase_roll = int(round(float(self._e04_initial_state[4]))) % (
            E04_STRING_RING_COUNT
        )
        nodes, _face_counts, _face_indices = e04_string_surface_mesh(figure_index)
        rings = nodes.reshape(
            E04_STRING_RING_COUNT,
            E04_STRING_RADIAL_VERTEX_COUNT,
            3,
        )
        rings = np.roll(rings, phase_roll, axis=0)
        return self._transform_figure_points(rings.reshape(-1, 3))

    def _apply_task_pose_randomization(self) -> None:
        self._write_table_height()
        _transition_index, transition = self._transition()
        source_index = E04_FIGURE_INDEX[transition.source]
        target_index = E04_FIGURE_INDEX[transition.target]
        source_centers = self._rolled_centers(source_index)
        target_centers = self._rolled_centers(target_index)
        reset_nodes = self._rolled_nodes(source_index)

        self._e04_source_centers_np = source_centers.copy()
        self._e04_target_centers_np = target_centers.copy()
        self._e02_reset_centers_np = source_centers.copy()
        self._e02_centers_np = source_centers.copy()
        self._e02_velocities_np = np.zeros_like(source_centers)
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
                    "E04 tensor reset write did not reach PhysX: "
                    f"max_node_error={max_error:.6f}m"
                )

        source_rms = self._shape_rms(self._e02_centers_np, source_centers)
        self._e04_source_verified = bool(source_rms <= 1e-8)
        saved_points = self._e02_centers_np
        self._e02_centers_np = self._e04_target_centers_np
        try:
            self._e04_target_crossing_count = int(
                self._rope_contact_geometry()["projected_crossing_count"]
            )
        finally:
            self._e02_centers_np = saved_points
        self._e04_last_metrics = self._rope_metrics(update_history=False)
        print(
            "[HandsOn/IsaacLab] Reset E04 closed-loop string figure: "
            f"transition={transition.source}->{transition.target}, "
            f"source_verified={self._e04_source_verified}, "
            f"phase_roll={int(self._e04_initial_state[4])}, "
            f"mirror={bool(round(float(self._e04_initial_state[5])))}, "
            f"yaw={math.degrees(self._e04_initial_state[2]):+.1f}deg, "
            f"table_delta_h={self.delta_h[0]:.4f}m",
            flush=True,
        )

    def _rope_pose(self) -> np.ndarray:
        center = np.mean(self._e02_centers_np, axis=0)
        centered = self._e02_centers_np[:, :2] - center[:2]
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        _values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        if axis[0] < 0.0:
            axis *= -1.0
        yaw = math.atan2(float(axis[1]), float(axis[0]))
        return np.asarray(
            [*center, math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
            dtype=np.float64,
        )

    def _rope_contact_geometry(self) -> dict[str, Any]:
        points = self._e02_centers_np
        next_points = np.roll(points, -1, axis=0)
        first_start = points[_E04_PAIR_FIRST]
        first_delta = next_points[_E04_PAIR_FIRST] - first_start
        second_start = points[_E04_PAIR_SECOND]
        second_delta = next_points[_E04_PAIR_SECOND] - second_start

        # Five samples per link give a conservative, vectorized proximity
        # estimate.  Exact XY intersections below replace that estimate with
        # their interpolated Z gap, so actual crossings are never missed.
        first_samples = (
            first_start[:, None, :]
            + _E04_SEGMENT_SAMPLES[None, :, None] * first_delta[:, None, :]
        )
        second_samples = (
            second_start[:, None, :]
            + _E04_SEGMENT_SAMPLES[None, :, None] * second_delta[:, None, :]
        )
        pairwise = first_samples[:, :, None, :] - second_samples[:, None, :, :]
        sampled_distance = np.min(np.linalg.norm(pairwise, axis=-1), axis=(1, 2))

        r = first_delta[:, :2]
        s = second_delta[:, :2]
        denominator = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
        delta = second_start[:, :2] - first_start[:, :2]
        nonparallel = np.abs(denominator) >= 1e-10
        safe_denominator = np.where(nonparallel, denominator, 1.0)
        first_parameter = (
            delta[:, 0] * s[:, 1] - delta[:, 1] * s[:, 0]
        ) / safe_denominator
        second_parameter = (
            delta[:, 0] * r[:, 1] - delta[:, 1] * r[:, 0]
        ) / safe_denominator
        intersects = (
            nonparallel
            & (first_parameter >= 0.02)
            & (first_parameter <= 0.98)
            & (second_parameter >= 0.02)
            & (second_parameter <= 0.98)
        )
        first_z = first_start[:, 2] + first_parameter * first_delta[:, 2]
        second_z = second_start[:, 2] + second_parameter * second_delta[:, 2]
        z_gap = np.abs(first_z - second_z)
        effective_distance = np.where(
            intersects,
            np.minimum(sampled_distance, z_gap),
            sampled_distance,
        )
        crossing_mask = intersects & (z_gap <= _E04_MAX_CROSSING_Z_GAP)
        return {
            "nonlocal_contact_count": int(
                np.count_nonzero(effective_distance <= _E04_CONTACT_DISTANCE)
            ),
            "projected_crossing_count": int(np.count_nonzero(crossing_mask)),
            "min_nonlocal_distance": float(np.min(effective_distance)),
        }

    @staticmethod
    def _shape_rms(points: np.ndarray, target: np.ndarray) -> float:
        points = np.asarray(points, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        centered_points = points - np.mean(points, axis=0)
        centered_target = target - np.mean(target, axis=0)
        return float(
            np.sqrt(np.mean(np.sum(np.square(centered_points - centered_target), axis=1)))
        )

    def _hand_string_distances(self) -> tuple[float, float]:
        points = self._e02_centers_np
        distances: list[float] = []
        for side in ("right", "left"):
            minimum = math.inf
            for hand_point in self._a06_hand_points(side):
                minimum = min(
                    minimum,
                    float(np.min(np.linalg.norm(points - hand_point, axis=1))),
                )
            distances.append(minimum)
        return float(distances[0]), float(distances[1])

    def _rope_metrics(self, *, update_history: bool) -> dict[str, Any]:
        del update_history
        geometry = self._rope_contact_geometry()
        target_geometry_points = self._e02_centers_np
        current_crossings = geometry["projected_crossing_count"]
        target_crossings = self._e04_target_crossing_count

        source_rms = self._shape_rms(
            target_geometry_points,
            self._e04_source_centers_np,
        )
        target_rms = self._shape_rms(
            target_geometry_points,
            self._e04_target_centers_np,
        )
        centroid_drift = float(
            np.linalg.norm(
                np.mean(target_geometry_points, axis=0)
                - np.mean(self._e04_target_centers_np, axis=0)
            )
        )
        crossing_error = abs(int(current_crossings) - int(target_crossings))
        speed = np.linalg.norm(self._e02_velocities_np, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(speed))))
        max_speed = float(np.max(speed))
        right_distance, left_distance = self._hand_string_distances()
        bounds_min = np.min(target_geometry_points, axis=0) - E04_STRING_RADIUS
        bounds_max = np.max(target_geometry_points, axis=0) + E04_STRING_RADIUS
        table_top = E04_TABLE_TOP_Z + float(self.delta_h[0])

        source_changed = bool(
            self._e04_source_verified and source_rms >= _E04_SOURCE_DEPARTURE_RMS
        )
        target_matched = bool(
            target_rms <= _E04_TARGET_RMS_TOLERANCE
            and centroid_drift <= _E04_MAX_CENTROID_DRIFT
            and crossing_error <= 1
        )
        collision_safe = bool(
            geometry["min_nonlocal_distance"] >= _E04_MIN_NONLOCAL_DISTANCE
        )
        on_table = bool(
            bounds_min[2] >= table_top - 0.025
            and bounds_max[2] <= table_top + 0.34
        )
        in_workspace = bool(
            bounds_min[0] >= _E04_TABLE_X_BOUNDS[0] - 0.03
            and bounds_max[0] <= _E04_TABLE_X_BOUNDS[1] + 0.03
            and bounds_min[1] >= _E04_TABLE_Y_BOUNDS[0] - 0.03
            and bounds_max[1] <= _E04_TABLE_Y_BOUNDS[1] + 0.03
        )
        hands_supporting = bool(
            right_distance <= _E04_HAND_SUPPORT_DISTANCE
            and left_distance <= _E04_HAND_SUPPORT_DISTANCE
        )
        stable = bool(
            rms_speed <= _E04_STABLE_RMS_SPEED
            and max_speed <= _E04_STABLE_MAX_SPEED
        )
        success_now = bool(
            source_changed
            and target_matched
            and collision_safe
            and on_table
            and in_workspace
            and hands_supporting
            and stable
        )
        return {
            **geometry,
            "source_rms": source_rms,
            "target_rms": target_rms,
            "centroid_drift": centroid_drift,
            "target_crossing_count": int(target_crossings),
            "crossing_error": int(crossing_error),
            "rms_link_speed": rms_speed,
            "max_link_speed": max_speed,
            "right_hand_distance": right_distance,
            "left_hand_distance": left_distance,
            "source_verified": bool(self._e04_source_verified),
            "source_changed": source_changed,
            "target_matched": target_matched,
            "collision_safe": collision_safe,
            "on_table": on_table,
            "in_workspace": in_workspace,
            "hands_supporting": hands_supporting,
            "stable": stable,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "success_now": success_now,
        }

    def _refresh_rope_state(self) -> None:
        super()._refresh_rope_state()
        self._e04_last_metrics = dict(self._e02_last_metrics)

    def _build_observation_space(self) -> spaces.Dict:
        figure_count = len(E04_RECORDED_FIGURE_STEPS)
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "string_pose": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "string_keypoints": spaces.Box(
                -np.inf,
                np.inf,
                shape=(_E04_KEYPOINT_COUNT * 3,),
                dtype=np.float64,
            ),
            "target_keypoints": spaces.Box(
                -np.inf,
                np.inf,
                shape=(_E04_KEYPOINT_COUNT * 3,),
                dtype=np.float64,
            ),
            "source_figure": spaces.Box(
                0.0, 1.0, shape=(figure_count,), dtype=np.float64
            ),
            "target_figure": spaces.Box(
                0.0, 1.0, shape=(figure_count,), dtype=np.float64
            ),
            "transition_indices": spaces.Box(
                0.0, float(figure_count - 1), shape=(2,), dtype=np.float64
            ),
            "figure_metrics": spaces.Box(
                -np.inf, np.inf, shape=(20,), dtype=np.float64
            ),
            "string_initial_state": spaces.Box(
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
        metrics = self._e04_last_metrics or self._rope_metrics(update_history=False)
        tcp, gripper = self._robot_state()
        transition_index, transition = self._transition()
        source_index = E04_FIGURE_INDEX[transition.source]
        target_index = E04_FIGURE_INDEX[transition.target]
        source_selector = np.zeros(len(E04_FIGURE_KEYS), dtype=np.float64)
        target_selector = np.zeros(len(E04_FIGURE_KEYS), dtype=np.float64)
        source_selector[source_index] = 1.0
        target_selector[target_index] = 1.0
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "string_pose": self._rope_pose(),
            "string_keypoints": self._e02_centers_np[
                self._e02_keypoint_indices
            ].reshape(-1).astype(np.float64, copy=True),
            "target_keypoints": self._e04_target_centers_np[
                self._e02_keypoint_indices
            ].reshape(-1).astype(np.float64, copy=True),
            "source_figure": source_selector,
            "target_figure": target_selector,
            "transition_indices": np.asarray(
                [source_index, target_index], dtype=np.float64
            ),
            "figure_metrics": np.asarray(
                [
                    metrics["source_rms"],
                    metrics["target_rms"],
                    metrics["centroid_drift"],
                    metrics["projected_crossing_count"],
                    metrics["target_crossing_count"],
                    metrics["crossing_error"],
                    metrics["nonlocal_contact_count"],
                    metrics["min_nonlocal_distance"],
                    metrics["rms_link_speed"],
                    metrics["max_link_speed"],
                    metrics["right_hand_distance"],
                    metrics["left_hand_distance"],
                    metrics["source_verified"],
                    metrics["source_changed"],
                    metrics["target_matched"],
                    metrics["collision_safe"],
                    metrics["on_table"],
                    metrics["in_workspace"],
                    metrics["hands_supporting"],
                    self._success_counter / max(1, self._success_steps_required()),
                ],
                dtype=np.float64,
            ),
            "string_initial_state": self._e04_initial_state.astype(
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
        self._e04_last_metrics = self._rope_metrics(update_history=False)
        self._e02_last_metrics = dict(self._e04_last_metrics)
        return bool(self._e04_last_metrics["success_now"])

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._e04_last_metrics or self._rope_metrics(update_history=False)
        _index, transition = self._transition()
        target_label = E04_FIGURE_LABELS[E04_FIGURE_INDEX[transition.target]]
        return [
            self._condition_row(
                "e04_source_changed",
                "The sampled source figure has been changed",
                metrics["source_changed"],
            ),
            self._condition_row(
                "e04_target_matched",
                f"The requested next figure ({target_label}) is matched",
                metrics["target_matched"],
            ),
            self._condition_row(
                "e04_supported",
                "Both hands support the collision-safe loop inside the workspace",
                metrics["hands_supporting"]
                and metrics["collision_safe"]
                and metrics["on_table"]
                and metrics["in_workspace"],
            ),
            self._condition_row(
                "success_hold",
                f"The requested figure remains stable for {self._success_steps_required()} frames",
                self._success_counter >= self._success_steps_required(),
            ),
        ]

    def _compute_failure(self) -> bool:
        metrics = self._e04_last_metrics or self._rope_metrics(update_history=False)
        center = np.mean(self._e02_centers_np, axis=0)
        return bool(
            metrics["bounds_min"][2] < 0.55
            or center[0] < -0.90
            or center[0] > 0.60
            or abs(float(center[1])) > 1.10
        )

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        transition_index, transition = self._transition()
        info.update(
            {
                "task_id": "E04",
                "transition_index": transition_index,
                "source_figure": transition.source,
                "source_figure_label": E04_FIGURE_LABELS[
                    E04_FIGURE_INDEX[transition.source]
                ],
                "target_figure": transition.target,
                "target_figure_label": E04_FIGURE_LABELS[
                    E04_FIGURE_INDEX[transition.target]
                ],
                "transition_operation": transition.operation,
                "recorded_figure_sequence": list(E04_FIGURE_KEYS),
                "cycle_transition_count": len(E04_CYCLE_TRANSITIONS),
                "closed_loop": True,
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
            state_parts.get("string_initial_state", self._e04_initial_state),
            dtype=np.float64,
        ).ravel()
        if initial.size >= 7:
            self._e04_initial_state[:7] = initial[:7]
        self._e04_initial_state[7] = float(self.delta_h[0])
        if self._robots:
            self._restore_task_scene_state()
        self._apply_task_pose_randomization()
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()


__all__ = ["HandsOnE04CatsCradleIsaacLabEnv"]
