"""Native IsaacLab environment for folding a cloth and wiping reachable table."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .e_collection_env import HandsOnEIsaacLabEnv, _tensor_to_numpy
from .e05_fold_wipe_scene import (
    E05_CLOTH_CONTACT_OFFSET,
    E05_CLOTH_NODE_COUNT,
    E05_CLOTH_TRIANGLE_COUNT,
    E05_CLOTH_REST_OFFSET,
    E05_CLOTH_ROOT_PATH,
    E05_CLOTH_SIM_MESH_PATH,
    E05_TABLE_TOP_Z,
    E05_TASK_NAME,
    E05_WIPE_GRID_CELL_COUNT,
    E05_WIPE_GRID_CELL_SIZE,
    e05_initial_cloth_nodes,
    e05_reachable_wipe_masks,
    e05_wipe_grid_centers,
    spawn_e05_surface_deformable_cloth,
)


_E05_FOLD_RATIO_THRESHOLD = 0.48
_E05_FOLD_AREA_RATIO_THRESHOLD = 0.68
_E05_FOLD_MAINTAIN_RATIO = 0.58
_E05_FOLD_MAINTAIN_AREA_RATIO = 0.76
_E05_CONTACT_HEIGHT = 0.020
_E05_MIN_CONTACT_NODES = 10
_E05_HAND_CONTACT_DISTANCE = 0.075
_E05_WIPE_MARK_RADIUS = 0.75 * E05_WIPE_GRID_CELL_SIZE
_E05_UNION_COVERAGE_THRESHOLD = 0.90
_E05_SIDE_COVERAGE_THRESHOLD = 0.86
_E05_MIN_WIPE_PATH_LENGTH = 0.80
_E05_STABLE_RMS_SPEED = 0.040
_E05_STABLE_MAX_SPEED = 0.22
_E05_TABLE_X_BOUNDS = (-0.55, 0.25)
_E05_TABLE_Y_BOUNDS = (-0.80, 0.80)


class HandsOnE05FoldWipeIsaacLabEnv(HandsOnEIsaacLabEnv):
    """Two-stage E05: physically fold the cloth, then wipe reachable cells."""

    _TASK_NAME = E05_TASK_NAME
    _TASK_ID = "E05"
    _SURFACE_LOGICAL_NAME = "e05_cloth"
    _SURFACE_ROOT_PATH = E05_CLOTH_ROOT_PATH
    _SURFACE_ASSET_LABEL = "procedural microfiber wiping cloth"

    def __init__(self, task_name: str = E05_TASK_NAME, **kwargs: Any):
        self._e05_grid_centers = e05_wipe_grid_centers()
        (
            self._e05_right_reachable_mask,
            self._e05_left_reachable_mask,
            self._e05_reachable_mask,
            self._e05_reach_metadata,
        ) = e05_reachable_wipe_masks()
        self._e05_covered_mask = np.zeros(E05_WIPE_GRID_CELL_COUNT, dtype=bool)
        self._e05_fold_completed = False
        self._e05_wipe_started = False
        self._e05_wipe_path_length = 0.0
        self._e05_last_wipe_centroid: np.ndarray | None = None
        self._e05_initial_state = np.zeros(6, dtype=np.float64)
        self._e05_last_metrics: dict[str, Any] = {}
        super().__init__(task_name=task_name, **kwargs)

    def _interface_surface_nodes(self) -> np.ndarray:
        return e05_initial_cloth_nodes()

    def _surface_mesh_metadata(self) -> dict[str, Any]:
        # E05 authors a 17x13 cloth sheet, not the E01 GarmentLab shirt.
        # Keep the native readback topology guards bound to the actual mesh.
        return {
            "label": self._SURFACE_ASSET_LABEL,
            "sim_vertex_count": E05_CLOTH_NODE_COUNT,
            "sim_triangle_count": E05_CLOTH_TRIANGLE_COUNT,
            "visual_vertex_count": E05_CLOTH_NODE_COUNT,
            "visual_triangle_count": E05_CLOTH_TRIANGLE_COUNT,
        }

    def _spawn_surface_deformable(self) -> Any:
        if self._stage is None:
            raise RuntimeError("E05 stage is unavailable before cloth spawn")
        return spawn_e05_surface_deformable_cloth(self._stage)

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        """Keep the thin cloth and dexterous fingers from tunneling."""
        del sim_utils
        if self._stage is None:
            return 0, 0

        from pxr import PhysxSchema, Usd, UsdPhysics

        tuned_bodies = 0
        cloth_root = self._stage.GetPrimAtPath(E05_CLOTH_ROOT_PATH)
        cloth_ccd = cloth_root.GetAttribute(
            "physxDeformableBody:enableSpeculativeCCD"
        )
        if cloth_ccd.IsValid():
            cloth_ccd.Set(True)
            tuned_bodies += 1

        robots_root = self._stage.GetPrimAtPath("/World/HandsOnRobots")
        if robots_root.IsValid():
            for prim in Usd.PrimRange(robots_root):
                path_lower = str(prim.GetPath()).lower()
                is_hand = "palm" in path_lower or "finger" in path_lower
                if is_hand and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
                    rigid_api.CreateSolverPositionIterationCountAttr().Set(64)
                    rigid_api.CreateSolverVelocityIterationCountAttr().Set(8)
                    rigid_api.CreateMaxDepenetrationVelocityAttr().Set(0.60)
                    tuned_bodies += 1

        tuned_colliders = 0
        simulation_mesh = self._stage.GetPrimAtPath(E05_CLOTH_SIM_MESH_PATH)
        if simulation_mesh.IsValid():
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(simulation_mesh)
            collision_api.CreateContactOffsetAttr().Set(E05_CLOTH_CONTACT_OFFSET)
            collision_api.CreateRestOffsetAttr().Set(E05_CLOTH_REST_OFFSET)
            tuned_colliders = 1
        return tuned_bodies, tuned_colliders

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        HandsOnABCIsaacLabEnv._sample_reset_state(
            self,
            clear_manual_request=clear_manual_request,
        )
        if self.randomize:
            self.delta_h[:] = self._random.uniform(0.0, 0.025)
            self._e05_initial_state = np.asarray(
                [
                    self._random.uniform(-0.025, 0.025),
                    self._random.uniform(-0.055, 0.055),
                    math.radians(self._random.uniform(-15.0, 15.0)),
                    self._random.uniform(0.0, 0.010),
                    float(self._random.randint(0, 3)),
                    float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
        else:
            self.delta_h[:] = 0.0
            self._e05_initial_state = np.zeros(6, dtype=np.float64)
        self._e_initial_state = self._e05_initial_state.copy()
        self._e05_covered_mask[:] = False
        self._e05_fold_completed = False
        self._e05_wipe_started = False
        self._e05_wipe_path_length = 0.0
        self._e05_last_wipe_centroid = None
        self._e05_last_metrics = {}
        self._e_last_metrics = {}

    def _garment_reset_nodes(self) -> np.ndarray:
        nodes = self._e_base_nodes_np.copy()
        center = np.mean(nodes, axis=0)
        dx, dy, yaw, wrinkle, variant, delta_h = self._e05_initial_state
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        centered_xy = nodes[:, :2] - center[:2]
        nodes[:, :2] = centered_xy @ rotation.T + center[:2]
        if wrinkle > 0.0:
            x_scale = max(1e-9, float(np.max(np.abs(centered_xy[:, 0]))))
            y_scale = max(1e-9, float(np.max(np.abs(centered_xy[:, 1]))))
            normalized_x = centered_xy[:, 0] / x_scale
            normalized_y = centered_xy[:, 1] / y_scale
            variant_index = int(round(float(variant))) % 3
            if variant_index == 0:
                profile = np.sin(math.pi * normalized_x) * np.cos(
                    math.pi * normalized_y
                )
            elif variant_index == 1:
                profile = np.sin(2.0 * math.pi * normalized_y + normalized_x)
            else:
                profile = np.cos(1.5 * math.pi * normalized_x - normalized_y)
            nodes[:, 2] += float(wrinkle) * (0.5 + 0.5 * profile)
        nodes[:, 0] += float(dx)
        nodes[:, 1] += float(dy)
        nodes[:, 2] += float(delta_h)
        return nodes

    def _coverage_ratio(self, mask: np.ndarray) -> float:
        target_count = int(np.count_nonzero(mask))
        if target_count == 0:
            return 0.0
        return float(np.count_nonzero(self._e05_covered_mask & mask) / target_count)

    def _cloth_metrics(self) -> dict[str, Any]:
        nodes = self._e_nodes_np
        velocities = self._e_velocities_np
        lower, upper = self._e_fold_masks
        lower_center = np.mean(nodes[lower, :2], axis=0)
        upper_center = np.mean(nodes[upper, :2], axis=0)
        half_distance = float(np.linalg.norm(lower_center - upper_center))
        fold_ratio = half_distance / max(self._e_baseline_half_distance, 1e-9)
        area_ratio = self._oriented_xy_area(nodes) / max(self._e_baseline_area, 1e-9)
        bounds_min = np.min(nodes, axis=0)
        bounds_max = np.max(nodes, axis=0)
        centroid = np.mean(nodes, axis=0)
        table_top = E05_TABLE_TOP_Z + float(self.delta_h[0])
        contact_mask = nodes[:, 2] <= table_top + _E05_CONTACT_HEIGHT
        contact_node_count = int(np.count_nonzero(contact_mask))
        hand_distance = self._minimum_hand_distance(nodes)
        speed = np.linalg.norm(velocities, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(speed))))
        max_speed = float(np.max(speed))

        compact_fold = bool(
            fold_ratio <= _E05_FOLD_RATIO_THRESHOLD
            and area_ratio <= _E05_FOLD_AREA_RATIO_THRESHOLD
            and bounds_max[2] - bounds_min[2] <= 0.12
        )
        fold_maintained = bool(
            fold_ratio <= _E05_FOLD_MAINTAIN_RATIO
            and area_ratio <= _E05_FOLD_MAINTAIN_AREA_RATIO
        )
        on_table = bool(
            bounds_min[2] >= table_top - 0.025
            and bounds_max[2] <= table_top + 0.26
        )
        in_workspace = bool(
            bounds_min[0] >= _E05_TABLE_X_BOUNDS[0] - 0.03
            and bounds_max[0] <= _E05_TABLE_X_BOUNDS[1] + 0.03
            and bounds_min[1] >= _E05_TABLE_Y_BOUNDS[0] - 0.03
            and bounds_max[1] <= _E05_TABLE_Y_BOUNDS[1] + 0.03
        )
        stable = bool(
            rms_speed <= _E05_STABLE_RMS_SPEED
            and max_speed <= _E05_STABLE_MAX_SPEED
        )
        contact_valid = bool(
            self._e05_fold_completed
            and fold_maintained
            and on_table
            and in_workspace
            and contact_node_count >= _E05_MIN_CONTACT_NODES
            and hand_distance <= _E05_HAND_CONTACT_DISTANCE
        )
        union_coverage = self._coverage_ratio(self._e05_reachable_mask)
        right_coverage = self._coverage_ratio(self._e05_right_reachable_mask)
        left_coverage = self._coverage_ratio(self._e05_left_reachable_mask)
        coverage_complete = bool(
            union_coverage >= _E05_UNION_COVERAGE_THRESHOLD
            and right_coverage >= _E05_SIDE_COVERAGE_THRESHOLD
            and left_coverage >= _E05_SIDE_COVERAGE_THRESHOLD
        )
        success_now = bool(
            self._e05_fold_completed
            and fold_maintained
            and coverage_complete
            and self._e05_wipe_path_length >= _E05_MIN_WIPE_PATH_LENGTH
            and on_table
            and in_workspace
            and stable
        )
        return {
            "fold_ratio": float(fold_ratio),
            "area_ratio": float(area_ratio),
            "half_centroid_distance": half_distance,
            "z_span": float(bounds_max[2] - bounds_min[2]),
            "hand_distance": float(hand_distance),
            "rms_nodal_speed": rms_speed,
            "max_nodal_speed": max_speed,
            "contact_node_count": contact_node_count,
            "compact_fold": compact_fold,
            "fold_completed": bool(self._e05_fold_completed),
            "fold_maintained": fold_maintained,
            "wipe_started": bool(self._e05_wipe_started),
            "contact_valid": contact_valid,
            "wipe_path_length": float(self._e05_wipe_path_length),
            "union_coverage": union_coverage,
            "right_coverage": right_coverage,
            "left_coverage": left_coverage,
            "covered_cell_count": int(
                np.count_nonzero(self._e05_covered_mask & self._e05_reachable_mask)
            ),
            "target_cell_count": int(np.count_nonzero(self._e05_reachable_mask)),
            "coverage_complete": coverage_complete,
            "on_table": on_table,
            "in_workspace": in_workspace,
            "stable": stable,
            "centroid": centroid,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "contact_mask": contact_mask,
            "success_now": success_now,
        }

    def _update_task_progress(self, metrics: dict[str, Any]) -> None:
        if not self._e05_fold_completed:
            if (
                metrics["compact_fold"]
                and metrics["stable"]
                and metrics["on_table"]
                and metrics["in_workspace"]
            ):
                self._e05_fold_completed = True
                self._e05_last_wipe_centroid = None
            return

        if not metrics["contact_valid"]:
            self._e05_last_wipe_centroid = None
            return
        contact_points = self._e_nodes_np[metrics["contact_mask"], :2]
        contact_centroid = np.mean(contact_points, axis=0)
        distances_squared = np.sum(
            np.square(
                self._e05_grid_centers[:, None, :] - contact_points[None, :, :]
            ),
            axis=2,
        )
        touched = np.min(distances_squared, axis=1) <= _E05_WIPE_MARK_RADIUS**2
        self._e05_covered_mask |= touched & self._e05_reachable_mask
        if self._e05_last_wipe_centroid is not None:
            increment = float(
                np.linalg.norm(contact_centroid - self._e05_last_wipe_centroid)
            )
            if 0.001 <= increment <= 0.08:
                self._e05_wipe_path_length += increment
        self._e05_last_wipe_centroid = contact_centroid
        self._e05_wipe_started = True

    def _garment_metrics(self) -> dict[str, Any]:
        return self._cloth_metrics()

    def _refresh_garment_state(self) -> None:
        if self._e_garment_view is not None:
            self._e_nodes_np = _tensor_to_numpy(
                self._e_garment_view.get_simulation_nodal_positions()
            )[0]
            self._e_velocities_np = _tensor_to_numpy(
                self._e_garment_view.get_simulation_nodal_velocities()
            )[0]
        metrics = self._cloth_metrics()
        self._update_task_progress(metrics)
        self._e05_last_metrics = self._cloth_metrics()
        self._e_last_metrics = dict(self._e05_last_metrics)

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "cloth_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "cloth_keypoints": spaces.Box(
                -np.inf, np.inf, shape=(24,), dtype=np.float64
            ),
            "cloth_aabb": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "fold_metrics": spaces.Box(
                -np.inf, np.inf, shape=(12,), dtype=np.float64
            ),
            "wipe_metrics": spaces.Box(
                -np.inf, np.inf, shape=(16,), dtype=np.float64
            ),
            "wipe_target_mask": spaces.Box(
                0.0,
                1.0,
                shape=(E05_WIPE_GRID_CELL_COUNT,),
                dtype=np.float64,
            ),
            "wipe_coverage_mask": spaces.Box(
                0.0,
                1.0,
                shape=(E05_WIPE_GRID_CELL_COUNT,),
                dtype=np.float64,
            ),
            "task_phase": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float64),
            "cloth_initial_state": spaces.Box(
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

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._e05_last_metrics or self._cloth_metrics()
        tcp, gripper = self._robot_state()
        if metrics["coverage_complete"]:
            phase_index = 2
        elif self._e05_fold_completed:
            phase_index = 1
        else:
            phase_index = 0
        phase = np.zeros(3, dtype=np.float64)
        phase[phase_index] = 1.0
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "cloth_pose": self._garment_pose(),
            "cloth_keypoints": self._e_nodes_np[
                self._e_keypoint_indices
            ].reshape(-1).astype(np.float64, copy=True),
            "cloth_aabb": np.concatenate(
                [metrics["bounds_min"], metrics["bounds_max"]]
            ).astype(np.float64),
            "fold_metrics": np.asarray(
                [
                    metrics["fold_ratio"],
                    metrics["area_ratio"],
                    metrics["half_centroid_distance"],
                    metrics["z_span"],
                    metrics["hand_distance"],
                    metrics["rms_nodal_speed"],
                    metrics["max_nodal_speed"],
                    metrics["compact_fold"],
                    metrics["fold_completed"],
                    metrics["fold_maintained"],
                    metrics["on_table"],
                    metrics["in_workspace"],
                ],
                dtype=np.float64,
            ),
            "wipe_metrics": np.asarray(
                [
                    metrics["wipe_started"],
                    metrics["contact_valid"],
                    metrics["contact_node_count"],
                    metrics["wipe_path_length"],
                    metrics["union_coverage"],
                    metrics["right_coverage"],
                    metrics["left_coverage"],
                    metrics["covered_cell_count"],
                    metrics["target_cell_count"],
                    metrics["coverage_complete"],
                    _E05_UNION_COVERAGE_THRESHOLD,
                    _E05_SIDE_COVERAGE_THRESHOLD,
                    _E05_MIN_WIPE_PATH_LENGTH,
                    metrics["stable"],
                    metrics["success_now"],
                    self._success_counter / max(1, self._success_steps_required()),
                ],
                dtype=np.float64,
            ),
            "wipe_target_mask": self._e05_reachable_mask.astype(
                np.float64, copy=True
            ),
            "wipe_coverage_mask": self._e05_covered_mask.astype(
                np.float64, copy=True
            ),
            "task_phase": phase,
            "cloth_initial_state": self._e05_initial_state.astype(
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
        self._e05_last_metrics = self._cloth_metrics()
        self._e_last_metrics = dict(self._e05_last_metrics)
        return bool(self._e05_last_metrics["success_now"])

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._e05_last_metrics or self._cloth_metrics()
        rows = [
            self._condition_row(
                "e05_fold_first",
                "The wiping cloth was folded before wiping started",
                metrics["fold_completed"],
            ),
            self._condition_row(
                "e05_fold_maintained",
                "The cloth remains folded while contacting the table",
                metrics["fold_maintained"],
            ),
            self._condition_row(
                "e05_reachable_coverage",
                "At least 90% of the C01 arm-reachable table mask is wiped",
                metrics["coverage_complete"],
            ),
            self._condition_row(
                "e05_wipe_path",
                f"The folded cloth swept at least {_E05_MIN_WIPE_PATH_LENGTH:.2f} m",
                metrics["wipe_path_length"] >= _E05_MIN_WIPE_PATH_LENGTH,
            ),
            self._condition_row(
                "success_hold",
                f"The final cloth remains stable for {self._success_steps_required()} frames",
                self._success_counter >= self._success_steps_required(),
            ),
        ]
        return rows

    def _task_info(self) -> dict[str, Any]:
        metrics = self._e05_last_metrics or self._cloth_metrics()
        return {
            "task_id": "E05",
            "task_name": self.task_name,
            "cloth_asset": str(self._e_asset_path or self._SURFACE_ASSET_LABEL),
            "cloth_node_count": int(self._e_nodes_np.shape[0]),
            "task_phase": (
                "complete"
                if metrics["coverage_complete"]
                else "wipe"
                if self._e05_fold_completed
                else "fold"
            ),
            "reachability_source": str(
                "c01_rokae_wuji_reachability.npz"
            ),
            "reachability_resolution_m": self._e05_reach_metadata[
                "resolution_m"
            ],
            "reachability_samples_per_arm": int(
                self._e05_reach_metadata["samples_per_arm"]
            ),
            "reachable_cell_count": int(
                self._e05_reach_metadata["union_cell_count"]
            ),
            "reachable_area_m2": self._e05_reach_metadata["union_area_m2"],
            "success_counter": int(self._success_counter),
            "success_steps_required": int(self._success_steps_required()),
            **{
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "centroid",
                    "bounds_min",
                    "bounds_max",
                    "contact_mask",
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
            self._restore_recorded_table_height(float(delta[0]))
        initial = np.asarray(
            state_parts.get("cloth_initial_state", self._e05_initial_state),
            dtype=np.float64,
        ).ravel()
        if initial.size >= 5:
            self._e05_initial_state[:5] = initial[:5]
        self._e05_initial_state[5] = float(self.delta_h[0])
        self._e_initial_state = self._e05_initial_state.copy()
        self._e05_covered_mask[:] = False
        self._e05_fold_completed = False
        self._e05_wipe_started = False
        self._e05_wipe_path_length = 0.0
        self._e05_last_wipe_centroid = None
        self._apply_task_pose_randomization()
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()


__all__ = ["HandsOnE05FoldWipeIsaacLabEnv"]
