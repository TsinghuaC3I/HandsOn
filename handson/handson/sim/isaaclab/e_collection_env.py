"""Native IsaacLab collection environment for E01 garment folding."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv, _randomized_physx_pose
from .bimanual_assembly_env import _pose_vector_to_matrix
from .e_collection_scene import (
    E01_CONTACT_OFFSET,
    E01_GARMENT_ROOT_PATH,
    E01_PHYSICS_HZ,
    E01SurfacePhysicsTuning,
    E01_TASK_NAME,
    E_TASK_NAMES,
    selected_garmentlab_preset_metadata,
    spawn_e01_garmentlab_surface_deformable,
)


_E_TABLE_TOP_Z = 0.89
_E_TABLE_X_BOUNDS = (-0.55, 0.25)
_E_TABLE_Y_BOUNDS = (-0.80, 0.80)
_E_FOLD_RATIO_THRESHOLD = 0.48
_E_AREA_RATIO_THRESHOLD = 0.70
_F01_DIAGNOSTIC_TASK_NAME = "f01_cotton_cloth_mvp"
_F01_PARTIAL_FOLD_RATIO_THRESHOLD = 0.90
_F01_PARTIAL_AREA_RATIO_THRESHOLD = 0.96
_F01_RIGHT_TARGET_X_BOUNDS = (-0.32, 0.02)
_F01_RIGHT_TARGET_Y_BOUNDS = (-0.44, -0.20)
_F01_GRASP_ATTACH_MIN_FINGER_DISTANCE = 0.010
_F01_GRASP_ATTACH_MAX_FINGER_DISTANCE = 0.050
_F01_GRASP_RELEASE_FINGER_DISTANCE = 0.065
_F01_GRASP_ATTACH_NODE_RADIUS = 0.030
_F01_GRASP_MAX_NODES_PER_HAND = 6
_F01_GRASP_MAX_NODE_SPEED = 0.65
_F01_GRASP_MAX_TARGET_ERROR = 0.055
_F01_HAND_STATIC_FRICTION = 2.2
_F01_HAND_DYNAMIC_FRICTION = 1.6
_F01_HAND_CONTACT_OFFSET = 0.0025
_F01_HAND_REST_OFFSET = 0.0002
_F01_WUJI_TIP_LOCAL_OFFSETS = {
    "right": (
        (-0.00105, -0.00020, 0.0313),
        (-0.00105, 0.00020, 0.0267),
        (-0.00104909, 0.00020104, 0.0267),
        (-0.00105961, 0.000190451, 0.0267),
        (-0.00104464, 0.00021131, 0.0267),
    ),
    "left": (
        (-0.00105, 0.00020, 0.0313),
        (-0.00105210, 0.000210869, 0.0267),
        (-0.00105040, 0.000202672, 0.0267),
        (-0.00104998, 0.000202557, 0.0267),
        (-0.00105041, 0.000201047, 0.0267),
    ),
}
_E_HAND_RELEASE_DISTANCE = 0.065
_E_STABLE_RMS_SPEED = 0.005
_E_STABLE_MAX_SPEED = 0.05
# The authored PhysX surface deformable caps nodal speed at
# ``contactOffset * 120 Hz``.  Derive the guard from that same scene constant
# so contact tuning cannot silently leave diagnostics on a stale limit.  A
# small tolerance covers float32/solver rounding.
_E_CONFIGURED_MAX_NODAL_SPEED = E01_CONTACT_OFFSET * E01_PHYSICS_HZ
_E_MAX_NODAL_SPEED_TOLERANCE = 0.06
_E_MAX_CONTROL_NODE_DISPLACEMENT = 0.015
_E_RESET_SETTLE_MAX_SECONDS = 4.0
# A tensor reset explicitly writes zero nodal velocity.  The first one or two
# PhysX reads can therefore look quiet before gravity, elastic strain, and
# table contacts have had time to respond.  Do not treat that transient as an
# equilibrium: always simulate a short dwell and require a consecutive quiet
# window before exposing the episode.
_E_RESET_SETTLE_MIN_SECONDS = 1.0
_E_RESET_SETTLE_STABLE_SECONDS = 0.50
_E_RESET_SETTLE_MAX_WINDOW_DRIFT = 0.002


def _interface_garment_nodes() -> np.ndarray:
    """Small deterministic shirt-like surface for no-Isaac interface tests."""
    x_values = np.linspace(-0.31, 0.31, 21)
    y_values = np.linspace(-0.19, 0.19, 13)
    points = np.asarray(
        [
            (-0.15 + x, y, _E_TABLE_TOP_Z + 0.004)
            for y in y_values
            for x in x_values
            if abs(y) <= 0.115 or abs(x) <= 0.17
        ],
        dtype=np.float64,
    )
    return points


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    elif hasattr(value, "warp"):
        value = value.warp
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64).copy()


class HandsOnEIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual GarmentLab shirt-folding task with live surface tensors."""

    _TASK_NAME = E01_TASK_NAME
    _TASK_ID = "E01"
    _SURFACE_LOGICAL_NAME = "garment"
    _SURFACE_ROOT_PATH = E01_GARMENT_ROOT_PATH

    def __init__(
        self,
        task_name: str = E01_TASK_NAME,
        *,
        surface_physics_tuning: E01SurfacePhysicsTuning | None = None,
        relaxed_garment_readback_guards: bool = False,
        diagnostic_task_name: str | None = None,
        enable_f01_grasp_assist: bool = False,
        **kwargs: Any,
    ):
        if task_name != self._TASK_NAME:
            raise ValueError(
                f"Unsupported surface-deformable task {task_name!r}; "
                f"expected {self._TASK_NAME!r}"
            )

        # Virtual setup/reset hooks run inside the shared constructor.
        fallback = self._interface_surface_nodes()
        self._e_surface_sim_view: Any | None = None
        self._e_garment_view: Any | None = None
        self._e_garment_indices: Any | None = None
        self._e_asset_path = None
        self._e_base_nodes_np = fallback.copy()
        self._e_nodes_np = fallback.copy()
        self._e_velocities_np = np.zeros_like(fallback)
        self._e_reset_nodes_np = fallback.copy()
        self._e_expected_node_count = int(fallback.shape[0])
        self._e_simulation_triangle_count = 0
        self._e_last_valid_nodes_np = fallback.copy()
        self._e_last_valid_velocities_np = np.zeros_like(fallback)
        self._e_last_control_nodes_np = fallback.copy()
        self._e_readback_failed = False
        self._e_failure_reason: str | None = None
        self._e_failure_diagnostics: dict[str, Any] = {}
        self._e_last_control_node_displacement = 0.0
        self._e_last_readback_max_speed = 0.0
        self._e_reset_settle_steps = 0
        self._e_reset_settle_seconds = 0.0
        self._e_reset_settle_rms_speed = 0.0
        self._e_reset_settle_max_speed = 0.0
        self._e_reset_settle_window_drift = 0.0
        self._e_reset_settle_stable_steps = 0
        self._e_reset_settle_converged = False
        self._e_reset_settle_timed_out = False
        self._e_fold_axis = np.asarray([1.0, 0.0], dtype=np.float64)
        self._e_fold_masks = (
            np.arange(fallback.shape[0] // 2, dtype=np.int64),
            np.arange(fallback.shape[0] // 2, fallback.shape[0], dtype=np.int64),
        )
        self._e_surface_mesh_metadata = self._surface_mesh_metadata()
        self._e_surface_physics_tuning = (
            surface_physics_tuning or E01SurfacePhysicsTuning()
        )
        self._e_relaxed_readback_guards = bool(
            relaxed_garment_readback_guards
        )
        self.diagnostic_task_name = str(diagnostic_task_name or "")
        self._f01_grasp_assist_enabled = bool(enable_f01_grasp_assist)
        self._f01_grasp_assist_ever_applied = False
        self._e_surface_asset_label = self._e_surface_mesh_metadata["label"]
        self._e_surface_sim_node_count_target = int(
            self._e_surface_mesh_metadata["sim_vertex_count"]
        )
        self._e_surface_sim_triangle_count_target = int(
            self._e_surface_mesh_metadata["sim_triangle_count"]
        )
        self._e_surface_visual_vertex_count_target = int(
            self._e_surface_mesh_metadata["visual_vertex_count"]
        )
        self._e_surface_visual_triangle_count_target = int(
            self._e_surface_mesh_metadata["visual_triangle_count"]
        )
        self._e_keypoint_indices = np.arange(8, dtype=np.int64)
        self._e_baseline_half_distance = 1.0
        self._e_baseline_area = 1.0
        self._e_initial_state = np.zeros(6, dtype=np.float64)
        self._e_last_metrics: dict[str, Any] = {}
        self._f01_grasp_assist: dict[str, dict[str, Any] | None] = {
            "right": None,
            "left": None,
        }
        self._f01_grasp_body_indices: dict[str, int] = {}
        self._f01_grasp_assist_disabled = False
        self._f01_grasp_assist_metrics: dict[str, dict[str, Any]] = {
            side: {
                "active": False,
                "pair": "",
                "node_count": 0,
                "reason": "reset",
            }
            for side in ("right", "left")
        }
        self._configure_garment_tracking(self._e_base_nodes_np)
        super().__init__(task_name=task_name, **kwargs)

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        kwargs = super()._in_process_scene_rebuild_kwargs(seed)
        kwargs.update(
            {
                "surface_physics_tuning": self._e_surface_physics_tuning,
                "relaxed_garment_readback_guards": bool(
                    self._e_relaxed_readback_guards
                ),
                "diagnostic_task_name": self.diagnostic_task_name or None,
                "enable_f01_grasp_assist": bool(
                    self._f01_grasp_assist_enabled
                ),
            }
        )
        return kwargs

    def _is_f01_diagnostic(self) -> bool:
        return self.diagnostic_task_name == _F01_DIAGNOSTIC_TASK_NAME

    def _uses_cloth_hand_contact_material(self) -> bool:
        """Enable physical finger contact tuning independently of task metrics."""
        return self._is_f01_diagnostic()

    def _interface_surface_nodes(self) -> np.ndarray:
        return _interface_garment_nodes()

    def _surface_mesh_metadata(self) -> dict[str, Any]:
        """Return the exact topology contract for this surface task."""

        return selected_garmentlab_preset_metadata()

    def _spawn_surface_deformable(self) -> Any:
        if self._stage is None:
            raise RuntimeError(
                f"{self._TASK_ID} stage is unavailable before surface spawn"
            )
        return spawn_e01_garmentlab_surface_deformable(
            self._stage,
            physics_tuning=self._e_surface_physics_tuning,
        )

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        if not self._uses_cloth_hand_contact_material() or self._stage is None:
            return super()._configure_task_collision_safety(sim_utils)

        material_path = "/World/PhysicsMaterials/HandsOnF01CottonHand"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=_F01_HAND_STATIC_FRICTION,
            dynamic_friction=_F01_HAND_DYNAMIC_FRICTION,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)

        bound_links, bound_colliders, candidate_prims = (
            self._bind_f01_hand_contact_material(material_path)
        )
        print(
            "[HandsOn/IsaacLab] F01 cotton hand contact enhancement authored: "
            f"finger_candidates={candidate_prims}, "
            f"links={bound_links}, colliders={bound_colliders}, "
            f"static_friction={_F01_HAND_STATIC_FRICTION:.1f}, "
            f"dynamic_friction={_F01_HAND_DYNAMIC_FRICTION:.1f}, "
            f"contact_offset={_F01_HAND_CONTACT_OFFSET:.4f}",
            flush=True,
        )
        return bound_links, bound_colliders

    def _bind_f01_hand_contact_material(
        self, material_path: str
    ) -> tuple[int, int, int]:
        if self._stage is None:
            return 0, 0, 0

        from pxr import PhysxSchema, UsdPhysics, UsdShade

        hand_material = UsdShade.Material(
            self._stage.GetPrimAtPath(material_path)
        )
        bound_links = 0
        bound_colliders = 0
        candidate_prims = 0
        for prim in self._stage.Traverse():
            path = str(prim.GetPath()).lower()
            is_live_hand = path.startswith("/world/") and "finger" in path
            if not is_live_hand:
                continue
            candidate_prims += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(
                    hand_material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bound_links += 1
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(
                hand_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not collision_api:
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact_attr = collision_api.GetContactOffsetAttr()
            if not contact_attr or not contact_attr.IsValid():
                contact_attr = collision_api.CreateContactOffsetAttr()
            contact_attr.Set(_F01_HAND_CONTACT_OFFSET)
            rest_attr = collision_api.GetRestOffsetAttr()
            if not rest_attr or not rest_attr.IsValid():
                rest_attr = collision_api.CreateRestOffsetAttr()
            rest_attr.Set(_F01_HAND_REST_OFFSET)
            bound_colliders += 1
        return bound_links, bound_colliders, candidate_prims

    # ------------------------------------------------------------------
    # Surface-deformable setup and reset
    # ------------------------------------------------------------------

    def _task_specific_isaac_kit_args(self) -> str | None:
        return "--/persistent/physics/enableDeformableBeta=true"

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        del sim_utils
        self._e_asset_path = self._spawn_surface_deformable()
        print(
            f"[HandsOn/IsaacLab] {self._TASK_ID} asset authored as a native "
            f"{self._e_surface_mesh_metadata.get('solver_label', 'surface deformable')}: {self._e_asset_path}",
            flush=True,
        )

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        if self._uses_cloth_hand_contact_material():
            links, colliders, candidates = (
                self._bind_f01_hand_contact_material(
                    "/World/PhysicsMaterials/HandsOnF01CottonHand"
                )
            )
            print(
                "[HandsOn/IsaacLab] F01 cotton hand contact enhancement live: "
                f"finger_candidates={candidates}, links={links}, "
                f"colliders={colliders}",
                flush=True,
            )
        self._bind_e01_surface_view(initialize_baseline=True)

    @staticmethod
    def _surface_tensor_body(value: Any, *, label: str) -> np.ndarray:
        """Return the single body's ``(node, xyz)`` tensor without reducing it."""
        array = _tensor_to_numpy(value)
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 3:
            raise ValueError(
                f"{label}_tensor_shape={tuple(int(size) for size in array.shape)}"
            )
        return array[0]

    def _surface_array_like(self, reference: Any, values: np.ndarray) -> Any:
        """Create a write buffer for Isaac 2.x Torch or 3.x Warp views."""

        if hasattr(reference, "torch"):
            reference = reference.torch
        elif hasattr(reference, "warp"):
            reference = reference.warp
        payload = np.asarray(values, dtype=np.float32).reshape(reference.shape)
        if hasattr(reference, "detach"):
            return self._torch.as_tensor(
                payload,
                dtype=reference.dtype,
                device=reference.device,
            )
        import warp as wp

        return wp.array(
            payload,
            dtype=reference.dtype,
            device=reference.device,
        )

    def _surface_indices_like(self, reference: Any, count: int) -> Any:
        """Create int32 indices in the active low-level tensor backend."""

        if hasattr(reference, "torch"):
            reference = reference.torch
        elif hasattr(reference, "warp"):
            reference = reference.warp
        if hasattr(reference, "detach"):
            return self._torch.arange(
                int(count),
                dtype=self._torch.long,
                device=reference.device,
            )
        import warp as wp

        return wp.array(
            np.arange(int(count), dtype=np.int32),
            dtype=wp.int32,
            device=reference.device,
        )

    @staticmethod
    def _surface_zeros_like(value: Any) -> Any:
        if hasattr(value, "detach"):
            import torch

            return torch.zeros_like(value)
        import warp as wp

        return wp.zeros_like(value)

    @staticmethod
    def _basic_nodal_state_issue(
        nodes: np.ndarray,
        velocities: np.ndarray,
        *,
        expected_node_count: int | None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Validate topology and finite values before any geometric reduction."""
        nodes = np.asarray(nodes, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        diagnostics: dict[str, Any] = {
            "position_shape": tuple(int(size) for size in nodes.shape),
            "velocity_shape": tuple(int(size) for size in velocities.shape),
            "expected_node_count": (
                None
                if expected_node_count is None
                else int(expected_node_count)
            ),
        }
        if nodes.ndim != 2 or nodes.shape[1:] != (3,) or nodes.shape[0] == 0:
            return "invalid_nodal_position_shape", diagnostics
        if velocities.ndim != 2 or velocities.shape[1:] != (3,):
            return "invalid_nodal_velocity_shape", diagnostics

        # These checks deliberately precede norm/max/mean/eigh/min/ptp calls.
        if not bool(np.all(np.isfinite(nodes))):
            return "nonfinite_nodal_positions", diagnostics
        if not bool(np.all(np.isfinite(velocities))):
            return "nonfinite_nodal_velocities", diagnostics
        if velocities.shape[0] != nodes.shape[0]:
            return "nodal_position_velocity_count_mismatch", diagnostics
        if (
            expected_node_count is not None
            and nodes.shape[0] != int(expected_node_count)
        ):
            diagnostics["readback_node_count"] = int(nodes.shape[0])
            return "garment_node_count_changed", diagnostics
        diagnostics["readback_node_count"] = int(nodes.shape[0])
        return None, diagnostics

    def _dynamic_nodal_state_issue(
        self,
        nodes: np.ndarray,
        velocities: np.ndarray,
        *,
        enforce_control_frame_displacement: bool = True,
    ) -> tuple[str | None, dict[str, Any]]:
        """Validate a live surface readback against the current control frame.

        Reset settling may disable the control-frame displacement check for
        its first PhysX frame because the freshly authored surface is expected
        to relax. Shape, finite-value, speed, and stability checks remain
        active during that phase.
        """
        issue, diagnostics = self._basic_nodal_state_issue(
            nodes,
            velocities,
            expected_node_count=self._e_expected_node_count,
        )
        diagnostics.update(
            {
                "max_nodal_speed_limit": float(
                    self._e_surface_physics_tuning.resolved_max_linear_velocity
                    + _E_MAX_NODAL_SPEED_TOLERANCE
                ),
                "max_control_node_displacement_limit": float(
                    _E_MAX_CONTROL_NODE_DISPLACEMENT
                ),
                "control_frame_displacement_check_enabled": bool(
                    enforce_control_frame_displacement
                ),
            }
        )
        if issue is not None:
            return issue, diagnostics

        speed = np.linalg.norm(velocities, axis=1)
        max_speed = float(np.max(speed))
        diagnostics["max_nodal_speed"] = max_speed
        if max_speed > (
            self._e_surface_physics_tuning.resolved_max_linear_velocity
            + _E_MAX_NODAL_SPEED_TOLERANCE
        ) and not self._e_relaxed_readback_guards:
            return "nodal_speed_limit_exceeded", diagnostics

        previous = self._e_last_control_nodes_np
        previous_issue, _ = self._basic_nodal_state_issue(
            previous,
            self._e_last_valid_velocities_np,
            expected_node_count=self._e_expected_node_count,
        )
        if previous_issue is not None:
            diagnostics["previous_state_issue"] = previous_issue
            return "previous_valid_nodal_state_invalid", diagnostics
        displacement = np.linalg.norm(nodes - previous, axis=1)
        max_displacement = float(np.max(displacement))
        diagnostics["max_control_node_displacement"] = max_displacement
        if (
            enforce_control_frame_displacement
            and max_displacement > _E_MAX_CONTROL_NODE_DISPLACEMENT
            and not self._e_relaxed_readback_guards
        ):
            return "control_frame_node_displacement_limit_exceeded", diagnostics
        return None, diagnostics

    @staticmethod
    def _nodal_speed_summary(velocities: np.ndarray) -> tuple[float, float]:
        """Compute RMS/max speed only after shape and finite-value checks."""
        velocities = np.asarray(velocities, dtype=np.float64)
        if (
            velocities.ndim != 2
            or velocities.shape[1:] != (3,)
            or velocities.shape[0] == 0
        ):
            raise ValueError("Nodal velocities must have shape (node, 3)")
        if not bool(np.all(np.isfinite(velocities))):
            raise ValueError("Nodal velocities must be finite")
        speed = np.linalg.norm(velocities, axis=1)
        return (
            float(np.sqrt(np.mean(np.square(speed)))),
            float(np.max(speed)),
        )

    @staticmethod
    def _nodal_speed_is_stable(rms_speed: float, max_speed: float) -> bool:
        return bool(
            math.isfinite(float(rms_speed))
            and math.isfinite(float(max_speed))
            and float(rms_speed) <= _E_STABLE_RMS_SPEED
            and float(max_speed) <= _E_STABLE_MAX_SPEED
        )

    @staticmethod
    def _reset_settle_step_budget(physics_dt: float) -> int:
        if not math.isfinite(float(physics_dt)) or float(physics_dt) <= 0.0:
            raise ValueError("physics_dt must be finite and positive")
        return max(
            1,
            int(math.ceil(_E_RESET_SETTLE_MAX_SECONDS / float(physics_dt))),
        )

    @staticmethod
    def _reset_settle_minimum_steps(physics_dt: float) -> int:
        if not math.isfinite(float(physics_dt)) or float(physics_dt) <= 0.0:
            raise ValueError("physics_dt must be finite and positive")
        return max(
            1,
            int(math.ceil(_E_RESET_SETTLE_MIN_SECONDS / float(physics_dt))),
        )

    @staticmethod
    def _reset_settle_stable_steps_required(physics_dt: float) -> int:
        if not math.isfinite(float(physics_dt)) or float(physics_dt) <= 0.0:
            raise ValueError("physics_dt must be finite and positive")
        return max(
            1,
            int(math.ceil(_E_RESET_SETTLE_STABLE_SECONDS / float(physics_dt))),
        )

    def _clear_garment_failure_state(self) -> None:
        self._e_readback_failed = False
        self._e_failure_reason = None
        self._e_failure_diagnostics = {}
        self._e_last_control_node_displacement = 0.0
        self._e_last_readback_max_speed = 0.0
        self._e_reset_settle_steps = 0
        self._e_reset_settle_seconds = 0.0
        self._e_reset_settle_rms_speed = 0.0
        self._e_reset_settle_max_speed = 0.0
        self._e_reset_settle_window_drift = 0.0
        self._e_reset_settle_stable_steps = 0
        self._e_reset_settle_converged = False
        self._e_reset_settle_timed_out = False
        self._e_last_valid_nodes_np = self._e_nodes_np.copy()
        self._e_last_valid_velocities_np = self._e_velocities_np.copy()
        self._e_last_control_nodes_np = self._e_nodes_np.copy()

    def _reject_garment_readback(
        self,
        reason: str,
        diagnostics: Mapping[str, Any],
    ) -> None:
        """Latch a terminal failure and retain the last valid control frame."""
        if not self._e_readback_failed:
            self._e_readback_failed = True
            self._e_failure_reason = str(reason)
            self._e_failure_diagnostics = dict(diagnostics)
            details = ", ".join(
                f"{key}={value}"
                for key, value in sorted(self._e_failure_diagnostics.items())
            )
            print(
                f"[HandsOn/IsaacLab] {self._TASK_ID} garment state rejected: "
                f"reason={reason}; {details}; retaining last valid control frame",
                flush=True,
            )
        self._e_nodes_np = self._e_last_valid_nodes_np.copy()
        self._e_velocities_np = self._e_last_valid_velocities_np.copy()

    def _accept_garment_readback(
        self,
        nodes: np.ndarray,
        velocities: np.ndarray,
        diagnostics: Mapping[str, Any],
    ) -> None:
        self._e_nodes_np = np.asarray(nodes, dtype=np.float64).copy()
        self._e_velocities_np = np.asarray(velocities, dtype=np.float64).copy()
        self._e_last_valid_nodes_np = self._e_nodes_np.copy()
        self._e_last_valid_velocities_np = self._e_velocities_np.copy()
        self._e_last_control_nodes_np = self._e_nodes_np.copy()
        self._e_last_control_node_displacement = float(
            diagnostics.get("max_control_node_displacement", 0.0)
        )
        self._e_last_readback_max_speed = float(
            diagnostics.get("max_nodal_speed", 0.0)
        )

    def _ensure_metric_state_is_safe(self) -> None:
        """Fail closed if an in-process caller injected an invalid state."""
        issue, diagnostics = self._basic_nodal_state_issue(
            self._e_nodes_np,
            self._e_velocities_np,
            expected_node_count=self._e_expected_node_count,
        )
        if issue is not None:
            self._reject_garment_readback(issue, diagnostics)

    def _create_surface_tensor_view(self, physics_sim_view):
        return physics_sim_view.create_surface_deformable_body_view(
            self._SURFACE_ROOT_PATH
        )

    def _bind_e01_surface_view(self, *, initialize_baseline: bool) -> None:
        """Bind the current E01 actor after initial spawn or hard reset."""
        # A surface view created from an Articulation's long-lived simulation
        # view can retain the removed actor's GPU buffer after a GUI/Fabric
        # recook.  Give E01 its own fresh SimulationView every time instead.
        import omni.physics.tensors as physics_tensors

        surface_backend = (
            "warp"
            if bool(getattr(self, "_isaaclab_quaternions_are_xyzw", False))
            else "torch"
        )
        physics_sim_view = physics_tensors.create_simulation_view(
            surface_backend
        )
        physics_sim_view.set_subspace_roots("/")
        view = self._create_surface_tensor_view(physics_sim_view)
        view_count = int(view.count)
        view_valid = bool(view.check()) if view_count else False
        if view_count != 1:
            raise RuntimeError(
                f"{self._TASK_ID} surface-deformable tensor view is invalid: "
                f"count={view_count}, check={view_valid}, "
                f"root={self._SURFACE_ROOT_PATH}"
            )
        try:
            positions = view.get_simulation_nodal_positions()
            velocities = view.get_simulation_nodal_velocities()
            nodes_np = self._surface_tensor_body(
                positions, label="nodal_position"
            )
            velocities_np = self._surface_tensor_body(
                velocities, label="nodal_velocity"
            )
        except Exception as exc:
            raise RuntimeError(
                f"{self._TASK_ID} surface tensor exists but PhysX did not initialize its "
                f"nodal state (check={view_valid})"
            ) from exc
        expected_node_count = (
            None if initialize_baseline else self._e_expected_node_count
        )
        issue, diagnostics = self._basic_nodal_state_issue(
            nodes_np,
            velocities_np,
            expected_node_count=expected_node_count,
        )
        if issue is not None:
            raise RuntimeError(
                f"{self._TASK_ID} surface tensor initialized with invalid state: "
                f"reason={issue}, diagnostics={diagnostics}"
            )
        simulation_node_count = int(nodes_np.shape[0])
        simulation_triangle_count = int(
            view.max_simulation_elements_per_body
        )
        if simulation_node_count != self._e_surface_sim_node_count_target:
            raise RuntimeError(
                f"{self._TASK_ID} native PhysX node count changed from selected preset: "
                f"nodes={simulation_node_count}, "
                f"expected={self._e_surface_sim_node_count_target}"
            )
        if simulation_triangle_count != self._e_surface_sim_triangle_count_target:
            raise RuntimeError(
                f"{self._TASK_ID} native PhysX triangle count changed from selected preset: "
                f"triangles={simulation_triangle_count}, "
                f"expected={self._e_surface_sim_triangle_count_target}"
            )
        if (
            not initialize_baseline
            and self._e_simulation_triangle_count
            and simulation_triangle_count != self._e_simulation_triangle_count
        ):
            raise RuntimeError(
                f"{self._TASK_ID} native PhysX topology changed after reset: "
                f"triangles={simulation_triangle_count}, "
                f"expected={self._e_simulation_triangle_count}"
            )
        self._e_surface_sim_view = physics_sim_view
        self._e_garment_view = view
        self._e_garment_indices = self._surface_indices_like(
            positions,
            int(view.count),
        )
        self._e_nodes_np = nodes_np.copy()
        self._e_velocities_np = velocities_np.copy()
        self._e_expected_node_count = simulation_node_count
        self._e_simulation_triangle_count = simulation_triangle_count
        self._e_reset_nodes_np = self._e_nodes_np.copy()
        self._clear_garment_failure_state()
        if initialize_baseline:
            self._e_base_nodes_np = self._e_nodes_np.copy()
            self._configure_garment_tracking(self._e_base_nodes_np)
        print(
            f"[HandsOn/IsaacLab] {self._TASK_ID} surface tensor ready: "
            f"native_nodes={simulation_node_count}, "
            f"native_triangles={simulation_triangle_count}, "
            f"target_triangles={self._e_surface_sim_triangle_count_target}, "
            f"visual={self._e_surface_visual_vertex_count_target}v/"
            f"{self._e_surface_visual_triangle_count_target}t",
            flush=True,
        )

    def _configure_garment_tracking(self, nodes: np.ndarray) -> None:
        nodes = np.asarray(nodes, dtype=np.float64)
        issue, diagnostics = self._basic_nodal_state_issue(
            nodes,
            np.zeros_like(nodes),
            expected_node_count=None,
        )
        if issue is not None:
            raise ValueError(
                "Cannot configure E01 garment tracking from invalid nodes: "
                f"reason={issue}, diagnostics={diagnostics}"
            )
        centered_xy = nodes[:, :2] - np.mean(nodes[:, :2], axis=0)
        covariance = centered_xy.T @ centered_xy / max(1, nodes.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        self._e_fold_axis = axis.astype(np.float64)
        projection = centered_xy @ self._e_fold_axis
        median = float(np.median(projection))
        low = np.flatnonzero(projection <= median).astype(np.int64)
        high = np.flatnonzero(projection > median).astype(np.int64)
        if low.size == 0 or high.size == 0:
            midpoint = max(1, nodes.shape[0] // 2)
            low = np.arange(midpoint, dtype=np.int64)
            high = np.arange(midpoint, nodes.shape[0], dtype=np.int64)
        self._e_fold_masks = (low, high)

        perpendicular = np.asarray(
            [-self._e_fold_axis[1], self._e_fold_axis[0]], dtype=np.float64
        )
        directions = (
            self._e_fold_axis,
            -self._e_fold_axis,
            perpendicular,
            -perpendicular,
            self._e_fold_axis + perpendicular,
            self._e_fold_axis - perpendicular,
            -self._e_fold_axis + perpendicular,
            -self._e_fold_axis - perpendicular,
        )
        selected: list[int] = []
        for direction in directions:
            index = int(np.argmax(centered_xy @ direction))
            if index not in selected:
                selected.append(index)
        radial_order = np.argsort(np.linalg.norm(centered_xy, axis=1))[::-1]
        for index in radial_order:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == 8:
                break
        while len(selected) < 8:
            selected.append(selected[-1] if selected else 0)
        self._e_keypoint_indices = np.asarray(selected[:8], dtype=np.int64)

        low_center = np.mean(nodes[low, :2], axis=0)
        high_center = np.mean(nodes[high, :2], axis=0)
        self._e_baseline_half_distance = max(
            1e-6, float(np.linalg.norm(low_center - high_center))
        )
        self._e_baseline_area = max(1e-6, self._oriented_xy_area(nodes))

    @staticmethod
    def _oriented_xy_area(nodes: np.ndarray) -> float:
        nodes = np.asarray(nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1:] != (3,) or nodes.shape[0] == 0:
            raise ValueError(
                "E01 garment area requires a non-empty (node, xyz) array"
            )
        if not bool(np.all(np.isfinite(nodes))):
            raise ValueError("E01 garment area requires finite node positions")
        xy = nodes[:, :2]
        centered = xy - np.mean(xy, axis=0)
        covariance = centered.T @ centered / max(1, xy.shape[0] - 1)
        _values, vectors = np.linalg.eigh(covariance)
        projected = centered @ vectors
        extent = np.ptp(projected, axis=0)
        return float(max(1e-9, extent[0] * extent[1]))

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        if self.randomize:
            self.delta_h[:] = self._random.uniform(0.0, 0.025)
            self._e_initial_state = np.asarray(
                [
                    self._random.uniform(-0.04, 0.04),
                    self._random.uniform(-0.07, 0.07),
                    math.radians(self._random.uniform(-18.0, 18.0)),
                    self._random.uniform(0.0, 0.018),
                    float(self._random.randint(0, 3)),
                    float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
        else:
            self.delta_h[:] = 0.0
            self._e_initial_state = np.zeros(6, dtype=np.float64)
        self._e_last_metrics = {}
        self._clear_garment_failure_state()
        self._f01_grasp_assist_ever_applied = False
        for side in ("right", "left"):
            self._clear_f01_grasp_assist(side, reason="reset")

    def _garment_reset_nodes(self) -> np.ndarray:
        nodes = self._e_base_nodes_np.copy()
        center = np.mean(nodes, axis=0)
        dx, dy, yaw, wrinkle, variant, delta_h = self._e_initial_state
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        nodes[:, :2] = (nodes[:, :2] - center[:2]) @ rotation.T + center[:2]

        if wrinkle > 0.0:
            centered = self._e_base_nodes_np[:, :2] - np.mean(
                self._e_base_nodes_np[:, :2], axis=0
            )
            along = centered @ self._e_fold_axis
            across_axis = np.asarray(
                [-self._e_fold_axis[1], self._e_fold_axis[0]], dtype=np.float64
            )
            across = centered @ across_axis
            along /= max(1e-6, float(np.max(np.abs(along))))
            across /= max(1e-6, float(np.max(np.abs(across))))
            variant_index = int(round(float(variant))) % 3
            if variant_index == 0:
                profile = np.sin(math.pi * along) * np.cos(math.pi * across)
            elif variant_index == 1:
                profile = np.sin(2.0 * math.pi * across + 0.5 * math.pi * along)
            else:
                profile = np.cos(1.5 * math.pi * along - math.pi * across)
            nodes[:, 2] += float(wrinkle) * (0.5 + 0.5 * profile)

        nodes[:, 0] += float(dx)
        nodes[:, 1] += float(dy)
        nodes[:, 2] += float(delta_h)
        return nodes

    def _write_table_height(self) -> None:
        if not self._task_reset_body_states:
            return
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
                    randomized, dtype=root_pose.dtype, device=root_pose.device
                ),
                indices,
            )
            view.set_velocities(self._torch.zeros_like(root_velocity), indices)
            return

    def _reset_scene_state(self) -> None:
        interface_only = not self._robots
        super()._reset_scene_state()
        if interface_only:
            # The shared ABC reset intentionally returns early when AppLauncher
            # is disabled.  Still apply E01's deterministic nodal transform so
            # schema/replay tests exercise the same initial-state contract.
            self._apply_task_pose_randomization()
            return
        self._settle_garment_after_reset()

    def _settle_garment_after_reset(self) -> None:
        """Advance until nodal speed stays stable after a minimum dwell."""
        if (
            not self._robots
            or self._sim is None
            or self._e_garment_view is None
        ):
            return
        physics_dt = float(getattr(self, "_sim_dt", self._physics_dt))
        step_budget = self._reset_settle_step_budget(physics_dt)
        minimum_steps = self._reset_settle_minimum_steps(physics_dt)
        stable_steps_required = self._reset_settle_stable_steps_required(
            physics_dt
        )
        stable_steps = 0
        stable_window_start: np.ndarray | None = None
        for step_index in range(step_budget):
            self._step_isaaclab()
            try:
                nodes = self._surface_tensor_body(
                    self._e_garment_view.get_simulation_nodal_positions(),
                    label="nodal_position",
                )
                velocities = self._surface_tensor_body(
                    self._e_garment_view.get_simulation_nodal_velocities(),
                    label="nodal_velocity",
                )
            except Exception as exc:
                self._reject_garment_readback(
                    "reset_settle_surface_tensor_readback_error",
                    {"error": str(exc), "settle_step": step_index + 1},
                )
                return
            issue, diagnostics = self._dynamic_nodal_state_issue(
                nodes,
                velocities,
                # Reset writes an undeformed, zero-velocity surface and then
                # lets PhysX resolve gravity, strain, and table contact. The
                # first settle frame can therefore move more than the normal
                # control-frame budget even when its speed is within the
                # configured PhysX cap. This is expected reset relaxation;
                # speed and the stability window still guard the initial state.
                enforce_control_frame_displacement=step_index > 0,
            )
            diagnostics["settle_step"] = step_index + 1
            if issue is not None:
                self._reject_garment_readback(issue, diagnostics)
                return
            self._accept_garment_readback(nodes, velocities, diagnostics)
            rms_speed, max_speed = self._nodal_speed_summary(velocities)
            self._e_reset_settle_steps = step_index + 1
            self._e_reset_settle_seconds = (step_index + 1) * physics_dt
            self._e_reset_settle_rms_speed = rms_speed
            self._e_reset_settle_max_speed = max_speed
            if self._nodal_speed_is_stable(rms_speed, max_speed):
                if stable_window_start is None:
                    stable_window_start = nodes.copy()
                    stable_steps = 1
                    window_drift = 0.0
                else:
                    window_drift = float(
                        np.max(
                            np.linalg.norm(
                                nodes - stable_window_start,
                                axis=1,
                            )
                        )
                    )
                    if window_drift > _E_RESET_SETTLE_MAX_WINDOW_DRIFT:
                        # Low instantaneous speed is not enough: a cloth can
                        # creep coherently while every individual node stays
                        # below the speed threshold.  Restart the quiet window
                        # from this frame and require another complete dwell.
                        stable_window_start = nodes.copy()
                        stable_steps = 1
                        window_drift = 0.0
                    else:
                        stable_steps += 1
            else:
                stable_steps = 0
                stable_window_start = None
                window_drift = 0.0
            self._e_reset_settle_window_drift = window_drift
            self._e_reset_settle_stable_steps = stable_steps
            if (
                step_index + 1 >= minimum_steps
                and stable_steps >= stable_steps_required
            ):
                self._e_reset_settle_converged = True
                return

        if (
            self._e_reset_settle_steps >= minimum_steps
            and stable_steps > 0
            and self._nodal_speed_is_stable(
                self._e_reset_settle_rms_speed,
                self._e_reset_settle_max_speed,
            )
            and self._e_reset_settle_window_drift
            <= _E_RESET_SETTLE_MAX_WINDOW_DRIFT
        ):
            self._e_reset_settle_converged = True
            print(
                f"[HandsOn/IsaacLab] {self._TASK_ID} reset settle reached "
                "speed/drift thresholds at the step budget boundary: "
                f"stable_steps={stable_steps}, "
                f"required={stable_steps_required}",
                flush=True,
            )
            return

        self._e_reset_settle_timed_out = True
        self._e_failure_diagnostics = {
            "settle_steps": int(self._e_reset_settle_steps),
            "settle_seconds": float(self._e_reset_settle_seconds),
            "rms_nodal_speed": float(self._e_reset_settle_rms_speed),
            "max_nodal_speed": float(self._e_reset_settle_max_speed),
            "rms_speed_threshold": float(_E_STABLE_RMS_SPEED),
            "max_speed_threshold": float(_E_STABLE_MAX_SPEED),
            "stable_window_drift": float(
                self._e_reset_settle_window_drift
            ),
            "stable_window_drift_threshold": float(
                _E_RESET_SETTLE_MAX_WINDOW_DRIFT
            ),
            "minimum_settle_steps": int(minimum_steps),
            "stable_steps": int(stable_steps),
            "stable_steps_required": int(stable_steps_required),
        }
        if self._e_relaxed_readback_guards:
            print(
                f"[HandsOn/IsaacLab] {self._TASK_ID} reset settle timed out "
                "under relaxed diagnostic guards: "
                f"diagnostics={self._e_failure_diagnostics}",
                flush=True,
            )
            return

        self._e_failure_reason = "reset_nodal_state_did_not_settle"
        print(
            f"[HandsOn/IsaacLab] {self._TASK_ID} reset settle timed out: "
            f"diagnostics={self._e_failure_diagnostics}",
            flush=True,
        )

    def _apply_task_pose_randomization(self) -> None:
        self._write_table_height()
        reset_nodes = self._garment_reset_nodes()
        issue, diagnostics = self._basic_nodal_state_issue(
            reset_nodes,
            np.zeros_like(reset_nodes),
            expected_node_count=self._e_expected_node_count,
        )
        if issue is not None:
            raise RuntimeError(
                "E01 generated an invalid garment reset state: "
                f"reason={issue}, diagnostics={diagnostics}"
            )
        self._e_reset_nodes_np = reset_nodes.copy()
        self._e_nodes_np = reset_nodes.copy()
        self._e_velocities_np = np.zeros_like(reset_nodes)
        self._clear_garment_failure_state()
        if self._e_garment_view is not None and self._e_garment_indices is not None:
            positions = self._e_garment_view.get_simulation_nodal_positions()
            position_tensor = self._surface_array_like(
                positions,
                reset_nodes,
            )
            self._e_garment_view.set_simulation_nodal_positions(
                position_tensor, self._e_garment_indices
            )
            self._e_garment_view.set_simulation_nodal_velocities(
                self._surface_zeros_like(position_tensor),
                self._e_garment_indices,
            )
            try:
                readback = self._surface_tensor_body(
                    self._e_garment_view.get_simulation_nodal_positions(),
                    label="nodal_position",
                )
                velocity_readback = self._surface_tensor_body(
                    self._e_garment_view.get_simulation_nodal_velocities(),
                    label="nodal_velocity",
                )
            except Exception as exc:
                self._reject_garment_readback(
                    "reset_surface_tensor_readback_error",
                    {"error": str(exc)},
                )
                raise RuntimeError(
                    "E01 could not verify the PhysX garment reset write"
                ) from exc
            issue, diagnostics = self._basic_nodal_state_issue(
                readback,
                velocity_readback,
                expected_node_count=self._e_expected_node_count,
            )
            if issue is not None:
                self._reject_garment_readback(issue, diagnostics)
                raise RuntimeError(
                    "E01 tensor reset produced invalid nodal state: "
                    f"reason={issue}, diagnostics={diagnostics}"
                )
            max_error = float(
                np.max(np.linalg.norm(readback - reset_nodes, axis=1))
            )
            if max_error > 1e-4:
                self._reject_garment_readback(
                    "reset_node_position_mismatch",
                    {"max_node_error": max_error},
                )
                raise RuntimeError(
                    "E01 tensor reset write did not reach PhysX: "
                    f"max_node_error={max_error:.6f}m"
                )
            if self._sim is not None:
                self._sim.forward()
                try:
                    readback = self._surface_tensor_body(
                        self._e_garment_view.get_simulation_nodal_positions(),
                        label="nodal_position",
                    )
                    velocity_readback = self._surface_tensor_body(
                        self._e_garment_view.get_simulation_nodal_velocities(),
                        label="nodal_velocity",
                    )
                except Exception as exc:
                    self._reject_garment_readback(
                        "reset_surface_forward_readback_error",
                        {"error": str(exc)},
                    )
                    raise RuntimeError(
                        "E01 could not verify the PhysX garment reset forward"
                    ) from exc
                issue, diagnostics = self._basic_nodal_state_issue(
                    readback,
                    velocity_readback,
                    expected_node_count=self._e_expected_node_count,
                )
                if issue is not None:
                    self._reject_garment_readback(issue, diagnostics)
                    raise RuntimeError(
                        "E01 tensor reset forward produced invalid nodal state: "
                        f"reason={issue}, diagnostics={diagnostics}"
                    )
                max_error = float(
                    np.max(np.linalg.norm(readback - reset_nodes, axis=1))
                )
                if max_error > 1e-4:
                    self._reject_garment_readback(
                        "reset_forward_node_position_mismatch",
                        {"max_node_error": max_error},
                    )
                    raise RuntimeError(
                        "E01 tensor reset forward did not retain PhysX nodes: "
                        f"max_node_error={max_error:.6f}m"
                    )
                self._e_reset_nodes_np = readback.copy()
                self._accept_garment_readback(
                    readback,
                    velocity_readback,
                    {
                        "max_control_node_displacement": 0.0,
                        "max_nodal_speed": float(
                            np.max(np.linalg.norm(velocity_readback, axis=1))
                        ),
                    },
                )
        print(
            f"[HandsOn/IsaacLab] Reset {self._TASK_ID} surface through its dedicated "
            "PhysX tensor view: "
            f"table_delta_h={self.delta_h[0]:.4f}m, "
            f"xy=({self._e_initial_state[0]:+.4f},"
            f"{self._e_initial_state[1]:+.4f})m, "
            f"yaw={math.degrees(self._e_initial_state[2]):+.1f}deg, "
            f"wrinkle={self._e_initial_state[3]:.4f}m, "
            f"variant={int(self._e_initial_state[4])}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Live observations and success criteria
    # ------------------------------------------------------------------

    def _clear_f01_grasp_assist(self, side: str, *, reason: str) -> None:
        self._f01_grasp_assist[side] = None
        self._f01_grasp_assist_metrics[side] = {
            "active": False,
            "pair": "",
            "node_count": 0,
            "reason": str(reason),
        }

    def _f01_wuji_fingertip_positions(
        self, side: str
    ) -> dict[str, np.ndarray]:
        robot = self._robots.get(side)
        if robot is None:
            return {}
        positions: dict[str, np.ndarray] = {}
        for finger, local_offset in enumerate(
            _F01_WUJI_TIP_LOCAL_OFFSETS[side],
            start=1,
        ):
            body_name = f"{side}_finger{finger}_link4"
            cache_key = f"{side}:{body_name}"
            body_index = self._f01_grasp_body_indices.get(cache_key)
            if body_index is None:
                try:
                    body_index = next(
                        index
                        for index, name in enumerate(robot.body_names)
                        if body_name in str(name)
                    )
                except (StopIteration, TypeError):
                    continue
                self._f01_grasp_body_indices[cache_key] = int(body_index)
            try:
                pose = robot.data.body_pose_w[0, int(body_index), :7]
                if hasattr(pose, "detach"):
                    pose = pose.detach().cpu().numpy()
                pose_np = np.asarray(pose, dtype=np.float64).reshape(7)
            except Exception:
                continue
            if not bool(np.all(np.isfinite(pose_np))):
                continue
            matrix = _pose_vector_to_matrix(pose_np)
            position = (
                matrix[:3, 3]
                + matrix[:3, :3]
                @ np.asarray(local_offset, dtype=np.float64)
            )
            if bool(np.all(np.isfinite(position))):
                positions[f"finger{finger}"] = position
        return positions

    @staticmethod
    def _f01_grasp_candidate(
        nodes: np.ndarray,
        tips: Mapping[str, np.ndarray],
    ) -> dict[str, Any] | None:
        thumb = tips.get("finger1")
        if thumb is None:
            return None
        nodes = np.asarray(nodes, dtype=np.float64)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for opposing_name in ("finger2", "finger3", "finger4", "finger5"):
            opposing = tips.get(opposing_name)
            if opposing is None:
                continue
            finger_distance = float(np.linalg.norm(thumb - opposing))
            if not (
                _F01_GRASP_ATTACH_MIN_FINGER_DISTANCE
                <= finger_distance
                <= _F01_GRASP_ATTACH_MAX_FINGER_DISTANCE
            ):
                continue
            pinch_center = 0.5 * (thumb + opposing)
            node_distances = np.linalg.norm(nodes - pinch_center, axis=1)
            nearby = np.flatnonzero(
                node_distances <= _F01_GRASP_ATTACH_NODE_RADIUS
            )
            if nearby.size == 0:
                continue
            order = nearby[np.argsort(node_distances[nearby])]
            selected = order[:_F01_GRASP_MAX_NODES_PER_HAND].astype(np.int64)
            score = float(node_distances[selected[0]]) + 0.1 * finger_distance
            candidates.append(
                (
                    score,
                    {
                        "pair": ("finger1", opposing_name),
                        "finger_distance": finger_distance,
                        "pinch_center": pinch_center,
                        "indices": selected,
                        "offsets": nodes[selected] - pinch_center,
                    },
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _f01_limit_grasp_velocity(velocity: np.ndarray) -> np.ndarray:
        result = np.asarray(velocity, dtype=np.float64).copy()
        speed = np.linalg.norm(result, axis=1)
        limited = speed > _F01_GRASP_MAX_NODE_SPEED
        if np.any(limited):
            result[limited] *= (
                _F01_GRASP_MAX_NODE_SPEED
                / np.maximum(speed[limited], 1e-12)
            )[:, None]
        return result

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        self._apply_f01_grasp_assist()

    def _apply_f01_grasp_assist(self) -> None:
        if (
            not self._is_f01_diagnostic()
            or not self._f01_grasp_assist_enabled
            or self._f01_grasp_assist_disabled
            or self._e_garment_view is None
            or self._e_garment_indices is None
            or self._e_readback_failed
        ):
            return

        tip_positions = {
            side: self._f01_wuji_fingertip_positions(side)
            for side in ("right", "left")
        }
        if not any(tip_positions.values()):
            return
        try:
            position_reference = (
                self._e_garment_view.get_simulation_nodal_positions()
            )
            velocity_reference = (
                self._e_garment_view.get_simulation_nodal_velocities()
            )
            nodes = self._surface_tensor_body(
                position_reference,
                label="f01_grasp_nodal_position",
            )
            velocities = self._surface_tensor_body(
                velocity_reference,
                label="f01_grasp_nodal_velocity",
            )
        except Exception as exc:
            self._f01_grasp_assist_disabled = True
            self._add_task_state_warning(
                f"F01 grasp assist disabled after tensor read failure: {exc}"
            )
            return

        dt = float(getattr(self, "_sim_dt", 1.0 / E01_PHYSICS_HZ))
        if not math.isfinite(dt) or dt <= 0.0:
            return
        velocity_changed = False
        claimed_nodes: set[int] = set()
        for side in ("right", "left"):
            tips = tip_positions[side]
            state = self._f01_grasp_assist[side]
            if state is None:
                candidate = self._f01_grasp_candidate(nodes, tips)
                if candidate is None:
                    self._f01_grasp_assist_metrics[side] = {
                        "active": False,
                        "pair": "",
                        "node_count": 0,
                        "reason": "waiting_for_cloth_pinch",
                    }
                    continue
                candidate_indices = np.asarray(
                    candidate["indices"], dtype=np.int64
                )
                available = np.asarray(
                    [
                        int(index) not in claimed_nodes
                        for index in candidate_indices
                    ],
                    dtype=bool,
                )
                if not np.any(available):
                    self._f01_grasp_assist_metrics[side] = {
                        "active": False,
                        "pair": "",
                        "node_count": 0,
                        "reason": "cloth_nodes_claimed_by_other_hand",
                    }
                    continue
                state = {
                    "pair": tuple(candidate["pair"]),
                    "indices": candidate_indices[available],
                    "offsets": np.asarray(
                        candidate["offsets"], dtype=np.float64
                    )[available],
                }
                self._f01_grasp_assist[side] = state

            pair = tuple(state["pair"])
            if len(pair) != 2 or any(name not in tips for name in pair):
                self._clear_f01_grasp_assist(side, reason="fingertip_missing")
                continue
            first, second = tips[pair[0]], tips[pair[1]]
            finger_distance = float(np.linalg.norm(first - second))
            if finger_distance > _F01_GRASP_RELEASE_FINGER_DISTANCE:
                self._clear_f01_grasp_assist(side, reason="fingers_open")
                continue

            pinch_center = 0.5 * (first + second)
            indices = np.asarray(state["indices"], dtype=np.int64)
            offsets = np.asarray(state["offsets"], dtype=np.float64)
            available = np.asarray(
                [int(index) not in claimed_nodes for index in indices],
                dtype=bool,
            )
            if not np.any(available):
                self._clear_f01_grasp_assist(
                    side, reason="cloth_nodes_claimed_by_other_hand"
                )
                continue
            if not np.all(available):
                indices = indices[available]
                offsets = offsets[available]
                state["indices"] = indices
                state["offsets"] = offsets
            claimed_nodes.update(int(index) for index in indices)
            target = pinch_center + offsets
            error = target - nodes[indices]
            max_error = float(np.max(np.linalg.norm(error, axis=1)))
            if max_error > _F01_GRASP_MAX_TARGET_ERROR:
                self._clear_f01_grasp_assist(
                    side, reason="target_moved_too_far"
                )
                continue

            desired_velocity = self._f01_limit_grasp_velocity(error / dt)
            velocities[indices] = (
                0.15 * velocities[indices] + 0.85 * desired_velocity
            )
            velocity_changed = velocity_changed or bool(
                np.any(np.abs(desired_velocity) > 1e-8)
            )
            self._f01_grasp_assist_metrics[side] = {
                "active": True,
                "pair": f"{pair[0]}/{pair[1]}",
                "node_count": int(indices.size),
                "finger_distance": finger_distance,
                "max_target_error": max_error,
                "reason": "active",
            }

        if not velocity_changed:
            return
        issue, diagnostics = self._dynamic_nodal_state_issue(nodes, velocities)
        if issue is not None:
            self._reject_garment_readback(issue, diagnostics)
            return
        try:
            self._e_garment_view.set_simulation_nodal_velocities(
                self._surface_array_like(velocity_reference, velocities),
                self._e_garment_indices,
            )
        except Exception as exc:
            self._f01_grasp_assist_disabled = True
            self._add_task_state_warning(
                f"F01 grasp assist disabled after velocity write failure: {exc}"
            )
            return
        self._f01_grasp_assist_ever_applied = True
        self._record_trajectory_state_write(
            "_apply_f01_grasp_assist",
            subject=self._SURFACE_LOGICAL_NAME,
        )
        self._accept_garment_readback(nodes, velocities, diagnostics)

    def _refresh_garment_state(self) -> None:
        if self._e_garment_view is not None and not self._e_readback_failed:
            try:
                nodes = self._surface_tensor_body(
                    self._e_garment_view.get_simulation_nodal_positions(),
                    label="nodal_position",
                )
                velocities = self._surface_tensor_body(
                    self._e_garment_view.get_simulation_nodal_velocities(),
                    label="nodal_velocity",
                )
            except Exception as exc:
                self._reject_garment_readback(
                    "surface_tensor_readback_error",
                    {"error": str(exc)},
                )
            else:
                issue, diagnostics = self._dynamic_nodal_state_issue(
                    nodes,
                    velocities,
                )
                if issue is not None:
                    self._reject_garment_readback(issue, diagnostics)
                else:
                    self._accept_garment_readback(
                        nodes,
                        velocities,
                        diagnostics,
                    )
        else:
            self._ensure_metric_state_is_safe()
        self._e_last_metrics = self._garment_metrics()

    def _refresh_object_poses(self) -> None:
        super()._refresh_object_poses()
        self._refresh_garment_state()
        self._poses[self._SURFACE_LOGICAL_NAME] = self._garment_pose()

    def _garment_pose(self) -> np.ndarray:
        self._ensure_metric_state_is_safe()
        nodes = self._e_nodes_np
        centroid = np.mean(nodes, axis=0)
        centered = nodes[:, :2] - centroid[:2]
        covariance = centered.T @ centered / max(1, nodes.shape[0] - 1)
        _values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, 1]
        if axis[0] < 0.0:
            axis = -axis
        yaw = math.atan2(float(axis[1]), float(axis[0]))
        return np.asarray(
            [
                *centroid,
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            ],
            dtype=np.float64,
        )

    def _minimum_hand_distance(self, nodes: np.ndarray) -> float:
        stride = max(1, nodes.shape[0] // 512)
        samples = nodes[::stride]
        minimum = math.inf
        point_count = 0
        for side in ("right", "left"):
            for point in self._a06_hand_points(side):
                point_count += 1
                minimum = min(
                    minimum,
                    float(np.min(np.linalg.norm(samples - point, axis=1))),
                )
        return minimum if point_count else math.nan

    def _garment_metrics(self) -> dict[str, Any]:
        self._ensure_metric_state_is_safe()
        nodes = self._e_nodes_np
        velocities = self._e_velocities_np
        lower, upper = self._e_fold_masks
        lower_center = np.mean(nodes[lower, :2], axis=0)
        upper_center = np.mean(nodes[upper, :2], axis=0)
        half_distance = float(np.linalg.norm(lower_center - upper_center))
        fold_ratio = half_distance / self._e_baseline_half_distance
        area_ratio = self._oriented_xy_area(nodes) / self._e_baseline_area
        speed = np.linalg.norm(velocities, axis=1)
        rms_speed = float(np.sqrt(np.mean(np.square(speed))))
        max_speed = float(np.max(speed))
        bounds_min = np.min(nodes, axis=0)
        bounds_max = np.max(nodes, axis=0)
        centroid = np.mean(nodes, axis=0)
        hand_distance = self._minimum_hand_distance(nodes)
        table_top = _E_TABLE_TOP_Z + float(self.delta_h[0])
        compact = bool(
            fold_ratio <= _E_FOLD_RATIO_THRESHOLD
            and area_ratio <= _E_AREA_RATIO_THRESHOLD
        )
        partial_overlap = bool(
            fold_ratio <= _F01_PARTIAL_FOLD_RATIO_THRESHOLD
            or area_ratio <= _F01_PARTIAL_AREA_RATIO_THRESHOLD
        )
        on_table = bool(
            bounds_min[2] >= table_top - 0.025
            and bounds_max[2] <= table_top + 0.24
        )
        in_workspace = bool(
            bounds_min[0] >= _E_TABLE_X_BOUNDS[0] - 0.02
            and bounds_max[0] <= _E_TABLE_X_BOUNDS[1] + 0.02
            and bounds_min[1] >= _E_TABLE_Y_BOUNDS[0] - 0.02
            and bounds_max[1] <= _E_TABLE_Y_BOUNDS[1] + 0.02
        )
        right_target_center = np.asarray(
            [
                0.5
                * (_F01_RIGHT_TARGET_X_BOUNDS[0] + _F01_RIGHT_TARGET_X_BOUNDS[1]),
                0.5
                * (_F01_RIGHT_TARGET_Y_BOUNDS[0] + _F01_RIGHT_TARGET_Y_BOUNDS[1]),
            ],
            dtype=np.float64,
        )
        in_right_target = bool(
            _F01_RIGHT_TARGET_X_BOUNDS[0]
            <= float(centroid[0])
            <= _F01_RIGHT_TARGET_X_BOUNDS[1]
            and _F01_RIGHT_TARGET_Y_BOUNDS[0]
            <= float(centroid[1])
            <= _F01_RIGHT_TARGET_Y_BOUNDS[1]
        )
        hands_released = bool(
            math.isfinite(hand_distance)
            and hand_distance >= _E_HAND_RELEASE_DISTANCE
        )
        stable = self._nodal_speed_is_stable(rms_speed, max_speed)
        collection_eligible = bool(
            not self._is_f01_diagnostic()
            or not self._f01_grasp_assist_ever_applied
        )
        displacement = np.linalg.norm(nodes - self._e_reset_nodes_np, axis=1)
        return {
            "fold_ratio": float(fold_ratio),
            "area_ratio": float(area_ratio),
            "half_centroid_distance": half_distance,
            "z_span": float(bounds_max[2] - bounds_min[2]),
            "hand_distance": float(hand_distance),
            "rms_nodal_speed": rms_speed,
            "max_nodal_speed": max_speed,
            "max_reset_displacement": float(np.max(displacement)),
            "compact_fold": compact,
            "partial_overlap": partial_overlap,
            "right_target": in_right_target,
            "right_target_x_min": float(_F01_RIGHT_TARGET_X_BOUNDS[0]),
            "right_target_x_max": float(_F01_RIGHT_TARGET_X_BOUNDS[1]),
            "right_target_y_min": float(_F01_RIGHT_TARGET_Y_BOUNDS[0]),
            "right_target_y_max": float(_F01_RIGHT_TARGET_Y_BOUNDS[1]),
            "right_target_distance": float(
                np.linalg.norm(centroid[:2] - right_target_center)
            ),
            "on_table": on_table,
            "in_workspace": in_workspace,
            "hands_released": hands_released,
            "stable": stable,
            "collection_eligible": collection_eligible,
            "centroid": centroid,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "success_now": bool(
                not self._e_readback_failed
                and (
                    partial_overlap
                    if self._is_f01_diagnostic()
                    else compact
                )
                and (
                    in_right_target
                    if self._is_f01_diagnostic()
                    else True
                )
                and on_table
                and in_workspace
                and hands_released
                and stable
                and collection_eligible
            ),
        }

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "garment_pose": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "garment_keypoints": spaces.Box(
                -np.inf, np.inf, shape=(24,), dtype=np.float64
            ),
            "garment_aabb": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "fold_metrics": spaces.Box(
                -np.inf, np.inf, shape=(12,), dtype=np.float64
            ),
            "garment_initial_state": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "table_delta_height": spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float64
            ),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces),
            # Complete simulator readback for recording/replay only.  Keep it
            # outside ``state`` so E01's established 102D policy contract does
            # not change when all 20 joints per Wuji hand are persisted.
            "e01_hand_qpos": spaces.Box(
                -np.inf,
                np.inf,
                shape=(40,),
                dtype=np.float64,
            ),
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

    def _e01_recording_hand_qpos(self) -> np.ndarray:
        """Return the complete right20+left20 hand state for recording.

        The detached PhysX runtime overrides this hook with joint readback.
        This base implementation retains the complete command state for the
        no-Isaac interface and legacy full-robot runtime.
        """

        hand_qpos = np.concatenate(
            [
                np.asarray(self._right_hand, dtype=np.float64).reshape(-1),
                np.asarray(self._left_hand, dtype=np.float64).reshape(-1),
            ]
        )
        if hand_qpos.shape != (40,):
            raise RuntimeError(
                "E01 recording hand state must use right20+left20 layout; "
                f"got {hand_qpos.shape}."
            )
        return hand_qpos.astype(np.float64, copy=True)

    def get_recording_schema(self) -> dict[str, Any]:
        """Declare action-aligned full hand state without changing policy state."""

        hand_field = {
            "shape": [40],
            "dtype": "float64",
            "layout": "right_wuji_20_then_left_wuji_20",
            "source": "simulator_joint_position_readback",
        }
        base_schema = super().get_recording_schema()
        if base_schema is None:
            return {
                "name": "handson.e01_actual_hand_state",
                "version": 1,
                "fields": {"e01_hand_qpos": hand_field},
            }

        schema = dict(base_schema)
        schema["name"] = "handson.e01_with_physx_ground_truth"
        schema["fields"] = {
            **dict(base_schema.get("fields", {})),
            "e01_hand_qpos": hand_field,
        }
        return schema

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._e_last_metrics or self._garment_metrics()
        tcp, gripper = self._robot_state()
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "garment_pose": self._garment_pose(),
            "garment_keypoints": self._e_nodes_np[
                self._e_keypoint_indices
            ].reshape(-1).astype(np.float64, copy=True),
            "garment_aabb": np.concatenate(
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
                    metrics["on_table"],
                    metrics["in_workspace"],
                    metrics["hands_released"],
                    self._success_counter / max(1, self._success_steps_required()),
                ],
                dtype=np.float64,
            ),
            "garment_initial_state": self._e_initial_state.astype(
                np.float64, copy=True
            ),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        observation: dict[str, Any] = {
            "state": state,
            "e01_hand_qpos": self._e01_recording_hand_qpos(),
        }
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        if self._e_failure_reason is not None:
            return False
        self._e_last_metrics = self._garment_metrics()
        return bool(self._e_last_metrics["success_now"])

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        if str(row.get("key") or "") == "success_hold":
            return "Hold all conditions stable"
        return str(row.get("label") or row.get("key") or "Condition")

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._e_last_metrics or self._garment_metrics()
        if self._is_f01_diagnostic():
            rows = [
                self._condition_row(
                    "f01_right_target",
                    "Garment centroid is in the right target zone",
                    metrics["right_target"],
                ),
                self._condition_row(
                    "f01_partial_overlap",
                    "Partial cloth overlap detected",
                    metrics["partial_overlap"],
                ),
                self._condition_row(
                    "f01_table",
                    "Garment remains on the tabletop workspace",
                    metrics["on_table"] and metrics["in_workspace"],
                ),
                self._condition_row(
                    "f01_readback",
                    "Cloth tensor readback is healthy",
                    not self._e_readback_failed,
                ),
                self._condition_row(
                    "f01_release",
                    "Both hands are released from the cloth",
                    metrics["hands_released"],
                ),
                self._condition_row(
                    "f01_stable",
                    "Cloth nodal velocity is stable",
                    metrics["stable"],
                ),
                self._condition_row(
                    "f01_unassisted",
                    "No direct nodal-velocity assist was used",
                    metrics["collection_eligible"],
                ),
            ]
            return self._append_success_hold_row(rows)

        rows = [
            self._condition_row(
                "e01_compact",
                "Garment folded into a compact two-layer footprint",
                metrics["compact_fold"],
            ),
            self._condition_row(
                "e01_table",
                "Garment remains on the tabletop workspace",
                metrics["on_table"] and metrics["in_workspace"],
            ),
            self._condition_row(
                "e01_release",
                "Both hands released at least 65 mm from the garment",
                metrics["hands_released"],
            ),
            self._condition_row(
                "e01_stable",
                "Cloth nodal velocity is stable",
                metrics["stable"],
            ),
        ]
        return self._append_success_hold_row(rows)

    def _compute_failure(self) -> bool:
        if self._e_failure_reason is not None:
            return True
        metrics = self._e_last_metrics or self._garment_metrics()
        center = metrics["centroid"]
        reason: str | None = None
        if metrics["bounds_min"][2] < 0.55:
            reason = "garment_below_safe_height"
        elif center[0] < -0.90 or center[0] > 0.60:
            reason = "garment_centroid_outside_x_safety_bounds"
        elif abs(float(center[1])) > 1.10:
            reason = "garment_centroid_outside_y_safety_bounds"
        if reason is None:
            return False
        self._e_failure_reason = reason
        self._e_failure_diagnostics = {
            "centroid": tuple(float(value) for value in center),
            "bounds_min": tuple(
                float(value) for value in metrics["bounds_min"]
            ),
            "bounds_max": tuple(
                float(value) for value in metrics["bounds_max"]
            ),
        }
        print(
            f"[HandsOn/IsaacLab] {self._TASK_ID} garment task failed: "
            f"reason={reason}; diagnostics={self._e_failure_diagnostics}",
            flush=True,
        )
        return True

    def _task_info(self) -> dict[str, Any]:
        metrics = self._e_last_metrics or self._garment_metrics()
        readback_max_speed = float(
            self._e_failure_diagnostics.get(
                "max_nodal_speed", self._e_last_readback_max_speed
            )
        )
        readback_max_displacement = float(
            self._e_failure_diagnostics.get(
                "max_control_node_displacement",
                self._e_last_control_node_displacement,
            )
        )
        return {
            "task_id": self._TASK_ID,
            "task_name": self.task_name,
            "garment_asset": str(
                self._e_asset_path or self._e_surface_asset_label
            ),
            "garment_node_count": int(self._e_nodes_np.shape[0]),
            "garment_expected_node_count": int(self._e_expected_node_count),
            "garment_simulation_triangle_count": int(
                self._e_simulation_triangle_count
            ),
            "garment_simulation_target_triangle_count": int(
                self._e_surface_sim_triangle_count_target
            ),
            "garment_visual_vertex_count": int(
                self._e_surface_visual_vertex_count_target
            ),
            "garment_visual_triangle_count": int(
                self._e_surface_visual_triangle_count_target
            ),
            "garment_state_valid": not self._e_readback_failed,
            "garment_state_preserved_from_last_valid_frame": bool(
                self._e_readback_failed
            ),
            "failure_reason": self._e_failure_reason,
            "garment_failure_reason": self._e_failure_reason,
            "garment_failure_diagnostics": dict(
                self._e_failure_diagnostics
            ),
            "garment_readback_max_nodal_speed": readback_max_speed,
            "garment_nodal_speed_failure_limit": float(
                self._e_surface_physics_tuning.resolved_max_linear_velocity
                + _E_MAX_NODAL_SPEED_TOLERANCE
            ),
            "garment_max_control_node_displacement": (
                readback_max_displacement
            ),
            "garment_control_node_displacement_failure_limit": float(
                _E_MAX_CONTROL_NODE_DISPLACEMENT
            ),
            "garment_readback_guards_relaxed": bool(
                self._e_relaxed_readback_guards
            ),
            "garment_contact_offset": float(
                self._e_surface_physics_tuning.contact_offset
            ),
            "garment_surface_thickness": float(
                self._e_surface_physics_tuning.surface_thickness
            ),
            "garment_surface_density": float(
                self._e_surface_physics_tuning.surface_density
            ),
            "garment_surface_stretch_stiffness": float(
                self._e_surface_physics_tuning.surface_stretch_stiffness
            ),
            "garment_surface_shear_stiffness": float(
                self._e_surface_physics_tuning.surface_shear_stiffness
            ),
            "garment_surface_bend_stiffness": float(
                self._e_surface_physics_tuning.surface_bend_stiffness
            ),
            "garment_elasticity_damping": float(
                self._e_surface_physics_tuning.elasticity_damping
            ),
            "f01_grasp_assist_enabled": bool(
                self._is_f01_diagnostic()
                and self._f01_grasp_assist_enabled
                and not self._f01_grasp_assist_disabled
            ),
            "f01_grasp_assist_ever_applied": bool(
                self._f01_grasp_assist_ever_applied
            ),
            **{
                f"f01_grasp_assist_{side}_{key}": value
                for side, side_metrics in self._f01_grasp_assist_metrics.items()
                for key, value in side_metrics.items()
            },
            "garment_reset_settle_converged": bool(
                self._e_reset_settle_converged
            ),
            "garment_reset_settle_timed_out": bool(
                self._e_reset_settle_timed_out
            ),
            "garment_reset_settle_steps": int(self._e_reset_settle_steps),
            "garment_reset_settle_seconds": float(
                self._e_reset_settle_seconds
            ),
            "garment_reset_settle_rms_speed": float(
                self._e_reset_settle_rms_speed
            ),
            "garment_reset_settle_max_speed": float(
                self._e_reset_settle_max_speed
            ),
            "garment_reset_settle_window_drift": float(
                self._e_reset_settle_window_drift
            ),
            "garment_reset_settle_stable_steps": int(
                self._e_reset_settle_stable_steps
            ),
            "garment_reset_settle_min_seconds": float(
                _E_RESET_SETTLE_MIN_SECONDS
            ),
            "garment_reset_settle_stable_seconds": float(
                _E_RESET_SETTLE_STABLE_SECONDS
            ),
            "garment_reset_settle_max_window_drift": float(
                _E_RESET_SETTLE_MAX_WINDOW_DRIFT
            ),
            "garment_reset_settle_max_seconds": float(
                _E_RESET_SETTLE_MAX_SECONDS
            ),
            "success_counter": int(self._success_counter),
            "success_steps_required": int(self._success_steps_required()),
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"centroid", "bounds_min", "bounds_max"}
            },
        }

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        delta = np.asarray(
            state_parts.get("table_delta_height", self.delta_h), dtype=np.float64
        ).ravel()
        if delta.size:
            self.delta_h[:] = float(delta[0])
            self._restore_recorded_table_height(float(delta[0]))
        initial = np.asarray(
            state_parts.get("garment_initial_state", self._e_initial_state),
            dtype=np.float64,
        ).ravel()
        if initial.size >= 5:
            self._e_initial_state[:5] = initial[:5]
        self._e_initial_state[5] = float(self.delta_h[0])
        self._apply_task_pose_randomization()
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()

    def close(self) -> None:
        self._e_garment_view = None
        self._e_garment_indices = None
        super().close()


__all__ = ["HandsOnEIsaacLabEnv"]
