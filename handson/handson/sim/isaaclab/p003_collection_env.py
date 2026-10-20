"""Native Isaac Sim 5.1 / CPU PhysX tabletop runtime for P0-03.

The exact Ruler035 bundle is referenced byte-for-byte and receives a
versioned, runtime-only physics overlay.  Human collection shapes the ruler
through raw hand contacts while the table supports it; the runtime applies no
compliant tether, vertical support wrench, or moving joint target.  No ruler
transform or joint state is written after the reset boundary.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from handson.runtime.gpu import P003_CPU_PHYSX_GROUND_TRUTH_EXCEPTION

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .physx_ground_truth import configure_physx_contact_reports_for_recording
from .p003_contract import (
    ASSET_RELATIVE_PATH,
    ASSET_SHA256,
    DIRECTION_TOLERANCE_DEG,
    FINAL_HAND_TABLE_CONTACT_FORCE_N,
    JOINT_1_PATH,
    JOINT_2_PATH,
    JOINT_TOLERANCE_DEG,
    P003_RUNTIME_POLICY_VERSION,
    P003PhysicalReadback,
    P003StageTracker,
    PLACEMENT_HEIGHT_TOLERANCE_M,
    PLACEMENT_XY_TOLERANCE_M,
    RELEASE_CONTACT_FORCE_N,
    RELEASE_HOLD_STEPS,
    RULER_ROOT_PATH,
    S0_BODY_PATH,
    S1_BODY_PATH,
    S2_BODY_PATH,
    SHAPE_HOLD_STEPS,
    STABLE_ANGULAR_SPEED_RADPS,
    STABLE_LINEAR_SPEED_MPS,
    TARGET_KEYS,
    TARGET_SHAPES,
    TASK_NAME,
    _hands_clear_of_table,
    _tabletop_support_metrics,
    evaluate_shape,
    sample_reset_state,
)
from .p003_physics_overlay import (
    DEFAULT_P003_PHYSICS_POLICY,
    P003_PHYSICS_POLICY_VERSION,
    apply_p003_physics_overlay,
    p003_physics_policy_metadata,
    reset_root_support_offset_m,
    set_p003_mass_scale,
)


_TABLE_TOP_Z_M = 0.89
_RESET_SUPPORT_CLEARANCE_M = 0.0005
_ROOT_SUPPORT_OFFSET_M = reset_root_support_offset_m(
    clearance_m=_RESET_SUPPORT_CLEARANCE_M
)
_MATERIAL_PATH = "/World/PhysicsMaterials/P003Ruler"
_LIGHT_ROOT = "/World/Task/P003Lights"
_RENDER_RESOLUTION_HW = (480, 640)

# Terminal failure bounds catch a broken simulation, not ordinary task error.
_EXPLOSION_LINEAR_SPEED_MPS = 5.0
_EXPLOSION_ANGULAR_SPEED_RADPS = 20.0
_MAX_CONTACT_PENETRATION_M = 0.005
_IMMEDIATE_CONTACT_PENETRATION_M = 0.010
_CONTACT_PENETRATION_DEBOUNCE_SUBSTEPS = 3

_BODY_PATHS = (S0_BODY_PATH, S1_BODY_PATH, S2_BODY_PATH)
_JOINT_PATHS = (JOINT_1_PATH, JOINT_2_PATH)
_SEGMENT_LENGTH_WEIGHTS = np.asarray(
    [0.249598, 0.249607, 0.224654], dtype=np.float64
)
_SEGMENT_LENGTH_WEIGHTS /= np.sum(_SEGMENT_LENGTH_WEIGHTS)
_LOCAL_DIRECTION_AXES = (
    np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
)
_STAGE_NAMES = P003StageTracker.STAGES
_PHYSICAL_REPLAY_CONTROL_PHASES = (
    # Keep old numeric phase codes append-only so historical recordings still
    # decode.  The v2 policy never publishes the legacy "lift" phase.
    "",
    "clearance",
    "approach",
    "descend",
    "grasp",
    "lift",
    "configure",
    "place",
    "release",
    "retreat",
)
_P003_SUCCESS_CONDITION_DISPLAY_LABELS = {
    "p003_role_grasp": "Endpoint roles acquired: right S0, left S2",
    "p003_tabletop_supported": "Ruler remains supported by the tabletop",
    "p003_joint_shape": (
        f"Both joints are within +/- {JOINT_TOLERANCE_DEG:.0f} deg of target"
    ),
    "p003_direction_shape": (
        "All segment directions are within +/- "
        f"{DIRECTION_TOLERANCE_DEG:.0f} deg"
    ),
    "p003_placed": (
        "All leaves within "
        f"{PLACEMENT_HEIGHT_TOLERANCE_M * 100:.0f} cm of table and "
        f"{PLACEMENT_XY_TOLERANCE_M * 100:.0f} cm of center"
    ),
    "p003_released": (
        f"Both hand forces <= {RELEASE_CONTACT_FORCE_N:.2f} N"
    ),
    "p003_stable": "Final motion is stable",
    "p003_table_clear": "Both hands clear of table for final success",
    "p003_sequence": (
        f"Ordered hold complete ({SHAPE_HOLD_STEPS}+{RELEASE_HOLD_STEPS} frames)"
    ),
}


def _p003_hand_table_state(readback: P003PhysicalReadback) -> np.ndarray:
    """Return additive replay-visible final hand/table guard state."""

    return np.asarray(
        [
            readback.hand_table_contact_force_n,
            float(readback.hand_table_contact_valid),
        ],
        dtype=np.float64,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _p003_mass_matches_policy(actual_kg: float, expected_kg: float) -> bool:
    """Accept only the float32 round trip used by Isaac 5.1 USD mass attrs."""

    return math.isclose(
        float(actual_kg),
        float(expected_kg),
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    )


def _quat_wxyz_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.full(3, math.nan, dtype=np.float64)
    w, x, y, z = q / norm
    qv = np.asarray([x, y, z], dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _pose_xyzw_to_wxyz(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(7)
    return np.concatenate([transform[:3], transform[6:7], transform[3:6]])


class HandsOnP003IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual tabletop configure-and-release runtime for Ruler035."""

    def _warm_native_cameras_for_reset(self) -> None:
        """Prime RTX once before reset requests its first image observation.

        Construction leaves the native sensors ready, but the first reset can
        request RGB before RTX has produced a sensor frame.  A render before
        the state reset matches the validated standalone camera path and does
        not advance physics.  The P0-03 fallback override below keeps this
        warm-up fail-closed.
        """

        if (
            bool(getattr(self, "_p003_native_cameras_warmed", False))
            or not bool(getattr(self, "image_obs", False))
            or getattr(self, "_sim", None) is None
            or bool(getattr(self, "_deferred_camera_render", False))
        ):
            return
        frames = self.render()
        expected_shapes = ((480, 640, 3), (480, 848, 3), (480, 848, 3))
        actual_shapes = tuple(np.asarray(frame).shape for frame in frames)
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                "P0-03 native RGB warm-up returned unexpected frame shapes: "
                f"expected={expected_shapes}, actual={actual_shapes}"
            )
        self._p003_native_cameras_warmed = True
        print(
            "[HandsOn/P003] Native RTX cameras warmed before reset observation",
            flush=True,
        )

    def _fallback_camera_frame(
        self,
        key: str,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        """Refuse synthetic placeholder frames during native P0-03 capture."""

        if (
            bool(getattr(self, "image_obs", False))
            and getattr(self, "_sim", None) is not None
            and not bool(getattr(self, "_deferred_camera_render", False))
        ):
            warnings = "; ".join(
                str(item) for item in getattr(self, "_isaac_warnings", ())
            )
            detail = f" Last Isaac warning(s): {warnings}" if warnings else ""
            raise RuntimeError(
                "P0-03 native RGB capture is unavailable; synthetic fallback "
                "frames are forbidden. Verify the RTX/Vulkan runtime, including "
                "libXt.so.6, libGLU.so.1, and the Isaac Sim shader caches."
                f"{detail}"
            )
        return super()._fallback_camera_frame(
            key,
            height=height,
            width=width,
        )

    def __init__(self, task_name: str = TASK_NAME, *args: Any, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"P0-03 runtime requires {TASK_NAME!r}, got {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).strip().lower()
        if physics_backend not in {"physx", "isaacsim", "isaac"}:
            raise ValueError("P0-03 is Isaac Sim 5.1 PhysX-only; Newton is forbidden")
        requested_device = str(kwargs.get("isaac_device", "cpu")).strip().lower()
        if requested_device != "cpu":
            raise ValueError(
                "P0-03 requires isaac_device='cpu' for its validated tabletop "
                "physics profile and analytic collision proxies"
            )
        kwargs["physics_backend"] = "physx"
        kwargs["isaac_device"] = "cpu"
        kwargs.setdefault("physics_dt", 1.0 / 240.0)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("time_limit", 30.0)
        kwargs.setdefault("render_resolution", _RENDER_RESOLUTION_HW)

        self._fast_shutdown_on_close = True
        self._physx_cpu_ground_truth_exception = (
            P003_CPU_PHYSX_GROUND_TRUTH_EXCEPTION
        )
        requested_seed = 0 if kwargs.get("seed") is None else int(kwargs["seed"])
        self._p003_reset_serial = -1
        self._p003_reset = sample_reset_state(requested_seed)
        self._p003_articulation_view: Any | None = None
        self._p003_body_views: dict[str, Any] = {}
        self._p003_contact_physics_view: Any | None = None
        self._p003_contact_views: dict[str, Any] = {}
        self._p003_contact_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._p003_mass_attrs: dict[str, Any] = {}
        self._p003_inertia_attrs: dict[str, Any] = {}
        self._p003_nominal_masses = {
            segment.label: float(segment.nominal_mass_kg)
            for segment in DEFAULT_P003_PHYSICS_POLICY.segments
        }
        self._p003_physics_overlay_result: Any | None = None
        self._p003_physics_overlay_metadata: dict[str, Any] = {}
        self._p003_material_api: Any | None = None
        self._p003_local_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._p003_collision_local_bounds: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._p003_table_path = ""
        self._p003_contact_contract_ready = False
        self._p003_reset_root_position = np.asarray(
            [
                self._p003_reset.root_x_m,
                self._p003_reset.root_y_m,
                _TABLE_TOP_Z_M + _ROOT_SUPPORT_OFFSET_M,
            ],
            dtype=np.float64,
        )
        self._p003_reset_body_poses = np.tile(
            np.asarray([0.0, 0.0, _TABLE_TOP_Z_M, 1.0, 0.0, 0.0, 0.0]),
            (3, 1),
        )
        self._p003_reset_object_center_position = np.average(
            self._p003_reset_body_poses[:, :3],
            axis=0,
            weights=_SEGMENT_LENGTH_WEIGHTS,
        )
        self._p003_drive_targets_deg = np.asarray(
            [self._p003_reset.joint_1_deg, self._p003_reset.joint_2_deg],
            dtype=np.float64,
        )
        self._p003_owner_substeps = 0
        self._p003_assist_active = False
        self._p003_assist_substeps = 0
        self._p003_control_phase = ""
        self._p003_configure_preparing = False
        self._p003_workspace_preparing = False
        self._p003_reanchor_requested = False
        self._p003_extended_reanchors_done: set[float] = set()
        self._p003_nonextended_reanchor_count = 0
        self._p003_nonextended_reanchor_armed = True
        self._p003_release_substeps = 0
        self._p003_grasp_anchors: dict[str, np.ndarray] = {}
        self._p003_grasp_forces_n = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_force_vectors_n = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_grasp_errors_m = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_error_vectors_m = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_grasp_angle_errors_rad = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_last_target_positions: dict[str, np.ndarray] = {}
        self._p003_grasp_assist_substeps = 0
        self._p003_level_delta_z = 0.0
        self._p003_clearance_support_active = False
        self._p003_v20_grasp_anchors: dict[str, np.ndarray] = {}
        self._p003_v20_grasp_forces_n = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_force_vectors_n = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_v20_grasp_errors_m = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_error_vectors_m = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_v20_grasp_angle_errors_rad = {
            "s0": 0.0,
            "s2": 0.0,
        }
        self._p003_v20_contact_loss_substeps = {"s0": 0, "s2": 0}
        self._p003_v20_last_target_positions: dict[str, np.ndarray] = {}
        self._p003_v20_grasp_assist_substeps = 0
        self._p003_v20_clearance_support_active = False
        self._p003_tracker = P003StageTracker(self._p003_reset.target_key)
        self._p003_tracker_last_step = -1
        self._p003_last_stage_metrics: dict[str, Any] = {}
        self._p003_last_readback: P003PhysicalReadback | None = None
        self._p003_last_shape_metrics: dict[str, Any] = {}
        self._p003_last_contact_forces: dict[str, float] = {}
        self._p003_contact_readback_valid: dict[str, bool] = {}
        self._p003_max_penetration_m = 0.0
        self._p003_ruler_table_max_penetration_m = 0.0
        self._p003_robot_table_max_penetration_m = 0.0
        self._p003_hand_table_max_penetration_m = 0.0
        self._p003_monitored_contact_max_penetration_m = 0.0
        self._p003_failure_latched = False
        self._p003_failure_reason = ""
        self._p003_failure_reported = False
        self._p003_penetration_debounce_substeps = 0
        self._p003_penetration_debounce_last_physics_substep = -1
        self._p003_penetration_hard_armed = False
        self._p003_physics_substeps = 0
        self._p003_object_state_writes_after_action_zero = 0
        self._p003_native_cameras_warmed = False
        self._physx_prefer_tensor_contact_capture = True
        super().__init__(task_name, *args, **kwargs)

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "ruler_segment_poses": spaces.Box(-np.inf, np.inf, shape=(3, 7)),
                "ruler_joint_angles": spaces.Box(-np.inf, np.inf, shape=(2,)),
                "ruler_joint_targets": spaces.Box(-np.inf, np.inf, shape=(2,)),
                "ruler_segment_directions": spaces.Box(-1.0, 1.0, shape=(3, 2)),
                "ruler_target_selector": spaces.Box(0.0, 1.0, shape=(4,)),
                "p003_role_contacts": spaces.Box(-np.inf, np.inf, shape=(8,)),
                "p003_task_state": spaces.Box(-np.inf, np.inf, shape=(18,)),
            }
        )
        # Keep the established flat policy-state ABI unchanged.  This
        # final-frame guard is recording-only telemetry and is forwarded by
        # HandsOnObsAdapter as an additive top-level stream.
        observation_space.spaces["p003_hand_table_state"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float64,
        )
        return observation_space

    def _app_launcher_device(self) -> str:
        """Render on one scheduler-selected GPU while PhysX stays on CPU."""

        raw_index = os.environ.get("HANDSON_P003_RENDER_GPU", "0").strip()
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                "HANDSON_P003_RENDER_GPU must be a non-negative integer, "
                f"got {raw_index!r}"
            ) from exc
        if index < 0:
            raise ValueError(
                "HANDSON_P003_RENDER_GPU must be a non-negative integer, "
                f"got {index}"
            )
        return f"cuda:{index}"

    def _task_specific_isaac_kit_args(self) -> str:
        """Keep native Vulkan rendering on the selected active GPU only."""

        render_device = self._app_launcher_device()
        render_gpu = int(render_device.rsplit(":", 1)[-1])
        return " ".join(
            (
                f"--/renderer/activeGpu={render_gpu}",
                "--/renderer/multiGpu/enabled=false",
                "--/renderer/multiGpu/autoEnable=false",
                f"--/renderer/multiGpu/activeGpus={render_gpu}",
                "--/renderer/multiGpu/maxGpuCount=1",
            )
        )

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("P0-03 stage is unavailable before asset composition")

        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

        repository_root = Path(__file__).resolve().parents[4]
        asset_path = repository_root / ASSET_RELATIVE_PATH
        if not asset_path.is_file():
            raise FileNotFoundError(f"P0-03 exact Ruler035 asset is missing: {asset_path}")
        actual_sha = _sha256(asset_path)
        if actual_sha != ASSET_SHA256:
            raise RuntimeError(
                f"P0-03 Ruler035 SHA mismatch: expected {ASSET_SHA256}, got {actual_sha}"
            )

        root = UsdGeom.Xform.Define(self._stage, RULER_ROOT_PATH).GetPrim()
        root.GetReferences().ClearReferences()
        root.GetReferences().AddReference(str(asset_path.resolve()))
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(
                float(self._p003_reset.root_x_m),
                float(self._p003_reset.root_y_m),
                _TABLE_TOP_Z_M + _ROOT_SUPPORT_OFFSET_M,
            )
        )
        xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(
            float(self._p003_reset.root_yaw_deg)
        )

        bodies = [self._stage.GetPrimAtPath(path) for path in _BODY_PATHS]
        joints = [self._stage.GetPrimAtPath(path) for path in _JOINT_PATHS]
        if not all(prim.IsValid() for prim in (*bodies, *joints)):
            raise RuntimeError("P0-03 composed Ruler035 topology is incomplete")
        if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError("P0-03 source articulation root API is missing")
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
        overlay_result = apply_p003_physics_overlay(
            self._stage,
            bodies,
            joints,
            float(self._p003_reset.mass_scale),
            root_path=RULER_ROOT_PATH,
            clearance_m=_RESET_SUPPORT_CLEARANCE_M,
            enable_collision_proxies=True,
        )
        if not overlay_result.collision_proxies_applied:
            raise RuntimeError(
                "P0-03 analytic collision proxies were not applied: "
                f"status={overlay_result.proxy_topology_status}, "
                f"reason={overlay_result.proxy_blocked_reason}"
            )
        if not overlay_result.joint_friction_authored:
            raise RuntimeError(
                "P0-03 tabletop physics overlay could not author passive "
                "joint friction on both hinges"
            )
        self._p003_physics_overlay_result = overlay_result
        self._p003_physics_overlay_metadata = dict(overlay_result.metadata)
        overlay_offset = float(overlay_result.reset_root_support_offset_m)
        if not math.isclose(
            overlay_offset,
            _ROOT_SUPPORT_OFFSET_M,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "P0-03 overlay/reset support offsets disagree: "
                f"overlay={overlay_offset}, authored={_ROOT_SUPPORT_OFFSET_M}"
            )
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        for label, prim in zip(("s0", "s1", "s2"), bodies):
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api.CreateDisableGravityAttr(False)
            report = (
                PhysxSchema.PhysxContactReportAPI.Get(self._stage, prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(prim)
            )
            report.CreateThresholdAttr().Set(0.0)
            mass_attr = overlay_result.mass_attrs_by_label.get(label)
            if mass_attr is None or not mass_attr.IsValid():
                raise RuntimeError(
                    f"P0-03 physics overlay omitted the {label} mass attribute"
                )
            scaled_mass = float(mass_attr.Get())
            nominal = float(self._p003_nominal_masses[label])
            if not math.isfinite(nominal) or nominal <= 0.0:
                raise RuntimeError(
                    f"P0-03 physics overlay authored invalid {label} mass: "
                    f"{scaled_mass}"
                )
            if not _p003_mass_matches_policy(
                scaled_mass,
                nominal * float(self._p003_reset.mass_scale),
            ):
                raise RuntimeError(
                    f"P0-03 physics overlay {label} mass disagrees with policy: "
                    f"actual={scaled_mass}, nominal={nominal}, "
                    f"scale={self._p003_reset.mass_scale}"
                )
            self._p003_mass_attrs[label] = mass_attr
            inertia_attr = overlay_result.inertia_attrs_by_label.get(label)
            if inertia_attr is None or not inertia_attr.IsValid():
                raise RuntimeError(
                    f"P0-03 physics overlay omitted the {label} inertia attribute"
                )
            self._p003_inertia_attrs[label] = inertia_attr

            local_range = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
            lower = np.asarray(tuple(local_range.GetMin()), dtype=np.float64)
            upper = np.asarray(tuple(local_range.GetMax()), dtype=np.float64)
            if not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
                raise RuntimeError(f"P0-03 invalid local bounds for {prim.GetPath()}")
            self._p003_local_bounds[label] = (lower, upper)

        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            raise RuntimeError(
                "P0-03 table rigid body is unavailable before simulation reset"
            )
        table_report = (
            PhysxSchema.PhysxContactReportAPI.Get(
                self._stage, table_prim.GetPath()
            )
            if table_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            else PhysxSchema.PhysxContactReportAPI.Apply(table_prim)
        )
        table_report.CreateThresholdAttr().Set(0.0)

        for joint_prim, initial_deg in zip(
            joints,
            (self._p003_reset.joint_1_deg, self._p003_reset.joint_2_deg),
        ):
            joint_state = (
                PhysxSchema.JointStateAPI.Get(joint_prim, "angular")
                if joint_prim.HasAPI(PhysxSchema.JointStateAPI, "angular")
                else PhysxSchema.JointStateAPI.Apply(joint_prim, "angular")
            )
            joint_state.GetPositionAttr().Set(float(initial_deg))
            joint_state.GetVelocityAttr().Set(0.0)

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self._p003_reset.friction),
            dynamic_friction=float(self._p003_reset.friction),
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(_MATERIAL_PATH, material_cfg)
        sim_utils.bind_physics_material(
            RULER_ROOT_PATH,
            _MATERIAL_PATH,
            stage=self._stage,
            stronger_than_descendants=True,
        )
        self._p003_material_api = UsdPhysics.MaterialAPI(
            self._stage.GetPrimAtPath(_MATERIAL_PATH)
        )

        self._p003_collision_local_bounds = {
            label: (
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
            )
            for label, (lower, upper) in (
                overlay_result.collision_local_bounds_by_label.items()
            )
        }
        if set(self._p003_collision_local_bounds) != {"s0", "s1", "s2"}:
            raise RuntimeError(
                "P0-03 physics overlay returned incomplete collider bounds: "
                f"{sorted(self._p003_collision_local_bounds)}"
            )
        for label, (lower, upper) in self._p003_collision_local_bounds.items():
            if not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
                raise RuntimeError(
                    f"P0-03 invalid overlay collider bounds for {label}: "
                    f"lower={lower.tolist()}, upper={upper.tolist()}"
                )
        print(
            "[HandsOn/P003] Tabletop physics overlay: "
            + json.dumps(
                {
                    "overlay": overlay_result.report_dict(),
                    "physical_collider_bounds": {
                        label: [lower.tolist(), upper.tolist()]
                        for label, (lower, upper) in (
                            self._p003_collision_local_bounds.items()
                        )
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

        dome = UsdLux.DomeLight.Define(self._stage, f"{_LIGHT_ROOT}/DomeLight")
        dome.CreateIntensityAttr(450.0)
        dome.CreateColorAttr(Gf.Vec3f(0.86, 0.90, 1.0))
        distant = UsdLux.DistantLight.Define(
            self._stage, f"{_LIGHT_ROOT}/DistantLight"
        )
        distant.CreateIntensityAttr(1800.0)
        distant.CreateAngleAttr(2.0)
        distant.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 15.0, 25.0))

    def _capture_task_scene_state(self, task_mjcf: Path) -> None:
        """Avoid restoring individual links after the root articulation."""

        super()._capture_task_scene_state(task_mjcf)
        self._task_reset_body_states = [
            entry
            for entry in self._task_reset_body_states
            if not entry[0].startswith(f"{RULER_ROOT_PATH}/")
        ]

    def _configure_task_simulation_settings(self) -> None:
        """Enable the CPU contact-report pipeline used by the task contract."""

        import carb

        settings = carb.settings.get_settings()
        settings.set_bool("/physics/disableContactProcessing", False)
        if bool(settings.get("/physics/disableContactProcessing")):
            raise RuntimeError("P0-03 could not enable PhysX contact processing")

    def _configure_ground_truth_contact_reports(self, task_mjcf: Path) -> None:
        """Bind formal contact truth to all three exact ruler rigid bodies.

        The placeholder task MJCF does not enumerate the separately composed
        Ruler035 USD articulation.  The generic task-name lookup would
        therefore configure only the table and leave the formal per-substep
        contact trace without a dynamic sensor.  Author the report API before
        the first simulation reset and expose the exact body paths explicitly.
        """

        if self._stage is None:
            raise RuntimeError(
                "P0-03 PhysX contact recording requires the constructed USD stage"
            )
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            raise RuntimeError("P0-03 table body is unavailable for contact recording")
        sensor_paths = (*_BODY_PATHS, str(table_prim.GetPath()))
        self._physx_contact_sensor_paths_preconfigured = tuple(sensor_paths)
        self._physx_contact_report_paths_preconfigured = (
            configure_physx_contact_reports_for_recording(
                self._stage,
                sensor_paths,
            )
        )

    def _actual_tcp_pose(self, side: str) -> np.ndarray:
        """Return the live PhysX palm pose, never the commanded TCP state."""

        if side not in {"left", "right"}:
            raise ValueError(f"unknown P0-03 robot side: {side}")
        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        matrix = self._native_end_effector_pose_matrix(side, fallback)
        return self._pose_matrix_to_vector(matrix)

    def set_p003_control_state(
        self,
        phase: str,
        *,
        configure_preparing: bool = False,
        workspace_preparing: bool = False,
    ) -> None:
        """Publish replay-visible policy phase without physical side effects."""

        self._p003_control_phase = str(phase)
        self._p003_configure_preparing = bool(configure_preparing)
        self._p003_workspace_preparing = bool(workspace_preparing)

    def get_physical_replay_task_controller_state(self) -> np.ndarray:
        """Return the action-aligned P0-03 controller state.

        The scripted policy publishes this state immediately before each raw
        action.  The v2 tabletop runtime treats it as diagnostic/replay
        metadata only; it cannot activate an object force or joint target.
        """

        try:
            phase_code = _PHYSICAL_REPLAY_CONTROL_PHASES.index(
                str(self._p003_control_phase)
            )
        except ValueError as exc:
            raise RuntimeError(
                "P0-03 published an unknown physical-replay control phase: "
                f"{self._p003_control_phase!r}"
            ) from exc
        return np.asarray(
            [
                phase_code,
                int(bool(self._p003_configure_preparing)),
                int(bool(self._p003_workspace_preparing)),
            ],
            dtype=np.uint8,
        )

    def set_physical_replay_task_controller_state(
        self,
        state: np.ndarray,
    ) -> None:
        """Install one recorded action-aligned P0-03 controller state."""

        array = np.asarray(state)
        if array.shape != (3,):
            raise ValueError(
                "P0-03 physical-replay controller state must have shape (3,), "
                f"got {array.shape}"
            )
        if array.dtype.kind not in "biu" or not np.all(
            np.equal(array, array.astype(np.uint8))
        ):
            raise ValueError(
                "P0-03 physical-replay controller state must contain uint8 "
                "integer values"
            )
        values = array.astype(np.uint8, copy=False)
        phase_code = int(values[0])
        if not 0 <= phase_code < len(_PHYSICAL_REPLAY_CONTROL_PHASES):
            raise ValueError(
                "P0-03 physical-replay controller phase code is invalid: "
                f"{phase_code}"
            )
        if int(values[1]) not in (0, 1) or int(values[2]) not in (0, 1):
            raise ValueError(
                "P0-03 physical-replay controller preparation flags must be "
                "binary"
            )
        self.set_p003_control_state(
            _PHYSICAL_REPLAY_CONTROL_PHASES[phase_code],
            configure_preparing=bool(values[1]),
            workspace_preparing=bool(values[2]),
        )

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
            raise RuntimeError(f"P0-03 contact filter {key} resolved no rigid bodies")
        view = physics_view.create_rigid_contact_view(
            sensor_path,
            filter_patterns=filter_paths,
            max_contact_data_count=max(128, len(filter_paths) * 16),
        )
        try:
            sensor_count = int(view.sensor_count) if view is not None else 0
            filter_count = int(view.filter_count) if view is not None else 0
        except (AttributeError, RuntimeError, TypeError, ValueError):
            sensor_count = 0
            filter_count = 0
        if sensor_count <= 0 or filter_count != len(filter_paths):
            raise RuntimeError(
                f"P0-03 contact view {key} is incomplete "
                f"(sensors={sensor_count}, filters={filter_count}, "
                f"expected_filters={len(filter_paths)})"
            )
        self._p003_contact_views[key] = view
        self._p003_contact_paths[key] = (sensor_path, tuple(filter_paths))

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        import omni.physics.tensors as physics_tensors

        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            raise RuntimeError("P0-03 PhysX tensor simulation view is unavailable")
        self._p003_articulation_view = physics_view.create_articulation_view(
            RULER_ROOT_PATH
        )
        self._p003_body_views = {
            label: physics_view.create_rigid_body_view(path)
            for label, path in zip(("s0", "s1", "s2"), _BODY_PATHS)
        }
        if int(self._p003_articulation_view.count) != 1 or any(
            int(view.count) != 1 for view in self._p003_body_views.values()
        ):
            raise RuntimeError("P0-03 articulation/body tensor views are incomplete")

        left_hand = self._robot_rigid_body_paths("left", hand_only=True)
        right_hand = self._robot_rigid_body_paths("right", hand_only=True)
        all_robot = self._robot_rigid_body_paths("left", hand_only=False)
        all_robot.extend(self._robot_rigid_body_paths("right", hand_only=False))
        table_contact_hands = [*left_hand, *right_hand]
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            raise RuntimeError("P0-03 table rigid body is unavailable")
        self._p003_table_path = str(table_prim.GetPath())

        # Own a fresh contact view because the long-lived robot tensor view
        # does not surface pairs created by this task after scene composition.
        # CPU PhysX uses NumPy; the formal CUDA ground-truth path must use
        # Torch because the GPU pipeline rejects the NumPy frontend.
        contact_frontend = "numpy" if self._isaac_device == "cpu" else "torch"
        contact_physics_view = physics_tensors.create_simulation_view(
            contact_frontend
        )
        contact_physics_view.set_subspace_roots("/")
        self._p003_contact_physics_view = contact_physics_view

        for side, hand_paths in (("left", left_hand), ("right", right_hand)):
            for label, body_path in zip(("s0", "s1", "s2"), _BODY_PATHS):
                self._create_contact_view(
                    contact_physics_view,
                    f"{side}_{label}",
                    body_path,
                    hand_paths,
                )
        self._create_contact_view(
            contact_physics_view,
            "robot_table",
            self._p003_table_path,
            all_robot,
        )
        self._create_contact_view(
            contact_physics_view,
            "hand_table",
            self._p003_table_path,
            table_contact_hands,
        )
        for label, body_path in zip(("s0", "s1", "s2"), _BODY_PATHS):
            self._create_contact_view(
                contact_physics_view,
                f"{label}_table",
                body_path,
                [self._p003_table_path],
            )
        self._p003_contact_contract_ready = True
        print(
            "[HandsOn/P003] Exact contact contract ready: "
            f"left_hand={len(left_hand)}, right_hand={len(right_hand)}, "
            f"robot_table_filters={len(all_robot)}, "
            f"hand_table_filters={len(table_contact_hands)}",
            flush=True,
        )

    def reset(self, seed: int | None = None, **kwargs: Any):
        if seed is not None:
            self._p003_reset_serial = -1
        self._warm_native_cameras_for_reset()
        return super().reset(seed=seed, **kwargs)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._p003_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._p003_reset_serial
        self._p003_reset = sample_reset_state(episode_seed)
        self._p003_drive_targets_deg = np.asarray(
            [self._p003_reset.joint_1_deg, self._p003_reset.joint_2_deg],
            dtype=np.float64,
        )
        self._p003_owner_substeps = 0
        self._p003_assist_active = False
        self._p003_assist_substeps = 0
        self._p003_control_phase = ""
        self._p003_configure_preparing = False
        self._p003_workspace_preparing = False
        self._p003_reanchor_requested = False
        self._p003_extended_reanchors_done = set()
        self._p003_nonextended_reanchor_count = 0
        self._p003_nonextended_reanchor_armed = True
        self._p003_release_substeps = 0
        self._p003_grasp_anchors = {}
        self._p003_grasp_forces_n = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_force_vectors_n = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_grasp_errors_m = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_error_vectors_m = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_grasp_angle_errors_rad = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_last_target_positions = {}
        self._p003_grasp_assist_substeps = 0
        self._p003_level_delta_z = 0.0
        self._p003_clearance_support_active = False
        self._p003_v20_grasp_anchors = {}
        self._p003_v20_grasp_forces_n = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_force_vectors_n = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_v20_grasp_errors_m = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_error_vectors_m = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_v20_grasp_angle_errors_rad = {
            "s0": 0.0,
            "s2": 0.0,
        }
        self._p003_v20_contact_loss_substeps = {"s0": 0, "s2": 0}
        self._p003_v20_last_target_positions = {}
        self._p003_v20_grasp_assist_substeps = 0
        self._p003_v20_clearance_support_active = False
        self._p003_tracker = P003StageTracker(self._p003_reset.target_key)
        self._p003_tracker_last_step = -1
        self._p003_last_stage_metrics = {}
        self._p003_last_shape_metrics = {}
        self._p003_last_contact_forces = {}
        self._p003_contact_readback_valid = {}
        self._p003_max_penetration_m = 0.0
        self._p003_ruler_table_max_penetration_m = 0.0
        self._p003_robot_table_max_penetration_m = 0.0
        self._p003_hand_table_max_penetration_m = 0.0
        self._p003_monitored_contact_max_penetration_m = 0.0
        self._p003_last_readback = None
        self._p003_failure_latched = False
        self._p003_failure_reason = ""
        self._p003_failure_reported = False
        self._p003_penetration_debounce_substeps = 0
        self._p003_penetration_debounce_last_physics_substep = -1
        self._p003_penetration_hard_armed = False
        self._p003_physics_substeps = 0
        self._p003_object_state_writes_after_action_zero = 0

    def _apply_randomized_dynamics(self) -> None:
        friction = float(self._p003_reset.friction)
        if self._p003_physics_overlay_result is not None:
            set_p003_mass_scale(
                self._p003_physics_overlay_result,
                float(self._p003_reset.mass_scale),
            )
            self._p003_physics_overlay_metadata = dict(
                self._p003_physics_overlay_result.metadata
            )
        else:
            self._p003_physics_overlay_metadata = p003_physics_policy_metadata(
                mass_scale=float(self._p003_reset.mass_scale),
                clearance_m=_RESET_SUPPORT_CLEARANCE_M,
            )
        if self._p003_material_api:
            static_attr = self._p003_material_api.GetStaticFrictionAttr()
            dynamic_attr = self._p003_material_api.GetDynamicFrictionAttr()
            if not static_attr or not static_attr.IsValid():
                static_attr = self._p003_material_api.CreateStaticFrictionAttr()
            if not dynamic_attr or not dynamic_attr.IsValid():
                dynamic_attr = self._p003_material_api.CreateDynamicFrictionAttr()
            static_attr.Set(friction)
            dynamic_attr.Set(friction)

    def _set_drive_targets(self, target_degrees: np.ndarray) -> None:
        targets_deg = np.asarray(target_degrees, dtype=np.float64).reshape(2)
        targets_deg[0] = float(np.clip(targets_deg[0], -180.0, 0.0))
        targets_deg[1] = float(np.clip(targets_deg[1], 0.0, 180.0))
        self._p003_drive_targets_deg = targets_deg.copy()
        view = self._p003_articulation_view
        if view is None or self._torch is None:
            return
        targets = view.get_dof_positions().clone()
        if targets.ndim != 2 or int(targets.shape[1]) != 2:
            raise RuntimeError("P0-03 articulation must expose exactly two DOFs")
        targets[0, :] = self._torch.as_tensor(
            np.deg2rad(targets_deg),
            dtype=targets.dtype,
            device=targets.device,
        )
        indices = self._torch.arange(
            int(view.count), dtype=self._torch.long, device=self._isaac_device
        )
        view.set_dof_position_targets(targets, indices)

    def _apply_task_pose_randomization(self) -> None:
        if self._torch is None:
            return
        entry = next(
            (
                value
                for value in self._task_reset_articulation_states
                if value[0] == RULER_ROOT_PATH
            ),
            None,
        )
        if entry is None:
            self._add_task_state_warning("P0-03 articulation reset state is unavailable")
            return
        path, view, indices, root_pose, root_velocity, joint_pos, joint_vel = entry
        pose = root_pose.clone()
        pose[:, :3] = self._torch.as_tensor(
            [
                self._p003_reset.root_x_m,
                self._p003_reset.root_y_m,
                _TABLE_TOP_Z_M + _ROOT_SUPPORT_OFFSET_M,
            ],
            dtype=pose.dtype,
            device=pose.device,
        )
        half_yaw = math.radians(float(self._p003_reset.root_yaw_deg)) / 2.0
        pose[:, 3:7] = self._torch.as_tensor(
            [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
            dtype=pose.dtype,
            device=pose.device,
        )
        positions = joint_pos.clone()
        positions[:, :] = self._torch.as_tensor(
            np.deg2rad(
                [self._p003_reset.joint_1_deg, self._p003_reset.joint_2_deg]
            ),
            dtype=positions.dtype,
            device=positions.device,
        )
        zero_root_velocity = self._torch.zeros_like(root_velocity)
        zero_joint_velocity = self._torch.zeros_like(joint_vel)
        view.set_root_transforms(pose, indices)
        view.set_root_velocities(zero_root_velocity, indices)
        view.set_dof_positions(positions, indices)
        view.set_dof_velocities(zero_joint_velocity, indices)
        self._record_authored_task_reset_write(
            path,
            transform_xyzw=pose,
            velocity=zero_root_velocity,
            dof_position=positions,
            dof_velocity=zero_joint_velocity,
        )
        self._p003_reset_root_position = np.asarray(
            [
                self._p003_reset.root_x_m,
                self._p003_reset.root_y_m,
                _TABLE_TOP_Z_M + _ROOT_SUPPORT_OFFSET_M,
            ],
            dtype=np.float64,
        )
        self._apply_randomized_dynamics()
        self._set_drive_targets(
            np.asarray(
                [self._p003_reset.joint_1_deg, self._p003_reset.joint_2_deg],
                dtype=np.float64,
            )
        )
        if self._sim is not None:
            self._sim.forward()
        self._p003_reset_body_poses = self._body_poses()
        self._p003_reset_root_position = self._p003_reset_body_poses[0, :3].copy()
        self._p003_reset_object_center_position = self._object_center_position(
            self._p003_reset_body_poses
        )
        print(
            "[HandsOn/P003] Seeded reset: "
            + json.dumps(
                {
                    **self._p003_reset.to_dict(),
                    "target_joint_angles_deg": list(
                        self._p003_reset.target.joint_targets_deg
                    ),
                    "physics_device": str(self._isaac_device),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @staticmethod
    def _view_numpy(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(shape)

    def _body_pose(self, label: str, fallback: np.ndarray) -> np.ndarray:
        view = self._p003_body_views.get(label)
        if view is None:
            return np.asarray(fallback, dtype=np.float64).reshape(7).copy()
        try:
            return _pose_xyzw_to_wxyz(self._view_numpy(view.get_transforms()[0], (7,)))
        except Exception:
            return np.asarray(fallback, dtype=np.float64).reshape(7).copy()

    def _body_poses(self) -> np.ndarray:
        return np.stack(
            [
                self._body_pose(label, self._p003_reset_body_poses[index])
                for index, label in enumerate(("s0", "s1", "s2"))
            ],
            axis=0,
        )

    def _object_center_position(self, poses: np.ndarray) -> np.ndarray:
        world_centers: list[np.ndarray] = []
        for pose, label in zip(
            np.asarray(poses, dtype=np.float64).reshape(3, 7),
            ("s0", "s1", "s2"),
        ):
            lower, upper = self._p003_local_bounds[label]
            local_center = 0.5 * (lower + upper)
            world_centers.append(
                pose[:3] + _quat_wxyz_rotate(pose[3:7], local_center)
            )
        return np.average(
            np.stack(world_centers, axis=0),
            axis=0,
            weights=_SEGMENT_LENGTH_WEIGHTS,
        )

    def _body_velocity(self, label: str) -> np.ndarray:
        view = self._p003_body_views.get(label)
        if view is None:
            return np.zeros(6, dtype=np.float64)
        try:
            return self._view_numpy(view.get_velocities()[0], (6,))
        except Exception:
            return np.full(6, math.inf, dtype=np.float64)

    def _joint_angles_deg(self) -> np.ndarray:
        view = self._p003_articulation_view
        if view is None:
            return np.asarray(
                [self._p003_reset.joint_1_deg, self._p003_reset.joint_2_deg],
                dtype=np.float64,
            )
        try:
            positions = self._view_numpy(view.get_dof_positions(), (int(view.count), -1))
            return np.rad2deg(positions[0, :2])
        except Exception:
            return np.full(2, math.nan, dtype=np.float64)

    def _segment_directions_task_xy(self, poses: np.ndarray) -> np.ndarray:
        world_directions = np.stack(
            [
                _quat_wxyz_rotate(pose[3:7], local_axis)
                for pose, local_axis in zip(poses, _LOCAL_DIRECTION_AXES)
            ],
            axis=0,
        )
        x_axis = world_directions[0]
        x_norm = float(np.linalg.norm(x_axis))
        plane_normal = _quat_wxyz_rotate(
            poses[0, 3:7], np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        )
        plane_normal = plane_normal - x_axis * (
            float(np.dot(plane_normal, x_axis)) / max(x_norm * x_norm, 1.0e-18)
        )
        normal_norm = float(np.linalg.norm(plane_normal))
        if (
            not np.isfinite(x_norm)
            or not np.isfinite(normal_norm)
            or x_norm <= 1.0e-9
            or normal_norm <= 1.0e-9
        ):
            return np.full((3, 2), math.nan, dtype=np.float64)
        x_axis = x_axis / x_norm
        plane_normal = plane_normal / normal_norm
        y_axis = np.cross(plane_normal, x_axis)
        task_xy = np.stack(
            [world_directions @ x_axis, world_directions @ y_axis], axis=1
        )
        norms = np.linalg.norm(task_xy, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 1.0e-9):
            return np.full((3, 2), math.nan, dtype=np.float64)
        # The task frame is intrinsic to the articulated ruler: +X points
        # from hinge 1 into S0, +Z is S0's physical hinge axis, and
        # +Y = +Z x +X.  This preserves mirror-direction semantics while
        # removing irrelevant global yaw, roll, and pitch of the ruler.
        return task_xy / norms[:, None]

    def _segment_bottom_z(self, poses: np.ndarray) -> np.ndarray:
        segment_minima: list[float] = []
        for pose, label in zip(poses, ("s0", "s1", "s2")):
            lower, upper = self._p003_collision_local_bounds[label]
            minima: list[float] = []
            for x in (lower[0], upper[0]):
                for y in (lower[1], upper[1]):
                    for z in (lower[2], upper[2]):
                        corner_world = pose[:3] + _quat_wxyz_rotate(
                            pose[3:7], np.asarray([x, y, z], dtype=np.float64)
                        )
                        minima.append(float(corner_world[2]))
            segment_minima.append(float(min(minima)))
        return np.asarray(segment_minima, dtype=np.float64)

    def _minimum_segment_bottom_z(self, poses: np.ndarray) -> float:
        segment_bottoms = self._segment_bottom_z(poses)
        return (
            float(np.min(segment_bottoms))
            if segment_bottoms.size
            else math.nan
        )

    def _contact_samples(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        view = self._p003_contact_views.get(key)
        if view is None:
            self._p003_contact_readback_valid[key] = False
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        try:
            force_matrix = self._view_numpy(
                view.get_contact_force_matrix(self._sim_dt),
                (int(view.sensor_count), int(view.filter_count), 3),
            )
            forces = np.linalg.norm(force_matrix, axis=-1).reshape(-1)
            force_readback_valid = bool(np.all(np.isfinite(forces)))
            self._p003_contact_readback_valid[key] = force_readback_valid
            if not force_readback_valid:
                # Keep recorder-facing observations finite while the explicit
                # validity bit makes the final success guard fail closed.
                forces = np.zeros_like(forces)
        except Exception as exc:
            self._p003_contact_readback_valid[key] = False
            self._add_task_state_warning(
                f"P0-03 contact force readback {key} failed: {exc}"
            )
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        selected_separations: list[np.ndarray] = []
        try:
            _normal_forces, _points, _normals, separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )
            separation_values = self._view_numpy(separations, (-1,))
            counts_np = self._view_numpy(
                counts, (int(view.sensor_count), int(view.filter_count))
            ).astype(np.int64)
            starts_np = self._view_numpy(starts, counts_np.shape).astype(np.int64)
            for count, start in zip(counts_np.ravel(), starts_np.ravel()):
                if count <= 0:
                    continue
                stop = int(start + count)
                selected_separations.append(separation_values[int(start):stop])
        except Exception as exc:
            self._add_task_state_warning(
                f"P0-03 contact separation readback {key} failed: {exc}"
            )
        selected = (
            np.concatenate(selected_separations)
            if selected_separations
            else np.zeros(0, dtype=np.float64)
        )
        return forces, selected

    def _read_contact_forces(self) -> tuple[dict[str, float], float]:
        forces: dict[str, float] = {}
        ruler_table_max_penetration = 0.0
        role_contact_max_penetration = 0.0
        robot_table_max_penetration = 0.0
        hand_table_max_penetration = 0.0
        for key in self._p003_contact_views:
            samples, separations = self._contact_samples(key)
            force_n = float(np.sum(samples))
            forces[key] = force_n
            if key in {"s0_table", "s1_table", "s2_table"} and separations.size:
                ruler_table_max_penetration = max(
                    ruler_table_max_penetration,
                    float(np.max(np.maximum(0.0, -separations))),
                )
            elif key in {"right_s0", "left_s2"} and separations.size:
                role_contact_max_penetration = max(
                    role_contact_max_penetration,
                    float(np.max(np.maximum(0.0, -separations))),
                )
            elif key == "robot_table" and separations.size:
                robot_table_max_penetration = max(
                    robot_table_max_penetration,
                    float(np.max(np.maximum(0.0, -separations))),
                )
            elif key == "hand_table" and separations.size:
                hand_table_max_penetration = max(
                    hand_table_max_penetration,
                    float(np.max(np.maximum(0.0, -separations))),
                )
        self._p003_last_contact_forces = forces
        self._p003_max_penetration_m = max(
            ruler_table_max_penetration,
            robot_table_max_penetration,
        )
        self._p003_ruler_table_max_penetration_m = (
            ruler_table_max_penetration
        )
        self._p003_robot_table_max_penetration_m = (
            robot_table_max_penetration
        )
        self._p003_hand_table_max_penetration_m = hand_table_max_penetration
        self._p003_monitored_contact_max_penetration_m = max(
            role_contact_max_penetration,
            ruler_table_max_penetration,
        )
        return forces, self._p003_max_penetration_m

    def _raw_physical_readback(self) -> P003PhysicalReadback:
        poses = self._body_poses()
        velocities = np.stack(
            [self._body_velocity(label) for label in ("s0", "s1", "s2")],
            axis=0,
        )
        forces, _penetration = self._read_contact_forces()
        directions = self._segment_directions_task_xy(poses)
        joint_angles = self._joint_angles_deg()
        finite = bool(
            np.all(np.isfinite(poses))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(directions))
            and np.all(np.isfinite(joint_angles))
        )
        return P003PhysicalReadback(
            joint_angles_deg=tuple(float(value) for value in joint_angles),
            segment_directions_task_xy=tuple(
                tuple(float(value) for value in row) for row in directions
            ),
            root_position_m=tuple(float(value) for value in poses[0, :3]),
            reset_root_position_m=tuple(
                float(value) for value in self._p003_reset_root_position
            ),
            object_center_position_m=tuple(
                float(value) for value in self._object_center_position(poses)
            ),
            reset_object_center_position_m=tuple(
                float(value)
                for value in self._p003_reset_object_center_position
            ),
            table_top_z_m=_TABLE_TOP_Z_M,
            segment_bottom_z_m=tuple(
                float(value) for value in self._segment_bottom_z(poses)
            ),
            minimum_segment_bottom_z_m=self._minimum_segment_bottom_z(poses),
            maximum_linear_speed_mps=float(
                np.max(np.linalg.norm(velocities[:, :3], axis=1))
            ),
            maximum_angular_speed_radps=float(
                np.max(np.linalg.norm(velocities[:, 3:], axis=1))
            ),
            left_s0_contact_force_n=forces.get("left_s0", 0.0),
            right_s2_contact_force_n=forces.get("right_s2", 0.0),
            left_s2_contact_force_n=forces.get("left_s2", 0.0),
            right_s0_contact_force_n=forces.get("right_s0", 0.0),
            left_all_ruler_contact_force_n=sum(
                forces.get(f"left_{label}", 0.0) for label in ("s0", "s1", "s2")
            ),
            right_all_ruler_contact_force_n=sum(
                forces.get(f"right_{label}", 0.0) for label in ("s0", "s1", "s2")
            ),
            robot_table_contact_force_n=forces.get("robot_table", 0.0),
            hand_table_contact_force_n=forces.get("hand_table", 0.0),
            hand_table_contact_valid=bool(
                self._p003_contact_readback_valid.get("hand_table", False)
            ),
            finite_state=finite,
        )

    def _physical_readback(self) -> P003PhysicalReadback:
        """Return unsynthesized PhysX pose, velocity, and contact readback."""

        return self._raw_physical_readback()

    def _refresh_readback(self) -> P003PhysicalReadback:
        self._p003_last_readback = self._physical_readback()
        self._p003_last_shape_metrics = evaluate_shape(
            self._p003_last_readback, self._p003_reset.target
        )
        return self._p003_last_readback

    def _latch_failure(self, readback: P003PhysicalReadback) -> bool:
        reasons: list[str] = []
        if not readback.finite_state:
            reasons.append("nonfinite_state")
        if readback.maximum_linear_speed_mps > _EXPLOSION_LINEAR_SPEED_MPS:
            reasons.append("linear_explosion")
        if readback.maximum_angular_speed_radps > _EXPLOSION_ANGULAR_SPEED_RADPS:
            reasons.append("angular_explosion")
        # Endpoint role mistakes and hand/table contact are recoverable.  Only
        # deep ruler/table penetration during an actual recorded/operator
        # action can become terminal.  A single 240 Hz sample is debounced;
        # gross 10 mm penetration remains immediately irrecoverable.
        self._p003_penetration_hard_armed = bool(
            getattr(self, "_trajectory_action_phase_active", False)
        )
        penetration_m = float(self._p003_ruler_table_max_penetration_m)
        physics_substep = int(self._p003_physics_substeps)
        fresh_substep = bool(
            physics_substep
            != self._p003_penetration_debounce_last_physics_substep
        )
        if self._p003_penetration_hard_armed and fresh_substep:
            self._p003_penetration_debounce_last_physics_substep = physics_substep
            if penetration_m > _MAX_CONTACT_PENETRATION_M:
                self._p003_penetration_debounce_substeps += 1
            else:
                self._p003_penetration_debounce_substeps = 0
        else:
            if not self._p003_penetration_hard_armed:
                self._p003_penetration_debounce_substeps = 0
        deep_penetration = bool(
            self._p003_penetration_hard_armed
            and (
                penetration_m > _IMMEDIATE_CONTACT_PENETRATION_M
                or self._p003_penetration_debounce_substeps
                >= _CONTACT_PENETRATION_DEBOUNCE_SUBSTEPS
            )
        )
        if deep_penetration:
            reasons.append("ruler_table_penetration")
        if reasons and not self._p003_failure_latched:
            self._p003_failure_latched = True
            self._p003_failure_reason = ",".join(reasons)
        if self._p003_failure_latched and not self._p003_failure_reported:
            self._p003_failure_reported = True
            print(
                "[HandsOn/P003] Failure latched: "
                + json.dumps(
                    {
                        "env_step": int(self.env_step),
                        "reason": self._p003_failure_reason,
                        "readback": asdict(readback),
                        "contact_forces_n": self._p003_last_contact_forces,
                        "max_penetration_m": self._p003_max_penetration_m,
                        "ruler_table_max_penetration_m": (
                            self._p003_ruler_table_max_penetration_m
                        ),
                        "penetration_debounce_substeps": (
                            self._p003_penetration_debounce_substeps
                        ),
                        "penetration_hard_armed": (
                            self._p003_penetration_hard_armed
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return self._p003_failure_latched

    def _clear_disabled_assist_state(self) -> None:
        """Keep the legacy recorder ABI explicit while all assists stay off."""

        self._p003_owner_substeps = 0
        self._p003_assist_active = False
        self._p003_assist_substeps = 0
        self._p003_grasp_anchors = {}
        self._p003_v20_grasp_anchors = {}
        self._p003_grasp_forces_n = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_forces_n = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_force_vectors_n = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_v20_grasp_force_vectors_n = dict(
            self._p003_grasp_force_vectors_n
        )
        self._p003_grasp_errors_m = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_errors_m = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_error_vectors_m = {
            "s0": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
        }
        self._p003_v20_grasp_error_vectors_m = dict(
            self._p003_grasp_error_vectors_m
        )
        self._p003_grasp_angle_errors_rad = {"s0": 0.0, "s2": 0.0}
        self._p003_v20_grasp_angle_errors_rad = {"s0": 0.0, "s2": 0.0}
        self._p003_grasp_assist_substeps = 0
        self._p003_v20_grasp_assist_substeps = 0
        self._p003_clearance_support_active = False
        self._p003_v20_clearance_support_active = False

    def _after_task_physics_substep(self) -> None:
        self._p003_physics_substeps += 1
        readback = self._refresh_readback()
        self._latch_failure(readback)
        # P0-03 v2 is deliberately human/contact controlled.  Do not call the
        # historical compliant-grasp wrench or advance a joint target here.
        self._clear_disabled_assist_state()

    def _refresh_stage_tracker(self) -> dict[str, Any]:
        if self._p003_tracker_last_step == int(self.env_step):
            return self._p003_last_stage_metrics
        readback = self._refresh_readback()
        self._p003_last_stage_metrics = self._p003_tracker.update(readback)
        self._p003_tracker_last_step = int(self.env_step)
        return self._p003_last_stage_metrics

    def _compute_success_now(self) -> bool:
        metrics = self._refresh_stage_tracker()
        return bool(metrics.get("success", False) and not self._p003_failure_latched)

    def _compute_failure(self) -> bool:
        readback = self._p003_last_readback or self._refresh_readback()
        return self._latch_failure(readback)

    def _success_steps_required(self) -> int:
        # The stage tracker itself enforces the 15-step shape hold and 30-step
        # released/stable hold at control frequency.
        return 1

    def _target_prompt(self) -> str:
        return self._p003_reset.target.prompt

    def _task_info(self) -> dict[str, Any]:
        readback = self._p003_last_readback or self._refresh_readback()
        stage = self._refresh_stage_tracker()
        shape = self._p003_last_shape_metrics
        return {
            "p003_target_key": self._p003_reset.target_key,
            "p003_target_title": self._p003_reset.target.title,
            "p003_target_prompt": self._target_prompt(),
            "p003_target_description": self._p003_reset.target.shape_description,
            "p003_target_joint_angles_deg": list(
                self._p003_reset.target.joint_targets_deg
            ),
            "p003_stage": self._p003_tracker.stage,
            "p003_stage_index": int(self._p003_tracker.stage_index),
            "p003_shape_hold_count": int(self._p003_tracker.shape_hold_count),
            "p003_release_hold_count": int(self._p003_tracker.release_hold_count),
            "p003_stage_metrics": stage,
            "p003_shape_metrics": shape,
            "p003_runtime_policy_version": P003_RUNTIME_POLICY_VERSION,
            "p003_physics_policy_version": P003_PHYSICS_POLICY_VERSION,
            "p003_physics_overlay_metadata": dict(
                self._p003_physics_overlay_metadata
            ),
            "p003_joint_target_assist_enabled": False,
            "p003_compliant_grasp_enabled": False,
            "p003_vertical_support_enabled": False,
            "p003_wrong_role_contact_is_terminal": False,
            "p003_tabletop_support": _tabletop_support_metrics(readback),
            "p003_physical_readback": asdict(readback),
            "p003_contact_forces_n": dict(self._p003_last_contact_forces),
            "p003_raw_collision_forces_n": dict(
                self._p003_last_contact_forces
            ),
            "p003_compliant_grasp_owners": sorted(
                self._p003_grasp_anchors
            ),
            "p003_compliant_grasp_forces_n": dict(
                self._p003_grasp_forces_n
            ),
            "p003_compliant_grasp_force_vectors_n": dict(
                self._p003_grasp_force_vectors_n
            ),
            "p003_compliant_grasp_errors_m": dict(
                self._p003_grasp_errors_m
            ),
            "p003_compliant_grasp_error_vectors_m": dict(
                self._p003_grasp_error_vectors_m
            ),
            "p003_compliant_grasp_angle_errors_rad": dict(
                self._p003_grasp_angle_errors_rad
            ),
            "p003_control_phase": self._p003_control_phase,
            "p003_configure_preparing": bool(
                self._p003_configure_preparing
            ),
            "p003_workspace_preparing": bool(
                self._p003_workspace_preparing
            ),
            "p003_extended_reanchors_done_deg": sorted(
                self._p003_extended_reanchors_done
            ),
            "p003_nonextended_reanchor_count": int(
                self._p003_nonextended_reanchor_count
            ),
            "p003_max_penetration_m": float(self._p003_max_penetration_m),
            "p003_ruler_table_max_penetration_m": float(
                self._p003_ruler_table_max_penetration_m
            ),
            "p003_robot_table_max_penetration_m": float(
                self._p003_robot_table_max_penetration_m
            ),
            "p003_hand_table_max_penetration_m": float(
                self._p003_hand_table_max_penetration_m
            ),
            "p003_monitored_contact_max_penetration_m": float(
                self._p003_monitored_contact_max_penetration_m
            ),
            "p003_robot_table_contact_force_n": float(
                readback.robot_table_contact_force_n
            ),
            "p003_hand_table_contact_force_n": float(
                readback.hand_table_contact_force_n
            ),
            "p003_hand_table_contact_readback_valid": bool(
                readback.hand_table_contact_valid
            ),
            "p003_hands_clear_of_table": bool(
                _hands_clear_of_table(readback)
            ),
            "p003_hand_table_success_threshold_n": (
                FINAL_HAND_TABLE_CONTACT_FORCE_N
            ),
            "p003_hand_table_contact_is_terminal": False,
            "p003_penetration_hard_armed": bool(
                self._p003_penetration_hard_armed
            ),
            "p003_penetration_debounce_substeps": int(
                self._p003_penetration_debounce_substeps
            ),
            "p003_penetration_debounce_required_substeps": (
                _CONTACT_PENETRATION_DEBOUNCE_SUBSTEPS
            ),
            "p003_penetration_debounce_threshold_m": (
                _MAX_CONTACT_PENETRATION_M
            ),
            "p003_penetration_immediate_threshold_m": (
                _IMMEDIATE_CONTACT_PENETRATION_M
            ),
            "p003_failure_latched": bool(self._p003_failure_latched),
            "p003_failure_reason": self._p003_failure_reason,
            "p003_contact_gated_assist_active": bool(self._p003_assist_active),
            "p003_contact_gated_assist_substeps": int(self._p003_assist_substeps),
            "p003_compliant_grasp_substeps": int(
                self._p003_grasp_assist_substeps
            ),
            "p003_drive_targets_deg": self._p003_drive_targets_deg.tolist(),
            "p003_contact_contract_ready": bool(self._p003_contact_contract_ready),
            "p003_reset": self._p003_reset.to_dict(),
            "p003_physics_device": str(self._isaac_device),
            "p003_object_state_writes_after_action_zero": int(
                self._p003_object_state_writes_after_action_zero
            ),
        }

    def _task_status_text(self) -> str:
        angles = self._joint_angles_deg()
        target = self._p003_reset.target
        return (
            f"{self._p003_reset.target_key} | "
            f"{self._p003_tracker.stage_index + 1}/{len(_STAGE_NAMES)} | "
            f"q1 {angles[0]:.0f}/{target.joint_targets_deg[0]:.0f} | "
            f"q2 {angles[1]:.0f}/{target.joint_targets_deg[1]:.0f} deg"
        )

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        key = str(row.get("key") or "")
        if key in _P003_SUCCESS_CONDITION_DISPLAY_LABELS:
            return _P003_SUCCESS_CONDITION_DISPLAY_LABELS[key]
        return HandsOnABCIsaacLabEnv._success_condition_display_label(row)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        readback = self._p003_last_readback or self._refresh_readback()
        stage = self._refresh_stage_tracker()
        shape = self._p003_last_shape_metrics
        return [
            self._condition_row(
                "p003_role_grasp",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_role_grasp"],
                bool(stage.get("roles_acquired", False)),
            ),
            self._condition_row(
                "p003_tabletop_supported",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS[
                    "p003_tabletop_supported"
                ],
                bool(stage.get("tabletop_supported", False)),
            ),
            self._condition_row(
                "p003_joint_shape",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_joint_shape"],
                bool(shape.get("joint_angles_pass", False)),
            ),
            self._condition_row(
                "p003_direction_shape",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_direction_shape"],
                bool(shape.get("segment_directions_pass", False)),
            ),
            self._condition_row(
                "p003_placed",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_placed"],
                bool(stage.get("placed", False)),
            ),
            self._condition_row(
                "p003_released",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_released"],
                bool(stage.get("released", False)),
            ),
            self._condition_row(
                "p003_stable",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_stable"],
                bool(stage.get("stable", False)),
            ),
            self._condition_row(
                "p003_table_clear",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_table_clear"],
                bool(stage.get("hands_clear_of_table", False)),
            ),
            self._condition_row(
                "p003_sequence",
                _P003_SUCCESS_CONDITION_DISPLAY_LABELS["p003_sequence"],
                bool(self._p003_tracker.success),
            ),
        ]

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        readback = self._p003_last_readback or self._refresh_readback()
        poses = self._body_poses()
        directions = np.asarray(readback.segment_directions_task_xy, dtype=np.float64)
        joint_angles_rad = np.deg2rad(np.asarray(readback.joint_angles_deg))
        target_angles_rad = np.deg2rad(
            np.asarray(self._p003_reset.target.joint_targets_deg, dtype=np.float64)
        )
        selector = np.zeros(len(TARGET_KEYS), dtype=np.float64)
        selector[TARGET_KEYS.index(self._p003_reset.target_key)] = 1.0
        stage = self._p003_last_stage_metrics or self._refresh_stage_tracker()
        shape = self._p003_last_shape_metrics
        state = observation["state"]
        state["ruler_segment_poses"] = poses
        state["ruler_joint_angles"] = joint_angles_rad
        state["ruler_joint_targets"] = target_angles_rad
        state["ruler_segment_directions"] = directions
        state["ruler_target_selector"] = selector
        state["p003_role_contacts"] = np.asarray(
            [
                readback.left_s0_contact_force_n,
                readback.right_s2_contact_force_n,
                readback.left_s2_contact_force_n,
                readback.right_s0_contact_force_n,
                readback.left_all_ruler_contact_force_n,
                readback.right_all_ruler_contact_force_n,
                readback.robot_table_contact_force_n,
                self._p003_max_penetration_m,
            ],
            dtype=np.float64,
        )
        observation["p003_hand_table_state"] = _p003_hand_table_state(readback)
        joint_errors = shape.get("joint_errors_deg", (math.inf, math.inf))
        direction_errors = shape.get(
            "direction_errors_deg", (math.inf, math.inf, math.inf)
        )
        state["p003_task_state"] = np.asarray(
            [
                float(self._p003_reset.seed),
                float(TARGET_KEYS.index(self._p003_reset.target_key)),
                float(self._p003_reset.root_x_m),
                float(self._p003_reset.root_y_m),
                float(self._p003_reset.root_yaw_deg),
                float(self._p003_reset.friction),
                float(self._p003_reset.mass_scale),
                float(self._p003_tracker.stage_index),
                float(self._p003_tracker.shape_hold_count),
                float(self._p003_tracker.release_hold_count),
                float(self._p003_assist_active),
                float(self._p003_assist_substeps),
                float(joint_errors[0]),
                float(joint_errors[1]),
                float(max(direction_errors)),
                float(stage.get("placed", False)),
                float(stage.get("released", False)),
                float(self._p003_failure_latched),
            ],
            dtype=np.float64,
        )
        return observation

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        """Restore only the initial articulation boundary for replay."""

        poses = np.asarray(
            state_parts.get("ruler_segment_poses", np.zeros((0, 7))),
            dtype=np.float64,
        )
        joints = np.asarray(
            state_parts.get("ruler_joint_angles", np.zeros(0)), dtype=np.float64
        ).reshape(-1)
        targets = np.asarray(
            state_parts.get("ruler_joint_targets", np.zeros(0)), dtype=np.float64
        ).reshape(-1)
        selector = np.asarray(
            state_parts.get("ruler_target_selector", np.zeros(0)), dtype=np.float64
        ).reshape(-1)
        restored = False
        if (
            poses.shape == (3, 7)
            and joints.size == 2
            and targets.size == 2
            and selector.size == len(TARGET_KEYS)
            and self._p003_articulation_view is not None
            and self._torch is not None
        ):
            target_index = int(np.argmax(selector))
            if not np.isclose(selector[target_index], 1.0) or not np.isclose(
                np.sum(selector), 1.0
            ):
                raise ValueError("Recorded P0-03 target selector is not one-hot")
            target_key = TARGET_KEYS[target_index]
            expected_targets = np.deg2rad(
                np.asarray(
                    TARGET_SHAPES[target_key].joint_targets_deg,
                    dtype=np.float64,
                )
            )
            if not np.allclose(targets, expected_targets, atol=1.0e-6, rtol=0.0):
                raise ValueError("Recorded P0-03 target angles disagree with selector")
            self._p003_reset = replace(self._p003_reset, target_key=target_key)
            self._p003_tracker = P003StageTracker(target_key)
            view = self._p003_articulation_view
            indices = self._torch.arange(
                int(view.count), dtype=self._torch.long, device=self._isaac_device
            )
            root = view.get_root_transforms().clone()
            velocities = self._torch.zeros_like(view.get_root_velocities())
            positions = view.get_dof_positions().clone()
            dof_velocities = self._torch.zeros_like(view.get_dof_velocities())
            root[:, :3] = self._torch.as_tensor(
                poses[0, :3], dtype=root.dtype, device=root.device
            )
            root[:, 3:7] = self._torch.as_tensor(
                [poses[0, 4], poses[0, 5], poses[0, 6], poses[0, 3]],
                dtype=root.dtype,
                device=root.device,
            )
            positions[:, :2] = self._torch.as_tensor(
                joints, dtype=positions.dtype, device=positions.device
            )
            view.set_root_transforms(root, indices)
            view.set_root_velocities(velocities, indices)
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(dof_velocities, indices)
            self._p003_reset_root_position = poses[0, :3].copy()
            self._p003_reset_object_center_position = (
                self._object_center_position(poses)
            )
            self._set_drive_targets(np.rad2deg(joints))
            restored = True
        self._refresh_object_poses()
        return {
            "restored": restored,
            "restored_task_articulation": RULER_ROOT_PATH if restored else "",
            "restored_joints": bool(restored),
            "restored_target": bool(restored),
            "nonroot_link_state_write": False,
        }


__all__ = ["HandsOnP003IsaacLabEnv"]
