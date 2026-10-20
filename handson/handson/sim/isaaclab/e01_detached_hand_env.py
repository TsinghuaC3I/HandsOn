"""E01 PhysX cloth runtime with only the two canonical Wuji hands live.

The 54D collection action remains unchanged, but palm poses are followed by a
finite world-frame wrench servo at the PhysX substep rate.  ROKAE arms and
online IK are deliberately absent; the existing post-episode planner can
reconstruct them from the saved palm trajectory.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import numpy as np

from ..isaacsim.runtime import IsaacSimUnavailable
from .abc_collection_env import _TASK_ROOT
from .bimanual_assembly_env import _blend_quat_wxyz
from .compat import make_simulation_cfg
from .e01_detached_hand_control import (
    E01_PLAN1_MAX_HAND_JOINT_STEP,
    E01_PLAN1_MAX_TCP_ROTATION_STEP,
    E01_PLAN1_MAX_TCP_TRANSLATION_STEP,
    E01_TABLE_TOP_Z,
    E01Plan1CommandFilter,
)
from .e01_detached_hand_scene import (
    E01_DETACHED_HAND_HOME_POSES,
    E01_HAND_CONTACT_OFFSET,
    E01_HAND_MAX_DEPENETRATION_VELOCITY,
    E01_HAND_REST_OFFSET,
    E01_HAND_SOLVER_POSITION_ITERATIONS,
    E01_HAND_SOLVER_VELOCITY_ITERATIONS,
    E01_WUJI_HAND_DIM,
    E01_WUJI_HAND_HOME,
    detached_wuji_hand_cfg,
    isaaclab_uses_xyzw_quaternions,
)
from .e_collection_env import HandsOnEIsaacLabEnv
from .e_collection_scene import E01_PHYSICS_HZ
from .physics_config import bounded_single_env_gpu_physx_kwargs
from .physx_ground_truth import configure_physx_contact_reports_for_recording
from .runtime_display import enforce_isaac_viewport_only, isaac_kit_args


# These gains are deliberately modest for the roughly 0.5 kg canonical Wuji
# articulation.  More importantly, the final force/torque is norm-limited, so
# neither a large tracker error nor a stale derivative can create an unbounded
# impulse at a cloth contact.  They remain fixed until the GPU contact sweep is
# complete; silently increasing them through environment variables would
# defeat that safety property.
E01_PALM_POSITION_KP_N_PER_M = 100.0
E01_PALM_LINEAR_VELOCITY_KD_N_S_PER_M = 10.0
E01_PALM_ORIENTATION_KP_NM_PER_RAD = 0.50
E01_PALM_ANGULAR_VELOCITY_KD_NM_S_PER_RAD = 0.05
E01_PALM_MAX_FORCE_N = 2.0
E01_PALM_MAX_TORQUE_NM = 0.05
E01_PALM_MAX_TRACKING_TRANSLATION_ERROR_M = 0.010
E01_PALM_MAX_TRACKING_ROTATION_ERROR_RAD = math.radians(5.0)
E01_CONTACT_PROXY_ROOT_PATH = "/World/E01HandContactProxies"
E01_CONTACT_PROXY_RADIUS_M = 0.032
E01_CONTACT_PROXY_TABLE_CLEARANCE_M = 0.001
E01_CONTACT_PROXY_CONTACT_OFFSET_M = 0.004
E01_CONTACT_PROXY_REST_OFFSET_M = 0.0005
E01_CONTACT_PROXY_MASS_KG = 0.05
E01_CLOTH_CONTACT_ASSIST_RADIUS_M = 0.075
E01_CLOTH_CONTACT_ASSIST_MARGIN_M = 0.008
E01_CLOTH_CONTACT_ASSIST_SURFACE_OFFSET_M = 0.002
E01_CLOTH_CONTACT_ASSIST_MAX_NODE_STEP_M = 0.0015
E01_CLOTH_CONTACT_ASSIST_Z_GAIN = 0.35
E01_CLOTH_CONTACT_ASSIST_XY_GAIN = 0.80
E01_CLOTH_CONTACT_ASSIST_LIFT_GAIN = 0.90
E01_CLOTH_CONTACT_ASSIST_GRASP_JOINT_DELTA_RAD = 0.025
E01_CLOTH_CONTACT_ASSIST_GRASP_MARGIN_M = 0.040
# The Plan1 filter bounds command-to-command motion, but a soft PhysX drive can
# lag behind that command for many frames while touching cloth.  Independently
# cap each drive target around actual joint readback so accumulated lag cannot
# turn the next four-substep interpolation into a large contact impulse.
E01_FINGER_MAX_TRACKING_ERROR_RAD = E01_PLAN1_MAX_HAND_JOINT_STEP


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(default) if not value else float(value)


class HandsOnE01DetachedHandsIsaacLabEnv(HandsOnEIsaacLabEnv):
    """Native surface-deformable E01 with arm-free substep hand control."""

    # The detached hands reset palm roots, wrench composers, contact proxies,
    # and command-filter state as one E01-specific transaction.  The generic
    # articulated-arm home helper only resets joints, so it must not replace
    # that contract during recording-origin stabilization.
    stable_recording_origin_default_robot_restore_enabled = False
    # Palm wrench/contact-proxy inputs are not represented by joint targets.
    recorded_drive_target_replay_supported = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hz = int(kwargs.get("hz", 30))
        expected_control_dt = 1.0 / hz if hz > 0 else 1.0 / 30.0
        if "control_dt" in kwargs and hz > 0 and not np.isclose(
            float(kwargs["control_dt"]),
            expected_control_dt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "E01 detached-hand control_dt must equal 1/hz so every "
                "tracker command is interpolated across the complete PhysX "
                f"interval; got control_dt={kwargs['control_dt']}, hz={hz}"
            )
        kwargs.setdefault("control_dt", expected_control_dt)
        kwargs.setdefault("physics_dt", 1.0 / E01_PHYSICS_HZ)
        self._plan1_max_tcp_translation_step = (
            E01_PLAN1_MAX_TCP_TRANSLATION_STEP
        )
        self._plan1_max_tcp_rotation_step = E01_PLAN1_MAX_TCP_ROTATION_STEP
        requested_contact_translation = _env_float(
            "HANDSON_ISAACLAB_E01_CONTACT_MAX_TCP_TRANSLATION_STEP",
            0.002,
        )
        requested_contact_rotation = math.radians(
            _env_float(
                "HANDSON_ISAACLAB_E01_CONTACT_MAX_TCP_ROTATION_STEP_DEG",
                1.0,
            )
        )
        if requested_contact_translation <= 0.0 or requested_contact_rotation <= 0.0:
            raise ValueError(
                "E01 detached-hand contact-safe translation and rotation steps "
                "must both be positive"
            )
        self._contact_max_tcp_translation_step = min(
            requested_contact_translation,
            self._plan1_max_tcp_translation_step,
        )
        self._contact_max_tcp_rotation_step = min(
            requested_contact_rotation,
            self._plan1_max_tcp_rotation_step,
        )
        self._plan1_command_filter = E01Plan1CommandFilter(
            max_tcp_translation_step=self._contact_max_tcp_translation_step,
            max_tcp_rotation_step=self._contact_max_tcp_rotation_step,
            max_hand_joint_step=E01_PLAN1_MAX_HAND_JOINT_STEP,
        )
        self._detached_motion_start_pose = {
            side: np.asarray(pose, dtype=np.float64)
            for side, pose in E01_DETACHED_HAND_HOME_POSES.items()
        }
        self._detached_motion_target_pose = {
            side: pose.copy()
            for side, pose in self._detached_motion_start_pose.items()
        }
        home_hand = np.asarray(E01_WUJI_HAND_HOME, dtype=np.float64)
        self._detached_motion_start_hand = {
            "right": home_hand.copy(),
            "left": home_hand.copy(),
        }
        self._detached_motion_target_hand = {
            side: hand.copy()
            for side, hand in self._detached_motion_start_hand.items()
        }
        self._detached_sim_pose = {
            side: pose.copy()
            for side, pose in self._detached_motion_start_pose.items()
        }
        self._detached_sim_hand = {
            side: hand.copy()
            for side, hand in self._detached_motion_start_hand.items()
        }
        self._detached_sim_velocity = {
            "right": np.zeros(6, dtype=np.float64),
            "left": np.zeros(6, dtype=np.float64),
        }
        self._palm_tracking_translation_error = {
            "right": 0.0,
            "left": 0.0,
        }
        self._palm_tracking_rotation_error = {
            "right": 0.0,
            "left": 0.0,
        }
        self._palm_tracking_translation_saturated = {
            "right": False,
            "left": False,
        }
        self._palm_tracking_rotation_saturated = {
            "right": False,
            "left": False,
        }
        self._palm_tracking_saturation_substeps = {"right": 0, "left": 0}
        self._palm_force_saturated = {"right": False, "left": False}
        self._palm_torque_saturated = {"right": False, "left": False}
        self._finger_tracking_error = {"right": 0.0, "left": 0.0}
        self._finger_tracking_saturated = {"right": False, "left": False}
        self._contact_proxy_radius = _env_float(
            "HANDSON_ISAACLAB_E01_CONTACT_PROXY_RADIUS",
            E01_CONTACT_PROXY_RADIUS_M,
        )
        if (
            not math.isfinite(self._contact_proxy_radius)
            or self._contact_proxy_radius <= 0.0
        ):
            raise ValueError(
                "E01 contact proxy radius must be finite and positive: "
                f"{self._contact_proxy_radius}"
            )
        self._contact_proxy_paths: dict[str, str] = {}
        self._contact_proxy_views: dict[str, Any] = {}
        self._contact_proxy_indices: dict[str, Any] = {}
        self._contact_proxy_last_pose = {
            side: np.asarray(
                [
                    E01_DETACHED_HAND_HOME_POSES[side][0],
                    E01_DETACHED_HAND_HOME_POSES[side][1],
                    E01_DETACHED_HAND_HOME_POSES[side][2],
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            for side in ("right", "left")
        }
        self._contact_proxy_grasp_reference: dict[
            str, tuple[float, float] | None
        ] = {"right": None, "left": None}
        self._cloth_contact_assist_last_proxy_pose = {
            side: pose.copy() for side, pose in self._contact_proxy_last_pose.items()
        }
        self._cloth_contact_assist_grasp_active = {"right": False, "left": False}
        self._cloth_contact_assist_disabled = False
        self._cloth_contact_assist_active_nodes = {"right": 0, "left": 0}
        self._detached_motion_substep = 0
        self._e_settled_baseline_captured = False
        # Interface-only tests have no IsaacLab package loaded and retain the
        # repository's WXYZ convention.  Runtime setup replaces this from the
        # installed public InitialStateCfg identity (2.x=WXYZ, 3.x=XYZW).
        self._isaaclab_quaternions_are_xyzw = False
        # Surface-deformable tensor resets do not yet provide an IPC-like
        # recoverable randomized initial state.  Keep physical reset state
        # deterministic in this runtime; requested randomization still applies
        # to task visuals and cameras.
        self._physical_randomization = False
        super().__init__(*args, **kwargs)
        covered_dt = self._n_substeps * self._physics_dt
        if not np.isclose(
            covered_dt,
            self._control_dt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "E01 detached-hand physics_dt must divide control_dt exactly; "
                f"got control_dt={self._control_dt}, "
                f"physics_dt={self._physics_dt}, substeps={self._n_substeps}"
            )
        self._sync_plan1_command_state()

    # ------------------------------------------------------------------
    # E01-only Isaac construction
    # ------------------------------------------------------------------

    def _setup_isaaclab(self) -> None:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise IsaacSimUnavailable(
                "Isaac Lab AppLauncher is not available in this Python environment."
            ) from exc

        launcher_args = {
            "headless": self._headless,
            "enable_cameras": self.image_obs,
            "width": self._render_width,
            "height": self._render_height,
            "device": self._isaac_device,
        }
        kit_args = isaac_kit_args(
            existing=self._task_specific_isaac_kit_args(),
            headless=self._headless,
        )
        if kit_args:
            launcher_args["kit_args"] = kit_args
        self._app_launcher = AppLauncher(launcher_args)
        self._simulation_app = self._app_launcher.app
        enforce_isaac_viewport_only(
            self._simulation_app,
            headless=self._headless,
        )
        self._setup_timeline_pause_guard()
        self._setup_keyboard_reset()

        try:
            try:
                from isaacsim.core.utils.extensions import enable_extension
            except ModuleNotFoundError:
                from isaacsim.core.experimental.utils.impl.app import (
                    enable_extension,
                )

            enable_extension("isaacsim.asset.importer.mjcf")
            self._simulation_app.update()
        except Exception as exc:
            self._isaac_warnings.append(f"Could not enable MJCF importer: {exc}")

        try:
            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation, ArticulationCfg
            from isaaclab.sim import SimulationContext
        except Exception as exc:
            raise IsaacSimUnavailable(
                "Isaac Lab APIs are not available in this Python environment."
            ) from exc

        self._isaaclab_quaternions_are_xyzw = (
            isaaclab_uses_xyzw_quaternions(ArticulationCfg.InitialStateCfg)
        )
        self._torch = torch
        sim_cfg = make_simulation_cfg(
            sim_utils,
            physics_backend="physx",
            device=self._isaac_device,
            dt=self._physics_dt,
            render_interval=max(
                1,
                int(round(self._control_dt / self._physics_dt)),
            ),
            gravity=(0.0, 0.0, -9.81),
            physx_kwargs={
                "solver_type": 1,
                "enable_ccd": False,
                "max_position_iteration_count": 64,
                "max_velocity_iteration_count": 8,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                **bounded_single_env_gpu_physx_kwargs(),
                "gpu_max_num_partitions": 8,
            },
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
            ),
        )
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view(
            eye=self._spec.head_eye,
            target=self._spec.head_target,
        )

        from .bimanual_assembly_scene import (
            _repair_imported_visual_materials,
            apply_reference_robot_appearance,
            randomize_task_scene_visuals,
        )
        from .task_preview import spawn_task_scene

        print(
            "[HandsOn/IsaacLab] Constructing E01 detached-Wuji collection scene...",
            flush=True,
        )
        task_mjcf, _ = spawn_task_scene(self.task_name, _TASK_ROOT, first_physx_safe=True)
        import omni.usd

        self._stage = omni.usd.get_context().get_stage()
        self._randomize_task_scene_visuals = randomize_task_scene_visuals
        self._configure_randomizable_table_physics()
        disabled_robot_prims = self._disable_imported_task_robots()
        disabled_center_prims = self._disable_imported_center_components()
        self._disable_imported_static_target_joints()
        self._spawn_task_specific_assets(sim_utils)
        sim_utils.update_stage()

        for side in ("right", "left"):
            prim_path = f"/World/HandsOnRobots/{side.capitalize()}Robot"
            self._robots[side] = Articulation(
                detached_wuji_hand_cfg(side, prim_path=prim_path)
            )
        sim_utils.update_stage()

        repaired_robot_materials = 0
        hidden_robot_colliders = 0
        self._reference_robot_visual_roots = tuple(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
            for side in ("right", "left")
        )

        for side in ("right", "left"):
            repaired, hidden = _repair_imported_visual_materials(
                self._stage,
                f"/World/HandsOnRobots/{side.capitalize()}Robot",
            )
            repaired_robot_materials += repaired
            hidden_robot_colliders += hidden
        reference_robot_appearance = apply_reference_robot_appearance(
            self._stage,
            robot_root_paths=self._reference_robot_visual_roots,
        )
        tuned_bodies, tuned_shapes = self._configure_task_collision_safety(
            sim_utils
        )
        contact_proxy_count = self._spawn_contact_proxy_colliders()
        hidden_camera_helpers = self._hide_camera_debug_helpers()
        if (
            repaired_robot_materials
            or hidden_robot_colliders
            or reference_robot_appearance
            or tuned_bodies
            or tuned_shapes
            or contact_proxy_count
            or hidden_camera_helpers
        ):
            sim_utils.update_stage()

        self._init_recording_cameras()
        if self._physx_ground_truth_enabled:
            self._configure_ground_truth_contact_reports(task_mjcf)
            sim_utils.update_stage()
        from .default_appearance import install_environment

        install_environment(self)
        self._sim.reset()
        self._sim_dt = float(self._sim.get_physics_dt())
        self._init_task_specific_runtime()
        self._capture_task_scene_state(task_mjcf)
        self._cache_joint_targets()
        self._sample_reset_state(clear_manual_request=True)
        self._reset_scene_state()
        self._apply_reset_visual_randomization()
        for _ in range(5):
            self._step_isaaclab()
        self._refresh_object_poses()
        self._capture_settled_garment_baseline()
        self._init_success_condition_overlay()
        self._update_success_condition_overlay()
        print(
            "[HandsOn/IsaacLab] E01 detached-Wuji runtime ready: "
            f"arms=0, online_ik=False, hands=2, hand_dofs=40, "
            f"physics_hz={1.0 / self._sim_dt:.0f}, "
            f"imported_robots_disabled={disabled_robot_prims}, "
            f"center_components_disabled={disabled_center_prims}, "
            f"materials_repaired={repaired_robot_materials}, "
            f"colliders_hidden={hidden_robot_colliders}, "
            f"ccd_bodies={tuned_bodies}, collision_shapes={tuned_shapes}, "
            f"contact_proxies={contact_proxy_count}",
            flush=True,
        )

    def _configure_ground_truth_contact_reports(self, task_mjcf: Any) -> None:
        """Include the live detached hands and proxies before the first reset.

        The shared task-name lookup only searches /World/Task. Detached hands
        and their contact proxies live outside that subtree, so a pure-cloth
        scene with a static worktable otherwise registers no rigid reporters.
        These reporters cover rigid contacts only; PhysX 5.1 does not expose
        deformable contact manifolds, which must not be fabricated here.
        """
        super()._configure_ground_truth_contact_reports(task_mjcf)
        from pxr import UsdPhysics

        paths = set(self._physx_contact_sensor_paths_preconfigured)
        roots = tuple(self._reference_robot_visual_roots)
        proxy_paths = set(self._contact_proxy_paths.values())
        for prim in self._stage.Traverse():
            path = str(prim.GetPath())
            if not (
                path in proxy_paths
                or any(path == root or path.startswith(root + "/") for root in roots)
            ):
                continue
            if not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            if enabled is False:
                continue
            paths.add(path)
        self._physx_contact_sensor_paths_preconfigured = tuple(sorted(paths))
        self._physx_contact_report_paths_preconfigured = (
            configure_physx_contact_reports_for_recording(self._stage, sorted(paths))
        )

    def _disable_imported_static_target_joints(self) -> int:
        """Remove MJCF helper joints that connect two non-rigid target Xforms."""

        if self._stage is None:
            return 0
        disabled = 0
        for side in ("right", "left"):
            for prefix in (f"{_TASK_ROOT}/Geometry", _TASK_ROOT):
                prim = self._stage.GetPrimAtPath(
                    f"{prefix}/target_{side}/PhysicsFixedJoint"
                )
                if prim.IsValid() and prim.IsActive():
                    prim.SetActive(False)
                    disabled += 1
        return disabled

    def _capture_settled_garment_baseline(self) -> bool:
        """Promote the first finite equilibrium to the reset baseline once.

        An imported or task-authored surface can be authored above its final
        table equilibrium, so the initial tensor state is not necessarily the
        settled state.  Capturing the already-settled simulation nodes avoids
        replaying that drop on every episode reset.  This updates only runtime
        tensor/tracking caches; it does not alter the USD mesh or PhysX rest
        shape.
        """

        if self._e_settled_baseline_captured:
            return False
        settle_timed_out = bool(
            self._e_failure_reason == "reset_nodal_state_did_not_settle"
            and self._e_reset_settle_timed_out
        )
        if (self._e_readback_failed or self._e_failure_reason is not None) and (
            not settle_timed_out
        ):
            raise RuntimeError(
                "Cannot capture E01 settled baseline after garment readback "
                f"failure: {self._e_failure_reason}"
            )
        if self._robots and not self._e_reset_settle_converged:
            if not settle_timed_out:
                raise RuntimeError(
                    "Cannot capture E01 settled baseline before reset nodal "
                    "velocity convergence"
                )
            print(
                "[HandsOn/IsaacLab] E01 reset did not fully settle before "
                "startup; using the finite timed-out garment state as the "
                "preview baseline.",
                flush=True,
            )

        nodes = np.asarray(self._e_nodes_np, dtype=np.float64)
        velocities = np.asarray(self._e_velocities_np, dtype=np.float64)
        issue, diagnostics = self._basic_nodal_state_issue(
            nodes,
            velocities,
            expected_node_count=self._e_expected_node_count,
        )
        if issue is not None:
            raise RuntimeError(
                "Cannot capture invalid E01 settled baseline: "
                f"reason={issue}, diagnostics={diagnostics}"
            )
        rms_speed, max_speed = self._nodal_speed_summary(velocities)
        moving_baseline = not self._nodal_speed_is_stable(rms_speed, max_speed)
        if moving_baseline and not settle_timed_out:
            raise RuntimeError(
                "Cannot capture a moving E01 reset baseline after warm-up: "
                f"rms_nodal_speed={rms_speed:.6f}, "
                f"max_nodal_speed={max_speed:.6f}"
            )
        if moving_baseline:
            print(
                "[HandsOn/IsaacLab] E01 reset baseline still has residual "
                f"motion after timeout: rms_nodal_speed={rms_speed:.6f}, "
                f"max_nodal_speed={max_speed:.6f}",
                flush=True,
            )

        self._e_base_nodes_np = nodes.copy()
        self._e_reset_nodes_np = nodes.copy()
        self._configure_garment_tracking(self._e_base_nodes_np)
        settle_diagnostics = (
            self._e_reset_settle_steps,
            self._e_reset_settle_seconds,
            self._e_reset_settle_rms_speed,
            self._e_reset_settle_max_speed,
            self._e_reset_settle_window_drift,
            self._e_reset_settle_stable_steps,
            self._e_reset_settle_converged,
            self._e_reset_settle_timed_out,
        )
        self._clear_garment_failure_state()
        (
            self._e_reset_settle_steps,
            self._e_reset_settle_seconds,
            self._e_reset_settle_rms_speed,
            self._e_reset_settle_max_speed,
            self._e_reset_settle_window_drift,
            self._e_reset_settle_stable_steps,
            self._e_reset_settle_converged,
            self._e_reset_settle_timed_out,
        ) = settle_diagnostics
        self._e_settled_baseline_captured = True
        baseline_label = "finite timed-out" if moving_baseline else "settled"
        print(
            "[HandsOn/IsaacLab] Captured first "
            f"{baseline_label} {self._SURFACE_LOGICAL_NAME} tensor state "
            "as the E01 reset baseline: "
            f"nodes={nodes.shape[0]}",
            flush=True,
        )
        return True

    def _disable_imported_center_components(self) -> int:
        """Remove the task MJCF's arm-center body from the hands-only view."""

        if self._stage is None:
            return 0
        # The MJCF converter can emit camera-helper fixed joints outside the
        # center body subtree.  Deactivating the body alone leaves those joints
        # with targets that resolve to inactive/non-existent prims, which makes
        # the PhysX schema plugin report an orphan joint during the first stage
        # update.  Disable only joints that explicitly target this intentionally
        # removed center component; the two live hand articulations and their
        # joints are untouched.  Keep this best-effort so interface-only tests
        # with a lightweight fake stage do not need PXR types.
        try:
            from pxr import Usd, UsdPhysics

            task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
            center_prefixes = {
                f"{_TASK_ROOT}/rokae_center_components",
            }
            if task_root.IsValid():
                for candidate in task_root.GetChildren():
                    if candidate.GetName().lower() == "rokae_center_components":
                        center_prefixes.add(str(candidate.GetPath()))
                for candidate in list(Usd.PrimRange(task_root)):
                    if not candidate.IsA(UsdPhysics.Joint):
                        continue
                    joint = UsdPhysics.Joint(candidate)
                    targets = (
                        list(joint.GetBody0Rel().GetTargets())
                        + list(joint.GetBody1Rel().GetTargets())
                    )
                    if any(
                        any(
                            str(target) == prefix
                            or str(target).startswith(prefix + "/")
                            for prefix in center_prefixes
                        )
                        for target in targets
                    ):
                        candidate.SetActive(False)
        except Exception:
            # The actual center component is still deactivated below.  A
            # missing optional PXR traversal must not break interface-only
            # construction or lightweight unit-test doubles.
            pass
        candidates = []
        direct = self._stage.GetPrimAtPath(
            f"{_TASK_ROOT}/rokae_center_components"
        )
        if direct.IsValid():
            candidates.append(direct)
        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        if task_root.IsValid():
            for prim in task_root.GetChildren():
                if prim.GetName().lower() == "rokae_center_components":
                    candidates.append(prim)
        disabled = 0
        seen: set[str] = set()
        for prim in candidates:
            path = str(prim.GetPath())
            if path in seen:
                continue
            seen.add(path)
            prim.SetActive(False)
            disabled += 1
        return disabled

    def _configure_task_collision_safety(
        self,
        sim_utils: Any,
    ) -> tuple[int, int]:
        """Tune only Wuji/garment contact; no arm bodies exist in this scene."""

        if self._stage is None:
            return 0, 0
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        roots = self._stage.GetPrimAtPath("/World/HandsOnRobots")
        if not roots.IsValid():
            return 0, 0
        for child in roots.GetChildren():
            sim_utils.make_uninstanceable(str(child.GetPath()), stage=self._stage)
        tuned_bodies = 0
        tuned_shapes = 0
        try:
            prim_range = Usd.PrimRange(roots, Usd.TraverseInstanceProxies())
        except (AttributeError, TypeError):
            prim_range = Usd.PrimRange(roots)
        for prim in prim_range:
            path = str(prim.GetPath()).lower()
            if "palm" not in path and "finger" not in path:
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
                rigid_api.CreateSolverPositionIterationCountAttr().Set(
                    E01_HAND_SOLVER_POSITION_ITERATIONS
                )
                rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                    E01_HAND_SOLVER_VELOCITY_ITERATIONS
                )
                rigid_api.CreateMaxDepenetrationVelocityAttr().Set(
                    E01_HAND_MAX_DEPENETRATION_VELOCITY
                )
                tuned_bodies += 1
            is_collision_geom = (
                prim.IsA(UsdGeom.Gprim)
                and (
                    "collision" in path
                    or "_col" in prim.GetName().lower()
                    or prim.GetName().lower().endswith("col")
                )
            )
            if prim.HasAPI(UsdPhysics.CollisionAPI) or is_collision_geom:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                physics_collision = UsdPhysics.CollisionAPI(prim)
                physics_collision.CreateCollisionEnabledAttr().Set(True)
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision_api.CreateContactOffsetAttr().Set(
                    E01_HAND_CONTACT_OFFSET
                )
                collision_api.CreateRestOffsetAttr().Set(E01_HAND_REST_OFFSET)
                tuned_shapes += 1
        if tuned_bodies > 0 and tuned_shapes == 0:
            print(
                "[HandsOn/IsaacLab] WARNING: E01 detached Wuji hand imported "
                "rigid bodies but no collision shapes. The garment will not "
                "respond to hand contact.",
                flush=True,
            )
        return tuned_bodies, tuned_shapes

    def _spawn_contact_proxy_colliders(self) -> int:
        """Author invisible kinematic pads used only for cloth contact."""

        if self._stage is None:
            return 0
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        UsdGeom.Xform.Define(self._stage, E01_CONTACT_PROXY_ROOT_PATH)
        authored = 0
        self._contact_proxy_paths = {}
        for side in ("right", "left"):
            path = f"{E01_CONTACT_PROXY_ROOT_PATH}/{side}_contact_proxy"
            proxy_pose = self._contact_proxy_pose(
                side,
                np.asarray(E01_DETACHED_HAND_HOME_POSES[side], dtype=np.float64),
                np.asarray(E01_WUJI_HAND_HOME, dtype=np.float64),
            )
            sphere = UsdGeom.Sphere.Define(self._stage, path)
            sphere.CreateRadiusAttr().Set(float(self._contact_proxy_radius))
            prim = sphere.GetPrim()
            prim.SetCustomDataByKey(
                "handson:classification", "collision_only_grasp_proxy"
            )
            prim.SetCustomDataByKey(
                "handson:collisionRole", "hand_contact_proxy"
            )
            prim.SetCustomDataByKey(
                "handson:visualContract", "computed_invisible"
            )
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*proxy_pose[:3]))
            UsdGeom.Imageable(prim).MakeInvisible()

            collision = UsdPhysics.CollisionAPI.Apply(prim)
            collision.CreateCollisionEnabledAttr().Set(True)
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateRigidBodyEnabledAttr().Set(True)
            rigid.CreateKinematicEnabledAttr().Set(True)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr().Set(E01_CONTACT_PROXY_MASS_KG)

            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr().Set(
                E01_CONTACT_PROXY_CONTACT_OFFSET_M
            )
            physx_collision.CreateRestOffsetAttr().Set(E01_CONTACT_PROXY_REST_OFFSET_M)
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rigid.CreateEnableSpeculativeCCDAttr().Set(True)
            physx_rigid.CreateSolverPositionIterationCountAttr().Set(
                E01_HAND_SOLVER_POSITION_ITERATIONS
            )
            physx_rigid.CreateSolverVelocityIterationCountAttr().Set(
                E01_HAND_SOLVER_VELOCITY_ITERATIONS
            )
            physx_rigid.CreateMaxDepenetrationVelocityAttr().Set(
                E01_HAND_MAX_DEPENETRATION_VELOCITY
            )
            filtered = UsdPhysics.FilteredPairsAPI.Apply(prim)
            filtered_rel = filtered.CreateFilteredPairsRel()
            for target in self._contact_proxy_filter_targets():
                filtered_rel.AddTarget(target)
            self._contact_proxy_paths[side] = path
            self._contact_proxy_last_pose[side] = proxy_pose.copy()
            authored += 1
        print(
            "[HandsOn/IsaacLab] E01 hand contact proxies authored: "
            f"count={authored}, radius_m={self._contact_proxy_radius:.3f}, "
            f"contact_offset_m={E01_CONTACT_PROXY_CONTACT_OFFSET_M:.4f}",
            flush=True,
        )
        return authored

    def _contact_proxy_filter_targets(self) -> tuple[Any, ...]:
        if self._stage is None:
            return ()
        targets = []
        for side in ("right", "left"):
            robot = self._stage.GetPrimAtPath(
                f"/World/HandsOnRobots/{side.capitalize()}Robot"
            )
            if robot.IsValid():
                targets.append(robot.GetPath())
        table = self._find_rigid_body_prim("table")
        if table is not None and table.IsValid():
            targets.append(table.GetPath())
        return tuple(targets)

    def _init_task_specific_runtime(self) -> None:
        super()._init_task_specific_runtime()
        self._bind_contact_proxy_views()

    def _bind_contact_proxy_views(self) -> None:
        self._contact_proxy_views = {}
        self._contact_proxy_indices = {}
        if self._torch is None or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            self._add_task_state_warning(
                "E01 hand contact proxy rigid-body view is unavailable"
            )
            return
        for side, path in self._contact_proxy_paths.items():
            try:
                view = physics_sim_view.create_rigid_body_view(path)
                if int(view.count) < 1:
                    self._add_task_state_warning(
                        f"E01 hand contact proxy has no PhysX body: {path}"
                    )
                    continue
                indices = self._surface_indices_like(
                    view.get_transforms(),
                    int(view.count),
                )
                self._contact_proxy_views[side] = view
                self._contact_proxy_indices[side] = indices
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not bind E01 {side} hand contact proxy {path}: {exc}"
                )

    def _contact_proxy_pose(
        self,
        side: str,
        pose: np.ndarray,
        hand: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        hand = np.asarray(hand, dtype=np.float64).reshape(E01_WUJI_HAND_DIM)
        guard = self._plan1_command_filter.collision_guards[side]
        table_top = E01_TABLE_TOP_Z + float(np.asarray(self.delta_h).reshape(-1)[0])
        minimum_z = guard.minimum_collision_z(pose, hand)
        guard_bottom_z = max(
            float(minimum_z),
            table_top + E01_CONTACT_PROXY_TABLE_CLEARANCE_M,
        )
        if self._contact_assist_hand_closed(hand):
            reference = self._contact_proxy_grasp_reference.get(side)
            if reference is None:
                reference = (float(pose[2]), float(guard_bottom_z))
                self._contact_proxy_grasp_reference[side] = reference
            reference_pose_z, reference_bottom_z = reference
            commanded_bottom_z = reference_bottom_z + max(
                float(pose[2]) - float(reference_pose_z),
                0.0,
            )
            bottom_z = max(float(guard_bottom_z), float(commanded_bottom_z))
        else:
            self._contact_proxy_grasp_reference[side] = None
            bottom_z = float(guard_bottom_z)
        return np.asarray(
            [
                pose[0],
                pose[1],
                bottom_z + float(self._contact_proxy_radius),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    def _contact_proxy_tensor(self, reference: Any, pose_wxyz: np.ndarray) -> Any:
        pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
        pose_xyzw = np.asarray(
            [pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[3]],
            dtype=np.float32,
        )
        return self._surface_array_like(
            reference,
            pose_xyzw.reshape(1, 7),
        )

    @staticmethod
    def _native_tensor_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "torch"):
            value = value.torch
        elif hasattr(value, "warp"):
            value = value.warp
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value).copy()

    def _write_contact_proxy_pose(
        self,
        side: str,
        pose: np.ndarray,
        hand: np.ndarray,
        *,
        reset: bool,
    ) -> None:
        view = self._contact_proxy_views.get(side)
        indices = self._contact_proxy_indices.get(side)
        if view is None or indices is None or self._torch is None:
            return
        proxy_pose = self._contact_proxy_pose(side, pose, hand)
        last_pose = self._contact_proxy_last_pose[side]
        try:
            transforms = view.get_transforms()
            payload = self._contact_proxy_tensor(transforms, proxy_pose)
            view.set_transforms(payload, indices)
            velocities = view.get_velocities()
            velocity_values = np.zeros_like(
                self._native_tensor_numpy(velocities)
            )
            if not reset:
                velocity_values[:, :3] = (
                    proxy_pose[:3] - last_pose[:3]
                ) / self._physics_dt
            velocity_payload = self._surface_array_like(
                velocities,
                velocity_values,
            )
            view.set_velocities(velocity_payload, indices)
            self._contact_proxy_last_pose[side] = proxy_pose.copy()
            if reset:
                self._cloth_contact_assist_last_proxy_pose[side] = proxy_pose.copy()
                self._cloth_contact_assist_grasp_active[side] = False
                self._cloth_contact_assist_active_nodes[side] = 0
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not update E01 {side} hand contact proxy: {exc}"
            )

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        self._apply_cloth_contact_assist()

    def _prepare_cloth_contact_assist(
        self,
        nodes: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Let a task inspect the raw post-PhysX surface state before assist.

        The default E01 runtime keeps the historical proximity-based recovery
        behavior.  A task with stricter contact semantics can override this
        hook and cache a per-side gate.  It is called after the surface tensor
        has been read back from ``SimulationContext.step`` and before any
        velocity is written by :meth:`_apply_cloth_contact_assist`, so a task
        can distinguish solver response from its own assistance.
        """

        return None

    def _cloth_contact_assist_allowed(
        self,
        side: str,
        *,
        nodes: np.ndarray,
        velocities: np.ndarray,
        indices: np.ndarray,
        proxy_pose: np.ndarray,
        previous_proxy_pose: np.ndarray,
        contact_active: bool,
        hand_closed: bool,
        grasp_candidate: bool,
    ) -> bool:
        """Return whether this side may write the local cloth assist state.

        This is intentionally permissive for the shared E01/F01 runtime.  A
        stricter task may require a prior physical response while retaining
        the same bounded assist implementation.
        """

        return True

    @staticmethod
    def _contact_assist_weights(distance: np.ndarray, radius: float) -> np.ndarray:
        normalized = np.clip(
            np.asarray(distance, dtype=np.float64) / max(float(radius), 1e-9),
            0.0,
            1.0,
        )
        return 0.5 * (1.0 + np.cos(math.pi * normalized))

    @staticmethod
    def _contact_assist_hand_closed(hand: np.ndarray) -> bool:
        hand = np.asarray(hand, dtype=np.float64).reshape(E01_WUJI_HAND_DIM)
        home = np.asarray(E01_WUJI_HAND_HOME, dtype=np.float64)
        return bool(
            np.all(np.isfinite(hand))
            and np.max(np.abs(hand - home))
            >= E01_CLOTH_CONTACT_ASSIST_GRASP_JOINT_DELTA_RAD
        )

    def _contact_assist_grasp_commanded(self, side: str) -> bool:
        candidates = [
            self._detached_sim_hand.get(side),
            self._detached_motion_target_hand.get(side),
            self._plan1_command_filter.hands.get(side),
            self._right_hand if side == "right" else self._left_hand,
        ]
        for hand in candidates:
            if hand is not None and self._contact_assist_hand_closed(hand):
                return True
        return False

    @staticmethod
    def _clip_contact_assist_displacement(displacement: np.ndarray) -> np.ndarray:
        result = np.asarray(displacement, dtype=np.float64).copy()
        norms = np.linalg.norm(result, axis=1)
        limited = norms > E01_CLOTH_CONTACT_ASSIST_MAX_NODE_STEP_M
        if np.any(limited):
            result[limited] *= (
                E01_CLOTH_CONTACT_ASSIST_MAX_NODE_STEP_M
                / np.maximum(norms[limited], 1e-12)
            )[:, None]
        return result

    def _apply_cloth_contact_assist(self) -> None:
        """Apply a local, speed-limited cloth response when PhysX misses contact."""

        if (
            self._cloth_contact_assist_disabled
            or self._e_garment_view is None
            or self._e_garment_indices is None
            or self._e_readback_failed
        ):
            return
        # Read the post-PhysX surface frame directly here.  The shared
        # observation refresh happens at the control boundary, while this
        # hook runs after every physics substep; using the cached control
        # frame would make a recovery write (and BC025's contact witness)
        # depend on stale cloth coordinates.
        try:
            position_reference = (
                self._e_garment_view.get_simulation_nodal_positions()
            )
            velocity_reference = (
                self._e_garment_view.get_simulation_nodal_velocities()
            )
            nodes = self._surface_tensor_body(
                position_reference,
                label="contact_assist_nodal_position",
            )
            velocities = self._surface_tensor_body(
                velocity_reference,
                label="contact_assist_nodal_velocity",
            )
        except Exception as exc:
            self._reject_garment_readback(
                "cloth_assist_surface_tensor_readback_error",
                {"error": str(exc)},
            )
            return
        issue, diagnostics = self._basic_nodal_state_issue(
            nodes,
            velocities,
            expected_node_count=self._e_expected_node_count,
        )
        if issue is not None:
            self._reject_garment_readback(issue, diagnostics)
            return

        dt = float(getattr(self, "_sim_dt", self._physics_dt))
        if not math.isfinite(dt) or dt <= 0.0:
            return
        self._prepare_cloth_contact_assist(nodes, velocities)
        table_floor = (
            E01_TABLE_TOP_Z
            + float(np.asarray(self.delta_h).reshape(-1)[0])
            + E01_CONTACT_PROXY_TABLE_CLEARANCE_M
        )
        any_moved = False
        radius = E01_CLOTH_CONTACT_ASSIST_RADIUS_M
        for side in ("right", "left"):
            self._cloth_contact_assist_active_nodes[side] = 0
            proxy_pose = self._contact_proxy_last_pose[side]
            previous_proxy_pose = self._cloth_contact_assist_last_proxy_pose[side]
            self._cloth_contact_assist_last_proxy_pose[side] = proxy_pose.copy()
            if not np.all(np.isfinite(proxy_pose)):
                self._cloth_contact_assist_grasp_active[side] = False
                continue
            delta_xy = proxy_pose[:2] - previous_proxy_pose[:2]
            distance = np.linalg.norm(nodes[:, :2] - proxy_pose[:2], axis=1)
            indices = np.flatnonzero(distance <= radius)
            if indices.size == 0:
                if not self._contact_assist_grasp_commanded(side):
                    self._cloth_contact_assist_grasp_active[side] = False
                continue

            local_z = nodes[indices, 2]
            proxy_bottom_z = float(proxy_pose[2] - self._contact_proxy_radius)
            local_contact_z = float(np.percentile(local_z, 75.0))
            contact_active = (
                proxy_bottom_z
                <= local_contact_z + E01_CLOTH_CONTACT_ASSIST_MARGIN_M
            )
            hand_closed = self._contact_assist_grasp_commanded(side)
            grasp_candidate = (
                contact_active
                or proxy_bottom_z
                <= local_contact_z + E01_CLOTH_CONTACT_ASSIST_GRASP_MARGIN_M
            )
            if not self._cloth_contact_assist_allowed(
                side,
                nodes=nodes,
                velocities=velocities,
                indices=indices,
                proxy_pose=proxy_pose,
                previous_proxy_pose=previous_proxy_pose,
                contact_active=contact_active,
                hand_closed=hand_closed,
                grasp_candidate=grasp_candidate,
            ):
                self._cloth_contact_assist_grasp_active[side] = False
                continue
            if hand_closed and grasp_candidate:
                self._cloth_contact_assist_grasp_active[side] = True
            elif not hand_closed:
                self._cloth_contact_assist_grasp_active[side] = False
            grasp_active = self._cloth_contact_assist_grasp_active[side]
            if not (contact_active or grasp_active):
                continue

            weights = self._contact_assist_weights(distance[indices], radius)
            target_z = proxy_bottom_z + E01_CLOTH_CONTACT_ASSIST_SURFACE_OFFSET_M
            if grasp_active:
                z_target = np.full(indices.shape, target_z, dtype=np.float64)
            else:
                z_target = np.minimum(local_z, target_z)
            dz = (
                (z_target - local_z)
                * weights
                * E01_CLOTH_CONTACT_ASSIST_Z_GAIN
            )
            if grasp_active:
                dz += (
                    max(float(proxy_pose[2] - previous_proxy_pose[2]), 0.0)
                    * weights
                    * E01_CLOTH_CONTACT_ASSIST_LIFT_GAIN
                )
            dxy = (
                delta_xy.reshape(1, 2)
                * weights.reshape(-1, 1)
                * E01_CLOTH_CONTACT_ASSIST_XY_GAIN
            )
            displacement = np.column_stack([dxy[:, 0], dxy[:, 1], dz])
            displacement = self._clip_contact_assist_displacement(displacement)
            if not np.any(np.abs(displacement) > 1e-9):
                continue
            next_z = nodes[indices, 2] + displacement[:, 2]
            below_table = next_z < table_floor
            if np.any(below_table):
                displacement[below_table, 2] = (
                    table_floor - nodes[indices[below_table], 2]
                )
            velocities[indices] = displacement / dt
            self._cloth_contact_assist_active_nodes[side] = int(
                np.count_nonzero(np.linalg.norm(displacement, axis=1) > 1e-9)
            )
            any_moved = True

        if not any_moved:
            # Keep the next substep's baseline tied to the actual PhysX frame
            # even when the task is not currently eligible to assist.
            self._accept_garment_readback(nodes, velocities, diagnostics)
            return

        issue, diagnostics = self._dynamic_nodal_state_issue(nodes, velocities)
        if issue is not None:
            self._reject_garment_readback(issue, diagnostics)
            return
        try:
            velocity_reference = (
                self._e_garment_view.get_simulation_nodal_velocities()
            )
            self._e_garment_view.set_simulation_nodal_velocities(
                self._surface_array_like(velocity_reference, velocities),
                self._e_garment_indices,
            )
        except Exception as exc:
            self._cloth_contact_assist_disabled = True
            self._add_task_state_warning(
                f"E01 cloth contact assist disabled after velocity write failure: {exc}"
            )
            return
        self._accept_garment_readback(nodes, velocities, diagnostics)

    # ------------------------------------------------------------------
    # Exact Plan1 command boundary and hand-only actuation
    # ------------------------------------------------------------------

    def _sync_plan1_command_state(self) -> None:
        command_filter = self._plan1_command_filter
        self._right_tcp_pose = command_filter.poses["right"].copy()
        self._left_tcp_pose = command_filter.poses["left"].copy()
        self._right_hand = command_filter.hands["right"].copy()
        self._left_hand = command_filter.hands["left"].copy()
        self._right_pose_commanded = False
        self._left_pose_commanded = False

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        visual_randomization = bool(self.randomize)
        self.randomize = False
        try:
            super()._sample_reset_state(clear_manual_request=clear_manual_request)
        finally:
            self.randomize = visual_randomization
        self._plan1_command_filter.reset()
        self._sync_plan1_command_state()
        self._detached_motion_start_pose = {
            "right": self._right_tcp_pose.copy(),
            "left": self._left_tcp_pose.copy(),
        }
        self._detached_motion_target_pose = {
            side: pose.copy()
            for side, pose in self._detached_motion_start_pose.items()
        }
        self._detached_motion_start_hand = {
            "right": self._right_hand.copy(),
            "left": self._left_hand.copy(),
        }
        self._detached_motion_target_hand = {
            side: hand.copy()
            for side, hand in self._detached_motion_start_hand.items()
        }
        self._detached_sim_pose = {
            side: pose.copy()
            for side, pose in self._detached_motion_start_pose.items()
        }
        self._detached_sim_hand = {
            side: hand.copy()
            for side, hand in self._detached_motion_start_hand.items()
        }
        self._detached_sim_velocity = {
            "right": np.zeros(6, dtype=np.float64),
            "left": np.zeros(6, dtype=np.float64),
        }
        for side in ("right", "left"):
            self._palm_tracking_translation_error[side] = 0.0
            self._palm_tracking_rotation_error[side] = 0.0
            self._palm_tracking_translation_saturated[side] = False
            self._palm_tracking_rotation_saturated[side] = False
            self._palm_tracking_saturation_substeps[side] = 0
            self._palm_force_saturated[side] = False
            self._palm_torque_saturated[side] = False
            self._finger_tracking_error[side] = 0.0
            self._finger_tracking_saturated[side] = False
            self._contact_proxy_grasp_reference[side] = None
            self._cloth_contact_assist_grasp_active[side] = False
            self._cloth_contact_assist_active_nodes[side] = 0
        self._detached_motion_substep = 0

    def _apply_action(self, action: Any) -> None:
        if action is None:
            return
        self._plan1_command_filter.apply(
            action,
            table_top_z=0.89 + float(self.delta_h[0]),
        )
        self._right_tcp_pose = self._plan1_command_filter.poses["right"].copy()
        self._left_tcp_pose = self._plan1_command_filter.poses["left"].copy()
        self._right_hand = self._plan1_command_filter.hands["right"].copy()
        self._left_hand = self._plan1_command_filter.hands["left"].copy()
        self._right_pose_commanded = True
        self._left_pose_commanded = True

    def _parse_bimanual_hand_angles(
        self,
        action: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expose the canonical 20+20 hand contract to policy/teleop wrappers."""

        right = np.asarray(
            action.get("right", np.zeros(7 + E01_WUJI_HAND_DIM)),
            dtype=np.float64,
        ).ravel()
        left = np.asarray(
            action.get("left", np.zeros(7 + E01_WUJI_HAND_DIM)),
            dtype=np.float64,
        ).ravel()
        return (
            right[7 : 7 + E01_WUJI_HAND_DIM],
            left[7 : 7 + E01_WUJI_HAND_DIM],
        )

    def _cache_joint_targets(self) -> None:
        self._joint_targets = {}
        self._hand_joint_ids = {"right": [], "left": []}
        self._arm_joint_ids = {}
        self._ee_body_indices = {}
        self._ee_jacobi_indices = {}
        self._ik_controllers = {}
        for side in ("right", "left"):
            robot = self._control_robot(side)
            joint_ids: list[int] = []
            for finger in range(1, 6):
                for joint in range(1, 5):
                    name = f"{side}_finger{finger}_joint{joint}"
                    matches = robot.find_joints(name)[0]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"Expected one detached Wuji joint {name}, got {matches}"
                        )
                    joint_ids.append(int(matches[0]))
            if len(joint_ids) != E01_WUJI_HAND_DIM:
                raise RuntimeError(
                    f"Expected {E01_WUJI_HAND_DIM} {side} hand joints, "
                    f"got {len(joint_ids)}"
                )
            palm_matches = robot.find_bodies(f"{side}_palm_link")[0]
            if len(palm_matches) != 1:
                raise RuntimeError(
                    f"Expected one detached {side} palm, got {palm_matches}"
                )
            self._hand_joint_ids[side] = joint_ids
            self._ee_body_indices[side] = int(palm_matches[0])
            self._joint_targets[side] = self._torch_tensor(
                robot.data.default_joint_pos
            ).clone()
            print(
                f"[HandsOn/IsaacLab] {side} detached control map: "
                f"arm_joints=0, hand_joints={len(joint_ids)}, "
                f"root={robot.body_names[int(palm_matches[0])]}",
                flush=True,
            )

    def _reset_robot_control_state(self) -> None:
        if self._torch is None:
            return
        for side in ("right", "left"):
            robot = self._control_robot(side)
            # A variable PD wrench is submitted through the instantaneous
            # composer below.  Clear both public composers before the one
            # reset-only state write so no pre-reset wrench can leak into the
            # next episode.
            robot.instantaneous_wrench_composer.reset()
            robot.permanent_wrench_composer.reset()
            pose = np.asarray(
                E01_DETACHED_HAND_HOME_POSES[side],
                dtype=np.float64,
            )
            isaaclab_pose = self._pose_wxyz_to_isaaclab(
                pose,
                uses_xyzw=self._isaaclab_quaternions_are_xyzw,
            )
            hand = np.asarray(E01_WUJI_HAND_HOME, dtype=np.float64)
            root_pose_tensor = self._torch_tensor(robot.data.root_pose_w)
            pose_tensor = self._torch.as_tensor(
                isaaclab_pose.reshape(1, 7),
                dtype=root_pose_tensor.dtype,
                device=robot.device,
            )
            robot.write_root_pose_to_sim(pose_tensor)
            robot.write_root_link_velocity_to_sim(
                self._torch.zeros(
                    (1, 6),
                    dtype=root_pose_tensor.dtype,
                    device=robot.device,
                )
            )
            joint_pos = self._torch_tensor(robot.data.default_joint_pos).clone()
            joint_vel = self._torch.zeros_like(joint_pos)
            joint_ids = self._hand_joint_ids[side]
            joint_pos[:, joint_ids] = self._torch.as_tensor(
                hand,
                dtype=joint_pos.dtype,
                device=joint_pos.device,
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            self._joint_targets[side] = joint_pos.clone()
            self._detached_sim_pose[side] = pose.copy()
            self._detached_sim_hand[side] = hand.copy()
            self._detached_sim_velocity[side] = np.zeros(6, dtype=np.float64)
            self._write_contact_proxy_pose(
                side,
                pose,
                hand,
                reset=True,
            )

    def _apply_joint_targets(self) -> None:
        if self._robots:
            self._refresh_detached_hand_readback()
        self._detached_motion_start_pose = {
            side: self._detached_sim_pose[side].copy()
            for side in ("right", "left")
        }
        self._detached_motion_target_pose = {
            "right": self._right_tcp_pose.copy(),
            "left": self._left_tcp_pose.copy(),
        }
        self._detached_motion_start_hand = {
            side: self._detached_sim_hand[side].copy()
            for side in ("right", "left")
        }
        requested_hands = {
            "right": self._right_hand,
            "left": self._left_hand,
        }
        self._detached_motion_target_hand = {}
        for side in ("right", "left"):
            bounded, tracking_error, saturated = (
                self._bound_hand_tracking_target(
                    self._detached_sim_hand[side],
                    requested_hands[side],
                    max_tracking_error=E01_FINGER_MAX_TRACKING_ERROR_RAD,
                )
            )
            self._detached_motion_target_hand[side] = bounded
            self._finger_tracking_error[side] = tracking_error
            self._finger_tracking_saturated[side] = saturated
        self._detached_motion_substep = 0

    @staticmethod
    def _interpolate_pose(
        start: np.ndarray,
        target: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        pose = np.empty(7, dtype=np.float64)
        pose[:3] = (1.0 - fraction) * start[:3] + fraction * target[:3]
        pose[3:7] = _blend_quat_wxyz(
            start[3:7],
            target[3:7],
            fraction,
        )
        return pose

    @staticmethod
    def _torch_tensor(value: Any) -> Any:
        """Return a torch view for both IsaacLab 2 tensors and 3 ProxyArray."""

        return value.torch if hasattr(value, "torch") else value

    @staticmethod
    def _pose_wxyz_to_isaaclab(
        pose: np.ndarray,
        *,
        uses_xyzw: bool,
    ) -> np.ndarray:
        result = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        if uses_xyzw:
            w, x, y, z = result[3:7].copy()
            result[3:7] = (x, y, z, w)
        return result

    @staticmethod
    def _pose_isaaclab_to_wxyz(
        pose: np.ndarray,
        *,
        uses_xyzw: bool,
    ) -> np.ndarray:
        result = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        if uses_xyzw:
            x, y, z, w = result[3:7].copy()
            result[3:7] = (w, x, y, z)
        return result

    @staticmethod
    def _bound_hand_tracking_target(
        actual_hand: np.ndarray,
        requested_hand: np.ndarray,
        *,
        max_tracking_error: float,
    ) -> tuple[np.ndarray, float, bool]:
        """Bound every finger drive target around finite PhysX readback."""

        if (
            not math.isfinite(float(max_tracking_error))
            or float(max_tracking_error) <= 0.0
        ):
            raise ValueError(
                "Finger tracking limit must be finite and positive: "
                f"{max_tracking_error}"
            )
        actual = np.asarray(actual_hand, dtype=np.float64).reshape(
            E01_WUJI_HAND_DIM
        )
        requested = np.asarray(requested_hand, dtype=np.float64).reshape(
            E01_WUJI_HAND_DIM
        )
        if not np.all(np.isfinite(actual)):
            # There is no safe frame around which to form a new drive target.
            return np.asarray(E01_WUJI_HAND_HOME, dtype=np.float64), math.inf, True
        if not np.all(np.isfinite(requested)):
            return actual.copy(), math.inf, True
        delta = requested - actual
        tracking_error = float(np.max(np.abs(delta)))
        bounded = actual + np.clip(
            delta,
            -float(max_tracking_error),
            float(max_tracking_error),
        )
        return (
            bounded,
            tracking_error,
            tracking_error > float(max_tracking_error),
        )

    def _write_detached_hand_substep(self) -> None:
        if self._torch is None:
            return
        substeps = max(1, int(self._n_substeps))
        fraction_start = min(
            1.0,
            self._detached_motion_substep / substeps,
        )
        fraction_end = min(
            1.0,
            (self._detached_motion_substep + 1) / substeps,
        )
        for side in ("right", "left"):
            robot = self._control_robot(side)
            previous_requested_pose = self._interpolate_pose(
                self._detached_motion_start_pose[side],
                self._detached_motion_target_pose[side],
                fraction_start,
            )
            requested_pose = self._interpolate_pose(
                self._detached_motion_start_pose[side],
                self._detached_motion_target_pose[side],
                fraction_end,
            )
            requested_hand = (
                (1.0 - fraction_end) * self._detached_motion_start_hand[side]
                + fraction_end * self._detached_motion_target_hand[side]
            )
            # Re-anchor at every PhysX readback too.  External contact may
            # push a finger away from the control-frame start after the first
            # substep, and the drive must remain bounded in that case as well.
            desired_hand, substep_hand_error, substep_hand_saturated = (
                self._bound_hand_tracking_target(
                    self._detached_sim_hand[side],
                    requested_hand,
                    max_tracking_error=E01_FINGER_MAX_TRACKING_ERROR_RAD,
                )
            )
            self._finger_tracking_error[side] = max(
                self._finger_tracking_error[side],
                substep_hand_error,
            )
            self._finger_tracking_saturated[side] = bool(
                self._finger_tracking_saturated[side]
                or substep_hand_saturated
            )
            actual_pose = self._detached_sim_pose[side]
            (
                previous_desired_pose,
                _previous_translation_error,
                _previous_rotation_error,
                _previous_translation_saturated,
                _previous_rotation_saturated,
            ) = self._bound_pose_tracking_error(
                actual_pose,
                previous_requested_pose,
                max_translation_error=(
                    E01_PALM_MAX_TRACKING_TRANSLATION_ERROR_M
                ),
                max_rotation_error=E01_PALM_MAX_TRACKING_ROTATION_ERROR_RAD,
            )
            (
                desired_pose,
                translation_error,
                rotation_error,
                translation_saturated,
                rotation_saturated,
            ) = self._bound_pose_tracking_error(
                actual_pose,
                requested_pose,
                max_translation_error=(
                    E01_PALM_MAX_TRACKING_TRANSLATION_ERROR_M
                ),
                max_rotation_error=E01_PALM_MAX_TRACKING_ROTATION_ERROR_RAD,
            )
            desired_velocity = self._bounded_pose_feedforward(
                previous_desired_pose,
                desired_pose,
                self._physics_dt,
                max_linear_speed=(
                    self._contact_max_tcp_translation_step / self._control_dt
                ),
                max_angular_speed=(
                    self._contact_max_tcp_rotation_step / self._control_dt
                ),
            )
            force_w, torque_w = self._bounded_pd_wrench(
                actual_pose=actual_pose,
                actual_velocity=self._detached_sim_velocity[side],
                desired_pose=desired_pose,
                desired_velocity=desired_velocity,
                position_kp=E01_PALM_POSITION_KP_N_PER_M,
                linear_velocity_kd=E01_PALM_LINEAR_VELOCITY_KD_N_S_PER_M,
                orientation_kp=E01_PALM_ORIENTATION_KP_NM_PER_RAD,
                angular_velocity_kd=(
                    E01_PALM_ANGULAR_VELOCITY_KD_NM_S_PER_RAD
                ),
                max_force=E01_PALM_MAX_FORCE_N,
                max_torque=E01_PALM_MAX_TORQUE_NM,
            )
            self._palm_tracking_translation_error[side] = translation_error
            self._palm_tracking_rotation_error[side] = rotation_error
            self._palm_tracking_translation_saturated[side] = (
                translation_saturated
            )
            self._palm_tracking_rotation_saturated[side] = rotation_saturated
            if translation_saturated or rotation_saturated:
                self._palm_tracking_saturation_substeps[side] += 1
            else:
                self._palm_tracking_saturation_substeps[side] = 0
            self._palm_force_saturated[side] = bool(
                np.linalg.norm(force_w) >= E01_PALM_MAX_FORCE_N * (1.0 - 1e-9)
            )
            self._palm_torque_saturated[side] = bool(
                np.linalg.norm(torque_w)
                >= E01_PALM_MAX_TORQUE_NM * (1.0 - 1e-9)
            )
            self._set_palm_world_wrench(
                robot,
                body_index=self._ee_body_indices[side],
                force_w=force_w,
                torque_w=torque_w,
            )
            self._write_contact_proxy_pose(
                side,
                requested_pose,
                requested_hand,
                reset=False,
            )
            # Finger links remain ordinary driven articulation links.  Runtime
            # state writes would teleport their collision shapes and defeat
            # continuous collision detection, so only the drive target changes
            # after reset.
            joint_target = self._joint_targets[side].clone()
            joint_ids = self._hand_joint_ids[side]
            joint_target[:, joint_ids] = self._torch.as_tensor(
                desired_hand,
                dtype=joint_target.dtype,
                device=joint_target.device,
            )
            robot.set_joint_position_target(joint_target)
            self._joint_targets[side] = joint_target
        self._detached_motion_substep += 1

    @staticmethod
    def _clamp_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        """Return a finite vector whose Euclidean norm does not exceed a cap."""

        result = np.asarray(vector, dtype=np.float64).reshape(3).copy()
        if not np.all(np.isfinite(result)):
            return np.zeros(3, dtype=np.float64)
        if not np.isfinite(max_norm) or max_norm <= 0.0:
            raise ValueError(f"Vector norm limit must be finite and positive: {max_norm}")
        norm = float(np.linalg.norm(result))
        if norm > max_norm:
            result *= max_norm / norm
        return result

    @classmethod
    def _bound_pose_tracking_error(
        cls,
        actual_pose: np.ndarray,
        requested_pose: np.ndarray,
        *,
        max_translation_error: float,
        max_rotation_error: float,
    ) -> tuple[np.ndarray, float, float, bool, bool]:
        """Bound requested pose error around actual palm link feedback."""

        actual = np.asarray(actual_pose, dtype=np.float64).reshape(7)
        requested = np.asarray(requested_pose, dtype=np.float64).reshape(7)
        cls._clamp_vector_norm(np.zeros(3), max_translation_error)
        cls._clamp_vector_norm(np.zeros(3), max_rotation_error)
        if (
            not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(requested))
            or float(np.linalg.norm(actual[3:7])) <= 1e-12
            or float(np.linalg.norm(requested[3:7])) <= 1e-12
        ):
            return actual.copy(), math.inf, math.inf, True, True

        bounded = requested.copy()
        translation_delta = requested[:3] - actual[:3]
        translation_error = float(np.linalg.norm(translation_delta))
        translation_saturated = translation_error > max_translation_error
        if translation_saturated:
            bounded[:3] = actual[:3] + cls._clamp_vector_norm(
                translation_delta,
                max_translation_error,
            )

        rotation_delta = cls._angular_velocity(
            actual[3:7],
            requested[3:7],
            1.0,
        )
        rotation_error = float(np.linalg.norm(rotation_delta))
        rotation_saturated = rotation_error > max_rotation_error
        if rotation_saturated:
            bounded[3:7] = cls._slerp_quat_wxyz(
                actual[3:7],
                requested[3:7],
                max_rotation_error / rotation_error,
            )
        else:
            bounded[3:7] /= float(np.linalg.norm(bounded[3:7]))
        return (
            bounded,
            translation_error,
            rotation_error,
            translation_saturated,
            rotation_saturated,
        )

    @staticmethod
    def _slerp_quat_wxyz(
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        """Shortest-path quaternion interpolation with exact angular fraction."""

        q0 = np.asarray(start_quat, dtype=np.float64).reshape(4).copy()
        q1 = np.asarray(end_quat, dtype=np.float64).reshape(4).copy()
        q0 /= float(np.linalg.norm(q0))
        q1 /= float(np.linalg.norm(q1))
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        fraction = float(np.clip(fraction, 0.0, 1.0))
        half_angle = math.acos(dot)
        if half_angle <= 1e-12:
            return q0
        sine = math.sin(half_angle)
        result = (
            math.sin((1.0 - fraction) * half_angle) / sine * q0
            + math.sin(fraction * half_angle) / sine * q1
        )
        return result / float(np.linalg.norm(result))

    @classmethod
    def _bounded_pose_feedforward(
        cls,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        dt: float,
        *,
        max_linear_speed: float,
        max_angular_speed: float,
    ) -> np.ndarray:
        """Compute a finite, guard-bounded world-frame desired velocity."""

        start = np.asarray(start_pose, dtype=np.float64).reshape(7)
        end = np.asarray(end_pose, dtype=np.float64).reshape(7)
        if (
            not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or not np.isfinite(dt)
            or dt <= 0.0
            or float(np.linalg.norm(start[3:7])) <= 1e-12
            or float(np.linalg.norm(end[3:7])) <= 1e-12
        ):
            return np.zeros(6, dtype=np.float64)
        linear = cls._clamp_vector_norm(
            (end[:3] - start[:3]) / dt,
            max_linear_speed,
        )
        angular = cls._clamp_vector_norm(
            cls._angular_velocity(start[3:7], end[3:7], dt),
            max_angular_speed,
        )
        return np.concatenate([linear, angular])

    @classmethod
    def _bounded_pd_wrench(
        cls,
        *,
        actual_pose: np.ndarray,
        actual_velocity: np.ndarray,
        desired_pose: np.ndarray,
        desired_velocity: np.ndarray,
        position_kp: float,
        linear_velocity_kd: float,
        orientation_kp: float,
        angular_velocity_kd: float,
        max_force: float,
        max_torque: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a norm-limited world-frame PD wrench for the palm link.

        Invalid simulation readback yields a zero wrench instead of allowing a
        NaN or stale command into PhysX.  Configuration errors remain loud.
        """

        gains = np.asarray(
            [
                position_kp,
                linear_velocity_kd,
                orientation_kp,
                angular_velocity_kd,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
            raise ValueError(f"PD gains must be finite and non-negative: {gains}")
        # Validate the force caps even when state is invalid.  This prevents a
        # bad deployment configuration from being hidden by a transient read.
        cls._clamp_vector_norm(np.zeros(3), max_force)
        cls._clamp_vector_norm(np.zeros(3), max_torque)

        actual_pose = np.asarray(actual_pose, dtype=np.float64).reshape(7)
        desired_pose = np.asarray(desired_pose, dtype=np.float64).reshape(7)
        actual_velocity = np.asarray(actual_velocity, dtype=np.float64).reshape(6)
        desired_velocity = np.asarray(desired_velocity, dtype=np.float64).reshape(6)
        state = np.concatenate(
            [actual_pose, desired_pose, actual_velocity, desired_velocity]
        )
        if (
            not np.all(np.isfinite(state))
            or float(np.linalg.norm(actual_pose[3:7])) <= 1e-12
            or float(np.linalg.norm(desired_pose[3:7])) <= 1e-12
        ):
            return (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

        position_error = desired_pose[:3] - actual_pose[:3]
        orientation_error = cls._angular_velocity(
            actual_pose[3:7],
            desired_pose[3:7],
            1.0,
        )
        force_w = (
            position_kp * position_error
            + linear_velocity_kd
            * (desired_velocity[:3] - actual_velocity[:3])
        )
        torque_w = (
            orientation_kp * orientation_error
            + angular_velocity_kd
            * (desired_velocity[3:6] - actual_velocity[3:6])
        )
        return (
            cls._clamp_vector_norm(force_w, max_force),
            cls._clamp_vector_norm(torque_w, max_torque),
        )

    def _set_palm_world_wrench(
        self,
        robot: Any,
        *,
        body_index: int,
        force_w: np.ndarray,
        torque_w: np.ndarray,
    ) -> None:
        """Submit one-step force/torque only to the root palm body.

        IsaacLab 2.3.2's instantaneous composer is the public API for a
        changing wrench and clears itself after ``write_data_to_sim``.  That
        fail-safe lifetime is preferable to carrying the last PD wrench across
        a skipped controller update.
        """

        root_link_pose = self._torch_tensor(robot.data.root_link_pose_w)
        force = self._torch.as_tensor(
            np.asarray(force_w, dtype=np.float64).reshape(1, 1, 3),
            dtype=root_link_pose.dtype,
            device=robot.device,
        )
        torque = self._torch.as_tensor(
            np.asarray(torque_w, dtype=np.float64).reshape(1, 1, 3),
            dtype=root_link_pose.dtype,
            device=robot.device,
        )
        robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=force,
            torques=torque,
            body_ids=[int(body_index)],
            is_global=True,
        )

    @staticmethod
    def _angular_velocity(
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """World angular velocity taking ``start`` to ``end`` in one substep."""

        q0 = np.asarray(start_quat, dtype=np.float64)
        q1 = np.asarray(end_quat, dtype=np.float64)
        q0 /= max(float(np.linalg.norm(q0)), 1e-12)
        q1 /= max(float(np.linalg.norm(q1)), 1e-12)
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        # q_delta = q1 * conjugate(q0), expressed in the world frame.
        delta = np.asarray(
            [
                w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0,
                -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
                -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
                -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0,
            ],
            dtype=np.float64,
        )
        delta /= max(float(np.linalg.norm(delta)), 1e-12)
        if delta[0] < 0.0:
            delta = -delta
        vector_norm = float(np.linalg.norm(delta[1:]))
        if vector_norm < 1e-12 or dt <= 0.0:
            return np.zeros(3, dtype=np.float64)
        angle = 2.0 * math.atan2(vector_norm, float(delta[0]))
        return delta[1:] * (angle / (vector_norm * dt))

    def _step_isaaclab(self) -> None:
        if self._sim is None:
            return
        self._resume_timeline_if_paused(source="Timeline pause")
        self._write_detached_hand_substep()
        for robot in self._robots.values():
            robot.write_data_to_sim()
        self._sim.step(render=False)
        for robot in self._robots.values():
            robot.update(self._sim_dt)
        self._refresh_detached_hand_readback()
        self._after_task_physics_substep()

    def _refresh_detached_hand_readback(self) -> None:
        """Cache actual PhysX link pose/velocity and finger state."""

        for side in ("right", "left"):
            robot = self._robots.get(side)
            body_index = self._ee_body_indices.get(side)
            joint_ids = self._hand_joint_ids.get(side, [])
            if robot is None or body_index is None or not joint_ids:
                continue
            try:
                pose = self._torch_tensor(robot.data.body_link_pose_w)[
                    0, int(body_index), :7
                ]
                velocity = self._torch_tensor(robot.data.body_link_vel_w)[
                    0, int(body_index), :6
                ]
                hand = self._torch_tensor(robot.data.joint_pos)[0, joint_ids]
                pose_np = pose.detach().cpu().numpy().astype(np.float64)
                velocity_np = velocity.detach().cpu().numpy().astype(np.float64)
                hand_np = hand.detach().cpu().numpy().astype(np.float64)
            except Exception:
                # A stale finite sample must not keep driving after an invalid
                # PhysX read.  NaN makes the pure controller emit zero wrench.
                self._detached_sim_velocity[side] = np.full(
                    6,
                    np.nan,
                    dtype=np.float64,
                )
                continue
            pose_np = self._pose_isaaclab_to_wxyz(
                pose_np,
                uses_xyzw=self._isaaclab_quaternions_are_xyzw,
            )
            pose_valid = bool(np.all(np.isfinite(pose_np)))
            velocity_valid = bool(np.all(np.isfinite(velocity_np)))
            if pose_valid:
                self._detached_sim_pose[side] = pose_np.copy()
            if pose_valid and velocity_valid:
                self._detached_sim_velocity[side] = velocity_np.copy()
            else:
                # Never combine a stale finite pose with a newer velocity (or
                # vice versa): the PD derivative would otherwise keep
                # applying force after a corrupt PhysX link readback.
                self._detached_sim_velocity[side] = np.full(
                    6,
                    np.nan,
                    dtype=np.float64,
                )
            if np.all(np.isfinite(hand_np)):
                self._detached_sim_hand[side] = hand_np.copy()

    def _robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._robots:
            return super()._robot_state()
        self._refresh_detached_hand_readback()
        tcp = np.concatenate(
            [
                self._detached_sim_pose["right"],
                self._detached_sim_pose["left"],
            ]
        ).astype(np.float64)
        gripper = np.zeros(32, dtype=np.float64)
        gripper[:16] = self._detached_sim_hand["right"][:16]
        gripper[16:32] = self._detached_sim_hand["left"][:16]
        return tcp, gripper

    def _e01_recording_hand_qpos(self) -> np.ndarray:
        """Return the actual PhysX 20+20 Wuji state for episode recording."""

        self._refresh_detached_hand_readback()
        hand_qpos = np.concatenate(
            [
                self._detached_sim_hand["right"],
                self._detached_sim_hand["left"],
            ]
        ).astype(np.float64, copy=True)
        if hand_qpos.shape != (2 * E01_WUJI_HAND_DIM,):
            raise RuntimeError(
                "Detached E01 hand recording state must have shape (40,), "
                f"got {hand_qpos.shape}"
            )
        return hand_qpos

    def _init_haorui_rokae_ik(self) -> None:
        """No online arm model exists in the detached E01 runtime."""

        self._haorui_ik_enabled = False

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        command_filter = self._plan1_command_filter
        info.update(
            {
                "control_runtime": "isaac_physx_detached_canonical_wuji",
                "simulated_arm_count": 0,
                "simulated_hand_count": 2,
                "online_arm_ik": False,
                "center_robot_visible": False,
                "physical_randomization": bool(
                    getattr(self, "_physical_randomization", False)
                ),
                "visual_randomization": bool(self.randomize),
                "palm_root_control": "bounded_world_pd_wrench_servo",
                "palm_wrench_composer": "instantaneous_one_step",
                "palm_wrench_frame": "world",
                "palm_wrench_runtime_validation": "pending_gpu_contact_sweep",
                "cloth_contact_proxy": "invisible_kinematic_sphere",
                "cloth_contact_proxy_count": int(len(self._contact_proxy_paths)),
                "cloth_contact_proxy_radius_m": float(self._contact_proxy_radius),
                "cloth_contact_proxy_bound_to_physx": bool(
                    len(self._contact_proxy_views) == 2
                ),
                "cloth_contact_assist": "local_speed_limited_velocity_response",
                "cloth_contact_assist_disabled": bool(
                    self._cloth_contact_assist_disabled
                ),
                "cloth_contact_assist_radius_m": float(
                    E01_CLOTH_CONTACT_ASSIST_RADIUS_M
                ),
                "cloth_contact_assist_grasp_active_right": bool(
                    self._cloth_contact_assist_grasp_active["right"]
                ),
                "cloth_contact_assist_grasp_active_left": bool(
                    self._cloth_contact_assist_grasp_active["left"]
                ),
                "cloth_contact_assist_active_nodes_right": int(
                    self._cloth_contact_assist_active_nodes["right"]
                ),
                "cloth_contact_assist_active_nodes_left": int(
                    self._cloth_contact_assist_active_nodes["left"]
                ),
                "palm_position_kp_n_per_m": E01_PALM_POSITION_KP_N_PER_M,
                "palm_linear_velocity_kd_n_s_per_m": (
                    E01_PALM_LINEAR_VELOCITY_KD_N_S_PER_M
                ),
                "palm_orientation_kp_nm_per_rad": (
                    E01_PALM_ORIENTATION_KP_NM_PER_RAD
                ),
                "palm_angular_velocity_kd_nm_s_per_rad": (
                    E01_PALM_ANGULAR_VELOCITY_KD_NM_S_PER_RAD
                ),
                "palm_max_force_n": E01_PALM_MAX_FORCE_N,
                "palm_max_torque_nm": E01_PALM_MAX_TORQUE_NM,
                "palm_max_tracking_translation_error_m": (
                    E01_PALM_MAX_TRACKING_TRANSLATION_ERROR_M
                ),
                "palm_max_tracking_rotation_error_deg": float(
                    np.degrees(E01_PALM_MAX_TRACKING_ROTATION_ERROR_RAD)
                ),
                "palm_max_linear_feedforward_m_s": float(
                    self._contact_max_tcp_translation_step / self._control_dt
                ),
                "palm_max_angular_feedforward_rad_s": float(
                    self._contact_max_tcp_rotation_step / self._control_dt
                ),
                "palm_tracking_translation_error_m_right": float(
                    self._palm_tracking_translation_error["right"]
                ),
                "palm_tracking_translation_error_m_left": float(
                    self._palm_tracking_translation_error["left"]
                ),
                "palm_tracking_rotation_error_deg_right": float(
                    np.degrees(self._palm_tracking_rotation_error["right"])
                ),
                "palm_tracking_rotation_error_deg_left": float(
                    np.degrees(self._palm_tracking_rotation_error["left"])
                ),
                "palm_tracking_translation_saturated_right": bool(
                    self._palm_tracking_translation_saturated["right"]
                ),
                "palm_tracking_translation_saturated_left": bool(
                    self._palm_tracking_translation_saturated["left"]
                ),
                "palm_tracking_rotation_saturated_right": bool(
                    self._palm_tracking_rotation_saturated["right"]
                ),
                "palm_tracking_rotation_saturated_left": bool(
                    self._palm_tracking_rotation_saturated["left"]
                ),
                "palm_tracking_saturation_substeps_right": int(
                    self._palm_tracking_saturation_substeps["right"]
                ),
                "palm_tracking_saturation_substeps_left": int(
                    self._palm_tracking_saturation_substeps["left"]
                ),
                "palm_force_saturated_right": bool(
                    self._palm_force_saturated["right"]
                ),
                "palm_force_saturated_left": bool(
                    self._palm_force_saturated["left"]
                ),
                "palm_torque_saturated_right": bool(
                    self._palm_torque_saturated["right"]
                ),
                "palm_torque_saturated_left": bool(
                    self._palm_torque_saturated["left"]
                ),
                "finger_control": "position_drive_only",
                "finger_max_tracking_error_rad": float(
                    E01_FINGER_MAX_TRACKING_ERROR_RAD
                ),
                "finger_tracking_error_rad_right": float(
                    self._finger_tracking_error["right"]
                ),
                "finger_tracking_error_rad_left": float(
                    self._finger_tracking_error["left"]
                ),
                "finger_tracking_saturated_right": bool(
                    self._finger_tracking_saturated["right"]
                ),
                "finger_tracking_saturated_left": bool(
                    self._finger_tracking_saturated["left"]
                ),
                "tcp_observation_source": "physx_palm_readback",
                "hand_root_control_hz": float(1.0 / self._physics_dt),
                "control_command_hz": float(1.0 / self._control_dt),
                "hand_interpolation_substeps": int(self._n_substeps),
                "plan1_reach_guard": True,
                "plan1_max_tcp_translation_step_m": float(
                    self._plan1_max_tcp_translation_step
                ),
                "plan1_max_tcp_rotation_step_deg": float(
                    np.degrees(self._plan1_max_tcp_rotation_step)
                ),
                "plan1_max_hand_joint_step_rad": float(
                    command_filter.max_hand_joint_step
                ),
                "contact_max_tcp_translation_step_m": float(
                    self._contact_max_tcp_translation_step
                ),
                "contact_max_tcp_rotation_step_deg": float(
                    np.degrees(self._contact_max_tcp_rotation_step)
                ),
                "arm_reach_guard_right_active": bool(
                    command_filter.arm_reach_guard_active["right"]
                ),
                "arm_reach_guard_left_active": bool(
                    command_filter.arm_reach_guard_active["left"]
                ),
                "hand_table_guard_right_active": bool(
                    command_filter.hand_table_guard_active["right"]
                ),
                "hand_table_guard_left_active": bool(
                    command_filter.hand_table_guard_active["left"]
                ),
            }
        )
        return info

    def close(self) -> None:
        super().close()


__all__ = [
    "HandsOnE01DetachedHandsIsaacLabEnv",
    "E01_CLOTH_CONTACT_ASSIST_GRASP_JOINT_DELTA_RAD",
    "E01_CLOTH_CONTACT_ASSIST_GRASP_MARGIN_M",
    "E01_CLOTH_CONTACT_ASSIST_MARGIN_M",
    "E01_CLOTH_CONTACT_ASSIST_MAX_NODE_STEP_M",
    "E01_CLOTH_CONTACT_ASSIST_RADIUS_M",
    "E01_CLOTH_CONTACT_ASSIST_SURFACE_OFFSET_M",
    "E01_CLOTH_CONTACT_ASSIST_XY_GAIN",
    "E01_CLOTH_CONTACT_ASSIST_Z_GAIN",
    "E01_CONTACT_PROXY_CONTACT_OFFSET_M",
    "E01_CONTACT_PROXY_RADIUS_M",
    "E01_CONTACT_PROXY_REST_OFFSET_M",
    "E01_CONTACT_PROXY_ROOT_PATH",
    "E01_CONTACT_PROXY_TABLE_CLEARANCE_M",
    "E01_FINGER_MAX_TRACKING_ERROR_RAD",
    "E01_PALM_ANGULAR_VELOCITY_KD_NM_S_PER_RAD",
    "E01_PALM_LINEAR_VELOCITY_KD_N_S_PER_M",
    "E01_PALM_MAX_FORCE_N",
    "E01_PALM_MAX_TORQUE_NM",
    "E01_PALM_MAX_TRACKING_ROTATION_ERROR_RAD",
    "E01_PALM_MAX_TRACKING_TRANSLATION_ERROR_M",
    "E01_PALM_ORIENTATION_KP_NM_PER_RAD",
    "E01_PALM_POSITION_KP_N_PER_M",
]
