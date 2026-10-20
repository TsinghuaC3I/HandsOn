"""GEN4 asset-transfer task on the established D-series collection lifecycle."""

from __future__ import annotations

import json
import math
import os
import numpy as np
from gymnasium import spaces

from ...tasks.eval_gen04_pass_glue_stick import contract as c
from ...tasks.eval_gen04_pass_glue_stick.semantics import (
    HandoffState, tray_geometry,
)
from .d_collection_env import HandsOnDIsaacLabEnv
from .gen4_scene import compose_gen4_assets, load_geometry, sample_layout
from .abc_collection_env import _quat_wxyz_to_matrix


class HandsOnGEN4IsaacLabEnv(HandsOnDIsaacLabEnv):
    """Same robot/control/recorder as D03; source-derived glue/tray geometry."""

    def __init__(self, task_name=c.TASK_NAME, **kwargs):
        if task_name != c.TASK_NAME:
            raise ValueError(f"GEN4 environment received a different task: {task_name}")
        self._gen4_geometry = load_geometry(required=kwargs.get("launch_isaac", True))
        self._gen4_handoff = HandoffState()
        self._gen4_support_view = None
        self._gen4_hand_bounds = []
        self._gen4_lost_steps = 0
        self._gen4_failure_step = -1
        self._gen4_terminal_reason = None
        self._gen4_reset_layout = {}
        kwargs.setdefault("hz", c.CONTROL_HZ)
        if kwargs.get("control_dt") is None:
            kwargs["control_dt"] = 1.0 / c.CONTROL_HZ
        kwargs.setdefault("time_limit", c.TIME_LIMIT_S)
        kwargs.setdefault("physics_dt", 1.0 / c.PHYSICS_HZ)
        super().__init__(task_name, **kwargs)
        self.stable_recording_origin_default_robot_restore_enabled = False

    def _task_specific_isaac_kit_args(self):
        # The isolated task checkout can own Kit caches/logs even when the
        # ordinary recorder, rather than a diagnostic script, launches it.
        portable_root = os.environ.get("HANDSON_ISAACSIM_PORTABLE_ROOT", "").strip()
        gpu_uuid = os.environ.get("HANDSON_GEN4_EXCLUSIVE_GPU_UUID", "").strip()
        if gpu_uuid and self._simulation_app is None:
            # The user's isolated validation run must recheck before *each*
            # ordinary fresh recorder child, including reset retries.
            import subprocess
            import xml.etree.ElementTree as ET
            from pathlib import Path
            raw = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid, "-q", "-x"], text=True)
            if portable_root:
                evidence = Path(portable_root) / "gpu_launch_checks"
                evidence.mkdir(parents=True, exist_ok=True)
                (evidence / f"{os.getpid()}.xml").write_text(raw)
            gpu = ET.fromstring(raw).find("gpu")
            idle = (
                gpu is not None and gpu.findtext("uuid") == gpu_uuid
                and gpu.find("processes") is not None and not gpu.findall("processes/process_info")
                and float(gpu.findtext("fb_memory_usage/used").split()[0]) < 100
                and float(gpu.findtext("utilization/gpu_util").split()[0]) == 0
            )
            if not idle:
                raise RuntimeError("GEN4 validation launch stopped: the authorized GPU is no longer idle.")
        return f"--portable-root {portable_root}" if portable_root else None

    def _d03_physx_arm_target_limits_enabled(self):
        # Preserve D03's tracker-discontinuity rejection without changing
        # robot drives or applying the small block's object velocity caps.
        return self._physics_backend == "physx"

    def _build_observation_space(self):
        result = {
            "state": spaces.Dict({
                key: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
                for key, shape in c.PROPRIO_SHAPES.items()
            }),
            "gen4_handoff_state": spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float64),
        }
        if self.image_obs:
            wrist_width, wrist_height = self._wrist_recording_camera_output_size
            image_shapes = {
                "random_camera" if self.randomize else "front": self._image_shape,
                "wrist_left": (wrist_height, wrist_width, 3),
                "wrist_right": (wrist_height, wrist_width, 3),
            }
            result["images"] = spaces.Dict({
                key: spaces.Box(0, 255, shape=shape, dtype=np.uint8)
                for key, shape in image_shapes.items()
            })
        return spaces.Dict(result)

    def _sample_reset_state(self, *, clear_manual_request):
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.stable_recording_origin_default_robot_restore_enabled = False
        self._gen4_handoff = HandoffState()
        self._gen4_lost_steps = 0
        self._gen4_failure_step = -1
        self._gen4_terminal_reason = None
        self._gen4_reset_layout = {}

    def _spawn_task_specific_assets(self, sim_utils):
        # ``prepare_gen4_mjcf`` only renames the imported owners.  The selected
        # glue/tray USDs must be composed here, before reset and first PhysX;
        # otherwise GEN4 presents D03's square placeholder and the handoff
        # predicate can never observe the glue-stick body.
        compose_gen4_assets(self._stage, "/World/Task")
        for prim in self._stage.GetPrimAtPath("/World/Task").GetChildren():
            if prim.GetName() in {c.GLUE_BODY, c.TRAY_BODY}:
                prim.SetCustomDataByKey("handson:legacyCollisionSource", "D03/B05")

    def _apply_task_pose_randomization(self):
        if not self._task_reset_body_states and not self._task_reset_articulation_states:
            return
        layout = sample_layout(self._random, randomize=self.randomize, geometry=self._gen4_geometry)
        for name, pose in layout.items():
            # GEN4 layout yaw is absolute. The shared D writer otherwise adds
            # it to the authored orientation (doubling the tray's 90° yaw).
            if not self._write_reset_pose(name, orientation_xyzw=np.asarray([0, 0, 0, 1]), **pose):
                raise RuntimeError(f"GEN4 physical reset did not write {name}")
        self._gen4_reset_layout = layout
        print("[GEN4] Seeded physical reset: " + json.dumps({
            "seed": getattr(self, "_seed", None), "positions": layout,
        }, sort_keys=True), flush=True)

    def _configure_task_collision_safety(self, sim_utils):
        if self._stage is None:
            return 0, 0
        from pxr import PhysxSchema, UsdPhysics, Usd, UsdGeom, Vt, Gf
        # These props have no moving joints. The MJCF importer's redundant
        # articulation roots disable normal sleep and retain contact jitter.
        # Reuse D03's damping/sleep values on actual free rigid bodies; contact
        # naturally wakes them. No runtime pose/velocity clamping is applied.
        self._configure_named_reset_stability(
            sim_utils, (c.GLUE_BODY, *c.DISTRACTORS),
            linear_damping=0.25, angular_damping=1.0, sleep_threshold=0.010,
            warning_label="GEN4 free rigid bodies", strip_redundant_articulation_root=True,
        )
        prim = self._find_rigid_body_prim(c.GLUE_BODY)
        if prim is None:
            raise RuntimeError("GEN4 glue rigid body is missing before first PhysX reset")
        # The imported glue mesh starts on the tray/table contact shell.  Use
        # a stronger passive brake during the reset settling window so the
        # authored pose reaches sleep without changing grasp semantics.
        glue_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        glue_rigid.CreateLinearDampingAttr().Set(2.0)
        glue_rigid.CreateAngularDampingAttr().Set(20.0)
        glue_rigid.CreateSleepThresholdAttr().Set(0.05)
        report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report.CreateThresholdAttr().Set(0.0)
        properties = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        properties.CreateEnableSpeculativeCCDAttr().Set(True)
        properties.CreateSolverPositionIterationCountAttr().Set(64)
        properties.CreateSolverVelocityIterationCountAttr().Set(8)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(self._gen4_geometry["glue"]["mass_kg"])
        from pxr import Gf
        axes = self._gen4_geometry["glue"]["principal_axes_wxyz"]
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(axes[0], Gf.Vec3f(*axes[1:])))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*self._gen4_geometry["glue"]["diagonal_inertia_kg_m2"]))
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*self._gen4_geometry["glue"].get("center_of_mass_m", [0, 0, 0])))
        for name in (c.GLUE_BODY, c.TRAY_BODY, *c.DISTRACTORS):
            owner = self._find_rigid_body_prim(name)
            if owner is None:
                raise RuntimeError(f"GEN4 physical owner is missing: {name}")
            # The tray and D03 clutter are fixtures for this handoff task.
            # Keep them kinematic so tiny imported-mesh contact torques do
            # not make the reset origin drift before recording starts.  The
            # glue stick remains dynamic and is the only task object that
            # must be grasped and placed by the operator.
            if name == c.TRAY_BODY or name in c.DISTRACTORS:
                UsdPhysics.RigidBodyAPI.Apply(owner).CreateKinematicEnabledAttr().Set(True)
            for collider in Usd.PrimRange(owner):
                if collider.HasAPI(UsdPhysics.CollisionAPI):
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(collider)
                    contact.CreateContactOffsetAttr().Set(0.001)
                    contact.CreateRestOffsetAttr().Set(0.0)
                    if collider.IsA(UsdGeom.Mesh):
                        # Keep physical size exact while cooking small source
                        # meshes in millimetres. Metre-valued inputs lose
                        # support vertices to PhysX's convex cooking tolerance.
                        mesh = UsdGeom.Mesh(collider)
                        xform = UsdGeom.Xformable(collider)
                        if not collider.HasAttribute("xformOp:scale:gen4CookingUnits"):
                            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((points * 1000).astype(np.float32)))
                            if mesh.GetExtentAttr().HasAuthoredValueOpinion():
                                extent = np.asarray(mesh.GetExtentAttr().Get(), dtype=np.float64)
                                mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy((extent * 1000).astype(np.float32)))
                            xform.AddScaleOp(opSuffix="gen4CookingUnits").Set(Gf.Vec3f(.001))
        return 1, len(self._gen4_geometry["glue"].get("collision_paths", []))

    def _init_task_specific_runtime(self):
        super()._init_task_specific_runtime()
        self._gen4_support_view = None
        if not self._robots or self._stage is None:
            return
        view = self._task_tensor_sim_view()
        glue = self._find_rigid_body_prim(c.GLUE_BODY)
        tray = self._find_rigid_body_prim(c.TRAY_BODY)
        if view is not None and glue is not None and tray is not None:
            self._gen4_support_view = view.create_rigid_contact_view(
                str(glue.GetPath()), filter_patterns=[str(tray.GetPath())],
                max_contact_data_count=128,
            )
        self._cache_gen4_hand_bounds()

    def _cache_gen4_hand_bounds(self):
        """Conservative source-collider spheres for stale-contact release only."""
        from pxr import Usd, UsdGeom, UsdPhysics
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
        self._gen4_hand_bounds = []
        for side, robot in self._robots.items():
            names = list(robot.body_names)
            for path in self._d_hand_body_paths(side):
                body = self._stage.GetPrimAtPath(path)
                if body.GetName() not in names:
                    continue
                bounds = []
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                        continue
                    box = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
                    if not box.IsEmpty():
                        bounds.extend([np.asarray(box.GetMin()), np.asarray(box.GetMax())])
                if bounds:
                    low, high = np.min(bounds, axis=0), np.max(bounds, axis=0)
                    center = (low + high) * 0.5
                    radius = float(np.linalg.norm((high - low) * 0.5))
                    self._gen4_hand_bounds.append((side, names.index(body.GetName()), center, radius))

    def _gen4_geometric_release(self, target):
        if not self._gen4_hand_bounds or self._gen4_geometry is None:
            return False
        low, high = np.asarray(self._gen4_geometry["glue"]["bounds_m"])
        rotation = _quat_wxyz_to_matrix(target[3:])
        # A complete collision-link sphere must clear the glue's occupied
        # bounding box. This can reject stale impulses but cannot fake contact.
        for side, index, center, radius in self._gen4_hand_bounds:
            matrix = self._robot_body_pose_matrix(self._robots[side], index)
            world = matrix[:3, :3] @ center + matrix[:3, 3]
            local = (world - target[:3]) @ rotation
            gap = np.maximum(np.maximum(low - local, local - high), 0)
            if float(np.linalg.norm(gap)) <= radius + 0.003:
                return False
        return True

    def _d_contact_reading(self, side, name):
        view = self._d_hand_contact_views.get((side, name))
        if view is None:
            return 0.0, False
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            forces = np.asarray(matrix, dtype=float)
            if (
                forces.ndim != 3 or forces.shape[0] != 1
                or forces.shape[1] < 1 or forces.shape[2] != 3
                or not np.isfinite(forces).all()
            ):
                return 0.0, False
            with np.errstate(over="ignore", invalid="ignore"):
                total_force = float(np.linalg.norm(forces, axis=-1).sum())
            return (total_force, True) if math.isfinite(total_force) else (0.0, False)
        except Exception as exc:
            key = (side, name)
            if key not in self._d_contact_view_errors:
                self._d_contact_view_errors.add(key)
                self._add_task_state_warning(f"GEN4 {side}-hand contact reading unavailable: {exc}")
            return 0.0, False

    def _gen4_metrics(self, *, update=False):
        target = self._poses.get(c.GLUE_BODY, self._fallback_pose(c.GLUE_BODY))
        tray = self._poses.get(c.TRAY_BODY, self._fallback_pose(c.TRAY_BODY))
        left_force, left_available = self._d_contact_reading("left", c.GLUE_BODY)
        right_force, right_available = self._d_contact_reading("right", c.GLUE_BODY)
        left = left_available and left_force >= c.HAND_CONTACT_FORCE_N
        right = right_available and right_force >= c.HAND_CONTACT_FORCE_N
        shape = self._gen4_geometry
        geometry = {"footprint_inside": False, "at_floor": False, "floor_gap_m": math.inf, "lowest_world_z": -math.inf}
        if shape is not None:
            geometry = tray_geometry(shape["glue"]["occupied_vertices_m"], target, tray, shape["tray"]["interior_xy_m"], shape["tray"]["floor_z_m"])
        lift = geometry["lowest_world_z"] - self._current_table_top_z()
        lifted = lift >= c.LIFT_CLEARANCE_M
        transfer = bool(-0.45 <= target[0] <= 0.12 and abs(target[1]) <= 0.18 and lifted)
        velocity = self._physics_body_velocity(c.GLUE_BODY)
        linear, angular = self._safe_speed(velocity), self._safe_speed(velocity, True)
        support_force = self._c04_contact_force(self._gen4_support_view)
        # A kinematic tray can expose an empty contact-force matrix even when
        # the measured glue footprint is seated on its floor.  The floor band
        # is derived from the source collision vertices, so it is a physical
        # support test rather than a pose shortcut; require low speed as well.
        support_by_geometry = bool(
            self._gen4_support_view is None
            and geometry["at_floor"]
            and linear <= c.LINEAR_SPEED_MAX
        )
        supported = bool(support_force > 0.05 or support_by_geometry)
        inside = bool(geometry["footprint_inside"] and geometry["at_floor"] and supported)
        contacts_available = left_available and right_available
        released = bool(contacts_available and not (left or right))
        if contacts_available and not released and inside and linear <= c.LINEAR_SPEED_MAX:
            released = self._gen4_geometric_release(target)
        stable = bool(inside and released and linear <= c.LINEAR_SPEED_MAX and angular <= c.ANGULAR_SPEED_MAX)
        state = self._gen4_handoff
        if update:
            state.advance(
                left=left, right=right, contacts_available=contacts_available,
                lifted=lifted, transfer=transfer, stable=stable,
                success_counter=self._success_counter,
            )
        metrics = {
            "handoff_stage": state.stage, "left_pickup_confirmed": state.left_confirmed,
            "both_hands_confirmed": state.both_confirmed, "right_only_confirmed": state.right_confirmed,
            "direct_right_contact_seen": state.direct_right_seen,
            "left_target_contact": bool(left), "right_target_contact": bool(right),
            "target_lift_height": float(lift), "target_lifted": bool(lifted),
            "target_in_transfer_zone": transfer, "target_inside_box": inside,
            "target_supported_by_tray": bool(supported),
            "target_support_force_n": float(support_force),
            "target_support_source": "contact" if support_force > 0.05 else ("geometry" if support_by_geometry else "none"),
            "target_floor_gap_m": geometry["floor_gap_m"],
            "target_released": released, "target_linear_speed": linear,
            "target_angular_speed": angular, "placement_stable": stable,
        }
        if update:
            self._d_last_metrics = metrics
        return metrics

    def _compute_success_now(self):
        metrics = self._gen4_metrics(update=True)
        return bool(metrics["placement_stable"] and self._gen4_handoff.stage >= 3)

    def _success_condition_rows(self):
        m = self._gen4_metrics()
        s = self._gen4_handoff
        checks = (s.left_confirmed, s.both_confirmed, s.right_confirmed, m["placement_stable"])
        rows = [self._condition_row(f"gen4_{i}", label, ok) for i, (label, ok) in enumerate(zip(c.CONDITION_LABELS, checks))]
        rows.append(self._condition_row("gen4_success_hold", "Released placement stable for 8 frames", self._success_counter >= c.SUCCESS_FRAMES))
        return rows

    @staticmethod
    def _success_condition_display_label(row):
        return str(row.get("label") or row.get("key", "Condition"))

    def _compute_observation(self):
        tcp, gripper = self._d_robot_state()
        m = self._gen4_metrics()
        target = self._poses.get(c.GLUE_BODY, self._fallback_pose(c.GLUE_BODY))
        tray = self._poses.get(c.TRAY_BODY, self._fallback_pose(c.TRAY_BODY))
        s = self._gen4_handoff
        state = {
            "tcp_pose": tcp, "gripper_pose": gripper,
            "target_pose": target.copy(), "box_pose": tray.copy(),
            "target_to_box": (target[:3] - tray[:3]).astype(np.float64),
            "handoff_metrics": np.asarray([s.stage, m["left_target_contact"], m["right_target_contact"], m["target_lift_height"], m["target_in_transfer_zone"], s.both_confirmed, s.right_confirmed, m["target_inside_box"], m["target_linear_speed"], self._success_counter], dtype=np.float64),
            "distractor_poses": np.concatenate([self._poses.get(name, self._fallback_pose(name)).copy() for name in c.DISTRACTORS]),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        result = {"state": state, "gen4_handoff_state": np.concatenate([s.vector(), [self._success_counter, self._gen4_lost_steps]])}
        if self.image_obs:
            front, left, right = self.render()
            result["images"] = {"random_camera" if self.randomize else "front": front, "wrist_left": left, "wrist_right": right}
        return result

    def _time_limit_is_truncation(self):
        return True

    def get_recording_schema(self):
        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        result["fields"] = {**schema["fields"], "gen4_handoff_state": {
            "shape": [10], "dtype": "float64",
            "source": "gen4_action_aligned_task_state",
            "layout": list(c.HIDDEN_STATE_FIELDS),
            "visibility": "hidden_task_logic", "replay_domain": "task_state",
            "physical_replay": True, "replay_restore_supported": False,
            "replay_atol": 0.0,
        }}
        result["task_state"] = {
            "task_name": c.TASK_NAME,
            "field_names": ["gen4_handoff_state"],
            "hidden_field_names": ["gen4_handoff_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return result

    def _compute_failure(self):
        if self._gen4_terminal_reason is not None:
            return True
        if self._sim is None or self.env_step < 1 or self._gen4_failure_step == self.env_step:
            return False
        self._gen4_failure_step = self.env_step
        pose = self._poses.get(c.GLUE_BODY)
        if pose is None or self._gen4_geometry is None:
            return False
        # A target completely below the tabletop cannot be recovered by the
        # intended table-level operation. Ordinary table drops remain retryable.
        from ...tasks.eval_gen04_pass_glue_stick.semantics import transformed_points
        points = transformed_points(self._gen4_geometry["glue"]["occupied_vertices_m"], pose)
        lost = bool(points[:, 2].max() < self._current_table_top_z() - 0.10)
        self._gen4_lost_steps = self._gen4_lost_steps + 1 if lost else 0
        if self._gen4_lost_steps >= 15:
            self._gen4_terminal_reason = {
                "reason": "glue_stick_below_recoverable_workspace", "body": c.GLUE_BODY,
                "control_step": int(self.env_step), "consecutive_frames": 15,
                "observed_top_z_m": float(points[:, 2].max()),
                "threshold_z_m": self._current_table_top_z() - 0.10,
            }
            print("[GEN4] First terminal reason: " + json.dumps(self._gen4_terminal_reason), flush=True)
            return True
        return False

    def _task_status_text(self):
        names = ("Left-hand pickup", "Two-hand contact", "Right-hand takeover", "Place and release", "Complete")
        return f"GEN4 | {names[self._gen4_handoff.stage]} | Stable {min(self._success_counter, 8)}/8"

    def _task_info(self):
        result = super()._task_info()
        result.update(task_id=c.TASK_ID, task_name=c.TASK_NAME, handoff_stage=self._gen4_handoff.stage)
        if self._gen4_terminal_reason is not None:
            result["terminal_reason"] = dict(self._gen4_terminal_reason)
        return result

    def restore_recorded_initial_state(self, state_parts):
        restored = []
        pairs = [(c.GLUE_BODY, state_parts.get("target_pose")), (c.TRAY_BODY, state_parts.get("box_pose"))]
        clutter = np.asarray(state_parts.get("distractor_poses", []), dtype=float).reshape(-1, 7)
        pairs.extend(zip(c.DISTRACTORS, clutter))
        for name, value in pairs:
            view = self._task_body_views.get(name)
            if view is not None and value is not None and self._set_task_body_world_pose(view, np.asarray(value, dtype=float).reshape(7)):
                restored.append(name)
        hidden = state_parts.get("gen4_handoff_state")
        if hidden is not None:
            hidden = np.asarray(hidden, dtype=float).reshape(10)
            self._gen4_handoff = HandoffState.from_vector(hidden[:8])
            self._success_counter, self._gen4_lost_steps = map(int, hidden[8:])
        self._d_last_metrics = {}
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()

    def get_physical_replay_reset_contract(self):
        result = dict(super().get_physical_replay_reset_contract())
        result["task_name"] = c.TASK_NAME
        result["gen4_handoff_state_layout"] = list(c.HIDDEN_STATE_FIELDS)
        result["gen4_assistance"] = "none"
        return result
