"""Continuous FU015 + C05 using their existing assets and physical predicates."""

from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    _C05_BUTTON_PRESS_THRESHOLD, _C05_SUCCESS_STEPS, _C05_MOUSEPAD_RADIUS,
    _TABLE_TOP_Z, _randomized_physx_pose, _quat_wxyz_to_matrix,
)
from .fu015_collection_env import HandsOnFU015IsaacLabEnv, _right_tcp_opening_intent
from .fu015_contract import SUCCESS_HOLD_STEPS, TARGET_TOLERANCE_DEG, REQUIRED_CONTACT_FORCE_N, FU015BilateralOwnerGate
from .xy2_contract import (
    TASK_ID, TASK_NAME, SOURCE_TASK_IDS, TIME_LIMIT_SECONDS,
    FAILURE_CODES, RECORDING_LAYOUTS, RECORDING_ATOLS,
    XY2Progress, laptop_retained, sample_layout,
)


class HandsOnXY2IsaacLabEnv(HandsOnFU015IsaacLabEnv):
    _runtime_task_name = TASK_NAME

    def _is_b320_collection_task(self) -> bool:
        # The public XY2 identifier has no B320 prefix, but it inherits
        # FU015's physical-palm command origin and realtime servo pacing.
        return True

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        self._reset_xy2_left_hinge_control()
        self._xy2_randomize = bool(kwargs.get("randomize", True))
        target = kwargs.pop("target_object", None)
        self._xy2_target_angle = None if target in (None, "", "random") else float(target)
        self._xy2_progress = XY2Progress(_C05_SUCCESS_STEPS)
        self._xy2_layout = None
        self._xy2_composed_layout = None
        self._xy2_last_evaluated_frame = -1
        self._xy2_success_now = False
        self._xy2_failure_reported = False
        self._xy2_capture_button_baseline = False
        self._xy2_mouse_collision_vertices_local = np.empty((0, 3), dtype=np.float64)
        # XY2 uses the shared FU015 scene but does not expose FU015's tensor
        # contact views in the composed laptop layout.  Fall back to the
        # canonical PhysX callback report for recording capture.
        self._physx_prefer_tensor_contact_capture = False
        kwargs.setdefault("time_limit", TIME_LIMIT_SECONDS)
        kwargs.setdefault("randomize", self._xy2_randomize)
        super().__init__(task_name, **kwargs)
        # FU015's constructor enables tensor contact capture globally; XY2's
        # composed scene has no tensor views, so retain the callback path.
        self._physx_prefer_tensor_contact_capture = False

    def _reset_xy2_left_hinge_control(self):
        self._xy2_previous_left_tcp_z = None
        self._xy2_left_opening_intent = False
        self._xy2_left_display_contact = False
        self._xy2_left_owner_gate = FU015BilateralOwnerGate()

    def _before_task_physics_substep(self):
        if self._fu015_stability_probe_substeps % max(1, int(self._n_substeps)) == 0:
            z = float(self._left_tcp_pose[2])
            self._xy2_left_opening_intent = _right_tcp_opening_intent(
                self._xy2_previous_left_tcp_z, z)
            self._xy2_previous_left_tcp_z = z if np.isfinite(z) else None
        force = float(self._fu015_raw_contact_diagnostic.get("left_display_force_n", 0.0))
        self._xy2_left_display_contact = bool(
            np.isfinite(force) and force >= REQUIRED_CONTACT_FORCE_N)
        super()._before_task_physics_substep()

    def _hinge_control_contact(self):
        # XY2 permits either hand to operate the display.
        return super()._hinge_control_contact() or self._xy2_left_display_contact

    def _hinge_control_opening_intent(self):
        # Lift intent and display contact must belong to the same hand.
        return bool(
            (super()._hinge_control_contact() and super()._hinge_control_opening_intent())
            or (self._xy2_left_display_contact and self._xy2_left_opening_intent))

    def _hinge_assist_owner_ready(self, owner_ready, metrics):
        force = float(self._fu015_raw_contact_diagnostic.get("left_display_force_n", 0.0))
        sample = bool(np.isfinite(force) and force >= REQUIRED_CONTACT_FORCE_N
                      and metrics["base_stable"] and not self._fu015_failure_latched)
        for _ in range(max(1, int(self._n_substeps))):
            self._xy2_left_owner_gate.update(sample)
        return owner_ready or self._xy2_left_owner_gate.ready

    def reset(self, seed=None, *, target_label=None, **kwargs):
        # Ordinary collection UI and the formal certifier pass target_label
        # to reset(), rather than to the environment constructor.
        if target_label is not None:
            self._xy2_target_angle = None if target_label in ("", "random") else float(target_label)
        return super().reset(seed=seed, **kwargs)

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        kwargs = super()._in_process_scene_rebuild_kwargs(seed)
        # ABC's physical target name is "mouse". XY2's public target selector
        # is the laptop angle, which must survive ordinary recorder hard resets.
        kwargs["target_object"] = self._xy2_target_angle
        return kwargs

    def _initial_laptop_reset(self, seed: int):
        self._xy2_layout = sample_layout(
            seed, randomize=self._xy2_randomize, target_angle=self._xy2_target_angle,
        )
        if self._xy2_composed_layout is None:
            self._xy2_composed_layout = self._xy2_layout
        return self._xy2_layout.laptop

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._reset_xy2_left_hinge_control()
        self._xy2_progress.reset()
        self._xy2_last_evaluated_frame = -1
        self._xy2_success_now = False
        self._xy2_failure_reported = False
        self._xy2_capture_button_baseline = True

    def _task_extra_pose_names(self) -> set[str]:
        return super()._task_extra_pose_names() | {"fu015_laptop_base", "fu015_laptop_display"}

    def _configure_task_simulation_settings(self) -> None:
        super()._configure_task_simulation_settings()
        from pxr import PhysxSchema, Usd, UsdPhysics

        # Match FU015's contact sensors before PhysX initialization, including
        # preview runs where full recording contact capture is disabled. The
        # inherited robot/table sensor also needs an explicit table reporter.
        for name in ("mouse", "table"):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                raise RuntimeError(f"XY2 {name} contact sensor is unavailable")
            report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)
        mouse_root = self._find_rigid_body_prim("mouse")
        if mouse_root is not None:
            # The old C05 mouse is free to move, but its small buttons must
            # settle without the 29 mm first-frame launch seen in XY2.
            for body in Usd.PrimRange(mouse_root):
                if not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                api = PhysxSchema.PhysxRigidBodyAPI(body) if body.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI.Apply(body)
                api.CreateLinearDampingAttr().Set(0.45)
                api.CreateAngularDampingAttr().Set(1.20)
                api.CreateSleepThresholdAttr().Set(0.02)
        # FU015 already authors the laptop's native world-fixed articulation.
        # "link_3" belongs to the C05 mouse, not the laptop: making it
        # kinematic here disables the free mouse/button root's dynamics before
        # the first PhysX reset and prevents physical mouse manipulation.
        self._cache_xy2_mouse_collision_vertices(self._find_rigid_body_prim("mouse"))

    @staticmethod
    def _xy2_collider_points(collider: Any):
        from pxr import Usd, UsdGeom

        if collider.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(collider).GetPointsAttr().Get() or []
        if not collider.IsA(UsdGeom.Boundable):
            return []
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
            UsdGeom.Boundable(collider), Usd.TimeCode.Default(),
        )
        if extent is None or len(extent) != 2:
            return []
        return [(x, y, z) for x in (extent[0][0], extent[1][0])
                for y in (extent[0][1], extent[1][1])
                for z in (extent[0][2], extent[1][2])]

    def _cache_xy2_mouse_collision_vertices(self, mouse: Any) -> None:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self._xy2_mouse_collision_vertices_local = np.empty((0, 3), dtype=np.float64)
        cache = UsdGeom.XformCache()
        # PhysX poses have no scale. Remove scale only from the inverse body
        # transform, retaining the real dimensions in every collider's world
        # points, including scale authored on the body or its ancestors.
        world_to_body_pose = cache.GetLocalToWorldTransform(mouse).RemoveScaleShear().GetInverse()
        vertices = []
        for collider in Usd.PrimRange(mouse, Usd.TraverseInstanceProxies()):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
                continue
            owner = collider
            while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != mouse:
                continue
            collider_to_body = cache.GetLocalToWorldTransform(collider) * world_to_body_pose
            for point in self._xy2_collider_points(collider):
                vertices.append(collider_to_body.Transform(Gf.Vec3d(*point)))
        points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            raise RuntimeError("XY2 mouse has no finite source collision geometry")
        self._xy2_mouse_collision_vertices_local = points

    def _init_task_specific_runtime(self) -> None:
        super()._init_task_specific_runtime()
        if not self._robots:
            return
        physics_view = self._task_tensor_sim_view()
        mouse = self._find_rigid_body_prim("mouse")
        table = self._find_rigid_body_prim("table")
        if mouse is None or table is None:
            raise RuntimeError("XY2 mouse/table support bodies are unavailable")
        # The mousepad is visual-only, including its photo-referenced black fabric. Its physical support is
        # the unchanged table underneath it, not a new pad collider.
        self._create_fu015_contact_view(
            physics_view, "xy2_mouse_table", str(mouse.GetPath()), [str(table.GetPath())],
        )

    def _xy2_mouse_bottom_height_error_m(self) -> float:
        points = self._xy2_mouse_collision_vertices_local
        if not len(points) or not np.isfinite(points).all():
            return float("inf")
        pose = self._physics_body_pose("mouse")
        if pose is None and not self._launch_isaac:
            pose = self._pose_for_role("target")
        if pose is None or not np.isfinite(pose).all() or np.linalg.norm(pose[3:7]) < 1e-9:
            return float("inf")
        rotation = _quat_wxyz_to_matrix(pose[3:7])
        bottom_z = float(np.min(points @ rotation[2]) + pose[2])
        return bottom_z - float(self._pose_for_role("receptacle")[2])

    def _xy2_mouse_supported(self, *, bottom_height_error_m: float | None = None) -> bool:
        if bottom_height_error_m is None:
            bottom_height_error_m = self._xy2_mouse_bottom_height_error_m()
        # PhysX contact buffers can retain a previous positive force after
        # separation. Check current source geometry as well as the contact.
        if not np.isfinite(bottom_height_error_m) or abs(bottom_height_error_m) > .004:
            return False
        points, forces = self._contact_samples("xy2_mouse_table")
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        forces = np.asarray(forces, dtype=np.float64).reshape(-1)
        if not len(points) or len(points) != len(forces):
            return False
        pad = self._pose_for_role("receptacle")
        valid = (
            np.isfinite(points).all(axis=1) & np.isfinite(forces)
            & (forces > 1e-4)
            & (np.abs(points[:, 2] - pad[2]) <= .004)
            & (np.linalg.norm(points[:, :2] - pad[:2], axis=1) <= _C05_MOUSEPAD_RADIUS)
        )
        return bool(np.any(valid))

    def _xy2_mouse_click_metrics(self, *, update_failure: bool) -> dict[str, Any]:
        # Keep C05 unchanged; XY2 accepts a click on the supported pad
        # regardless of travel distance. Preserve C05's early-click failure.
        metrics = self._c05_mouse_click_metrics(update_failure=False)
        bottom_error = self._xy2_mouse_bottom_height_error_m()
        supported = self._xy2_mouse_supported(bottom_height_error_m=bottom_error)
        on_pad = bool(metrics["c05_mouse_on_pad"] and supported)
        ready = on_pad
        left = metrics["c05_left_button_travel"] >= _C05_BUTTON_PRESS_THRESHOLD
        right = metrics["c05_right_button_travel"] >= _C05_BUTTON_PRESS_THRESHOLD
        early = bool((left or right) and not ready)
        if update_failure and early:
            self._c05_failed_early_click = True
        metrics.update({
            "xy2_mouse_supported": supported,
            "xy2_mouse_bottom_height_error_m": bottom_error,
            "c05_mouse_on_pad": on_pad,
            "c05_ready_for_click": ready,
            "c05_early_click_now": early,
            "c05_failed_early_click": bool(self._c05_failed_early_click),
            "c05_no_early_click": not self._c05_failed_early_click,
            "c05_left_clicked": bool(ready and left and not right and not self._c05_failed_early_click),
        })
        return metrics

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        progress = self._xy2_progress
        if progress.phase != 1 or self._xy2_capture_button_baseline:
            return
        neutral = all(self._c05_button_travel(name) < _C05_BUTTON_PRESS_THRESHOLD
                      for name in ("mouse_joint_0", "mouse_joint_1"))
        if neutral:
            progress.buttons_armed = True
            return
        # Read the physical pose at the press, rather than the previous
        # control frame. The recorded click latch survives button release.
        for name in ("mouse", "mousepad"):
            pose = self._physics_body_pose(name)
            if pose is not None:
                self._poses[name] = pose.copy()
        progress.observe_buttons(
            laptop=self._xy2_laptop_metrics(),
            mouse=self._xy2_mouse_click_metrics(update_failure=True),
            buttons_neutral=False,
        )

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        super()._spawn_task_specific_assets(sim_utils)
        # Author the joint sample BEFORE the first SimulationContext.reset().
        # Later writes use the same shared reset transaction as the source tasks.
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        cache = UsdGeom.XformCache()
        for name, xy, yaw in (
            ("mouse", self._xy2_layout.mouse_xy, self._xy2_layout.mouse_yaw_deg),
            ("mousepad", self._xy2_layout.pad_xy, 0.0),
        ):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                raise RuntimeError(f"XY2 missing source body {name}")
            world = cache.GetLocalToWorldTransform(prim)
            desired = world * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
            z = _TABLE_TOP_Z if name == "mousepad" else float(world.ExtractTranslation()[2])
            # MJCF imports mouse links as siblings. Transform their common
            # parent so buttons and root agree before the first physics step.
            group = prim.GetParent() if name == "mouse" else prim
            if name == "mouse":
                collision_bottoms = [
                    cache.GetLocalToWorldTransform(collider).Transform(Gf.Vec3d(*point))[2]
                    for collider in Usd.PrimRange(group, Usd.TraverseInstanceProxies())
                    if collider.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is not False
                    for point in self._xy2_collider_points(collider)
                ]
                if not collision_bottoms:
                    raise RuntimeError("XY2 mouse has no source collision geometry")
                bottom = min(collision_bottoms)
                # Leave a small positive support gap; the old mesh is more
                # faceted than the current import and .001 m still enters the
                # contact shell at some yaw values.
                z += _TABLE_TOP_Z + .004 - float(bottom)
            desired.SetTranslateOnly(Gf.Vec3d(float(xy[0]), float(xy[1]), z))
            group_world = cache.GetLocalToWorldTransform(group) * world.GetInverse() * desired
            parent = cache.GetLocalToWorldTransform(group.GetParent())
            xform = UsdGeom.Xformable(group)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(group_world * parent.GetInverse())
            cache.Clear()

    def _apply_task_pose_randomization(self) -> None:
        super()._apply_task_pose_randomization()
        if self._torch is None:
            return
        bodies = {entry[0]: entry[1:5] for entry in self._task_reset_body_states}
        articulations = {entry[0]: entry[1:5] for entry in self._task_reset_articulation_states}
        for name, xy, yaw in (
            ("mouse", self._xy2_layout.mouse_xy,
             self._xy2_layout.mouse_yaw_deg - self._xy2_composed_layout.mouse_yaw_deg),
            ("mousepad", self._xy2_layout.pad_xy, 0.0),
        ):
            prim = self._find_rigid_body_prim(name)
            path = str(prim.GetPath()) if prim is not None else ""
            state = self._reset_transform_state_for_path(path, bodies, articulations)
            if state is None:
                raise RuntimeError(f"XY2 source {name} is absent from physical reset state")
            kind, view, indices, root_pose, velocity = state
            initial = root_pose.detach().cpu().numpy().copy()
            pose = _randomized_physx_pose(
                initial, xy_offset=np.asarray(xy) - initial[0, :2],
                z_offset=0.0, yaw_degrees=yaw,
            )
            if name == "mouse":
                # The shared C05 root is a free body whose authored height
                # predates the 0.89 m collection table. The pre-PhysX
                # authoring hook fixes the first scene, but the regular reset
                # transaction writes the cached root pose again. Recompute
                # the root height from the unchanged legacy collision mesh so
                # the mouse starts supported at every sampled yaw.
                points = self._xy2_mouse_collision_vertices_local
                if len(points) and np.isfinite(points).all():
                    # PhysX reset poses use (x, y, z, qx, qy, qz, qw), while
                    # the shared matrix helper consumes wxyz.
                    quat_wxyz = np.asarray(
                        [pose[0, 6], pose[0, 3], pose[0, 4], pose[0, 5]],
                        dtype=np.float64,
                    )
                    rotation = _quat_wxyz_to_matrix(quat_wxyz)
                    lowest = float(np.min(points @ rotation[2]))
                    pose[:, 2] = _TABLE_TOP_Z + 0.004 - lowest
            tensor = self._torch.as_tensor(pose, dtype=root_pose.dtype, device=root_pose.device)
            zeros = self._torch.zeros_like(velocity)
            if kind == "articulation":
                view.set_root_transforms(tensor, indices)
                view.set_root_velocities(zeros, indices)
                self._record_authored_task_reset_write(path, transform_xyzw=tensor, velocity=zeros)
            else:
                self._write_rigid_reset_state(path, view, indices, tensor, zeros)
        self._xy2_capture_button_baseline = True

    def _refresh_object_poses(self) -> None:
        super()._refresh_object_poses()
        if self._xy2_capture_button_baseline:
            readings = {name: self._physics_dof_position(name)
                        for name in ("mouse_joint_0", "mouse_joint_1")}
            if all(value is not None for value in readings.values()):
                self._c05_button_initial_positions = {name: float(value[0]) for name, value in readings.items()}
                self._xy2_capture_button_baseline = False

    def _fallback_pose(self, name: str) -> np.ndarray:
        if self._xy2_layout is not None and name in {"mouse", "link_3", "mousepad"}:
            xy = self._xy2_layout.pad_xy if name == "mousepad" else self._xy2_layout.mouse_xy
            quat = [1., 0., 0., 0.] if name == "mousepad" else [.5, .5, .5, .5]
            return np.asarray([*xy, .89 if name == "mousepad" else .892, *quat])
        return super()._fallback_pose(name)

    def _xy2_laptop_metrics(self):
        metrics = dict(self._fu015_last_metrics or self._refresh_fu015_metrics())
        force = float(self._fu015_raw_contact_diagnostic.get("left_display_force_n", 0.0))
        left_contact = bool(np.isfinite(force) and force >= REQUIRED_CONTACT_FORCE_N)
        metrics["display_owner_ready"] = bool(
            (metrics["right_display_contact"] and self._fu015_bilateral_owner_gate.ready)
            or (left_contact and self._xy2_left_owner_gate.ready))
        metrics["success_now"] = laptop_retained(metrics)
        return metrics

    def _compute_success_now(self) -> bool:
        if self._xy2_last_evaluated_frame == self.env_step:
            return self._xy2_success_now
        self._xy2_last_evaluated_frame = self.env_step
        progress = self._xy2_progress
        previous_phase = progress.phase
        laptop = self._xy2_laptop_metrics()
        mouse = self._xy2_mouse_click_metrics(
            update_failure=progress.phase == 1,
        )
        self._last_c05_success_metrics = mouse
        neutral = all(self._c05_button_travel(name) < _C05_BUTTON_PRESS_THRESHOLD
                      for name in ("mouse_joint_0", "mouse_joint_1"))
        self._xy2_success_now = progress.update(
            frame=int(self.env_step), laptop=laptop,
            mouse_xy=self._pose_for_role("target")[:2], mouse=mouse,
            buttons_neutral=neutral,
        )
        if previous_phase == 0 and progress.phase == 1:
            self._c05_mouse_start_xy = progress.mouse_start_xy.copy()
            self._c05_failed_early_click = False
        return self._xy2_success_now

    def _compute_failure(self) -> bool:
        source_failure = super()._compute_failure()
        if source_failure and not self._xy2_progress.failure_reason:
            self._xy2_progress.failure_reason = "fu015_physical_failure"
        failed = bool(self._xy2_progress.failure_reason)
        if failed and not self._xy2_failure_reported:
            import json
            print("[HandsOn/XY2] " + json.dumps({
                "event": "first_terminal", "reason": self._xy2_progress.failure_reason,
                "phase": self._xy2_progress.phase, "frame": int(self.env_step),
            }), flush=True)
            self._xy2_failure_reported = True
        return failed

    def _success_steps_required(self) -> int:
        # The two original windows are already evaluated by XY2Progress.
        return 1

    def _time_limit_is_truncation(self) -> bool:
        return True

    def _build_observation_space(self):
        result = super()._build_observation_space()
        for name, layout in RECORDING_LAYOUTS.items():
            result.spaces[name] = spaces.Box(-np.inf, np.inf, shape=(len(layout),), dtype=np.float64)
        return result

    def _compute_observation(self):
        obs = super()._compute_observation()
        p = self._xy2_progress
        obs["xy2_history"] = np.asarray([
            p.phase, p.laptop_steps, p.mouse_steps, p.buttons_armed,
            p.a_completed_frame, *p.mouse_start_xy,
            self._c05_failed_early_click, self._xy2_capture_button_baseline,
            self._c05_button_initial_positions["mouse_joint_0"],
            self._c05_button_initial_positions["mouse_joint_1"],
            self._xy2_last_evaluated_frame, FAILURE_CODES[p.failure_reason],
            p.left_click_seen,
        ], dtype=np.float64)
        obs["xy2_laptop_control"] = np.asarray([
            self._fu015_drive_target_deg, self._fu015_assist_active,
            self._fu015_assist_substeps, self._fu015_bilateral_owner_gate.consecutive_substeps,
            self._fu015_previous_right_tcp_z_m if self._fu015_previous_right_tcp_z_m is not None else self._right_tcp_pose[2],
            self._fu015_right_command_delta_z_m, self._fu015_right_opening_intent,
            self._fu015_right_display_contact_active, self._fu015_hinge_contact_lock_active,
            self._fu015_hinge_lock_angle_deg, self._fu015_stability_probe_substeps,
        ], dtype=np.float64)
        obs["xy2_left_laptop_control"] = np.asarray([
            self._xy2_previous_left_tcp_z if self._xy2_previous_left_tcp_z is not None else self._left_tcp_pose[2],
            self._xy2_left_opening_intent, self._xy2_left_display_contact,
            self._xy2_left_owner_gate.consecutive_substeps,
        ], dtype=np.float64)
        obs["xy2_fu015_state"] = obs["state"]["fu015_task_state"].astype(np.float64)
        # FU015 adds float64 poses to float32 shared spaces. Preserve values,
        # while making XY2's declared public observation contract exact.
        obs["state"] = {
            key: np.asarray(obs["state"][key], dtype=space.dtype)
            for key, space in self.observation_space["state"].spaces.items()
        }
        return obs

    def get_recording_schema(self):
        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        fields = {name: dict(value) for name, value in result["fields"].items()}
        for name, layout in RECORDING_LAYOUTS.items():
            fields[name] = {
                "shape": [len(layout)], "dtype": "float64",
                "source": "xy2_action_aligned_task_state", "layout": list(layout),
                "visibility": "hidden_task_logic", "replay_domain": "task_state",
                "physical_replay": True, "replay_restore_supported": False,
                "replay_atol": RECORDING_ATOLS[name],
            }
        result["fields"] = fields
        result["task_state"] = {
            "task_name": TASK_NAME, "field_names": list(RECORDING_LAYOUTS),
            "hidden_field_names": list(RECORDING_LAYOUTS),
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return result

    def _task_info(self):
        return {
            **super()._task_info(), **self._last_c05_success_metrics,
            "task_id": TASK_ID, "task_name": TASK_NAME, "source_task_ids": SOURCE_TASK_IDS,
            "xy2_phase": self._xy2_progress.phase,
            "xy2_laptop_hold_steps": self._xy2_progress.laptop_steps,
            "xy2_mouse_hold_steps": self._xy2_progress.mouse_steps,
            "xy2_left_click_seen": self._xy2_progress.left_click_seen,
            "xy2_first_terminal_reason": self._xy2_progress.failure_reason,
            "xy2_layout": self._xy2_layout.to_dict(),
        }

    def _task_status_text(self):
        metrics = self._xy2_laptop_metrics()
        return (f"Angle {metrics['hinge_deg']:.1f} / {metrics['target_open_deg']:.0f} deg "
                f"(+/-{TARGET_TOLERANCE_DEG:g})")

    @staticmethod
    def _success_condition_display_label(row):
        # XY2 rows already contain ASCII operator text, including the target.
        return str(row["label"])

    def _success_condition_rows(self):
        metrics = self._xy2_laptop_metrics()
        p = self._xy2_progress
        mouse = self._last_c05_success_metrics or self._xy2_mouse_click_metrics(update_failure=False)
        return [
            self._condition_row("xy2_laptop",
                f"A: Hold target {p.laptop_steps}/{SUCCESS_HOLD_STEPS}", p.phase > 0),
            self._condition_row("xy2_retained", "Keep laptop angle and base stable", laptop_retained(metrics)),
            self._condition_row("c05_mouse_on_pad", "Mouse supported on pad", mouse["c05_mouse_on_pad"]),
            self._condition_row("xy2_left_click_seen", "Left click on pad registered", p.left_click_seen),
            self._condition_row("xy2_complete",
                f"Final stability {p.mouse_steps}/{_C05_SUCCESS_STEPS}", self._xy2_success_now),
        ]
