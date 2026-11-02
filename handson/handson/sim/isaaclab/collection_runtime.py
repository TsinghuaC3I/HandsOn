"""Shared runtime behavior across the archived evaluation environments.

Only methods with identical syntax and matching library bindings are shared.
Task-specific constants, reset distributions, control variants and native
success conditions remain on their environment classes when they differ.
The extraction fixture preserves the pre-refactor method fingerprints.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np


class CollectionRuntimeSupport:
    """Common methods; environment subclasses own all state and configuration."""

    def _resolve_target_object(self, target_object: str | None) -> str:
        if target_object:
            value = str(target_object).strip()
            if value in self._spec.target_objects:
                return value
        return self._spec.target_objects[0]

    def _task_specific_isaac_kit_args(self) -> str | None:
        """Return optional Kit settings that must be present before launch."""
        return None

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        """Author extra task assets before the first PhysX reset."""
        return None

    def _init_task_specific_runtime(self) -> None:
        """Create task tensor views after the first PhysX reset."""
        return None

    def _configure_named_reset_stability(
        self,
        sim_utils,
        names: tuple[str, ...],
        *,
        linear_damping: float,
        angular_damping: float,
        sleep_threshold: float,
        warning_label: str,
        strip_redundant_articulation_root: bool = False,
        locked_rot_axis: int | None = None,
    ) -> int:
        """Apply wakeable PhysX damping to independent task rigid bodies."""
        from pxr import PhysxSchema, UsdPhysics

        paths: list[str] = []
        for name in names:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not tune {warning_label} reset stability for {name}"
                )
                continue
            path = str(prim.GetPath())
            paths.append(path)
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if not paths:
            return 0
        sim_utils.update_stage()

        tuned = 0
        for path in paths:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            if strip_redundant_articulation_root:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                ("GetLinearDampingAttr", "CreateLinearDampingAttr", linear_damping),
                ("GetAngularDampingAttr", "CreateAngularDampingAttr", angular_damping),
                ("GetSleepThresholdAttr", "CreateSleepThresholdAttr", sleep_threshold),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            if locked_rot_axis is not None:
                locked_rot_attr = rigid_api.GetLockedRotAxisAttr()
                if not locked_rot_attr or not locked_rot_attr.IsValid():
                    locked_rot_attr = rigid_api.CreateLockedRotAxisAttr()
                locked_rot_attr.Set(int(locked_rot_axis))
            # MJCF free joints are imported as one-link articulation roots in
            # some cached USDs.  Their articulation sleep opinion overrides
            # the link-level value, so author the same physical threshold at
            # both levels while leaving the bodies wakeable during collection.
            if UsdPhysics.ArticulationRootAPI(prim):
                articulation_api = PhysxSchema.PhysxArticulationAPI(prim)
                if not articulation_api:
                    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                sleep_attr = articulation_api.GetSleepThresholdAttr()
                if not sleep_attr or not sleep_attr.IsValid():
                    sleep_attr = articulation_api.CreateSleepThresholdAttr()
                sleep_attr.Set(sleep_threshold)
            tuned += 1
        return tuned

    def _c04_right_hand_body_paths(self) -> list[str]:
        if self._stage is None:
            return []

        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath("/World/HandsOnRobots/RightRobot")
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if name == "right_palm_link" or name.startswith("right_finger"):
                paths.append(str(prim.GetPath()))
        return paths

    def _c04_contact_force(self, view: Any | None) -> float:
        if view is None:
            return 0.0
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            forces = np.asarray(matrix, dtype=np.float64)
            if forces.size == 0:
                return 0.0
            return float(np.linalg.norm(forces, axis=-1).sum())
        except Exception:
            return 0.0

    def _c04_trigger_contact_force(self) -> float:
        return self._c04_contact_force(self._c04_trigger_contact_view)

    def _c04_grasp_contact_force(self) -> float:
        return float(
            sum(self._c04_contact_force(view) for view in self._c04_grasp_contact_views)
        )

    def _c06_right_hand_shutter_contact_force(self) -> float:
        return self._c04_contact_force(self._c06_shutter_contact_view)

    def _c12_right_hand_block_contact_force(self) -> float:
        return float(
            sum(
                self._c04_contact_force(view)
                for view in self._c12_block_contact_views.values()
            )
        )

    def _reset_c04_trigger_cycle(self) -> None:
        self._c04_trigger_phase = "idle"
        self._c04_trigger_cycle_latched = False
        self._c04_trigger_contact_seen = False
        self._c04_grasp_seen = False
        self._c04_trigger_contact_grace = 0
        self._c04_grasp_contact_grace = 0
        self._c04_grasp_ready_steps = 0
        self._c04_trigger_armed = False
        self._c04_trigger_pulled = False
        self._c04_water_emitting = False
        self._c04_failed_trigger_outside = False
        self._last_c04_success_metrics = {}

    def _update_c04_spray_visual(self) -> None:
        if self.task_name != "t50_water_plant":
            return
        self._set_c04_spray_visible(self._c04_water_emitting)

    def _control_robot(self, side: str):
        try:
            return self._robots[side]
        except KeyError as exc:
            raise RuntimeError(f"IsaacLab {side} robot is not initialized") from exc

    @staticmethod
    def _tensor_view(value: Any) -> Any:
        return value.torch if hasattr(value, "torch") else value

    def _reset_robot_body_points(self, side: str) -> tuple[np.ndarray, ...]:
        """Return live reset-home arm/hand body origins for clearance sampling."""
        points: list[np.ndarray] = []
        robot = self._robots.get(side)
        if robot is not None:
            try:
                body_names = list(robot.body_names)
                body_poses = robot.data.body_pose_w[0]
                for index, name in enumerate(body_names):
                    normalized = str(name).lower()
                    if normalized in {f"base_{side}", f"base_{side}/base_{side}"}:
                        continue
                    point = body_poses[int(index), :3]
                    if hasattr(point, "detach"):
                        point = point.detach().cpu().numpy()
                    point_np = np.asarray(point, dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(point_np)):
                        points.append(point_np)
            except Exception:
                points = []
        if points:
            return tuple(points)

        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        return (np.asarray(fallback[:3], dtype=np.float64).copy(),)

    @staticmethod
    def _point_segment_distance_xy(
        point_xy: np.ndarray,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
    ) -> float:
        start = np.asarray(start_xy, dtype=np.float64)
        delta = np.asarray(end_xy, dtype=np.float64) - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-12:
            return float(np.linalg.norm(np.asarray(point_xy) - start))
        fraction = float(
            np.clip(
                np.dot(np.asarray(point_xy, dtype=np.float64) - start, delta)
                / denominator,
                0.0,
                1.0,
            )
        )
        return float(
            np.linalg.norm(
                np.asarray(point_xy, dtype=np.float64) - (start + fraction * delta)
            )
        )

    def _c04_reset_spray_pose(self) -> tuple[np.ndarray, float]:
        xy_by_name, yaw_degrees = self._c04_reset_layout()
        return xy_by_name["spray"].copy(), yaw_degrees

    @staticmethod
    def _reset_transform_state_for_path(
        path: str,
        rigid_states: Mapping[str, tuple[Any, Any, Any, Any]],
        articulation_states: Mapping[str, tuple[Any, Any, Any, Any]],
    ) -> tuple[str, Any, Any, Any, Any] | None:
        """Prefer an articulation root writer over a child rigid-body writer."""
        state = articulation_states.get(path)
        if state is not None:
            return ("articulation", *state)
        state = rigid_states.get(path)
        if state is not None:
            return ("rigid", *state)
        return None

    def _write_rigid_reset_state(
        self,
        path: str,
        view: Any,
        indices: Any,
        transforms: Any,
        velocities: Any,
    ) -> None:
        """Write a rigid reset pose without assigning velocity to kinematics."""
        view.set_transforms(transforms, indices)
        if path not in self._task_kinematic_body_paths:
            view.set_velocities(velocities, indices)
            self._record_authored_task_reset_write(
                path,
                transform_xyzw=transforms,
                velocity=velocities,
            )
        else:
            self._record_authored_task_reset_write(
                path,
                transform_xyzw=transforms,
            )

    @staticmethod
    def _c08_point_segment_distance(
        point_xy: np.ndarray,
        segment_start: np.ndarray,
        segment_end: np.ndarray,
    ) -> float:
        point = np.asarray(point_xy, dtype=np.float64).reshape(2)
        start = np.asarray(segment_start, dtype=np.float64).reshape(2)
        end = np.asarray(segment_end, dtype=np.float64).reshape(2)
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + fraction * segment)))

    @classmethod
    def _a01_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._a01_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    @classmethod
    def _a02_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._a02_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    @classmethod
    def _a03_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._a03_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    @classmethod
    def _b01_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._b01_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    @staticmethod
    def _free_body_names(task_mjcf: Path) -> tuple[str, ...]:
        root = ET.parse(task_mjcf).getroot()
        names: list[str] = []
        for body in root.iter("body"):
            has_free_joint = any(
                child.tag == "freejoint"
                or (child.tag == "joint" and child.attrib.get("type") == "free")
                for child in body
            )
            name = body.attrib.get("name")
            if has_free_joint and name:
                names.append(name)
        return tuple(names)

    def _task_state_candidate_names(self, task_mjcf: Path) -> set[str]:
        names = set(self._spec.target_objects) | {
            self._spec.receptacle,
            "wood",
            "table",
        }
        names.update(self._task_extra_pose_names())
        names.update(self._free_body_names(task_mjcf))
        return names

    def _restore_task_scene_state(self) -> None:
        for (
            path,
            view,
            indices,
            root_pose,
            root_velocity,
            joint_pos,
            joint_vel,
        ) in self._task_reset_articulation_states:
            try:
                view.set_root_transforms(root_pose, indices)
                view.set_root_velocities(root_velocity, indices)
                if joint_pos.shape[-1] > 0:
                    view.set_dof_positions(joint_pos, indices)
                    view.set_dof_velocities(joint_vel, indices)
                self._record_authored_task_reset_write(
                    path,
                    transform_xyzw=root_pose,
                    velocity=root_velocity,
                    dof_position=joint_pos,
                    dof_velocity=joint_vel,
                )
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not restore articulation {path}: {exc}"
                )
        for path, view, indices, pose, velocity in self._task_reset_body_states:
            try:
                self._write_rigid_reset_state(
                    path,
                    view,
                    indices,
                    pose,
                    velocity,
                )
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not restore rigid body {path}: {exc}"
                )

    def _add_task_state_warning(self, detail: str) -> None:
        message = f"Task reset state warning: {detail}"
        if message not in self._isaac_warnings:
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _native_camera_prim_path(
        self,
        side: str,
        names: tuple[str, ...],
    ) -> str | None:
        if self._stage is None:
            return None
        from pxr import Usd, UsdGeom

        root_path = f"/World/HandsOnRobots/{side.capitalize()}Robot"
        root = self._stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        cameras = {
            prim.GetName(): str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Camera)
        }
        for name in names:
            if name in cameras:
                return cameras[name]
        return None

    def restore_recorded_initial_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        """Restore recorded table and task-object poses before action replay."""
        selector = np.asarray(
            state_parts.get("target_selector", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if selector.size:
            target_index = int(
                np.clip(
                    np.argmax(selector),
                    0,
                    len(self._spec.target_objects) - 1,
                )
            )
            self._target_object_name = self._spec.target_objects[target_index]

        delta = np.asarray(
            state_parts.get("table_delta_height", self.delta_h),
            dtype=np.float64,
        ).ravel()
        if delta.size:
            self.delta_h[:] = float(delta[0])
            self._restore_recorded_table_height(float(delta[0]))

        recorded_poses: dict[str, np.ndarray] = {}
        object_poses = np.asarray(
            state_parts.get("object_poses", np.zeros((0, 7))),
            dtype=np.float64,
        ).reshape(-1, 7)
        for name, pose in zip(self._spec.target_objects, object_poses):
            recorded_poses[name] = pose

        if (
            not selector.size
            and object_poses.size
            and "target_object_pose" in state_parts
        ):
            target_pose = np.asarray(
                state_parts.get("target_object_pose", np.zeros(0)),
                dtype=np.float64,
            ).ravel()
            if target_pose.size >= 3:
                candidates = list(zip(self._spec.target_objects, object_poses))
                if candidates:
                    distances = np.asarray(
                        [
                            np.linalg.norm(
                                np.asarray(pose[:3], dtype=np.float64) - target_pose[:3]
                            )
                            for _name, pose in candidates
                        ],
                        dtype=np.float64,
                    )
                    finite = np.isfinite(distances)
                    if np.any(finite):
                        target_index = int(
                            np.argmin(np.where(finite, distances, np.inf))
                        )
                        self._target_object_name = candidates[target_index][0]

        for state_key, role in self._spec.aliases.items():
            if state_key not in state_parts:
                continue
            if role == "target":
                name = self._target_object_name
            elif role == "receptacle":
                name = self._spec.receptacle
            else:
                name = role
            pose = np.asarray(state_parts[state_key], dtype=np.float64).ravel()
            if pose.size >= 7:
                recorded_poses[name] = pose[:7]

        restored_views: set[int] = set()
        for name, pose in recorded_poses.items():
            view = self._task_body_views.get(name)
            if view is None or id(view) in restored_views:
                continue
            if self._set_task_body_world_pose(view, pose):
                restored_views.add(id(view))

        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()

    def restore_recorded_visual_state(
        self,
        state_parts: Mapping[str, np.ndarray],
        *,
        robot_iterations: int = 8,
    ) -> dict[str, Any]:
        """Restore recorded task objects and drive robot visuals toward state.

        Historical ABC zarr episodes only store TCP and hand command state, not
        the complete PhysX articulation qpos.  This method is therefore a
        best-effort visual reconstruction path: it restores task objects exactly
        and repeatedly solves/writes the robot joint targets from the recorded
        TCP + gripper state before rendering.
        """
        self.restore_recorded_initial_state(state_parts)
        self._restore_recorded_robot_command_state(state_parts)
        self._snap_robot_to_command_state(robot_iterations)
        if self._sim is not None:
            self._sim.forward()
        if self._scene is not None:
            try:
                self._scene.update(0.0)
            except Exception:
                pass
        self._refresh_object_poses()
        return self._compute_observation()

    def _restore_recorded_robot_command_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> None:
        tcp_pose = np.asarray(
            state_parts.get("tcp_pose", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if tcp_pose.size >= 7:
            right_pose = tcp_pose[:7].astype(np.float64, copy=True)
            if np.all(np.isfinite(right_pose)) and not np.allclose(right_pose[:7], 0.0):
                quat_norm = float(np.linalg.norm(right_pose[3:7]))
                if quat_norm > 1e-8:
                    right_pose[3:7] /= quat_norm
                    self._right_tcp_pose = right_pose
                    self._right_pose_commanded = True
                    self._arm_pose_filters["right"] = None
        if tcp_pose.size >= 14:
            left_pose = tcp_pose[7:14].astype(np.float64, copy=True)
            if np.all(np.isfinite(left_pose)) and not np.allclose(left_pose[:7], 0.0):
                quat_norm = float(np.linalg.norm(left_pose[3:7]))
                if quat_norm > 1e-8:
                    left_pose[3:7] /= quat_norm
                    self._left_tcp_pose = left_pose
                    self._left_pose_commanded = True
                    self._arm_pose_filters["left"] = None

        gripper_pose = np.asarray(
            state_parts.get("gripper_pose", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if not gripper_pose.size:
            return

        if self._spec.bimanual and gripper_pose.size >= 32:
            right_size = min(16, self._right_hand.size)
            left_size = min(16, self._left_hand.size)
            if right_size:
                self._right_hand[:right_size] = gripper_pose[:right_size]
            if left_size:
                self._left_hand[:left_size] = gripper_pose[16 : 16 + left_size]
            return

        right_size = min(self._right_hand.size, gripper_pose.size)
        if right_size:
            self._right_hand[:right_size] = gripper_pose[:right_size]

    def _snap_robot_to_command_state(self, robot_iterations: int) -> None:
        if not self._robots or self._torch is None:
            return
        iterations = max(1, int(robot_iterations))
        for _ in range(iterations):
            self._apply_joint_targets()
            self._write_joint_targets_as_state()
            if self._sim is not None:
                self._sim.forward()
            if self._scene is not None:
                try:
                    self._scene.update(0.0)
                except Exception:
                    pass
            for robot in self._robots.values():
                try:
                    robot.update(0.0)
                except Exception:
                    pass

    def _write_joint_targets_as_state(self) -> None:
        for side, robot in self._robots.items():
            target = self._joint_targets.get(side)
            if target is None:
                continue
            try:
                joint_pos = target.clone()
                joint_vel = self._torch.zeros_like(joint_pos)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_position_target(joint_pos)
                robot.write_data_to_sim()
            except Exception as exc:
                message = f"{side} robot visual state restore failed: {exc}"
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _set_task_body_world_pose(self, view: Any, pose_wxyz: np.ndarray) -> bool:
        if self._torch is None:
            return False
        try:
            transforms = view.get_transforms().clone()
            velocities = view.get_velocities().clone()
            original_transforms = transforms.clone()
            original_velocities = velocities.clone()
            pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
            transforms[:, :3] = self._torch.as_tensor(
                pose[:3],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            transforms[:, 3:7] = self._torch.as_tensor(
                [pose[4], pose[5], pose[6], pose[3]],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            try:
                # A velocity failure must not leave a transform-only teleport.
                view.set_velocities(
                    self._torch.zeros_like(velocities),
                    indices,
                )
                view.set_transforms(transforms, indices)
            except Exception:
                try:
                    view.set_velocities(original_velocities, indices)
                    view.set_transforms(original_transforms, indices)
                except Exception:
                    pass
                raise
            subject = next(
                (
                    name
                    for name, candidate in self._task_body_views.items()
                    if candidate is view
                ),
                None,
            )
            self._record_trajectory_state_write(
                "_set_task_body_world_pose",
                subject=subject,
            )
            return True
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not restore recorded object pose: {exc}"
            )
            return False

    def _apply_reset_visual_randomization(self) -> None:
        from .paper_reset_visual_rng import preserve
        from .default_appearance import refresh_environment

        preserve(self)
        refresh_environment(self)

    def _step_isaaclab(self) -> None:
        if self._sim is None:
            return
        self._resume_timeline_if_paused(source="Timeline pause")
        self._update_b01_release_assist()
        for robot in self._robots.values():
            robot.write_data_to_sim()
        # Defer rendering until render(), after every physics substep has
        # finished and the wrist-mount body tensors contain the latest pose.
        self._sim.step(render=False)
        if self.task_name == "t08_hammer_nail":
            # A hammer impact can begin and end inside one 30 Hz control
            # interval. Sample it at the PhysX rate so a real strike is not
            # missed while still requiring the contact tensor below.
            self._refresh_object_poses()
            self._update_c01_hammer_hit_state()
        self._update_c04_trigger_state()
        self._update_c04_spray_visual()
        for robot in self._robots.values():
            robot.update(self._sim_dt)
        self._after_task_physics_substep()

    def _after_task_physics_substep(self) -> None:
        """Task-only high-frequency event hook; robot stepping stays shared."""
        if self.task_name == "t08_hammer_nail":
            self._update_c01_grasp_assists()
        elif self.task_name == "c07_pitcher_pour_cup":
            self._update_c07_pitcher_grasp_assist()
        elif self.task_name == "c08_tongs_ice_cubes":
            self._update_c08_tongs_assist()
        elif self.task_name == "c09_staple_paper_sheets":
            self._update_c09_paper_assist()
        elif self.task_name == "t02_connect_charger_and_cable":
            self._update_c10_connector_assist()
        elif self.task_name == "c11_xylophone_sequence":
            self._update_c11_mallet_assist()
            self._update_c11_hit_events()
        elif self.task_name == "t11_line_up_blocks":
            if not self._c12_direct_hand_contact and self._c12_direct_hand_touch_now():
                self._c12_direct_hand_contact = True
                self._manual_reset_requested = True
                print(
                    "[HandsOn/IsaacLab] C12 direct hand/block contact; resetting episode",
                    flush=True,
                )
            elif not self._c12_direct_hand_contact:
                self._update_c12_tool_assist()

    def _hand_command_max(self, side: str) -> float:
        values = self._right_hand if side == "right" else self._left_hand
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _raw_right_hand_command_max(self) -> float:
        values = np.asarray(
            getattr(self, "_c_assist_raw_right_hand", self._right_hand),
            dtype=np.float64,
        )
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _actual_hand_joint_pose(self, side: str) -> np.ndarray:
        """Read a visible hand pose at contact, falling back to the raw glove."""
        raw_name = f"_c_assist_raw_{side}_hand"
        commanded = self._right_hand if side == "right" else self._left_hand
        fallback = (
            np.asarray(
                getattr(self, raw_name, commanded),
                dtype=np.float64,
            )
            .reshape(-1)
            .copy()
        )
        robot = self._robots.get(side)
        joint_ids = self._hand_joint_ids.get(side, [])
        if robot is None or not joint_ids:
            return fallback
        try:
            values = robot.data.joint_pos[0, joint_ids]
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            actual = np.asarray(values, dtype=np.float64).reshape(-1)
        except Exception:
            return fallback
        if actual.size != fallback.size or not np.all(np.isfinite(actual)):
            return fallback
        return actual.copy()

    def _actual_right_hand_joint_pose(self) -> np.ndarray:
        """Compatibility wrapper for the established C07/C08 right hand."""
        return self._actual_hand_joint_pose("right")

    def _raw_hand_command_max(self, side: str) -> float:
        values = np.asarray(
            getattr(
                self,
                f"_c_assist_raw_{side}_hand",
                self._right_hand if side == "right" else self._left_hand,
            ),
            dtype=np.float64,
        )
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _raw_tcp_pose(self, side: str) -> np.ndarray:
        fallback = self._right_tcp_pose if side == "right" else self._left_tcp_pose
        return (
            np.asarray(
                getattr(self, f"_c_assist_raw_{side}_tcp_pose", fallback),
                dtype=np.float64,
            )
            .reshape(7)
            .copy()
        )

    def _current_body_pose(self, name: str) -> np.ndarray | None:
        pose = self._physics_body_pose(name)
        if pose is None:
            pose = self._poses.get(name)
        if pose is None:
            return None
        pose_np = np.asarray(pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(pose_np)):
            return None
        return pose_np.copy()

    @staticmethod
    def _rotation_error_vector(
        target_rotation: np.ndarray,
        current_rotation: np.ndarray,
    ) -> np.ndarray:
        error = (
            np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
            @ np.asarray(current_rotation, dtype=np.float64).reshape(3, 3).T
        )
        vector = 0.5 * np.asarray(
            [
                error[2, 1] - error[1, 2],
                error[0, 2] - error[2, 0],
                error[1, 0] - error[0, 1],
            ],
            dtype=np.float64,
        )
        sin_angle = float(np.linalg.norm(vector))
        cos_angle = float(np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(math.atan2(sin_angle, cos_angle))
        if sin_angle <= 1e-8 or angle <= 1e-8:
            return np.zeros(3, dtype=np.float64)
        return vector * (angle / sin_angle)

    @staticmethod
    def _wuji_tip_names(side: str) -> tuple[str, tuple[str, ...]]:
        return (
            f"{side}_finger1_tip",
            tuple(f"{side}_finger{index}_tip" for index in range(2, 6)),
        )

    @staticmethod
    def _point_box_surface_metrics(
        point: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return signed distance and outward normal for an axis-aligned box."""
        local = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        half = np.asarray(half_extents, dtype=np.float64).reshape(3)
        outside = np.maximum(np.abs(local) - half, 0.0)
        outside_distance = float(np.linalg.norm(outside))
        if outside_distance > 1e-12:
            closest = np.clip(local, -half, half)
            delta = local - closest
            return outside_distance, delta / max(float(np.linalg.norm(delta)), 1e-12)

        face_clearance = half - np.abs(local)
        axis = int(np.argmin(face_clearance))
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0 if float(local[axis]) >= 0.0 else -1.0
        return -float(face_clearance[axis]), normal

    @staticmethod
    def _segment_intersects_aabb(
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
        *,
        margin: float = 0.0,
    ) -> bool:
        first = np.asarray(start, dtype=np.float64).reshape(3) - np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        second = np.asarray(end, dtype=np.float64).reshape(3) - np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        half = np.asarray(half_extents, dtype=np.float64).reshape(3) + max(
            float(margin),
            0.0,
        )
        direction = second - first
        lower = 0.0
        upper = 1.0
        for axis in range(3):
            if abs(float(direction[axis])) <= 1e-12:
                if abs(float(first[axis])) > float(half[axis]):
                    return False
                continue
            inverse = 1.0 / float(direction[axis])
            near = (-float(half[axis]) - float(first[axis])) * inverse
            far = (float(half[axis]) - float(first[axis])) * inverse
            if near > far:
                near, far = far, near
            lower = max(lower, near)
            upper = min(upper, far)
            if lower > upper:
                return False
        return True

    def _assist_hand_tracking_ready(
        self,
        side: str,
        state: dict[str, Any],
        *,
        position_tolerance: float,
        angle_tolerance: float,
    ) -> bool:
        target_value = state.get("hand_target_matrix")
        if target_value is None:
            state["hand_tracking_ready"] = True
            return True
        if not bool(state.get("hand_target_installed")):
            # The target is only copied into the robot command by
            # _postprocess_hand_action at the next control frame.  Advancing
            # the object during the intervening PhysX substeps would make it
            # visibly move before the hand has even received that command.
            state["hand_tracking_ready"] = False
            return False
        installed_value = state.get("hand_target_installed_matrix")
        if installed_value is not None:
            installed = np.asarray(
                installed_value,
                dtype=np.float64,
            ).reshape(4, 4)
            requested = np.asarray(
                target_value,
                dtype=np.float64,
            ).reshape(4, 4)
            if not np.allclose(
                installed,
                requested,
                rtol=0.0,
                atol=1e-10,
            ):
                # A PhysX substep has staged a newer guide target than the one
                # copied into the robot command at the control-frame boundary.
                # The object must wait for that exact target to be installed.
                state["hand_target_installed"] = False
                state["hand_tracking_ready"] = False
                return False
        palm = self._robot_link_pose_matrix(side)
        if palm is None:
            state["hand_tracking_ready"] = False
            return False
        target = np.asarray(
            installed_value if installed_value is not None else target_value,
            dtype=np.float64,
        ).reshape(4, 4)
        position_error = float(np.linalg.norm(palm[:3, 3] - target[:3, 3]))
        angle_error = float(
            np.linalg.norm(
                self._rotation_error_vector(
                    target[:3, :3],
                    palm[:3, :3],
                )
            )
        )
        ready = bool(
            position_error <= float(position_tolerance)
            and angle_error <= float(angle_tolerance)
        )
        state["hand_tracking_position_error"] = position_error
        state["hand_tracking_angle_error"] = angle_error
        state["hand_tracking_ready"] = ready
        return ready

    @staticmethod
    def _map_raw_hand_handoff(
        raw: np.ndarray,
        raw_anchor: np.ndarray,
        hand_anchor: np.ndarray,
    ) -> np.ndarray:
        """Apply raw translation/rotation deltas without a rotating lever arm."""
        current = np.asarray(raw, dtype=np.float64).reshape(4, 4)
        source = np.asarray(raw_anchor, dtype=np.float64).reshape(4, 4)
        destination = np.asarray(
            hand_anchor,
            dtype=np.float64,
        ).reshape(4, 4)
        target = destination.copy()
        target[:3, 3] = destination[:3, 3] + current[:3, 3] - source[:3, 3]
        target[:3, :3] = current[:3, :3] @ source[:3, :3].T @ destination[:3, :3]
        return target

    @classmethod
    def _bounded_pose_step(
        cls,
        current: np.ndarray,
        goal: np.ndarray,
        *,
        dt: float,
        max_linear_speed: float,
        max_angular_speed: float,
    ) -> np.ndarray:
        start = np.asarray(current, dtype=np.float64).reshape(4, 4)
        end = np.asarray(goal, dtype=np.float64).reshape(4, 4)
        distance = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
        angle = float(
            np.linalg.norm(
                cls._rotation_error_vector(
                    end[:3, :3],
                    start[:3, :3],
                )
            )
        )
        alpha = 1.0
        step_dt = max(float(dt), 0.0)
        if distance > 1e-9:
            alpha = min(
                alpha,
                float(max_linear_speed) * step_dt / distance,
            )
        if angle > 1e-9:
            alpha = min(
                alpha,
                float(max_angular_speed) * step_dt / angle,
            )
        return cls._c07_interpolate_pose(start, end, alpha)

    @classmethod
    def _c07_lock_pitcher_target_xy(
        cls,
        target: np.ndarray,
        cup_xy: np.ndarray,
    ) -> np.ndarray:
        locked = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
        outlet = cls._c07_spout_outlet_world(locked)
        locked[:2, 3] += np.asarray(cup_xy, dtype=np.float64).reshape(2) - outlet[:2]
        return locked

    @staticmethod
    def _c07_particle_settle_local(index: int) -> np.ndarray:
        slots = np.asarray(
            [
                (0.000, 0.000, -0.022),
                (0.010, 0.000, -0.010),
                (-0.010, 0.000, -0.010),
                (0.000, 0.010, -0.010),
                (0.000, -0.010, -0.010),
                (0.008, 0.008, 0.004),
                (-0.008, 0.008, 0.004),
                (0.000, -0.008, 0.004),
            ],
            dtype=np.float64,
        )
        return slots[int(index) % len(slots)].copy()

    @staticmethod
    def _c07_particle_release_tilt_degrees(
        state: dict[str, Any],
        pitcher_target: np.ndarray,
    ) -> float:
        del state
        target_matrix = np.asarray(
            pitcher_target,
            dtype=np.float64,
        ).reshape(4, 4)
        # atan2 preserves the full 0..180-degree arc.  asin would fold a
        # 150-degree pour back to 30 degrees and disable the drain again.
        total_tilt = math.degrees(
            math.atan2(
                float(-target_matrix[2, 1]),
                float(target_matrix[2, 2]),
            )
        )
        return float(np.clip(total_tilt, 0.0, 180.0))

    @staticmethod
    def _c08_tongs_tip_positions(
        root_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fixed_local = np.asarray([0.135, -0.018, 0.012], dtype=np.float64)
        moving_root = np.asarray([-0.120, 0.018, 0.012], dtype=np.float64)
        moving_arm = np.asarray([0.255, 0.0, 0.0], dtype=np.float64)
        cosine = math.cos(float(joint_angle))
        sine = math.sin(float(joint_angle))
        moving_rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        moving_local = moving_root + moving_rotation @ moving_arm
        root = np.asarray(root_matrix, dtype=np.float64).reshape(4, 4)
        fixed = root[:3, 3] + root[:3, :3] @ fixed_local
        moving = root[:3, 3] + root[:3, :3] @ moving_local
        return fixed, moving, 0.5 * (fixed + moving)

    def _c08_tongs_scene_clearance(
        self,
        root_matrix: np.ndarray,
        joint_angle: float,
    ) -> float:
        """Return the minimum signed clearance from the full tongs to the table."""
        # The tongs must intentionally contact ice and may pass close to the
        # cup during pickup/drop. Only the table remains a blocking surface.
        minimum_clearance = math.inf
        table_top = float(self._current_table_top_z())

        for segment, radius in self._c08_tongs_collision_capsules(
            root_matrix,
            joint_angle,
        ):
            points = self._c08_capsule_centerline_samples(segment)
            minimum_clearance = min(
                minimum_clearance,
                float(np.min(points[:, 2] - radius - table_top)),
            )
        for center, rotation, half_extents in self._c08_tongs_collision_boxes(
            root_matrix,
            joint_angle,
        ):
            # The support of an OBB along world Z is the sum of each local
            # half extent projected onto that axis.  Treating the new flat
            # boxes as their former capsules would miss a corner whenever the
            # tongs pitch or roll during a swept hand command.
            vertical_support = float(np.dot(np.abs(rotation[2, :]), half_extents))
            minimum_clearance = min(
                minimum_clearance,
                float(center[2] - vertical_support - table_top),
            )
        return float(minimum_clearance)

    @classmethod
    def _c08_inner_pad_metrics(
        cls,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        ice_position: np.ndarray,
        ice_rotation: np.ndarray | None = None,
    ) -> tuple[dict[str, float | bool], dict[str, float | bool]]:
        fixed_segment, moving_segment = cls._c08_tongs_pad_segments(
            tongs_matrix,
            joint_angle,
        )
        rotation = (
            np.eye(3, dtype=np.float64)
            if ice_rotation is None
            else np.asarray(ice_rotation, dtype=np.float64).reshape(3, 3)
        )
        return (
            cls._c08_pad_inner_surface_metrics(
                fixed_segment,
                moving_segment,
                ice_position,
                rotation,
            ),
            cls._c08_pad_inner_surface_metrics(
                moving_segment,
                fixed_segment,
                ice_position,
                rotation,
            ),
        )

    @classmethod
    def _c08_lock_tongs_target_xy(
        cls,
        target: np.ndarray,
        joint_angle: float,
        cup_xy: np.ndarray,
    ) -> np.ndarray:
        locked = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
        _fixed, _moving, midpoint = cls._c08_tongs_tip_positions(
            locked,
            joint_angle,
        )
        locked[:2, 3] += np.asarray(cup_xy, dtype=np.float64).reshape(2) - midpoint[:2]
        return locked

    @staticmethod
    def _abort_c08_release(grasp: dict[str, Any]) -> None:
        """Give control back without detaching the held cube."""
        grasp.update(
            {
                "phase": "holding_ice",
                "pour_xy_lock": False,
                "pour_lock_pending": False,
                "pour_hand_target_installed": False,
                "hand_target_matrix": None,
                "hand_target_installed": False,
                "release_armed": False,
                "release_retry_blocked": True,
                "release_suspended": False,
                "reanchor_after_release_abort": True,
                "drop_hand_pose": None,
                "drop_tongs_pose": None,
                "drop_goal_pose": None,
                "align_elapsed_s": 0.0,
                "cup_missing_elapsed_s": 0.0,
                "release_elapsed_s": 0.0,
                "open_command_applied": False,
                "open_command_success_s": 0.0,
            }
        )
        grasp.pop("hand_target_installed_matrix", None)
        grasp.pop("hand_target_installed_frame", None)

    @staticmethod
    def _c09_other_paper(name: str) -> str:
        return "c09_paper_1" if name == "c09_paper_0" else "c09_paper_0"

    def _c09_grasp_side_for_paper(self, name: str) -> str | None:
        for side, state in self._c09_paper_grasps.items():
            if str(state.get("name")) == name:
                return side
        return None

    def _c09_clear_hand_targets(self) -> None:
        for state in self._c09_paper_grasps.values():
            state["hand_target_matrix"] = None
            state["hand_target_installed"] = False

    def _c09_pin_stapler_root(self) -> None:
        stapler = self._current_body_matrix("c09_stapler")
        if stapler is None:
            return
        if self._c09_stapler_anchor is None:
            self._c09_stapler_anchor = stapler.copy()
        anchor = np.asarray(
            self._c09_stapler_anchor,
            dtype=np.float64,
        ).reshape(4, 4)
        self._set_body_rigid_grasp_target(
            "c09_stapler",
            anchor,
            previous_target=anchor,
        )

    def _b01_hand_command_max(self) -> float:
        return self._hand_command_max("right")

    def _b02_nut_grasp_assist_enabled(self) -> bool:
        # Disabled while testing source-matched B02 MuJoCo parameters.  Keeping
        # the helper state readable in ``info`` makes it obvious during
        # collection that no velocity-based nut attachment is active.
        return False

    def _b02_wuji_fingertip_positions(self) -> dict[str, np.ndarray]:
        """Compatibility wrapper used by the existing right-hand tasks."""
        return self._wuji_fingertip_positions("right")

    def _set_deferred_robot_body_hidden(self, hidden: bool) -> int:
        if self._stage is None:
            return 0
        from pxr import UsdGeom

        if self._deferred_robot_body_visibility is None:
            self._deferred_robot_body_visibility = (
                self._deferred_robot_body_visibility_records()
            )
            print(
                "[HandsOn/IsaacLab] Deferred arm render body filter: "
                f"{len(self._deferred_robot_body_visibility)} robot visual prims",
                flush=True,
            )
        for path, original in self._deferred_robot_body_visibility:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            visibility = UsdGeom.Imageable(prim).GetVisibilityAttr()
            visibility.Set(UsdGeom.Tokens.invisible if hidden else original)
        return len(self._deferred_robot_body_visibility)

    def _refresh_object_poses(self) -> None:
        extras = {"wood", "table"}
        extras.update(self._task_extra_pose_names())
        for name in set(self._spec.target_objects) | {self._spec.receptacle} | extras:
            self._poses[name] = self._prim_pose(name, self._fallback_pose(name))
        if (
            self.task_name == "t14_move_and_click_mouse"
            and self._c05_capture_start_pending
        ):
            self._c05_mouse_start_xy = self._pose_for_role("target")[:2].copy()
            for name in ("mouse_joint_0", "mouse_joint_1"):
                reading = self._physics_dof_position(name)
                self._c05_button_initial_positions[name] = (
                    float(reading[0]) if reading is not None else 0.0
                )
            self._c05_capture_start_pending = False

    def _pose_for_role(self, role: str) -> np.ndarray:
        if role == "target":
            return self._poses.get(
                self._target_object_name, self._fallback_pose(self._target_object_name)
            )
        if role == "receptacle":
            return self._poses.get(
                self._spec.receptacle, self._fallback_pose(self._spec.receptacle)
            )
        return self._poses.get(role, self._fallback_pose(role))

    def _physics_body_pose(self, name: str) -> np.ndarray | None:
        view = self._task_body_views.get(name)
        if view is None:
            return None
        try:
            transform = view.get_transforms()[0]
            if hasattr(transform, "detach"):
                transform = transform.detach().cpu().numpy()
            transform = np.asarray(transform, dtype=np.float64).reshape(7)
            # PhysX tensor transforms use xyzw quaternion ordering.
            return np.concatenate(
                [transform[:3], transform[6:7], transform[3:6]]
            ).astype(np.float64)
        except Exception:
            return None

    def _physics_body_velocity(self, name: str) -> np.ndarray | None:
        view = self._task_body_views.get(name)
        if view is None:
            return None
        try:
            velocity = view.get_velocities()[0]
            if hasattr(velocity, "detach"):
                velocity = velocity.detach().cpu().numpy()
            return np.asarray(velocity, dtype=np.float64).reshape(6)
        except Exception:
            return None

    def _physics_dof_position(self, name: str) -> tuple[float, float] | None:
        entry = self._task_dof_views.get(name)
        if entry is None:
            entry = next(
                (
                    value
                    for dof_name, value in self._task_dof_views.items()
                    if name in dof_name
                ),
                None,
            )
        if entry is None:
            return None
        view, index, initial = entry
        try:
            positions = view.get_dof_positions()
            value = positions[0, int(index)]
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            return float(value), float(initial)
        except Exception:
            return None

    def _compute_success(self) -> bool:
        success_now = self._compute_success_now()
        if success_now:
            self._success_counter += 1
        else:
            self._success_counter = 0
        success = self._success_counter >= self._success_steps_required()
        self._update_success_condition_overlay()
        return success

    def _compute_failure(self) -> bool:
        return bool(
            (self.task_name == "t50_water_plant" and self._c04_failed_trigger_outside)
            or (
                self.task_name == "t14_move_and_click_mouse"
                and self._c05_failed_early_click
            )
        )

    def _c05_button_travel(self, name: str) -> float:
        reading = self._physics_dof_position(name)
        if reading is None:
            return 0.0
        baseline = float(self._c05_button_initial_positions.get(name, 0.0))
        return max(0.0, float(reading[0]) - baseline)

    def _body_local_point_world(
        self, body_name: str, local_point: np.ndarray
    ) -> np.ndarray:
        matrix = self._pose_matrix_for_name(body_name)
        point = np.asarray([*np.asarray(local_point, dtype=np.float64).reshape(3), 1.0])
        return (matrix @ point)[:3]

    @staticmethod
    def _point_segment_distance_3d(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        point_np = np.asarray(point, dtype=np.float64).reshape(3)
        start_np = np.asarray(start, dtype=np.float64).reshape(3)
        segment = np.asarray(end, dtype=np.float64).reshape(3) - start_np
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return float(np.linalg.norm(point_np - start_np))
        fraction = float(
            np.clip(np.dot(point_np - start_np, segment) / length_sq, 0.0, 1.0)
        )
        return float(np.linalg.norm(point_np - (start_np + fraction * segment)))

    def _c07_pitcher_pour_cup_success_now(self) -> bool:
        self._last_c07_success_metrics = self._c07_pitcher_pour_metrics()
        self._set_task_visual_names_visible(
            {f"c07_cup_goal_rim_{index}" for index in range(8)},
            bool(self._last_c07_success_metrics["c07_enough_inside"]),
        )
        return bool(self._last_c07_success_metrics["c07_success_now"])

    def _c08_tongs_ice_cubes_success_now(self) -> bool:
        self._last_c08_success_metrics = self._c08_tongs_ice_metrics()
        self._set_task_visual_names_visible(
            {
                "c08_cup_goal_rim_xp",
                "c08_cup_goal_rim_xn",
                "c08_cup_goal_rim_yp",
                "c08_cup_goal_rim_yn",
            },
            bool(self._last_c08_success_metrics["c08_all_ice_inside"]),
        )
        return bool(self._last_c08_success_metrics["c08_success_now"])

    def _c09_staple_paper_sheets_success_now(self) -> bool:
        self._last_c09_success_metrics = self._c09_staple_metrics()
        self._set_task_visual_names_visible(
            {"c09_staple_done"},
            bool(self._last_c09_success_metrics["c09_success_now"]),
        )
        return bool(self._last_c09_success_metrics["c09_success_now"])

    def _apply_c10_progressive_capture(
        self,
        poses: dict[str, np.ndarray],
        geometry: dict[str, Any],
    ) -> None:
        """Report capture state without magnetizing a connector held by nobody.

        The previous implementation drove any free rigid body inside the broad
        funnel to a canonical pose.  C10 guidance now lives in the connector's
        real hand-grasp state, so this compatibility hook is deliberately
        read-only.
        """
        self._c10_assist_metrics = {
            "c10_plug_capture_active": any(
                str(state.get("name")) == "c10_charger_plug"
                and str(state.get("phase")) != "carry"
                for state in self._c10_connector_grasps.values()
            ),
            "c10_cable_capture_active": any(
                str(state.get("name")) == "c10_charging_cable"
                and str(state.get("phase")) != "carry"
                for state in self._c10_connector_grasps.values()
            ),
        }

    def _c10_plug_latch_handoff_pending(self) -> bool:
        return any(
            str(state.get("name")) == "c10_charger_plug"
            and str(state.get("phase")) == "latched_handoff"
            and not bool(state.get("hand_target_installed"))
            for state in self._c10_connector_grasps.values()
        )

    def _set_c10_connector_kinematic(
        self,
        name: str,
        enabled: bool,
    ) -> bool:
        return self._set_task_body_kinematic(
            name,
            enabled,
            warning_label="C10 connector",
        )

    def _set_task_body_kinematic(
        self,
        name: str,
        enabled: bool,
        *,
        warning_label: str = "task body",
    ) -> bool:
        if self._stage is None:
            return False
        try:
            from pxr import UsdPhysics

            prim = self._find_rigid_body_prim(name)
            if prim is None:
                return False
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            attribute = rigid_api.GetKinematicEnabledAttr()
            if not attribute.IsValid():
                attribute = rigid_api.CreateKinematicEnabledAttr()
            attribute.Set(bool(enabled))
            return True
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not set {warning_label} kinematic state for {name}: {exc}"
            )
            return False

    def _c10_connection_metrics(self) -> dict[str, Any]:
        plug = self._poses.get(
            "c10_charger_plug", self._fallback_pose("c10_charger_plug")
        )
        cable = self._poses.get(
            "c10_charging_cable", self._fallback_pose("c10_charging_cable")
        )
        strip = self._poses.get(
            "c10_power_strip", self._fallback_pose("c10_power_strip")
        )
        geometry = self._c10_connection_geometry(plug, cable, strip)
        first_connected_now = bool(geometry["first_connected_now"])
        second_connected_now = bool(geometry["second_connected_now"])
        order_valid = bool(
            self._c10_first_connected_seen and not self._c10_order_violation
        )
        first_connected = bool(self._c10_plug_latched)
        second_connected = bool(self._c10_cable_latched)
        workspace_ok = bool(
            float(np.linalg.norm(strip[:2])) <= 1.0
            and plug[2] >= self._current_table_top_z() - 0.03
            and cable[2] >= self._current_table_top_z() - 0.03
        )
        success_now = bool(order_valid and second_connected and workspace_ok)
        return {
            "c10_first_connected": first_connected,
            "c10_second_connected": bool(second_connected),
            "c10_order_valid": bool(order_valid),
            "c10_workspace_ok": bool(workspace_ok),
            "c10_success_now": bool(success_now),
            "c10_first_connected_now": bool(first_connected_now),
            "c10_order_violation": bool(self._c10_order_violation),
            "c10_plug_latched": bool(self._c10_plug_latched),
            "c10_cable_latched": bool(self._c10_cable_latched),
            "c10_plug_latch_stable_steps": int(
                self._c10_latch_stable_steps["c10_charger_plug"]
            ),
            "c10_cable_latch_stable_steps": int(
                self._c10_latch_stable_steps["c10_charging_cable"]
            ),
            "c10_plug_strip_xy_error": float(geometry["plug_lateral_error"]),
            "c10_plug_strip_z_error": abs(float(geometry["plug_shoulder_gap"])),
            "c10_plug_insertion_depth": float(geometry["plug_insertion_depth"]),
            "c10_plug_axis_alignment": float(geometry["plug_axis_alignment"]),
            "c10_plug_secondary_alignment": float(geometry["plug_secondary_alignment"]),
            "c10_cable_plug_xy_error": float(geometry["cable_lateral_error"]),
            "c10_cable_plug_z_error": abs(float(geometry["cable_shoulder_gap"])),
            "c10_cable_insertion_depth": float(geometry["cable_insertion_depth"]),
            "c10_cable_axis_alignment": float(geometry["cable_axis_alignment"]),
            "c10_cable_secondary_alignment": float(
                geometry["cable_secondary_alignment"]
            ),
        }

    def _c11_expected_key(self) -> str | None:
        if self._c11_sequence_progress >= len(self._c11_target_sequence):
            return None
        return str(self._c11_target_sequence[self._c11_sequence_progress])

    def _c11_return_guided_target(
        self,
        carried: np.ndarray,
    ) -> np.ndarray:
        target = np.asarray(carried, dtype=np.float64).reshape(4, 4).copy()
        zone = self._current_body_matrix("c11_return_zone")
        if zone is None:
            return target
        target[:2, 3] += zone[:2, 3] - target[:2, 3]
        return target

    def _c11_key_contact_force(self, key_name: str) -> float:
        return self._c04_contact_force(self._c11_key_contact_views.get(key_name))

    def _c12_tool_block_contact_force(self, name: str) -> tuple[float, bool]:
        view = self._c12_tool_block_contact_views.get(name)
        if view is None:
            return 0.0, False
        return self._c04_contact_force(view), True

    def _c12_set_block_kinematic(self, name: str, enabled: bool) -> bool:
        applied = self._set_task_body_kinematic(
            name,
            enabled,
            warning_label="C12 block",
        )
        return bool(applied or self._stage is None)

    def _c12_pin_settled_blocks(self) -> bool:
        for name, target_value in tuple(self._c12_settled_targets.items()):
            target = np.asarray(
                target_value,
                dtype=np.float64,
            ).reshape(4, 4)
            applied, _linear, _angular = self._set_body_rigid_grasp_target(
                name,
                target,
                previous_target=target,
            )
            if not applied:
                self._manual_reset_requested = True
                return False
        return True

    @staticmethod
    def _clip_polygon_to_axis_aligned_box(
        polygon: np.ndarray,
        half_extents: np.ndarray,
    ) -> np.ndarray:
        clipped = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        for axis in range(2):
            for bound, keep_less in (
                (float(half_extents[axis]), True),
                (-float(half_extents[axis]), False),
            ):
                if len(clipped) == 0:
                    return clipped
                output: list[np.ndarray] = []
                previous = clipped[-1]
                previous_inside = (
                    float(previous[axis]) <= bound
                    if keep_less
                    else float(previous[axis]) >= bound
                )
                for current in clipped:
                    current_inside = (
                        float(current[axis]) <= bound
                        if keep_less
                        else float(current[axis]) >= bound
                    )
                    if current_inside != previous_inside:
                        denominator = float(current[axis] - previous[axis])
                        if abs(denominator) > 1e-12:
                            fraction = (bound - float(previous[axis])) / denominator
                            output.append(previous + fraction * (current - previous))
                    if current_inside:
                        output.append(current)
                    previous = current
                    previous_inside = current_inside
                clipped = np.asarray(output, dtype=np.float64).reshape(-1, 2)
        return clipped

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        vertices = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(vertices) < 3:
            return 0.0
        return 0.5 * abs(
            float(
                np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
                - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
            )
        )

    def _compute_success_now(self) -> bool:
        if self.task_name == "t26_place_fruit_in_bucket":
            return self._target_on_bucket_bottom()
        if self.task_name == "t25_place_cup_on_plate":
            return self._target_on_tray_center()
        if self.task_name == "t24_place_ball_in_bowl":
            return self._target_inside_bowl_precise()
        if self.task_name == "t19_orient_cup_upright":
            return self._cup_bottom_down()
        if self.task_name in {"t04_cover_pitcher", "t03_cover_box"}:
            return self._lid_fit_success()
        if self.task_name == "b02_nut_bolt_assembly":
            return self._nut_bolt_success()
        if self.task_name == "t09_insert_peg":
            return self._peg_socket_success()
        if self.task_name == "t44_stack_blocks":
            return self._block_stack_stable()
        if self.task_name == "t08_hammer_nail":
            return self._t08_hammer_nail_success()
        if self.task_name == "t36_scoop_marbles":
            return self._spoon_marbles_in_bowl()
        if self.task_name == "t50_water_plant":
            return self._t50_water_plant_success_now()
        if self.task_name == "t14_move_and_click_mouse":
            return self._t14_move_and_click_mouse_success_now()
        if self.task_name == "t23_photograph_logo":
            return self._t23_photograph_logo_success_now()
        if self.task_name == "c07_pitcher_pour_cup":
            return self._c07_pitcher_pour_cup_success_now()
        if self.task_name == "c08_tongs_ice_cubes":
            return self._c08_tongs_ice_cubes_success_now()
        if self.task_name == "c09_staple_paper_sheets":
            return self._c09_staple_paper_sheets_success_now()
        if self.task_name == "t02_connect_charger_and_cable":
            return self._t02_connect_charger_and_cable_success_now()
        if self.task_name == "c11_xylophone_sequence":
            return self._c11_xylophone_sequence_success_now()
        if self.task_name == "t11_line_up_blocks":
            return self._t11_line_up_blocks_success_now()

        return False

    def _target_inside_bowl_precise(self) -> bool:
        local, _target, _bowl = self._target_local_to_receptacle()
        radial = float(np.linalg.norm(local[:2]))
        return bool(radial <= 0.095 and 0.010 <= float(local[2]) <= 0.115)

    def _target_on_tray_center(self) -> bool:
        target = self._pose_for_role("target")
        tray = self._pose_for_role("receptacle")
        target_is_brown_cup = self._target_object_name == "025_mug"
        xy_error = float(np.linalg.norm(target[:2] - tray[:2]))
        z_above = float(target[2] - tray[2])
        bottom_down = self._z_axis_alignment(target) >= float(np.cos(np.deg2rad(25.0)))
        not_dropped = target[2] >= self._current_table_top_z() - 0.02
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= 0.025
            and float(np.linalg.norm(velocity[3:])) <= 0.35
        )
        return bool(
            target_is_brown_cup
            and xy_error <= 0.065
            and 0.015 <= z_above <= 0.10
            and bottom_down
            and not_dropped
            and stable
        )

    def _cup_bottom_down(self) -> bool:
        metrics = self._a06_cup_success_metrics()
        self._last_a06_success_metrics = metrics
        return bool(metrics["a06_success_now"])

    def _a06_hand_points(self, side: str) -> tuple[np.ndarray, ...]:
        points: list[np.ndarray] = []
        robot = self._robots.get(side)
        if robot is not None:
            try:
                body_names = list(robot.body_names)
                body_poses = robot.data.body_pose_w[0]
                for index, name in enumerate(body_names):
                    normalized = str(name).lower()
                    if "palm" not in normalized and "finger" not in normalized:
                        continue
                    point = body_poses[int(index), :3]
                    if hasattr(point, "detach"):
                        point = point.detach().cpu().numpy()
                    point_np = np.asarray(point, dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(point_np)):
                        points.append(point_np)
            except Exception:
                points = []
        if points:
            return tuple(points)

        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        matrix = self._native_end_effector_pose_matrix(side, fallback)
        if matrix is None:
            return ()
        point_np = np.asarray(matrix[:3, 3], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point_np)):
            return ()
        return (point_np,)

    def _init_success_condition_overlay(self) -> None:
        if (
            self._headless
            or not self._launch_isaac
            or self._success_condition_window is not None
            or self._success_condition_frame is not None
        ):
            return
        try:
            import omni.ui as ui

            viewport_window = None
            try:
                from omni.kit.viewport.utility import get_active_viewport_window

                viewport_window = get_active_viewport_window()
            except Exception:
                viewport_window = None

            if viewport_window is not None:
                self._success_condition_frame = viewport_window.get_frame(
                    "handson.success_condition_overlay"
                )
                self._success_condition_frame.visible = True
                self._build_success_condition_overlay(ui, self._success_condition_frame)
            else:
                self._success_condition_window = ui.Window(
                    "HandsOn",
                    width=420,
                    height=320,
                    visible=True,
                )
                for attr, value in (
                    ("position_x", max(20, int(self._render_width) - 440)),
                    ("position_y", 30),
                ):
                    try:
                        setattr(self._success_condition_window, attr, value)
                    except Exception:
                        pass
                self._build_success_condition_overlay(
                    ui,
                    self._success_condition_window.frame,
                )
        except Exception as exc:
            if not self._success_condition_overlay_failed:
                self._success_condition_overlay_failed = True
                self._add_task_state_warning(
                    f"Could not create Isaac success-condition overlay: {exc}"
                )

    @staticmethod
    def _condition_row(key: str, label: str, ok: bool) -> dict[str, Any]:
        return {"key": key, "label": label, "ok": bool(ok)}

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        try:
            if self.task_name == "t26_place_fruit_in_bucket":
                rows = self._a01_success_condition_rows()
            elif self.task_name == "t25_place_cup_on_plate":
                rows = self._a02_success_condition_rows()
            elif self.task_name == "t24_place_ball_in_bowl":
                rows = self._a03_success_condition_rows()
            elif self.task_name == "t19_orient_cup_upright":
                rows = self._a06_success_condition_rows()
            elif self.task_name == "t04_cover_pitcher":
                rows = self._b01_success_condition_rows()
            elif self.task_name == "b02_nut_bolt_assembly":
                rows = self._b02_success_condition_rows()
            elif self.task_name == "t09_insert_peg":
                rows = self._b03_success_condition_rows()
            elif self.task_name == "t44_stack_blocks":
                rows = self._b04_success_condition_rows()
            elif self.task_name == "t03_cover_box":
                rows = self._b05_success_condition_rows()
            elif self.task_name == "t08_hammer_nail":
                rows = self._c01_success_condition_rows()
            elif self.task_name == "t36_scoop_marbles":
                rows = self._c02_success_condition_rows()
            elif self.task_name == "t50_water_plant":
                rows = self._c04_success_condition_rows()
            elif self.task_name == "t14_move_and_click_mouse":
                rows = self._c05_success_condition_rows()
            elif self.task_name == "t23_photograph_logo":
                rows = self._c06_success_condition_rows()
            elif self.task_name == "c07_pitcher_pour_cup":
                rows = self._c07_success_condition_rows()
            elif self.task_name == "c08_tongs_ice_cubes":
                rows = self._c08_success_condition_rows()
            elif self.task_name == "c09_staple_paper_sheets":
                rows = self._c09_success_condition_rows()
            elif self.task_name == "t02_connect_charger_and_cable":
                rows = self._c10_success_condition_rows()
            elif self.task_name == "c11_xylophone_sequence":
                rows = self._c11_success_condition_rows()
            elif self.task_name == "t11_line_up_blocks":
                rows = self._c12_success_condition_rows()
            else:
                rows = [
                    self._condition_row(
                        "event_success_unavailable",
                        "该任务需要按钮/触发器事件，Isaac 当前未接入事件判定",
                        False,
                    )
                ]
        except Exception as exc:
            rows = [
                self._condition_row(
                    "condition_panel_error",
                    f"判定条件读取失败：{exc}",
                    False,
                )
            ]
        return self._append_success_hold_row(rows)

    def _c07_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c07_success_metrics or self._c07_pitcher_pour_metrics()
        return [
            self._condition_row(
                "c07_enough_inside",
                "至少 80% 代理物进入目标杯",
                bool(metrics["c07_enough_inside"]),
            ),
            self._condition_row(
                "c07_spill_limited",
                "桌面洒落数量不超过 1 个",
                bool(metrics["c07_spill_limited"]),
            ),
            self._condition_row(
                "c07_tilt_seen",
                "大把手杯已经发生有效倾倒",
                bool(metrics["c07_tilt_seen"]),
            ),
            self._condition_row(
                "c07_pitcher_centered_on_pad",
                "大把手杯完整放入回收垫中心",
                bool(metrics["c07_pitcher_centered_on_pad"]),
            ),
            self._condition_row(
                "c07_pitcher_on_pad_height",
                "大把手杯底已经落到回收垫上",
                bool(metrics["c07_pitcher_on_pad_height"]),
            ),
            self._condition_row(
                "c07_pitcher_stable",
                "大把手杯已停止移动",
                bool(metrics["c07_pitcher_stable"]),
            ),
            self._condition_row(
                "c07_hand_released",
                "手已完全离开大把手杯",
                bool(metrics["c07_hand_released"]),
            ),
            self._condition_row(
                "c07_pitcher_returned",
                "大把手杯回正并稳定放回",
                bool(metrics["c07_pitcher_returned"]),
            ),
        ]

    def _c08_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c08_success_metrics or self._c08_tongs_ice_metrics()
        return [
            self._condition_row(
                "c08_all_ice_inside",
                "4 个冰块均进入杯内",
                bool(metrics["c08_all_ice_inside"]),
            ),
            self._condition_row(
                "c08_tongs_cycles",
                "食品夹完成至少 4 次闭合",
                bool(metrics["c08_tongs_cycles"]),
            ),
            self._condition_row(
                "c08_cup_upright", "玻璃杯未倾倒", bool(metrics["c08_cup_upright"])
            ),
        ]

    def _c09_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c09_success_metrics or self._c09_staple_metrics()
        return [
            self._condition_row(
                "c09_stapler_opened",
                "订书机上盖已打开",
                bool(metrics["c09_stapler_opened"]),
            ),
            self._condition_row(
                "c09_papers_overlap",
                "两张纸重叠率达到要求",
                bool(metrics["c09_papers_overlap"]),
            ),
            self._condition_row(
                "c09_papers_in_zone",
                "纸张进入有效装订区",
                bool(metrics["c09_papers_in_zone"]),
            ),
            self._condition_row(
                "c09_stapler_pressed",
                "订书机压合越过阈值",
                bool(metrics["c09_stapler_pressed"]),
            ),
        ]

    def _c10_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c10_success_metrics or self._c10_connection_metrics()
        return [
            self._condition_row(
                "c10_first_connected",
                "充电头先插入排插",
                bool(metrics["c10_first_connected"]),
            ),
            self._condition_row(
                "c10_second_connected",
                "充电线接头插入充电头",
                bool(metrics["c10_second_connected"]),
            ),
            self._condition_row(
                "c10_order_valid", "插接顺序正确", bool(metrics["c10_order_valid"])
            ),
        ]

    def _c12_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c12_success_metrics or self._c12_push_line_metrics()
        return [
            self._condition_row(
                "c12_targets_overlap",
                "三个木块与目标轮廓重合率均大于 90%",
                bool(metrics["c12_targets_overlap"]),
            ),
            self._condition_row(
                "c12_blocks_on_line",
                "三个木块中心贴近基准线",
                bool(metrics["c12_blocks_on_line"]),
            ),
            self._condition_row(
                "c12_blocks_aligned",
                "三个木块朝向与基准线对齐",
                bool(metrics["c12_blocks_aligned"]),
            ),
            self._condition_row(
                "c12_spacing_ok",
                "相邻木块间距在设定范围内",
                bool(metrics["c12_spacing_ok"]),
            ),
            self._condition_row(
                "c12_tool_used", "已使用直角尺/刮板推动", bool(metrics["c12_tool_used"])
            ),
            self._condition_row(
                "c12_no_direct_grasp",
                "手未直接触碰木块",
                bool(metrics["c12_no_direct_grasp"]),
            ),
        ]

    def _append_success_hold_row(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        required_steps = self._success_steps_required()
        if required_steps > 1:
            hold_label = f"以上条件连续保持 {required_steps} 帧"
            hold_key = "success_hold"
            if self.task_name == "c07_pitcher_pour_cup":
                hold_label = "放稳并脱手连续保持 0.5 秒"
                hold_key = "c07_release_hold"
            elif self.task_name == "t09_insert_peg":
                hold_label = "插入且插销/插座稳定连续保持 0.5 秒"
                hold_key = "b03_stable_hold"
            rows.append(
                self._condition_row(
                    hold_key,
                    hold_label,
                    self._success_counter >= required_steps,
                )
            )
        return rows

    def _a02_success_condition_rows(self) -> list[dict[str, Any]]:
        target = self._pose_for_role("target")
        tray = self._pose_for_role("receptacle")
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= 0.025
            and float(np.linalg.norm(velocity[3:])) <= 0.35
        )
        return [
            self._condition_row(
                "a02_target_is_mug",
                "目标是棕色杯子",
                self._target_object_name == "025_mug",
            ),
            self._condition_row(
                "a02_centered_on_tray",
                "杯子放在托盘中心附近",
                float(np.linalg.norm(target[:2] - tray[:2])) <= 0.065,
            ),
            self._condition_row(
                "a02_height_on_tray",
                "杯底高度在托盘上方",
                0.015 <= float(target[2] - tray[2]) <= 0.10,
            ),
            self._condition_row(
                "a02_bottom_down",
                "杯子底朝下放正",
                self._z_axis_alignment(target) >= float(np.cos(np.deg2rad(25.0))),
            ),
            self._condition_row(
                "a02_not_dropped",
                "杯子没有掉到桌面下方",
                target[2] >= self._current_table_top_z() - 0.02,
            ),
            self._condition_row("a02_stable", "杯子已经稳定不滑动", stable),
        ]

    def _a03_success_condition_rows(self) -> list[dict[str, Any]]:
        local, _target, _bowl = self._target_local_to_receptacle()
        return [
            self._condition_row(
                "a03_ball_in_bowl_xy",
                "球在碗口范围内",
                float(np.linalg.norm(local[:2])) <= 0.095,
            ),
            self._condition_row(
                "a03_ball_in_bowl_z",
                "球的高度在碗里面",
                0.010 <= float(local[2]) <= 0.115,
            ),
        ]

    def _a06_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_a06_success_metrics or self._a06_cup_success_metrics()
        return [
            self._condition_row(
                "a06_opening_up", "杯口朝上", bool(metrics["a06_opening_up"])
            ),
            self._condition_row(
                "a06_on_table", "杯子放在桌面高度", bool(metrics["a06_on_table"])
            ),
            self._condition_row(
                "a06_hand_clear", "手已经离开杯子", bool(metrics["a06_hand_clear"])
            ),
        ]

    def _b01_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_b01_lid_success_metrics
        if not metrics:
            metrics = self._b01_lid_insert_metrics(
                self._pose_for_role("target"),
                self._pose_for_role("receptacle"),
                update_hand_clear_hold=False,
            )
        return [
            self._condition_row(
                "b01_insert_inside",
                "杯盖下方凸起整体进入杯口",
                bool(metrics["b01_lid_insert_inside"]),
            ),
            self._condition_row(
                "b01_z_close", "杯盖高度接近完全扣合", bool(metrics["b01_lid_z_close"])
            ),
            self._condition_row(
                "b01_top_aligned", "杯盖基本放平", bool(metrics["b01_lid_top_aligned"])
            ),
            self._condition_row(
                "b01_moved", "杯盖已被移动到杯子上", bool(metrics["b01_lid_moved"])
            ),
            self._condition_row(
                "b01_stable", "杯盖已经稳定不滑动", bool(metrics["b01_lid_stable"])
            ),
            self._condition_row(
                "b01_hand_clear", "手已经离开杯盖", bool(metrics["b01_hand_clear"])
            ),
            self._condition_row(
                "b01_hand_clear_held",
                "手离开杯盖保持 0.5 秒",
                bool(metrics["b01_hand_clear_held"]),
            ),
        ]

    def _c04_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c04_success_metrics
        if not metrics:
            aim_metrics = self._c04_spray_aim_metrics()
            metrics = {
                **aim_metrics,
                "spray_grasp_ok": self._c04_hand_gripping_spray(),
                "spray_trigger_contact_ok": False,
                "c04_water_emitting": self._c04_water_emitting,
            }
        return [
            self._condition_row(
                "c04_spray_grasped",
                "手已经抓住喷壶",
                bool(metrics.get("spray_grasp_ok", False)),
            ),
            self._condition_row(
                "c04_trigger_contact",
                "手指确实按到扳机",
                bool(metrics.get("spray_trigger_contact_ok", False)),
            ),
            self._condition_row(
                "c04_water_emitting",
                "扳机保持按下并正在喷水",
                bool(metrics.get("c04_water_emitting", False)),
            ),
            self._condition_row(
                "c04_water_hits_target",
                "水流命中植株或盆土",
                bool(metrics.get("spray_hits_soil_or_plant", False)),
            ),
        ]

    def _c05_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c05_success_metrics or self._c05_mouse_click_metrics(
            update_failure=False
        )
        return [
            self._condition_row(
                "c05_mouse_moved",
                "鼠标已经离开初始位置并移动足够距离",
                bool(metrics["c05_mouse_moved"]),
            ),
            self._condition_row(
                "c05_mouse_on_pad",
                "鼠标已经移动到鼠标垫范围内",
                bool(metrics["c05_mouse_on_pad"]),
            ),
            self._condition_row(
                "c05_no_early_click",
                "移动完成前没有压到左右按键",
                bool(metrics["c05_no_early_click"]),
            ),
            self._condition_row(
                "c05_left_clicked",
                "鼠标到位后按下左键",
                bool(metrics["c05_left_clicked"]),
            ),
        ]

    def _c06_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c06_success_metrics or self._c06_photograph_metrics()
        return [
            self._condition_row(
                "c06_camera_aimed",
                "相机镜头对准标志",
                bool(metrics["c06_camera_aimed"]),
            ),
            self._condition_row(
                "c06_right_hand_contact",
                "右手手指接触快门按钮",
                bool(metrics["c06_right_hand_contact"]),
            ),
            self._condition_row(
                "c06_shutter_pressed",
                "快门按钮已经压下",
                bool(metrics["c06_shutter_pressed"]),
            ),
        ]

    @staticmethod
    def _point_to_segment_distance_xy(
        point: np.ndarray,
        segment_start: np.ndarray,
        segment_end: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        point_np = np.asarray(point, dtype=np.float64).reshape(2)
        start = np.asarray(segment_start, dtype=np.float64).reshape(2)
        end = np.asarray(segment_end, dtype=np.float64).reshape(2)
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-12:
            closest = start
        else:
            t = float(
                np.clip(np.dot(point_np - start, segment) / segment_len_sq, 0.0, 1.0)
            )
            closest = start + t * segment
        return float(np.linalg.norm(point_np - closest)), closest

    def _c01_nail_depth(self) -> float:
        state = self._physics_dof_position("nail_slide_joint")
        if state is None:
            return 0.0
        current, initial = state
        return float(np.clip(initial - current, 0.0, 0.030))

    def _stack_target_height(
        self,
        names: list[str] | tuple[str, ...],
        vertical_half_extents: dict[str, float] | None = None,
    ) -> float:
        extents = vertical_half_extents or {
            name: self._stack_vertical_half_extent(
                name, self._poses.get(name, self._fallback_pose(name))
            )
            for name in names
        }
        return float(sum(2.0 * float(extents[name]) for name in names))

    def _stack_tower_height(
        self,
        poses: dict[str, np.ndarray],
        names: list[str] | tuple[str, ...],
        vertical_half_extents: dict[str, float] | None = None,
    ) -> float:
        if not names:
            return 0.0
        extents = vertical_half_extents or {
            name: self._stack_vertical_half_extent(name, poses[name]) for name in names
        }
        table_z = self._current_table_top_z()
        top_z = max(float(poses[name][2]) + float(extents[name]) for name in names)
        return float(top_z - table_z)

    def _block_stack_max_speeds(
        self,
        names: list[str] | tuple[str, ...],
    ) -> tuple[float, float, bool]:
        max_linear_speed = 0.0
        max_angular_speed = 0.0
        available_count = 0
        for name in names:
            velocity = self._physics_body_velocity(name)
            if velocity is None:
                continue
            available_count += 1
            max_linear_speed = max(
                max_linear_speed,
                float(np.linalg.norm(velocity[:3])),
            )
            max_angular_speed = max(
                max_angular_speed,
                float(np.linalg.norm(velocity[3:])),
            )
        return (
            max_linear_speed,
            max_angular_speed,
            available_count == len(names),
        )

    def _block_stack_hand_clearance(
        self,
        poses: dict[str, np.ndarray],
    ) -> tuple[float, bool]:
        """Compatibility wrapper for callers that only need distance/release."""
        metrics = self._block_stack_hand_metrics(poses)
        return float(metrics["min_distance"]), bool(metrics["released"])

    def render(self):
        frames: dict[str, np.ndarray] = {}
        if self._sim is not None and self._cameras:
            self._update_recording_cameras()
            if self._deferred_arm_render:
                self._sim.render()
                for name, fallback_key in (
                    ("head_camera", "ego"),
                    ("wrist_left_camera", "wrist_left"),
                    ("wrist_right_camera", "wrist_right"),
                ):
                    camera = self._cameras.get(name)
                    if camera is None:
                        continue
                    camera.update(self._control_dt)
                    frames[name] = self._camera_rgb_frame(name, fallback_key)
            else:
                # A stable visibility set is shared by the viewport and all
                # recording cameras. Per-camera visibility switching polluted
                # RTX/DLSS temporal history and required four render passes.
                self._set_wrist_obstructions_hidden(True)
                self._sim.render()
                for name, fallback_key in (
                    ("head_camera", "ego"),
                    ("wrist_left_camera", "wrist_left"),
                    ("wrist_right_camera", "wrist_right"),
                ):
                    camera = self._cameras.get(name)
                    if camera is None:
                        continue
                    camera.update(self._control_dt)
                    frames[name] = self._camera_rgb_frame(name, fallback_key)
        for name, fallback_key in (
            ("head_camera", "ego"),
            ("wrist_left_camera", "wrist_left"),
            ("wrist_right_camera", "wrist_right"),
        ):
            if name not in frames:
                frames[name] = self._camera_rgb_frame(name, fallback_key)
        return [
            frames["head_camera"],
            frames["wrist_left_camera"],
            frames["wrist_right_camera"],
        ]

    @staticmethod
    def _resize_rgb_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        y_idx = np.clip(
            (np.arange(height) * src_h / height).astype(np.int64), 0, src_h - 1
        )
        x_idx = np.clip(
            (np.arange(width) * src_w / width).astype(np.int64), 0, src_w - 1
        )
        return frame[y_idx][:, x_idx, :3].astype(np.uint8, copy=True)

    def _fallback_camera_frame(
        self,
        key: str,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        frame = np.zeros(
            (
                self._render_height if height is None else int(height),
                self._render_width if width is None else int(width),
                3,
            ),
            dtype=np.uint8,
        )
        colors = {
            "ego": (58, 66, 70),
            "wrist_left": (64, 74, 92),
            "wrist_right": (92, 72, 62),
        }
        frame[:, :] = colors.get(key, (48, 48, 48))
        h, w, _ = frame.shape
        frame[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, :] = (154, 154, 145)
        return frame

    def get_end_effector_pose_matrix(self):
        right = self._native_end_effector_pose_matrix("right", self._right_tcp_pose)
        if not self._spec.bimanual:
            return right
        left = self._native_end_effector_pose_matrix("left", self._left_tcp_pose)
        return right, left
