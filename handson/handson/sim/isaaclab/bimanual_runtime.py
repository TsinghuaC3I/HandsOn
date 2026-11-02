"""Shared runtime behavior across the archived evaluation environments.

Only methods with identical syntax and matching library bindings are shared.
Task-specific constants, reset distributions, control variants and native
success conditions remain on their environment classes when they differ.
The extraction fixture preserves the pre-refactor method fingerprints.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import time


class BimanualRuntimeSupport:
    """Common methods; environment subclasses own all state and configuration."""

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        """Return the constructor recipe for a fresh scene in the current app."""

        return {
            "seed": int(seed),
            "control_dt": float(self._control_dt),
            "physics_dt": float(self._physics_dt),
            "time_limit": float(self._time_limit),
            "render_mode": self.render_mode,
            "image_obs": bool(self.image_obs),
            "randomize": bool(self.randomize),
            "hand_type": self.hand_type,
            "arm_type": self.arm_type,
            "launch_isaac": True,
            "headless": bool(self._headless),
            "render_resolution": (
                int(self._render_height),
                int(self._render_width),
            ),
            "hz": int(self.hz),
            "isaac_device": str(self._isaac_device),
        }

    def _unsubscribe_scene_input_callbacks(self) -> None:
        if (
            self._keyboard_subscription is not None
            and self._keyboard_input_interface is not None
        ):
            try:
                self._keyboard_input_interface.unsubscribe_to_keyboard_events(
                    self._keyboard,
                    self._keyboard_subscription,
                )
            except Exception:
                pass
        self._keyboard_subscription = None
        self._keyboard_input_interface = None
        self._keyboard = None
        self._timeline_pause_subscription = None
        self._timeline = None

    @staticmethod
    def _normalize_arm_ik_backend(value: str) -> str:
        backend = str(value or "").strip().lower()
        if backend in {"haorui", "haorui_new"}:
            return "haorui"
        if backend in {
            "",
            "dls",
            "sim",
            "isaac",
            "isaaclab",
            "isaaclab_dls",
            "differential_ik",
        }:
            return "dls"
        print(
            f"[HandsOn/IsaacLab] Unknown arm IK backend {backend!r}; using isaaclab_dls",
            flush=True,
        )
        return "dls"

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random

    def get_recording_schema(self) -> dict[str, Any] | None:
        """Return the optional action-aligned PhysX ground-truth schema."""

        if not self._physx_ground_truth_enabled:
            return None
        if self._physx_ground_truth_trace is None:
            self._initialize_physx_ground_truth_recording()
        if self._physx_ground_truth_trace is None:
            return None
        schema = dict(self._physx_ground_truth_trace.schema())
        schema["reset_replay_contract"] = self.get_physical_replay_reset_contract()
        return schema

    def _attach_physx_ground_truth(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._physx_ground_truth_enabled:
            return observation
        if self._physx_ground_truth_trace is None:
            self._initialize_physx_ground_truth_recording()
        if self._physx_ground_truth_trace is None:
            raise RuntimeError("PhysX ground-truth trace failed to initialize.")
        result = dict(observation)
        result.update(self._physx_ground_truth_trace.snapshot())
        result.update(self._physx_ground_truth_trace.substep_snapshot(consume=True))
        return result

    def _setup_timeline_pause_guard(self) -> None:
        """Keep interactive collection running when Kit receives a pause command."""
        if self._headless:
            return
        try:
            import omni.timeline

            self._timeline = omni.timeline.get_timeline_interface()
            event_stream = self._timeline.get_timeline_event_stream()
            self._timeline_pause_subscription = (
                event_stream.create_subscription_to_pop_by_type(
                    int(omni.timeline.TimelineEventType.PAUSE),
                    self._on_timeline_pause,
                    order=-100,
                    name="HandsOn IsaacLab collection pause guard",
                )
            )
            print(
                "[HandsOn/IsaacLab] Timeline pause guard enabled: Space will not pause collection",
                flush=True,
            )
        except Exception as exc:
            message = f"Timeline pause guard unavailable: {exc}"
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _on_timeline_pause(self, _event=None) -> None:
        self._resume_timeline_if_paused(source="Timeline pause")

    def _resume_timeline_if_paused(self, *, source: str = "") -> bool:
        timeline = self._timeline
        if timeline is None:
            return False
        try:
            is_stopped = getattr(timeline, "is_stopped", None)
            if callable(is_stopped) and is_stopped():
                return False
            if timeline.is_playing():
                return False
            timeline.play()
            if source:
                print(
                    f"[HandsOn/IsaacLab] {source} ignored; collection resumed immediately",
                    flush=True,
                )
            return True
        except Exception as exc:
            message = f"Could not resume paused Timeline: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return False

    def _setup_keyboard_reset(self) -> None:
        if self._headless:
            return
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("default app window is unavailable")
            keyboard = app_window.get_keyboard()
            input_interface = carb.input.acquire_input_interface()
            self._keyboard = keyboard
            self._keyboard_input_interface = input_interface
            self._keyboard_subscription = input_interface.subscribe_to_keyboard_events(
                keyboard,
                lambda event, *args, **kwargs: self._on_keyboard_event(
                    carb.input, event
                ),
            )
            print(
                "[HandsOn/IsaacLab] Keyboard controls enabled: R resets; "
                "V restores the task view; Space is ignored",
                flush=True,
            )
        except Exception as exc:
            message = f"Keyboard reset unavailable: {exc}"
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _on_keyboard_event(self, carb_input, event) -> bool:
        event_type = getattr(event, "type", None)
        key_input = getattr(event, "input", None)
        press_type = getattr(carb_input.KeyboardEventType, "KEY_PRESS", None)
        event_type_name = str(event_type).lower()
        is_press = event_type == press_type or (
            "press" in event_type_name and "release" not in event_type_name
        )
        key_name = str(key_input).lower()
        is_r_key = key_input == getattr(
            carb_input.KeyboardInput, "R", None
        ) or key_name in {"r", "keyboardinput.r"}
        is_v_key = key_input == getattr(
            carb_input.KeyboardInput, "V", None
        ) or key_name in {
            "v",
            "keyboardinput.v",
        }
        is_space_key = key_input == getattr(
            carb_input.KeyboardInput, "SPACE", None
        ) or key_name in {
            "space",
            "keyboardinput.space",
        }
        if is_press and is_space_key:
            # Returning False consumes the event before Kit's default Space
            # shortcut can pause the Timeline.  If that shortcut ran first,
            # resume here as well so SimulationContext.step() cannot block.
            self._resume_timeline_if_paused(source="Space pause")
            return False
        if is_press and is_r_key:
            # Freeze the camera before Kit or the recorder gets another event
            # cycle.  Reading it only when the USD Stage is later destroyed is
            # too late: Kit's own R handling may already have reframed it.
            self._manual_reset_viewport_camera_state = (
                self._capture_viewport_camera_state()
            )
            self._manual_reset_requested = True
            # The recorder owns the reset transaction: it discards the current
            # trajectory, recreates the spool, re-randomizes the scene, waits
            # for the stable recording origin, then rebases teleoperation.
            # Resetting here as well used to create a second, concurrent scene
            # reset and let Kit's default R shortcut see the same key event.
            print(
                "[HandsOn/IsaacLab] Keyboard reset requested; queued for recorder "
                f"(camera_frozen={self._manual_reset_viewport_camera_state is not None})",
                flush=True,
            )
            return False
        if is_press and is_v_key:
            self._reset_viewport_camera_to_default()
            return False
        return True

    def _consume_manual_reset_request(self) -> bool:
        requested = bool(self._manual_reset_requested)
        self._manual_reset_requested = False
        return requested

    def _apply_keyboard_reset_immediately(self) -> None:
        if self._keyboard_reset_in_progress:
            return
        if self._sim is None:
            return
        self._keyboard_reset_in_progress = True
        try:
            self._begin_recording_origin_reset_write_capture()
            self._sample_reset_state(clear_manual_request=False)
            self._reset_scene_state()
            self._apply_reset_visual_randomization()
            self._capture_recording_origin_authored_reset_snapshot()
            self._step_isaaclab()
            self._refresh_object_poses()
            self._manual_reset_already_applied = True
        except Exception as exc:
            message = (
                f"Immediate keyboard reset failed; recorder fallback will reset: {exc}"
            )
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
        finally:
            self._keyboard_reset_in_progress = False

    def _init_haorui_rokae_ik(self) -> None:
        if self._arm_ik_backend != "haorui":
            self._haorui_ik_enabled = False
            print("[HandsOn/IsaacLab] ROKAE arm IK backend=isaaclab_dls", flush=True)
            return
        try:
            from ..envs.rokae_wuji_ik_replay import rokae as rokae08

            self._haorui_rokae = rokae08
            self._haorui_ik_models = {
                "right": rokae08.load_model("right"),
                "left": rokae08.load_model("left"),
            }
            self._haorui_ik_solvers = {
                side: rokae08.make_solver(side) for side in self._haorui_ik_models
            }
            for solver in self._haorui_ik_solvers.values():
                solver.warmup()
            self._haorui_ik_enabled = True
            print("[HandsOn/IsaacLab] ROKAE arm IK backend=haorui_new", flush=True)
        except Exception as exc:
            self._haorui_ik_enabled = False
            message = f"Haorui ROKAE IK unavailable; fallback=isaaclab_dls reason={exc}"
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _control_robot(self, side: str):
        if self._scene is None:
            raise RuntimeError("IsaacLab control scene is not initialized")
        return self._scene[f"{side}_robot"]

    def _reset_robot_control_state(self) -> None:
        for side in ("right", "left"):
            robot = self._control_robot(side)
            root_state = robot.data.default_root_state.clone()
            if self._scene is not None and hasattr(self._scene, "env_origins"):
                root_state[:, :3] += self._scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

        self._restore_default_robot_joint_state(
            reset_controllers=True,
            sync=False,
        )

    def _restore_default_robot_joint_state(
        self,
        *,
        reset_controllers: bool = True,
        sync: bool = False,
    ) -> dict[str, Any]:
        """Atomically restore robot joints and every matching control target.

        A joint-state teleport alone is not a complete reset for an implicit
        actuator.  Its previous position target remains live and accelerates
        the joint again on the next PhysX frame.  Keep the simulator state,
        actuator buffers, and Python-side command state on the same default
        pose so the pre-action recording boundary cannot inherit an old
        teleoperation target.
        """

        targets: dict[str, Any] = {}
        for side in ("right", "left"):
            try:
                robot = self._control_robot(side)
            except RuntimeError:
                continue

            joint_pos = robot.data.default_joint_pos.clone()
            if self._torch is not None:
                joint_vel = self._torch.zeros_like(joint_pos)
            else:
                joint_vel = robot.data.default_joint_vel.clone()
                zero_ = getattr(joint_vel, "zero_", None)
                if callable(zero_):
                    zero_()
                else:
                    joint_vel[...] = 0.0

            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            set_velocity_target = getattr(
                robot,
                "set_joint_velocity_target",
                None,
            )
            if callable(set_velocity_target):
                set_velocity_target(joint_vel)
            write_data = getattr(robot, "write_data_to_sim", None)
            if callable(write_data):
                write_data()

            target = joint_pos.clone()
            self._joint_targets[side] = target
            targets[side] = target.clone()

            joint_ids = list(self._hand_joint_ids.get(side, ()))
            hand_command = getattr(self, f"_{side}_hand", None)
            if joint_ids and isinstance(hand_command, np.ndarray):
                values = target[:, joint_ids].detach().cpu().numpy().reshape(-1)
                count = min(hand_command.size, values.size)
                hand_command[:count] = values[:count]

            setattr(self, f"_{side}_pose_commanded", False)

        self._arm_pose_filters = {"right": None, "left": None}
        if reset_controllers:
            for controller in self._ik_controllers.values():
                reset = getattr(controller, "reset", None)
                if callable(reset):
                    reset()

        if sync:
            if self._sim is not None:
                self._sim.forward()
            if self._scene is not None:
                try:
                    self._scene.update(0.0)
                except Exception:
                    pass
            for side in targets:
                try:
                    self._control_robot(side).update(0.0)
                except Exception:
                    pass
        return targets

    def _reset_scene_state(self) -> None:
        if self._scene is None:
            return
        scene = self._scene
        self._reset_robot_control_state()

        table = scene["table"]
        table_state = table.data.default_root_state.clone()
        table_state[:, :3] += scene.env_origins
        table_state[:, 2] += float(self.delta_h[0])
        table.write_root_pose_to_sim(table_state[:, :7])

        for name in ("peg", "socket"):
            rigid_object = scene[name]
            root_state = rigid_object.data.default_root_state.clone()
            pose = self._peg_ori_pose if name == "peg" else self._socket_ori_pose
            pose_tensor = self._torch.tensor(
                pose.reshape(1, 7),
                dtype=root_state.dtype,
                device=root_state.device,
            ).repeat(root_state.shape[0], 1)
            pose_tensor[:, :3] += scene.env_origins
            root_state[:, :7] = pose_tensor
            rigid_object.write_root_pose_to_sim(root_state[:, :7])
            rigid_object.write_root_velocity_to_sim(root_state[:, 7:])

        scene.reset()
        scene.write_data_to_sim()

    def reset(self, seed: int | None = None, **_: Any):
        if seed is not None:
            self._seed = int(seed)
            self._random.seed(self._seed)

        if self._manual_reset_already_applied:
            self._manual_reset_already_applied = False
            self._manual_reset_requested = False
            self._manual_reset_viewport_camera_state = None
            self._refresh_object_poses()
            info = {
                "succeed": False,
                "manual_reset_applied": True,
                "isaac_warnings": tuple(self._isaac_warnings),
            }
            info.update(self._task_info())
            return self._attach_physx_ground_truth(self._compute_observation()), info

        self._begin_recording_origin_reset_write_capture()
        self._sample_reset_state(clear_manual_request=True)
        self._reset_scene_state()
        self._apply_reset_visual_randomization()
        self._reset_physics_steps = 0
        self._capture_recording_origin_authored_reset_snapshot()
        self._step_isaaclab()
        self._reset_physics_steps += 1
        self._refresh_object_poses()
        info = {
            "succeed": False,
            "isaac_warnings": tuple(self._isaac_warnings),
        }
        info.update(self._task_info())
        return self._attach_physx_ground_truth(self._compute_observation()), info

    def _apply_reset_visual_randomization(self) -> None:
        if self.randomize and self._launch_isaac:
            try:
                self._randomize_scene_visuals(1, self._random)
            except Exception as exc:
                message = f"Visual randomization unavailable: {exc}"
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def step(self, action):
        start_time = time.monotonic()

        # Keyboard events are delivered while Kit is pumping a simulation
        # frame.  If R was handled between recorder iterations, return the
        # reset transition before an action from the old episode can be
        # applied.
        if self._manual_reset_requested:
            return self._manual_reset_step_result()

        self._apply_action(action)
        self._apply_joint_targets()
        return self.step_with_current_joint_targets(start_time=start_time)

    def step_with_current_joint_targets(self, *, start_time: float | None = None):
        """Advance one control step using targets already written to the drives."""

        previous_action_phase = self._trajectory_action_phase_active
        previous_substep = self._trajectory_action_phase_substep_index
        self._trajectory_action_phase_active = True
        self._trajectory_action_phase_substep_index = -1
        try:
            return self._step_with_current_joint_targets_impl(start_time=start_time)
        finally:
            self._trajectory_action_phase_active = previous_action_phase
            self._trajectory_action_phase_substep_index = previous_substep

    def _log_control_performance(self, timing: dict[str, float | int]) -> None:
        self._control_perf_steps += 1
        now = time.monotonic()
        elapsed = now - self._control_perf_window_start
        if elapsed < 2.0:
            return
        actual_hz = self._control_perf_steps / max(elapsed, 1e-9)
        realtime_factor = (
            self._control_perf_steps * self._control_dt / max(elapsed, 1e-9)
        )
        print(
            "[HandsOn/IsaacLab] control timing: "
            f"target={self.hz}Hz actual={actual_hz:.1f}Hz "
            f"rtf={realtime_factor:.2f} "
            f"work={float(timing['work_ms']):.1f}ms "
            f"render={float(timing['render_ms']):.1f}ms",
            flush=True,
        )
        self._control_perf_window_start = now
        self._control_perf_steps = 0

    def _manual_reset_step_result(self):
        self._refresh_object_poses()
        manual_reset = self._consume_manual_reset_request()
        info = {
            "succeed": False,
            "manual_reset": manual_reset,
            "manual_reset_applied": bool(self._manual_reset_already_applied),
            "isaac_warnings": tuple(self._isaac_warnings),
        }
        info.update(self._task_info())
        return (
            self._attach_physx_ground_truth(self._compute_observation()),
            0.0,
            False,
            False,
            info,
        )

    def _task_info(self) -> dict[str, Any]:
        return {}

    def _compute_failure(self) -> bool:
        return False

    def _step_isaaclab(self) -> None:
        if self._scene is None or self._sim is None:
            return
        # IsaacLab's SimulationContext.step() blocks while the Timeline is
        # paused.  Recover before entering it in case a pause event landed
        # between two control steps.
        self._resume_timeline_if_paused(source="Timeline pause")
        self._scene.write_data_to_sim()
        # Camera rendering is performed exactly once by render(), after all
        # PhysX substeps have completed and wrist poses are current.
        self._sim.step(render=False)
        self._scene.update(self._sim_dt)

    def _apply_action(self, action) -> None:
        if action is None:
            return
        if isinstance(action, dict):
            right = np.asarray(action.get("right", []), dtype=np.float64)
            left = np.asarray(action.get("left", []), dtype=np.float64)
        else:
            flat = np.asarray(action, dtype=np.float64).reshape(-1)
            if flat.size >= 14 + 2 * self._hand_action_dim:
                right = np.concatenate(
                    [flat[:7], flat[14 : 14 + self._hand_action_dim]]
                )
                left = np.concatenate(
                    [
                        flat[7:14],
                        flat[
                            14 + self._hand_action_dim : 14 + 2 * self._hand_action_dim
                        ],
                    ]
                )
            elif flat.size >= 7 + self._hand_action_dim:
                right = flat
                left = np.asarray([], dtype=np.float64)
            else:
                return

        if right.size >= 7 and not np.allclose(right[:7], 0.0):
            self._right_tcp_pose = right[:7].astype(np.float64, copy=True)
            self._right_pose_commanded = True
        if left.size >= 7 and not np.allclose(left[:7], 0.0):
            self._left_tcp_pose = left[:7].astype(np.float64, copy=True)
            self._left_pose_commanded = True
        if right.size >= 7 + self._hand_action_dim:
            self._right_hand = right[7 : 7 + self._hand_action_dim].astype(
                np.float64, copy=True
            )
        if left.size >= 7 + self._hand_action_dim:
            self._left_hand = left[7 : 7 + self._hand_action_dim].astype(
                np.float64, copy=True
            )
        postprocess = getattr(self, "_postprocess_hand_action", None)
        if callable(postprocess):
            postprocess()

    def _arm_reference_q(self, side: str, robot) -> np.ndarray | None:
        arm_joint_ids = self._arm_joint_ids.get(side)
        if arm_joint_ids is None:
            return None
        target = self._joint_targets.get(side)
        source = target if target is not None else robot.data.joint_pos
        return (
            source[:, arm_joint_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
            .reshape(-1)
        )

    def _refresh_object_poses(self) -> None:
        if self._scene is None:
            return
        self._peg_ori_pose = self._rigid_object_pose("peg", self._peg_ori_pose)
        self._socket_ori_pose = self._rigid_object_pose("socket", self._socket_ori_pose)

    def _rigid_object_pose(self, name: str, fallback: np.ndarray) -> np.ndarray:
        try:
            root_state = self._scene[name].data.root_state_w[0, :7]
            return root_state.detach().cpu().numpy().astype(np.float64, copy=True)
        except Exception:
            return fallback.astype(np.float64, copy=True)

    def _compute_success(self) -> bool:
        distance = float(
            np.linalg.norm(self._peg_ori_pose[:3] - self._socket_ori_pose[:3])
        )
        if distance < 0.025:
            self._success_counter += 1
        else:
            self._success_counter = 0
        return self._success_counter >= 30

    def render(self):
        frames: dict[str, np.ndarray] = {}
        if self._sim is not None and self._scene is not None:
            self._update_recording_cameras()
            # Keep the six near-field housing/adapter visuals hidden for every
            # camera. Switching their visibility between head and wrist passes
            # invalidates RTX temporal history, flashes the viewport, and
            # requires four renders. One stable scene gives every camera a
            # coherent frame from a single render.
            self._set_wrist_obstructions_hidden(True)
            self._sim.render()
            for name, fallback_key in (
                ("head_camera", "ego"),
                ("wrist_left_camera", "wrist_left"),
                ("wrist_right_camera", "wrist_right"),
            ):
                self._scene[name].update(self._control_dt)
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

    def _compute_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        if self.image_obs:
            ego, wrist_left, wrist_right = self.render()
            obs["images"] = {
                "random_camera" if self.randomize else "ego": ego,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }

        gripper_pose = np.zeros(32, dtype=np.float64)
        right_dim = min(16, self._right_hand.size)
        left_dim = min(16, self._left_hand.size)
        gripper_pose[:right_dim] = self._right_hand[:right_dim]
        gripper_pose[16 : 16 + left_dim] = self._left_hand[:left_dim]

        obs["state"] = {
            "tcp_pose": np.concatenate(
                [self._right_tcp_pose, self._left_tcp_pose]
            ).astype(np.float64),
            "gripper_pose": gripper_pose,
            "socket_ori_pose": self._socket_ori_pose.astype(np.float64, copy=True),
            "peg_ori_pose": self._peg_ori_pose.astype(np.float64, copy=True),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        return obs

    def get_end_effector_pose_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._body_pose_matrix(
                "right_robot", "right_palm_link", self._right_tcp_pose
            ),
            self._body_pose_matrix("left_robot", "left_palm_link", self._left_tcp_pose),
        )

    def close(self) -> None:
        self._unsubscribe_scene_input_callbacks()
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:
                pass
            self._sim = None
        self._scene = None
        simulation_app = self._simulation_app
        self._simulation_app = None
        self._app_launcher = None
        if simulation_app is not None:
            try:
                simulation_app.close(
                    wait_for_replicator=False,
                    skip_cleanup=bool(getattr(self, "_fast_shutdown_on_close", False)),
                )
            except TypeError:
                try:
                    simulation_app.close(wait_for_replicator=False)
                except TypeError:
                    try:
                        simulation_app.close()
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
