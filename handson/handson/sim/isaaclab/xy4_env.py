"""Continuous HS005 + KA021, accepting either ordered scrub direction."""

from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _HS005_BOTTLE_RESET_XY, _HS005_BRUSH_STABLE_RESET_POSE_XYZW,
    _HS005_MAX_BOTTLE_FINAL_DISPLACEMENT, _HS005_MIN_BOTTLE_UP_ALIGNMENT,
    _HS005_MAX_FINAL_LINEAR_SPEED, _HS005_MAX_FINAL_ANGULAR_SPEED,
    _HS005_CONTACT_FORCE_EPSILON, _HS005_RELEASE_CLEARANCE, _sample_hs005_reset_offsets,
)
from .b320_bimanual_env import (
    HandsOnB320BimanualIsaacLabEnv, _yaw_quaternion,
    _KA021_CUP_PARK_CENTER_BOUNDS_XY,
)
from .b320_bimanual_scene import KA021_SHARED_SCENE_ROOT_PATH
from .xy4_contract import TASK_NAME, CONTROL_HZ, PHYSICS_HZ, XY4Sequence
from .xy4_scene import layout_clearances, PAIR_CLEARANCE_M, RESET_MAX_ATTEMPTS, PARK_CENTER_BOUNDS_XY

from .xy4_contract import HIDDEN_SCALARS, HIDDEN_FIELDS


class HandsOnXY4IsaacLabEnv(HandsOnB320BimanualIsaacLabEnv):
    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"XY4 runtime does not implement {task_name}")
        for key, expected in (("physics_dt", 1 / PHYSICS_HZ), ("control_dt", 1 / CONTROL_HZ)):
            if key in kwargs and not np.isclose(kwargs[key], expected, rtol=0, atol=1e-12):
                raise ValueError(f"XY4 requires {key}={expected}")
            kwargs[key] = expected
        self._xy4_sequence = XY4Sequence()
        self._xy4_scrub_direction = 0
        self._xy4_layout: dict[str, np.ndarray] = {}
        self._xy4_layout_attempts = 0
        self._xy4_fill_update_step = -1
        self._xy4_success_latched = False
        self._xy4_right_bottle_contact_view = None
        self._hs005_material_path = f"{KA021_SHARED_SCENE_ROOT_PATH}/Looks/HS005PhysicsMaterial"
        super().__init__(task_name, **kwargs)

    def _is_b320_collection_task(self) -> bool:
        # XY4 composes the B320 runtime under a new naming family. Preserve its
        # physical TCP reset zero, control pacing and servo diagnostics.
        return True

    def _apply_raised_hand_home(self, robot_cfg, side):
        applied = super()._apply_raised_hand_home(robot_cfg, side)
        if applied:
            # Native PhysX shows the original elbow intersecting the kitchen's
            # front counter. Raise it within the unchanged joint limits and
            # compensate at the wrist, preserving the approach orientation.
            robot_cfg.init_state.joint_pos[f"joint1_{side}(_1)?"] = (
                -.9 if side == "right" else .9)
            robot_cfg.init_state.joint_pos[f"joint4_{side}(_1)?"] = 1.5
            robot_cfg.init_state.joint_pos[f"joint7_{side}(_1)?"] = (
                .6 if side == "right" else -.6)
        return applied

    def accept_stable_recording_origin(self, report):
        # The shared origin gate can settle the arms after reset's first TCP
        # sync. Start the first operator command at the accepted physical palms.
        self._sync_b320_tcp_command_poses_to_physical_state()
        acceptance = HandsOnABCIsaacLabEnv.accept_stable_recording_origin(self, report)
        return {**(acceptance or {}), "accepted": True,
                "tcp_command_origin": "accepted_physx_palms"}

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        self._xy4_sequence.reset()
        self._xy4_scrub_direction = 0
        self._xy4_fill_update_step = -1
        self._xy4_success_latched = False
        self._xy4_layout = {}
        super()._sample_reset_state(clear_manual_request=clear_manual_request)

    def _compute_success(self) -> bool:
        success = bool(super()._compute_success())
        if success:
            # The episode is terminal at this point.  Keep the terminal
            # condition panel green even if a later UI refresh samples a
            # small post-success bottle drift.
            self._xy4_success_latched = True
        return success

    def _sample_ka021_reset_pose(self) -> np.ndarray:
        # Both source samplers use the same environment RNG. The new condition
        # only rejects collisions introduced by composing the two layouts.
        for attempt in range(1, RESET_MAX_ATTEMPTS + 1):
            offset = _sample_hs005_reset_offsets(self._random, randomize=self.randomize)
            cup = super()._sample_ka021_reset_pose()
            if self.randomize:
                # Reuse the source draw without changing the three-object RNG
                # sequence. Its distant pad range leaves XY4's main view.
                old_low, old_high = np.asarray(_KA021_CUP_PARK_CENTER_BOUNDS_XY)
                new_low, new_high = np.asarray(PARK_CENTER_BOUNDS_XY)
                fraction = (self._ka021_cup_park_position_w[:2] - old_low) / (old_high - old_low)
                self._ka021_cup_park_position_w[:2] = new_low + fraction * (new_high - new_low)
            bottle = np.asarray([
                *(_HS005_BOTTLE_RESET_XY + offset[:2]), 0.895,
                *_yaw_quaternion(float(offset[2])),
            ])
            brush = np.asarray([
                *(_HS005_BRUSH_STABLE_RESET_POSE_XYZW[:2] + offset[3:5]),
                _HS005_BRUSH_STABLE_RESET_POSE_XYZW[2],
                *_yaw_quaternion(float(offset[5])),
            ])
            poses = {"hs005_bottle": bottle, "hs005_brush": brush, "ceramic_mug": cup}
            if min(layout_clearances(poses).values()) < PAIR_CLEARANCE_M:
                continue
            self._xy4_layout = {name: pose.copy() for name, pose in poses.items()}
            self._xy4_layout_attempts = attempt
            self._hs005_bottle_reset_pose = bottle.copy()
            self._hs005_brush_reset_pose = brush.copy()
            self._hs005_reset_parameters[:] = [
                *offset, self._hs005_friction,
                self._hs005_bottle_mass_scale, self._hs005_brush_mass_scale,
            ]
            return cup
        raise RuntimeError("XY4 cannot sample a supported separated three-object layout")

    def _fallback_b320_pose(self, name: str) -> np.ndarray:
        if name in self._xy4_layout:
            return self._xy4_layout[name].copy()
        # Before the first seeded reset, describe the unchanged authored props.
        if name == "hs005_bottle":
            return np.asarray([*_HS005_BOTTLE_RESET_XY, 0.895, 1., 0., 0., 0.])
        if name == "hs005_brush":
            return np.asarray([*_HS005_BRUSH_STABLE_RESET_POSE_XYZW[:3], 1., 0., 0., 0.])
        return super()._fallback_b320_pose(name)

    def _randomized_pose(self, pose, *, fixture, object_name=None):
        if object_name in self._xy4_layout:
            return self._xy4_layout[object_name].copy()
        return super()._randomized_pose(pose, fixture=fixture, object_name=object_name)

    def _hs005_scrub_band_order(self, band: int) -> tuple[int, ...]:
        # Only a valid endpoint contact selects this episode's direction.
        if self._xy4_scrub_direction == 0 and band in (0, 2):
            self._xy4_scrub_direction = 1 if band == 0 else -1
        return (2, 1, 0) if self._xy4_scrub_direction == -1 else (0, 1, 2)

    def _after_task_physics_substep(self) -> None:
        # The shared post-substep observer below runs in collection and replay.
        return

    def _init_hs005_contact_views(self) -> None:
        super()._init_hs005_contact_views()
        self._xy4_right_bottle_contact_view = None
        if not self._robots:
            return
        physics = self._task_tensor_sim_view()
        bottle = self._find_rigid_body_prim("hs005_bottle")
        right_paths = self._hs005_hand_body_paths("right")
        try:
            if physics is None or bottle is None or not right_paths:
                raise RuntimeError("bottle or right-hand bodies unavailable")
            view = physics.create_rigid_contact_view(
                str(bottle.GetPath()), filter_patterns=right_paths,
                max_contact_data_count=max(64, len(right_paths) * 8),
            )
            if view is None or int(view.sensor_count) != 1 or int(view.filter_count) != len(right_paths):
                raise RuntimeError("incomplete right-hand/bottle contact view")
            # Keep HS005's three source-role channels unchanged. This observer
            # belongs only to XY4's final, two-hand bottle-release condition.
            self._xy4_right_bottle_contact_view = view
        except Exception as exc:
            self._add_task_state_warning(f"XY4 bottle release reporting unavailable: {exc}")

    def _release_task_physics_resources_for_scene_rebuild(self):
        self._xy4_right_bottle_contact_view = None
        return super()._release_task_physics_resources_for_scene_rebuild()

    def _handson_post_physics_substep_callback(self, substep_index, substep_count):
        super()._handson_post_physics_substep_callback(substep_index, substep_count)
        if not self._xy4_sequence.x_complete:
            self._update_hs005_physical_events()

    def physical_replay_post_substep_observer_is_safe(self, callback):
        return (getattr(callback, "__self__", None) is self and
                getattr(callback, "__func__", None) is
                HandsOnXY4IsaacLabEnv._handson_post_physics_substep_callback)

    def _bottle_release_metrics(self) -> dict[str, Any]:
        bottle = self._current_body_pose("hs005_bottle")
        values: dict[str, Any] = {}
        for side, view in (
            ("left", self._hs005_contact_views.get("left_bottle")),
            ("right", self._xy4_right_bottle_contact_view),
        ):
            force, readable = self._read_b320_contact_force_sample(view, self._physics_dt)
            points = np.asarray(self._a06_hand_points(side), dtype=np.float64).reshape(-1, 3)
            clear = bool(
                bottle is not None and len(points) > 0
                and np.all(np.isfinite(points))
                and np.all(np.isfinite(bottle[:3]))
                and np.min(np.linalg.norm(points - bottle[:3], axis=1))
                >= _HS005_RELEASE_CLEARANCE
            )
            values[f"xy4_{side}_bottle_contact_force_n"] = force
            values[f"xy4_{side}_bottle_contact_readable"] = readable
            values[f"xy4_{side}_bottle_clear"] = clear
        values["xy4_bottle_released"] = all(
            values[f"xy4_{side}_bottle_contact_readable"]
            and values[f"xy4_{side}_bottle_contact_force_n"] <= _HS005_CONTACT_FORCE_EPSILON
            and values[f"xy4_{side}_bottle_clear"]
            for side in ("left", "right")
        )
        return values

    def _bottle_retained(self, metrics: dict[str, Any], release: dict[str, Any]) -> bool:
        velocity = self._physics_body_velocity("hs005_bottle")
        return bool(
            metrics["hs005_runtime_valid"] and velocity is not None
            and metrics["hs005_bottle_displacement"] <= _HS005_MAX_BOTTLE_FINAL_DISPLACEMENT
            and metrics["hs005_bottle_up_alignment"] >= _HS005_MIN_BOTTLE_UP_ALIGNMENT
            and release["xy4_bottle_released"]
            and np.linalg.norm(velocity[:3]) <= _HS005_MAX_FINAL_LINEAR_SPEED
            and np.linalg.norm(velocity[3:]) <= _HS005_MAX_FINAL_ANGULAR_SPEED
        )

    def _metrics(self, *, update: bool) -> dict[str, Any]:
        step = int(self.env_step)
        if update:
            self._read_contacts(update_latches=True)
        scrub = self._hs005_metrics()
        self._last_hs005_success_metrics = scrub
        if update and self._xy4_sequence.advance_x(step=step, success=scrub["hs005_success_now"]):
            self._ka021_fill_hold_frames = 0
            self._ka021_water_filled = False
            self._ka021_fill_last_update_step = None
        enabled = self._xy4_sequence.y_enabled(step)
        count_fill = update and enabled and self._xy4_fill_update_step != step
        faucet, faucet_flags = self._faucet_metrics(update=count_fill)
        if count_fill:
            self._xy4_fill_update_step = step
        release = self._bottle_release_metrics()
        retained = self._bottle_retained(scrub, release)
        # Preserve all eight original Y predicates and its 15-frame terminal
        # stability. X uses its original physical 0.5 s release/stability gate.
        flags = (
            self._xy4_sequence.x_complete,
            *(bool(value) and enabled for value in faucet_flags),
            retained,
        )
        if self._xy4_success_latched:
            flags = (True,) * len(self._b320_contract.success_labels)
            retained = True
        self._b320_condition_flags = tuple(flags)
        values = {
            **faucet, **release, "xy4_x_complete": self._xy4_sequence.x_complete,
            "xy4_y_enabled": enabled, "xy4_bottle_retained": retained,
            "success_now": bool(all(flags)),
        }
        self._b320_last_metrics = values
        return values

    def _compute_failure(self) -> bool:
        if self._hs005_order_violation:
            self._b320_failure_reason = "hs005_scrub_band_order_violation"
            self._success_counter = 0
            return True
        return super()._compute_failure()

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        # XY4 keys are outside the shared B320 prefix; show the contract text.
        if row.get("label"):
            return str(row["label"])
        return HandsOnB320BimanualIsaacLabEnv._success_condition_display_label(row)

    def _task_status_text(self) -> str:
        if self._xy4_success_latched:
            return "XY4 complete"
        if not self._xy4_sequence.x_complete:
            return f"XY4: scrub {self._hs005_order_progress}/3 | release and stabilize"
        if not self._ka021_water_filled:
            return f"XY4: fill mug {self._ka021_fill_hold_frames}/15"
        return "XY4: park mug and close faucet | keep bottle upright"

    def _build_observation_space(self):
        result = super()._build_observation_space()
        result.spaces["xy4_hidden_state"] = spaces.Box(
            -np.inf, np.inf, shape=(len(HIDDEN_FIELDS),), dtype=np.float64,
        )
        return result

    def _hidden_observation(self) -> np.ndarray:
        previous = self._hs005_previous_bristle_world
        scalars = [float(getattr(self, key)) for key in HIDDEN_SCALARS]
        # An unavailable diagnostic distance has a finite, explicit sentinel.
        scalars = [v if np.isfinite(v) else 1e6 for v in scalars]
        return np.asarray([
            *scalars, *self._hs005_coverage_mask,
            float(previous is not None), *(previous if previous is not None else np.zeros(3)),
            self._xy4_sequence.x_complete_step, self._xy4_sequence.last_update_step,
            -1 if self._ka021_fill_last_update_step is None else self._ka021_fill_last_update_step,
            self._xy4_fill_update_step,
            *(self._hs005_bottle_reset_pose if self._hs005_bottle_reset_pose is not None else np.zeros(7)),
            *(self._hs005_brush_reset_pose if self._hs005_brush_reset_pose is not None else np.zeros(7)),
            *self._hs005_reset_parameters, *self._ka021_cup_park_position_w,
            self._xy4_scrub_direction,
        ], dtype=np.float64)

    def _compute_observation(self):
        observation = super()._compute_observation()
        observation["xy4_hidden_state"] = self._hidden_observation()
        return observation

    def get_recording_schema(self):
        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        result["fields"] = dict(result["fields"])
        result["fields"]["xy4_hidden_state"] = {
            "shape": [len(HIDDEN_FIELDS)], "dtype": "float64",
            "source": "xy4_action_aligned_task_state", "layout": list(HIDDEN_FIELDS),
            "visibility": "hidden_task_logic", "replay_domain": "task_state",
            "physical_replay": True, "replay_restore_supported": False,
            "replay_atol": 1e-6,
        }
        result["task_state"] = {
            "task_name": TASK_NAME, "field_names": ["xy4_hidden_state"],
            "hidden_field_names": ["xy4_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return result

    def get_physical_replay_reset_contract(self):
        contract = super().get_physical_replay_reset_contract()
        contract["xy4_reset"] = {
            "poses_wxyz": {name: pose.tolist() for name, pose in self._xy4_layout.items()},
            "hs005_parameters": self._hs005_reset_parameters.tolist(),
            "cup_park_position_w": self._ka021_cup_park_position_w.tolist(),
            "cup_park_xy_bounds_m": [list(values) for values in PARK_CENTER_BOUNDS_XY],
            "joint_layout_attempts": self._xy4_layout_attempts,
            "source_tasks": ["B320-HS005", "B320-KA021"],
            "phase_state_restored": False,
        }
        return contract

    def _ka021_faucet_metadata(self):
        metadata = super()._ka021_faucet_metadata()
        metadata["cup_park_xy_bounds_m"] = [list(values) for values in PARK_CENTER_BOUNDS_XY]
        return metadata

    def _task_info(self):
        info = super()._task_info()
        info.update({
            "task_id": "XY4", "xy4_source_tasks": ["B320-HS005", "B320-KA021"],
            # Unavailable contact diagnostics use infinity internally. Export
            # null in receipts; keep the physical checker and its state intact.
            "hs005_metrics": {
                key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
                for key, value in self._last_hs005_success_metrics.items()
            },
            "xy4_x_complete_step": self._xy4_sequence.x_complete_step,
            "xy4_layout_attempts": self._xy4_layout_attempts,
            "xy4_reset_poses_wxyz": {k: v.tolist() for k, v in self._xy4_layout.items()},
            "xy4_layout_clearances_m": layout_clearances(self._xy4_layout) if self._xy4_layout else {},
            "xy4_hidden_state_fields": list(HIDDEN_FIELDS),
        })
        return info
