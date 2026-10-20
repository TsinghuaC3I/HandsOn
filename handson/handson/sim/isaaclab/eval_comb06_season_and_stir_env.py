"""Continuous BC014 + KA012 task; no inter-stage object or robot resets."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces
import numpy as np

from .b320_bimanual_env import (
    HandsOnB320BimanualIsaacLabEnv,
    _BC014_FLOW_MAX_LENGTH_M, _BC014_FLOW_MAX_POUR_SECONDS,
    _BC014_FLOW_TILT_THRESHOLD_DEG, _HAND_CONTACT_N, _RELEASE_CONTACT_N,
    _quat_multiply, _yaw_quaternion,
)
from .b320_bimanual_scene import BC014_JAR_TARGET_HEIGHT_M
from .eval_comb06_season_and_stir_scene import (
    OBJECT_NAMES, POT_RECEIVER_HEIGHT_M, RESET_MAX_ATTEMPTS, TASK_NAME,
    TRANSLATION_HALF_RANGES_M, YAW_HALF_RANGES_DEG, pot_flow_projection,
    rotation_wxyz,
)


SEASON, TRANSFER, STIR = range(3)
_FAILURES = (None, "xy6_salt_pour_timeout", "xy6_pepper_pour_timeout", "xy6_both_pour_timeout")
XY6_HIDDEN_STATE_FIELDS = (
    "phase", "season_hold_frames", "last_update_step", "jars_released",
    "salt_hit_seconds", "pepper_hit_seconds", "salt_pour_seconds", "pepper_pour_seconds",
    "salt_hit_latched", "pepper_hit_latched", "overlap_seconds", "overlap_latched",
    "season_contact_flag", "season_salt_flag", "season_pepper_flag", "season_overlap_flag",
    "left_contact_seen", "right_contact_seen", "left_contact_lost",
    "previous_angle_valid", "previous_angle_rad", "accumulated_rad",
    *(f"sector_{i}" for i in range(6)),
    "anchor_valid", *(f"anchor_{i}" for i in range(7)), "anchor_frame",
    "success_counter", "failure_code",
    "salt_stream_emitting", "pepper_stream_emitting",
    *(f"stir_flag_{i}" for i in range(5)),
)
_HIDDEN_FIELDS = XY6_HIDDEN_STATE_FIELDS


class HandsOnXY6IsaacLabEnv(HandsOnB320BimanualIsaacLabEnv):
    _xy6_dedicated_runtime = True

    def _is_b320_collection_task(self):
        return True

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"XY6 runtime requires {TASK_NAME!r}")
        self._xy6_contact_groups: dict[int, dict[str, Any]] = {}
        self._xy6_release_views: dict[str, Any] = {}
        self._xy6_reset_poses: dict[str, np.ndarray] = {}
        self._xy6_reset_attempts = 0
        self._clear_xy6_state()
        super().__init__(task_name, **kwargs)

    def _clear_xy6_state(self):
        self._xy6_phase = SEASON
        self._xy6_season_hold = 0
        self._xy6_last_update = -1
        self._xy6_control_tick = -1
        self._xy6_jars_released = False
        self._xy6_season_flags = (False,) * 4
        self._xy6_stir_flags = (False,) * 5
        self._xy6_values: dict[str, Any] = {}
        self._xy6_flow_emitting = {"salt_jar": False, "pepper_jar": False}

    def _sample_reset_state(self, *, clear_manual_request: bool):
        self._clear_xy6_state()
        if self._xy6_contact_groups:
            self._b320_contact_views = self._xy6_contact_groups[SEASON]
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._xy6_reset_poses = self._sample_xy6_layout()
        self._b320_reset_poses.update({n: p.copy() for n, p in self._xy6_reset_poses.items()})
        if not self._launch_isaac:
            self._poses.update({n: p.copy() for n, p in self._xy6_reset_poses.items()})
        for role in ("salt_jar", "pepper_jar"):
            self._set_bc014_flow_visual(role, None)

    def _sample_xy6_layout(self):
        # Conservative plan-view envelopes are CPU prefilters, not collision
        # certification. Full robot/object geometry is checked by live audits.
        radii = {"salt_jar": .029, "pepper_jar": .029, "dutch_oven": .103,
                 "dutch_oven_lid": .071, "spoon": .13}
        for attempt in range(1, RESET_MAX_ATTEMPTS + 1):
            poses = {}
            for spec in self._b320_contract.object_specs:
                pose = np.array((*spec.home_translation, *spec.home_orientation_wxyz), dtype=np.float64)
                if self.randomize:
                    bounds = np.asarray(TRANSLATION_HALF_RANGES_M[spec.name])
                    pose[:2] += self._random.uniform(-bounds, bounds)
                    yaw = self._random.uniform(-YAW_HALF_RANGES_DEG[spec.name], YAW_HALF_RANGES_DEG[spec.name])
                    pose[3:7] = _quat_multiply(_yaw_quaternion(float(yaw)), pose[3:7])
                poses[spec.name] = pose
            clear = all(
                np.linalg.norm(poses[a][:2] - poses[b][:2]) >= radii[a] + radii[b] + .020
                for i, a in enumerate(OBJECT_NAMES) for b in OBJECT_NAMES[i + 1:]
            )
            reachable = self._ka012_pot_reset_is_reachable(poses["dutch_oven"])
            for role, side_y in (("salt_jar", .065), ("pepper_jar", -.065), ("spoon", -.065)):
                grasp = poses[role][:3] + np.array([0., 0., .06])
                reachable &= np.linalg.norm(grasp - np.array([-.75, side_y, 1.34])) <= .75
            if clear and reachable:
                self._xy6_reset_attempts = attempt
                return poses
        raise RuntimeError("XY6 bounded layout sampler found no clear reachable layout")

    def _randomized_pose(self, pose, *, fixture, object_name=None):
        if object_name in self._xy6_reset_poses:
            return self._xy6_reset_poses[object_name].copy()
        return super()._randomized_pose(pose, fixture=fixture, object_name=object_name)

    def _init_task_specific_runtime(self):
        super()._init_task_specific_runtime()
        self._xy6_contact_groups = {SEASON: dict(self._b320_contact_views), STIR: {}}
        self._xy6_release_views = {}
        if not self._robots:
            return
        physics = next(iter(self._robots.values()))._physics_sim_view
        for side, role in (("left", "dutch_oven"), ("right", "spoon")):
            filters = self._hand_body_paths(side)
            self._xy6_contact_groups[STIR][side] = physics.create_rigid_contact_view(
                self._role_body_path(role), filter_patterns=filters,
                max_contact_data_count=max(96, len(filters) * 12),
            )
        filters = self._hand_body_paths("left") + self._hand_body_paths("right")
        for role in ("salt_jar", "pepper_jar"):
            self._xy6_release_views[role] = physics.create_rigid_contact_view(
                self._role_body_path(role), filter_patterns=filters,
                max_contact_data_count=max(96, len(filters) * 12),
            )

    def _configure_task_simulation_settings(self):
        super()._configure_task_simulation_settings()
        # The pot-interior predicate checks the stream against actual shapes.
        # Enable those queries in ordinary collection, including headless runs.
        from pxr import PhysxSchema
        for prim in self._stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableSceneQuerySupportAttr().Set(True)

    def _read_contacts(self, *, update_latches):
        if self._xy6_contact_groups:
            group = SEASON if self._xy6_phase == SEASON else STIR
            self._b320_contact_views = self._xy6_contact_groups[group]
        previous_seen = bool(self._b320_contact_seen["left"])
        super()._read_contacts(update_latches=update_latches)
        if (update_latches and self._xy6_phase == STIR and previous_seen
                and self._contact_sample_readable("left")
                and self._b320_contact_force["left"] < _RELEASE_CONTACT_N):
            self._b320_left_contact_lost = True

    def _jars_are_released(self):
        samples = [self._read_b320_contact_force_sample(self._xy6_release_views.get(n), self._sim_dt)
                   for n in ("salt_jar", "pepper_jar")]
        return all(readable and force < _RELEASE_CONTACT_N for force, readable in samples)

    def _flow_path_clear(self, sample):
        if not self._launch_isaac:
            return True  # geometry-only interface; not physical collision evidence
        from omni.physx import get_physx_scene_query_interface
        # Start just outside the emitter cap, whose outward axis defines flow.
        start = sample["cap_position"] + sample["cap_axis"] * .003
        result = get_physx_scene_query_interface().raycast_closest(
            tuple(float(v) for v in start),
            tuple(float(v) for v in sample["cap_axis"]),
            max(0.0, sample["stream_length_m"] - .003),
        )
        return not bool(result.get("hit", False))

    def _seasoning_projection(self, role):
        sample = pot_flow_projection(
            self._pose(role), self._pose("dutch_oven"),
            jar_height=BC014_JAR_TARGET_HEIGHT_M,
            tilt_threshold=_BC014_FLOW_TILT_THRESHOLD_DEG,
            maximum_length=_BC014_FLOW_MAX_LENGTH_M,
        )
        if sample["projects_into_zone"]:
            sample["projects_into_zone"] = self._flow_path_clear(sample)
        return sample

    def _metrics(self, *, update):
        if update and self._xy6_last_update != self._xy6_control_tick:
            self._xy6_last_update = self._xy6_control_tick
            self._read_contacts(update_latches=True)
            if self._xy6_phase == SEASON:
                for role in ("salt_jar", "pepper_jar"):
                    self._bc014_goal_centers_xy[role] = self._pose("dutch_oven")[:2].copy()
                values, flags = super()._seasoning_metrics(update=True)
                self._xy6_season_flags = tuple(flags)
                self._xy6_values.update(values)
                self._xy6_flow_emitting = {
                    "salt_jar": bool(values["salt_stream_emitting"]),
                    "pepper_jar": bool(values["pepper_stream_emitting"]),
                }
                hits_complete = all(self._bc014_flow_hit_latched.values()) and self._bc014_flow_overlap_latched
                timed_out = [s for s, seconds in self._bc014_flow_pour_seconds.items()
                             if seconds > _BC014_FLOW_MAX_POUR_SECONDS + 1e-9]
                if timed_out and not hits_complete and self._b320_failure_reason is None:
                    code = 3 if len(timed_out) == 2 else 1 if timed_out[0] == "left" else 2
                    self._b320_failure_reason = _FAILURES[code]
                self._xy6_season_hold = self._xy6_season_hold + 1 if all(flags) else 0
                if self._xy6_season_hold >= self._b320_contract.stable_frames and self._b320_failure_reason is None:
                    self._xy6_phase = TRANSFER
                    self._b320_contact_seen = {"left": False, "right": False}
                    self._b320_contact_peak = {"left": 0.0, "right": 0.0}
                    self._b320_release_frame = {"left": None, "right": None}
                    for role in ("salt_jar", "pepper_jar"):
                        self._xy6_flow_emitting[role] = False
                        self._set_bc014_flow_visual(role, None)
            elif self._b320_failure_reason is None:
                ready = self._xy6_phase == STIR or (
                    self._jars_are_released()
                    and all(self._contact_at_least(s, _HAND_CONTACT_N) for s in ("left", "right"))
                )
                if ready:
                    # Invoke the atomic evaluator exactly once per control step.
                    # Its read path mutates angle history, so overlays/observations
                    # below use this cached result rather than invoking it again.
                    values, flags = super()._stir_metrics(update=True)
                    self._xy6_values.update(values)
                    self._xy6_stir_flags = tuple(flags)
                    if self._b320_stir_anchor_pose is not None:
                        self._xy6_phase = STIR
                        self._xy6_jars_released = True
        self._b320_condition_flags = (
            *self._xy6_season_flags, self._xy6_jars_released, *self._xy6_stir_flags,
        )
        values = {
            "phase": self._xy6_phase,
            "season_complete": self._xy6_phase > SEASON,
            "jars_released": self._xy6_jars_released,
            "salt_hit_seconds": self._bc014_flow_hit_seconds["left"],
            "pepper_hit_seconds": self._bc014_flow_hit_seconds["right"],
            "overlap_hit_seconds": self._bc014_flow_overlap_seconds,
            "salt_pour_seconds": self._bc014_flow_pour_seconds["left"],
            "pepper_pour_seconds": self._bc014_flow_pour_seconds["right"],
            "stir_accumulated_deg": abs(float(np.degrees(self._b320_stir_accumulated_rad))),
            "stir_checkpoint_count": len(self._b320_stir_sectors),
            "pot_drift_m": float(self._xy6_values.get("pot_drift_m", 0.0)),
            "success_now": bool(all(self._b320_condition_flags) and self._b320_failure_reason is None),
        }
        self._b320_last_metrics = values
        return values

    def _compute_failure(self):
        return bool(self._b320_failure_reason is not None or super()._compute_failure())

    def step_with_current_joint_targets(self, *, start_time=None):
        # The shared step increments env_step *after* task evaluation. Own a
        # control transaction token so a later reader cannot count that action
        # twice merely because the public step number has already advanced.
        self._xy6_control_tick += 1
        return super().step_with_current_joint_targets(start_time=start_time)

    def _hidden_state(self):
        anchor = self._b320_stir_anchor_pose
        return np.asarray([
            self._xy6_phase, self._xy6_season_hold, self._xy6_last_update, self._xy6_jars_released,
            *[self._bc014_flow_hit_seconds[s] for s in ("left", "right")],
            *[self._bc014_flow_pour_seconds[s] for s in ("left", "right")],
            *[self._bc014_flow_hit_latched[s] for s in ("left", "right")],
            self._bc014_flow_overlap_seconds, self._bc014_flow_overlap_latched,
            *self._xy6_season_flags,
            self._b320_contact_seen["left"], self._b320_contact_seen["right"], self._b320_left_contact_lost,
            self._b320_stir_previous_angle is not None,
            0.0 if self._b320_stir_previous_angle is None else self._b320_stir_previous_angle,
            self._b320_stir_accumulated_rad,
            *[i in self._b320_stir_sectors for i in range(6)],
            anchor is not None, *(np.zeros(7) if anchor is None else anchor),
            -1 if self._b320_stir_anchor_frame is None else self._b320_stir_anchor_frame,
            self._success_counter, _FAILURES.index(self._b320_failure_reason),
            self._xy6_flow_emitting["salt_jar"], self._xy6_flow_emitting["pepper_jar"],
            *self._xy6_stir_flags,
        ], dtype=np.float64)

    def _restore_xy6_hidden_state(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(len(_HIDDEN_FIELDS))
        h = dict(zip(_HIDDEN_FIELDS, values))
        self._xy6_phase = int(h["phase"])
        self._xy6_season_hold = int(h["season_hold_frames"])
        self._xy6_last_update = int(h["last_update_step"])
        self._xy6_control_tick = self._xy6_last_update
        self._xy6_jars_released = bool(h["jars_released"])
        for side, name in (("left", "salt"), ("right", "pepper")):
            self._bc014_flow_hit_seconds[side] = h[f"{name}_hit_seconds"]
            self._bc014_flow_pour_seconds[side] = h[f"{name}_pour_seconds"]
            self._bc014_flow_hit_latched[side] = bool(h[f"{name}_hit_latched"])
            self._b320_contact_seen[side] = bool(h[f"{side}_contact_seen"])
        self._bc014_flow_overlap_seconds = h["overlap_seconds"]
        self._bc014_flow_overlap_latched = bool(h["overlap_latched"])
        self._xy6_season_flags = tuple(bool(h[f"season_{n}_flag"]) for n in ("contact", "salt", "pepper", "overlap"))
        self._b320_left_contact_lost = bool(h["left_contact_lost"])
        self._b320_stir_previous_angle = h["previous_angle_rad"] if h["previous_angle_valid"] else None
        self._b320_stir_accumulated_rad = h["accumulated_rad"]
        self._b320_stir_sectors = {i for i in range(6) if h[f"sector_{i}"]}
        self._b320_stir_anchor_pose = np.asarray([h[f"anchor_{i}"] for i in range(7)]) if h["anchor_valid"] else None
        self._b320_stir_anchor_frame = int(h["anchor_frame"]) if h["anchor_valid"] else None
        self._success_counter = int(h["success_counter"])
        self._b320_failure_reason = _FAILURES[int(h["failure_code"])]
        self._xy6_flow_emitting = {
            "salt_jar": bool(h["salt_stream_emitting"]),
            "pepper_jar": bool(h["pepper_stream_emitting"]),
        }
        self._xy6_stir_flags = tuple(bool(h[f"stir_flag_{i}"]) for i in range(5))

    def _build_observation_space(self):
        result = super()._build_observation_space()
        result.spaces["xy6_hidden_state"] = spaces.Box(-np.inf, np.inf, (len(_HIDDEN_FIELDS),), dtype=np.float64)
        result.spaces["xy6_object_poses"] = spaces.Box(-np.inf, np.inf, (len(OBJECT_NAMES), 7), dtype=np.float64)
        return result

    def _compute_observation(self):
        observation = super()._compute_observation()
        state = observation["state"]
        left, right = ("salt_jar", "pepper_jar") if self._xy6_phase == SEASON else ("dutch_oven", "spoon")
        state["left_role_pose"] = self._pose(left).copy()
        state["right_role_pose"] = self._pose(right).copy()
        state["fixture_pose"] = self._pose("dutch_oven").copy()
        state["goal_pose"] = self._pose("dutch_oven").copy()
        state["goal_pose"][:3] += rotation_wxyz(state["goal_pose"][3:7]) @ np.array([0., POT_RECEIVER_HEIGHT_M, 0.])
        state["stage_state"] = np.asarray([
            self._xy6_phase, self._xy6_season_hold, self._xy6_jars_released,
            self._b320_left_contact_lost, len(self._b320_stir_sectors),
            self._b320_stir_accumulated_rad, self._success_counter, self.env_step,
        ], dtype=np.float64)
        observation["xy6_hidden_state"] = self._hidden_state()
        observation["xy6_object_poses"] = np.stack([self._pose(n) for n in OBJECT_NAMES])
        return observation

    def get_recording_schema(self):
        schema = super().get_recording_schema()
        if schema is None:
            return None
        schema = dict(schema)
        schema["fields"] = {n: dict(s) for n, s in schema["fields"].items()}
        for name, layout in (("xy6_hidden_state", _HIDDEN_FIELDS),):
            box = self.observation_space[name]
            schema["fields"][name] = {
                "shape": list(box.shape), "dtype": "float64", "layout": list(layout),
                "source": "xy6_action_aligned_task_state", "replay_domain": "task_state",
                "visibility": "hidden_task_logic" if name == "xy6_hidden_state" else "physical_state",
                "physical_replay": True, "replay_restore_supported": False, "replay_atol": 1e-6,
            }
        schema["task_state"] = {
            "task_name": TASK_NAME, "field_names": ["xy6_hidden_state"],
            "hidden_field_names": ["xy6_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return schema

    def restore_recorded_initial_state(self, state_parts):
        super().restore_recorded_initial_state(state_parts)
        if "xy6_object_poses" in state_parts:
            poses = np.asarray(state_parts["xy6_object_poses"]).reshape(len(OBJECT_NAMES), 7)
            for name, pose in zip(OBJECT_NAMES, poses):
                if self._launch_isaac:
                    view = self._task_body_views.get(name)
                    if view is None or not self._set_task_body_world_pose(view, pose):
                        raise RuntimeError(f"XY6 initial pose restore failed: {name}")
                self._poses[name] = pose.copy()
        if "xy6_hidden_state" in state_parts:
            self._restore_xy6_hidden_state(state_parts["xy6_hidden_state"])
        return self._compute_observation()

    def _restore_recorded_robot_command_state(self, state_parts):
        super()._restore_recorded_robot_command_state(state_parts)
        # XY6 records both complete 20-joint Wuji hands. The inherited legacy
        # ABC visual restorer splits Allegro hands at 16 instead.
        hands = np.asarray(state_parts.get("gripper_pose", ()), dtype=np.float64).ravel()
        if hands.size == 40:
            self._right_hand[:] = hands[:20]
            self._left_hand[:] = hands[20:]

    def _task_status_text(self):
        return (
            "XY6 - Pour salt and pepper into pot",
            "XY6 - Release jars; hold pot and spoon",
            f"XY6 - Stir: {abs(np.degrees(self._b320_stir_accumulated_rad)):.0f} deg / >650 deg",
        )[self._xy6_phase]

    def restore_recorded_visual_state(self, state_parts, *, robot_iterations=8):
        observation = super().restore_recorded_visual_state(
            state_parts, robot_iterations=robot_iterations,
        )
        for role in ("salt_jar", "pepper_jar"):
            sample = pot_flow_projection(
                self._pose(role), self._pose("dutch_oven"),
                jar_height=BC014_JAR_TARGET_HEIGHT_M,
                tilt_threshold=_BC014_FLOW_TILT_THRESHOLD_DEG,
                maximum_length=_BC014_FLOW_MAX_LENGTH_M,
            )
            sample["emitting"] = bool(self._xy6_flow_emitting[role] and sample["ray_valid"])
            self._set_bc014_flow_visual(role, sample)
        return observation

    def _task_info(self):
        info = super()._task_info()
        info["xy6_phase"] = self._xy6_phase
        info["xy6_hidden_state_fields"] = list(_HIDDEN_FIELDS)
        info["xy6_reset"] = self._xy6_reset_metadata()
        return info

    def _xy6_reset_metadata(self):
        return {
            "attempts": self._xy6_reset_attempts,
            "translation_half_ranges_m": dict(TRANSLATION_HALF_RANGES_M),
            "yaw_half_ranges_deg": dict(YAW_HALF_RANGES_DEG),
            "poses_wxyz": {n: p.tolist() for n, p in self._xy6_reset_poses.items()},
            "object_order": list(OBJECT_NAMES),
        }

    def get_physical_replay_reset_contract(self):
        contract = super().get_physical_replay_reset_contract()
        contract["xy6_reset"] = self._xy6_reset_metadata()
        contract["xy6_hidden_state_fields"] = list(_HIDDEN_FIELDS)
        contract["xy6_hidden_state"] = self._hidden_state().tolist()
        contract["xy6_composition"] = {
            "sources": ["t30_pour_seasonings", "t45_stir_pot"],
            "shared_receiver": "dutch_oven", "continuous_scene": True,
            "phase_labels": ["season", "transfer", "stir"],
        }
        return contract
