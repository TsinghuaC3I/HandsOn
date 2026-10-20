"""Build a stable, deterministic physical origin before teleoperation starts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..physx_ground_truth import PhysxGroundTruthTrace


STABLE_ORIGIN_ENV = "HANDSON_STABLE_RECORDING_ORIGIN"
JOINT_VELOCITY_LIMIT_ENV = "HANDSON_STABLE_JOINT_VELOCITY_LIMIT"
STABLE_ORIGIN_RESET_ATTEMPTS_ENV = "HANDSON_STABLE_ORIGIN_RESET_ATTEMPTS"
DEFAULT_LINEAR_VELOCITY_LIMIT = 0.001
DEFAULT_ANGULAR_VELOCITY_LIMIT = 0.002
DEFAULT_JOINT_VELOCITY_LIMIT = 0.001
DEFAULT_STABLE_SECONDS = 0.5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_STABILITY_SAMPLE_SECONDS = 0.05
_COLLECTION_INITIAL_SETTLE_SECONDS = 1.0
_COLLECTION_POST_SETTLE_TIMEOUT_SECONDS = 3.0
_COLLECTION_RESET_ATTEMPTS = 3
VELOCITY_CLAMP_SCOPE_NONE = "none"
VELOCITY_CLAMP_SCOPE_GATE_ONLY = "gate_only"
VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS = (
    "all_initialization_steps"
)
_VELOCITY_CLAMP_SCOPES = {
    VELOCITY_CLAMP_SCOPE_NONE,
    VELOCITY_CLAMP_SCOPE_GATE_ONLY,
    VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS,
}
_TASK_RESET_ATTEMPTS = {
    # A01 has five independently settling tabletop bodies, including rounded
    # and compound collision shapes.  A rejected sample is harmless, whereas
    # exhausting the generic three attempts tears down the collection window.
    "t26_place_fruit_in_bucket": 12,
}
_TASK_JOINT_VELOCITY_LIMITS = {
    # B02 intentionally uses the source MuJoCo soft-hand drives so the small
    # nut remains graspable.  PhysX leaves a sub-half-degree/second joint
    # residual while the fixed bolt and free nut are already below the strict
    # rigid-body limits.  Keep this narrow exception local to B02.
    "b02_nut_bolt_assembly": 0.008,
    # The left thumb remains in light reset contact with the camera.  The
    # measured PhysX solver jitter is about 0.0026 rad/s, while the rest of the
    # scene is settled.  This explicit near-zero limit was validated by strict
    # fresh-process replay rather than chosen to bypass the stability gate.
    "t23_photograph_logo": 0.003,
}
_TASK_ANGULAR_VELOCITY_LIMITS = {
    # Both C05 robot bases are fixed and every robot joint reads exactly zero,
    # but PhysX reports a repeatable ~0.00258 rad/s residual on a fixed-base
    # articulation root.  Task bodies settle below 0.0002 rad/s.  Keep a
    # narrow task-specific near-zero allowance instead of weakening every
    # task's origin contract.
    "t14_move_and_click_mouse": 0.003,
}
_TASK_STATIC_POSE_FALLBACK_LIMITS = {
    # A06 uses a 27-shape convex cup.  GPU PhysX can leave a non-integrated
    # contact-correction velocity in its sleeping tensor state even while the
    # transform is bitwise stationary.  Accept that narrow readback case only
    # when every physics-frame task pose is substantially tighter than motion
    # allowed by the ordinary velocity gate; robot velocities remain strict.
    "t19_orient_cup_upright": (1.0e-6, 1.0e-5),
}
_STATIC_POSE_FALLBACK_TASK = "t19_orient_cup_upright"


class RecordingOriginStabilityError(RuntimeError):
    """Raised when a reset state cannot reach its controlled origin in time."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def stable_recording_origin_kwargs_for_task(
    task_name: object,
) -> dict[str, Any]:
    """Return the strict recording-origin contract for a new episode."""

    # Every task first advances naturally so contact manifolds can converge,
    # then gets one canonical zero-velocity/sleep transaction.  The following
    # strict gate is deliberately write-free: if PhysX still corrects a pose or
    # regenerates velocity, the sample must keep settling or be rejected rather
    # than being made to look stable by a per-frame velocity clamp.
    kwargs: dict[str, Any] = {
        "stability_metric": "velocity",
        "initial_settle_seconds": _COLLECTION_INITIAL_SETTLE_SECONDS,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": True,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "timeout_seconds": _COLLECTION_POST_SETTLE_TIMEOUT_SECONDS,
    }
    normalized_task_name = str(task_name or "").strip().lower()
    angular_velocity_limit = _TASK_ANGULAR_VELOCITY_LIMITS.get(
        normalized_task_name
    )
    if angular_velocity_limit is not None:
        kwargs["angular_velocity_limit"] = angular_velocity_limit
    pose_fallback_limits = _TASK_STATIC_POSE_FALLBACK_LIMITS.get(
        normalized_task_name
    )
    if pose_fallback_limits is not None:
        kwargs.update(
            {
                "stability_metric": "velocity_or_static_pose",
                "pose_translation_limit": pose_fallback_limits[0],
                "pose_rotation_limit": pose_fallback_limits[1],
            }
        )
    return kwargs


def stable_recording_origin_reset_attempts_for_task(task_name: object) -> int:
    """Return how many freshly sampled reset poses may be tried."""

    configured = os.environ.get(STABLE_ORIGIN_RESET_ATTEMPTS_ENV)
    if configured is not None:
        try:
            attempts = int(configured)
        except ValueError as exc:
            raise ValueError(
                f"{STABLE_ORIGIN_RESET_ATTEMPTS_ENV} must be a positive integer; "
                f"got {configured!r}."
            ) from exc
        if attempts <= 0:
            raise ValueError(
                f"{STABLE_ORIGIN_RESET_ATTEMPTS_ENV} must be a positive integer; "
                f"got {configured!r}."
            )
        return attempts
    normalized_task_name = str(task_name or "").strip().lower()
    return _TASK_RESET_ATTEMPTS.get(
        normalized_task_name,
        _COLLECTION_RESET_ATTEMPTS,
    )


def stable_recording_origin_kwargs_from_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recover the exact stabilization inputs stored in a replay manifest."""

    if not isinstance(report, Mapping):
        return {}
    numeric_fields = {
        "linear_velocity_limit": "linear_velocity_limit",
        "angular_velocity_limit": "angular_velocity_limit",
        "joint_velocity_limit": "joint_velocity_limit",
        "initial_settle_seconds": "initial_settle_seconds",
        "pose_translation_limit": "pose_translation_limit",
        "pose_rotation_limit": "pose_rotation_limit",
        "stable_seconds_required": "stable_seconds",
        "stability_sample_seconds": "stability_sample_seconds",
        "timeout_seconds": "timeout_seconds",
    }
    kwargs: dict[str, Any] = {}
    for report_name, argument_name in numeric_fields.items():
        value = report.get(report_name)
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                "Recorded stable-origin parameter must be finite and "
                f"non-negative: {report_name}={value!r}"
            )
        kwargs[argument_name] = numeric
    metric = report.get("stability_metric")
    if metric is not None:
        kwargs["stability_metric"] = str(metric)
    zero_after_settle = report.get("post_settle_velocity_zeroing")
    if zero_after_settle is not None:
        if type(zero_after_settle) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "post_settle_velocity_zeroing must be boolean."
            )
        kwargs["post_settle_velocity_zeroing"] = zero_after_settle
    sleep_after_settle = report.get("post_settle_sleeping")
    if sleep_after_settle is not None:
        if type(sleep_after_settle) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter post_settle_sleeping must be boolean."
            )
        kwargs["post_settle_sleeping"] = sleep_after_settle
    velocity_clamp = report.get("stabilization_velocity_clamp")
    if velocity_clamp is not None:
        if type(velocity_clamp) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "stabilization_velocity_clamp must be boolean."
            )
    per_step_velocity_writes = report.get(
        "initialization_per_physics_step_velocity_writes"
    )
    if per_step_velocity_writes is not None:
        if type(per_step_velocity_writes) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "initialization_per_physics_step_velocity_writes must be "
                "boolean."
            )
        if velocity_clamp is not None and (
            per_step_velocity_writes != velocity_clamp
        ):
            raise ValueError(
                "Recorded stable-origin velocity clamp flags must match."
            )
    velocity_clamp_scope = report.get("stabilization_velocity_clamp_scope")
    # A few historical reports omitted one redundant boolean or explicitly
    # wrote scope=none before both booleans were emitted.  Normalize those
    # representations; enabled reports still need internally consistent step
    # and write counts in the certifier.
    if velocity_clamp is None and per_step_velocity_writes is not None:
        velocity_clamp = per_step_velocity_writes
    elif per_step_velocity_writes is None and velocity_clamp is not None:
        per_step_velocity_writes = velocity_clamp
    elif (
        velocity_clamp is None
        and per_step_velocity_writes is None
        and velocity_clamp_scope == VELOCITY_CLAMP_SCOPE_NONE
    ):
        velocity_clamp = False
        per_step_velocity_writes = False
    if velocity_clamp is not None:
        kwargs["stabilization_velocity_clamp"] = velocity_clamp
    if velocity_clamp_scope is None:
        # Reports written before the scope field existed clamped only the
        # strict gate.  Preserve that call history exactly during replay.
        if velocity_clamp is True:
            velocity_clamp_scope = VELOCITY_CLAMP_SCOPE_GATE_ONLY
        elif velocity_clamp is False:
            velocity_clamp_scope = VELOCITY_CLAMP_SCOPE_NONE
    if velocity_clamp_scope is not None:
        if velocity_clamp is None:
            raise ValueError(
                "Recorded stable-origin velocity clamp scope requires its "
                "boolean clamp flag."
            )
        if (
            not isinstance(velocity_clamp_scope, str)
            or velocity_clamp_scope not in _VELOCITY_CLAMP_SCOPES
        ):
            raise ValueError(
                "Recorded stable-origin parameter "
                "stabilization_velocity_clamp_scope is invalid: "
                f"{velocity_clamp_scope!r}."
            )
        if velocity_clamp is True and (
            velocity_clamp_scope == VELOCITY_CLAMP_SCOPE_NONE
        ):
            raise ValueError(
                "Recorded velocity clamp cannot use scope 'none'."
            )
        if velocity_clamp is False and (
            velocity_clamp_scope != VELOCITY_CLAMP_SCOPE_NONE
        ):
            raise ValueError(
                "Recorded disabled velocity clamp must use scope 'none'."
            )
        kwargs["stabilization_velocity_clamp_scope"] = (
            velocity_clamp_scope
        )
    return kwargs


def stable_recording_origin_enabled() -> bool:
    return os.environ.get(STABLE_ORIGIN_ENV, "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _resolve_joint_velocity_limit(
    raw_env: Any,
    requested: float | None,
) -> tuple[float, str]:
    configured = os.environ.get(JOINT_VELOCITY_LIMIT_ENV)
    if configured is not None:
        return float(configured), "environment_variable"
    if requested is not None:
        return float(requested), "argument"

    environment_limit = getattr(
        raw_env,
        "stable_recording_origin_joint_velocity_limit",
        None,
    )
    if environment_limit is not None:
        return float(environment_limit), "environment"

    task_name = str(getattr(raw_env, "task_name", "")).strip().lower()
    if task_name in _TASK_JOINT_VELOCITY_LIMITS:
        return float(_TASK_JOINT_VELOCITY_LIMITS[task_name]), "task_default"
    return float(DEFAULT_JOINT_VELOCITY_LIMIT), "default"


def _valid_states(
    snapshot: Mapping[str, Any],
    state_name: str,
    valid_name: str,
) -> list[np.ndarray]:
    states = np.asarray(snapshot.get(state_name, np.zeros((0, 13))))
    valid = np.asarray(snapshot.get(valid_name, np.zeros(0))).reshape(-1)
    if states.ndim != 2 or states.shape[-1] < 13:
        return []
    count = min(states.shape[0], valid.size)
    return [
        np.asarray(states[index, :13], dtype=np.float64)
        for index in range(count)
        if bool(valid[index])
    ]


def _object_motion(snapshot: Mapping[str, Any]) -> tuple[float, float] | None:
    states = _valid_states(
        snapshot,
        "physx_task_rigid_body_state",
        "physx_task_rigid_body_valid",
    )
    states.extend(
        _valid_states(
            snapshot,
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        )
    )
    linear_speeds = [
        float(np.linalg.norm(state[7:10]))
        for state in states
        if np.all(np.isfinite(state[7:13]))
    ]
    angular_speeds = [
        float(np.linalg.norm(state[10:13]))
        for state in states
        if np.all(np.isfinite(state[7:13]))
    ]

    articulation_velocity = np.asarray(
        snapshot.get(
            "physx_task_articulation_dof_velocity",
            np.zeros((0, 0)),
        ),
        dtype=np.float64,
    )
    articulation_valid = np.asarray(
        snapshot.get(
            "physx_task_articulation_dof_valid",
            np.zeros((0, 0)),
        )
    )
    if articulation_velocity.shape == articulation_valid.shape:
        valid_velocity = articulation_velocity[articulation_valid.astype(bool)]
        if valid_velocity.size:
            angular_speeds.append(
                float(np.max(np.abs(valid_velocity)))
            )

    deformable_velocity = np.asarray(
        snapshot.get("physx_deformable_velocity", np.zeros((0, 3))),
        dtype=np.float64,
    )
    deformable_valid = np.asarray(
        snapshot.get("physx_deformable_valid", np.zeros(0))
    ).reshape(-1)
    if (
        deformable_valid.size
        and bool(deformable_valid[0])
        and deformable_velocity.ndim == 2
        and deformable_velocity.shape[-1] == 3
        and deformable_velocity.size
    ):
        linear_speeds.append(
            float(np.max(np.linalg.norm(deformable_velocity, axis=-1)))
        )

    if not linear_speeds and not angular_speeds:
        return None
    return (
        max(linear_speeds, default=0.0),
        max(angular_speeds, default=0.0),
    )


def _task_motion_peaks(
    snapshot: Mapping[str, Any],
    trace: PhysxGroundTruthTrace,
    *,
    linear_velocity_limit: float,
    angular_velocity_limit: float,
    count: int = 5,
) -> list[dict[str, Any]]:
    peaks: dict[str, dict[str, Any]] = {}
    layouts = (
        (
            getattr(trace, "rigid_bodies", ()),
            "physx_task_rigid_body_state",
            "physx_task_rigid_body_valid",
        ),
        (
            getattr(trace, "articulations", ()),
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        ),
    )
    for entries, state_name, valid_name in layouts:
        states = np.asarray(snapshot.get(state_name, np.zeros((0, 13))))
        valid = np.asarray(snapshot.get(valid_name, np.zeros(0))).reshape(-1)
        for index, entry in enumerate(entries):
            if (
                index >= states.shape[0]
                or index >= valid.size
                or not bool(valid[index])
                or states.shape[-1] < 13
            ):
                continue
            state = np.asarray(states[index, :13], dtype=np.float64)
            if not np.all(np.isfinite(state[7:13])):
                continue
            linear = float(np.linalg.norm(state[7:10]))
            angular = float(np.linalg.norm(state[10:13]))
            score = max(
                linear / max(linear_velocity_limit, 1e-12),
                angular / max(angular_velocity_limit, 1e-12),
            )
            name = str(entry.name)
            previous = peaks.get(name)
            if previous is None or score > float(previous["limit_ratio"]):
                peaks[name] = {
                    "name": name,
                    "linear_velocity": linear,
                    "angular_velocity": angular,
                    "limit_ratio": score,
                }
    return sorted(
        peaks.values(),
        key=lambda item: float(item["limit_ratio"]),
        reverse=True,
    )[: max(0, int(count))]


def _task_pose_delta(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Return maximum consecutive task-body translation and rotation."""

    translations: list[float] = []
    rotations: list[float] = []
    for state_name, valid_name in (
        ("physx_task_rigid_body_state", "physx_task_rigid_body_valid"),
        (
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        ),
    ):
        before = np.asarray(previous.get(state_name, np.zeros((0, 13))))
        after = np.asarray(current.get(state_name, np.zeros((0, 13))))
        before_valid = np.asarray(
            previous.get(valid_name, np.zeros(0))
        ).reshape(-1)
        after_valid = np.asarray(
            current.get(valid_name, np.zeros(0))
        ).reshape(-1)
        if (
            before.ndim != 2
            or after.ndim != 2
            or before.shape != after.shape
            or before.shape[-1] < 7
            or before_valid.shape != after_valid.shape
            or before_valid.size != before.shape[0]
        ):
            return None
        before_valid = before_valid.astype(bool)
        after_valid = after_valid.astype(bool)
        # Comparing separately filtered lists can silently pair two different
        # bodies when validity changes between frames.  Require an identical
        # indexed body set for the whole consecutive-pose check.
        if not np.array_equal(before_valid, after_valid):
            return None
        for index in np.flatnonzero(before_valid):
            before_state = np.asarray(before[index, :7], dtype=np.float64)
            after_state = np.asarray(after[index, :7], dtype=np.float64)
            if not (
                np.all(np.isfinite(before_state))
                and np.all(np.isfinite(after_state))
            ):
                return None
            translations.append(
                float(np.linalg.norm(after_state[:3] - before_state[:3]))
            )
            before_quat = before_state[3:7]
            after_quat = after_state[3:7]
            before_norm = float(np.linalg.norm(before_quat))
            after_norm = float(np.linalg.norm(after_quat))
            if before_norm <= 1e-12 or after_norm <= 1e-12:
                return None
            cosine = abs(float(np.dot(before_quat, after_quat))) / (
                before_norm * after_norm
            )
            rotations.append(float(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))))
    if not translations and not rotations:
        return None
    return max(translations, default=0.0), max(rotations, default=0.0)


def _robot_motion(
    snapshot: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    root_states = _valid_states(
        snapshot,
        "physx_robot_root_state",
        "physx_robot_root_valid",
    )
    linear_speeds = [
        float(np.linalg.norm(state[7:10]))
        for state in root_states
        if np.all(np.isfinite(state[7:13]))
    ]
    angular_speeds = [
        float(np.linalg.norm(state[10:13]))
        for state in root_states
        if np.all(np.isfinite(state[7:13]))
    ]

    joint_velocity = np.asarray(
        snapshot.get("physx_robot_joint_velocity", np.zeros((0, 0))),
        dtype=np.float64,
    )
    joint_valid = np.asarray(
        snapshot.get("physx_robot_joint_valid", np.zeros((0, 0)))
    )
    joint_speeds: list[float] = []
    if joint_velocity.shape == joint_valid.shape:
        valid_velocity = joint_velocity[joint_valid.astype(bool)]
        valid_velocity = valid_velocity[np.isfinite(valid_velocity)]
        if valid_velocity.size:
            joint_speeds.append(float(np.max(np.abs(valid_velocity))))

    if not linear_speeds and not angular_speeds and not joint_speeds:
        return None
    return (
        max(linear_speeds, default=0.0),
        max(angular_speeds, default=0.0),
        max(joint_speeds, default=0.0),
    )


def _robot_joint_velocity_peaks(
    snapshot: Mapping[str, Any],
    trace: PhysxGroundTruthTrace,
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    velocity = np.asarray(
        snapshot.get("physx_robot_joint_velocity", np.zeros((0, 0))),
        dtype=np.float64,
    )
    valid = np.asarray(
        snapshot.get("physx_robot_joint_valid", np.zeros((0, 0)))
    ).astype(bool)
    peaks: list[dict[str, Any]] = []
    if velocity.shape != valid.shape:
        return peaks
    for side_index, entry in enumerate(trace.robots):
        if side_index >= velocity.shape[0]:
            continue
        for joint_index, joint_name in enumerate(entry.joint_names):
            if (
                joint_index >= velocity.shape[1]
                or not valid[side_index, joint_index]
                or not np.isfinite(velocity[side_index, joint_index])
            ):
                continue
            peaks.append(
                {
                    "side": entry.side,
                    "joint": joint_name,
                    "velocity": float(velocity[side_index, joint_index]),
                }
            )
    peaks.sort(key=lambda item: abs(float(item["velocity"])), reverse=True)
    return peaks[: max(0, int(count))]


def _rebuild_observation(env: Any, trace: PhysxGroundTruthTrace) -> Any:
    raw_env = getattr(env, "unwrapped", env)
    refresh = getattr(raw_env, "_refresh_object_poses", None)
    if callable(refresh):
        refresh()
    observation = raw_env._compute_observation()
    attach = getattr(raw_env, "_attach_physx_ground_truth", None)
    if callable(attach):
        observation = attach(observation)
    transform = getattr(env, "observation", None)
    if callable(transform):
        observation = transform(observation)
    trace.begin_substep_capture()
    return observation


def _zero_velocity_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    canonical = {
        name: np.asarray(value).copy()
        for name, value in snapshot.items()
    }
    for name in (
        "physx_robot_root_state",
        "physx_task_rigid_body_state",
        "physx_task_articulation_root_state",
    ):
        value = canonical.get(name)
        if value is not None and value.ndim >= 1 and value.shape[-1] >= 13:
            value[..., 7:13] = 0.0
    for name in (
        "physx_robot_joint_velocity",
        "physx_task_articulation_dof_velocity",
        "physx_deformable_velocity",
    ):
        value = canonical.get(name)
        if value is not None:
            value[...] = 0.0
    return canonical


def _write_canonical_state(
    trace: PhysxGroundTruthTrace,
    *,
    sync: bool = True,
) -> dict[str, int]:
    return trace.restore(
        _zero_velocity_snapshot(trace.snapshot()),
        trace.schema(),
        sync=sync,
    )


def _capture_robot_hold_targets(raw_env: Any) -> dict[str, Any]:
    """Capture one fixed post-reset joint target for each available robot."""

    robots = getattr(raw_env, "_robots", None)
    if not isinstance(robots, Mapping):
        robots = {}
    targets: dict[str, Any] = {}
    for side in ("right", "left"):
        robot = robots.get(side)
        if robot is None:
            control_robot = getattr(raw_env, "_control_robot", None)
            if callable(control_robot):
                robot = control_robot(side)
        if robot is not None:
            targets[side] = robot.data.joint_pos.clone()
    return targets


def _restore_default_robot_hold_targets(
    raw_env: Any,
    *,
    sync: bool,
) -> tuple[dict[str, Any], str]:
    """Restore the authored robot home and return its synchronized targets.

    Native Isaac Lab environments expose an atomic reset helper that writes
    default joint positions, exact-zero velocities, actuator drive targets,
    and Python-side command buffers together.  Older/interface-only fixtures
    fall back to their live readback for compatibility.
    """

    enabled = bool(
        getattr(
            raw_env,
            "stable_recording_origin_default_robot_restore_enabled",
            False,
        )
    )
    restore = getattr(raw_env, "_restore_default_robot_joint_state", None)
    if not enabled or not callable(restore):
        return _capture_robot_hold_targets(raw_env), "live_readback_fallback"

    restored = restore(reset_controllers=True, sync=sync)
    if not isinstance(restored, Mapping):
        raise RuntimeError(
            "Stable recording origin default robot restore must return targets."
        )
    targets = {
        str(side): target.clone()
        for side, target in restored.items()
        if target is not None
    }
    if not targets:
        raise RuntimeError(
            "Stable recording origin found no default robot joint targets."
        )
    return targets, "default_joint_state"


def _hold_robot_at_targets(
    raw_env: Any,
    targets: Mapping[str, Any],
    *,
    reset_controllers: bool = False,
) -> dict[str, int]:
    """Hold the robot at one fixed reset readback while physics settles."""

    robots = getattr(raw_env, "_robots", None)
    if not isinstance(robots, Mapping):
        robots = {}
    joint_targets = getattr(raw_env, "_joint_targets", None)
    hand_joint_ids = getattr(raw_env, "_hand_joint_ids", {})
    held_joints = 0
    held_hands = 0

    for side in ("right", "left"):
        target = targets.get(side)
        if target is None:
            continue
        robot = robots.get(side)
        if robot is None:
            control_robot = getattr(raw_env, "_control_robot", None)
            if callable(control_robot):
                robot = control_robot(side)
        if robot is None:
            continue

        robot.set_joint_position_target(target)
        set_velocity_target = getattr(
            robot,
            "set_joint_velocity_target",
            None,
        )
        if callable(set_velocity_target):
            velocity_target = target.clone()
            zero_ = getattr(velocity_target, "zero_", None)
            if callable(zero_):
                zero_()
            else:
                velocity_target[...] = 0.0
            set_velocity_target(velocity_target)
        write_data = getattr(robot, "write_data_to_sim", None)
        if callable(write_data):
            write_data()
        if isinstance(joint_targets, dict):
            joint_targets[side] = target.clone()
        held_joints += int(target.shape[-1])

        ids = list(hand_joint_ids.get(side, ()))
        hand_command = getattr(raw_env, f"_{side}_hand", None)
        if ids and isinstance(hand_command, np.ndarray):
            values = (
                target[:, ids]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            count = min(hand_command.size, values.size)
            hand_command[:count] = values[:count]
            held_hands += count

        setattr(raw_env, f"_{side}_pose_commanded", False)

    filters = getattr(raw_env, "_arm_pose_filters", None)
    if isinstance(filters, dict):
        filters.update({"right": None, "left": None})
    if reset_controllers:
        controllers = getattr(raw_env, "_ik_controllers", None)
        if isinstance(controllers, Mapping):
            for controller in controllers.values():
                reset = getattr(controller, "reset", None)
                if callable(reset):
                    reset()

    return {
        "robot_joints": held_joints,
        "hand_commands": held_hands,
    }


def stabilize_recording_origin(
    env: Any,
    observation: Any,
    info: Mapping[str, Any] | None,
    *,
    linear_velocity_limit: float = DEFAULT_LINEAR_VELOCITY_LIMIT,
    angular_velocity_limit: float = DEFAULT_ANGULAR_VELOCITY_LIMIT,
    joint_velocity_limit: float | None = None,
    initial_settle_seconds: float = 0.0,
    post_settle_velocity_zeroing: bool = False,
    post_settle_sleeping: bool = False,
    stabilization_velocity_clamp: bool = False,
    stabilization_velocity_clamp_scope: str | None = None,
    stability_metric: str = "velocity",
    pose_translation_limit: float = 0.0005,
    pose_rotation_limit: float = 0.01,
    stable_seconds: float = DEFAULT_STABLE_SECONDS,
    stability_sample_seconds: float = DEFAULT_STABILITY_SAMPLE_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[Any, dict[str, Any]]:
    """Create a physically settled recording boundary before record or replay."""

    output_info = dict(info or {})
    if not stable_recording_origin_enabled():
        return observation, output_info

    raw_env = getattr(env, "unwrapped", env)
    task_name = str(getattr(raw_env, "task_name", "")).strip().lower()
    if type(stabilization_velocity_clamp) is not bool:
        raise ValueError("stabilization_velocity_clamp must be boolean.")
    if stabilization_velocity_clamp_scope is None:
        # Direct legacy callers used the clamp only during the strict gate.
        # New collection profiles pass the all-step scope explicitly.
        stabilization_velocity_clamp_scope = (
            VELOCITY_CLAMP_SCOPE_GATE_ONLY
            if stabilization_velocity_clamp
            else VELOCITY_CLAMP_SCOPE_NONE
        )
    stabilization_velocity_clamp_scope = str(
        stabilization_velocity_clamp_scope
    ).strip().lower()
    if stabilization_velocity_clamp_scope not in _VELOCITY_CLAMP_SCOPES:
        raise ValueError(
            "unsupported stabilization velocity clamp scope: "
            f"{stabilization_velocity_clamp_scope!r}"
        )
    if stabilization_velocity_clamp and (
        stabilization_velocity_clamp_scope == VELOCITY_CLAMP_SCOPE_NONE
    ):
        raise ValueError("Enabled velocity clamp cannot use scope 'none'.")
    if not stabilization_velocity_clamp and (
        stabilization_velocity_clamp_scope != VELOCITY_CLAMP_SCOPE_NONE
    ):
        raise ValueError("Disabled velocity clamp must use scope 'none'.")
    stability_metric = str(stability_metric).strip().lower()
    if stability_metric not in {
        "velocity",
        "pose",
        "velocity_or_static_pose",
    }:
        raise ValueError(
            "unsupported recording-origin stability metric: "
            f"{stability_metric}"
        )
    if (
        stability_metric == "velocity_or_static_pose"
        and task_name != _STATIC_POSE_FALLBACK_TASK
    ):
        raise ValueError(
            "velocity_or_static_pose recording-origin stability is restricted "
            f"to {_STATIC_POSE_FALLBACK_TASK}; got {task_name!r}."
        )
    joint_velocity_limit, joint_velocity_limit_source = (
        _resolve_joint_velocity_limit(raw_env, joint_velocity_limit)
    )
    step_physics = getattr(raw_env, "_step_isaaclab", None)
    apply_targets = getattr(raw_env, "_apply_joint_targets", None)
    if not callable(step_physics) or not callable(apply_targets):
        raise RuntimeError(
            "Stable recording origin requires native Isaac physics stepping."
        )

    physics_dt = float(getattr(raw_env, "_physics_dt", 0.0))
    if not np.isfinite(physics_dt) or physics_dt <= 0.0:
        raise RuntimeError("Stable recording origin requires a positive physics dt.")

    trace = getattr(raw_env, "_physx_ground_truth_trace", None)
    if trace is None:
        trace = PhysxGroundTruthTrace(raw_env)

    scalar_limits = {
        "linear_velocity_limit": linear_velocity_limit,
        "angular_velocity_limit": angular_velocity_limit,
        "joint_velocity_limit": joint_velocity_limit,
        "initial_settle_seconds": initial_settle_seconds,
        "pose_translation_limit": pose_translation_limit,
        "pose_rotation_limit": pose_rotation_limit,
        "stable_seconds": stable_seconds,
        "stability_sample_seconds": stability_sample_seconds,
        "timeout_seconds": timeout_seconds,
    }
    if any(
        not np.isfinite(float(value)) or float(value) < 0.0
        for value in scalar_limits.values()
    ):
        raise ValueError(
            "Stable recording-origin limits and durations must be finite and "
            f"non-negative: {scalar_limits}"
        )
    if stable_seconds <= 0.0:
        raise ValueError("stable_seconds must be positive.")
    if stability_metric == "velocity_or_static_pose":
        approved_pose_limits = _TASK_STATIC_POSE_FALLBACK_LIMITS[
            _STATIC_POSE_FALLBACK_TASK
        ]
        if (
            float(pose_translation_limit) > approved_pose_limits[0]
            or float(pose_rotation_limit) > approved_pose_limits[1]
            or float(linear_velocity_limit) > DEFAULT_LINEAR_VELOCITY_LIMIT
            or float(angular_velocity_limit) > DEFAULT_ANGULAR_VELOCITY_LIMIT
            or float(joint_velocity_limit) > DEFAULT_JOINT_VELOCITY_LIMIT
            or float(stable_seconds) < DEFAULT_STABLE_SECONDS
            or not np.isclose(
                physics_dt,
                1.0 / 120.0,
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                "A06 static-pose fallback exceeds its approved 120 Hz "
                "stability contract."
            )

    stable_steps_required = max(
        1, int(np.ceil(float(stable_seconds) / physics_dt))
    )
    initial_settle_steps = max(
        0, int(np.ceil(float(initial_settle_seconds) / physics_dt))
    )
    # A06's fallback exists only for a non-integrated PhysX velocity readback.
    # Inspect every simulated frame so an actual transform change can never be
    # hidden between the ordinary 50 ms velocity samples.
    stability_sample_steps = (
        1
        if stability_metric == "velocity_or_static_pose"
        else max(
            1,
            int(np.ceil(float(stability_sample_seconds) / physics_dt)),
        )
    )
    stability_sample_interval_seconds = stability_sample_steps * physics_dt
    timeout_steps = max(
        1, int(np.ceil(float(timeout_seconds) / physics_dt))
    )
    if timeout_steps < stable_steps_required:
        raise ValueError(
            "timeout_seconds must allow at least one full stable interval: "
            f"stable_seconds={stable_seconds}, timeout_seconds={timeout_seconds}"
        )

    # The reset call advances one PhysX frame before returning its observation.
    # Never promote that solver-displaced live q (or an old drive target) to the
    # recording origin.  Restore the authored joint home, exact-zero qd, and
    # matching control/drive targets first.  The following canonical write is
    # then an atomic guard for every task and robot velocity before settling.
    robot_hold_targets, robot_hold_target_source = (
        _restore_default_robot_hold_targets(raw_env, sync=True)
    )
    default_joint_state_restores = int(
        robot_hold_target_source == "default_joint_state"
    )
    first_write = _write_canonical_state(trace)
    hold_target_sync = _hold_robot_at_targets(
        raw_env,
        robot_hold_targets,
        reset_controllers=True,
    )
    hold_target_sync["passes"] = 1
    stable_steps_observed = 0
    physics_steps = 0
    last_object_motion: tuple[float, float] | None = None
    last_robot_motion: tuple[float, float, float] | None = None
    last_snapshot: Mapping[str, Any] | None = None
    last_pose_delta: tuple[float, float] | None = None
    velocity_clamp_steps = 0
    velocity_clamp_write = None
    # Let reset contacts dissipate before applying the strict recording gate.
    # These frames are neither returned as observations nor recorded.
    for _ in range(initial_settle_steps):
        _hold_robot_at_targets(raw_env, robot_hold_targets)
        hold_target_sync["passes"] += 1
        apply_targets()
        step_physics()
        physics_steps += 1
        if stabilization_velocity_clamp_scope == (
            VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
        ):
            velocity_clamp_write = trace.zero_task_velocities()
            velocity_clamp_steps += 1
    post_settle_write = None
    post_settle_sleep = None
    put_to_sleep = None
    post_settle_robot_hold_target_source = robot_hold_target_source
    if post_settle_velocity_zeroing:
        # A-series reset objects can keep rolling while contacts find their
        # resting manifold.  The authored robot home can likewise have a tiny
        # contact-constrained equilibrium offset.  Fast-forward first, capture
        # that physically reached q as both state and drive target, then zero
        # velocities.  Capturing after the full settle interval (not after the
        # reset's single PhysX frame) prevents the drive from chasing a stale
        # or unverified joint pose during collection.
        robot_hold_targets = _capture_robot_hold_targets(raw_env)
        post_settle_robot_hold_target_source = (
            "settled_after_default_joint_state"
            if robot_hold_target_source == "default_joint_state"
            else "settled_live_readback"
        )
        _hold_robot_at_targets(
            raw_env,
            robot_hold_targets,
            reset_controllers=True,
        )
        hold_target_sync["passes"] += 1
        post_settle_write = _write_canonical_state(trace)
    if post_settle_sleeping:
        put_to_sleep = getattr(trace, "put_task_bodies_to_sleep", None)
        if not callable(put_to_sleep):
            raise RuntimeError(
                "Stable recording origin requires task-body sleep support."
            )
        post_settle_sleep = put_to_sleep()
    pose_reference = (
        trace.snapshot()
        if stability_metric in {"pose", "velocity_or_static_pose"}
        else None
    )
    steps_since_sample = 0
    stability_samples = 0
    task_velocity_stable = False
    task_pose_stable = False
    robot_velocity_stable = False
    gate_task_velocity_stable = False
    gate_task_static_pose_stable = False
    gate_robot_velocity_stable = False
    gate_static_pose_fallback_samples = 0
    gate_pose_samples = 0
    gate_max_task_translation_delta: float | None = None
    gate_max_task_rotation_delta: float | None = None
    gate_max_fallback_translation_delta: float | None = None
    gate_max_fallback_rotation_delta: float | None = None
    velocity_streak_steps = 0
    pose_streak_steps = 0
    pose_streak_task_velocity_stable = True
    pose_streak_fallback_samples = 0
    pose_streak_samples = 0
    pose_streak_max_translation_delta: float | None = None
    pose_streak_max_rotation_delta: float | None = None
    pose_streak_max_fallback_translation_delta: float | None = None
    pose_streak_max_fallback_rotation_delta: float | None = None
    for timeout_step in range(timeout_steps):
        _hold_robot_at_targets(raw_env, robot_hold_targets)
        hold_target_sync["passes"] += 1
        apply_targets()
        step_physics()
        physics_steps += 1
        steps_since_sample += 1
        if stabilization_velocity_clamp:
            velocity_clamp_write = trace.zero_task_velocities()
            velocity_clamp_steps += 1
        if (
            steps_since_sample < stability_sample_steps
            and timeout_step + 1 < timeout_steps
        ):
            continue
        last_snapshot = trace.snapshot()
        stability_samples += 1
        last_object_motion = _object_motion(last_snapshot)
        last_robot_motion = _robot_motion(last_snapshot)
        if last_object_motion is None:
            raise RuntimeError(
                "Stable recording origin found no task-object state readback."
            )
        if last_robot_motion is None:
            raise RuntimeError(
                "Stable recording origin found no robot velocity readback."
            )
        if stability_metric in {"pose", "velocity_or_static_pose"}:
            assert pose_reference is not None
            last_pose_delta = _task_pose_delta(pose_reference, last_snapshot)
            pose_reference = last_snapshot
            task_pose_stable = bool(
                last_pose_delta is not None
                and last_pose_delta[0] <= pose_translation_limit
                and last_pose_delta[1] <= pose_rotation_limit
            )
        task_velocity_stable = bool(
            last_object_motion[0] <= linear_velocity_limit
            and last_object_motion[1] <= angular_velocity_limit
        )
        robot_velocity_stable = bool(
            last_robot_motion[0] <= linear_velocity_limit
            and last_robot_motion[1] <= angular_velocity_limit
            and last_robot_motion[2] <= joint_velocity_limit
        )
        if stability_metric == "velocity_or_static_pose":
            # Once task velocities are explicitly zeroed after the simulated
            # frame they are no longer independent evidence that the task pose
            # was stationary.  Preserve A06's per-frame pose protection by
            # disabling the velocity-only acceptance branch under the global
            # initialization clamp; robot velocities remain strict.
            velocity_stable_now = bool(
                not stabilization_velocity_clamp
                and robot_velocity_stable
                and task_velocity_stable
            )
            pose_stable_now = bool(
                robot_velocity_stable and task_pose_stable
            )
            velocity_streak_steps = (
                velocity_streak_steps + steps_since_sample
                if velocity_stable_now
                else 0
            )
            if pose_stable_now:
                assert last_pose_delta is not None
                if pose_streak_steps == 0:
                    pose_streak_task_velocity_stable = True
                    pose_streak_fallback_samples = 0
                    pose_streak_samples = 0
                    pose_streak_max_translation_delta = None
                    pose_streak_max_rotation_delta = None
                    pose_streak_max_fallback_translation_delta = None
                    pose_streak_max_fallback_rotation_delta = None
                pose_streak_steps += steps_since_sample
                pose_streak_samples += 1
                pose_streak_task_velocity_stable &= task_velocity_stable
                pose_streak_max_translation_delta = max(
                    pose_streak_max_translation_delta or 0.0,
                    last_pose_delta[0],
                )
                pose_streak_max_rotation_delta = max(
                    pose_streak_max_rotation_delta or 0.0,
                    last_pose_delta[1],
                )
                if stabilization_velocity_clamp or not task_velocity_stable:
                    pose_streak_fallback_samples += 1
                    pose_streak_max_fallback_translation_delta = max(
                        pose_streak_max_fallback_translation_delta or 0.0,
                        last_pose_delta[0],
                    )
                    pose_streak_max_fallback_rotation_delta = max(
                        pose_streak_max_fallback_rotation_delta or 0.0,
                        last_pose_delta[1],
                    )
            else:
                pose_streak_steps = 0

            # The two criteria have independent continuous windows.  Never
            # combine a moving-but-slow frame with a static-but-noisy frame to
            # manufacture one accepted interval.
            if velocity_streak_steps >= stable_steps_required:
                stable_steps_observed = velocity_streak_steps
                gate_task_velocity_stable = True
                gate_robot_velocity_stable = True
                break
            if pose_streak_steps >= stable_steps_required:
                stable_steps_observed = pose_streak_steps
                gate_task_velocity_stable = (
                    False
                    if stabilization_velocity_clamp
                    else pose_streak_task_velocity_stable
                )
                gate_task_static_pose_stable = True
                gate_robot_velocity_stable = True
                gate_static_pose_fallback_samples = (
                    pose_streak_fallback_samples
                )
                gate_pose_samples = pose_streak_samples
                gate_max_task_translation_delta = (
                    pose_streak_max_translation_delta
                )
                gate_max_task_rotation_delta = (
                    pose_streak_max_rotation_delta
                )
                gate_max_fallback_translation_delta = (
                    pose_streak_max_fallback_translation_delta
                )
                gate_max_fallback_rotation_delta = (
                    pose_streak_max_fallback_rotation_delta
                )
                break
            stable_steps_observed = max(
                velocity_streak_steps,
                pose_streak_steps,
            )
        else:
            stable_now = (
                task_pose_stable
                if stability_metric == "pose"
                else bool(task_velocity_stable and robot_velocity_stable)
            )
            if stable_now:
                stable_steps_observed += steps_since_sample
                if stable_steps_observed >= stable_steps_required:
                    gate_task_velocity_stable = task_velocity_stable
                    gate_task_static_pose_stable = task_pose_stable
                    gate_robot_velocity_stable = robot_velocity_stable
                    break
            else:
                stable_steps_observed = 0
        steps_since_sample = 0
    else:
        assert last_object_motion is not None
        assert last_robot_motion is not None
        assert last_snapshot is not None
        task_motion_peaks = _task_motion_peaks(
            last_snapshot,
            trace,
            linear_velocity_limit=linear_velocity_limit,
            angular_velocity_limit=angular_velocity_limit,
        )
        raise RecordingOriginStabilityError(
            "The robot and task objects did not remain physically stable "
            "before the "
            "recording-origin timeout: "
            f"linear={max(last_object_motion[0], last_robot_motion[0]):.6g} m/s "
            f"(limit {linear_velocity_limit:.6g}), "
            "angular="
            f"{max(last_object_motion[1], last_robot_motion[1]):.6g} rad/s "
            f"(limit {angular_velocity_limit:.6g}), "
            f"robot_joint={last_robot_motion[2]:.6g} rad/s "
            f"(limit {joint_velocity_limit:.6g}), "
            f"steps={physics_steps}, timeout={timeout_seconds:.6g}s, "
            f"metric={stability_metric}, pose_delta={last_pose_delta}."
            f" Fastest task bodies: {task_motion_peaks}."
            f" Fastest joints: "
            f"{_robot_joint_velocity_peaks(last_snapshot, trace)}",
            diagnostics={
                "failure_phase": "stability_timeout",
                "physics_steps": int(physics_steps),
                "stability_samples": int(stability_samples),
                "hold_target_sync_passes": int(
                    hold_target_sync["passes"]
                ),
            },
        )

    assert last_object_motion is not None
    assert last_robot_motion is not None
    assert last_snapshot is not None
    gate_snapshot = last_snapshot
    gate_object_motion = last_object_motion
    gate_robot_motion = last_robot_motion

    # The stability gate necessarily ends after a simulated frame, which can
    # leave a small gravity/contact residual even though every threshold was
    # satisfied.  Establish the actual pre-action boundary only after the gate:
    # atomically preserve the accepted poses while setting all public task and
    # joint velocities to exactly zero, put dynamic bodies back to sleep, then
    # read the boundary back without advancing physics.  No state write or
    # physics step is allowed after this point and before action[0].
    # Keep the joint pose that actually passed the physical gate.  Teleporting
    # back to the authored home here would create a new, unverified state that
    # could move again on action[0].  The canonical restore synchronizes all
    # derived body/TCP caches without advancing physics; then align every drive
    # target to that accepted q so the first no-op has no position error to
    # chase.  The canonical all-state write is deliberately last so its
    # synchronized readback also contains those aligned targets.
    final_robot_hold_targets = _capture_robot_hold_targets(raw_env)
    final_hold_target_sync = _hold_robot_at_targets(
        raw_env,
        final_robot_hold_targets,
        reset_controllers=True,
    )
    final_boundary_write = _write_canonical_state(trace, sync=True)
    final_boundary_sleep = None
    if post_settle_sleeping:
        assert callable(put_to_sleep)
        final_boundary_sleep = put_to_sleep()
    final_snapshot = trace.snapshot()
    final_object_motion = _object_motion(final_snapshot)
    final_robot_motion = _robot_motion(final_snapshot)
    if final_object_motion is None:
        raise RuntimeError(
            "Stable recording origin found no final task-object readback."
        )
    if final_robot_motion is None:
        raise RuntimeError(
            "Stable recording origin found no final robot velocity readback."
        )
    final_position = np.asarray(
        final_snapshot.get("physx_robot_joint_position", np.zeros((0, 0))),
        dtype=np.float64,
    )
    final_target = np.asarray(
        final_snapshot.get("physx_robot_joint_target", np.zeros((0, 0))),
        dtype=np.float64,
    )
    final_position_valid = np.asarray(
        final_snapshot.get("physx_robot_joint_valid", np.zeros((0, 0)))
    ).astype(bool)
    final_target_valid = np.asarray(
        final_snapshot.get(
            "physx_robot_joint_target_valid",
            np.zeros((0, 0)),
        )
    ).astype(bool)
    comparable = bool(
        final_position.shape == final_target.shape
        and final_position.shape == final_position_valid.shape
        and final_position.shape == final_target_valid.shape
    )
    valid_targets = (
        final_position_valid & final_target_valid
        if comparable
        else np.zeros((0, 0), dtype=bool)
    )
    if not comparable or not np.any(valid_targets):
        raise RuntimeError(
            "Stable recording origin found no comparable final robot drive "
            "targets."
        )
    final_robot_position_target_error = float(
        np.max(np.abs(final_position[valid_targets] - final_target[valid_targets]))
    )
    if (
        final_object_motion[0] != 0.0
        or final_object_motion[1] != 0.0
        or final_robot_motion[2] != 0.0
        or final_robot_position_target_error != 0.0
    ):
        raise RecordingOriginStabilityError(
            "The final pre-action recording boundary did not read back exact "
            "zero task/joint velocity and an exactly aligned robot target "
            "without a physics step: "
            f"task_linear={final_object_motion[0]:.9g}, "
            f"task_angular={final_object_motion[1]:.9g}, "
            f"robot_joint={final_robot_motion[2]:.9g}, "
            f"robot_target_error={final_robot_position_target_error:.9g}.",
            diagnostics={
                "failure_phase": "final_boundary_readback",
                "physics_steps": int(physics_steps),
                "stability_samples": int(stability_samples),
                "hold_target_sync_passes": int(
                    hold_target_sync["passes"]
                ),
            },
        )
    last_snapshot = final_snapshot
    last_object_motion = final_object_motion
    last_robot_motion = final_robot_motion
    final_max_linear_velocity = max(
        last_object_motion[0],
        last_robot_motion[0],
    )
    final_max_angular_velocity = max(
        last_object_motion[1],
        last_robot_motion[1],
    )
    raw_env.env_step = 0
    for counter_name in ("_success_counter", "_failure_counter"):
        if hasattr(raw_env, counter_name):
            setattr(raw_env, counter_name, 0)

    report = {
        "strategy": "continuous_current_target_all_velocity_settle",
        "stability_metric": stability_metric,
        "physics_steps": physics_steps,
        "physics_dt": physics_dt,
        "initial_settle_seconds": float(initial_settle_seconds),
        "initial_settle_steps": initial_settle_steps,
        "velocity_zeroing_passes": (
            3 if post_settle_write is not None else 2
        ),
        "post_settle_velocity_zeroing": bool(
            post_settle_velocity_zeroing
        ),
        "post_settle_state_write": post_settle_write,
        "post_settle_sleeping": bool(post_settle_sleeping),
        "post_settle_sleep": post_settle_sleep,
        "gate_max_task_linear_velocity": float(gate_object_motion[0]),
        "gate_max_task_angular_velocity": float(gate_object_motion[1]),
        "gate_max_robot_linear_velocity": float(gate_robot_motion[0]),
        "gate_max_robot_angular_velocity": float(gate_robot_motion[1]),
        "gate_max_robot_joint_velocity": float(gate_robot_motion[2]),
        "gate_task_velocity_stable": bool(gate_task_velocity_stable),
        "gate_task_static_pose_stable": bool(
            gate_task_static_pose_stable
        ),
        "gate_robot_velocity_stable": bool(gate_robot_velocity_stable),
        "gate_static_pose_fallback_samples": (
            gate_static_pose_fallback_samples
        ),
        "gate_pose_samples": gate_pose_samples,
        "gate_max_task_translation_delta": gate_max_task_translation_delta,
        "gate_max_task_rotation_delta": gate_max_task_rotation_delta,
        "gate_max_fallback_translation_delta": (
            gate_max_fallback_translation_delta
        ),
        "gate_max_fallback_rotation_delta": (
            gate_max_fallback_rotation_delta
        ),
        "gate_robot_joint_velocity_peaks": _robot_joint_velocity_peaks(
            gate_snapshot,
            trace,
        ),
        "final_boundary_state_write": final_boundary_write,
        "final_boundary_restore_sync": True,
        "robot_hold_target_source": robot_hold_target_source,
        "post_settle_robot_hold_target_source": (
            post_settle_robot_hold_target_source
        ),
        "default_joint_state_restores": default_joint_state_restores,
        "final_default_joint_state_restore": False,
        "final_robot_hold_target_source": "accepted_settled_joint_state",
        "final_hold_target_sync": final_hold_target_sync,
        "final_robot_position_target_error": (
            final_robot_position_target_error
        ),
        "final_boundary_sleep": final_boundary_sleep,
        "final_boundary_physics_steps_after_write": 0,
        "final_boundary_task_and_joint_velocity_exact_zero": True,
        "stabilization_velocity_clamp": bool(
            stabilization_velocity_clamp
        ),
        "stabilization_velocity_clamp_scope": (
            stabilization_velocity_clamp_scope
        ),
        "stabilization_velocity_clamp_steps": velocity_clamp_steps,
        "stabilization_velocity_clamp_write": velocity_clamp_write,
        "linear_velocity_limit": float(linear_velocity_limit),
        "angular_velocity_limit": float(angular_velocity_limit),
        "joint_velocity_limit": float(joint_velocity_limit),
        "joint_velocity_limit_source": joint_velocity_limit_source,
        "task_name": str(getattr(raw_env, "task_name", "")),
        "stable_seconds_required": float(stable_seconds),
        "stable_steps_required": stable_steps_required,
        "stable_steps_observed": stable_steps_observed,
        "stability_sample_seconds": float(stability_sample_seconds),
        "stability_sample_steps": stability_sample_steps,
        "stability_sample_interval_seconds": (
            stability_sample_interval_seconds
        ),
        "full_frame_pose_gate": bool(
            stability_metric == "velocity_or_static_pose"
        ),
        "stability_samples": stability_samples,
        "timeout_seconds": float(timeout_seconds),
        "pose_translation_limit": float(pose_translation_limit),
        "pose_rotation_limit": float(pose_rotation_limit),
        "final_max_task_translation_delta": (
            None if last_pose_delta is None else float(last_pose_delta[0])
        ),
        "final_max_task_rotation_delta": (
            None if last_pose_delta is None else float(last_pose_delta[1])
        ),
        "final_max_linear_velocity": float(final_max_linear_velocity),
        "final_max_angular_velocity": float(final_max_angular_velocity),
        "final_max_robot_joint_velocity": float(last_robot_motion[2]),
        "final_max_task_linear_velocity": float(last_object_motion[0]),
        "final_max_task_angular_velocity": float(last_object_motion[1]),
        "final_robot_joint_velocity_peaks": _robot_joint_velocity_peaks(
            last_snapshot,
            trace,
        ),
        "first_state_write": first_write,
        "hold_target_sync": hold_target_sync,
        "state_writes": True,
        "initialization_per_physics_step_velocity_writes": bool(
            stabilization_velocity_clamp
        ),
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }
    raw_env._physical_recording_origin = dict(report)
    output_info["physical_recording_origin"] = dict(report)
    return _rebuild_observation(env, trace), output_info


__all__ = [
    "JOINT_VELOCITY_LIMIT_ENV",
    "RecordingOriginStabilityError",
    "STABLE_ORIGIN_ENV",
    "STABLE_ORIGIN_RESET_ATTEMPTS_ENV",
    "VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS",
    "VELOCITY_CLAMP_SCOPE_GATE_ONLY",
    "VELOCITY_CLAMP_SCOPE_NONE",
    "stable_recording_origin_kwargs_for_task",
    "stable_recording_origin_kwargs_from_report",
    "stable_recording_origin_reset_attempts_for_task",
    "stable_recording_origin_enabled",
    "stabilize_recording_origin",
]
