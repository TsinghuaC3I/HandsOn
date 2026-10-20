from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

import glfw
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from handson.tasks.tracker_episode_protocol import (
    TrackerAutoBaselineClient,
    release_tracker_episode_guard,
    request_tracker_episode_baseline,
)

INTERVENTION_START_DELAY_SECONDS = 2.0
RESET_REBASE_DELAY_SECONDS = 0.5
WORLD_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
HAND_POLICY_DIM = 16
WUJI_NATIVE_HAND_DIM = 20
TRACKER_HOLD_POS_EPS_M = 0.006
TRACKER_HOLD_ROT_EPS_DEG = 1.0
TRACKER_BASELINE_WINDOW_SECONDS = 0.30
TRACKER_BASELINE_POS_SPREAD_M = 0.008
TRACKER_BASELINE_ROT_SPREAD_DEG = 2.0
TRACKER_BASELINE_MAX_WAIT_SECONDS = 1.0
TRACKER_BASELINE_MIN_SAMPLES = 6
TRACKER_BASELINE_MAX_SAMPLE_GAP_SECONDS = 0.10
TRACKER_STRICT_PACKET_MAX_AGE_SECONDS = 0.25
TRACKER_REBASE_GAP_SECONDS = 0.15
TRACKER_RECONNECT_MAX_POSITION_DELTA_M = 0.12
TRACKER_RECONNECT_MAX_ROTATION_DELTA_DEG = 35.0
# Match the Mode A bridge's per-sample discontinuity limits.  Ordinary tracker
# motion is smoothed below these values; a larger accepted step means the
# bridge reacquired a tracker or adopted a relocated SteamVR coordinate frame.
TRACKER_JUMP_TRANSLATION_M = 0.12
TRACKER_JUMP_ROTATION_DEG = 35.0
TELEOP_FILTER_ALPHA = 0.55
TELEOP_FILTER_DEADBAND_M = 0.001
TELEOP_FILTER_ROT_DEADBAND_DEG = 0.5
TELEOP_FILTER_MAX_STEP_M = 0.06
TELEOP_FILTER_MAX_ROT_STEP_DEG = 10.0
TELEOP_MAX_TRACKER_DELTA_M = 0.34
B02_NUT_BOLT_TASK_NAME = "b02_nut_bolt_assembly"
B02_WUJI_CLOSURE_START_RAD = 0.35
B02_WUJI_CLOSURE_FULL_RAD = 0.90
B02_WUJI_CLOSED_SPREAD_TARGET = np.array(
    [-0.06, 0.00, 0.10, 0.16],
    dtype=np.float64,
)
B02_WUJI_CLOSED_FLEXION_BONUS_RAD = 0.12


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_attr(env, name: str):
    current = env
    for _ in range(8):
        if current is None:
            break
        value = getattr(current, name, None)
        if value is not None:
            return value
        current = getattr(current, "env", None)
    return None


def _consume_teleop_action_extras(env) -> np.ndarray:
    """Collect one-shot task controls that must survive action replay."""
    callback = _env_attr(env, "consume_teleop_action_extras")
    if not callable(callback):
        return np.empty(0, dtype=np.float64)
    values = np.asarray(callback(), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Teleoperation action extras must be finite.")
    return values


def _condition_live_teleop_action(
    env,
    action: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply an optional task-owned conditioner before physics and recording."""

    callback = _env_attr(env, "condition_live_teleop_action")
    if not callable(callback):
        return action
    conditioned = callback(action)
    if not isinstance(conditioned, dict):
        conditioned = dict(conditioned)
    for side in ("right", "left"):
        if side not in conditioned:
            raise ValueError(f"Conditioned teleoperation action is missing {side!r}.")
        values = np.asarray(conditioned[side], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Conditioned teleoperation action for {side!r} must be finite."
            )
        conditioned[side] = values
    return conditioned


def _consume_manual_tracker_rebase_request(env) -> bool:
    """Consume the Isaac window's one-shot Z-key tracker rebase request."""

    callback = _env_attr(env, "consume_manual_tracker_rebase_request")
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception as exc:
        print(
            f"[Teleop][WARN] Could not consume tracker rebase request: {exc}",
            flush=True,
        )
        return False


def _consume_guidance_route_complete_request(env) -> bool:
    """Consume the one-shot Space request that seals a hands-only guide."""

    callback = _env_attr(env, "consume_guidance_route_complete_request")
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception as exc:
        print(
            f"[Teleop][WARN] Could not consume guidance route completion: {exc}",
            flush=True,
        )
        return False


def _is_rokae_wuji_teleop(env) -> bool:
    arm_type = str(_env_attr(env, "arm_type") or os.environ.get("HANDSON_ARM_TYPE", "panda"))
    hand_type = str(_env_attr(env, "hand_type") or "")
    return arm_type.lower() == "rokae" or hand_type.lower() in {"wuji1", "wuji2"}


def _is_wuji_hand_env(env) -> bool:
    hand_type = str(_env_attr(env, "hand_type") or "")
    return hand_type.lower() in {"wuji1", "wuji2"}


def _is_b02_nut_bolt_task(env) -> bool:
    task_name = str(_env_attr(env, "task_name") or "")
    return task_name.lower() == B02_NUT_BOLT_TASK_NAME


def _supports_direct_hand_targets(env) -> bool:
    return bool(_env_attr(env, "_hand_target_control") or _env_attr(env, "_hands_only"))


def _rokae_wuji_controller(env):
    return _env_attr(env, "_rokae_wuji") or _env_attr(env, "_rokae_wuji_control")


def _single_teleop_start_pose(env) -> np.ndarray:
    controller = _rokae_wuji_controller(env)
    if (
        controller is not None
        and bool(getattr(controller, "hand_target_control", False))
        and hasattr(controller, "body_pose_matrix")
    ):
        return np.asarray(controller.body_pose_matrix(), dtype=np.float64)
    return np.asarray(env.get_end_effector_pose_matrix(), dtype=np.float64)


def _dual_teleop_start_poses(env) -> tuple[np.ndarray, np.ndarray]:
    controller = _rokae_wuji_controller(env)
    if (
        controller is not None
        and bool(getattr(controller, "hand_target_control", False))
        and hasattr(controller, "body_pose_matrix")
    ):
        return (
            np.asarray(controller.body_pose_matrix("right_palm_link"), dtype=np.float64),
            np.asarray(controller.body_pose_matrix("left_palm_link"), dtype=np.float64),
        )
    right_pose, left_pose = env.get_end_effector_pose_matrix()
    return (
        np.asarray(right_pose, dtype=np.float64),
        np.asarray(left_pose, dtype=np.float64),
    )


def _teleop_target_frame(env=None) -> str:
    frame = os.environ.get("HANDSON_TELEOP_TARGET_FRAME", "").strip().lower()
    frame = frame.replace("-", "_")
    if not frame:
        if _env_flag("HANDSON_TELEOP_SIM_ALIGNED"):
            if _is_rokae_wuji_teleop(env) and not _env_flag(
                "HANDSON_TELEOP_ALLOW_WUJI_SIM_FRAME"
            ):
                return "tracker"
            return "sim"
        return "tracker"
    if frame in {"sim", "simulation", "relative", "sim_relative"}:
        if _is_rokae_wuji_teleop(env) and not _env_flag(
            "HANDSON_TELEOP_ALLOW_WUJI_SIM_FRAME"
        ):
            return "tracker"
        return "sim"
    if frame in {"tracker", "absolute", "world", "tracker_absolute"}:
        return "tracker"
    raise ValueError(
        "HANDSON_TELEOP_TARGET_FRAME must be 'sim'/'relative' or "
        f"'tracker'/'absolute', got {frame!r}."
    )


@dataclass(frozen=True)
class SingleArmTeleopConfig:
    vive_udp_host: str = "127.0.0.1"
    vive_udp_port: int = 5012
    hand_udp_host: str = "127.0.0.1"
    hand_udp_port: int = 5014
    pose_scale: float = 1.0
    filter_alpha: float | None = None
    filter_deadband_m: float | None = None
    filter_rot_deadband_deg: float | None = None
    filter_max_step_m: float | None = None
    filter_max_rot_step_deg: float | None = None
    max_tracker_delta_m: float | None = None
    command_on_tracker_motion_only: bool = False
    tracker_motion_deadband_m: float = 0.0
    tracker_motion_rot_deadband_deg: float = 0.0
    rebase_on_tracker_jump: bool = True
    tracker_jump_translation_m: float = TRACKER_JUMP_TRANSLATION_M
    tracker_jump_rotation_deg: float = TRACKER_JUMP_ROTATION_DEG
    tracker_packet_max_age_s: float | None = None
    preserve_tracker_baseline_on_recovery: bool = True
    auto_rebase_on_tracker_recovery: bool = True
    rebase_on_stationary_drift: bool = False
    tracker_stationary_max_translation_m: float = 0.0
    tracker_stationary_max_rotation_deg: float = 0.0
    tracker_stationary_rebase_after_sec: float = 0.0


@dataclass(frozen=True)
class BimanualTeleopConfig:
    reset_rebase_delay_seconds: float = RESET_REBASE_DELAY_SECONDS
    vive_udp_host: str = "127.0.0.1"
    vive_udp_port: int = 5012
    right_hand_udp_host: str = "127.0.0.1"
    right_hand_udp_port: int = 5014
    left_hand_udp_host: str = "127.0.0.1"
    left_hand_udp_port: int = 5016
    pose_scale: float = 1.0
    filter_alpha: float | None = None
    filter_deadband_m: float | None = None
    filter_rot_deadband_deg: float | None = None
    filter_max_step_m: float | None = None
    filter_max_rot_step_deg: float | None = None
    # Live-only Cartesian velocity feed-forward.  This never changes the
    # outward action shape: recorders still receive absolute TCP pose targets
    # and replay keeps using the recorded pose/DLS path.
    # Enabled for the common dual-arm teleop path.  A task with unusually
    # sensitive contact dynamics can opt out explicitly with ``False``.
    velocity_assist: bool = True
    velocity_assist_gain: float = 0.35
    velocity_assist_lookahead_s: float = 0.04
    velocity_assist_max_linear_m_s: float = 0.25
    velocity_assist_max_angular_rad_s: float = float(np.deg2rad(90.0))
    velocity_assist_stop_speed_m_s: float = 0.008
    hand_filter_max_step_rad: float | None = None
    # Optional one-shot hand takeover ramp.  Unlike
    # ``hand_filter_max_step_rad`` this is active only for the first bounded
    # number of control frames after teleoperation is enabled.  It prevents a
    # live glove pose from becoming an impulse at action[0], then returns to
    # exact per-frame hand targets with no steady-state filtering.
    hand_takeover_max_step_rad: float | None = None
    hand_takeover_max_control_steps: int = 0
    hand_packet_max_age_s: float | None = None
    max_tracker_delta_m: float | None = None
    max_tracker_rotation_delta_deg: float | None = None
    tracker_packet_max_age_s: float | None = None
    rebase_on_tracker_jump: bool = True
    tracker_jump_translation_m: float = TRACKER_JUMP_TRANSLATION_M
    tracker_jump_rotation_deg: float = TRACKER_JUMP_ROTATION_DEG
    # Rejecting input must also cancel the controller's previously latched
    # target. All collection tasks hold measured palms on loss/discontinuity.
    fail_closed_rebase_on_tracker_jump: bool = True
    # The baseline-preserving recovery remains available when automatic
    # rebasing is disabled. By default, a fresh live pose immediately maps to
    # the measured palms. Only missing input requires a bridge request.
    preserve_tracker_baseline_on_recovery: bool = True
    auto_rebase_on_tracker_recovery: bool = True
    # A continuous, intentional workspace excursion is different from a
    # packet discontinuity.  Tasks may opt into fail-closed handling for the
    # former, but the default saturates the Cartesian target at
    # ``max_tracker_*`` while preserving the episode baseline.  Reusing the
    # jump flag here made long reaches repeatedly hold/rebase and discard the
    # operator's accumulated arm motion.
    fail_closed_on_tracker_excursion: bool = False
    # When enabled, teleoperation is armed only from an atomic tracker pair
    # that has remained stable for the baseline window.  This prevents one
    # fresh wrist and one stale wrist from becoming an episode baseline.
    require_stable_pair_on_enable: bool = False
    # Some runtimes spend a long time proving a stable physical origin after
    # ``reset()``.  Refreshing the bridge-owned episode baseline immediately
    # before intervention keeps glove repositioning during that initialization
    # window from being mistaken for a tracker relocation.
    refresh_tracker_baseline_on_enable: bool = False


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(matrix).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _unit_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return WORLD_Z_AXIS
    return axis / norm


def _rotation_component_deg(rot_matrix: np.ndarray, axis_world: np.ndarray) -> float:
    axis = _unit_axis(axis_world)
    rotvec = R.from_matrix(rot_matrix).as_rotvec()
    return float(np.rad2deg(np.dot(rotvec, axis)))


def _target_pose_from_tracker_pose(
    ee_start: np.ndarray,
    tracker_start: np.ndarray,
    tracker_now: np.ndarray,
    pose_scale: float,
    target_frame: str = "tracker",
) -> np.ndarray:
    """Apply the tracker-relative delta to the EE pose captured at enable."""
    try:
        tracker_delta = np.linalg.inv(tracker_start) @ tracker_now
    except np.linalg.LinAlgError:
        return ee_start.copy()

    if target_frame == "sim":
        tracker_to_ee_basis = ee_start[:3, :3].T @ tracker_start[:3, :3]
        tracker_delta[:3, :3] = (
            tracker_to_ee_basis @ tracker_delta[:3, :3] @ tracker_to_ee_basis.T
        )
        tracker_delta[:3, 3] = tracker_to_ee_basis @ tracker_delta[:3, 3]

    tracker_delta[:3, 3] *= pose_scale
    target = ee_start @ tracker_delta
    return target


def _target_pose_from_bridge_world_pose(
    ee_start: np.ndarray,
    tracker_start: np.ndarray,
    tracker_now: np.ndarray,
    pose_scale: float,
    max_tracker_delta_m: float | None = None,
    max_tracker_rotation_delta_deg: float | None = None,
) -> np.ndarray:
    try:
        tracker_to_ee = np.linalg.inv(tracker_start) @ ee_start
    except np.linalg.LinAlgError:
        return ee_start.copy()
    # Tracker orientation and translation are independent measurements.  Applying
    # the full transform would turn a wrist rotation into a spurious position arc.
    target = ee_start.copy()
    target[:3, :3] = tracker_now[:3, :3] @ tracker_to_ee[:3, :3]
    if max_tracker_rotation_delta_deg is not None:
        max_rotation = max(0.0, float(max_tracker_rotation_delta_deg))
        start_rotation = R.from_matrix(ee_start[:3, :3])
        target_rotation = R.from_matrix(target[:3, :3])
        rotation_delta = start_rotation.inv() * target_rotation
        rotation_angle = float(rotation_delta.magnitude())
        max_rotation_rad = float(np.deg2rad(max_rotation))
        if max_rotation_rad > 0.0 and rotation_angle > max_rotation_rad:
            target[:3, :3] = (
                start_rotation
                * R.from_rotvec(
                    rotation_delta.as_rotvec()
                    * (max_rotation_rad / rotation_angle)
                )
            ).as_matrix()
    translation = pose_scale * (tracker_now[:3, 3] - tracker_start[:3, 3])
    translation_norm = float(np.linalg.norm(translation))
    if max_tracker_delta_m is None:
        max_translation = max(
            0.0,
            _env_float(
                "HANDSON_TELEOP_MAX_TRACKER_DELTA_M",
                TELEOP_MAX_TRACKER_DELTA_M,
            ),
        )
    else:
        max_translation = max(0.0, float(max_tracker_delta_m))
    if max_translation > 0.0 and translation_norm > max_translation:
        translation *= max_translation / translation_norm
    target[:3, 3] = ee_start[:3, 3] + translation
    return target


def _rotation_angle_deg(rot_matrix: np.ndarray) -> float:
    return float(np.rad2deg(R.from_matrix(rot_matrix).magnitude()))


def _tracker_timeout_seconds(config) -> float:
    try:
        timeout = float(config.tracker_packet_max_age_s)
    except (AttributeError, TypeError, ValueError):
        return TRACKER_STRICT_PACKET_MAX_AGE_SECONDS
    return timeout if np.isfinite(timeout) and timeout > 0.0 else TRACKER_STRICT_PACKET_MAX_AGE_SECONDS


def _valid_tracker_transform(transform: np.ndarray) -> bool:
    return bool(
        transform.shape == (4, 4)
        and np.all(np.isfinite(transform))
        and np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)
        and np.linalg.det(transform[:3, :3]) > 0.0
    )


def _tracker_history_is_fresh(history, config) -> bool:
    if len(history) < TRACKER_BASELINE_MIN_SAMPLES:
        return False
    times = np.asarray([sample[0] for sample in history], dtype=np.float64)
    gaps = np.diff(times)
    return bool(
        np.all(np.isfinite(times))
        and times[-1] - times[0] >= TRACKER_BASELINE_WINDOW_SECONDS * 0.9
        and np.all(gaps > 0.0)
        and np.all(gaps <= TRACKER_BASELINE_MAX_SAMPLE_GAP_SECONDS)
        and 0.0 <= time.monotonic() - times[-1] <= _tracker_timeout_seconds(config)
    )


def _queue_tracker_hold(wrapper, now: float) -> None:
    """Called under the tracker lock; physical readback stays on action thread."""
    if not getattr(wrapper, "_tracker_rebase_hold_pending", False):
        wrapper._tracker_history = deque()
        wrapper._tracker_fault_hold_pose = None
        wrapper._tracker_baseline_wait_started_at = now
        recovery = (
            "automatically requesting a fresh baseline at the current palms."
            if getattr(wrapper.config, "auto_rebase_on_tracker_recovery", False)
            else "waiting for fresh tracking near the last trusted pose."
        )
        print(f"[Teleop][WARN] Tracker input rejected or unavailable; holding measured palms; {recovery}", flush=True)
    wrapper._tracker_rebase_requested = False
    wrapper._tracker_rebase_hold_pending = True


def _cancel_auto_tracker_rebase(wrapper) -> None:
    client = getattr(wrapper, "_auto_tracker_baseline_client", None)
    if client is not None:
        client.close()
        wrapper._auto_tracker_baseline_client = None


def _auto_tracker_rebase_action(wrapper, *, dual: bool):
    with wrapper._tracker_lock:
        return _auto_tracker_rebase_action_locked(wrapper, dual=dual)


def _auto_tracker_rebase_action_locked(wrapper, *, dual: bool):
    """Rebase in this control frame using a fresh pose, with no dwell timer."""
    if not (wrapper.intervened
            and getattr(wrapper, "_tracker_rebase_hold_pending", False)
            and getattr(wrapper.config, "auto_rebase_on_tracker_recovery", False)):
        _cancel_auto_tracker_rebase(wrapper)
        return None
    hold_action = wrapper._physical_palm_hold_action()
    poses = None
    history = getattr(wrapper, "_tracker_history", [])
    if history:
        sample = history[-1]
        age = time.monotonic() - float(sample[0])
        if (len(sample) == (3 if dual else 2)
                and 0.0 <= age <= _tracker_timeout_seconds(wrapper.config)
                and all(_valid_tracker_transform(p) for p in sample[1:])):
            # This pose has already arrived through the live bridge. A jump
            # changes the reference, not the palm target: adopt it immediately
            # at measured physics instead of waiting for another packet/ACK.
            poses = [p.copy() for p in sample[1:]]
    client = getattr(wrapper, "_auto_tracker_baseline_client", None)
    if getattr(wrapper, "_rokae_wuji_teleop", False) and (poses is None or client is not None):
        try:
            if client is None:
                client = TrackerAutoBaselineClient()
                wrapper._auto_tracker_baseline_client = client
                print("[Teleop] Automatic tracker rebase requested; waiting only for a valid live pose.", flush=True)
            packet = client.poll()
        except OSError:
            return hold_action
        count = 24 if dual else 12
        if packet is not None and len(packet) in ({24 * 8} if dual else {12 * 8, 24 * 8}):
            values = np.frombuffer(packet, dtype=np.float64, count=count).reshape(-1, 3, 4)
            poses = []
            for value in values:
                candidate = np.eye(4)
                candidate[:3, :] = value
                poses.append(candidate)
    if poses is None or not all(_valid_tracker_transform(p) for p in poses):
        return hold_action

    # Read physical palms at adoption, not the servo goal from before the
    # dropout. The returned action is this same pose for the entire frame.
    if dual:
        physical = dict(zip(("right", "left"), _dual_teleop_start_poses(wrapper.env)))
        baseline = dict(zip(("right", "left"), poses))
        wrapper.latest_tracker_right = poses[0].copy()
        wrapper.latest_tracker_left = poses[1].copy()
        wrapper.tracker_start_world = {k: p.copy() for k, p in baseline.items()}
        wrapper.ee_start = {k: p.copy() for k, p in physical.items()}
        wrapper.hold_pose = {k: p.copy() for k, p in physical.items()}
        wrapper._tracker_fault_hold_pose = {k: p.copy() for k, p in physical.items()}
    else:
        physical = _single_teleop_start_pose(wrapper.env).copy()
        wrapper.latest_tracker = poses[0].copy()
        wrapper.tracker_start_world = poses[0].copy()
        wrapper.ee_start = physical.copy()
        wrapper.hold_pose = physical.copy()
        wrapper._tracker_fault_hold_pose = physical.copy()
        wrapper._last_commanded_tracker_pose = poses[0].copy()
        wrapper._tracker_stationary_since = None
        wrapper._tracker_stationary_drift_eligible = True
    now = time.monotonic()
    wrapper._last_tracker_packet_time = now
    wrapper._tracker_history.clear()
    wrapper._tracker_history.append((now, *(p.copy() for p in poses)))
    wrapper._tracker_baseline_wait_started_at = None
    wrapper._tracker_rebase_requested = False
    wrapper._tracker_rebase_hold_pending = False
    hold_action = wrapper._physical_palm_hold_action()
    wrapper._tracker_fault_hold_pose = None
    _cancel_auto_tracker_rebase(wrapper)
    print("[Teleop] Automatic tracker rebase complete at the current palms; no Z required.", flush=True)
    return hold_action


def _tracker_pose_jump_exceeds(
    previous: np.ndarray,
    current: np.ndarray,
    config: SingleArmTeleopConfig | BimanualTeleopConfig,
) -> bool:
    """Detect a SteamVR relocalization discontinuity, not ordinary motion."""

    if not config.rebase_on_tracker_jump:
        return False
    translation_limit = max(0.0, float(config.tracker_jump_translation_m))
    rotation_limit = max(0.0, float(config.tracker_jump_rotation_deg))
    translation = float(
        np.linalg.norm(current[:3, 3] - previous[:3, 3])
    )
    rotation = _rotation_angle_deg(
        current[:3, :3] @ previous[:3, :3].T
    )
    return bool(
        (translation_limit > 0.0 and translation > translation_limit)
        or (rotation_limit > 0.0 and rotation > rotation_limit)
    )


def _config_float(config, attr: str, env_name: str, default: float) -> float:
    value = getattr(config, attr, None) if config is not None else None
    if value is None:
        return _env_float(env_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return _env_float(env_name, default)


def _target_filter_alpha(config=None) -> float:
    return float(
        np.clip(
            _config_float(
                config,
                "filter_alpha",
                "HANDSON_TELEOP_FILTER_ALPHA",
                TELEOP_FILTER_ALPHA,
            ),
            0.0,
            1.0,
        )
    )


def _target_filter_deadband_pos(config=None) -> float:
    return max(
        0.0,
        _config_float(
            config,
            "filter_deadband_m",
            "HANDSON_TELEOP_FILTER_DEADBAND_M",
            TELEOP_FILTER_DEADBAND_M,
        ),
    )


def _target_filter_deadband_rot_rad(config=None) -> float:
    return max(
        0.0,
        np.deg2rad(
            _config_float(
                config,
                "filter_rot_deadband_deg",
                "HANDSON_TELEOP_FILTER_ROT_DEADBAND_DEG",
                TELEOP_FILTER_ROT_DEADBAND_DEG,
            )
        ),
    )


def _target_filter_max_step(config=None) -> float:
    return max(
        0.0,
        _config_float(
            config,
            "filter_max_step_m",
            "HANDSON_TELEOP_FILTER_MAX_STEP_M",
            TELEOP_FILTER_MAX_STEP_M,
        ),
    )


def _target_filter_max_rot_step_rad(config=None) -> float:
    return max(
        0.0,
        np.deg2rad(
            _config_float(
                config,
                "filter_max_rot_step_deg",
                "HANDSON_TELEOP_FILTER_MAX_ROT_STEP_DEG",
                TELEOP_FILTER_MAX_ROT_STEP_DEG,
            )
        ),
    )


def _pose_with_limited_delta(previous: np.ndarray, target: np.ndarray, config=None) -> np.ndarray:
    limited = np.asarray(target, dtype=np.float64).copy()
    previous = np.asarray(previous, dtype=np.float64)

    pos_delta = limited[:3, 3] - previous[:3, 3]
    pos_dist = float(np.linalg.norm(pos_delta))
    max_step = _target_filter_max_step(config)
    if max_step > 0.0 and pos_dist > max_step:
        limited[:3, 3] = previous[:3, 3] + pos_delta * (max_step / pos_dist)

    prev_rot = R.from_matrix(previous[:3, :3])
    target_rot = R.from_matrix(target[:3, :3])
    rot_delta = prev_rot.inv() * target_rot
    rot_angle = float(rot_delta.magnitude())
    max_rot_step = _target_filter_max_rot_step_rad(config)
    if max_rot_step > 0.0 and rot_angle > max_rot_step:
        limited[:3, :3] = (
            prev_rot
            * R.from_rotvec(rot_delta.as_rotvec() * (max_rot_step / rot_angle))
        ).as_matrix()
    return limited


def _blend_pose(previous: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    previous = np.asarray(previous, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    blended = target.copy()
    blended[:3, 3] = previous[:3, 3] + alpha * (target[:3, 3] - previous[:3, 3])

    prev_rot = R.from_matrix(previous[:3, :3])
    target_rot = R.from_matrix(target[:3, :3])
    rot_delta = prev_rot.inv() * target_rot
    blended[:3, :3] = (prev_rot * R.from_rotvec(alpha * rot_delta.as_rotvec())).as_matrix()
    return blended


def _filter_target_pose(previous: np.ndarray | None, target: np.ndarray, config=None) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    if previous is None:
        return target.copy()

    previous = np.asarray(previous, dtype=np.float64)
    pos_delta = target[:3, 3] - previous[:3, 3]
    pos_dist = float(np.linalg.norm(pos_delta))
    rot_delta = R.from_matrix(previous[:3, :3]).inv() * R.from_matrix(target[:3, :3])
    rot_angle = float(rot_delta.magnitude())
    if (
        pos_dist <= _target_filter_deadband_pos(config)
        and rot_angle <= _target_filter_deadband_rot_rad(config)
    ):
        return previous.copy()

    limited = _pose_with_limited_delta(previous, target, config)
    return _blend_pose(previous, limited, _target_filter_alpha(config))


def _pose_velocity(
    previous: np.ndarray | None,
    current: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return a finite-difference world-frame twist ``[v, omega]``.

    The helper is intentionally independent of IsaacLab.  The teleop wrapper
    uses it to publish a transient live hint while the dataset continues to
    store the pose target produced by ``_filter_target_pose``.
    """

    if previous is None or not np.isfinite(float(dt)) or float(dt) <= 0.0:
        return np.zeros(6, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return np.zeros(6, dtype=np.float64)
    dt = float(dt)
    linear = (current[:3, 3] - previous[:3, 3]) / dt
    try:
        rotation_delta = R.from_matrix(previous[:3, :3]).inv() * R.from_matrix(
            current[:3, :3]
        )
        angular = rotation_delta.as_rotvec() / dt
    except ValueError:
        return np.zeros(6, dtype=np.float64)
    velocity = np.concatenate([linear, angular]).astype(np.float64, copy=False)
    return velocity if np.all(np.isfinite(velocity)) else np.zeros(6, dtype=np.float64)


def _clip_vector_norm(vector: np.ndarray, maximum: float | None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        return np.zeros_like(vector)
    try:
        maximum = float(maximum)
    except (TypeError, ValueError):
        return vector.copy()
    if not np.isfinite(maximum) or maximum <= 0.0:
        return vector.copy()
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1e-12:
        return vector.copy()
    return vector * (maximum / norm)


def _create_udp_socket(ip, port, timeout=0.2, reuseaddr=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if reuseaddr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.settimeout(timeout)
    return sock


def _recv_latest_udp_datagram(sock, max_size: int) -> tuple[bytes, object]:
    """Receive one datagram, then discard any backlog in favor of the newest."""
    data, address = sock.recvfrom(max_size)
    timeout = sock.gettimeout()
    sock.setblocking(False)
    try:
        while True:
            try:
                data, address = sock.recvfrom(max_size)
            except (BlockingIOError, InterruptedError, socket.timeout):
                return data, address
    finally:
        sock.settimeout(timeout)


def _parse_udp_hand_angles(
    data: bytes,
    *,
    expected_dim: int | None = None,
) -> np.ndarray | None:
    try:
        hand_angles = np.frombuffer(data, dtype=np.float64)
    except ValueError:
        return None
    if expected_dim is not None:
        if hand_angles.size != int(expected_dim) or not np.all(
            np.isfinite(hand_angles)
        ):
            return None
        return hand_angles.astype(np.float64, copy=True)
    if hand_angles.size >= WUJI_NATIVE_HAND_DIM:
        return hand_angles[:WUJI_NATIVE_HAND_DIM].astype(np.float64, copy=True)

    parsed = np.zeros(HAND_POLICY_DIM, dtype=np.float64)
    parsed[: min(parsed.size, hand_angles.size)] = hand_angles[: parsed.size]
    return parsed


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _condition_b02_wuji_hand_action(env, hand_action: np.ndarray) -> np.ndarray:
    hand = np.asarray(hand_action, dtype=np.float64)
    if (
        hand.size != WUJI_NATIVE_HAND_DIM
        or not _is_wuji_hand_env(env)
        or not _is_b02_nut_bolt_task(env)
    ):
        return hand.copy()

    qpos = hand.reshape(5, 4).copy()
    finger_flexion = np.maximum(qpos[1:, (0, 2, 3)], 0.0).mean(axis=1)
    closure = (finger_flexion - B02_WUJI_CLOSURE_START_RAD) / (
        B02_WUJI_CLOSURE_FULL_RAD - B02_WUJI_CLOSURE_START_RAD
    )
    blend = _smoothstep01(closure)

    spread = qpos[1:, 1]
    qpos[1:, 1] = spread + (B02_WUJI_CLOSED_SPREAD_TARGET - spread) * blend

    flexion_columns = (0, 2, 3)
    qpos[1:, flexion_columns] += B02_WUJI_CLOSED_FLEXION_BONUS_RAD * blend[:, None]
    return qpos.reshape(WUJI_NATIVE_HAND_DIM)


def _get_mujoco_window(env):
    # Gymnasium 1.x wrappers do not forward arbitrary attributes from their
    # wrapped environments.  Plan-mode tasks add an observation wrapper below
    # the teleop wrapper, so inspect the complete wrapper chain rather than
    # looking only at the immediate child.
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        for viewer_attr in ("_mj_viewer", "_viewer"):
            viewer = getattr(current, viewer_attr, None)
            if viewer is None:
                continue
            window = getattr(viewer, "window", None)
            if window is not None:
                return window
            nested_window = getattr(
                getattr(viewer, "viewer", None),
                "window",
                None,
            )
            if nested_window is not None:
                return nested_window
        current = getattr(current, "env", None)
    return None


def _maybe_set_key_callback(env, callback, current_window=None):
    window = _get_mujoco_window(env)
    if window is None:
        return current_window, False
    if window is current_window:
        return current_window, True
    try:
        if not glfw.init():
            return current_window, False
        glfw.set_key_callback(window, callback)
    except Exception:
        return current_window, False
    return window, True


class SingleArmViveHandTeleopWrapper(gym.ActionWrapper):
    def __init__(self, env, config: SingleArmTeleopConfig):
        super().__init__(env)
        self.config = config
        self.intervened = False
        self.reset_trigger = False
        self._guidance_route_complete_requested = False
        self.tracker_start_world = None
        self.ee_start = None
        self.hold_pose = None
        self._filtered_target_pose = None
        self._last_commanded_tracker_pose: np.ndarray | None = None
        self._tracker_fault_hold_pose: np.ndarray | None = None
        self.latest_tracker: np.ndarray | None = None
        self._single_wuji_native_hand = _is_wuji_hand_env(self.env)
        self._hand_action_dim = (
            WUJI_NATIVE_HAND_DIM if self._single_wuji_native_hand else HAND_POLICY_DIM
        )
        # Start from the environment's authored/reset hand target.  A missing
        # glove stream must hold that pose instead of commanding an all-zero
        # hand and letting the fingers visibly fall at startup.
        self.latest_allegro_angles = self._current_hand_action()

        self._tracker_lock = threading.Lock()
        self._tracker_history: deque[tuple[float, np.ndarray]] = deque()
        self._last_tracker_packet_time: float | None = None
        self._tracker_baseline_wait_started_at: float | None = None
        self._tracker_rebase_requested = False
        self._tracker_rebase_hold_pending = False
        self._manual_tracker_rebase_requested = False
        self._tracker_stationary_since: float | None = None
        self._tracker_stationary_drift_eligible = True
        self._hand_lock = threading.Lock()
        self._stop_tracker_thread = False
        self._stop_hand_thread = False
        self._key_callback_window = None
        self._last_tracker_motion_log = 0.0
        self._rokae_wuji_teleop = _is_rokae_wuji_teleop(self.env)
        self.target_frame = _teleop_target_frame(self.env)
        self.direct_hand_targets = (
            _supports_direct_hand_targets(self.env)
            and (self.target_frame == "tracker" or self._rokae_wuji_teleop)
        )
        self.direct_tracker_passthrough = bool(
            _env_attr(self.env, "_direct_tracker_passthrough")
        )

        self._vive_socket = _create_udp_socket(
            config.vive_udp_host,
            config.vive_udp_port,
            timeout=0.5,
        )
        self._hand_socket = _create_udp_socket(
            config.hand_udp_host,
            config.hand_udp_port,
            timeout=0.1,
            reuseaddr=True,
        )

        self._vive_thread = threading.Thread(target=self._recv_vive_loop, daemon=True)
        self._hand_thread = threading.Thread(target=self._recv_hand_loop, daemon=True)
        self._vive_thread.start()
        self._hand_thread.start()

        self._ensure_key_callback()

    def _ensure_key_callback(self):
        self._key_callback_window, _ = _maybe_set_key_callback(
            self.env,
            self.glfw_on_key,
            current_window=self._key_callback_window,
        )

    def _establish_episode_tracker_baseline(self) -> bool:
        """Make the current physical and simulated poses this episode's zero."""

        if not bool(getattr(self, "_rokae_wuji_teleop", False)):
            return True
        with self._tracker_lock:
            self.latest_tracker = None
            self._last_tracker_packet_time = None
            self._tracker_history.clear()

        packet = request_tracker_episode_baseline()
        if packet is None:
            print(
                "[Teleop][WARN] Mode A did not acknowledge the new episode "
                "tracker baseline; holding the arm until a live pose arrives.",
                flush=True,
            )
            return False

        pose = np.frombuffer(packet, dtype=np.float64, count=12).reshape(3, 4)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :] = pose
        packet_time = time.monotonic()
        with self._tracker_lock:
            self.latest_tracker = transform
            self._last_tracker_packet_time = packet_time
            self._tracker_history.clear()
            self._tracker_history.append((packet_time, transform.copy()))
        print(
            "[Teleop] New episode tracker baseline established at the current pose.",
            flush=True,
        )
        return True

    def _consume_manual_tracker_rebase_action(self) -> np.ndarray | None:
        requested = bool(
            getattr(self, "_manual_tracker_rebase_requested", False)
        )
        self._manual_tracker_rebase_requested = False
        requested = (
            _consume_manual_tracker_rebase_request(self.env) or requested
        )
        if not requested:
            return None
        _cancel_auto_tracker_rebase(self)
        if not self.intervened:
            print(
                "[Teleop] Ignored Z tracker rebase because teleoperation is disabled.",
                flush=True,
            )
            return None

        physical_pose = _single_teleop_start_pose(self.env).copy()
        tracker_pose: np.ndarray | None = None
        if bool(getattr(self, "_rokae_wuji_teleop", False)):
            packet = request_tracker_episode_baseline()
            if packet is not None and len(packet) >= 12 * 8:
                values = np.frombuffer(packet, dtype=np.float64, count=12)
                candidate = np.eye(4, dtype=np.float64)
                candidate[:3, :] = values.reshape(3, 4)
                if np.all(np.isfinite(candidate)):
                    tracker_pose = candidate
        else:
            with self._tracker_lock:
                if self.latest_tracker is not None:
                    tracker_pose = self.latest_tracker.copy()

        if tracker_pose is None:
            print(
                "[Teleop][WARN] Z tracker rebase was not acknowledged; holding "
                "the current simulated palm and keeping the previous mapping.",
                flush=True,
            )
        else:
            packet_time = time.monotonic()
            with self._tracker_lock:
                self.latest_tracker = tracker_pose.copy()
                self.tracker_start_world = tracker_pose.copy()
                self._last_tracker_packet_time = packet_time
                self._tracker_history.clear()
                self._tracker_history.append((packet_time, tracker_pose.copy()))
                self._tracker_baseline_wait_started_at = None
                self._tracker_rebase_requested = False
                self._tracker_rebase_hold_pending = False
                self._tracker_fault_hold_pose = None
                self._last_commanded_tracker_pose = tracker_pose.copy()
                self._tracker_stationary_since = None
                self._tracker_stationary_drift_eligible = True
            self.ee_start = physical_pose.copy()
            self._filtered_target_pose = physical_pose.copy()
            self.hold_pose = physical_pose.copy()
            print(
                "[Teleop] Z tracker rebase complete: the current glove pose now "
                "maps to the current simulated palm without a jump.",
                flush=True,
            )

        # Explicitly overwrite any Cartesian target latched before the freeze.
        return np.concatenate(
            [
                physical_pose[:3, 3],
                _quat_wxyz_from_matrix(physical_pose[:3, :3]),
            ],
            axis=0,
        )

    def set_intervened(self, enabled: bool, *, start_delay: bool = True):
        enabled = bool(enabled)
        if enabled == self.intervened:
            return self.intervened

        _cancel_auto_tracker_rebase(self)
        self.intervened = enabled
        self._tracker_fault_hold_pose = None
        if self.intervened:
            if start_delay:
                print(
                    f"Teleoperation will be enabled in "
                    f"{INTERVENTION_START_DELAY_SECONDS:.1f}s...",
                    flush=True,
                )
                time.sleep(INTERVENTION_START_DELAY_SECONDS)
            with self._tracker_lock:
                self._tracker_rebase_requested = False
                self._tracker_rebase_hold_pending = False
                stable_tracker = (
                    self._stable_tracker_pose_locked() if start_delay else None
                )
                self.tracker_start_world = (
                    stable_tracker
                    if stable_tracker is not None
                    else (
                        None
                        if self.latest_tracker is None
                        else self.latest_tracker.copy()
                    )
                )
                # Arm recovery now, before the receiver can fill latest_tracker.
                # A packet arriving between enable and the first action would
                # otherwise bypass the missing-input hold and leave a moving
                # glove waiting indefinitely for a stationary initial baseline.
                self._tracker_rebase_hold_pending = bool(
                    self.tracker_start_world is None
                    and getattr(self.config, "preserve_tracker_baseline_on_recovery", False)
                )
                self._tracker_baseline_wait_started_at = (
                    time.monotonic()
                    if self.tracker_start_world is None
                    else None
                )
                self._last_commanded_tracker_pose = (
                    None
                    if self.tracker_start_world is None
                    else self.tracker_start_world.copy()
                )
                self._tracker_stationary_since = None
                self._tracker_stationary_drift_eligible = True
            try:
                self.ee_start = _single_teleop_start_pose(self.env).copy()
                self._filtered_target_pose = self.ee_start.copy()
                print(
                    "Teleop config: "
                    f"pose_scale={self.config.pose_scale:.2f}, "
                    f"target_frame={self.target_frame}",
                    flush=True,
                )
            except Exception:
                self.ee_start = None
                self.intervened = False
                print(
                    "Warning: failed to read EE pose; teleop was not enabled.",
                    flush=True,
                )
        else:
            self.tracker_start_world = None
            self._tracker_baseline_wait_started_at = None
            self._tracker_rebase_requested = False
            self._tracker_rebase_hold_pending = False
            self.ee_start = None
            self._filtered_target_pose = None
            self._last_commanded_tracker_pose = None
            self._tracker_stationary_since = None
            self._tracker_stationary_drift_eligible = True
            self._capture_hold_pose()

        print(
            f"Teleoperation {'enabled' if self.intervened else 'disabled'}.",
            flush=True,
        )
        return self.intervened

    def _recv_vive_loop(self):
        while not self._stop_tracker_thread:
            try:
                data, _ = _recv_latest_udp_datagram(self._vive_socket, 2048)
                if len(data) < 12 * 8:
                    if getattr(self.config, "preserve_tracker_baseline_on_recovery", False):
                        with self._tracker_lock:
                            _queue_tracker_hold(self, time.monotonic())
                            self._tracker_history.clear()
                    continue
                pose = np.frombuffer(data, dtype=np.float64, count=12).reshape(3, 4)
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :] = pose
                self._accept_tracker_pose(transform, packet_time=time.monotonic())
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

    def _accept_tracker_pose(self, transform: np.ndarray, *, packet_time: float) -> bool:
        if not getattr(self.config, "preserve_tracker_baseline_on_recovery", False):
            if not _valid_tracker_transform(transform):
                return False
            self._accept_legacy_tracker_pose(transform, packet_time=packet_time)
            return True
        with self._tracker_lock:
            valid = _valid_tracker_transform(transform)
            previous = self.latest_tracker
            jump = valid and previous is not None and _tracker_pose_jump_exceeds(
                previous, transform, self.config
            )
            last_time = getattr(self, "_last_tracker_packet_time", None)
            gap = (self.intervened and last_time is not None
                   and packet_time - last_time > _tracker_timeout_seconds(self.config))
            if not valid or jump or gap:
                _queue_tracker_hold(self, packet_time)
            if not valid:
                self._tracker_history.clear()
                return False
            if not hasattr(self, "_tracker_history"):
                self._tracker_history = deque()
            if self._tracker_history and _tracker_pose_jump_exceeds(
                self._tracker_history[-1][1], transform, self.config
            ):
                self._tracker_history.clear()
            self._last_tracker_packet_time = packet_time
            self._tracker_history.append((packet_time, transform.copy()))
            cutoff = packet_time - TRACKER_BASELINE_WINDOW_SECONDS
            while len(self._tracker_history) > 3 and self._tracker_history[1][0] < cutoff:
                self._tracker_history.popleft()
            if self._tracker_rebase_hold_pending:
                return False
            self.latest_tracker = transform.copy()
            return True

    def _accept_legacy_tracker_pose(self, transform: np.ndarray, *, packet_time: float) -> None:
        with self._tracker_lock:
            tracker_jump = bool(
                self.latest_tracker is not None
                and _tracker_pose_jump_exceeds(
                    self.latest_tracker,
                    transform,
                    self.config,
                )
            )
            if (
                self._last_tracker_packet_time is None
                or packet_time - self._last_tracker_packet_time
                > TRACKER_REBASE_GAP_SECONDS
                or tracker_jump
            ):
                self._tracker_history.clear()
                if self.intervened:
                    # The action thread owns baseline replacement so a
                    # receive callback cannot invalidate a mapping that
                    # is already being evaluated.
                    self._tracker_rebase_requested = True
                    self._tracker_rebase_hold_pending = True
                    self._tracker_baseline_wait_started_at = packet_time
                if tracker_jump:
                    print(
                        "[Teleop] Tracker relocalization jump rejected; "
                        "waiting for a stable pose and rebasing in place.",
                        flush=True,
                    )
            self._last_tracker_packet_time = packet_time
            self.latest_tracker = transform
            self._tracker_history.append((packet_time, transform.copy()))
            history_cutoff = packet_time - TRACKER_BASELINE_WINDOW_SECONDS
            while (
                len(self._tracker_history) > 1
                and self._tracker_history[0][0] < history_cutoff
            ):
                self._tracker_history.popleft()

    def _stable_tracker_pose_locked(self) -> np.ndarray | None:
        history = getattr(self, "_tracker_history", None)
        if getattr(self.config, "preserve_tracker_baseline_on_recovery", False):
            if history is None or not _tracker_history_is_fresh(history, self.config):
                return None
        if history is None:
            return None if self.latest_tracker is None else self.latest_tracker.copy()
        if len(history) < 2:
            return None
        if history[-1][0] - history[0][0] < TRACKER_BASELINE_WINDOW_SECONDS * 0.9:
            return None

        latest = history[-1][1]
        max_pos_spread = max(
            float(np.linalg.norm(sample[:3, 3] - latest[:3, 3]))
            for _, sample in history
        )
        max_rot_spread = max(
            _rotation_angle_deg(sample[:3, :3] @ latest[:3, :3].T)
            for _, sample in history
        )
        if (
            max_pos_spread > TRACKER_BASELINE_POS_SPREAD_M
            or max_rot_spread > TRACKER_BASELINE_ROT_SPREAD_DEG
        ):
            return None
        return latest.copy()

    def _tracker_is_stationary_locked(self) -> bool:
        """Return whether the recent tracker window contains only idle drift."""

        if not self.config.rebase_on_stationary_drift:
            return False
        history = getattr(self, "_tracker_history", None)
        if history is None or len(history) < 2:
            return False
        if history[-1][0] - history[0][0] < TRACKER_BASELINE_WINDOW_SECONDS * 0.9:
            return False

        latest = history[-1][1]
        max_pos_spread = max(
            float(np.linalg.norm(sample[:3, 3] - latest[:3, 3]))
            for _, sample in history
        )
        max_rot_spread = max(
            _rotation_angle_deg(sample[:3, :3] @ latest[:3, :3].T)
            for _, sample in history
        )
        return bool(
            max_pos_spread
            <= max(0.0, self.config.tracker_stationary_max_translation_m)
            and max_rot_spread
            <= max(0.0, self.config.tracker_stationary_max_rotation_deg)
        )

    def _recv_hand_loop(self):
        while not self._stop_hand_thread:
            try:
                data, _ = _recv_latest_udp_datagram(self._hand_socket, 4096)
                if not data:
                    continue
                parsed = _parse_udp_hand_angles(data)
                if parsed is None:
                    continue
                with self._hand_lock:
                    self.latest_allegro_angles = parsed
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

    def glfw_on_key(self, window, key, scancode, action, mods):
        del window
        del scancode
        del mods

        if action == glfw.PRESS and key == glfw.KEY_SEMICOLON:
            self.set_intervened(not self.intervened)

        if action == glfw.PRESS and key == glfw.KEY_R:
            self.reset_trigger = True

        if action == glfw.PRESS and key == glfw.KEY_SPACE:
            # Space is a route delimiter only in hands-only guidance.  Keep the
            # flag on the wrapper as well as the Isaac env because either GLFW
            # or Kit's keyboard subscription may receive the event first.
            self._guidance_route_complete_requested = True

        if action == glfw.PRESS and key == glfw.KEY_Z:
            self._manual_tracker_rebase_requested = True

    def _vive_action(self):
        if not self.intervened or self.ee_start is None:
            return None
        direct_tracker_passthrough = bool(
            getattr(self, "direct_tracker_passthrough", False)
        )

        tracker_stationary = False
        with self._tracker_lock:
            if getattr(self.config, "preserve_tracker_baseline_on_recovery", False):
                now = time.monotonic()
                last_time = getattr(self, "_last_tracker_packet_time", None)
                if self.latest_tracker is None or (
                    last_time is not None and now - last_time > _tracker_timeout_seconds(self.config)
                ):
                    _queue_tracker_hold(self, now)
                if getattr(self, "_tracker_rebase_hold_pending", False):
                    if getattr(self.config, "auto_rebase_on_tracker_recovery", False):
                        return self._physical_palm_hold_action()
                    stable = self._stable_tracker_pose_locked()
                    if stable is None or (
                        self.latest_tracker is not None
                        and _tracker_pose_jump_exceeds(self.latest_tracker, stable, self.config)
                    ):
                        return self._physical_palm_hold_action()
                    hold_action = self._physical_palm_hold_action()
                    if self.tracker_start_world is None:
                        self.tracker_start_world = stable.copy()
                        self.ee_start = self._tracker_fault_hold_pose.copy()
                    self.latest_tracker = stable.copy()
                    self._last_commanded_tracker_pose = stable.copy()
                    self._tracker_rebase_hold_pending = False
                    self._tracker_rebase_requested = False
                    self._tracker_baseline_wait_started_at = None
                    self._tracker_fault_hold_pose = None
                    self._tracker_history.clear()
                    print("[Teleop] Original tracker recovered; palm hold released.", flush=True)
                    return hold_action
            if self.latest_tracker is None:
                return None
            tracker_now = self.latest_tracker.copy()
            if getattr(self, "_tracker_rebase_requested", False):
                self.tracker_start_world = None
                self._tracker_rebase_requested = False
                self._last_commanded_tracker_pose = None
                self._tracker_stationary_since = None
                self._tracker_stationary_drift_eligible = True
            if self.tracker_start_world is None:
                rebase_hold_pending = bool(
                    getattr(self, "_tracker_rebase_hold_pending", False)
                )
                stable_tracker = self._stable_tracker_pose_locked()
                baseline_wait_started_at = getattr(
                    self,
                    "_tracker_baseline_wait_started_at",
                    None,
                )
                if baseline_wait_started_at is None:
                    self._tracker_baseline_wait_started_at = time.monotonic()
                elif (
                    stable_tracker is None
                    and not getattr(self.config, "preserve_tracker_baseline_on_recovery", False)
                    and time.monotonic() - baseline_wait_started_at
                    >= TRACKER_BASELINE_MAX_WAIT_SECONDS
                ):
                    # Some SteamVR stacks retain a few millimetres of idle
                    # jitter forever.  After a bounded hold window, use the
                    # newest real packet as a zero-motion baseline rather than
                    # leaving teleoperation permanently disabled.
                    stable_tracker = tracker_now
                if stable_tracker is None:
                    if rebase_hold_pending:
                        physical_pose = _single_teleop_start_pose(self.env).copy()
                        self.ee_start = physical_pose.copy()
                        self._filtered_target_pose = physical_pose.copy()
                        return np.concatenate(
                            [
                                physical_pose[:3, 3],
                                _quat_wxyz_from_matrix(physical_pose[:3, :3]),
                            ],
                            axis=0,
                        )
                    return self._zero_pose_action()
                self.tracker_start_world = stable_tracker
                self._tracker_baseline_wait_started_at = None
                self._tracker_rebase_hold_pending = False
                self.ee_start = _single_teleop_start_pose(self.env).copy()
                self._filtered_target_pose = self.ee_start.copy()
                self._last_commanded_tracker_pose = stable_tracker.copy()
                if rebase_hold_pending or self.config.command_on_tracker_motion_only:
                    # A packet gap/relocalization rebases at the palm's real
                    # physical pose.  Submit that pose once so a previously
                    # latched Cartesian servo cannot continue chasing its old
                    # goal while the tracker is stationary.  This is a drive
                    # target, not a joint/root state or velocity write.
                    return np.concatenate(
                        [
                            self.ee_start[:3, 3],
                            _quat_wxyz_from_matrix(self.ee_start[:3, :3]),
                        ],
                        axis=0,
                    )
                return self._zero_pose_action()
            tracker_stationary = self._tracker_is_stationary_locked()

        motion_only = bool(self.config.command_on_tracker_motion_only)
        if motion_only and self.config.rebase_on_stationary_drift:
            now = time.monotonic()
            if tracker_stationary:
                if self._tracker_stationary_since is None:
                    self._tracker_stationary_since = now
            else:
                self._tracker_stationary_since = None
                # Once the recent window contains an intentional-speed move,
                # do not reinterpret the following stationary hold as drift
                # while the physical arm is still travelling to that goal.
                self._tracker_stationary_drift_eligible = False

            stationary_since = self._tracker_stationary_since
            if (
                tracker_stationary
                and stationary_since is not None
                and now - stationary_since
                >= max(0.0, self.config.tracker_stationary_rebase_after_sec)
                and self.tracker_start_world is not None
            ):
                baseline_pos_delta = float(
                    np.linalg.norm(
                        tracker_now[:3, 3]
                        - self.tracker_start_world[:3, 3]
                    )
                )
                baseline_rot_delta = _rotation_angle_deg(
                    tracker_now[:3, :3]
                    @ self.tracker_start_world[:3, :3].T
                )
                if (
                    baseline_pos_delta
                    > max(0.0, self.config.tracker_motion_deadband_m)
                    or baseline_rot_delta
                    > max(0.0, self.config.tracker_motion_rot_deadband_deg)
                ):
                    physical_pose = _single_teleop_start_pose(self.env).copy()
                    mapped_target = _target_pose_from_bridge_world_pose(
                        self.ee_start,
                        self.tracker_start_world,
                        tracker_now,
                        1.0,
                        max_tracker_delta_m=self.config.max_tracker_delta_m,
                    )
                    target_pos_error = float(
                        np.linalg.norm(
                            mapped_target[:3, 3] - physical_pose[:3, 3]
                        )
                    )
                    target_rot_error = _rotation_angle_deg(
                        mapped_target[:3, :3] @ physical_pose[:3, :3].T
                    )
                    safe_after_deliberate_motion = bool(
                        target_pos_error <= TRACKER_HOLD_POS_EPS_M
                        and target_rot_error <= TRACKER_HOLD_ROT_EPS_DEG
                    )
                    if (
                        not self._tracker_stationary_drift_eligible
                        and not safe_after_deliberate_motion
                    ):
                        # The tracker was deliberately moved and the simulated
                        # palm has not arrived yet.  Leave the latched Cartesian
                        # command alone; rebasing here would truncate the move.
                        return self._zero_pose_action()
                    with self._tracker_lock:
                        self.tracker_start_world = tracker_now.copy()
                        self._last_commanded_tracker_pose = tracker_now.copy()
                    self.ee_start = physical_pose.copy()
                    self._filtered_target_pose = physical_pose.copy()
                    self._tracker_stationary_since = now
                    self._tracker_stationary_drift_eligible = True
                    print(
                        "[Teleop] Absorbed stationary tracker drift and "
                        "rebased at the physical palm pose.",
                        flush=True,
                    )
                    return np.concatenate(
                        [
                            physical_pose[:3, 3],
                            _quat_wxyz_from_matrix(physical_pose[:3, :3]),
                        ],
                        axis=0,
                    )

        if motion_only:
            previous_tracker = self._last_commanded_tracker_pose
            if previous_tracker is None:
                previous_tracker = self.tracker_start_world
            pos_delta = float(
                np.linalg.norm(
                    tracker_now[:3, 3] - previous_tracker[:3, 3]
                )
            )
            rot_delta = _rotation_angle_deg(
                tracker_now[:3, :3] @ previous_tracker[:3, :3].T
            )
            if (
                pos_delta <= max(0.0, self.config.tracker_motion_deadband_m)
                and rot_delta
                <= max(0.0, self.config.tracker_motion_rot_deadband_deg)
            ):
                return self._zero_pose_action()
        elif (
            self._rokae_wuji_teleop
            and self.direct_hand_targets
            and not direct_tracker_passthrough
        ):
            pos_delta = np.linalg.norm(
                tracker_now[:3, 3] - self.tracker_start_world[:3, 3]
            )
            rot_delta = _rotation_angle_deg(
                tracker_now[:3, :3] @ self.tracker_start_world[:3, :3].T
            )
            if (
                pos_delta <= TRACKER_HOLD_POS_EPS_M
                and rot_delta <= TRACKER_HOLD_ROT_EPS_DEG
            ):
                return self._zero_pose_action()

        if self.direct_hand_targets:
            target_pose = _target_pose_from_bridge_world_pose(
                self.ee_start,
                self.tracker_start_world,
                tracker_now,
                1.0,
                max_tracker_delta_m=self.config.max_tracker_delta_m,
            )
        elif self._rokae_wuji_teleop:
            target_pose = _target_pose_from_bridge_world_pose(
                self.ee_start,
                self.tracker_start_world,
                tracker_now,
                self.config.pose_scale,
                max_tracker_delta_m=self.config.max_tracker_delta_m,
            )
        else:
            target_pose = _target_pose_from_tracker_pose(
                self.ee_start,
                self.tracker_start_world,
                tracker_now,
                self.config.pose_scale,
                self.target_frame,
            )

        if not direct_tracker_passthrough:
            target_pose = _filter_target_pose(
                self._filtered_target_pose,
                target_pose,
                self.config,
            )
        self._filtered_target_pose = target_pose.copy()
        if motion_only:
            self._last_commanded_tracker_pose = tracker_now.copy()

        now = time.monotonic()
        if now - self._last_tracker_motion_log >= 2.0:
            pos_delta = np.linalg.norm(
                tracker_now[:3, 3] - self.tracker_start_world[:3, 3]
            )
            tracker_dz = tracker_now[2, 3] - self.tracker_start_world[2, 3]
            target_dz = target_pose[2, 3] - self.ee_start[2, 3]
            tracker_delta_xyz = tracker_now[:3, 3] - self.tracker_start_world[:3, 3]
            target_delta_xyz = target_pose[:3, 3] - self.ee_start[:3, 3]
            rot_delta = tracker_now[:3, :3] @ self.tracker_start_world[:3, :3].T
            target_rot_delta = target_pose[:3, :3] @ self.ee_start[:3, :3].T
            palm_axis = self.tracker_start_world[:3, 0]
            palm_rot = _rotation_component_deg(rot_delta, palm_axis)
            target_palm_rot = _rotation_component_deg(target_rot_delta, palm_axis)
            print(
                "tracker delta: "
                f"pos={pos_delta:.3f}m "
                f"xyz=({tracker_delta_xyz[0]:+.3f},"
                f"{tracker_delta_xyz[1]:+.3f},"
                f"{tracker_delta_xyz[2]:+.3f})m "
                f"target_xyz=({target_delta_xyz[0]:+.3f},"
                f"{target_delta_xyz[1]:+.3f},"
                f"{target_delta_xyz[2]:+.3f})m "
                f"dz={tracker_dz:+.3f}m "
                f"target_dz={target_dz:+.3f}m "
                f"palm_rot={palm_rot:+.1f}deg "
                f"target_palm_rot={target_palm_rot:+.1f}deg "
                f"rot={_rotation_angle_deg(rot_delta):.1f}deg",
                flush=True,
            )
            self._last_tracker_motion_log = now

        return np.concatenate(
            [target_pose[:3, 3], _quat_wxyz_from_matrix(target_pose[:3, :3])],
            axis=0,
        )

    def _zero_pose_action(self):
        return np.zeros(7, dtype=np.float64)

    def _physical_palm_hold_action(self) -> np.ndarray:
        # Latch once: continuously following measured drift would not hold
        # the position at which the tracker fault was detected.
        if getattr(self, "_tracker_fault_hold_pose", None) is None:
            self._tracker_fault_hold_pose = _single_teleop_start_pose(self.env).copy()
        pose = self._tracker_fault_hold_pose
        self._filtered_target_pose = pose.copy()
        self._tracker_hold_action_this_frame = True
        return np.concatenate([pose[:3, 3], _quat_wxyz_from_matrix(pose[:3, :3])])

    def _hold_pose_action(self):
        if self._rokae_wuji_teleop and self.direct_hand_targets:
            return self._zero_pose_action()
        if self.hold_pose is None:
            self._capture_hold_pose()
        ee_pose = self.hold_pose
        return np.concatenate(
            [ee_pose[:3, 3], _quat_wxyz_from_matrix(ee_pose[:3, :3])],
            axis=0,
        )

    def _capture_hold_pose(self):
        self.hold_pose = _single_teleop_start_pose(self.env).copy()

    def _current_hand_action(self) -> np.ndarray:
        if self._single_wuji_native_hand:
            # Isaac collection environments expose their live command buffer
            # directly.  It is synchronized to the authored default joints by
            # every full reset, so it is the correct no-packet hold target.
            isaac_hand = _env_attr(self.env, "_right_hand")
            if isaac_hand is not None:
                values = np.asarray(isaac_hand, dtype=np.float64).reshape(-1)
                result = np.zeros(self._hand_action_dim, dtype=np.float64)
                count = min(result.size, values.size)
                result[:count] = values[:count]
                return result
            hand = _env_attr(self.env, "_hand")
            data = _env_attr(self.env, "_data")
            qpos_ids = getattr(hand, "qpos_ids", None)
            if data is not None and qpos_ids is not None:
                return np.asarray(data.qpos[qpos_ids], dtype=np.float64).copy()
        return np.zeros(self._hand_action_dim, dtype=np.float64)

    def _hand_action(self, action: np.ndarray):
        if self.intervened:
            with self._hand_lock:
                return self.latest_allegro_angles.copy()
        if action is None or action.shape[0] <= 7:
            with self._hand_lock:
                return self.latest_allegro_angles.copy()
        hand_payload = np.asarray(action[7:], dtype=np.float64)
        if hand_payload.size == 0 or np.allclose(hand_payload, 0.0):
            with self._hand_lock:
                return self.latest_allegro_angles.copy()
        if action is not None and action.shape[0] >= 7 + WUJI_NATIVE_HAND_DIM:
            return np.asarray(action[7 : 7 + WUJI_NATIVE_HAND_DIM], dtype=np.float64)
        if action is not None and action.shape[0] >= 7 + HAND_POLICY_DIM:
            return np.asarray(action[7 : 7 + HAND_POLICY_DIM], dtype=np.float64)
        with self._hand_lock:
            return self.latest_allegro_angles.copy()

    def action(self, action: np.ndarray):
        self._tracker_hold_action_this_frame = False
        pose_action = self._consume_manual_tracker_rebase_action()
        if pose_action is None:
            pose_action = self._vive_action()
        automatic_action = _auto_tracker_rebase_action(self, dual=False)
        if automatic_action is not None:
            pose_action = automatic_action
        if pose_action is None:
            pose_action = self._hold_pose_action()
        hand_action = _condition_b02_wuji_hand_action(self.env, self._hand_action(action))
        operator_action = _consume_teleop_action_extras(self.env)
        with self._tracker_lock:
            if self.intervened and getattr(self, "_tracker_rebase_hold_pending", False):
                automatic_action = _auto_tracker_rebase_action_locked(self, dual=False)
                pose_action = automatic_action if automatic_action is not None else self._physical_palm_hold_action()
        return np.concatenate(
            [pose_action, hand_action, operator_action],
            axis=0,
        )

    def step(self, action):
        self._ensure_key_callback()
        sample_started = time.monotonic()
        teleop_action = self.action(action)
        action_sampled = time.monotonic()
        obs, rew, done, truncated, info = self.env.step(teleop_action)
        step_finished = time.monotonic()
        info = dict(info)
        info["intervene_action"] = teleop_action
        route_complete = bool(self._guidance_route_complete_requested)
        self._guidance_route_complete_requested = False
        route_complete = (
            _consume_guidance_route_complete_request(self.env) or route_complete
        )
        if route_complete:
            info["guidance_route_complete"] = True
        with self._tracker_lock:
            packet_time = self._last_tracker_packet_time
        info["teleop_timing"] = {
            "tracker_packet_age_ms": (
                None
                if packet_time is None
                else float(max(0.0, action_sampled - packet_time) * 1000.0)
            ),
            "action_sampling_ms": float((action_sampled - sample_started) * 1000.0),
            "environment_step_ms": float((step_finished - action_sampled) * 1000.0),
        }
        if self.reset_trigger:
            info["manual_reset"] = True
            self.reset_trigger = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        _cancel_auto_tracker_rebase(self)
        resume_intervention = bool(self.intervened)
        obs, info = self.env.reset(**kwargs)
        self._ensure_key_callback()
        self.intervened = False
        self._tracker_fault_hold_pose = None
        self.reset_trigger = False
        self._guidance_route_complete_requested = False
        self.tracker_start_world = None
        self._tracker_baseline_wait_started_at = None
        self._tracker_rebase_requested = False
        self._tracker_rebase_hold_pending = False
        self._manual_tracker_rebase_requested = False
        self.ee_start = None
        self._filtered_target_pose = None
        self._last_commanded_tracker_pose = None
        self._tracker_stationary_since = None
        self._tracker_stationary_drift_eligible = True
        self._establish_episode_tracker_baseline()
        self._capture_hold_pose()
        with self._hand_lock:
            self.latest_allegro_angles = self._current_hand_action()
        if resume_intervention:
            print(
                f"[Teleop] Reset pause: hold for "
                f"{RESET_REBASE_DELAY_SECONDS:.1f}s to reposition the glove.",
                flush=True,
            )
            time.sleep(RESET_REBASE_DELAY_SECONDS)
            # Tracker packets continue arriving during the pause. Rebase from
            # the final glove pose so repositioning does not move the robot.
            self.set_intervened(True, start_delay=False)
        return obs, info

    def close(self):
        _cancel_auto_tracker_rebase(self)
        if bool(getattr(self, "_rokae_wuji_teleop", False)):
            release_tracker_episode_guard()
        self._stop_tracker_thread = True
        self._stop_hand_thread = True
        if self._vive_thread.is_alive():
            self._vive_thread.join(timeout=1.0)
        if self._hand_thread.is_alive():
            self._hand_thread.join(timeout=1.0)
        try:
            self._vive_socket.close()
        except Exception:
            pass
        try:
            self._hand_socket.close()
        except Exception:
            pass
        self.env.close()


class DualArmViveHandTeleopWrapper(gym.ActionWrapper):
    def __init__(self, env, config: BimanualTeleopConfig):
        super().__init__(env)
        self.config = config
        self.intervened = False
        self.reset_trigger = False
        self._guidance_route_complete_requested = False
        self.tracker_start_world = None
        self.ee_start = None
        self.hold_pose = None
        self._tracker_fault_hold_pose: dict[str, np.ndarray] | None = None
        self._filtered_target_pose = {"right": None, "left": None}
        # A transient, live-only twist hint consumed by the IsaacLab H01
        # controller.  It is deliberately not concatenated into the action,
        # so the recorder/replayer contract remains pose based.
        self._live_velocity_hint_enabled = bool(
            getattr(config, "velocity_assist", False)
        )
        self.latest_tracker_right: np.ndarray | None = None
        self.latest_tracker_left: np.ndarray | None = None
        self._dual_wuji_native_hand = _is_wuji_hand_env(self.env)
        self._hand_action_dim = (
            WUJI_NATIVE_HAND_DIM if self._dual_wuji_native_hand else HAND_POLICY_DIM
        )
        self.latest_allegro_angles_right = np.zeros(self._hand_action_dim, dtype=np.float64)
        self.latest_allegro_angles_left = np.zeros(self._hand_action_dim, dtype=np.float64)
        self._commanded_hand_angles_right = self.latest_allegro_angles_right.copy()
        self._commanded_hand_angles_left = self.latest_allegro_angles_left.copy()
        self._hand_takeover_steps_remaining = 0
        self._last_hand_packet_time_right: float | None = None
        self._last_hand_packet_time_left: float | None = None
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14 + 2 * self._hand_action_dim,),
            dtype=np.float32,
        )
        self.hands_only = _env_flag("HANDSON_HANDS_ONLY")
        self._rokae_wuji_teleop = _is_rokae_wuji_teleop(self.env)
        self.target_frame = _teleop_target_frame(self.env)
        target_control = self.hands_only or _supports_direct_hand_targets(self.env)
        self.direct_hand_targets = target_control and (
            self.target_frame == "tracker" or self._rokae_wuji_teleop
        )

        self._tracker_lock = threading.Lock()
        self._tracker_history: deque[
            tuple[float, np.ndarray, np.ndarray]
        ] = deque()
        self._last_tracker_packet_time: float | None = None
        self._tracker_baseline_wait_started_at: float | None = None
        self._tracker_rebase_requested = False
        self._tracker_rebase_hold_pending = False
        self._manual_tracker_rebase_requested = False
        self._hand_lock = threading.Lock()
        self._stop_tracker_thread = False
        self._stop_hand_thread = False
        self._key_callback_window = None
        self._last_tracker_motion_log = 0.0

        self._vive_socket = _create_udp_socket(
            config.vive_udp_host,
            config.vive_udp_port,
            timeout=0.5,
        )
        self._right_hand_socket = _create_udp_socket(
            config.right_hand_udp_host,
            config.right_hand_udp_port,
            timeout=0.1,
            reuseaddr=True,
        )
        self._left_hand_socket = _create_udp_socket(
            config.left_hand_udp_host,
            config.left_hand_udp_port,
            timeout=0.1,
            reuseaddr=True,
        )

        self._vive_thread = threading.Thread(target=self._recv_vive_loop, daemon=True)
        self._hand_thread = threading.Thread(target=self._recv_hand_loop, daemon=True)
        self._vive_thread.start()
        self._hand_thread.start()

        self._ensure_key_callback()

    def _publish_live_velocity_hint(
        self,
        previous_targets: dict[str, np.ndarray] | None,
        current_targets: dict[str, np.ndarray] | None,
    ) -> None:
        """Publish one transient H01 twist hint without changing the action."""

        config = getattr(self, "config", None)
        if not bool(
            getattr(
                self,
                "_live_velocity_hint_enabled",
                getattr(config, "velocity_assist", False),
            )
        ):
            return
        callback = _env_attr(
            getattr(self, "env", None),
            "set_live_teleop_velocity_hint",
        )
        if not callable(callback):
            return
        if previous_targets is None or current_targets is None:
            callback(None)
            return

        control_dt = _env_attr(self.env, "_control_dt")
        if control_dt is None:
            control_hz = _env_attr(self.env, "hz")
            try:
                control_dt = 1.0 / float(control_hz)
            except (TypeError, ValueError, ZeroDivisionError):
                control_dt = 1.0 / 30.0
        try:
            control_dt = max(float(control_dt), 1e-6)
        except (TypeError, ValueError):
            control_dt = 1.0 / 30.0

        max_linear = getattr(self.config, "velocity_assist_max_linear_m_s", None)
        max_angular = getattr(
            self.config,
            "velocity_assist_max_angular_rad_s",
            None,
        )
        payload: dict[str, object] = {
            "_gain": getattr(self.config, "velocity_assist_gain", 0.35),
            "_lookahead_s": getattr(
                self.config,
                "velocity_assist_lookahead_s",
                0.04,
            ),
            "_max_linear_m_s": max_linear,
            "_max_angular_rad_s": max_angular,
            "_stop_speed_m_s": getattr(
                self.config,
                "velocity_assist_stop_speed_m_s",
                0.008,
            ),
        }
        for side in ("right", "left"):
            velocity = _pose_velocity(
                previous_targets.get(side),
                current_targets[side],
                control_dt,
            )
            velocity[:3] = _clip_vector_norm(velocity[:3], max_linear)
            velocity[3:] = _clip_vector_norm(velocity[3:], max_angular)
            payload[side] = velocity
        callback(payload)

    def _clear_live_velocity_hint(self) -> None:
        config = getattr(self, "config", None)
        if not bool(
            getattr(
                self,
                "_live_velocity_hint_enabled",
                getattr(config, "velocity_assist", False),
            )
        ):
            return
        callback = _env_attr(
            getattr(self, "env", None),
            "set_live_teleop_velocity_hint",
        )
        if callable(callback):
            callback(None)

    def _ensure_key_callback(self):
        self._key_callback_window, _ = _maybe_set_key_callback(
            self.env,
            self.glfw_on_key,
            current_window=self._key_callback_window,
        )

    def _establish_episode_tracker_baseline(self) -> bool:
        """Make both current trackers relative to the freshly reset palms."""

        if not bool(getattr(self, "_rokae_wuji_teleop", False)):
            return True
        with self._tracker_lock:
            self.latest_tracker_right = None
            self.latest_tracker_left = None
            self._last_tracker_packet_time = None
            self._tracker_history.clear()

        packet = request_tracker_episode_baseline()
        if packet is None or len(packet) < 24 * 8:
            print(
                "[Teleop][WARN] Mode A did not acknowledge a complete two-"
                "tracker episode baseline; holding both arms until live poses "
                "arrive.",
                flush=True,
            )
            return False

        poses = np.frombuffer(packet, dtype=np.float64, count=24)
        right = np.eye(4, dtype=np.float64)
        right[:3, :] = poses[:12].reshape(3, 4)
        left = np.eye(4, dtype=np.float64)
        left[:3, :] = poses[12:24].reshape(3, 4)
        packet_time = time.monotonic()
        with self._tracker_lock:
            self.latest_tracker_right = right
            self.latest_tracker_left = left
            self._last_tracker_packet_time = packet_time
            self._tracker_history.clear()
            self._tracker_history.append(
                (packet_time, right.copy(), left.copy())
            )
        print(
            "[Teleop] New episode tracker baseline established at the current "
            "two-hand pose.",
            flush=True,
        )
        return True

    def _consume_manual_tracker_rebase_action(
        self,
    ) -> dict[str, np.ndarray] | None:
        requested = bool(
            getattr(self, "_manual_tracker_rebase_requested", False)
        )
        self._manual_tracker_rebase_requested = False
        requested = (
            _consume_manual_tracker_rebase_request(self.env) or requested
        )
        if not requested:
            return None
        _cancel_auto_tracker_rebase(self)
        if not self.intervened:
            print(
                "[Teleop] Ignored Z tracker rebase because teleoperation is disabled.",
                flush=True,
            )
            return None

        right_pose, left_pose = _dual_teleop_start_poses(self.env)
        physical_poses = {
            "right": right_pose.copy(),
            "left": left_pose.copy(),
        }
        tracker_poses: dict[str, np.ndarray] | None = None
        if bool(getattr(self, "_rokae_wuji_teleop", False)):
            packet = request_tracker_episode_baseline()
            if packet is not None and len(packet) >= 24 * 8:
                values = np.frombuffer(packet, dtype=np.float64, count=24)
                right_tracker = np.eye(4, dtype=np.float64)
                right_tracker[:3, :] = values[:12].reshape(3, 4)
                left_tracker = np.eye(4, dtype=np.float64)
                left_tracker[:3, :] = values[12:24].reshape(3, 4)
                if np.all(np.isfinite(right_tracker)) and np.all(
                    np.isfinite(left_tracker)
                ):
                    tracker_poses = {
                        "right": right_tracker,
                        "left": left_tracker,
                    }
        else:
            with self._tracker_lock:
                if (
                    self.latest_tracker_right is not None
                    and self.latest_tracker_left is not None
                ):
                    tracker_poses = {
                        "right": self.latest_tracker_right.copy(),
                        "left": self.latest_tracker_left.copy(),
                    }

        if tracker_poses is None:
            print(
                "[Teleop][WARN] Z tracker rebase was not acknowledged; holding "
                "the current simulated palms and keeping the previous mapping.",
                flush=True,
            )
        else:
            packet_time = time.monotonic()
            with self._tracker_lock:
                self.latest_tracker_right = tracker_poses["right"].copy()
                self.latest_tracker_left = tracker_poses["left"].copy()
                self.tracker_start_world = {
                    side: pose.copy() for side, pose in tracker_poses.items()
                }
                self._last_tracker_packet_time = packet_time
                self._tracker_history.clear()
                self._tracker_history.append(
                    (
                        packet_time,
                        tracker_poses["right"].copy(),
                        tracker_poses["left"].copy(),
                    )
                )
                self._tracker_baseline_wait_started_at = None
                self._tracker_rebase_requested = False
                self._tracker_rebase_hold_pending = False
                self._tracker_fault_hold_pose = None
            self.ee_start = {
                side: pose.copy() for side, pose in physical_poses.items()
            }
            self._filtered_target_pose = {
                side: pose.copy() for side, pose in physical_poses.items()
            }
            self.hold_pose = {
                side: pose.copy() for side, pose in physical_poses.items()
            }
            print(
                "[Teleop] Z tracker rebase complete: the current glove poses now "
                "map to the current simulated palms without a jump.",
                flush=True,
            )

        return {
            side: np.concatenate(
                [pose[:3, 3], _quat_wxyz_from_matrix(pose[:3, :3])],
                axis=0,
            )
            for side, pose in physical_poses.items()
        }

    def set_intervened(self, enabled: bool, *, start_delay: bool = True):
        enabled = bool(enabled)
        if enabled == self.intervened:
            return self.intervened

        _cancel_auto_tracker_rebase(self)
        self.intervened = enabled
        self._tracker_fault_hold_pose = None
        if self.intervened:
            if start_delay:
                print(
                    f"Teleoperation will be enabled in "
                    f"{INTERVENTION_START_DELAY_SECONDS:.1f}s...",
                    flush=True,
                )
                time.sleep(INTERVENTION_START_DELAY_SECONDS)
            if (
                bool(getattr(self, "_rokae_wuji_teleop", False))
                and bool(
                    getattr(
                        self.config,
                        "refresh_tracker_baseline_on_enable",
                        False,
                    )
                )
            ):
                self._establish_episode_tracker_baseline()
            with self._tracker_lock:
                self._tracker_rebase_requested = False
                self._tracker_rebase_hold_pending = False
                require_stable_pair = bool(
                    getattr(self.config, "require_stable_pair_on_enable", False)
                )
                stable_pair = (
                    self._stable_tracker_pair_locked()
                    if require_stable_pair
                    else None
                )
                if require_stable_pair:
                    self.tracker_start_world = stable_pair
                    self._tracker_rebase_hold_pending = stable_pair is None
                else:
                    self.tracker_start_world = (
                        None
                        if (
                            self.latest_tracker_right is None
                            or self.latest_tracker_left is None
                        )
                        else {
                            "right": self.latest_tracker_right.copy(),
                            "left": self.latest_tracker_left.copy(),
                        }
                    )
                # Match the single-arm enable path: arm recovery before a
                # late first pair can bypass the missing-input check. A moving
                # pair must be rebased at the held palms, not wait indefinitely
                # for a stationary baseline after reset.
                if self.tracker_start_world is None and (
                    getattr(self, "_rokae_wuji_teleop", False)
                    or self.config.preserve_tracker_baseline_on_recovery
                ):
                    self._tracker_rebase_hold_pending = True
                self._tracker_baseline_wait_started_at = (
                    time.monotonic()
                    if self.tracker_start_world is None
                    else None
                )
            try:
                right_pose, left_pose = _dual_teleop_start_poses(self.env)
                self.ee_start = {
                    "right": right_pose.copy(),
                    "left": left_pose.copy(),
                }
                self._filtered_target_pose = {
                    "right": right_pose.copy(),
                    "left": left_pose.copy(),
                }
                try:
                    takeover_step = float(
                        self.config.hand_takeover_max_step_rad or 0.0
                    )
                    takeover_steps = int(
                        self.config.hand_takeover_max_control_steps
                    )
                except (TypeError, ValueError):
                    takeover_step = 0.0
                    takeover_steps = 0
                self._hand_takeover_steps_remaining = (
                    max(0, takeover_steps) if takeover_step > 0.0 else 0
                )
                if self._hand_takeover_steps_remaining:
                    print(
                        "[Teleop] One-shot hand takeover armed: "
                        f"max_step={takeover_step:.3f}rad, "
                        f"max_frames={self._hand_takeover_steps_remaining}; "
                        "steady-state glove targets remain direct.",
                        flush=True,
                    )
                print(
                    "Teleop config: "
                    f"pose_scale={self.config.pose_scale:.2f}, "
                    f"target_frame={self.target_frame}",
                    flush=True,
                )
            except Exception:
                self.ee_start = None
                self.intervened = False
                print(
                    "Warning: failed to read EE pose; teleop was not enabled.",
                    flush=True,
                )
        else:
            self._clear_live_velocity_hint()
            self.tracker_start_world = None
            self._tracker_baseline_wait_started_at = None
            self._tracker_rebase_requested = False
            self._tracker_rebase_hold_pending = False
            self.ee_start = None
            self._filtered_target_pose = {"right": None, "left": None}
            self._capture_hold_pose()

        print(
            f"Teleoperation {'enabled' if self.intervened else 'disabled'}.",
            flush=True,
        )
        return self.intervened

    def _accept_tracker_pair(
        self,
        right_transform: np.ndarray,
        left_transform: np.ndarray,
        *,
        packet_time: float,
    ) -> tuple[str, ...]:
        """Accept a live pair while containing SteamVR frame relocations.

        A bad pair never overwrites the trusted pair. Collection defaults
        cancel old servo goals and require fresh, stable samples near that
        trusted pair before releasing the physical hold.
        """

        jump_sides: list[str] = []
        rejected_bridge_packet = False
        started_fail_closed_recovery = False
        with self._tracker_lock:
            if not hasattr(self, "_tracker_history"):
                self._tracker_history = deque()
            bridge_guard_authoritative = bool(
                getattr(self, "_rokae_wuji_teleop", False)
                or self.config.preserve_tracker_baseline_on_recovery
            )
            fail_closed_recovery = bool(
                bridge_guard_authoritative
                and getattr(
                    self.config,
                    "fail_closed_rebase_on_tracker_jump",
                    False,
                )
            )
            recovery_active = bool(
                fail_closed_recovery
                and getattr(self, "_tracker_rebase_hold_pending", False)
            )
            if not (_valid_tracker_transform(right_transform) and _valid_tracker_transform(left_transform)):
                _queue_tracker_hold(self, packet_time)
                self._tracker_history.clear()
                return ("right", "left")
            last_time = getattr(self, "_last_tracker_packet_time", None)
            if (fail_closed_recovery and self.intervened and last_time is not None
                    and packet_time - last_time > self._tracker_packet_max_age_seconds()):
                _queue_tracker_hold(self, packet_time)
                recovery_active = True
            if bridge_guard_authoritative:
                for side, previous, candidate in (
                    (
                        "right",
                        self.latest_tracker_right,
                        right_transform,
                    ),
                    (
                        "left",
                        self.latest_tracker_left,
                        left_transform,
                    ),
                ):
                    if previous is None:
                        continue
                    exceeded = _tracker_pose_jump_exceeds(
                        previous,
                        candidate,
                        self.config,
                    )
                    if exceeded:
                        jump_sides.append(side)

                if fail_closed_recovery and (jump_sides or recovery_active):
                    rejected_bridge_packet = True
                    started_fail_closed_recovery = not recovery_active
                    self._last_tracker_packet_time = packet_time
                    if started_fail_closed_recovery:
                        self._tracker_history.clear()
                        self._tracker_fault_hold_pose = None
                    elif self._tracker_history:
                        previous_candidate = self._tracker_history[-1]
                        candidate_discontinuous = bool(
                            _tracker_pose_jump_exceeds(
                                previous_candidate[1],
                                right_transform,
                                self.config,
                            )
                            or _tracker_pose_jump_exceeds(
                                previous_candidate[2],
                                left_transform,
                                self.config,
                            )
                        )
                        if candidate_discontinuous:
                            self._tracker_history.clear()
                    # Keep the last trusted pair untouched.  Candidate pairs
                    # live only in history until both wrists remain stable long
                    # enough to be rebased together by the action thread.
                    self._tracker_history.append(
                        (
                            packet_time,
                            right_transform.copy(),
                            left_transform.copy(),
                        )
                    )
                    history_cutoff = packet_time - TRACKER_BASELINE_WINDOW_SECONDS
                    while (
                        len(self._tracker_history) > 3
                        and self._tracker_history[1][0] < history_cutoff
                    ):
                        self._tracker_history.popleft()
                    self._tracker_rebase_requested = False
                    self._tracker_rebase_hold_pending = True
                    if started_fail_closed_recovery:
                        self._tracker_baseline_wait_started_at = packet_time
                elif jump_sides:
                    rejected_bridge_packet = True
                    self._last_tracker_packet_time = packet_time
                    self._tracker_history.clear()
            else:
                if (
                    self.latest_tracker_right is not None
                    and _tracker_pose_jump_exceeds(
                        self.latest_tracker_right,
                        right_transform,
                        self.config,
                    )
                ):
                    jump_sides.append("right")
                if (
                    self.latest_tracker_left is not None
                    and _tracker_pose_jump_exceeds(
                        self.latest_tracker_left,
                        left_transform,
                        self.config,
                    )
                ):
                    jump_sides.append("left")

            packet_gap = bool(
                not bridge_guard_authoritative
                and (
                    self._last_tracker_packet_time is None
                    or packet_time - self._last_tracker_packet_time
                    > TRACKER_REBASE_GAP_SECONDS
                )
            )
            if not rejected_bridge_packet:
                if packet_gap or jump_sides:
                    self._tracker_history.clear()
                if (packet_gap or jump_sides) and self.intervened:
                    # Let the action thread clear and replace its own mapping
                    # baseline while holding the same lock.  Mutating the
                    # baseline here could race a normal mapping operation.
                    self._tracker_rebase_requested = True
                    self._tracker_baseline_wait_started_at = packet_time
                    self._tracker_rebase_hold_pending = True

                self.latest_tracker_right = right_transform
                self.latest_tracker_left = left_transform
                self._last_tracker_packet_time = packet_time
                self._tracker_history.append(
                    (
                        packet_time,
                        right_transform.copy(),
                        left_transform.copy(),
                    )
                )
                history_cutoff = packet_time - TRACKER_BASELINE_WINDOW_SECONDS
                while (
                    len(self._tracker_history) > 1
                    and self._tracker_history[0][0] < history_cutoff
                ):
                    self._tracker_history.popleft()

        if started_fail_closed_recovery:
            sides = " and ".join(jump_sides)
            recovery_requirement = (
                "automatic rebasing is acknowledged from fresh tracker input."
                if getattr(self.config, "auto_rebase_on_tracker_recovery", False)
                else "the original trusted tracker pair returns."
                if getattr(self.config, "preserve_tracker_baseline_on_recovery", False)
                else "a stable two-wrist pair can be rebased safely."
            )
            print(
                f"[Teleop][WARN] Rejected relocated bridge pose on {sides}; "
                f"holding both physical palms until {recovery_requirement}",
                flush=True,
            )
        elif rejected_bridge_packet and not bool(
            getattr(
                self.config,
                "fail_closed_rebase_on_tracker_jump",
                False,
            )
        ):
            sides = " and ".join(jump_sides)
            print(
                f"[Teleop][WARN] Rejected relocated bridge pose on {sides}; "
                "kept the last trusted pair; the next credible packet resumes "
                "immediately.",
                flush=True,
            )
        elif jump_sides and not rejected_bridge_packet:
            sides = " and ".join(jump_sides)
            print(
                f"[Teleop] Tracker relocalization jump detected on {sides}; "
                "absorbing it and rebasing both arms at their physical poses.",
                flush=True,
            )
        return tuple(jump_sides)

    def _strict_tracker_pair_required(self) -> bool:
        return bool(
            getattr(self.config, "require_stable_pair_on_enable", False)
            or getattr(
                self.config,
                "fail_closed_rebase_on_tracker_jump",
                False,
            )
        )

    def _tracker_packet_max_age_seconds(self) -> float:
        return _tracker_timeout_seconds(self.config)

    def _stable_tracker_pair_locked(self) -> dict[str, np.ndarray] | None:
        """Return the latest pair only after both trackers stay stable."""

        history = getattr(self, "_tracker_history", None)
        strict_pair = self._strict_tracker_pair_required()
        minimum_samples = TRACKER_BASELINE_MIN_SAMPLES if strict_pair else 2
        if history is None or len(history) < minimum_samples:
            return None
        sample_times = np.asarray(
            [float(sample[0]) for sample in history],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(sample_times)):
            return None
        if sample_times[-1] - sample_times[0] < TRACKER_BASELINE_WINDOW_SECONDS * 0.9:
            return None
        if strict_pair:
            sample_gaps = np.diff(sample_times)
            if np.any(sample_gaps <= 0.0) or np.any(
                sample_gaps > TRACKER_BASELINE_MAX_SAMPLE_GAP_SECONDS
            ):
                return None
            latest_age = max(0.0, time.monotonic() - float(sample_times[-1]))
            if latest_age > self._tracker_packet_max_age_seconds():
                return None

        latest_right = history[-1][1]
        latest_left = history[-1][2]
        for index, latest in ((1, latest_right), (2, latest_left)):
            max_position_spread = max(
                float(np.linalg.norm(sample[index][:3, 3] - latest[:3, 3]))
                for sample in history
            )
            max_rotation_spread = max(
                _rotation_angle_deg(
                    sample[index][:3, :3] @ latest[:3, :3].T
                )
                for sample in history
            )
            if (
                max_position_spread > TRACKER_BASELINE_POS_SPREAD_M
                or max_rotation_spread > TRACKER_BASELINE_ROT_SPREAD_DEG
            ):
                return None
        return {
            "right": latest_right.copy(),
            "left": latest_left.copy(),
        }

    def _recovered_tracker_pair_locked(self) -> dict[str, np.ndarray] | None:
        """Return a stable pair only when it matches the last trusted pair."""

        stable_pair = self._stable_tracker_pair_locked()
        if stable_pair is None:
            return None
        trusted = {
            "right": self.latest_tracker_right,
            "left": self.latest_tracker_left,
        }
        for side in ("right", "left"):
            previous = trusted[side]
            candidate = stable_pair[side]
            if previous is None:
                continue
            if bool(
                getattr(self.config, "preserve_tracker_baseline_on_recovery", False)
            ):
                if _tracker_pose_jump_exceeds(previous, candidate, self.config):
                    return None
                continue
            position_delta = float(
                np.linalg.norm(candidate[:3, 3] - previous[:3, 3])
            )
            rotation_delta = _rotation_angle_deg(
                candidate[:3, :3] @ previous[:3, :3].T
            )
            if (
                position_delta > TRACKER_RECONNECT_MAX_POSITION_DELTA_M
                or rotation_delta > TRACKER_RECONNECT_MAX_ROTATION_DELTA_DEG
            ):
                return None
        return stable_pair

    def _physical_palm_hold_action(self) -> dict[str, np.ndarray]:
        """Latch the measured palms once and cancel the old Cartesian goals."""

        if getattr(self, "_tracker_fault_hold_pose", None) is None:
            right_pose, left_pose = _dual_teleop_start_poses(self.env)
            self._tracker_fault_hold_pose = {
                "right": right_pose.copy(), "left": left_pose.copy(),
            }
        right_pose = self._tracker_fault_hold_pose["right"]
        left_pose = self._tracker_fault_hold_pose["left"]
        self._clear_live_velocity_hint()
        self._tracker_hold_action_this_frame = True
        self._filtered_target_pose = {
            "right": right_pose.copy(),
            "left": left_pose.copy(),
        }
        return {
            "right": np.concatenate(
                [
                    right_pose[:3, 3],
                    _quat_wxyz_from_matrix(right_pose[:3, :3]),
                ],
                axis=0,
            ),
            "left": np.concatenate(
                [
                    left_pose[:3, 3],
                    _quat_wxyz_from_matrix(left_pose[:3, :3]),
                ],
                axis=0,
            ),
        }

    def _recv_vive_loop(self):
        while not self._stop_tracker_thread:
            try:
                data, _ = _recv_latest_udp_datagram(self._vive_socket, 2048)
                if self._strict_tracker_pair_required() and len(data) < 24 * 8:
                    # A one-wrist packet cannot refresh an atomic two-wrist
                    # stream by copying the same transform onto both sides.
                    with self._tracker_lock:
                        _queue_tracker_hold(self, time.monotonic())
                        self._tracker_history.clear()
                    continue
                if len(data) < 12 * 8:
                    continue
                if len(data) >= 24 * 8:
                    arr = np.frombuffer(data, dtype=np.float64, count=24)
                    right_pose = arr[:12].reshape(3, 4)
                    left_pose = arr[12:24].reshape(3, 4)
                else:
                    pose = np.frombuffer(data, dtype=np.float64, count=12).reshape(3, 4)
                    right_pose = pose
                    left_pose = pose

                right_transform = np.eye(4, dtype=np.float64)
                right_transform[:3, :] = right_pose
                left_transform = np.eye(4, dtype=np.float64)
                left_transform[:3, :] = left_pose
                self._accept_tracker_pair(
                    right_transform,
                    left_transform,
                    packet_time=time.monotonic(),
                )
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

    def _accept_hand_packet(
        self,
        side: str,
        data: bytes,
        *,
        packet_time: float,
    ) -> bool:
        """Atomically publish one dimensionally valid live hand packet."""

        if side not in {"right", "left"}:
            raise ValueError(f"Unknown hand side: {side!r}")
        parsed = _parse_udp_hand_angles(
            data,
            expected_dim=self._hand_action_dim,
        )
        if parsed is None:
            return False
        with self._hand_lock:
            setattr(self, f"latest_allegro_angles_{side}", parsed)
            setattr(self, f"_last_hand_packet_time_{side}", float(packet_time))
        return True

    def _recv_hand_loop(self):
        while not self._stop_hand_thread:
            try:
                right_data, _ = _recv_latest_udp_datagram(
                    self._right_hand_socket,
                    4096,
                )
                if right_data:
                    self._accept_hand_packet(
                        "right",
                        right_data,
                        packet_time=time.monotonic(),
                    )
            except socket.timeout:
                pass
            except Exception:
                time.sleep(0.01)

            try:
                left_data, _ = _recv_latest_udp_datagram(
                    self._left_hand_socket,
                    4096,
                )
                if left_data:
                    self._accept_hand_packet(
                        "left",
                        left_data,
                        packet_time=time.monotonic(),
                    )
            except socket.timeout:
                pass
            except Exception:
                time.sleep(0.01)

    def glfw_on_key(self, window, key, scancode, action, mods):
        del window
        del scancode
        del mods

        if action == glfw.PRESS and key == glfw.KEY_SEMICOLON:
            self.set_intervened(not self.intervened)

        if action == glfw.PRESS and key == glfw.KEY_R:
            self.reset_trigger = True

        if action == glfw.PRESS and key == glfw.KEY_SPACE:
            self._guidance_route_complete_requested = True

        if action == glfw.PRESS and key == glfw.KEY_Z:
            self._manual_tracker_rebase_requested = True

    def _vive_action(self):
        if not self.intervened or self.ee_start is None:
            return None

        with self._tracker_lock:
            now = time.monotonic()
            preserve_baseline = bool(
                getattr(self, "_rokae_wuji_teleop", False)
                or self.config.preserve_tracker_baseline_on_recovery
            )
            strict_pair = self._strict_tracker_pair_required()
            last_packet_time = getattr(self, "_last_tracker_packet_time", None)
            stale_timeout = (
                self._tracker_packet_max_age_seconds()
                if strict_pair
                else TRACKER_REBASE_GAP_SECONDS
            )
            if (
                (strict_pair or not preserve_baseline)
                and last_packet_time is not None
                and now - last_packet_time > stale_timeout
                and not getattr(self, "_tracker_rebase_hold_pending", False)
            ):
                # Do not wait for the first reconnect packet to discover a
                # dropout.  Cancel the currently latched Cartesian goal at the
                # physical palms as soon as the live pair goes stale.
                if not preserve_baseline:
                    self.tracker_start_world = None
                self._tracker_rebase_requested = False
                self._tracker_rebase_hold_pending = True
                self._tracker_fault_hold_pose = None
                self._tracker_baseline_wait_started_at = now
                history = getattr(self, "_tracker_history", None)
                if history is not None:
                    history.clear()
                if strict_pair:
                    recovery_requirement = (
                        "automatic rebasing is acknowledged from fresh tracker input."
                        if getattr(self.config, "auto_rebase_on_tracker_recovery", False)
                        else "the original trusted tracker pair returns."
                        if getattr(
                            self.config, "preserve_tracker_baseline_on_recovery", False
                        )
                        else "a fresh stable two-wrist pair can be rebased safely."
                    )
                    print(
                        "[Teleop][WARN] Tracker pair is stale; holding both "
                        f"physical palms until {recovery_requirement}",
                        flush=True,
                    )
            if (
                preserve_baseline
                and getattr(self, "_tracker_rebase_hold_pending", False)
            ):
                if getattr(self.config, "auto_rebase_on_tracker_recovery", False):
                    return self._physical_palm_hold_action()
                preserve_recovery_baseline = bool(
                    getattr(
                        self.config,
                        "preserve_tracker_baseline_on_recovery",
                        False,
                    )
                )
                recovered_pair = (
                    self._stable_tracker_pair_locked()
                    if strict_pair and not preserve_recovery_baseline
                    else self._recovered_tracker_pair_locked()
                )
                # The explicit legacy rebase option retains its bounded
                # recovery wait. Collection defaults never adopt a relocated
                # candidate on stability or elapsed time alone.
                recovery_timed_out = False
                if recovered_pair is None and not preserve_recovery_baseline:
                    baseline_started = getattr(
                        self,
                        "_tracker_baseline_wait_started_at",
                        None,
                    )
                    recovery_timed_out = bool(
                        baseline_started is not None
                        and now - float(baseline_started)
                        >= TRACKER_BASELINE_MAX_WAIT_SECONDS
                    )
                    if recovery_timed_out:
                        history = getattr(self, "_tracker_history", None)
                        if history:
                            candidate = history[-1]
                            candidate_right = np.asarray(
                                candidate[1], dtype=np.float64
                            )
                            candidate_left = np.asarray(
                                candidate[2], dtype=np.float64
                            )
                            if np.all(np.isfinite(candidate_right)) and np.all(
                                np.isfinite(candidate_left)
                            ):
                                recovered_pair = {
                                    "right": candidate_right.copy(),
                                    "left": candidate_left.copy(),
                                }
                                print(
                                    "[Teleop][WARN] Stable two-wrist recovery "
                                    f"timed out after {TRACKER_BASELINE_MAX_WAIT_SECONDS:.1f}s; "
                                    "using the newest finite pair and rebasing "
                                    "at the physical palms.",
                                    flush=True,
                                )
                if recovered_pair is None:
                    return self._physical_palm_hold_action()

                self.latest_tracker_right = recovered_pair["right"]
                self.latest_tracker_left = recovered_pair["left"]
                self._tracker_rebase_hold_pending = False
                self._tracker_rebase_requested = False
                self._tracker_baseline_wait_started_at = None
                self._tracker_history.clear()
                if self.tracker_start_world is None or (
                    (strict_pair or recovery_timed_out) and not preserve_recovery_baseline
                ):
                    # A relocation is adopted only after both tracker poses are
                    # jointly stable.  Rebase them at the palms' current
                    # physical poses so the newly adopted coordinate frame
                    # cannot create an arm command discontinuity.
                    right_pose, left_pose = _dual_teleop_start_poses(self.env)
                    self.tracker_start_world = {
                        side: pose.copy()
                        for side, pose in recovered_pair.items()
                    }
                    self.ee_start = {
                        "right": right_pose.copy(),
                        "left": left_pose.copy(),
                    }
                    self.hold_pose = {
                        "right": right_pose.copy(),
                        "left": left_pose.copy(),
                    }
                    self._filtered_target_pose = {
                        "right": right_pose.copy(),
                        "left": left_pose.copy(),
                    }
                    if recovery_timed_out:
                        print(
                            "[Teleop] Timeout fallback rebase accepted; both "
                            "trackers now map from the current physical palms.",
                            flush=True,
                        )
                    else:
                        print(
                            "[Teleop] Stable two-wrist recovery accepted; rebased "
                            "both trackers at the current physical palms.",
                            flush=True,
                        )
                elif preserve_recovery_baseline:
                    print(
                        "[Teleop] Tracker pair returned near the last trusted "
                        "poses; original episode baseline preserved.",
                        flush=True,
                    )
                # Submit the physical hold target once more at the recovery
                # boundary.  The next action resumes from the trusted mapping.
                hold_action = self._physical_palm_hold_action()
                self._tracker_fault_hold_pose = None
                return hold_action
            if (
                self.latest_tracker_right is None
                or self.latest_tracker_left is None
            ):
                if strict_pair:
                    self._tracker_rebase_requested = False
                    self._tracker_rebase_hold_pending = True
                    if self._tracker_baseline_wait_started_at is None:
                        self._tracker_baseline_wait_started_at = now
                    return self._physical_palm_hold_action()
                return None
            tracker_right = self.latest_tracker_right.copy()
            tracker_left = self.latest_tracker_left.copy()
            if getattr(self, "_tracker_rebase_requested", False):
                self.tracker_start_world = None
                self._tracker_rebase_requested = False
            if self.tracker_start_world is None:
                rebase_hold_pending = bool(
                    getattr(self, "_tracker_rebase_hold_pending", False)
                )
                stable_pair = self._stable_tracker_pair_locked()
                if stable_pair is None:
                    if rebase_hold_pending:
                        right_pose, left_pose = _dual_teleop_start_poses(self.env)
                        self.ee_start = {
                            "right": right_pose.copy(),
                            "left": left_pose.copy(),
                        }
                        self._filtered_target_pose = {
                            "right": right_pose.copy(),
                            "left": left_pose.copy(),
                        }
                        return {
                            "right": np.concatenate(
                                [
                                    right_pose[:3, 3],
                                    _quat_wxyz_from_matrix(right_pose[:3, :3]),
                                ],
                                axis=0,
                            ),
                            "left": np.concatenate(
                                [
                                    left_pose[:3, 3],
                                    _quat_wxyz_from_matrix(left_pose[:3, :3]),
                                ],
                                axis=0,
                            ),
                        }
                    return self._zero_pose_action()

                self.tracker_start_world = stable_pair
                self._tracker_baseline_wait_started_at = None
                self._tracker_rebase_hold_pending = False
                right_pose, left_pose = _dual_teleop_start_poses(self.env)
                self.ee_start = {
                    "right": right_pose.copy(),
                    "left": left_pose.copy(),
                }
                self._filtered_target_pose = {
                    "right": right_pose.copy(),
                    "left": left_pose.copy(),
                }
                if rebase_hold_pending:
                    # Overwrite an Isaac/DLS or MuJoCo Cartesian servo's old
                    # latched goal.  A zero pose action would leave that goal
                    # active and the arm could keep travelling after rebasing.
                    return {
                        "right": np.concatenate(
                            [
                                right_pose[:3, 3],
                                _quat_wxyz_from_matrix(right_pose[:3, :3]),
                            ],
                            axis=0,
                        ),
                        "left": np.concatenate(
                            [
                                left_pose[:3, 3],
                                _quat_wxyz_from_matrix(left_pose[:3, :3]),
                            ],
                            axis=0,
                        ),
                    }
                return self._zero_pose_action()

        if self._rokae_wuji_teleop and self.direct_hand_targets:
            right_pos_delta = np.linalg.norm(
                tracker_right[:3, 3] - self.tracker_start_world["right"][:3, 3]
            )
            left_pos_delta = np.linalg.norm(
                tracker_left[:3, 3] - self.tracker_start_world["left"][:3, 3]
            )
            right_rot_delta = _rotation_angle_deg(
                tracker_right[:3, :3] @ self.tracker_start_world["right"][:3, :3].T
            )
            left_rot_delta = _rotation_angle_deg(
                tracker_left[:3, :3] @ self.tracker_start_world["left"][:3, :3].T
            )
            fail_closed_excursion = bool(
                getattr(
                    self.config,
                    "fail_closed_on_tracker_excursion",
                    False,
                )
            )
            maximum_position = getattr(
                self.config,
                "max_tracker_delta_m",
                None,
            )
            maximum_rotation = getattr(
                self.config,
                "max_tracker_rotation_delta_deg",
                None,
            )
            excursion_sides: list[str] = []
            for side, position, rotation in (
                ("right", float(right_pos_delta), float(right_rot_delta)),
                ("left", float(left_pos_delta), float(left_rot_delta)),
            ):
                position_exceeded = bool(
                    maximum_position is not None
                    and float(maximum_position) > 0.0
                    and position > float(maximum_position)
                )
                rotation_exceeded = bool(
                    maximum_rotation is not None
                    and float(maximum_rotation) > 0.0
                    and rotation > float(maximum_rotation)
                )
                if position_exceeded or rotation_exceeded:
                    excursion_sides.append(side)
            if fail_closed_excursion and excursion_sides:
                packet_time = self._last_tracker_packet_time or time.monotonic()
                with self._tracker_lock:
                    self._tracker_history.clear()
                    self._tracker_history.append(
                        (
                            packet_time,
                            tracker_right.copy(),
                            tracker_left.copy(),
                        )
                    )
                    self._tracker_rebase_requested = False
                    self._tracker_rebase_hold_pending = True
                    self._tracker_baseline_wait_started_at = packet_time
                details = "; ".join(
                    (
                        f"{side}: pos={position:.3f}m, rot={rotation:.1f}deg"
                    )
                    for side, position, rotation in (
                        ("right", float(right_pos_delta), float(right_rot_delta)),
                        ("left", float(left_pos_delta), float(left_rot_delta)),
                    )
                    if side in excursion_sides
                )
                position_limit = (
                    "disabled"
                    if maximum_position is None
                    else f"{float(maximum_position):.3f}m"
                )
                rotation_limit = (
                    "disabled"
                    if maximum_rotation is None
                    else f"{float(maximum_rotation):.1f}deg"
                )
                print(
                    "[Teleop][WARN] Tracker excursion rejected "
                    f"({details}; limits pos={position_limit}, "
                    f"rot={rotation_limit}); holding both physical palms "
                    "until a stable two-wrist rebase.",
                    flush=True,
                )
                return self._physical_palm_hold_action()
            if (
                right_pos_delta <= TRACKER_HOLD_POS_EPS_M
                and left_pos_delta <= TRACKER_HOLD_POS_EPS_M
                and right_rot_delta <= TRACKER_HOLD_ROT_EPS_DEG
                and left_rot_delta <= TRACKER_HOLD_ROT_EPS_DEG
            ):
                return self._zero_pose_action()

        if self.direct_hand_targets:
            target_right = _target_pose_from_bridge_world_pose(
                self.ee_start["right"],
                self.tracker_start_world["right"],
                tracker_right,
                1.0,
                max_tracker_delta_m=self.config.max_tracker_delta_m,
                max_tracker_rotation_delta_deg=(
                    self.config.max_tracker_rotation_delta_deg
                ),
            )
            target_left = _target_pose_from_bridge_world_pose(
                self.ee_start["left"],
                self.tracker_start_world["left"],
                tracker_left,
                1.0,
                max_tracker_delta_m=self.config.max_tracker_delta_m,
                max_tracker_rotation_delta_deg=(
                    self.config.max_tracker_rotation_delta_deg
                ),
            )
        elif self._rokae_wuji_teleop:
            target_right = _target_pose_from_bridge_world_pose(
                self.ee_start["right"],
                self.tracker_start_world["right"],
                tracker_right,
                self.config.pose_scale,
                max_tracker_delta_m=self.config.max_tracker_delta_m,
                max_tracker_rotation_delta_deg=(
                    self.config.max_tracker_rotation_delta_deg
                ),
            )
            target_left = _target_pose_from_bridge_world_pose(
                self.ee_start["left"],
                self.tracker_start_world["left"],
                tracker_left,
                self.config.pose_scale,
                max_tracker_delta_m=self.config.max_tracker_delta_m,
                max_tracker_rotation_delta_deg=(
                    self.config.max_tracker_rotation_delta_deg
                ),
            )
        else:
            target_right = _target_pose_from_tracker_pose(
                self.ee_start["right"],
                self.tracker_start_world["right"],
                tracker_right,
                self.config.pose_scale,
                self.target_frame,
            )
            target_left = _target_pose_from_tracker_pose(
                self.ee_start["left"],
                self.tracker_start_world["left"],
                tracker_left,
                self.config.pose_scale,
                self.target_frame,
            )

        previous_targets = {
            side: pose.copy()
            for side, pose in self._filtered_target_pose.items()
            if pose is not None
        }
        target_right = _filter_target_pose(
            self._filtered_target_pose.get("right"),
            target_right,
            self.config,
        )
        target_left = _filter_target_pose(
            self._filtered_target_pose.get("left"),
            target_left,
            self.config,
        )
        self._filtered_target_pose = {
            "right": target_right.copy(),
            "left": target_left.copy(),
        }
        self._publish_live_velocity_hint(
            previous_targets,
            {"right": target_right, "left": target_left},
        )

        now = time.monotonic()
        if now - self._last_tracker_motion_log >= 2.0:
            pos_delta_right = np.linalg.norm(
                tracker_right[:3, 3] - self.tracker_start_world["right"][:3, 3]
            )
            pos_delta_left = np.linalg.norm(
                tracker_left[:3, 3] - self.tracker_start_world["left"][:3, 3]
            )
            tracker_dz_right = (
                tracker_right[2, 3] - self.tracker_start_world["right"][2, 3]
            )
            tracker_dz_left = (
                tracker_left[2, 3] - self.tracker_start_world["left"][2, 3]
            )
            target_dz_right = target_right[2, 3] - self.ee_start["right"][2, 3]
            target_dz_left = target_left[2, 3] - self.ee_start["left"][2, 3]
            tracker_delta_right_xyz = (
                tracker_right[:3, 3] - self.tracker_start_world["right"][:3, 3]
            )
            tracker_delta_left_xyz = (
                tracker_left[:3, 3] - self.tracker_start_world["left"][:3, 3]
            )
            target_delta_right_xyz = (
                target_right[:3, 3] - self.ee_start["right"][:3, 3]
            )
            target_delta_left_xyz = (
                target_left[:3, 3] - self.ee_start["left"][:3, 3]
            )
            rot_delta_right = (
                tracker_right[:3, :3] @ self.tracker_start_world["right"][:3, :3].T
            )
            rot_delta_left = (
                tracker_left[:3, :3] @ self.tracker_start_world["left"][:3, :3].T
            )
            target_rot_delta_right = (
                target_right[:3, :3] @ self.ee_start["right"][:3, :3].T
            )
            target_rot_delta_left = (
                target_left[:3, :3] @ self.ee_start["left"][:3, :3].T
            )
            palm_axis_right = self.tracker_start_world["right"][:3, 0]
            palm_axis_left = self.tracker_start_world["left"][:3, 0]
            palm_rot_right = _rotation_component_deg(rot_delta_right, palm_axis_right)
            palm_rot_left = _rotation_component_deg(rot_delta_left, palm_axis_left)
            target_palm_rot_right = _rotation_component_deg(
                target_rot_delta_right,
                palm_axis_right,
            )
            target_palm_rot_left = _rotation_component_deg(
                target_rot_delta_left,
                palm_axis_left,
            )
            print(
                "tracker delta: "
                f"right pos={pos_delta_right:.3f}m "
                f"xyz=({tracker_delta_right_xyz[0]:+.3f},"
                f"{tracker_delta_right_xyz[1]:+.3f},"
                f"{tracker_delta_right_xyz[2]:+.3f})m "
                f"target_xyz=({target_delta_right_xyz[0]:+.3f},"
                f"{target_delta_right_xyz[1]:+.3f},"
                f"{target_delta_right_xyz[2]:+.3f})m "
                f"dz={tracker_dz_right:+.3f}m "
                f"target_dz={target_dz_right:+.3f}m "
                f"palm_rot={palm_rot_right:+.1f}deg "
                f"target_palm_rot={target_palm_rot_right:+.1f}deg "
                f"rot={_rotation_angle_deg(rot_delta_right):.1f}deg; "
                f"left pos={pos_delta_left:.3f}m "
                f"xyz=({tracker_delta_left_xyz[0]:+.3f},"
                f"{tracker_delta_left_xyz[1]:+.3f},"
                f"{tracker_delta_left_xyz[2]:+.3f})m "
                f"target_xyz=({target_delta_left_xyz[0]:+.3f},"
                f"{target_delta_left_xyz[1]:+.3f},"
                f"{target_delta_left_xyz[2]:+.3f})m "
                f"dz={tracker_dz_left:+.3f}m "
                f"target_dz={target_dz_left:+.3f}m "
                f"palm_rot={palm_rot_left:+.1f}deg "
                f"target_palm_rot={target_palm_rot_left:+.1f}deg "
                f"rot={_rotation_angle_deg(rot_delta_left):.1f}deg",
                flush=True,
            )
            self._last_tracker_motion_log = now

        return {
            "right": np.concatenate(
                [target_right[:3, 3], _quat_wxyz_from_matrix(target_right[:3, :3])],
                axis=0,
            ),
            "left": np.concatenate(
                [target_left[:3, 3], _quat_wxyz_from_matrix(target_left[:3, :3])],
                axis=0,
            ),
        }

    def _zero_pose_action(self):
        zero = np.zeros(7, dtype=np.float64)
        return {"right": zero.copy(), "left": zero.copy()}

    def _hold_pose_action(self):
        if self._rokae_wuji_teleop and self.direct_hand_targets:
            return self._zero_pose_action()
        if self.hold_pose is None:
            self._capture_hold_pose()
        right_pose, left_pose = self.hold_pose["right"], self.hold_pose["left"]
        return {
            "right": np.concatenate(
                [right_pose[:3, 3], _quat_wxyz_from_matrix(right_pose[:3, :3])],
                axis=0,
            ),
            "left": np.concatenate(
                [left_pose[:3, 3], _quat_wxyz_from_matrix(left_pose[:3, :3])],
                axis=0,
            ),
        }

    def _capture_hold_pose(self):
        right_pose, left_pose = _dual_teleop_start_poses(self.env)
        self.hold_pose = {
            "right": right_pose.copy(),
            "left": left_pose.copy(),
        }

    def _current_hand_action(self) -> tuple[np.ndarray, np.ndarray]:
        hand = _env_attr(self.env, "_hand")
        data = _env_attr(self.env, "_data")
        side_qpos_ids = getattr(hand, "side_qpos_ids", None)
        if data is not None and isinstance(side_qpos_ids, dict):
            right_ids = side_qpos_ids.get("right")
            left_ids = side_qpos_ids.get("left")
            if right_ids is not None and left_ids is not None:
                return (
                    np.asarray(data.qpos[right_ids], dtype=np.float64).copy(),
                    np.asarray(data.qpos[left_ids], dtype=np.float64).copy(),
                )
        right_hand = _env_attr(self.env, "_right_hand")
        left_hand = _env_attr(self.env, "_left_hand")
        if right_hand is not None and left_hand is not None:
            right = np.asarray(right_hand, dtype=np.float64).reshape(-1)
            left = np.asarray(left_hand, dtype=np.float64).reshape(-1)
            if (
                right.size == self._hand_action_dim
                and left.size == self._hand_action_dim
                and np.all(np.isfinite(right))
                and np.all(np.isfinite(left))
            ):
                return right.copy(), left.copy()
        return (
            np.zeros(self._hand_action_dim, dtype=np.float64),
            np.zeros(self._hand_action_dim, dtype=np.float64),
        )

    def _slew_hand_action(
        self,
        right: np.ndarray,
        left: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound live glove targets once per recorded control frame."""

        requested = {
            "right": np.asarray(right, dtype=np.float64).reshape(-1),
            "left": np.asarray(left, dtype=np.float64).reshape(-1),
        }
        configured_step = getattr(self.config, "hand_filter_max_step_rad", None)
        if configured_step is None:
            maximum_step = 0.0
        else:
            try:
                maximum_step = max(0.0, float(configured_step))
            except (TypeError, ValueError):
                maximum_step = 0.0

        takeover_steps_remaining = int(
            max(0, getattr(self, "_hand_takeover_steps_remaining", 0))
        )
        takeover_active = takeover_steps_remaining > 0
        if takeover_active:
            try:
                takeover_step = max(
                    0.0,
                    float(self.config.hand_takeover_max_step_rad or 0.0),
                )
            except (TypeError, ValueError):
                takeover_step = 0.0
            if takeover_step > 0.0:
                maximum_step = takeover_step
            else:
                takeover_active = False
                self._hand_takeover_steps_remaining = 0

        filtered: dict[str, np.ndarray] = {}
        takeover_reached = True
        expected_shape = (int(self._hand_action_dim),)
        for side in ("right", "left"):
            target = requested[side]
            attr = f"_commanded_hand_angles_{side}"
            previous = getattr(self, attr, None)
            if previous is None or np.asarray(previous).shape != expected_shape:
                previous = np.zeros(expected_shape, dtype=np.float64)
            else:
                previous = np.asarray(previous, dtype=np.float64)
            if target.shape != expected_shape:
                target = previous.copy()
            else:
                target = np.where(np.isfinite(target), target, previous)
            if maximum_step > 0.0:
                target = previous + np.clip(
                    target - previous,
                    -maximum_step,
                    maximum_step,
                )
            if takeover_active and target.shape == expected_shape:
                takeover_reached = bool(
                    takeover_reached
                    and float(
                        np.max(np.abs(requested[side] - target))
                    )
                    <= 1e-4
                )
            target = target.astype(np.float64, copy=True)
            setattr(self, attr, target.copy())
            filtered[side] = target
        if takeover_active:
            self._hand_takeover_steps_remaining = max(
                0, takeover_steps_remaining - 1
            )
            if takeover_reached or self._hand_takeover_steps_remaining == 0:
                self._hand_takeover_steps_remaining = 0
                print(
                    "[Teleop] One-shot hand takeover complete; direct glove "
                    "targets active.",
                    flush=True,
                )
        return filtered["right"], filtered["left"]

    def _hand_packet_max_age_seconds(self) -> float | None:
        configured = getattr(self.config, "hand_packet_max_age_s", None)
        if configured is None:
            return None
        try:
            timeout = float(configured)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(timeout) or timeout <= 0.0:
            return None
        return timeout

    def _live_hand_action_locked(self) -> tuple[np.ndarray, np.ndarray]:
        timeout = self._hand_packet_max_age_seconds()
        if timeout is not None:
            now = time.monotonic()
            packet_times = (
                getattr(self, "_last_hand_packet_time_right", None),
                getattr(self, "_last_hand_packet_time_left", None),
            )
            both_fresh = all(
                packet_time is not None
                and np.isfinite(float(packet_time))
                and max(0.0, now - float(packet_time)) <= timeout
                for packet_time in packet_times
            )
            if not both_fresh:
                return (
                    np.asarray(
                        self._commanded_hand_angles_right,
                        dtype=np.float64,
                    ).copy(),
                    np.asarray(
                        self._commanded_hand_angles_left,
                        dtype=np.float64,
                    ).copy(),
                )
        return (
            self.latest_allegro_angles_right.copy(),
            self.latest_allegro_angles_left.copy(),
        )

    def _hand_action(self, action: np.ndarray):
        if self.intervened:
            with self._hand_lock:
                return self._live_hand_action_locked()

        hand_action = (
            action[14:]
            if (action is not None and action.shape[0] > 14)
            else np.array([], dtype=np.float64)
        )
        if hand_action.size == 0 or np.allclose(hand_action, 0.0):
            with self._hand_lock:
                return self._live_hand_action_locked()
        if hand_action.size >= 2 * WUJI_NATIVE_HAND_DIM:
            return (
                np.asarray(hand_action[:WUJI_NATIVE_HAND_DIM], dtype=np.float64),
                np.asarray(
                    hand_action[WUJI_NATIVE_HAND_DIM : 2 * WUJI_NATIVE_HAND_DIM],
                    dtype=np.float64,
                ),
            )
        if hand_action.size >= 2 * HAND_POLICY_DIM:
            return (
                np.asarray(hand_action[:HAND_POLICY_DIM], dtype=np.float64),
                np.asarray(hand_action[HAND_POLICY_DIM : 2 * HAND_POLICY_DIM], dtype=np.float64),
            )
        if hand_action.size >= HAND_POLICY_DIM:
            return (
                np.asarray(hand_action[:HAND_POLICY_DIM], dtype=np.float64),
                np.zeros(self._hand_action_dim, dtype=np.float64),
            )
        return (
            np.zeros(self._hand_action_dim, dtype=np.float64),
            np.zeros(self._hand_action_dim, dtype=np.float64),
        )

    def action(self, action: np.ndarray):
        # Do not leave a previous frame's velocity hint active while tracker
        # data is stale, rebasing, or teleoperation is disabled.
        self._tracker_hold_action_this_frame = False
        self._clear_live_velocity_hint()
        pose_action = self._consume_manual_tracker_rebase_action()
        if pose_action is not None:
            on_rebase = _env_attr(self.env, "on_live_teleop_pose_rebase")
            if callable(on_rebase):
                on_rebase()
        if pose_action is None:
            pose_action = self._vive_action()
        if pose_action is None:
            pose_action = self._hold_pose_action()

        automatic_action = _auto_tracker_rebase_action(self, dual=True)
        if automatic_action is not None:
            pose_action = automatic_action

        hand_right, hand_left = self._hand_action(action)
        hand_right = _condition_b02_wuji_hand_action(self.env, hand_right)
        hand_left = _condition_b02_wuji_hand_action(self.env, hand_left)
        hand_right, hand_left = self._slew_hand_action(hand_right, hand_left)
        teleop_action = {
            "right": np.concatenate([pose_action["right"], hand_right], axis=0),
            "left": np.concatenate([pose_action["left"], hand_left], axis=0),
        }
        teleop_action = _condition_live_teleop_action(self.env, teleop_action)
        # A receive callback may detect a fault while targets are being
        # filtered. Check again at the final action boundary. Task-specific
        # no-op/lock conditioning must not mask the command cancelling an old
        # Cartesian goal (notably YCB08's left-pan lock).
        with self._tracker_lock:
            if self.intervened and getattr(self, "_tracker_rebase_hold_pending", False):
                automatic_action = _auto_tracker_rebase_action_locked(self, dual=True)
                pose_action = automatic_action if automatic_action is not None else self._physical_palm_hold_action()
            if self._tracker_hold_action_this_frame:
                on_rebase = _env_attr(self.env, "on_live_teleop_pose_rebase")
                if callable(on_rebase):
                    on_rebase()
                for side in ("right", "left"):
                    teleop_action[side] = teleop_action[side].copy()
                    teleop_action[side][:7] = pose_action[side]
        return teleop_action

    def step(self, action):
        self._ensure_key_callback()
        sample_started = time.monotonic()
        teleop_action = self.action(action)
        action_sampled = time.monotonic()
        obs, rew, done, truncated, info = self.env.step(teleop_action)
        step_finished = time.monotonic()
        info = dict(info)
        info["intervene_action"] = teleop_action
        route_complete = bool(self._guidance_route_complete_requested)
        self._guidance_route_complete_requested = False
        route_complete = (
            _consume_guidance_route_complete_request(self.env) or route_complete
        )
        if route_complete:
            info["guidance_route_complete"] = True
        with self._tracker_lock:
            packet_time = self._last_tracker_packet_time
        info["teleop_timing"] = {
            "tracker_packet_age_ms": (
                None
                if packet_time is None
                else float(max(0.0, action_sampled - packet_time) * 1000.0)
            ),
            "action_sampling_ms": float((action_sampled - sample_started) * 1000.0),
            "environment_step_ms": float((step_finished - action_sampled) * 1000.0),
        }
        if self.reset_trigger:
            info["manual_reset"] = True
            self.reset_trigger = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        _cancel_auto_tracker_rebase(self)
        resume_intervention = bool(self.intervened)
        obs, info = self.env.reset(**kwargs)
        self._ensure_key_callback()
        self.intervened = False
        self._tracker_fault_hold_pose = None
        self.reset_trigger = False
        self._guidance_route_complete_requested = False
        self.tracker_start_world = None
        self._tracker_baseline_wait_started_at = None
        self._tracker_rebase_requested = False
        self._tracker_rebase_hold_pending = False
        self._manual_tracker_rebase_requested = False
        self.ee_start = None
        self._filtered_target_pose = {"right": None, "left": None}
        self._clear_live_velocity_hint()
        self._establish_episode_tracker_baseline()
        self._capture_hold_pose()
        with self._hand_lock:
            self._last_hand_packet_time_right = None
            self._last_hand_packet_time_left = None
            (
                self.latest_allegro_angles_right,
                self.latest_allegro_angles_left,
            ) = self._current_hand_action()
            self._commanded_hand_angles_right = (
                self.latest_allegro_angles_right.copy()
            )
            self._commanded_hand_angles_left = (
                self.latest_allegro_angles_left.copy()
            )
        if resume_intervention:
            try:
                reset_delay = max(0.0, float(self.config.reset_rebase_delay_seconds))
            except (TypeError, ValueError):
                reset_delay = RESET_REBASE_DELAY_SECONDS
            if reset_delay > 0.0:
                print(f"[Teleop] Reset pause: hold for {reset_delay:.1f}s "
                      "to reposition the gloves.", flush=True)
                time.sleep(reset_delay)
            self.set_intervened(True, start_delay=False)
        return obs, info

    def close(self):
        _cancel_auto_tracker_rebase(self)
        if bool(getattr(self, "_rokae_wuji_teleop", False)):
            release_tracker_episode_guard()
        self._stop_tracker_thread = True
        self._stop_hand_thread = True
        if self._vive_thread.is_alive():
            self._vive_thread.join(timeout=1.0)
        if self._hand_thread.is_alive():
            self._hand_thread.join(timeout=1.0)
        for sock in (
            self._vive_socket,
            self._right_hand_socket,
            self._left_hand_socket,
        ):
            try:
                sock.close()
            except Exception:
                pass
        self.env.close()
