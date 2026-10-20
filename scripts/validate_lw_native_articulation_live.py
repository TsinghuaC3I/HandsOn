#!/usr/bin/env python3
"""Fail-closed Isaac Sim 5.1 stability audit for the LW native tasks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any


TASK_CONFIG_MODULES = {
    "g05_folding_fan_bimanual_open": (
        "handson.tasks.g05_folding_fan_bimanual_open.config"
    ),
    "t18_open_micropipette_box": (
        "handson.tasks.t18_open_micropipette_box.config"
    ),
    "t49_unfold_ruler": (
        "handson.tasks.t49_unfold_ruler.config"
    ),
}
EXPECTED_TOPOLOGY = {
    "g05_folding_fan_bimanual_open": (20, 23, 19, True),
    "t18_open_micropipette_box": (2, 25, 1, True),
    "t49_unfold_ruler": (3, 7, 2, False),
}
TASKDEV_DIRS = {
    name: name for name in TASK_CONFIG_MODULES
}
G08_TASK_NAME = "t18_open_micropipette_box"
G10_TASK_NAME = "t49_unfold_ruler"
G10_DYNAMICS_POLICY = "g10_free_root_passive_hinge_damping_v2"
G10_LINK_LINEAR_DAMPING = 2.0
G10_LINK_ANGULAR_DAMPING = 8.0
G10_JOINT_FRICTION = 0.03
G10_DRIVE_STIFFNESS = 0.0
G10_DRIVE_DAMPING = 0.05
G10_DRIVE_MAX_FORCE_NM = 0.15
G10_DRIVE_TARGET_VELOCITY_DEG_S = 0.0
# Keep the live composed-stage assertion tied to the same contract constant
# used by the runtime scene.  A literal here previously drifted to -60° and
# made a correctly composed -90° lid look invalid to the validator.
G08_LID_HARD_STOP_DEG = 90.0
MAX_Q_DRIFT_RAD = math.radians(0.5)
MAX_ABS_QD_RAD_S = 0.08
MAX_ROOT_TRANSLATION_M = 0.002
MAX_ROOT_ROTATION_DEG = 1.0
G08_FIRST_COMMAND_LIFT_M = 0.25
G08_FIRST_COMMAND_MAX_JOINT_TARGET_STEP_RAD = 0.0305
G08_FIRST_COMMAND_MAX_PALM_DISPLACEMENT_M = 0.060
G08_FIRST_COMMAND_MAX_LID_MOTION_RAD = math.radians(0.5)
G08_FIRST_COMMAND_MAX_LID_SPEED_RAD_S = 0.20
G08_LID_PROBE_INITIAL_ANGLE_RAD = math.radians(-5.0)
G08_LID_PROBE_INITIAL_SPEED_RAD_S = -1.0
G08_LID_PROBE_CONTROL_STEPS = 60
G08_LID_PROBE_MAX_OPEN_ANGLE_RAD = math.radians(15.0)
G08_LID_PROBE_FINAL_OPEN_ANGLE_RAD = math.radians(2.0)
G08_LID_PROBE_MAX_OPENING_SPEED_RAD_S = 1.25
G08_EXPECTED_ASSET_SCALE = 0.40
G08_ROOT_NOMINAL_X_M = -0.280
G08_ROOT_NOMINAL_Y_M = 0.0
G08_ROOT_RANDOM_X_HALF_RANGE_M = 0.120
G08_ROOT_RANDOM_Y_HALF_RANGE_M = 0.120
G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG = 30.0


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_ready(value: Any) -> Any:
    """Normalize evidence to strict JSON, including nested NumPy values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_ready(evidence),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short").splitlines(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _origin_summary(info: dict[str, Any]) -> dict[str, Any]:
    origin = dict(info.get("physical_recording_origin") or {})
    keys = (
        "strategy",
        "stability_metric",
        "physics_steps",
        "physics_dt",
        "initial_settle_steps",
        "reset_max_translation_drift_m",
        "reset_max_rotation_drift_rad",
        "reset_max_coordinate_drift",
        "reset_drift_all_physics_frames_sampled",
        "initial_geometry_gate_passed",
        "initialization_contact_summary",
        "active_physics_layout_gate_final_exact_match",
        "gate_final_coordinate_exact_match",
        "task_readiness_required",
        "final_task_readiness",
        "task_recording_origin_acceptance",
        # Preserve the detailed fallback/static-pose gate evidence.  The
        # compact summary is written into taskdev evidence and must be enough
        # to audit the skill's initial-state contract without reopening a
        # transient child process log.
        "stability_sample_seconds",
        "stable_steps_required",
        "stable_steps_observed",
        "timeout_seconds",
        "linear_velocity_limit",
        "angular_velocity_limit",
        "joint_velocity_limit",
        "pose_translation_limit",
        "pose_rotation_limit",
        "gate_pose_samples",
        "gate_static_pose_fallback_samples",
        "gate_max_fallback_translation_delta",
        "gate_max_fallback_rotation_delta",
        "gate_max_task_translation_delta",
        "gate_max_task_rotation_delta",
        "gate_task_velocity_stable",
        "gate_task_static_pose_stable",
        "gate_robot_velocity_stable",
        "initial_geometry_gate_passed",
        "reset_pose_drift_checked",
        "reset_drift_sample_count",
        "reset_drift_expected_sample_count",
    )
    return {key: origin.get(key) for key in keys}


def _observation_summary(
    env: Any,
    observation: Any,
    *,
    phase: str,
) -> dict[str, list[int]]:
    """Require an in-space, finite observation and return its leaf shapes."""

    import numpy as np

    if not env.observation_space.contains(observation):
        raise RuntimeError(f"{phase} observation is outside its declared space")
    shapes: dict[str, list[int]] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                visit(item, f"{path}.{key}" if path else str(key))
            return
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise RuntimeError(
                f"{phase} observation leaf {path!r} is non-numeric: "
                f"{array.dtype}"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"{phase} observation leaf {path!r} is non-finite"
            )
        shapes[path] = [int(size) for size in array.shape]

    visit(observation, "")
    if not shapes:
        raise RuntimeError(f"{phase} observation has no numeric leaves")
    return shapes


def _truth_payload(raw: Any) -> dict[str, Any]:
    truth = raw._read_lw_truth()
    return {
        "q_rad": truth["q"],
        "qd_rad_s": truth["qd"],
        "dof_available": bool(truth["dof_available"]),
        "assigned_contact_available": bool(truth["contact_available"]),
        "left_force_n": float(truth["left_force"]),
        "right_force_n": float(truth["right_force"]),
        "support_available": bool(truth["support_available"]),
        "support_active": bool(truth["support_active"]),
        "support_force_n": float(truth["support_force"]),
        "support_contact_count": int(truth["support_contact_count"]),
        "support_min_separation_m": float(
            truth["support_min_separation_m"]
        ),
        "support_separation_valid": bool(truth["support_separation_valid"]),
        "support_clearance_m": float(truth["support_clearance_m"]),
        "support_clearance_valid": bool(truth["support_clearance_valid"]),
        "root_pose_wxyz": truth["root_pose"],
        "root_translation_error_m": float(
            truth["root_translation_error_m"]
        ),
        "root_rotation_error_deg": float(truth["root_rotation_error_deg"]),
    }


def _reset_and_accept_origin(env: Any, raw: Any, seed: int) -> dict[str, Any]:
    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin,
        stable_recording_origin_kwargs_for_task,
    )

    observation, info = env.reset(seed=seed)
    reset_observation = _observation_summary(
        env,
        observation,
        phase=f"seed {seed} reset",
    )
    authored_sample = raw._lw_reset_sample.as_dict()
    observation, info = stabilize_recording_origin(
        env,
        observation,
        info,
        **stable_recording_origin_kwargs_for_task(raw.task_name),
    )
    accepted_observation = _observation_summary(
        env,
        observation,
        phase=f"seed {seed} accepted recording origin",
    )
    if not raw._lw_origin_accepted:
        raise RuntimeError("stable recording origin was not accepted")
    summary = _origin_summary(info)
    readiness = summary.get("final_task_readiness")
    if not isinstance(readiness, dict) or readiness.get("ready") is not True:
        raise RuntimeError(f"final task readiness failed: {readiness!r}")
    return {
        "seed": seed,
        "authored_sample": authored_sample,
        "reset_observation_shapes": reset_observation,
        "accepted_observation_shapes": accepted_observation,
        "accepted_truth": _truth_payload(raw),
        "recording_origin": summary,
    }


def _audit_int(audit: dict[str, Any], key: str) -> int:
    try:
        return int(audit[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"invalid substep audit field {key!r}: {audit!r}") from exc


def _audit_float(audit: dict[str, Any], key: str) -> float:
    try:
        value = float(audit[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"invalid substep audit field {key!r}: {audit!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite substep audit field {key!r}: {audit!r}")
    return value


def _audit_range(audit: dict[str, Any], key: str) -> tuple[float, float]:
    values = audit.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise RuntimeError(f"invalid substep audit range {key!r}: {audit!r}")
    try:
        lower, upper = (float(values[0]), float(values[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"invalid substep audit range {key!r}: {audit!r}") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise RuntimeError(f"invalid substep audit range {key!r}: {audit!r}")
    return lower, upper


def _require_fresh_post_accept_substep_audits(raw: Any) -> None:
    audit = dict(raw._post_accept_substep_audit_snapshot())
    if _audit_int(audit, "sample_count") != 0:
        raise RuntimeError(
            "post-accept substep audit was not cleared at origin acceptance: "
            f"{audit!r}"
        )
    if raw.task_name == G08_TASK_NAME:
        support = dict(raw._g08_support_substep_audit_snapshot())
        if _audit_int(support, "sample_count") != 0:
            raise RuntimeError(
                "G08 support substep audit was not cleared at origin "
                f"acceptance: {support!r}"
            )


def _validate_post_accept_substep_audits(
    raw: Any,
    expected_sample_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    audit = dict(raw._post_accept_substep_audit_snapshot())
    sample_count = _audit_int(audit, "sample_count")
    if sample_count != expected_sample_count:
        raise RuntimeError(
            "post-accept physics-frame audit count mismatch: "
            f"{sample_count} != {expected_sample_count}"
        )
    unreadable_count = _audit_int(audit, "unreadable_count")
    if bool(audit.get("truth_unreadable")) or unreadable_count != 0:
        raise RuntimeError(
            "post-accept truth became unreadable during a physics frame: "
            f"{audit!r}"
        )

    max_q_drift = _audit_float(audit, "max_abs_q_error_rad")
    max_abs_qd = _audit_float(audit, "max_abs_qd_rad_s")
    max_root_translation = _audit_float(
        audit,
        "max_root_translation_error_m",
    )
    max_root_rotation = _audit_float(audit, "max_root_rotation_error_deg")
    if max_q_drift > MAX_Q_DRIFT_RAD:
        raise RuntimeError(f"DOF drift exceeded 0.5 deg: {max_q_drift}")
    if max_abs_qd > MAX_ABS_QD_RAD_S:
        raise RuntimeError(f"DOF speed exceeded 0.08 rad/s: {max_abs_qd}")
    # G05/G08 are fixed-root tasks and must remain inside their reset-pose
    # envelope during the post-accept zero-action audit. G10 deliberately has
    # a free tabletop root, so ordinary translation/rotation is not a failure;
    # its contract still latches a terminal state for a sustained gross escape.
    contract_spec = getattr(raw, "_lw_contract_spec", None)
    root_pose_constraint = bool(
        getattr(contract_spec, "root_pose_constraint", True)
    )
    if root_pose_constraint:
        if max_root_translation > MAX_ROOT_TRANSLATION_M:
            raise RuntimeError(
                f"root drift exceeded 2 mm: {max_root_translation}"
            )
        if max_root_rotation > MAX_ROOT_ROTATION_DEG:
            raise RuntimeError(
                f"root rotation drift exceeded 1 deg: {max_root_rotation}"
            )
    else:
        escape_translation = getattr(
            contract_spec, "root_escape_translation_limit_m", None
        )
        escape_rotation = getattr(
            contract_spec, "root_escape_rotation_limit_deg", None
        )
        if escape_translation is not None and max_root_translation > float(
            escape_translation
        ):
            raise RuntimeError(
                "free-root translation exceeded the G10 gross-escape envelope: "
                f"{max_root_translation} > {escape_translation}"
            )
        if escape_rotation is not None and max_root_rotation > float(
            escape_rotation
        ):
            raise RuntimeError(
                "free-root rotation exceeded the G10 gross-escape envelope: "
                f"{max_root_rotation} > {escape_rotation}"
            )

    if raw.task_name != G08_TASK_NAME:
        return audit, None

    support = dict(raw._g08_support_substep_audit_snapshot())
    support_sample_count = _audit_int(support, "sample_count")
    if support_sample_count != expected_sample_count:
        raise RuntimeError(
            "G08 support physics-frame audit count mismatch: "
            f"{support_sample_count} != {expected_sample_count}"
        )
    # G08 is now a reset-positioned native fixed-base articulation. Preserve
    # every support sample as diagnostic evidence, but never turn a missing or
    # low-force table manifold into a reset/stability/success gate.
    return audit, support


def _run_zero_action_stability(raw: Any, control_steps: int) -> dict[str, Any]:
    _require_fresh_post_accept_substep_audits(raw)
    observation_shapes: dict[str, list[int]] | None = None
    terminal: list[dict[str, Any]] = []
    for control_step in range(control_steps):
        result = raw.step(None)
        if not isinstance(result, tuple) or len(result) < 5:
            raise RuntimeError(
                f"zero-action step returned an invalid result: {result!r}"
            )
        current_shapes = _observation_summary(
            raw,
            result[0],
            phase=f"zero-action control step {control_step}",
        )
        if observation_shapes is None:
            observation_shapes = current_shapes
        elif current_shapes != observation_shapes:
            raise RuntimeError(
                "zero-action observation shapes changed: "
                f"{current_shapes!r} != {observation_shapes!r}"
            )
        terminated = bool(result[2])
        truncated = bool(result[3])
        if terminated or truncated:
            terminal.append(
                {
                    "control_step": control_step,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": result[4],
                }
            )
            break

    if terminal:
        raise RuntimeError(f"zero-action rollout terminated: {terminal!r}")
    if raw._lw_state.failure_reason:
        raise RuntimeError(
            f"task failure latched during zero-action rollout: "
            f"{raw._lw_state.failure_reason}"
        )
    physics_steps = control_steps * int(raw._n_substeps)
    audit, support = _validate_post_accept_substep_audits(
        raw,
        physics_steps,
    )
    return {
        "control_steps": control_steps,
        "physics_steps": physics_steps,
        "observation_shapes": observation_shapes,
        "post_accept_substep_audit": audit,
        "g08_support_substep_audit": support,
        "max_q_drift_rad": audit["max_abs_q_error_rad"],
        "max_abs_qd_rad_s": audit["max_abs_qd_rad_s"],
        "max_root_translation_drift_m": audit[
            "max_root_translation_error_m"
        ],
        "max_root_rotation_drift_deg": audit[
            "max_root_rotation_error_deg"
        ],
        "min_support_force_n": (
            None if support is None else support["min_support_force_n"]
        ),
        "min_support_contact_count": (
            None if support is None else support["min_support_contact_count"]
        ),
        "support_clearance_range_m": (
            None if support is None else support["support_clearance_range_m"]
        ),
        "support_separation_range_m": (
            None
            if support is None
            else support["support_separation_range_m"]
        ),
        "final_truth": _truth_payload(raw),
    }


def _run_g08_first_command_guard(raw: Any) -> dict[str, Any]:
    """Prove the first non-idle tracker target is bounded on live PhysX."""

    import numpy as np

    if raw.task_name != G08_TASK_NAME:
        raise RuntimeError("G08 first-command guard was requested for another task")
    if not raw._physx_contact_arm_target_limits_enabled():
        raise RuntimeError("G08 live arm target limits are disabled")
    if int(raw._lw_reset_tcp_sync_count) < 1:
        raise RuntimeError("G08 reset did not synchronize TCP command zero")

    side = "right"
    robot = raw._robots[side]
    palm_index = int(raw._ee_body_indices[side])

    def palm_pose() -> np.ndarray:
        robot.update(0.0)
        return np.asarray(
            raw._pose_matrix_to_vector(
                raw._robot_body_pose_matrix(robot, palm_index)
            ),
            dtype=np.float64,
        ).reshape(7)

    before_palm = palm_pose()
    command_zero = np.asarray(raw._right_tcp_pose, dtype=np.float64).reshape(7)
    command_zero_position_error_m = float(
        np.linalg.norm(command_zero[:3] - before_palm[:3])
    )
    if command_zero_position_error_m > 0.010:
        raise RuntimeError(
            "G08 command zero drifted from the physical right palm before the "
            f"first target: {command_zero_position_error_m:.6f} m"
        )

    before_arm_target = np.asarray(
        raw._arm_reference_q(side, robot), dtype=np.float64
    ).reshape(-1)
    before_truth = _truth_payload(raw)
    command = command_zero.copy()
    command[2] += G08_FIRST_COMMAND_LIFT_M
    right_action = np.concatenate(
        [command, np.asarray(raw._right_hand, dtype=np.float64).reshape(-1)]
    )
    result = raw.step({"right": right_action})
    if not isinstance(result, tuple) or len(result) < 5:
        raise RuntimeError(f"G08 first-command step returned {result!r}")
    if bool(result[2]) or bool(result[3]):
        raise RuntimeError(
            "G08 first-command step terminated unexpectedly: "
            f"terminated={bool(result[2])}, truncated={bool(result[3])}, "
            f"info={result[4]!r}"
        )
    if raw._lw_state.failure_reason:
        raise RuntimeError(
            "G08 first-command step latched a task failure: "
            f"{raw._lw_state.failure_reason}"
        )

    after_arm_target = np.asarray(
        raw._arm_reference_q(side, robot), dtype=np.float64
    ).reshape(-1)
    after_palm = palm_pose()
    after_truth = _truth_payload(raw)
    max_joint_target_step_rad = float(
        np.max(np.abs(after_arm_target - before_arm_target))
    )
    palm_displacement_m = float(
        np.linalg.norm(after_palm[:3] - before_palm[:3])
    )
    before_q = float(np.asarray(before_truth["q_rad"]).reshape(-1)[0])
    after_q = float(np.asarray(after_truth["q_rad"]).reshape(-1)[0])
    after_qd = float(np.asarray(after_truth["qd_rad_s"]).reshape(-1)[0])
    lid_motion_rad = abs(after_q - before_q)
    lid_speed_rad_s = abs(after_qd)

    if max_joint_target_step_rad > G08_FIRST_COMMAND_MAX_JOINT_TARGET_STEP_RAD:
        raise RuntimeError(
            "G08 first arm target bypassed the joint-step limiter: "
            f"{max_joint_target_step_rad:.6f} rad"
        )
    if palm_displacement_m > G08_FIRST_COMMAND_MAX_PALM_DISPLACEMENT_M:
        raise RuntimeError(
            "G08 right palm jumped on its first target: "
            f"{palm_displacement_m:.6f} m"
        )
    if lid_motion_rad > G08_FIRST_COMMAND_MAX_LID_MOTION_RAD:
        raise RuntimeError(
            "G08 lid moved during the guarded first lift: "
            f"{math.degrees(lid_motion_rad):.6f} deg"
        )
    if lid_speed_rad_s > G08_FIRST_COMMAND_MAX_LID_SPEED_RAD_S:
        raise RuntimeError(
            "G08 lid accelerated during the guarded first lift: "
            f"{lid_speed_rad_s:.6f} rad/s"
        )

    return {
        "tcp_sync_count": int(raw._lw_reset_tcp_sync_count),
        "arm_target_limits_enabled": True,
        "command_lift_m": G08_FIRST_COMMAND_LIFT_M,
        "command_zero_position_error_m": command_zero_position_error_m,
        "max_joint_target_step_rad": max_joint_target_step_rad,
        "max_joint_target_step_limit_rad": (
            G08_FIRST_COMMAND_MAX_JOINT_TARGET_STEP_RAD
        ),
        "palm_displacement_m": palm_displacement_m,
        "palm_displacement_limit_m": G08_FIRST_COMMAND_MAX_PALM_DISPLACEMENT_M,
        "lid_motion_rad": lid_motion_rad,
        "lid_motion_limit_rad": G08_FIRST_COMMAND_MAX_LID_MOTION_RAD,
        "lid_speed_rad_s": lid_speed_rad_s,
        "lid_speed_limit_rad_s": G08_FIRST_COMMAND_MAX_LID_SPEED_RAD_S,
        "before_truth": before_truth,
        "after_truth": after_truth,
    }


def _run_g08_passive_lid_impulse_probe(raw: Any) -> dict[str, Any]:
    """Apply a small opening impulse and prove the passive lid cannot fly up."""

    if raw.task_name != G08_TASK_NAME:
        raise RuntimeError("G08 lid impulse probe was requested for another task")
    if raw._lw_articulation_view is None or raw._torch is None:
        raise RuntimeError("G08 lid impulse probe has no live articulation view")
    if len(raw._lw_dof_indices) != 1:
        raise RuntimeError(
            f"G08 lid impulse probe expected one DOF, got {raw._lw_dof_indices!r}"
        )

    view = raw._lw_articulation_view
    positions = view.get_dof_positions().clone()
    velocities = view.get_dof_velocities().clone()
    dof_index = int(raw._lw_dof_indices[0])
    positions[:, dof_index] = float(G08_LID_PROBE_INITIAL_ANGLE_RAD)
    velocities[:, dof_index] = float(G08_LID_PROBE_INITIAL_SPEED_RAD_S)
    articulation_indices = raw._torch.tensor(
        [0],
        dtype=raw._torch.long,
        device=positions.device,
    )
    view.set_dof_positions(positions, articulation_indices)
    view.set_dof_velocities(velocities, articulation_indices)

    samples: list[dict[str, float]] = []
    min_q_rad = float(G08_LID_PROBE_INITIAL_ANGLE_RAD)
    max_abs_qd_rad_s = abs(float(G08_LID_PROBE_INITIAL_SPEED_RAD_S))
    max_opening_speed_rad_s = max(
        0.0,
        -float(G08_LID_PROBE_INITIAL_SPEED_RAD_S),
    )
    max_closing_speed_rad_s = max(
        0.0,
        float(G08_LID_PROBE_INITIAL_SPEED_RAD_S),
    )
    for step_index in range(G08_LID_PROBE_CONTROL_STEPS):
        result = raw.step(None)
        if not isinstance(result, tuple) or len(result) < 5:
            raise RuntimeError(f"G08 lid impulse probe step returned {result!r}")
        if bool(result[2]) or bool(result[3]):
            raise RuntimeError(
                "G08 lid impulse probe terminated unexpectedly: "
                f"step={step_index}, terminated={bool(result[2])}, "
                f"truncated={bool(result[3])}, info={result[4]!r}"
            )
        truth = _truth_payload(raw)
        q_rad = float(truth["q_rad"][0])
        qd_rad_s = float(truth["qd_rad_s"][0])
        min_q_rad = min(min_q_rad, q_rad)
        max_abs_qd_rad_s = max(max_abs_qd_rad_s, abs(qd_rad_s))
        max_opening_speed_rad_s = max(max_opening_speed_rad_s, -qd_rad_s)
        max_closing_speed_rad_s = max(max_closing_speed_rad_s, qd_rad_s)
        if step_index in (0, 1, 2, 4, 9, 19, 39, 59):
            samples.append(
                {
                    "control_step": int(step_index + 1),
                    "q_rad": q_rad,
                    "qd_rad_s": qd_rad_s,
                }
            )

    final_truth = _truth_payload(raw)
    final_q_rad = float(final_truth["q_rad"][0])
    if min_q_rad < -float(G08_LID_PROBE_MAX_OPEN_ANGLE_RAD):
        raise RuntimeError(
            "G08 passive lid flew open under the small impulse: "
            f"max_open={math.degrees(-min_q_rad):.3f} deg"
        )
    if final_q_rad < -float(G08_LID_PROBE_FINAL_OPEN_ANGLE_RAD):
        raise RuntimeError(
            "G08 passive lid did not settle closed after the small impulse: "
            f"final_open={math.degrees(-final_q_rad):.3f} deg"
        )
    if max_opening_speed_rad_s > float(
        G08_LID_PROBE_MAX_OPENING_SPEED_RAD_S
    ):
        raise RuntimeError(
            "G08 passive lid accelerated open under the small impulse: "
            f"max_opening_speed={max_opening_speed_rad_s:.6f} rad/s"
        )

    return {
        "initial_angle_rad": float(G08_LID_PROBE_INITIAL_ANGLE_RAD),
        "initial_speed_rad_s": float(G08_LID_PROBE_INITIAL_SPEED_RAD_S),
        "control_steps": int(G08_LID_PROBE_CONTROL_STEPS),
        "max_open_angle_rad": -min_q_rad,
        "max_open_angle_limit_rad": float(G08_LID_PROBE_MAX_OPEN_ANGLE_RAD),
        "max_abs_speed_rad_s": max_abs_qd_rad_s,
        "max_opening_speed_rad_s": max_opening_speed_rad_s,
        "max_opening_speed_limit_rad_s": float(
            G08_LID_PROBE_MAX_OPENING_SPEED_RAD_S
        ),
        "max_closing_speed_rad_s": max_closing_speed_rad_s,
        "final_open_angle_rad": max(0.0, -final_q_rad),
        "final_open_angle_limit_rad": float(G08_LID_PROBE_FINAL_OPEN_ANGLE_RAD),
        "samples": samples,
        "final_truth": final_truth,
    }


def _audit_g08_reset_randomization(
    resets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify every live reset sample stays inside the widened XY/yaw box."""

    unique_by_seed: dict[int, dict[str, Any]] = {}
    for reset in resets:
        sample = dict(reset.get("authored_sample") or {})
        seed = int(reset["seed"])
        if sample.get("randomize") is not True:
            raise RuntimeError(f"G08 live reset was not randomized: {sample!r}")
        try:
            root_x_m = float(sample["root_x_m"])
            root_y_m = float(sample["root_y_m"])
            root_yaw_deg = float(sample["root_yaw_deg"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"G08 live reset randomization sample is invalid: {sample!r}"
            ) from exc
        if not (
            abs(root_x_m - G08_ROOT_NOMINAL_X_M)
            <= G08_ROOT_RANDOM_X_HALF_RANGE_M + 1.0e-9
            and abs(root_y_m - G08_ROOT_NOMINAL_Y_M)
            <= G08_ROOT_RANDOM_Y_HALF_RANGE_M + 1.0e-9
            and abs(root_yaw_deg)
            <= G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG + 1.0e-9
        ):
            raise RuntimeError(
                "G08 live reset escaped the widened randomization envelope: "
                f"{sample!r}"
            )
        existing = unique_by_seed.get(seed)
        if existing is not None and existing != sample:
            raise RuntimeError(
                f"G08 same-seed reset changed inside one run: seed={seed}"
            )
        unique_by_seed[seed] = sample

    if len(unique_by_seed) < 2:
        raise RuntimeError(
            "G08 live randomization audit needs at least two distinct seeds"
        )
    samples = list(unique_by_seed.values())
    x_values = [float(sample["root_x_m"]) for sample in samples]
    y_values = [float(sample["root_y_m"]) for sample in samples]
    yaw_values = [float(sample["root_yaw_deg"]) for sample in samples]
    if len({(x, y, yaw) for x, y, yaw in zip(x_values, y_values, yaw_values)}) != len(
        samples
    ):
        raise RuntimeError("G08 distinct seeds produced duplicate XY/yaw poses")
    return {
        "nominal_xy_m": [G08_ROOT_NOMINAL_X_M, G08_ROOT_NOMINAL_Y_M],
        "declared_x_range_m": [
            G08_ROOT_NOMINAL_X_M - G08_ROOT_RANDOM_X_HALF_RANGE_M,
            G08_ROOT_NOMINAL_X_M + G08_ROOT_RANDOM_X_HALF_RANGE_M,
        ],
        "declared_y_range_m": [
            G08_ROOT_NOMINAL_Y_M - G08_ROOT_RANDOM_Y_HALF_RANGE_M,
            G08_ROOT_NOMINAL_Y_M + G08_ROOT_RANDOM_Y_HALF_RANGE_M,
        ],
        "declared_yaw_range_deg": [
            -G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG,
            G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG,
        ],
        "distinct_seed_count": len(samples),
        "observed_x_range_m": [min(x_values), max(x_values)],
        "observed_y_range_m": [min(y_values), max(y_values)],
        "observed_yaw_range_deg": [min(yaw_values), max(yaw_values)],
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASK_CONFIG_MODULES), required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--reset-trials", type=int, default=2)
    parser.add_argument("--control-steps", type=int, default=60)
    parser.add_argument("--hard-reset", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reset_trials < 1:
        parser.error("--reset-trials must be at least 1")
    if args.control_steps < 60:
        parser.error("--control-steps must be at least 60 (240 physics steps)")

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"

    repository_root = Path(__file__).resolve().parents[1]
    output = args.output or (
        repository_root
        / "taskdev"
        / "tasks"
        / TASKDEV_DIRS[args.task]
        / "evidence"
        / "live_initial_stability.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "schema": "handson.lw_native_articulation_live_stability.v1",
        "task_name": args.task,
        "seed": args.seed,
        "status": "running",
        "git": _git_metadata(repository_root),
    }
    env = None
    raw = None
    try:
        module = importlib.import_module(TASK_CONFIG_MODULES[args.task])
        env = module.TaskConfig().get_native_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=True,
            sim_backend="isaacsim",
            physics_backend="physx",
            launch_isaac=True,
            headless=True,
            image_obs=False,
            seed=args.seed,
            isaac_device="cuda:0",
        )
        raw = env.unwrapped
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        body_count, collider_count, dof_count, fixed_base = EXPECTED_TOPOLOGY[
            args.task
        ]
        metatype = raw._lw_articulation_view.get_metatype(0)
        composed_joint_limits_deg: dict[str, list[float]] = {}
        g08_lid_dynamics: dict[str, Any] | None = None
        g10_ruler_dynamics: dict[str, Any] | None = None
        if args.task == G08_TASK_NAME:
            # Verify the task-scoped override on the *composed* stage, not only
            # in the pure contract or YAML.  The source USD remains hashed by
            # validate_lw_asset_files(); this read proves the integration
            # layer authored -90/0 degrees where PhysX discovers the joint.
            joint_path = (
                f"{raw._lw_scene_spec.asset_path}/"
                f"{raw._lw_scene_spec.right_body_name}/"
                f"{raw._lw_scene_spec.dof_names[0]}"
            )
            joint_prim = raw._stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid() or not joint_prim.IsA(
                UsdPhysics.RevoluteJoint
            ):
                raise RuntimeError(
                    f"G08 composed lid joint is missing or not revolute: {joint_path}"
                )
            joint = UsdPhysics.RevoluteJoint(joint_prim)
            lower_limit = float(joint.GetLowerLimitAttr().Get())
            upper_limit = float(joint.GetUpperLimitAttr().Get())
            composed_joint_limits_deg[raw._lw_scene_spec.dof_names[0]] = [
                lower_limit,
                upper_limit,
            ]
            if not (
                math.isclose(
                    lower_limit,
                    -G08_LID_HARD_STOP_DEG,
                    abs_tol=1.0e-6,
                )
                and math.isclose(upper_limit, 0.0, abs_tol=1.0e-6)
            ):
                raise RuntimeError(
                    "G08 composed lid limits are not the task hard stop: "
                    f"path={joint_path}, lower={lower_limit}, upper={upper_limit}"
                )
            if not bool(raw._lw_manifest.get("fixed_root", False)):
                raise RuntimeError(
                    "G08 spawn manifest lost fixed_root=true before live capture"
                )

            lid_prim = raw._stage.GetPrimAtPath(
                raw._lw_scene_spec.right_body_path
            )
            if not lid_prim.IsValid() or not lid_prim.HasAPI(
                UsdPhysics.MassAPI
            ):
                raise RuntimeError(
                    "G08 composed lid is missing its repaired MassAPI: "
                    f"{raw._lw_scene_spec.right_body_path}"
                )
            if not lid_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                raise RuntimeError(
                    "G08 composed lid is missing PhysxRigidBodyAPI"
                )
            if not joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
                raise RuntimeError(
                    "G08 composed lid joint is missing PhysxJointAPI; "
                    "joint friction would not be active"
                )
            mass_api = UsdPhysics.MassAPI(lid_prim)
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(lid_prim)
            joint_api = PhysxSchema.PhysxJointAPI(joint_prim)
            mass_kg = float(mass_api.GetMassAttr().Get())
            com_local_m = tuple(
                float(value) for value in mass_api.GetCenterOfMassAttr().Get()
            )
            diagonal_inertia_kg_m2 = tuple(
                float(value)
                for value in mass_api.GetDiagonalInertiaAttr().Get()
            )
            angular_damping = float(
                rigid_api.GetAngularDampingAttr().Get()
            )
            joint_friction = float(
                joint_api.GetJointFrictionAttr().Get()
            )
            scale = tuple(float(value) for value in raw._lw_scene_spec.scale)
            effective_hinge_inertia = float(
                diagonal_inertia_kg_m2[0]
                + mass_kg
                * (
                    (com_local_m[1] * scale[1]) ** 2
                    + (com_local_m[2] * scale[2]) ** 2
                )
            )
            g08_lid_dynamics = {
                "policy": str(
                    lid_prim.GetCustomDataByKey(
                        "handson:g08LidDynamicsPolicy"
                    )
                    or ""
                ),
                "mass_kg": mass_kg,
                "com_local_m": list(com_local_m),
                "asset_scale": list(scale),
                "diagonal_inertia_kg_m2": list(diagonal_inertia_kg_m2),
                "effective_hinge_inertia_kg_m2": effective_hinge_inertia,
                "angular_damping": angular_damping,
                "joint_friction": joint_friction,
                "physx_joint_api_applied": True,
            }
            if not (
                math.isclose(mass_kg, 0.300, abs_tol=1.0e-6)
                and all(
                    math.isclose(actual, expected, abs_tol=1.0e-6)
                    for actual, expected in zip(
                        com_local_m,
                        (0.0, -0.2411680296, 0.0275114477),
                        strict=True,
                    )
                )
                and all(
                    math.isfinite(value) and value > 1.0e-8
                    for value in diagonal_inertia_kg_m2
                )
                and all(
                    math.isclose(
                        value,
                        G08_EXPECTED_ASSET_SCALE,
                        abs_tol=1.0e-6,
                    )
                    for value in scale
                )
                and effective_hinge_inertia > 0.0038
                and math.isclose(angular_damping, 2.0, abs_tol=1.0e-6)
                and math.isclose(joint_friction, 0.10, abs_tol=1.0e-6)
                and g08_lid_dynamics["policy"]
                == "g08_measured_lid_dynamics_v3_scale040"
            ):
                raise RuntimeError(
                    "G08 composed lid dynamics readback changed: "
                    f"{g08_lid_dynamics!r}"
                )
        elif args.task == G10_TASK_NAME:
            if bool(raw._lw_manifest.get("fixed_root", True)):
                raise RuntimeError(
                    "G10 spawn manifest unexpectedly constrained the free root"
                )
            for dof_name, expected_limits in {
                "Prop001_joint": (-180.0, 180.0),
                "Prop002_joint": (-180.0, 180.0),
            }.items():
                joint_path = f"{raw._lw_scene_spec.asset_path}/"
                joint_path += (
                    "Ruler035_Prop001/Prop001_joint"
                    if dof_name == "Prop001_joint"
                    else "Ruler035_Prop002/Prop002_joint"
                )
                joint_prim = raw._stage.GetPrimAtPath(joint_path)
                if not joint_prim.IsValid() or not joint_prim.IsA(
                    UsdPhysics.RevoluteJoint
                ):
                    raise RuntimeError(
                        f"G10 composed ruler joint is missing or not revolute: {joint_path}"
                    )
                joint = UsdPhysics.RevoluteJoint(joint_prim)
                lower_limit = float(joint.GetLowerLimitAttr().Get())
                upper_limit = float(joint.GetUpperLimitAttr().Get())
                composed_joint_limits_deg[dof_name] = [
                    lower_limit,
                    upper_limit,
                ]
                if not (
                    math.isclose(lower_limit, expected_limits[0], abs_tol=1.0e-6)
                    and math.isclose(upper_limit, expected_limits[1], abs_tol=1.0e-6)
                ):
                    raise RuntimeError(
                        "G10 composed ruler limits are not symmetric ±180 degrees: "
                        f"path={joint_path}, lower={lower_limit}, upper={upper_limit}"
                    )
            asset_root = raw._stage.GetPrimAtPath(
                raw._lw_scene_spec.asset_path
            )
            policy = str(
                asset_root.GetCustomDataByKey("handson:g10DynamicsPolicy")
                or ""
            )
            body_dynamics: dict[str, dict[str, float]] = {}
            for body_name in (
                raw._lw_scene_spec.left_body_name,
                raw._lw_scene_spec.root_body_name,
                raw._lw_scene_spec.right_body_name,
            ):
                body_prim = raw._stage.GetPrimAtPath(
                    f"{raw._lw_scene_spec.asset_path}/{body_name}"
                )
                rigid_api = PhysxSchema.PhysxRigidBodyAPI(body_prim)
                body_dynamics[body_name] = {
                    "linear_damping": float(
                        rigid_api.GetLinearDampingAttr().Get()
                    ),
                    "angular_damping": float(
                        rigid_api.GetAngularDampingAttr().Get()
                    ),
                }
            joint_dynamics: dict[str, dict[str, Any]] = {}
            for dof_name in raw._lw_scene_spec.dof_names:
                joint_path = f"{raw._lw_scene_spec.asset_path}/"
                joint_path += (
                    "Ruler035_Prop001/Prop001_joint"
                    if dof_name == "Prop001_joint"
                    else "Ruler035_Prop002/Prop002_joint"
                )
                joint_prim = raw._stage.GetPrimAtPath(joint_path)
                joint_api = PhysxSchema.PhysxJointAPI(joint_prim)
                drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                joint_dynamics[dof_name] = {
                    "friction": float(
                        joint_api.GetJointFrictionAttr().Get()
                    ),
                    "drive": {
                        "type": str(drive.GetTypeAttr().Get()),
                        "stiffness": float(drive.GetStiffnessAttr().Get()),
                        "damping": float(drive.GetDampingAttr().Get()),
                        "max_force_nm": float(drive.GetMaxForceAttr().Get()),
                        "target_velocity_deg_s": float(
                            drive.GetTargetVelocityAttr().Get()
                        ),
                    },
                }
            g10_ruler_dynamics = {
                "policy": policy,
                "body": body_dynamics,
                "joint": joint_dynamics,
            }
            if (
                policy != G10_DYNAMICS_POLICY
                or any(
                    not math.isclose(
                        values["linear_damping"],
                        G10_LINK_LINEAR_DAMPING,
                        abs_tol=1.0e-6,
                    )
                    or not math.isclose(
                        values["angular_damping"],
                        G10_LINK_ANGULAR_DAMPING,
                        abs_tol=1.0e-6,
                    )
                    for values in body_dynamics.values()
                )
                or any(
                    not math.isclose(
                        values["friction"],
                        G10_JOINT_FRICTION,
                        abs_tol=1.0e-6,
                    )
                    or values["drive"]["type"] != "force"
                    or not math.isclose(
                        values["drive"]["stiffness"],
                        G10_DRIVE_STIFFNESS,
                        abs_tol=1.0e-6,
                    )
                    or not math.isclose(
                        values["drive"]["damping"],
                        G10_DRIVE_DAMPING,
                        abs_tol=1.0e-6,
                    )
                    or not math.isclose(
                        values["drive"]["max_force_nm"],
                        G10_DRIVE_MAX_FORCE_NM,
                        abs_tol=1.0e-6,
                    )
                    or not math.isclose(
                        values["drive"]["target_velocity_deg_s"],
                        G10_DRIVE_TARGET_VELOCITY_DEG_S,
                        abs_tol=1.0e-6,
                    )
                    for values in joint_dynamics.values()
                )
            ):
                raise RuntimeError(
                    "G10 composed passive dynamics readback changed: "
                    f"{g10_ruler_dynamics!r}"
                )
        asset_root = raw._stage.GetPrimAtPath(raw._lw_scene_spec.asset_path)
        stage_body_count = 0
        stage_collider_count = 0
        nonrendering_prim_audit: list[dict[str, Any]] = []
        renderable_collision_only_paths: list[str] = []
        grip_paths = tuple(str(path) for path in raw._lw_manifest["grip_paths"])
        for prim in Usd.PrimRange(asset_root):
            stage_body_count += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            stage_collider_count += int(prim.HasAPI(UsdPhysics.CollisionAPI))
            path = str(prim.GetPath())
            classification = str(
                prim.GetCustomDataByKey("handson:renderClassification") or ""
            )
            collision_only = bool(
                classification.startswith("collision_only")
                or "/Collisions/" in path
                or any(
                    path == grip_path or path.startswith(grip_path + "/")
                    for grip_path in grip_paths
                )
            )
            if not collision_only:
                continue
            imageable = UsdGeom.Imageable(prim)
            computed_visibility = (
                str(imageable.ComputeVisibility())
                if imageable
                else "not_imageable"
            )
            entry = {
                "path": path,
                "classification": classification or "path_classified",
                "has_collision_api": bool(
                    prim.HasAPI(UsdPhysics.CollisionAPI)
                ),
                "computed_visibility": computed_visibility,
            }
            nonrendering_prim_audit.append(entry)
            if imageable and computed_visibility != str(UsdGeom.Tokens.invisible):
                renderable_collision_only_paths.append(path)
        if renderable_collision_only_paths:
            raise RuntimeError(
                "collision-only/helper prims remain renderable in the final "
                f"composed stage: {renderable_collision_only_paths!r}"
            )
        topology = {
            "physics_hz": round(1.0 / raw._physics_dt),
            "control_hz": round(1.0 / raw._control_dt),
            "substeps": int(raw._n_substeps),
            "body_count": int(raw._lw_manifest["body_count"]),
            "collider_count": int(raw._lw_manifest["collider_count"]),
            "source_body_count": int(raw._lw_manifest["source_body_count"]),
            "source_collider_count": int(
                raw._lw_manifest["source_collider_count"]
            ),
            "stage_body_count": stage_body_count,
            "stage_collider_count": stage_collider_count,
            "physx_link_count": int(metatype.link_count),
            "physx_dof_count": int(metatype.dof_count),
            "physx_link_names": list(metatype.link_names),
            "dof_names": list(raw._lw_scene_spec.dof_names),
            "articulation_path": raw._lw_articulation_path,
            "fixed_base": bool(metatype.fixed_base),
            "manifest_fixed_root": bool(raw._lw_manifest.get("fixed_root", False)),
            "composed_joint_limits_deg": composed_joint_limits_deg,
            "g08_lid_dynamics": g08_lid_dynamics,
            "g10_ruler_dynamics": g10_ruler_dynamics,
            "assigned_contact_views": sorted(raw._lw_contact_views),
            "support_contact_view": raw._lw_support_contact_view is not None,
            "fixture_reset_overlap": sorted(
                str(state[0])
                for state in raw._task_reset_body_states
                if str(state[0]).startswith(raw._lw_scene_spec.asset_path)
            ),
            "nonrendering_prim_audit": nonrendering_prim_audit,
            "renderable_collision_only_paths": (
                renderable_collision_only_paths
            ),
        }
        expected = {
            "physics_hz": 120,
            "control_hz": 30,
            "substeps": 4,
            "body_count": body_count,
            "collider_count": collider_count,
            "dof_count": dof_count,
            "fixed_base": fixed_base,
        }
        actual = {
            key: topology[key]
            for key in (
                "physics_hz",
                "control_hz",
                "substeps",
                "body_count",
                "collider_count",
                "fixed_base",
            )
        }
        actual["dof_count"] = len(topology["dof_names"])
        if actual != expected:
            raise RuntimeError(
                f"live topology mismatch: actual={actual}, expected={expected}"
            )
        if (
            topology["stage_body_count"] != topology["body_count"]
            or topology["stage_collider_count"] != topology["collider_count"]
        ):
            raise RuntimeError(
                "LW manifest topology does not match the composed live stage: "
                f"{topology!r}"
            )
        if (
            topology["physx_link_count"] != body_count
            or topology["physx_dof_count"] != dof_count
        ):
            raise RuntimeError(
                "LW active PhysX articulation inventory changed: "
                f"{topology!r}"
            )
        if topology["assigned_contact_views"] != ["left", "right"]:
            raise RuntimeError(f"assigned contact views changed: {topology!r}")
        if bool(topology["support_contact_view"]) != (
            args.task == "t18_open_micropipette_box"
        ):
            raise RuntimeError(f"support contact view changed: {topology!r}")
        if topology["fixture_reset_overlap"]:
            raise RuntimeError(
                "articulation child rigid bodies leaked into reset truth: "
                f"{topology['fixture_reset_overlap']}"
            )
        evidence["topology"] = topology

        resets: list[dict[str, Any]] = []
        first = _reset_and_accept_origin(env, raw, args.seed)
        resets.append(first)
        evidence["zero_action_stability"] = _run_zero_action_stability(
            raw,
            args.control_steps,
        )
        if args.task == G08_TASK_NAME:
            evidence["first_command_guard"] = _run_g08_first_command_guard(raw)

        repeated = _reset_and_accept_origin(env, raw, args.seed)
        resets.append(repeated)
        if first["authored_sample"] != repeated["authored_sample"]:
            raise RuntimeError("same-seed authored reset is not deterministic")
        if args.task == G08_TASK_NAME:
            evidence["passive_lid_impulse_probe"] = (
                _run_g08_passive_lid_impulse_probe(raw)
            )

        for offset in range(1, args.reset_trials):
            resets.append(
                _reset_and_accept_origin(env, raw, args.seed + offset)
            )

        if args.hard_reset:
            hard_seed = args.seed + 100003
            hard_reset_result = raw.hard_reset_physx_scene(seed=hard_seed)
            evidence["hard_reset_result"] = hard_reset_result
            resets.append(_reset_and_accept_origin(env, raw, hard_seed))

        if args.task == G08_TASK_NAME:
            evidence["reset_randomization"] = _audit_g08_reset_randomization(
                resets
            )

        evidence["reset_trials"] = resets
        evidence["asset_manifest_sha256"] = _sha256(
            repository_root
            / "handson"
            / "handson"
            / "sim"
            / "envs"
            / "usd"
            / "lw_native_articulation"
            / "manifest.json"
        )
        evidence["status"] = "pass"
        _write_evidence(output, evidence)
        print(f"LW_LIVE_STABILITY_PASS {args.task} {output}", flush=True)
        return 0
    except Exception as exc:
        evidence["status"] = "fail"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["traceback"] = traceback.format_exc()
        if raw is not None:
            try:
                evidence["failure_truth"] = _truth_payload(raw)
            except Exception as truth_exc:
                evidence["failure_truth_error"] = (
                    f"{type(truth_exc).__name__}: {truth_exc}"
                )
        _write_evidence(output, evidence)
        print(f"LW_LIVE_STABILITY_FAIL {args.task} {output}: {exc}", flush=True)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
