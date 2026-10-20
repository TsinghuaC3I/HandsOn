#!/usr/bin/env python3
"""Strict 10-second priority-zero gate for the exact HS014 production child."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from isaaclab.app import AppLauncher  # noqa: E402


PHYSICS_HZ = 120
CONTROL_HZ = 30
TASK_NAME = "bench320_hs014_bottle_stabilized_cap_opening"
STABLE_RECORDING_ORIGIN_ENV = "HANDSON_STABLE_RECORDING_ORIGIN"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reset-count",
        type=int,
        default=1,
        help=(
            "Must be exactly 1: HS014's replay contract requires one seeded "
            "first reset per fresh Isaac process. Run multiple seeds as "
            "separate process invocations."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    parser.add_argument(
        "--diagnostic-continue-after-initial-failure",
        action="store_true",
        help=(
            "Collect no-action geometry diagnostics after the strict initial "
            "gate rejects a seed. Diagnostic runs can never pass."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _quat_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(4)
    right = np.asarray(right, dtype=np.float64).reshape(4)
    left /= max(float(np.linalg.norm(left)), 1.0e-12)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    alignment = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return 2.0 * math.acos(alignment)


def _json_safe(value: Any) -> Any:
    """Return strict-JSON data without hiding a non-finite physics readback."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _require_hs014_physical_recording_origin(
    info: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a disabled, skipped, truncated, or unaccepted shared gate."""

    if not isinstance(info, Mapping):
        raise RuntimeError("HS014 stable-origin gate returned non-mapping info")
    raw_origin = info.get("physical_recording_origin")
    if not isinstance(raw_origin, Mapping):
        raise RuntimeError(
            "HS014 live gate is missing physical_recording_origin; "
            "the shared stable-origin gate was disabled or skipped"
        )
    origin = dict(raw_origin)
    if str(origin.get("task_name", "")).strip().lower() != TASK_NAME:
        raise RuntimeError("HS014 physical_recording_origin has the wrong task_name")
    stable_seconds = float(profile.get("stable_seconds", 0.0))
    if not math.isclose(stable_seconds, 10.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            "HS014 production stable-origin profile must require exactly 10 seconds"
        )
    expected_steps = int(round(stable_seconds * PHYSICS_HZ))
    required_steps = origin.get("stable_steps_required")
    observed_steps = origin.get("stable_steps_observed")
    if (
        type(required_steps) is not int
        or required_steps != expected_steps
        or type(observed_steps) is not int
        or observed_steps < required_steps
    ):
        raise RuntimeError(
            "HS014 physical_recording_origin does not prove the complete "
            f"{expected_steps}-frame production gate"
        )
    acceptance = origin.get("task_recording_origin_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("accepted") is not True:
        raise RuntimeError(
            "HS014 physical_recording_origin lacks exact task acceptance"
        )
    return origin


def _exit_code_for_result(result: Mapping[str, Any]) -> int:
    """Return success only for an exact positive validator result."""

    return 0 if result.get("passed") is True else 2


def _state(environment: Any, bottle_name: str, cap_name: str) -> dict[str, Any]:
    truth = dict(environment._opening_truth())
    support_force, support_penetration = environment._contact_reading(
        "bottle_support"
    )
    cap_bottle_force, cap_bottle_penetration = environment._contact_reading(
        "cap_bottle"
    )
    bottle_awake, bottle_awake_available, bottle_awake_error = (
        environment._body_awake_readback(bottle_name)
    )
    cap_awake, cap_awake_available, cap_awake_error = (
        environment._body_awake_readback(cap_name)
    )
    bottle_velocity = environment._physics_body_velocity(bottle_name)
    cap_velocity = environment._physics_body_velocity(cap_name)
    bottle_velocity = (
        np.full(6, np.nan, dtype=np.float64)
        if bottle_velocity is None
        else np.asarray(bottle_velocity, dtype=np.float64).reshape(6)
    )
    cap_velocity = (
        np.full(6, np.nan, dtype=np.float64)
        if cap_velocity is None
        else np.asarray(cap_velocity, dtype=np.float64).reshape(6)
    )
    bottle = np.asarray(truth["bottle_pose"], dtype=np.float64).reshape(7)
    cap = np.asarray(truth["cap_pose"], dtype=np.float64).reshape(7)
    finite = bool(
        np.all(np.isfinite(bottle))
        and np.all(np.isfinite(cap))
        and np.all(np.isfinite(bottle_velocity))
        and np.all(np.isfinite(cap_velocity))
        and np.isfinite(support_force)
        and np.isfinite(support_penetration)
        and np.isfinite(cap_bottle_force)
        and np.isfinite(cap_bottle_penetration)
        and all(
            np.isfinite(float(truth[key]))
            for key in (
                "axis_error_deg",
                "cap_radial_offset_m",
                "axial_separation_m",
                "rotation_progress_rad",
                "bottle_displacement_m",
                "bottle_orientation_drift_deg",
                "bottle_yaw_drift_deg",
                "bottle_tilt_deg",
                "cap_linear_speed_mps",
                "cap_angular_speed_radps",
            )
        )
    )
    return {
        "finite": finite,
        "bottle": bottle,
        "cap": cap,
        "bottle_velocity": bottle_velocity,
        "cap_velocity": cap_velocity,
        "bottle_support_force_n": float(support_force),
        "bottle_support_penetration_m": float(support_penetration),
        "cap_bottle_force_n": float(cap_bottle_force),
        "cap_bottle_penetration_m": float(cap_bottle_penetration),
        "cap_bottle_contact_diagnostics": dict(
            getattr(environment, "_hs014_contact_diagnostics", {}).get(
                "cap_bottle", {}
            )
        ),
        "bottle_awake": bool(bottle_awake),
        "cap_awake": bool(cap_awake),
        "awake_readback_available": bool(
            bottle_awake_available and cap_awake_available
        ),
        "awake_readback_errors": [
            error
            for error in (bottle_awake_error, cap_awake_error)
            if error
        ],
        "truth": truth,
        "failure": bool(environment._hs014_failure_latched),
        "failure_reason": str(environment._hs014_failure_reason),
        "stage": int(environment._hs014_stage),
        "joint_enabled": {
            key: bool(environment._read_joint_enabled(key))
            for key in environment._hs014_joint_enabled
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if int(args.reset_count) != 1:
        raise SystemExit(
            "HS014 requires exactly one public seeded reset per fresh Isaac "
            "process; invoke this validator separately for every seed."
        )
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
    # stabilize_recording_origin() intentionally no-ops without this explicit
    # opt-in.  A validator must exercise the production gate, not silently
    # audit only the public reset.
    os.environ[STABLE_RECORDING_ORIGIN_ENV] = "1"
    # Contact-report APIs must be authored before the first sim.reset(); the
    # shared production gate cannot safely add them after tensor views exist.
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, Any] = {
        "schema": "handson.hs014.priority_zero_live.v2",
        "passed": False,
        "runtime": {
            "passed": False,
            "requested_device": args.device,
            "lifecycle": (
                "fresh_construct_then_public_seeded_reset_then_shared_"
                "production_recording_origin"
            ),
        },
    }
    rc = _exit_code_for_result(result)
    try:
        from handson.sim.isaaclab.hs014_collection_env import (
            BOTTLE_NAME,
            CAP_SEATED_AXIAL_MAX_M,
            CAP_SEATED_AXIAL_MIN_M,
            CAP_SEATED_AXIS_ERROR_MAX_DEG,
            CAP_SEATED_RADIAL_MAX_M,
            CAP_NAME,
            HS014IsaacLabEnv,
        )
        from handson.sim.isaaclab.priority_zero_robot_audit import (
            RobotStabilityAudit,
            initial_robot_state_report,
            priority_zero_runtime_contract,
        )
        from handson.sim.isaaclab.recording_origin import (
            stabilize_recording_origin,
            stable_recording_origin_kwargs_for_task,
        )

        initial_gate_failures: list[dict[str, Any]] = []
        if args.diagnostic_continue_after_initial_failure:
            strict_initial_gate = (
                HS014IsaacLabEnv._validate_task_initial_state_before_ready
            )

            def diagnostic_initial_gate(environment, *gate_args, **gate_kwargs):
                try:
                    return strict_initial_gate(
                        environment,
                        *gate_args,
                        **gate_kwargs,
                    )
                except RuntimeError as exc:
                    initial_gate_failures.append(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "report": dict(
                                getattr(
                                    environment,
                                    "_hs014_initial_state_report",
                                    {},
                                )
                            ),
                        }
                    )
                    return {
                        "diagnostic_initial_gate_bypassed": True,
                        "error": str(exc),
                    }

            HS014IsaacLabEnv._validate_task_initial_state_before_ready = (
                diagnostic_initial_gate
            )

        environment = HS014IsaacLabEnv(
            seed=int(args.seed_start),
            control_dt=1.0 / CONTROL_HZ,
            physics_dt=1.0 / PHYSICS_HZ,
            time_limit=90.0,
            render_mode="rgb_array",
            image_obs=False,
            randomize=True,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            isaac_device=args.device,
            physics_backend="physx",
            existing_simulation_app=app,
            existing_app_launcher=launcher,
        )
        if environment._sim is None:
            raise RuntimeError("HS014 exact production child has no PhysX context")
        runtime_contract = priority_zero_runtime_contract(
            environment,
            requested_device=args.device,
            physics_hz=PHYSICS_HZ,
            control_hz=CONTROL_HZ,
        )
        result["runtime"].update(
            {
                "passed": True,
                "simulator": "Isaac Sim 5.1.0.0",
                "isaaclab": "2.3.2",
                "physics": "PhysX",
                "exact_production_child": "HS014IsaacLabEnv",
                "verified_contract": runtime_contract,
                "physics_schema_readback": dict(
                    environment._hs014_runtime_physics_schema
                ),
            }
        )

        robot_inventory = {
            side: {
                "arm_joints": len(environment._arm_joint_ids[side]),
                "hand_joints": len(environment._hand_joint_ids[side]),
            }
            for side in ("left", "right")
        }
        contact_inventory = {
            key: {
                "sensor_count": int(view.sensor_count),
                "filter_count": int(view.filter_count),
            }
            for key, view in environment._hs014_contact_views.items()
        }
        reset_rows: list[dict[str, Any]] = []
        result["reset_trials"] = reset_rows
        # Match the declared collection/replay lifecycle exactly: construct a
        # fresh process, perform the episode's public seeded reset, then run the
        # shared production recording-origin gate before action 0.
        seed = int(args.seed_start)
        observation, reset_info = environment.reset(seed=seed)
        info = dict(reset_info)
        robot_passed, robot_report = initial_robot_state_report(environment)
        initial = _state(environment, BOTTLE_NAME, CAP_NAME)
        initial_readiness = dict(
            environment.stable_recording_origin_task_readiness()
        )
        reset_rows.append(
            {
                "seed": seed,
                "public_seeded_reset_after_fresh_construction": True,
                "public_reset_physics_steps": int(
                    environment._reset_physics_steps
                ),
                "finite": initial["finite"],
                "initial_failed": bool(
                    info.get("failed", False) or initial["failure"]
                ),
                "initial_success": bool(info.get("succeed", False)),
                "stage": initial["stage"],
                "fixed_joints_disabled": not any(
                    initial["joint_enabled"].values()
                ),
                "axial_separation_from_reset_m": float(
                    initial["truth"]["axial_separation_m"]
                ),
                "rotation_progress_rad": float(
                    initial["truth"]["rotation_progress_rad"]
                ),
                "cap_axial_position_m": float(
                    initial["truth"]["axial_position_m"]
                ),
                "cap_radial_offset_m": float(
                    initial["truth"]["cap_radial_offset_m"]
                ),
                "cap_axis_error_deg": float(initial["truth"]["axis_error_deg"]),
                "bottle_displacement_m": float(
                    initial["truth"]["bottle_displacement_m"]
                ),
                "bottle_orientation_drift_deg": float(
                    initial["truth"]["bottle_orientation_drift_deg"]
                ),
                "bottle_linear_speed_mps": float(
                    np.linalg.norm(initial["bottle_velocity"][:3])
                ),
                "bottle_angular_speed_radps": float(
                    np.linalg.norm(initial["bottle_velocity"][3:])
                ),
                "cap_linear_speed_mps": float(
                    np.linalg.norm(initial["cap_velocity"][:3])
                ),
                "cap_angular_speed_radps": float(
                    np.linalg.norm(initial["cap_velocity"][3:])
                ),
                "bottle_support_force_n": float(
                    initial["bottle_support_force_n"]
                ),
                "bottle_support_penetration_m": float(
                    initial["bottle_support_penetration_m"]
                ),
                "cap_bottle_force_n": float(initial["cap_bottle_force_n"]),
                "cap_bottle_penetration_m": float(
                    initial["cap_bottle_penetration_m"]
                ),
                "cap_bottle_contact_diagnostics": dict(
                    initial["cap_bottle_contact_diagnostics"]
                ),
                "bottle_awake": bool(initial["bottle_awake"]),
                "cap_awake": bool(initial["cap_awake"]),
                "awake_readback_available": bool(
                    initial["awake_readback_available"]
                ),
                "robot_initial_state_passed": robot_passed,
                "robot_initial_state": robot_report,
                "task_readiness": initial_readiness,
            }
        )
        if not initial["finite"]:
            raise RuntimeError("HS014 initial physical state is non-finite")
        robot_stability = RobotStabilityAudit.start(environment)
        origin_profile = stable_recording_origin_kwargs_for_task(
            environment.task_name
        )
        if not math.isclose(
            float(args.stability_seconds),
            float(origin_profile["stable_seconds"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "HS014 live evidence must use the production stable_seconds "
                f"value {origin_profile['stable_seconds']}; got "
                f"{args.stability_seconds}"
            )
        requested_steps = int(
            round(float(origin_profile["stable_seconds"]) * PHYSICS_HZ)
        )
        physics_steps = 0
        nonfinite_samples = 0
        peak_bottle_linear_speed = 0.0
        peak_bottle_angular_speed = 0.0
        peak_cap_linear_speed = 0.0
        peak_cap_angular_speed = 0.0
        max_bottle_translation_per_frame_m = 0.0
        max_bottle_rotation_per_frame_rad = 0.0
        max_cap_translation_per_frame_m = 0.0
        max_cap_rotation_per_frame_rad = 0.0
        previous_bottle = initial["bottle"].copy()
        previous_cap = initial["cap"].copy()
        supported_physics_samples = 0
        awake_physics_samples = 0
        task_ready_physics_samples = 0
        first_task_readiness_failure: dict[str, Any] | None = None
        max_support_penetration_m = 0.0
        unexpected_terminal = False
        terminal_info: dict[str, Any] = {}
        origin_error: dict[str, Any] | None = None
        stable_info: dict[str, Any] = {}
        original_step_physics = environment._step_isaaclab

        def audit_step_physics() -> None:
            nonlocal physics_steps
            nonlocal nonfinite_samples
            nonlocal peak_bottle_linear_speed
            nonlocal peak_bottle_angular_speed
            nonlocal peak_cap_linear_speed
            nonlocal peak_cap_angular_speed
            nonlocal max_bottle_translation_per_frame_m
            nonlocal max_bottle_rotation_per_frame_rad
            nonlocal max_cap_translation_per_frame_m
            nonlocal max_cap_rotation_per_frame_rad
            nonlocal previous_bottle
            nonlocal previous_cap
            nonlocal supported_physics_samples
            nonlocal awake_physics_samples
            nonlocal task_ready_physics_samples
            nonlocal first_task_readiness_failure
            nonlocal max_support_penetration_m
            original_step_physics()
            physics_steps += 1
            current = _state(environment, BOTTLE_NAME, CAP_NAME)
            if not current["finite"]:
                nonfinite_samples += 1
            bottle_velocity = current["bottle_velocity"]
            cap_velocity = current["cap_velocity"]
            peak_bottle_linear_speed = max(
                peak_bottle_linear_speed,
                float(np.linalg.norm(bottle_velocity[:3])),
            )
            peak_bottle_angular_speed = max(
                peak_bottle_angular_speed,
                float(np.linalg.norm(bottle_velocity[3:])),
            )
            peak_cap_linear_speed = max(
                peak_cap_linear_speed,
                float(np.linalg.norm(cap_velocity[:3])),
            )
            peak_cap_angular_speed = max(
                peak_cap_angular_speed,
                float(np.linalg.norm(cap_velocity[3:])),
            )
            max_bottle_translation_per_frame_m = max(
                max_bottle_translation_per_frame_m,
                float(np.linalg.norm(current["bottle"][:3] - previous_bottle[:3])),
            )
            max_bottle_rotation_per_frame_rad = max(
                max_bottle_rotation_per_frame_rad,
                _quat_angle_rad(current["bottle"][3:7], previous_bottle[3:7]),
            )
            max_cap_translation_per_frame_m = max(
                max_cap_translation_per_frame_m,
                float(np.linalg.norm(current["cap"][:3] - previous_cap[:3])),
            )
            max_cap_rotation_per_frame_rad = max(
                max_cap_rotation_per_frame_rad,
                _quat_angle_rad(current["cap"][3:7], previous_cap[3:7]),
            )
            previous_bottle = current["bottle"].copy()
            previous_cap = current["cap"].copy()
            if current["bottle_support_force_n"] >= 0.01:
                supported_physics_samples += 1
            if (
                current["awake_readback_available"]
                and current["bottle_awake"]
                and current["cap_awake"]
            ):
                awake_physics_samples += 1
            task_readiness = dict(
                environment.stable_recording_origin_task_readiness()
            )
            if bool(task_readiness.get("ready", False)):
                task_ready_physics_samples += 1
            elif first_task_readiness_failure is None:
                first_task_readiness_failure = {
                    "physics_step": physics_steps,
                    "readiness": task_readiness,
                }
            max_support_penetration_m = max(
                max_support_penetration_m,
                float(current["bottle_support_penetration_m"]),
            )
            robot_stability.sample(environment)

        environment._step_isaaclab = audit_step_physics
        try:
            observation, stable_info = stabilize_recording_origin(
                environment,
                observation,
                reset_info,
                **origin_profile,
            )
            _require_hs014_physical_recording_origin(
                stable_info,
                profile=origin_profile,
            )
        except Exception as exc:
            origin_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            environment._step_isaaclab = original_step_physics
        final = _state(environment, BOTTLE_NAME, CAP_NAME)

        bottle_translation_drift_m = float(
            np.linalg.norm(final["bottle"][:3] - initial["bottle"][:3])
        )
        bottle_rotation_drift_rad = _quat_angle_rad(
            final["bottle"][3:7], initial["bottle"][3:7]
        )
        cap_translation_drift_m = float(
            np.linalg.norm(final["cap"][:3] - initial["cap"][:3])
        )
        cap_rotation_drift_rad = _quat_angle_rad(
            final["cap"][3:7], initial["cap"][3:7]
        )
        final_bottle_linear_speed = float(
            np.linalg.norm(final["bottle_velocity"][:3])
        )
        final_bottle_angular_speed = float(
            np.linalg.norm(final["bottle_velocity"][3:])
        )
        final_cap_linear_speed = float(
            np.linalg.norm(final["cap_velocity"][:3])
        )
        final_cap_angular_speed = float(
            np.linalg.norm(final["cap_velocity"][3:])
        )
        bottle_velocity_or_static_pose = bool(
            (
                final_bottle_linear_speed <= 0.001
                and final_bottle_angular_speed <= 0.002
            )
            or (
                max_bottle_translation_per_frame_m <= 1.0e-6
                and max_bottle_rotation_per_frame_rad <= 1.0e-5
            )
        )
        cap_velocity_or_static_pose = bool(
            (
                final_cap_linear_speed <= 0.001
                and final_cap_angular_speed <= 0.002
            )
            or (
                max_cap_translation_per_frame_m <= 1.0e-6
                and max_cap_rotation_per_frame_rad <= 1.0e-5
            )
        )
        final_robot_passed, final_robot_report = initial_robot_state_report(
            environment
        )
        final_readiness = dict(
            environment.stable_recording_origin_task_readiness()
        )
        origin_report = (
            dict(stable_info.get("physical_recording_origin", {}))
            if isinstance(stable_info, Mapping)
            else {}
        )
        accepted_boundary = (
            dict(origin_report.get("task_recording_origin_acceptance", {}))
            if isinstance(
                origin_report.get("task_recording_origin_acceptance"),
                Mapping,
            )
            else {}
        )
        if origin_error is not None:
            accepted_boundary = {
                "accepted": False,
                "origin_error": origin_error,
            }

        reset_pass = bool(
            len(reset_rows) == 1
            and all(
                row["finite"]
                and row["public_seeded_reset_after_fresh_construction"]
                and row["public_reset_physics_steps"] == PHYSICS_HZ
                and not row["initial_failed"]
                and not row["initial_success"]
                and row["fixed_joints_disabled"]
                and row["awake_readback_available"]
                and row["bottle_awake"]
                and row["cap_awake"]
                and abs(row["axial_separation_from_reset_m"]) <= 0.002
                and CAP_SEATED_AXIAL_MIN_M
                <= row["cap_axial_position_m"]
                <= CAP_SEATED_AXIAL_MAX_M
                and row["cap_radial_offset_m"] <= CAP_SEATED_RADIAL_MAX_M
                and row["cap_axis_error_deg"] <= CAP_SEATED_AXIS_ERROR_MAX_DEG
                and row["cap_bottle_force_n"] >= 0.01
                and row["cap_bottle_penetration_m"] <= 0.0005
                and abs(row["rotation_progress_rad"]) <= 1.0e-6
                and row["bottle_displacement_m"] <= 0.005
                and row["bottle_orientation_drift_deg"] <= 2.0
                and row["bottle_linear_speed_mps"] <= 0.05
                and row["bottle_angular_speed_radps"] <= 0.75
                and row["cap_linear_speed_mps"] <= 0.05
                and row["cap_angular_speed_radps"] <= 0.75
                and row["bottle_support_force_n"] >= 0.01
                and row["bottle_support_penetration_m"] <= 0.0005
                and row["robot_initial_state_passed"]
                and bool(row["task_readiness"].get("ready", False))
                for row in reset_rows
            )
        )
        stability_pass = bool(
            origin_error is None
            and physics_steps >= 10 * PHYSICS_HZ
            and nonfinite_samples == 0
            and not unexpected_terminal
            and final["finite"]
            and not final["failure"]
            and not any(final["joint_enabled"].values())
            and abs(float(final["truth"]["axial_separation_m"])) <= 0.005
            and abs(float(final["truth"]["rotation_progress_rad"])) <= 1.0e-6
            and peak_bottle_linear_speed <= 0.05
            and peak_bottle_angular_speed <= 0.75
            and peak_cap_linear_speed <= 0.05
            and peak_cap_angular_speed <= 0.75
            and bottle_velocity_or_static_pose
            and cap_velocity_or_static_pose
            and bottle_translation_drift_m <= 0.005
            and bottle_rotation_drift_rad <= math.radians(3.0)
            and cap_translation_drift_m <= 0.005
            and cap_rotation_drift_rad <= math.radians(3.0)
            and supported_physics_samples == physics_steps
            and awake_physics_samples == physics_steps
            and task_ready_physics_samples == physics_steps
            and final["bottle_support_force_n"] >= 0.01
            and max_support_penetration_m <= 0.0005
            and robot_stability.passed(
                physics_steps=physics_steps,
                physics_hz=PHYSICS_HZ,
            )
            and final_robot_passed
            and bool(final_readiness.get("ready", False))
            and bool(accepted_boundary.get("accepted", False))
        )
        bottle_prim_path = str(
            environment._hs014_composition.get("bottle", {}).get("prim_path", "")
        )
        physics_schema_readback = environment._hs014_runtime_physics_schema
        physics_schema_checks = physics_schema_readback.get("checks", {})
        cap_mass_readback = physics_schema_readback.get("bodies", {}).get("cap", {})
        expected_cap_mass_properties = {
            "mass_kg": 0.012,
            "center_of_mass_local_m": [0.0, 0.013, 0.0],
            "diagonal_inertia_kg_m2": [
                3.635590e-6,
                4.745227e-6,
                3.635590e-6,
            ],
            "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        cap_mass_property_keys = tuple(expected_cap_mass_properties)
        cap_mass_property_tolerances = {
            "mass_kg": 1.0e-8,
            "center_of_mass_local_m": 1.0e-8,
            "diagonal_inertia_kg_m2": 1.0e-10,
            "principal_axes_wxyz": 1.0e-8,
        }
        cap_composition = environment._hs014_composition.get("cap", {})
        collision_topology = physics_schema_readback.get("collision_topology", {})
        helper_material_values = collision_topology.get(
            "helper_physics_material", {}
        ).get("values", {})
        helper_material_values_match = bool(
            isinstance(helper_material_values, Mapping)
            and all(
                key in helper_material_values
                and math.isclose(
                    float(helper_material_values[key]),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                for key, expected in (
                    ("static_friction", 0.0),
                    ("dynamic_friction", 0.0),
                    ("restitution", 0.0),
                )
            )
            and helper_material_values.get("friction_combine_mode") == "min"
        )
        cap_bottle_diagnostics = final.get(
            "cap_bottle_contact_diagnostics", {}
        )
        checks = {
            "verified_isaac51_isaaclab232_cuda_physx_runtime": bool(
                runtime_contract.get("passed", False)
            ),
            "post_reset_physics_schema_readback": bool(
                physics_schema_readback.get("passed", False)
            ),
            "post_reset_cap_mass_properties_readback": bool(
                all(
                    physics_schema_checks.get(f"cap_{key}") is True
                    for key in cap_mass_property_keys
                )
                and math.isclose(
                    float(cap_mass_readback.get("mass_kg", math.nan)),
                    expected_cap_mass_properties["mass_kg"],
                    rel_tol=0.0,
                    abs_tol=cap_mass_property_tolerances["mass_kg"],
                )
                and all(
                    np.allclose(
                        cap_mass_readback.get(key, []),
                        expected_cap_mass_properties[key],
                        rtol=0.0,
                        atol=cap_mass_property_tolerances[key],
                    )
                    for key in cap_mass_property_keys
                    if key != "mass_kg"
                )
            ),
            "composed_cap_thin_wall_mass_properties": bool(
                math.isclose(
                    float(cap_composition.get("mass_kg", math.nan)),
                    expected_cap_mass_properties["mass_kg"],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and all(
                    np.allclose(
                        cap_composition.get(key, []),
                        expected_cap_mass_properties[key],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    for key in cap_mass_property_keys
                    if key != "mass_kg"
                )
            ),
            "post_reset_torque_neutral_cradle_and_material_readback": bool(
                collision_topology.get("passed", False)
                and collision_topology.get("contact_truth_topology")
                == "three_sphere_cradle_to_com_sphere"
                and len(collision_topology.get("bottle_support_paths", [])) == 3
                and len(collision_topology.get("unfiltered_contact_pairs", [])) == 3
                and math.isclose(
                    float(collision_topology.get("center_delta_z_m", math.nan)),
                    0.0143,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    float(collision_topology.get("center_distance_m", math.nan)),
                    0.0145,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and collision_topology.get("helper_physics_material", {}).get(
                    "passed"
                )
                is True
                and helper_material_values_match
                and collision_topology.get("root_contact_view_sensor_path")
                == str(cap_composition.get("prim_path", ""))
                and collision_topology.get("root_contact_view_filter_path")
                == str(
                    environment._hs014_composition.get("bottle", {}).get(
                        "prim_path", ""
                    )
                )
            ),
            "three_sphere_cradle_exact_contact_readback": bool(
                int(cap_bottle_diagnostics.get("contact_count", 0)) >= 3
                and cap_bottle_diagnostics.get("contact_truth_topology")
                == "three_sphere_cradle_to_com_sphere"
                and cap_bottle_diagnostics.get("unfiltered_contact_pairs")
                == collision_topology.get("unfiltered_contact_pairs")
                and cap_bottle_diagnostics.get(
                    "helper_physics_material_readback_passed"
                )
                is True
                and cap_bottle_diagnostics.get(
                    "root_contact_view_represents_three_sphere_cradle"
                )
                is True
            ),
            "strict_initial_gate_not_bypassed": not bool(
                args.diagnostic_continue_after_initial_failure
            ),
            "exact_production_child": True,
            "dual_rokae_wuji1": robot_inventory
            == {
                "left": {"arm_joints": 7, "hand_joints": 20},
                "right": {"arm_joints": 7, "hand_joints": 20},
            },
            "six_required_contact_views": len(contact_inventory) == 6
            and all(row["sensor_count"] >= 1 for row in contact_inventory.values())
            and contact_inventory.get("bottle_support", {}).get("filter_count") == 1,
            "canonical_table_height": abs(float(environment.delta_h[0])) <= 1.0e-12,
            "corrected_source_axis_and_collision": bool(
                environment._hs014_composition.get("bottle", {}).get(
                    "bottle_composed_axis"
                )
                == "+Z_from_base_to_open_mouth"
                and environment._hs014_composition.get("bottle", {}).get(
                    "bottle_visual_translation_m"
                )
                == [0.0, 0.0, -0.2]
                and environment._hs014_composition.get("bottle", {}).get(
                    "collision_model"
                )
                == "source_convex_decomposition_dynamic"
                and environment._hs014_composition.get("bottle", {}).get(
                    "contact_offset_m"
                )
                == 0.0005
                and environment._hs014_composition.get("bottle", {}).get(
                    "rest_offset_m"
                )
                == 0.0
                and environment._hs014_composition.get("cap", {}).get(
                    "contact_offset_m"
                )
                == 0.0005
                and environment._hs014_composition.get("cap", {}).get(
                    "rest_offset_m"
                )
                == 0.0
                and environment._hs014_composition.get("bottle", {}).get(
                    "solver_position_iteration_count"
                )
                == 16
                and environment._hs014_composition.get("bottle", {}).get(
                    "solver_velocity_iteration_count"
                )
                == 8
                and environment._hs014_composition.get("cap", {}).get(
                    "solver_position_iteration_count"
                )
                == 16
                and environment._hs014_composition.get("cap", {}).get(
                    "solver_velocity_iteration_count"
                )
                == 8
                and not environment._hs014_composition.get("bottle", {}).get(
                    "proxy_collision_paths"
                )
                and not any(
                    environment._stage.GetPrimAtPath(
                        f"{bottle_prim_path}/{name}"
                    ).IsValid()
                    for name in ("BodyProxy", "NeckProxy", "StabilityBaseProxy")
                )
            ),
            "fresh_process_public_reset_ready_state": reset_pass,
            "production_ten_second_recording_origin": stability_pass,
        }
        result.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "robot_inventory": robot_inventory,
                "contact_inventory": contact_inventory,
                "diagnostic_initial_gate_failures": initial_gate_failures,
                "stability": {
                    "physics_steps": physics_steps,
                    "physics_hz": PHYSICS_HZ,
                    "simulated_seconds": physics_steps / PHYSICS_HZ,
                    "production_origin_profile": origin_profile,
                    "production_origin_error": origin_error,
                    "production_origin_report": origin_report,
                    "nonfinite_sample_count": nonfinite_samples,
                    "unexpected_terminal": unexpected_terminal,
                    "terminal_info": terminal_info,
                    "peak_bottle_linear_speed_mps": peak_bottle_linear_speed,
                    "peak_bottle_angular_speed_radps": peak_bottle_angular_speed,
                    "peak_cap_linear_speed_mps": peak_cap_linear_speed,
                    "peak_cap_angular_speed_radps": peak_cap_angular_speed,
                    "final_bottle_linear_speed_mps": final_bottle_linear_speed,
                    "final_bottle_angular_speed_radps": final_bottle_angular_speed,
                    "final_cap_linear_speed_mps": final_cap_linear_speed,
                    "final_cap_angular_speed_radps": final_cap_angular_speed,
                    "max_bottle_translation_per_frame_m": (
                        max_bottle_translation_per_frame_m
                    ),
                    "max_bottle_rotation_per_frame_rad": (
                        max_bottle_rotation_per_frame_rad
                    ),
                    "max_cap_translation_per_frame_m": (
                        max_cap_translation_per_frame_m
                    ),
                    "max_cap_rotation_per_frame_rad": max_cap_rotation_per_frame_rad,
                    "bottle_velocity_or_static_pose": bottle_velocity_or_static_pose,
                    "cap_velocity_or_static_pose": cap_velocity_or_static_pose,
                    "bottle_supported_physics_samples": supported_physics_samples,
                    "both_bodies_awake_physics_samples": awake_physics_samples,
                    "task_ready_physics_samples": task_ready_physics_samples,
                    "first_task_readiness_failure": first_task_readiness_failure,
                    "bottle_support_sample_fraction": (
                        supported_physics_samples / max(physics_steps, 1)
                    ),
                    "final_bottle_support_force_n": float(
                        final["bottle_support_force_n"]
                    ),
                    "max_bottle_support_penetration_m": max_support_penetration_m,
                    "bottle_translation_drift_m": bottle_translation_drift_m,
                    "bottle_rotation_drift_deg": math.degrees(
                        bottle_rotation_drift_rad
                    ),
                    "cap_translation_drift_m": cap_translation_drift_m,
                    "cap_rotation_drift_deg": math.degrees(cap_rotation_drift_rad),
                    "robot": robot_stability.to_dict(),
                    "final_robot_state_passed": final_robot_passed,
                    "final_robot_state": final_robot_report,
                    "final_task_readiness": final_readiness,
                    "task_acceptance_boundary": accepted_boundary,
                },
                "release_blockers_outside_zero_action": {
                    "corrected_source_collider_hand_sweep_pending": True,
                    "first_simulation_context_reset_readback_pending": True,
                    "source_asset_license_independently_verified": False,
                },
            }
        )
        rc = _exit_code_for_result(result)
    except Exception as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                _json_safe(result),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        # This executable owns its bounded Kit process.  SimulationApp.close
        # can replace a failed gate with process status zero, so persist and
        # flush the evidence before exiting with the truthful result.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
