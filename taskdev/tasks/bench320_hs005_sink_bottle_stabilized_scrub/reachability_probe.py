#!/usr/bin/env python3
"""Run a camera-free live PhysX reachability probe for B320-HS005."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=1705)
    result.add_argument("--tail-frames", type=int, default=6)
    result.add_argument(
        "--reset-only",
        action="store_true",
        help="validate the physical reset and exit before automated arm motion",
    )
    AppLauncher.add_app_launcher_args(result)
    return result


class ResetOnlyComplete(RuntimeError):
    """Internal control flow after reset-only evidence has been populated."""


def jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def orientation_error_rad(target, actual) -> float:
    import numpy as np

    target_quat = np.asarray(target[3:7], dtype=np.float64)
    actual_quat = np.asarray(actual[3:7], dtype=np.float64)
    target_quat /= np.linalg.norm(target_quat)
    actual_quat /= np.linalg.norm(actual_quat)
    cosine = float(np.clip(abs(np.dot(target_quat, actual_quat)), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


def pose_delta(reset_pose, current_pose) -> dict[str, float]:
    import numpy as np

    reset = np.asarray(reset_pose, dtype=np.float64).reshape(7)
    current = np.asarray(current_pose, dtype=np.float64).reshape(7)
    return {
        "translation_m": float(np.linalg.norm(current[:3] - reset[:3])),
        "rotation_rad": orientation_error_rad(reset, current),
    }


def arm_joint_margin(environment, side: str) -> float:
    import numpy as np

    robot = environment._robots[side]
    joint_ids = environment._arm_joint_ids[side]
    positions = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
    limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
    margins = np.minimum(positions - limits[:, 0], limits[:, 1] - positions)
    return float(np.min(margins))


def physical_tcp_readback(environment, observation):
    """Read both live palms, including the post-failure diagnostic path."""
    import numpy as np

    if "physx_tcp_state" in observation and "physx_tcp_valid" in observation:
        return (
            np.asarray(observation["physx_tcp_state"], dtype=np.float64),
            np.asarray(observation["physx_tcp_valid"]).astype(bool),
        )
    poses = []
    valid = []
    for side in ("right", "left"):
        matrix = environment._robot_link_pose_matrix(side)
        valid.append(matrix is not None)
        poses.append(
            np.zeros(13, dtype=np.float64)
            if matrix is None
            else np.pad(environment._pose_matrix_to_vector(matrix), (0, 6))
        )
    return np.stack(poses), np.asarray(valid, dtype=bool)


def hs005_task_event(info) -> dict:
    """Keep the small set of physical event fields needed to audit a phase."""
    keys = (
        "hs005_runtime_valid",
        "hs005_left_contact_force",
        "hs005_right_contact_force",
        "hs005_scrub_contact_force",
        "hs005_left_contact_seen",
        "hs005_right_contact_seen",
        "hs005_right_safe_grasp_seen",
        "hs005_right_safe_grasp_now",
        "hs005_right_bristle_touch_now",
        "hs005_right_grasp_surface_error",
        "hs005_right_grasp_tip_count",
        "hs005_bristle_local_z",
        "hs005_bristle_radial_error",
        "hs005_current_band",
        "hs005_coverage_mask",
        "hs005_order_progress",
        "hs005_order_violation",
        "hs005_path_length",
        "hs005_coverage_complete",
        "hs005_bottle_displacement",
        "hs005_bottle_up_alignment",
        "hs005_release_seen",
        "hs005_stability_steps",
        "hs005_stage_index",
        "hs005_success_now",
    )
    return {key: info[key] for key in keys if key in info}


def main() -> int:
    args = parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = "cuda:0"
    launcher = AppLauncher(args)
    app = launcher.app

    import numpy as np

    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
    from handson.sim.isaaclab.recording_origin import stabilize_recording_origin
    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.automated_policy import (
        HS005AutomatedPolicy,
    )

    environment = None
    started = time.time()
    result = {
        "schema": "handson.b320_hs005.reachability_probe.v1",
        "seed": int(args.seed),
        "pass": False,
    }
    rc = 1
    try:
        environment = HandsOnABCIsaacLabEnv(
            "t37_scrub_bottle",
            seed=int(args.seed),
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / 240.0,
            time_limit=45.0,
            render_mode=None,
            image_obs=False,
            randomize=True,
            randomize_dynamics=True,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            hz=30,
            isaac_device=args.device,
            physics_backend="physx",
            existing_simulation_app=app,
            existing_app_launcher=launcher,
        )
        observation, reset_info = environment.reset(seed=int(args.seed))
        reset_failure = None
        try:
            observation, reset_info = stabilize_recording_origin(
                environment,
                observation,
                reset_info,
                initial_settle_seconds=0.50,
                stable_seconds=0.25,
                stability_sample_seconds=0.25,
                timeout_seconds=12.0,
                linear_velocity_limit=0.03,
                angular_velocity_limit=0.40,
                pose_translation_limit=0.001,
                pose_rotation_limit=0.02,
            )
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            reset_failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
            }
            # Continue only as a diagnostic.  The overall probe remains failed,
            # while live readback identifies the body and arm errors in one run.
            observation = environment._compute_observation()
            reset_info = {
                **dict(reset_info),
                "physical_recording_origin": (
                    diagnostics if isinstance(diagnostics, dict) else {}
                ),
            }
        reset_body_drift = {}
        for name, reset_pose in (
            ("bottle", environment._hs005_bottle_reset_pose),
            ("brush", environment._hs005_brush_reset_pose),
        ):
            current_pose = environment._current_body_pose(f"hs005_{name}")
            if reset_pose is not None and current_pose is not None:
                reset_body_drift[name] = {
                    **pose_delta(reset_pose, current_pose),
                    "reset_pose_wxyz": np.asarray(reset_pose).tolist(),
                    "current_pose_wxyz": np.asarray(current_pose).tolist(),
                }
        state = observation["state"]
        physical_tcp, physical_valid = physical_tcp_readback(
            environment,
            observation,
        )
        if args.reset_only:
            body_velocities = np.asarray(
                state["hs005_body_velocities"],
                dtype=np.float64,
            ).reshape(2, 6)
            drift_pass = bool(
                reset_body_drift
                and all(
                    float(row["translation_m"]) <= 0.002
                    and float(row["rotation_rad"]) <= 0.02
                    for row in reset_body_drift.values()
                )
            )
            reset_only_pass = bool(
                reset_failure is None
                and np.all(physical_valid)
                and np.all(np.isfinite(physical_tcp))
                and np.all(np.isfinite(body_velocities))
                and drift_pass
                and np.max(np.linalg.norm(body_velocities[:, :3], axis=1)) <= 0.03
                and np.max(np.linalg.norm(body_velocities[:, 3:], axis=1)) <= 0.40
            )
            result.update(
                {
                    "pass": reset_only_pass,
                    "reset_only": True,
                    "reset_origin_pass": reset_failure is None,
                    "reset_origin_failure": reset_failure,
                    "reset_body_drift": reset_body_drift,
                    "physical_tcp_valid": physical_valid.tolist(),
                    "physical_tcp_state": physical_tcp.tolist(),
                    "body_velocities": body_velocities.tolist(),
                    "reset_parameters": state["hs005_reset_parameters"],
                    "stable_recording_origin": reset_info.get(
                        "physical_recording_origin", {}
                    ),
                }
            )
            rc = 0 if reset_only_pass else 2
            raise ResetOnlyComplete
        policy = HS005AutomatedPolicy(
            state["hs005_bottle_pose"],
            state["hs005_brush_pose"],
            state["tcp_pose"],
            physical_tcp,
            physical_valid,
            state["hs005_reset_parameters"],
        )
        phase_ends = []
        total = 0
        for phase, frames in policy.waypoint_schedule:
            total += int(frames)
            phase_ends.append((phase, total))

        frame_rows = []
        phase_index = 0
        final_info = dict(reset_info)
        for frame in range(policy.total_frames):
            live_tcp, live_valid = physical_tcp_readback(environment, observation)
            action = policy.action(live_tcp, live_valid)
            desired_targets = policy.last_desired_target_palm
            observation, _reward, terminated, truncated, info = environment.step(action)
            physical, valid = physical_tcp_readback(environment, observation)
            targets = np.stack((action[:7], action[7:14])).astype(np.float64)
            task_event = hs005_task_event(info)
            row = {
                "frame": frame + 1,
                "phase": phase_ends[phase_index][0],
                "valid": valid.tolist(),
                "target_pose_wxyz": targets.tolist(),
                "desired_goal_pose_wxyz": desired_targets.tolist(),
                "actual_pose_wxyz": physical[:, :7].tolist(),
                "position_error_m": [
                    float(np.linalg.norm(targets[side, :3] - physical[side, :3]))
                    for side in range(2)
                ],
                "orientation_error_rad": [
                    orientation_error_rad(targets[side], physical[side])
                    for side in range(2)
                ],
                "goal_position_error_m": [
                    float(
                        np.linalg.norm(
                            desired_targets[side, :3] - physical[side, :3]
                        )
                    )
                    for side in range(2)
                ],
                "goal_orientation_error_rad": [
                    orientation_error_rad(desired_targets[side], physical[side])
                    for side in range(2)
                ],
                "joint_margin_rad": [
                    arm_joint_margin(environment, "right"),
                    arm_joint_margin(environment, "left"),
                ],
                "task_event": task_event,
            }
            frame_rows.append(row)
            final_info = dict(info)
            if frame + 1 >= phase_ends[phase_index][1] and phase_index + 1 < len(phase_ends):
                phase_index += 1
            if terminated or truncated:
                break

        tail_count = max(1, int(args.tail_frames))
        phase_reports = []
        start = 0
        for phase, end in phase_ends:
            end = min(end, len(frame_rows))
            if end <= start:
                break
            tail = frame_rows[max(start, end - tail_count) : end]
            phase_rows = frame_rows[start:end]
            events = [row["task_event"] for row in phase_rows]

            def peak(key, default=0.0):
                return float(max(float(event.get(key, default)) for event in events))

            finite_surface_errors = [
                float(event["hs005_right_grasp_surface_error"])
                for event in events
                if math.isfinite(float(event.get("hs005_right_grasp_surface_error", math.inf)))
            ]
            phase_reports.append(
                {
                    "phase": phase,
                    "frames": end - start,
                    "tail_max_position_error_m": np.max(
                        [row["position_error_m"] for row in tail], axis=0
                    ).tolist(),
                    "tail_max_orientation_error_rad": np.max(
                        [row["orientation_error_rad"] for row in tail], axis=0
                    ).tolist(),
                    "minimum_joint_margin_rad": np.min(
                        [row["joint_margin_rad"] for row in tail], axis=0
                    ).tolist(),
                    "end_goal_position_error_m": tail[-1][
                        "goal_position_error_m"
                    ],
                    "end_goal_orientation_error_rad": tail[-1][
                        "goal_orientation_error_rad"
                    ],
                    "peak_contact_force": {
                        "left_bottle": peak("hs005_left_contact_force"),
                        "right_brush": peak("hs005_right_contact_force"),
                        "brush_bottle": peak("hs005_scrub_contact_force"),
                    },
                    "minimum_right_grasp_surface_error_m": (
                        min(finite_surface_errors) if finite_surface_errors else None
                    ),
                    "maximum_right_grasp_tip_count": int(
                        peak("hs005_right_grasp_tip_count")
                    ),
                    "maximum_order_progress": int(peak("hs005_order_progress")),
                    "maximum_scrub_path_length_m": peak("hs005_path_length"),
                    "maximum_bottle_displacement_m": peak(
                        "hs005_bottle_displacement"
                    ),
                    "end_task_event": tail[-1]["task_event"],
                    "end": tail[-1],
                }
            )
            start = end

        max_tail_position = float(
            max(max(report["tail_max_position_error_m"]) for report in phase_reports)
        )
        max_tail_orientation = float(
            max(max(report["tail_max_orientation_error_rad"]) for report in phase_reports)
        )
        min_joint_margin = float(
            min(min(report["minimum_joint_margin_rad"]) for report in phase_reports)
        )
        max_end_goal_position = float(
            max(max(report["end_goal_position_error_m"]) for report in phase_reports)
        )
        max_end_goal_orientation = float(
            max(max(report["end_goal_orientation_error_rad"]) for report in phase_reports)
        )
        reachability_pass = bool(
            max_tail_position <= 0.01
            and max_tail_orientation <= 0.15
            and max_end_goal_position <= 0.01
            and max_end_goal_orientation <= 0.15
            and min_joint_margin >= 0.05
        )
        overall_pass = bool(reset_failure is None and reachability_pass)
        result.update(
            {
                "pass": overall_pass,
                "reset_origin_pass": reset_failure is None,
                "reset_origin_failure": reset_failure,
                "reset_body_drift": reset_body_drift,
                "reachability_pass": reachability_pass,
                "reachability_thresholds": {
                    "position_error_m_max": 0.01,
                    "orientation_error_rad_max": 0.15,
                    "joint_margin_rad_min": 0.05,
                },
                "max_tail_position_error_m": max_tail_position,
                "max_tail_orientation_error_rad": max_tail_orientation,
                "minimum_joint_margin_rad": min_joint_margin,
                "max_end_goal_position_error_m": max_end_goal_position,
                "max_end_goal_orientation_error_rad": max_end_goal_orientation,
                "frame_count": len(frame_rows),
                "phase_reports": phase_reports,
                "reset_parameters": state["hs005_reset_parameters"],
                "stable_recording_origin": reset_info["physical_recording_origin"],
                "final_task_info": {
                    key: value
                    for key, value in final_info.items()
                    if str(key).startswith("hs005_")
                },
                "task_tcp_contract": policy.reachability_report,
            }
        )
        rc = 0 if overall_pass else 2
    except ResetOnlyComplete:
        pass
    except Exception as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, dict):
            result["exception"]["diagnostics"] = diagnostics
        rc = 1
    finally:
        result["elapsed_wall_seconds"] = time.time() - started
        result["exit_code"] = rc
        result["cleanup_policy"] = {
            "evidence_written_before_teardown": True,
            "kit_cleanup_watchdog_seconds": 20.0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(jsonable(result), indent=2, sort_keys=True), flush=True)

        # Headless Kit teardown has already blocked after otherwise complete
        # HS005 probes.  Keep the normal close path, but terminate with the
        # already sealed producer result if teardown does not return.
        import threading

        shutdown_watchdog = threading.Timer(20.0, lambda: os._exit(rc))
        shutdown_watchdog.daemon = True
        shutdown_watchdog.start()
        try:
            if environment is not None:
                environment.close()
            app.close()
        except Exception as exc:
            shutdown_watchdog.cancel()
            result["environment_close_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            rc = 1
            result["pass"] = False
            result["exit_code"] = rc
            args.output.write_text(
                json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            shutdown_watchdog.cancel()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
