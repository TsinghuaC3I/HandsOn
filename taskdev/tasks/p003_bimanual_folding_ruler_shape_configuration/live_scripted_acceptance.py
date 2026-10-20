"""Run one fresh-process P0-03 scripted physical acceptance episode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import numpy as np

from handson.sim.isaaclab.p003_collection_env import HandsOnP003IsaacLabEnv
from handson.sim.isaaclab.p003_contract import (
    FINAL_HAND_TABLE_CONTACT_FORCE_N,
    P003_RUNTIME_POLICY_VERSION,
    P003ScriptedPolicy,
    PLACEMENT_XY_TOLERANCE_M,
    RELEASE_CONTACT_FORCE_N,
    STABLE_ANGULAR_SPEED_RADPS,
    STABLE_LINEAR_SPEED_MPS,
    TASK_NAME,
    TIMEOUT_STEPS,
    _tabletop_support_metrics,
)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=TIMEOUT_STEPS)
    parser.add_argument("--image-obs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = HandsOnP003IsaacLabEnv(
        TASK_NAME,
        seed=int(args.seed),
        launch_isaac=True,
        headless=True,
        image_obs=bool(args.image_obs),
        render_mode="rgb_array",
        randomize=True,
        physics_backend="physx",
        isaac_device="cpu",
        time_limit=30.0,
    )
    report: dict[str, object] = {}
    try:
        observation, reset_info = env.reset(seed=int(args.seed))
        initial_body_poses = env._body_poses().copy()
        policy = P003ScriptedPolicy(
            env._p003_reset,
            right_home_pose=env._right_tcp_pose.copy(),
            left_home_pose=env._left_tcp_pose.copy(),
            initial_body_poses=initial_body_poses,
        )
        phase_events: list[dict[str, object]] = []
        stage_events: list[dict[str, object]] = []
        samples: list[dict[str, object]] = []
        tabletop_trace: list[dict[str, object]] = []
        contact_trace: list[dict[str, object]] = []
        role_force_by_path_peak_n: dict[str, dict[str, float]] = {
            "right_s0": {},
            "left_s2": {},
        }
        previous_phase = policy.phase
        previous_stage = env._p003_tracker.stage
        terminated = False
        truncated = False
        last_info = dict(reset_info)

        for step in range(min(int(args.steps), TIMEOUT_STEPS)):
            readback = env._p003_last_readback or env._refresh_readback()
            right_actual_tcp_pose = env._actual_tcp_pose("right")
            left_actual_tcp_pose = env._actual_tcp_pose("left")
            action = policy.action(
                int(env.env_step),
                readback=readback,
                stage=env._p003_tracker.stage,
                body_poses=env._body_poses(),
                right_tcp_pose=right_actual_tcp_pose,
                left_tcp_pose=left_actual_tcp_pose,
            )
            env.set_p003_control_state(
                policy.phase,
                configure_preparing=policy.configure_preparing,
                workspace_preparing=policy.workspace_preparing,
            )
            observation, _reward, terminated, truncated, last_info = env.step(action)
            raw_contact = dict(env._p003_last_contact_forces)
            if any(
                float(raw_contact.get(key, 0.0)) > 0.0
                for key in (
                    "right_s0",
                    "left_s2",
                    "right_s2",
                    "left_s0",
                    "robot_table",
                )
            ):
                contact_trace.append(
                    {
                        "step": step,
                        "policy_phase": policy.phase,
                        "task_stage": env._p003_tracker.stage,
                        "raw_collision_forces_n": raw_contact,
                        "right_closed": bool(
                            max(abs(float(value)) for value in env._right_hand)
                            > 0.15
                        ),
                        "left_closed": bool(
                            max(abs(float(value)) for value in env._left_hand)
                            > 0.15
                        ),
                    }
                )
            for contact_key in role_force_by_path_peak_n:
                try:
                    contact_view = env._p003_contact_views[contact_key]
                    force_matrix = env._view_numpy(
                        contact_view.get_contact_force_matrix(env._sim_dt),
                        (
                            int(contact_view.sensor_count),
                            int(contact_view.filter_count),
                            3,
                        ),
                    )
                    per_filter_force = np.sum(
                        np.linalg.norm(force_matrix, axis=-1), axis=0
                    )
                    filter_paths = env._p003_contact_paths[contact_key][1]
                    peak = role_force_by_path_peak_n[contact_key]
                    for path, force in zip(filter_paths, per_filter_force):
                        peak[path] = max(
                            float(peak.get(path, 0.0)), float(force)
                        )
                except Exception as exc:
                    role_force_by_path_peak_n[contact_key] = {
                        "diagnostic_error": str(exc)
                    }
            if policy.phase in {"configure", "release"}:
                tabletop_trace.append(
                    {
                        "step": step,
                        "policy_phase": policy.phase,
                        "physical_readback": last_info.get(
                            "p003_physical_readback", {}
                        ),
                        "raw_collision_forces_n": dict(
                            env._p003_last_contact_forces
                        ),
                    }
                )
            if policy.phase != previous_phase:
                phase_events.append(
                    {
                        "step": step,
                        "from": previous_phase,
                        "to": policy.phase,
                    }
                )
                previous_phase = policy.phase
            if env._p003_tracker.stage != previous_stage:
                stage_events.append(
                    {
                        "step": step,
                        "from": previous_stage,
                        "to": env._p003_tracker.stage,
                    }
                )
                previous_stage = env._p003_tracker.stage
            if step % 15 == 0 or terminated or truncated:
                current = env._p003_last_readback or env._refresh_readback()
                samples.append(
                    {
                        "step": step,
                        "policy_phase": policy.phase,
                        "task_stage": env._p003_tracker.stage,
                        "joint_angles_deg": list(current.joint_angles_deg),
                        "minimum_segment_bottom_z_m": float(
                            current.minimum_segment_bottom_z_m
                        ),
                        "maximum_linear_speed_mps": float(
                            current.maximum_linear_speed_mps
                        ),
                        "maximum_angular_speed_radps": float(
                            current.maximum_angular_speed_radps
                        ),
                        "body_velocities_mps_radps": {
                            label: env._body_velocity(label).tolist()
                            for label in ("s0", "s1", "s2")
                        },
                        "left_s0_contact_force_n": float(
                            current.left_s0_contact_force_n
                        ),
                        "right_s2_contact_force_n": float(
                            current.right_s2_contact_force_n
                        ),
                        "left_s2_contact_force_n": float(
                            current.left_s2_contact_force_n
                        ),
                        "right_s0_contact_force_n": float(
                            current.right_s0_contact_force_n
                        ),
                        "robot_table_contact_force_n": float(
                            current.robot_table_contact_force_n
                        ),
                        "max_penetration_m": float(
                            env._p003_monitored_contact_max_penetration_m
                        ),
                        "legacy_robot_inclusive_max_penetration_m": float(
                            env._p003_max_penetration_m
                        ),
                        "assist_substeps": int(env._p003_assist_substeps),
                        "runtime_policy_version": last_info.get(
                            "p003_runtime_policy_version"
                        ),
                        "physics_policy_version": last_info.get(
                            "p003_physics_policy_version"
                        ),
                        "tabletop_support": last_info.get(
                            "p003_tabletop_support", {}
                        ),
                        "control_phase": env._p003_control_phase,
                        "configure_preparing": bool(
                            env._p003_configure_preparing
                        ),
                        "workspace_preparing": bool(
                            env._p003_workspace_preparing
                        ),
                        "workspace_target_x_m": getattr(
                            policy, "_configure_workspace_target_x", None
                        ),
                        "workspace_count": int(
                            getattr(policy, "_configure_workspace_count", 0)
                        ),
                        "root_x_m": float(current.root_position_m[0]),
                        "object_center_position_m": list(
                            current.object_center_position_m
                        ),
                        "reset_object_center_position_m": list(
                            current.reset_object_center_position_m
                        ),
                        "raw_collision_forces_n": dict(
                            env._p003_last_contact_forces
                        ),
                        "right_actual_tcp_pose_wxyz_m": (
                            env._actual_tcp_pose("right").tolist()
                        ),
                        "left_actual_tcp_pose_wxyz_m": (
                            env._actual_tcp_pose("left").tolist()
                        ),
                        "right_commanded_tcp_pose_wxyz_m": (
                            env._right_tcp_pose.tolist()
                        ),
                        "left_commanded_tcp_pose_wxyz_m": (
                            env._left_tcp_pose.tolist()
                        ),
                        "right_command_pose_wxyz_m": action[:7].tolist(),
                        "left_command_pose_wxyz_m": action[7:14].tolist(),
                    }
                )
            if terminated or truncated:
                break

        final_readback = env._p003_last_readback or env._refresh_readback()
        head = None
        head_key = None
        if isinstance(observation, dict):
            image_observation = observation.get("images", observation)
            if not isinstance(image_observation, dict):
                image_observation = {}
            for key in (
                "ego",
                "random_camera",
                "head",
                "head_camera",
                "camera_0",
            ):
                candidate = image_observation.get(key)
                if isinstance(candidate, np.ndarray) and candidate.ndim >= 3:
                    head = candidate
                    head_key = key
                    break
        camera_metrics = {
            "present": head is not None,
            "observation_key": head_key,
            "shape": list(head.shape) if head is not None else [],
            "mean_rgb": float(np.mean(head)) if head is not None else None,
            "black_fraction": (
                float(np.mean(np.all(head[..., :3] == 0, axis=-1)))
                if head is not None
                else None
            ),
        }
        robot_table_force_by_path: dict[str, float] = {}
        try:
            robot_table_view = env._p003_contact_views["robot_table"]
            force_matrix = env._view_numpy(
                robot_table_view.get_contact_force_matrix(env._sim_dt),
                (
                    int(robot_table_view.sensor_count),
                    int(robot_table_view.filter_count),
                    3,
                ),
            )
            per_filter_force = np.sum(
                np.linalg.norm(force_matrix, axis=-1), axis=0
            )
            filter_paths = env._p003_contact_paths["robot_table"][1]
            robot_table_force_by_path = {
                path: float(force)
                for path, force in zip(filter_paths, per_filter_force)
                if float(force) > 0.0
            }
        except Exception as exc:
            robot_table_force_by_path = {"diagnostic_error": str(exc)}
        support = _tabletop_support_metrics(final_readback)
        center_error_m = float(
            np.linalg.norm(
                np.asarray(final_readback.object_center_position_m[:2])
                - np.asarray(final_readback.reset_object_center_position_m[:2])
            )
        )
        contract_checks = {
            "runtime_policy_version": bool(
                last_info.get("p003_runtime_policy_version")
                == P003_RUNTIME_POLICY_VERSION
            ),
            "joint_target_assist_disabled": bool(
                last_info.get("p003_joint_target_assist_enabled") is False
                and int(env._p003_assist_substeps) == 0
            ),
            "compliant_grasp_disabled": bool(
                last_info.get("p003_compliant_grasp_enabled") is False
                and int(env._p003_grasp_assist_substeps) == 0
            ),
            "vertical_support_disabled": bool(
                last_info.get("p003_vertical_support_enabled") is False
            ),
            "wrong_role_contact_recoverable": bool(
                last_info.get("p003_wrong_role_contact_is_terminal") is False
            ),
            "all_segments_final_table_supported": bool(
                support["final_supported"]
            ),
            "centered": bool(center_error_m <= PLACEMENT_XY_TOLERANCE_M),
            "both_hands_released": bool(
                final_readback.left_all_ruler_contact_force_n
                <= RELEASE_CONTACT_FORCE_N
                and final_readback.right_all_ruler_contact_force_n
                <= RELEASE_CONTACT_FORCE_N
            ),
            "low_velocity": bool(
                final_readback.maximum_linear_speed_mps
                <= STABLE_LINEAR_SPEED_MPS
                and final_readback.maximum_angular_speed_radps
                <= STABLE_ANGULAR_SPEED_RADPS
            ),
            "final_hands_clear": bool(
                final_readback.hand_table_contact_valid
                and final_readback.hand_table_contact_force_n
                <= FINAL_HAND_TABLE_CONTACT_FORCE_N
            ),
            "no_object_state_write_after_action_zero": bool(
                env._p003_object_state_writes_after_action_zero == 0
            ),
        }
        report = {
            "format": "handson.p003.live_scripted_acceptance_v2",
            "seed": int(args.seed),
            "passed": bool(
                env._p003_tracker.success
                and not env._p003_failure_latched
                and not truncated
                and all(contract_checks.values())
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps_executed": int(env.env_step),
            "reset": env._p003_reset.to_dict(),
            "target_prompt": env._target_prompt(),
            "policy": policy.metadata(),
            "phase_events": phase_events,
            "stage_events": stage_events,
            "final_policy_phase": policy.phase,
            "final_task_stage": env._p003_tracker.stage,
            "final_readback": asdict(final_readback),
            "contract_checks": contract_checks,
            "final_tabletop_support": support,
            "final_center_error_m": center_error_m,
            "final_body_poses_wxyz_m": env._body_poses().tolist(),
            "drive_targets_deg": env._p003_drive_targets_deg.tolist(),
            "assist_substeps": int(env._p003_assist_substeps),
            "raw_collision_forces_n": dict(env._p003_last_contact_forces),
            "robot_table_force_by_path_n": robot_table_force_by_path,
            "role_force_by_path_peak_n": role_force_by_path_peak_n,
            "failure_latched": bool(env._p003_failure_latched),
            "failure_reason": env._p003_failure_reason,
            "object_state_writes_after_action_zero": int(
                env._p003_object_state_writes_after_action_zero
            ),
            "max_penetration_m": float(
                env._p003_monitored_contact_max_penetration_m
            ),
            "legacy_robot_inclusive_max_penetration_m": float(
                env._p003_max_penetration_m
            ),
            "camera": camera_metrics,
            "samples": samples,
            "contact_trace": contact_trace,
            "tabletop_trace": tabletop_trace,
            "last_info": {
                key: value
                for key, value in dict(last_info).items()
                if key.startswith("p003_") or key in {"succeed", "failed"}
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(_json_safe(report), sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
