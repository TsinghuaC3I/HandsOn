"""Fresh-process native-scene reset/stability smoke for P0-03."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from handson.sim.isaaclab.p003_collection_env import HandsOnP003IsaacLabEnv
from handson.sim.isaaclab.p003_contract import (
    P003_RUNTIME_POLICY_VERSION,
    TASK_NAME,
    _tabletop_support_metrics,
)


def _direction_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual /= np.linalg.norm(actual)
    expected /= np.linalg.norm(expected)
    return math.degrees(
        math.acos(float(np.clip(np.dot(actual, expected), -1.0, 1.0)))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = HandsOnP003IsaacLabEnv(
        TASK_NAME,
        seed=args.seed,
        launch_isaac=True,
        headless=True,
        image_obs=False,
        render_mode="rgb_array",
        randomize=True,
        physics_backend="physx",
        isaac_device="cpu",
        time_limit=30.0,
    )
    try:
        initial = env._refresh_readback()
        initial_poses = env._body_poses()
        reset = env._p003_reset
        expected_initial_directions = np.asarray(
            [
                [1.0, 0.0],
                [
                    math.cos(math.radians(initial.joint_angles_deg[0])),
                    math.sin(math.radians(initial.joint_angles_deg[0])),
                ],
                [
                    math.cos(
                        math.radians(
                            initial.joint_angles_deg[0]
                            + initial.joint_angles_deg[1]
                            + 180.0
                        )
                    ),
                    math.sin(
                        math.radians(
                            initial.joint_angles_deg[0]
                            + initial.joint_angles_deg[1]
                            + 180.0
                        )
                    ),
                ],
            ],
            dtype=np.float64,
        )
        direction_errors = [
            _direction_error_deg(actual, expected)
            for actual, expected in zip(
                initial.segment_directions_task_xy, expected_initial_directions
            )
        ]
        terminal = False
        last_info: dict[str, object] = {}
        for _ in range(int(args.steps)):
            _obs, _reward, terminated, truncated, info = env.step(
                np.zeros(54, dtype=np.float64)
            )
            last_info = dict(info)
            terminal = bool(terminated or truncated)
            if terminal:
                break
        final = env._refresh_readback()
        final_poses = env._body_poses()
        left_actual_tcp = env._actual_tcp_pose("left")
        right_actual_tcp = env._actual_tcp_pose("right")
        tcp_distances = {
            "left_to_s2_m": float(
                np.linalg.norm(left_actual_tcp[:3] - final_poses[2, :3])
            ),
            "right_to_s0_m": float(
                np.linalg.norm(right_actual_tcp[:3] - final_poses[0, :3])
            ),
        }
        initial_support = _tabletop_support_metrics(initial)
        final_support = _tabletop_support_metrics(final)
        no_action_joint_drift_deg = np.abs(
            np.asarray(final.joint_angles_deg, dtype=np.float64)
            - np.asarray(initial.joint_angles_deg, dtype=np.float64)
        )
        checks = {
            "contact_contract_ready": bool(env._p003_contact_contract_ready),
            "finite_initial": bool(initial.finite_state),
            "finite_final": bool(final.finite_state),
            "initial_joint_reset": bool(
                np.max(
                    np.abs(
                        np.asarray(initial.joint_angles_deg)
                        - np.asarray([reset.joint_1_deg, reset.joint_2_deg])
                    )
                )
                <= 2.0
            ),
            "initial_direction_contract": bool(max(direction_errors) <= 1.0),
            "all_segments_initially_table_supported": bool(
                initial_support["final_supported"]
            ),
            "all_segments_finally_table_supported": bool(
                final_support["final_supported"]
            ),
            "no_action_joint_drift_within_one_degree": bool(
                np.max(no_action_joint_drift_deg) <= 1.0
            ),
            "no_terminal": not terminal,
            "no_failure": not bool(env._p003_failure_latched),
            "runtime_policy_version": bool(
                last_info.get("p003_runtime_policy_version")
                == P003_RUNTIME_POLICY_VERSION
            ),
            "passive_hinges_no_joint_target_assist": bool(
                last_info.get("p003_joint_target_assist_enabled") is False
                and not bool(env._p003_assist_active)
                and int(env._p003_assist_substeps) == 0
            ),
            "no_compliant_or_vertical_tether": bool(
                last_info.get("p003_compliant_grasp_enabled") is False
                and last_info.get("p003_vertical_support_enabled") is False
                and int(env._p003_grasp_assist_substeps) == 0
            ),
            "wrong_role_contact_is_recoverable": bool(
                last_info.get("p003_wrong_role_contact_is_terminal") is False
            ),
            "passive_drive_target_unchanged": bool(
                np.allclose(
                    env._p003_drive_targets_deg,
                    [reset.joint_1_deg, reset.joint_2_deg],
                    atol=1.0e-6,
                    rtol=0.0,
                )
            ),
            "penetration_within_contract": bool(
                env._p003_monitored_contact_max_penetration_m <= 0.005
            ),
            "no_object_state_write_after_action_zero": bool(
                env._p003_object_state_writes_after_action_zero == 0
            ),
            "target_prompt_bound": bool(
                reset.target_key in env._target_prompt()
                or reset.target.title.lower().split()[0]
                in env._target_prompt().lower()
            ),
        }
        report = {
            "format": "handson.p003.runtime_reset_smoke_v2",
            "seed": int(args.seed),
            "steps_requested": int(args.steps),
            "terminal": terminal,
            "passed": bool(all(checks.values())),
            "checks": checks,
            "reset": reset.to_dict(),
            "target_prompt": env._target_prompt(),
            "initial_readback": asdict(initial),
            "final_readback": asdict(final),
            "initial_tabletop_support": initial_support,
            "final_tabletop_support": final_support,
            "initial_body_poses_wxyz_m": initial_poses.tolist(),
            "final_body_poses_wxyz_m": final_poses.tolist(),
            "initial_direction_errors_deg": direction_errors,
            "no_action_joint_drift_deg": no_action_joint_drift_deg.tolist(),
            "left_actual_tcp_pose_wxyz_m": left_actual_tcp.tolist(),
            "right_actual_tcp_pose_wxyz_m": right_actual_tcp.tolist(),
            "left_commanded_tcp_pose_wxyz_m": env._left_tcp_pose.tolist(),
            "right_commanded_tcp_pose_wxyz_m": env._right_tcp_pose.tolist(),
            "tcp_role_distances": tcp_distances,
            "drive_targets_deg": env._p003_drive_targets_deg.tolist(),
            "contact_forces_n": dict(env._p003_last_contact_forces),
            "max_penetration_m": float(
                env._p003_monitored_contact_max_penetration_m
            ),
            "legacy_robot_inclusive_max_penetration_m": float(
                env._p003_max_penetration_m
            ),
            "failure_reason": env._p003_failure_reason,
            "last_info": {
                key: value
                for key, value in last_info.items()
                if key.startswith("p003_") or key in {"succeed", "failed"}
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
