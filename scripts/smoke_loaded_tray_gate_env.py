#!/usr/bin/env python3
"""Launch B320-BC005 and verify reset/step/close over multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handson.sim.isaaclab.loaded_tray_gate_env import (
    HandsOnLoadedTrayGateIsaacLabEnv,
)
from handson.sim.isaaclab.loaded_tray_gate_scene import PAYLOAD_NAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps-per-seed", type=int, default=3)
    parser.add_argument("--isaac-device", default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    return parser.parse_args()


def _emit_report(report: dict[str, object], result_json: Path | None) -> None:
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(report_text + "\n", encoding="utf-8")
    print(report_text, flush=True)


def main() -> int:
    args = _parse_args()
    if args.seeds < 1 or args.steps_per_seed < 1:
        raise ValueError("--seeds and --steps-per-seed must be positive")

    env = HandsOnLoadedTrayGateIsaacLabEnv(
        launch_isaac=True,
        headless=True,
        image_obs=False,
        randomize=True,
        hz=0,
        time_limit=45.0,
        isaac_device=args.isaac_device,
    )
    results: list[dict[str, object]] = []
    last_seed_diagnostics: dict[str, object] = {}
    try:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for seed in range(args.seeds):
            observation, reset_info = env.reset(seed=seed)
            initial_poses = {
                "loaded_tray": observation["state"]["tray_pose"][:3].copy(),
                **{
                    name: observation["state"]["payload_poses"][
                        index * 7 : index * 7 + 3
                    ].copy()
                    for index, name in enumerate(PAYLOAD_NAMES)
                },
            }
            if reset_info["succeed"] or reset_info.get("failed", False):
                raise AssertionError(
                    f"seed {seed}: reset was terminal: {reset_info}"
                )
            if reset_info["task_stage"] != "approach":
                raise AssertionError(
                    f"seed {seed}: pre-satisfied stage {reset_info['task_stage']}"
                )
            if reset_info["left_role_owned"] or reset_info["right_role_owned"]:
                raise AssertionError(f"seed {seed}: reset pre-owned a handle role")
            reset_hand_forces = {
                "left": float(reset_info["left_tray_contact_force_n"]),
                "right": float(reset_info["right_tray_contact_force_n"]),
            }
            reset_gate_forces = {
                "hands": float(reset_info["hand_gate_contact_force_n"]),
                "task_bodies": float(reset_info["task_gate_contact_force_n"]),
                "total": float(reset_info["gate_contact_force_n"]),
            }
            last_seed_diagnostics = {
                "seed": seed,
                "phase": "reset",
                "tray_position_m": observation["state"]["tray_pose"][:3].tolist(),
                "handle_distances_m": {
                    "left": float(reset_info["left_handle_distance_m"]),
                    "right": float(reset_info["right_handle_distance_m"]),
                },
                "hand_contact_forces_n": reset_hand_forces,
                "gate_contact_forces_n": reset_gate_forces,
            }
            if max(reset_hand_forces.values()) > 0.05:
                raise AssertionError(
                    f"seed {seed}: reset hand/tray pre-contact {reset_hand_forces}"
                )
            if max(reset_gate_forces.values()) > 0.05:
                raise AssertionError(
                    f"seed {seed}: reset gate pre-contact {reset_gate_forces}"
                )
            # A reset transform is read before the next ordinary PhysX step,
            # so the contact view is valid but its support impulse is still
            # zero.  Require the exact deck geometry here; the no-op loop
            # below must then produce real nonzero tray support for both
            # payloads before the seed can pass.
            if not all(reset_info["payload_geometrically_on_tray"].values()):
                raise AssertionError(
                    f"seed {seed}: payload reset geometry is off tray: "
                    f"{reset_info['payload_geometrically_on_tray']}"
                )
            if not all(reset_info["payload_support_available"].values()):
                raise AssertionError(
                    f"seed {seed}: payload support views unavailable: "
                    f"{reset_info['payload_support_available']}"
                )
            if float(reset_info["max_gate_penetration_m"]) > 0.0001:
                raise AssertionError(
                    f"seed {seed}: reset gate penetration "
                    f"{reset_info['max_gate_penetration_m']}"
                )
            if not (
                reset_info["left_tray_contact_available"]
                and reset_info["right_tray_contact_available"]
                and reset_info["gate_contact_available"]
            ):
                raise AssertionError(
                    f"seed {seed}: contact views unavailable: {reset_info}"
                )

            info = reset_info
            for step_index in range(args.steps_per_seed):
                observation, _reward, terminated, truncated, info = env.step(action)
                noop_hand_forces = {
                    "left": float(info["left_tray_contact_force_n"]),
                    "right": float(info["right_tray_contact_force_n"]),
                }
                noop_gate_forces = {
                    "hands": float(info["hand_gate_contact_force_n"]),
                    "task_bodies": float(info["task_gate_contact_force_n"]),
                    "total": float(info["gate_contact_force_n"]),
                }
                last_seed_diagnostics = {
                    "seed": seed,
                    "phase": "noop",
                    "noop_step": step_index,
                    "tray_position_m": observation["state"]["tray_pose"][:3].tolist(),
                    "handle_distances_m": {
                        "left": float(info["left_handle_distance_m"]),
                        "right": float(info["right_handle_distance_m"]),
                    },
                    "hand_contact_forces_n": noop_hand_forces,
                    "gate_contact_forces_n": noop_gate_forces,
                }
                if terminated or truncated or info["succeed"] or info.get("failed", False):
                    raise AssertionError(
                        f"seed {seed}: no-op step was terminal: {info}"
                    )
                if info["left_role_owned"] or info["right_role_owned"]:
                    raise AssertionError(
                        f"seed {seed}: no-op step pre-owned a handle role"
                    )
                if max(noop_hand_forces.values()) > 0.05:
                    raise AssertionError(
                        f"seed {seed}: no-op hand/tray pre-contact "
                        f"{noop_hand_forces}"
                    )
                if max(noop_gate_forces.values()) > 0.05:
                    raise AssertionError(
                        f"seed {seed}: no-op gate pre-contact {noop_gate_forces}"
                    )
            final_poses = {
                "loaded_tray": observation["state"]["tray_pose"][:3],
                **{
                    name: observation["state"]["payload_poses"][
                        index * 7 : index * 7 + 3
                    ]
                    for index, name in enumerate(PAYLOAD_NAMES)
                },
            }
            displacements = {
                name: float(np.linalg.norm(final_poses[name] - initial_poses[name]))
                for name in initial_poses
            }
            if max(displacements.values()) > 0.020:
                raise AssertionError(
                    f"seed {seed}: unstable reset displacement {displacements}"
                )
            if not info["payloads_on_tray"]:
                raise AssertionError(f"seed {seed}: payload left tray on no-op")
            if not bool(info["finite_and_in_workspace"]):
                raise AssertionError(f"seed {seed}: nonfinite/out-of-workspace state")
            if float(info["max_gate_penetration_m"]) > 0.0001:
                raise AssertionError(
                    f"seed {seed}: no-op gate penetration "
                    f"{info['max_gate_penetration_m']}"
                )
            results.append(
                {
                    "seed": seed,
                    "stage": info["task_stage"],
                    "max_gate_penetration_m": float(
                        info["max_gate_penetration_m"]
                    ),
                    "max_noop_displacement_m": max(displacements.values()),
                    "payloads_on_tray": bool(info["payloads_on_tray"]),
                    "reset_hand_contact_forces_n": reset_hand_forces,
                    "noop_hand_contact_forces_n": noop_hand_forces,
                    "reset_gate_contact_forces_n": reset_gate_forces,
                    "noop_gate_contact_forces_n": noop_gate_forces,
                    "roles_owned_after_noop": [
                        bool(info["left_role_owned"]),
                        bool(info["right_role_owned"]),
                    ],
                    "contact_views_available": True,
                }
            )
        _emit_report(
            {
                "task": "loaded_tray_through_narrow_gate",
                "seeds": args.seeds,
                "steps_per_seed": args.steps_per_seed,
                "status": "PASS",
                "results": results,
            },
            args.result_json,
        )
    except BaseException as exc:
        _emit_report(
            {
                "task": "loaded_tray_through_narrow_gate",
                "seeds": args.seeds,
                "steps_per_seed": args.steps_per_seed,
                "status": "FAIL",
                "completed_seeds": len(results),
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "last_seed_diagnostics": last_seed_diagnostics,
                "results": results,
            },
            args.result_json,
        )
        raise
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
