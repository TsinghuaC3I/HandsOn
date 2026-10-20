#!/usr/bin/env python3
"""Exercise the production YCB06 reset/retry path in real CUDA PhysX.

Runs without tracker sockets, recording demos, or changing a live collection.
The optional left-arm probe commands only this script's simulated robot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "handson")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=296128758)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-left-arm", action="store_true")
    parser.add_argument("--probe-wakeup", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    os.environ.update(
        HANDSON_RECORD_PHYSX_GROUND_TRUTH="1",
        HANDSON_STABLE_RECORDING_ORIGIN="1",
        HANDSON_ISAAC_IN_PROCESS_HARD_RESET="1",
        HANDSON_ISAACLAB_DEFER_CAMERA_RENDER="1",
        HANDSON_ISAACLAB_DEFER_ARM_RENDER="1",
    )
    from scripts import record_demos_zarr as recorder
    from handson.tasks.t28_place_tableware.config import TaskConfig

    env = TaskConfig().get_native_environment(
        policy_mode=True, seed=args.seed, randomize=True, launch_isaac=True,
        headless=True, image_obs=False, render_mode="rgb_array",
        hand_type="wuji1", arm_type="rokae",
    )
    raw = env.unwrapped
    report = {"seed": args.seed, "requested_resets": args.count, "resets": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2))

    try:
        index = 0
        for ordinal in range(args.count):
            started = time.monotonic()
            events = []
            _, info, index, seed = recorder._reset_recording_episode_with_stable_origin(
                env, {}, sim_backend="isaacsim", task_id=raw.task_name,
                collection_run_seed=args.seed, reset_attempt_index=index,
                seed_resets=True, reset_attempt_observer=events.append,
            )
            origin = info["physical_recording_origin"]
            if origin.get("initial_geometry_gate_passed") is not True:
                raise AssertionError("Production reset did not pass its geometry gate")
            row = {
                "reset": ordinal, "seed": seed,
                "seconds": time.monotonic() - started,
                "attempts": origin["reset_stabilization_attempt"],
                "physics_steps": origin["physics_steps"],
                "translation_drift_m": origin["reset_max_translation_drift_m"],
                "rotation_drift_rad": origin["reset_max_rotation_drift_rad"],
                "penetration_m": origin["initialization_contact_summary"]["maximum_penetration_depth_m"],
                "group_yaw_deg": raw._ycb06_reset_group_yaw,
                "object_poses": {n: raw._physics_body_pose(n).tolist()
                                 for n in ("plate", "fork", "knife", "spoon", "mug")},
                "contact_filters": {n: int(v.filter_count)
                                    for n, v in raw._ycb06_contact_views.items()},
                "rejected_attempts": [event for event in events
                                      if event.get("outcome") == "rejected_unstable_origin"],
            }
            report["resets"].append(row)
            save()
            print("YCB06_RESET_PASS", json.dumps(row), flush=True)

        if args.probe_left_arm:
            from handson.sim.isaaclab.abc_collection_env import _matrix_to_quat_wxyz
            right_before, left_before = raw.get_end_effector_pose_matrix()
            left_target = left_before.copy()
            left_target[0, 3] -= 0.025
            left_action = np.concatenate([
                left_target[:3, 3], _matrix_to_quat_wxyz(left_target[:3, :3]),
                np.asarray(raw._left_hand).ravel(),
            ])
            for _ in range(30):
                raw.step({"right": np.zeros_like(left_action), "left": left_action})
            right_after, left_after = raw.get_end_effector_pose_matrix()
            report["left_arm_probe"] = {
                "left_displacement_m": float(np.linalg.norm(left_after[:3, 3] - left_before[:3, 3])),
                "right_displacement_m": float(np.linalg.norm(right_after[:3, 3] - right_before[:3, 3])),
            }
            save()
            if report["left_arm_probe"]["left_displacement_m"] < 0.005:
                raise AssertionError("Left arm failed to follow its Cartesian command")
        if args.probe_wakeup:
            import omni.physx
            import omni.usd
            from pxr import PhysicsSchemaTools

            report["wakeup_probe"] = {}
            for name in ("plate", "fork", "knife", "spoon", "mug"):
                # Each target starts at a newly accepted production origin.
                # Only the separate impactor is moved/woken explicitly.
                _, _, index, _ = recorder._reset_recording_episode_with_stable_origin(
                    env, {}, sim_backend="isaacsim", task_id=raw.task_name,
                    collection_run_seed=args.seed, reset_attempt_index=index,
                    seed_resets=True,
                )
                simulation = omni.physx.get_physx_simulation_interface()
                stage_id = omni.usd.get_context().get_stage_id()
                physics_view = raw._robots["right"]._physics_sim_view
                target_path = raw._find_rigid_body_prim(name).GetPath()
                body_id = PhysicsSchemaTools.sdfPathToInt(target_path)
                before = raw._physics_body_pose(name).copy()
                sleeping_before = simulation.is_sleeping(stage_id, body_id)
                mover_name = "plate" if name == "mug" else "mug"
                mover_path = str(raw._find_rigid_body_prim(mover_name).GetPath())
                view = physics_view.create_rigid_body_view(mover_path)
                contact = physics_view.create_rigid_contact_view(
                    str(target_path), filter_patterns=[mover_path],
                    max_contact_data_count=128,
                )
                transforms = view.get_transforms().clone()
                transforms[0, :3] = raw._torch.as_tensor(
                    before[:3], dtype=transforms.dtype, device=transforms.device,
                )
                direction = np.array([1.0, 0.0])
                distance = 0.22
                if name in ("fork", "knife", "spoon"):
                    # Approach the exposed end, away from the fixed riser.
                    direction = before[:2] - raw._ycb06_reset_fixture_xy
                    direction /= np.linalg.norm(direction)
                    distance = 0.12
                transforms[0, :2] += raw._torch.as_tensor(
                    distance * direction, dtype=transforms.dtype,
                    device=transforms.device,
                )
                transforms[0, 2] = raw._current_table_top_z() + (
                    0.0418 if mover_name == "mug" else 0.0127
                )
                velocity = raw._torch.zeros_like(view.get_velocities())
                velocity[0, :2] = raw._torch.as_tensor(
                    -1.5 * direction, dtype=velocity.dtype, device=velocity.device,
                )
                indices = raw._torch.arange(
                    view.count, dtype=raw._torch.long, device=velocity.device,
                )
                view.set_transforms(transforms, indices)
                view.set_velocities(velocity, indices)
                view.wake_up(indices.cpu())
                peak_force = peak_motion = 0.0
                for _ in range(90):
                    raw._step_isaaclab()
                    force = contact.get_contact_force_matrix(raw._physics_dt)
                    peak_force = max(peak_force, float(force.norm().item()))
                    after = raw._physics_body_pose(name)
                    peak_motion = max(peak_motion, float(np.linalg.norm(after[:3] - before[:3])))
                result = {
                    "sleeping_before_contact": sleeping_before,
                    "contact_force_n": peak_force,
                    "maximum_displacement_m": peak_motion,
                }
                report["wakeup_probe"][name] = result
                save()
                if not sleeping_before or peak_force <= 0 or peak_motion < 0.0001:
                    raise AssertionError(f"{name} did not respond to physical contact: {result}")
        report["passed"] = True
        save()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
