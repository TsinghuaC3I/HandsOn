#!/usr/bin/env python3
"""Real Isaac contact-channel/reset regression; never generates success demos."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def observation_errors(space, value, prefix=""):
    if hasattr(space, "spaces"):
        errors = []
        if set(space.spaces) != set(value):
            errors.append({"path": prefix, "missing": sorted(set(space.spaces)-set(value)),
                           "extra": sorted(set(value)-set(space.spaces))})
        for key in space.spaces.keys() & value.keys():
            errors.extend(observation_errors(space[key], value[key], prefix + "." + key))
        return errors
    if space.contains(value):
        return []
    array = np.asarray(value)
    return [{"path": prefix, "shape": array.shape, "expected_shape": space.shape,
             "dtype": str(array.dtype), "expected_dtype": str(space.dtype),
             "nonfinite_count": int(np.count_nonzero(~np.isfinite(array))),
             "min": float(np.min(array)) if array.size else None,
             "max": float(np.max(array)) if array.size else None}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2303, 17, 991])
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    assert importlib.metadata.version("isaacsim").startswith("5.1.")
    assert os.environ.get("HANDSON_ENABLE_ISAACSIM_NEWTON", "0") == "0"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    from handson.tasks.t35_rotate_propeller.config import TaskConfig
    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin, stable_recording_origin_kwargs_for_task,
    )
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "handson/handson/sim/isaaclab/ycb03_airplane_env.py",
        "handson/handson/sim/isaaclab/abc_collection_env.py",
        "handson/handson/sim/isaaclab/bimanual_assembly_env.py",
        "handson/handson/tasks/t35_rotate_propeller/semantics.py",
        "handson/handson/tasks/t35_rotate_propeller/config.py",
        "handson/handson/sim/isaaclab/task_preview.py",
        "handson/handson/evaluation/native.py",
        "scripts/record_demos_zarr.py",
    ]
    report = {
        "scope": "live_idle_contact_integrity_and_sensor_fault_reset_regression",
        "success_demonstration": False,
        "isaacsim": importlib.metadata.version("isaacsim"),
        "source_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths},
        "seeds": [],
        "pass": False,
    }
    env = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        env = TaskConfig().get_native_environment(
            policy_mode=True, launch_isaac=True, render_mode="human",
            image_obs=False, randomize=True, hand_type="wuji1",
        )
        raw = env.unwrapped
        report["timing"] = {
            "control_dt": raw.control_dt, "physics_dt": raw.physics_dt,
            "physics_substeps_per_control": raw._n_substeps,
        }
        assert np.isclose(raw.control_dt, 1.0 / 30.0)
        assert np.isclose(raw.physics_dt, 1.0 / 120.0) and raw._n_substeps == 4
        from pxr import PhysxSchema, UsdPhysics
        airplane_prim = raw._find_rigid_body_prim("ycb03_airplane")
        assert airplane_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        articulation_api = PhysxSchema.PhysxArticulationAPI(airplane_prim)
        report["airplane_solver_iterations"] = {
            "position": articulation_api.GetSolverPositionIterationCountAttr().Get(),
            "velocity": articulation_api.GetSolverVelocityIterationCountAttr().Get(),
        }
        for seed in args.seeds:
            obs, info = env.reset(seed=seed)
            obs, info = stabilize_recording_origin(
                env, obs, info, **stable_recording_origin_kwargs_for_task(raw.task_name)
            )
            origin = info["physical_recording_origin"]
            assert origin["initial_geometry_gate_passed"], origin
            assert origin["gate_robot_velocity_stable"], origin
            assert origin["gate_task_velocity_stable"] or origin["gate_task_static_pose_stable"], origin
            samples = []
            frame_durations = []
            previous_time = float(np.asarray(obs["physx_sim_time"]).ravel()[0])
            for _ in range(args.steps):
                obs, _, terminated, truncated, step_info = raw.step_with_current_joint_targets()
                current_time = float(np.asarray(obs["physx_sim_time"]).ravel()[0])
                frame_durations.append(current_time - previous_time)
                assert abs(frame_durations[-1] - 1.0 / 30.0) < 1.0e-6
                previous_time = current_time
                s = raw._ycb03_last_sample
                assert s is not None
                samples.append(dataclasses.asdict(s))
                assert s.finite and s.contacts_valid, (samples[-1], raw._task_info())
                assert not terminated and not truncated, step_info
                assert not raw._compute_failure(), raw._task_info()
                assert not raw._compute_success_now(), "idle airplane must not succeed"
                # The task's public interface is the TaskConfig adapter. ABC
                # keeps additional internal diagnostics for unrelated tasks;
                # they are not policy/recorder fields for this task.
                errors = observation_errors(env.observation_space, env.observation(obs))
                if errors:
                    report["observation_errors"] = errors
                    raise AssertionError(f"live observation violates declared schema: {errors}")
            row = {
                "seed": seed, "control_steps": len(samples),
                "contact_valid_samples": sum(s["contacts_valid"] for s in samples),
                "actual_frame_dt_range": [min(frame_durations), max(frame_durations)],
                "initialization": origin,
                "last_sample": samples[-1],
                "contact_validity": raw._ycb03_contact_validity.copy(),
                "maximum_root_linear_speed_mps": max(s["linear_speed_m_s"] for s in samples),
                "maximum_root_angular_speed_radps": max(s["angular_speed_rad_s"] for s in samples),
            }
            report["seeds"].append(row)
            print("CONTACT_INTEGRITY_SEED_PASS", seed, len(samples), flush=True)

        # Deliberately remove one sensor handle, without changing any body,
        # joint, action, semantic counter, or success threshold.
        release_views = raw._ycb03_any_contact_views["left"]
        removed = release_views.pop()
        try:
            raw.step_with_current_joint_targets()
            assert raw._compute_failure()
            assert not raw._compute_success_now()
            assert raw._task_info()["terminal_reason"] == "invalid_airplane_contact_state"
            assert not raw._task_info()["ycb03_hands_released"]
        finally:
            release_views.append(removed)
        raw.step_with_current_joint_targets()
        assert raw._compute_failure(), "restoring a sensor cannot clear the terminal error"
        obs, info = env.reset(seed=args.seeds[0])
        obs, info = stabilize_recording_origin(
            env, obs, info, **stable_recording_origin_kwargs_for_task(raw.task_name)
        )
        report["post_fault_reset_initialization"] = info["physical_recording_origin"]
        raw.step_with_current_joint_targets()
        assert raw._ycb03_last_sample.contacts_valid
        assert not raw._compute_failure()
        report["live_missing_sensor_rejected_and_reset_recovered"] = True
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=json_value) + "\n")
        print("CONTACT_INTEGRITY_REPORT", str(args.output), "pass=", report["pass"], flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
