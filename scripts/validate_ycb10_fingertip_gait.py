#!/usr/bin/env python3
"""Real Isaac reset/contact regression; this does not generate a success demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[104729, 104759, 104761]
    )
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    assert importlib.metadata.version("isaacsim").startswith("5.1.")
    assert os.environ.get("HANDSON_ENABLE_ISAACSIM_NEWTON", "0") == "0"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin,
        stable_recording_origin_kwargs_for_task,
    )
    from handson.sim.isaaclab.ycb10_semantics import YCB10SemanticState
    from handson.tasks.t33_reposition_marker_grasp.config import TaskConfig

    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "handson/handson/sim/isaaclab/ycb10_semantics.py",
        "handson/handson/sim/isaaclab/abc_collection_env.py",
        "handson/handson/sim/isaaclab/bimanual_assembly_env.py",
        "handson/handson/tasks/t33_reposition_marker_grasp/config.py",
        "handson/handson/sim/isaaclab/task_preview.py",
        "handson/handson/evaluation/native.py",
        "scripts/record_demos_zarr.py",
        "scripts/validate_ycb10_fingertip_gait.py",
    ]
    report = {
        "scope": "live_idle_distal_contact_integrity_and_sensor_fault_reset",
        "success_demonstration": False,
        "isaacsim": importlib.metadata.version("isaacsim"),
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths
        },
        "seeds": [],
        "pass": False,
    }
    xml_root = root / "handson/handson/sim/envs/xmls"
    objects_xml = xml_root / "t33_reposition_marker_grasp_objects.xml"
    asset_paths = {
        objects_xml,
        xml_root / "ycb10_collection_scene.xml",
        xml_root / "arena_arm_hand_33_reposition_marker_grasp_rokae_wuji1.xml",
        xml_root / "lightwheel_ycb/large_marker/model.xml",
    }
    for element in ET.parse(objects_xml).getroot().iter():
        if element.tag in {"mesh", "texture"} and element.get("file"):
            asset_paths.add(xml_root / element.get("file"))
    report["task_asset_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(asset_paths)
    }
    env = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        env = TaskConfig().get_native_environment(
            policy_mode=True,
            launch_isaac=True,
            render_mode="human",
            image_obs=False,
            randomize=True,
            hand_type="wuji1",
        )
        raw = env.unwrapped
        assert np.isclose(raw._control_dt, 1.0 / 30.0)
        assert np.isclose(raw._physics_dt, 1.0 / 120.0)
        assert np.isclose(raw._sim_dt, 1.0 / 120.0)
        assert raw._n_substeps == 4
        report["timing_contract"] = {
            "control_dt": raw._control_dt,
            "physics_dt": raw._physics_dt,
            "physics_substeps": raw._n_substeps,
        }
        # This branch's formal recorder defers cameras by constructing the
        # public task with image_obs=False; no newer private flags are needed.
        assert not raw.image_obs
        assert os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] == "1"
        assert os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] == "0"
        assert not raw._cameras, "online validation must create no recording cameras"
        paths = raw._c04_right_hand_body_paths()
        report["fingertip_paths"] = {
            str(index): raw._ycb10_fingertip_paths(paths, index)
            for index in range(1, 6)
        }
        for index in range(1, 6):
            view = raw._ycb10_finger_contact_views[index]
            assert view.sensor_count == 1
            assert view.filter_count == len(report["fingertip_paths"][str(index)]) > 0
        report["online_camera_handles"] = sorted(raw._cameras)
        for seed in args.seeds:
            obs, info = env.reset(seed=seed)
            reset_pose = raw._physics_body_pose("ycb10_marker").copy()
            obs, info = stabilize_recording_origin(
                env, obs, info, **stable_recording_origin_kwargs_for_task(raw.task_name)
            )
            origin = info["physical_recording_origin"]
            assert origin["initial_geometry_gate_passed"], origin
            assert origin["gate_robot_velocity_stable"], origin
            assert (
                origin["gate_task_velocity_stable"]
                or origin["gate_task_static_pose_stable"]
            ), origin
            max_linear = max_angular = 0.0
            previous_sim_time = None
            sim_time_deltas = []
            for _ in range(args.steps):
                obs, _, terminated, truncated, step_info = (
                    raw.step_with_current_joint_targets()
                )
                assert np.all(raw._ycb10_last_contact_validity == 1), raw._task_info()
                assert not terminated and not truncated, step_info
                assert not raw._compute_failure(), raw._task_info()
                assert not raw._compute_success_now(), "idle marker must not succeed"
                recorded_time = obs.get("physx_sim_time", obs.get("state", {}).get("physx_sim_time"))
                assert recorded_time is not None, "missing action-aligned PhysX clock"
                sim_time = float(np.asarray(recorded_time).reshape(-1)[0])
                if previous_sim_time is not None:
                    delta = sim_time - previous_sim_time
                    assert np.isclose(delta, 1.0 / 30.0, rtol=0.0, atol=1e-7), delta
                    sim_time_deltas.append(delta)
                previous_sim_time = sim_time
                adapted = env.observation(obs)
                assert env.observation_space.contains(adapted), {
                    key: (np.asarray(value).shape, str(np.asarray(value).dtype))
                    for key, value in adapted.items()
                }
                hidden = raw._ycb10_state.hidden_vector()
                assert np.array_equal(
                    YCB10SemanticState.from_hidden_vector(hidden).hidden_vector(),
                    hidden,
                )
                velocity = raw._physics_body_velocity("ycb10_marker")
                max_linear = max(max_linear, float(np.linalg.norm(velocity[:3])))
                max_angular = max(max_angular, float(np.linalg.norm(velocity[3:])))
            report["seeds"].append(
                {
                    "seed": seed,
                    "control_steps": args.steps,
                    "reset_pose": reset_pose,
                    "initialization": origin,
                    "last_marker_pose": raw._physics_body_pose("ycb10_marker"),
                    "contact_validity": raw._ycb10_last_contact_validity.copy(),
                    "maximum_linear_speed_mps": max_linear,
                    "maximum_angular_speed_radps": max_angular,
                    "control_step_sim_time_range": [min(sim_time_deltas), max(sim_time_deltas)],
                    "last_metrics": dict(raw._ycb10_state.last_metrics),
                }
            )
            print("YCB10_CONTACT_SEED_PASS", seed, args.steps, flush=True)

        # Remove only a required sensor handle. Never write physical state or
        # semantic progress to fabricate a positive task trajectory.
        removed = raw._ycb10_finger_contact_views.pop(2)
        try:
            raw.step_with_current_joint_targets()
            assert raw._compute_failure()
            assert not raw._compute_success_now()
            assert raw._ycb10_state.first_terminal_reason == "invalid_contact_state"
            assert raw._ycb10_last_contact_validity[1] == 0
        finally:
            raw._ycb10_finger_contact_views[2] = removed
        raw.step_with_current_joint_targets()
        assert raw._compute_failure(), "sensor repair cannot clear the terminal reason"
        obs, info = env.reset(seed=args.seeds[0])
        repeated_pose = raw._physics_body_pose("ycb10_marker").copy()
        obs, info = stabilize_recording_origin(
            env, obs, info, **stable_recording_origin_kwargs_for_task(raw.task_name)
        )
        raw.step_with_current_joint_targets()
        assert np.all(raw._ycb10_last_contact_validity == 1)
        assert not raw._compute_failure()
        assert raw._ycb10_state.earned_axial == 0.0
        assert not raw._ycb10_state.transfer_active
        assert not raw._ycb10_state.final_samples
        report["post_fault_reset_initialization"] = info["physical_recording_origin"]
        report["same_seed_post_step_pose_delta"] = float(
            np.max(np.abs(repeated_pose - report["seeds"][0]["reset_pose"]))
        )
        report["position_span_xy"] = np.ptp(
            np.asarray([row["reset_pose"][:2] for row in report["seeds"]]), axis=0
        )
        assert np.all(report["position_span_xy"] > 0), report["position_span_xy"]
        report["live_missing_sensor_rejected_and_reset_recovered"] = True
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=json_value) + "\n")
        print(
            "YCB10_CONTACT_REPORT",
            str(args.output),
            "pass=",
            report["pass"],
            flush=True,
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
