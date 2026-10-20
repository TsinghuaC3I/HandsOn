#!/usr/bin/env python3
"""Isolated native contact-sensor check; never produces a demonstration.

After a normal stable reset, drop the real bottle from a collision-clear pose
above each stationary palm. This diagnostic initial placement is not an
operator action or evidence of full-task completion. PhysX alone advances the
subsequent contact, and a normal reset verifies that release can recover.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import traceback

import numpy as np

from live_probe import HandsOnXY4IsaacLabEnv, TASK_NAME, live_shapes, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=90)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from collection_ui.server import ISAAC_COLLECTION_REALTIME_ENV
    os.environ.update(ISAAC_COLLECTION_REALTIME_ENV)
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    report = {
        "isaacsim": importlib.metadata.version("isaacsim"),
        "scope": "native bottle contact views and release recovery only",
        "diagnostic_object_placement": True,
        "full_task_success_verified": False,
        "rows": [],
    }
    assert report["isaacsim"].startswith("5.1.")
    env = None
    try:
        env = HandsOnXY4IsaacLabEnv(
            launch_isaac=True, image_obs=False, headless=True,
            render_mode="rgb_array", randomize=False, seed=0,
            isaac_device="cuda:0", hand_type="wuji1", arm_type="rokae",
        )
        from handson.sim.isaaclab.recording_origin import (
            stable_recording_origin_kwargs_for_task, stabilize_recording_origin,
        )
        report["online_camera_names"] = list(env._cameras)
        assert not env._cameras
        report["contact_views"] = {}
        for side, view in (("left", env._hs005_contact_views.get("left_bottle")),
                           ("right", env._xy4_right_bottle_contact_view)):
            expected_paths = env._hs005_hand_body_paths(side)
            assert view is not None and view.sensor_count == 1
            assert view.filter_count == len(expected_paths)
            report["contact_views"][side] = {
                "sensor_count": view.sensor_count, "filter_count": view.filter_count,
                "expected_filter_paths": expected_paths,
            }
        for side in ("left", "right"):
            obs, info = env.reset(seed=0)
            obs, info = stabilize_recording_origin(
                env, obs, info, **stable_recording_origin_kwargs_for_task(TASK_NAME),
            )
            initial = env._bottle_release_metrics()
            assert initial["xy4_bottle_released"], initial
            inventory = live_shapes(env, f"release_probe_{side}")
            palms = [r for r in inventory if r["owner"] and
                     r["owner"].endswith(f"/{side}_palm_link")]
            hand = [r for r in inventory if r["owner"] and
                    f"/{side.capitalize()}Robot/" in r["owner"] and
                    (f"/{side}_finger" in r["owner"] or r in palms)]
            bottle_path = env._b320_spawn_result.body_paths["hs005_bottle"]
            bottle_shapes = [r for r in inventory if r["owner"] == bottle_path]
            assert palms and hand and bottle_shapes
            # Choose the largest palm collider in XY; start above the entire
            # hand so the diagnostic placement has no hand/bottle overlap.
            palm = max(palms, key=lambda r: np.prod(np.diff(np.asarray(r["bounds"])[:, :2], axis=0)))
            drop = env._physics_body_pose("hs005_bottle").copy()
            low = np.min([r["bounds"][0] for r in bottle_shapes], axis=0)
            high = np.max([r["bounds"][1] for r in hand], axis=0)
            drop[:2] = np.mean(palm["bounds"], axis=0)[:2]
            drop[2] += high[2] + .015 - low[2]
            row = {
                "side": side, "origin": info["physical_recording_origin"],
                "initial": initial, "drop_pose_wxyz": drop, "samples": [],
            }
            report["rows"].append(row)
            env._set_task_body_world_pose(env._task_body_views["hs005_bottle"], drop)
            for step in range(args.steps):
                env.step_with_current_joint_targets()
                sample = env._bottle_release_metrics()
                sample["step"] = step
                row["samples"].append(sample)
                assert sample[f"xy4_{side}_bottle_contact_readable"]
            contacted = [s for s in row["samples"] if s[f"xy4_{side}_bottle_contact_force_n"] > 1e-6]
            row["contact_frames"] = len(contacted)
            row["peak_force_n"] = max(s[f"xy4_{side}_bottle_contact_force_n"] for s in row["samples"])
            assert contacted, f"No physical {side}-hand/bottle contact in diagnostic drop"
            assert all(not s["xy4_bottle_released"] for s in contacted)
            env.reset(seed=0)
            for _ in range(10):
                env.step_with_current_joint_targets()
            row["after_reset"] = env._bottle_release_metrics()
            assert row["after_reset"]["xy4_bottle_released"]
            assert not env._xy4_sequence.x_complete and not env._ka021_water_filled
            write(output / "report.json", report)
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write(output / "report.json", report)
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()


if __name__ == "__main__":
    main()
