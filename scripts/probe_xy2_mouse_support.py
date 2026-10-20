"""Isolated PhysX support diagnostic; never records a successful XY2 episode.

Initialize the mouse over its pad, lift its articulation once by 25 mm, then
let PhysX settle it. This tests contact readback, not a manipulation policy.
Use the authorized single-GPU launcher and the ordinary Isaac Sim 5.1 runtime.
"""

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import traceback

import numpy as np

from handson.sim.isaaclab.xy2_collection_env import HandsOnXY2IsaacLabEnv


class SupportProbe(HandsOnXY2IsaacLabEnv):
    def _initial_laptop_reset(self, seed):
        result = super()._initial_laptop_reset(seed)
        self._xy2_layout = replace(self._xy2_layout, mouse_xy=self._xy2_layout.pad_xy)
        return result


def sample(env):
    points, forces = env._contact_samples("xy2_mouse_table")
    legacy = env._c05_mouse_click_metrics(update_failure=False)
    metrics = env._xy2_mouse_click_metrics(update_failure=False)
    return {
        "mouse_pose": env._poses["mouse"].tolist(),
        "pad_pose": env._poses["mousepad"].tolist(),
        "points": points.tolist(), "forces_n": forces.tolist(),
        "source_on_pad": bool(legacy["c05_mouse_on_pad"]),
        "supported": bool(metrics["xy2_mouse_supported"]),
        "xy2_on_pad": bool(metrics["c05_mouse_on_pad"]),
        "bottom_height_error_m": metrics.get("xy2_mouse_bottom_height_error_m"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "diagnostic initialized poses and one root lift; no demonstration",
              "full_task_success_verified": False, "passed": False}
    env = None
    status = 1
    try:
        report["isaacsim"] = importlib.metadata.version("isaacsim")
        assert report["isaacsim"].startswith("5.1."), report["isaacsim"]
        env = SupportProbe(launch_isaac=True, image_obs=False, headless=True,
                           seed=0, target_object="60", randomize=False,
                           isaac_device="cuda:0", hand_type="wuji1", arm_type="rokae")
        env.reset(seed=0)
        from pxr import PhysxSchema

        report["ground_truth_enabled"] = bool(env._physx_ground_truth_enabled)
        report["mouse_contact_report_enabled"] = env._find_rigid_body_prim("mouse").HasAPI(
            PhysxSchema.PhysxContactReportAPI,
        )
        assert report["mouse_contact_report_enabled"]
        report["table_contact_report_enabled"] = env._find_rigid_body_prim("table").HasAPI(
            PhysxSchema.PhysxContactReportAPI,
        )
        assert report["table_contact_report_enabled"]
        report["source_collision_vertex_count"] = len(env._xy2_mouse_collision_vertices_local)
        command = np.concatenate([env._right_tcp_pose, env._left_tcp_pose,
                                  env._right_hand, env._left_hand])
        view = env._fu015_contact_views["xy2_mouse_table"]
        report["contact_view"] = {
            "sensor_count": int(view.sensor_count), "filter_count": int(view.filter_count),
            "paths": env._fu015_contact_paths["xy2_mouse_table"],
        }
        assert int(view.sensor_count) == int(view.filter_count) == 1
        report["physics_dt"] = env._physics_dt
        report["control_dt"] = env._control_dt
        for _ in range(10):
            env.step(command)
        report["settled"] = sample(env)
        assert report["settled"]["xy2_on_pad"], report["settled"]

        prim = env._find_rigid_body_prim("mouse")
        bodies = {entry[0]: entry[1:5] for entry in env._task_reset_body_states}
        articulations = {entry[0]: entry[1:5] for entry in env._task_reset_articulation_states}
        kind, mouse_view, indices, _, velocity = env._reset_transform_state_for_path(
            str(prim.GetPath()), bodies, articulations,
        )
        assert kind == "articulation", kind
        lifted = mouse_view.get_root_transforms().clone()
        lifted[:, 2] += .025
        mouse_view.set_root_transforms(lifted, indices)
        mouse_view.set_root_velocities(env._torch.zeros_like(velocity), indices)
        report["fall_samples"] = []
        for _ in range(25):
            env.step(command)
            report["fall_samples"].append(sample(env))
        assert any(row["source_on_pad"] and not row["xy2_on_pad"]
                   for row in report["fall_samples"]), "No unsupported source-predicate positive observed"
        assert report["fall_samples"][-1]["xy2_on_pad"], "Support did not recover after settling"
        env.reset(seed=0)
        command = np.concatenate([env._right_tcp_pose, env._left_tcp_pose,
                                  env._right_hand, env._left_hand])
        for _ in range(10):
            env.step(command)
        report["after_reset"] = sample(env)
        assert report["after_reset"]["xy2_on_pad"]
        assert env._xy2_progress.phase == 0 and not env._c05_failed_early_click
        report["passed"] = True
        status = 0
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
