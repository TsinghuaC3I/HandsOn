"""Post-fix native contact/interface check and optional public-action grasp trial.

No task-body pose or velocity setters are used. Preview cameras are enabled only
in this diagnostic process; this does not create a formal recorded episode.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

import numpy as np

from native_smoke import serial

TASK = Path(__file__).resolve().parent
OUTPUT = TASK / "evidence" / os.environ.get("GEN1_SUPERVISOR_OUTPUT", "supervisor_raw/contact_readback")


def main():
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    from PIL import Image
    from scipy.spatial.transform import Rotation, Slerp
    from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
    from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin, stable_recording_origin_kwargs_for_task,
    )
    from handson.data.recording_fields import (
        resolve_environment_recording_schema, declared_recording_field_names,
        select_next_observation_for_recording,
    )

    if (OUTPUT / "report.json").exists():
        raise FileExistsError(f"Use a fresh GEN1_SUPERVISOR_OUTPUT; existing report: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("isaacsim")
    assert version.startswith("5.1.")
    report = {
        "scope": "Post-fix native readback/interface/reset and physical public-action trial; no formal episode or operator acceptance",
        "isaacsim": version,
        "runtime_source_sha256": hashlib.sha256(Path(gen1.__file__).read_bytes()).hexdigest(),
        "readback_passed": False, "manipulation_attempted": False,
        "manipulation_succeeded": False, "stages": [], "cameras": {},
    }
    env = None
    try:
        env = TaskConfig().get_environment(
            policy_mode=True, render_mode="rgb_array", randomize=False,
            sim_backend="isaacsim", physics_backend="physx", launch_isaac=True,
            headless=True, image_obs=True, seed=7, isaac_device="cuda:0",
            arm_type="rokae", hand_type="wuji1",
        )
        raw = env.unwrapped
        obs, info = env.reset(seed=7)
        obs, info = stabilize_recording_origin(
            env, obs, info, **stable_recording_origin_kwargs_for_task(gen1.TASK_NAME),
        )
        report["physical_recording_origin"] = info["physical_recording_origin"]
        report["sampled_reset_pose_wxyz"] = raw._gen1_sampled_reset_pose.copy()
        neutral = np.r_[np.zeros(7), raw._right_hand].astype(np.float32)
        for _ in range(3):
            obs, _, terminated, truncated, _ = env.step(neutral)
            assert not terminated and not truncated
        metrics = gen1.current_metrics(raw)
        assert metrics["gen1_contacts_available"] and metrics["gen1_on_table"]
        assert metrics["gen1_hand_released"] and not metrics["gen1_success_now"]
        report["valid_native_metrics"] = metrics
        report["native_matrix_shapes"] = {
            role: list(view.get_contact_force_matrix(raw._sim_dt).shape)
            for role, view in raw._gen1_contact_views.items()
        }
        schema = resolve_environment_recording_schema(env)
        next_obs = select_next_observation_for_recording(
            obs, required_fields=declared_recording_field_names(schema),
        )
        report["interface"] = {
            "observation_space_contains": env.observation_space.contains(obs),
            "state_shape": list(obs["state"].shape),
            "task_state_shape": list(next_obs["gen1_task_state"].shape),
            "reset_pose_shape": list(next_obs["gen1_reset_pose_wxyz"].shape),
        }
        assert report["interface"]["observation_space_contains"]
        assert report["interface"]["state_shape"] == [50]

        def snapshot(stage):
            for _ in range(3):
                frames = raw.render()
            for name, frame in zip(("main", "wrist_left", "wrist_right"), frames, strict=True):
                array = np.asarray(frame, dtype=np.uint8)
                assert array.std() > 5 and array.ndim == 3 and array.shape[-1] == 3
                path = OUTPUT / f"{stage}_{name}.png"
                Image.fromarray(array).save(path)
                report["cameras"][f"{stage}_{name}"] = {
                    "file": path.name, "shape": list(array.shape), "std": float(array.std()),
                }

        snapshot("reset")
        report["readback_passed"] = True
        if os.environ.get("GEN1_SUPERVISOR_MANIPULATION", "0") != "1":
            return

        report["manipulation_attempted"] = True
        scale = float(os.environ.get("GEN1_SUPERVISOR_GRIP_SCALE", "0.5"))
        report["grip_scale"] = scale
        robot = raw._control_robot("right")
        palm = raw._robot_body_pose_matrix(robot, raw._ee_body_indices["right"])
        rest_rotation = Rotation.from_matrix(palm[:3, :3])
        rest_position = palm[:3, 3].copy()
        initial_bottle = raw._pose_for_role("target").copy()
        center = gen1.rotation_matrix(initial_bottle[3:]) @ np.array([0., 0., -.035]) + initial_bottle[:3]
        offset = np.asarray([
            float(value) for value in os.environ.get("GEN1_SUPERVISOR_GRASP_OFFSET", ".045,0,.11").split(",")
        ])
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("GEN1_SUPERVISOR_GRASP_OFFSET must contain three finite metres")
        report["grasp_local_offset_m"] = offset.copy()
        grasp_position = center - rest_rotation.apply(offset)
        above_position = grasp_position + [0., 0., .065]
        open_hand = raw._right_hand.copy()
        closed_hand = scale * raw._tool_power_grip_pose(open_hand)
        hand = open_hand.copy()
        last_goal = np.r_[rest_position, rest_rotation.as_quat(scalar_first=True)]
        stopped = False

        def step(goal, stage):
            nonlocal stopped
            _, _, terminated, truncated, info = env.step(np.r_[goal, hand].astype(np.float32))
            if terminated or truncated:
                stopped = True
                report["terminal"] = {"stage": stage, "terminated": bool(terminated), "truncated": bool(truncated), "succeed": bool(info.get("succeed"))}
                report["manipulation_succeeded"] = bool(info.get("succeed"))

        def move(position, rotation, frames, stage, end_hand=None):
            nonlocal last_goal, hand
            start = last_goal.copy()
            hand_start = hand.copy()
            rotations = Slerp([0., 1.], Rotation.concatenate([
                Rotation.from_quat(start[3:], scalar_first=True), rotation,
            ]))
            for alpha in np.linspace(0., 1., frames):
                if stopped:
                    break
                goal = np.r_[(1-alpha)*start[:3]+alpha*np.asarray(position), rotations(alpha).as_quat(scalar_first=True)]
                if end_hand is not None:
                    hand = (1-alpha)*hand_start + alpha*end_hand
                step(goal, stage)
                last_goal = goal
            physical_palm = raw._robot_body_pose_matrix(robot, raw._ee_body_indices["right"])
            report["stages"].append({
                "stage": stage, "command_pose_wxyz": last_goal.copy(),
                "physical_palm_matrix": physical_palm,
                "palm_position_error_m": float(np.linalg.norm(physical_palm[:3, 3] - last_goal[:3])),
                "bottle_pose": raw._pose_for_role("target").copy(),
                "metrics": gen1.current_metrics(raw),
            })
            print("GEN1_SUPERVISOR_STAGE", stage, serial(report["stages"][-1]), flush=True)

        move(above_position, rest_rotation, 90, "approach")
        move(grasp_position, rest_rotation, 90, "descend")
        move(grasp_position, rest_rotation, 75, "close", closed_hand)
        move(grasp_position, rest_rotation, 45, "grasp_hold")
        snapshot("grasp")
        move(grasp_position + [0., 0., .16], rest_rotation, 120, "lift")
        move(last_goal[:3], rest_rotation, 45, "lift_hold")
        lifted_bottle = raw._pose_for_role("target").copy()
        lift = float(lifted_bottle[2] - initial_bottle[2])
        report["physical_lift_m"] = lift
        snapshot("lift")
        if stopped or lift < .04:
            report["manipulation_limit"] = "Public-action grasp did not retain a physical lift of 4 cm; no object-state correction applied"
            return

        palm = raw._robot_body_pose_matrix(robot, raw._ee_body_indices["right"])
        palm_rotation = Rotation.from_matrix(palm[:3, :3])
        bottle_rotation = Rotation.from_quat(lifted_bottle[3:], scalar_first=True)
        desired_rotation = bottle_rotation.inv() * palm_rotation
        local_offset = palm_rotation.inv().apply(lifted_bottle[:3] - palm[:3, 3])
        desired_palm = lifted_bottle[:3] - desired_rotation.apply(local_offset)
        move(desired_palm, desired_rotation, 150, "reorient")
        move(desired_palm, desired_rotation, 45, "reorient_hold")
        snapshot("reoriented")
        upright_height = raw._current_table_top_z() - gen1.geometry().bottom_vertices[:, 2].min() + .003
        placed_center = lifted_bottle[:3].copy()
        placed_center[2] = upright_height
        placed_palm = placed_center - desired_rotation.apply(local_offset)
        move(placed_palm, desired_rotation, 120, "place")
        move(placed_palm, desired_rotation, 45, "place_hold")
        move(placed_palm, desired_rotation, 90, "open", open_hand)
        move(placed_palm + [0., 0., .10], desired_rotation, 90, "retreat")
        move(last_goal[:3], desired_rotation, 75, "released_hold")
        snapshot("final")
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (OUTPUT / "report.json").write_text(json.dumps(serial(report), indent=2) + "\n")
        print("GEN1_SUPERVISOR_REPORT", OUTPUT / "report.json", flush=True)
        if env is not None:
            env.unwrapped._fast_shutdown_on_close = True
            env.close()


if __name__ == "__main__":
    main()
