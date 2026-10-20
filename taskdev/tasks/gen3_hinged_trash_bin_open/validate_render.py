"""Fresh-process GEN3 three-camera appearance check at a paused physical state.

Run source and real_v12 separately with the safe run_render.sh allocation guard.
This script performs no task-body pose writes and never claims manipulation or
recording acceptance. Cross-process equality is measured, not inferred from seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def numeric(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise RuntimeError("Nonfinite direct PhysX tensor")
    return value.tolist()


def direct_physics(env) -> dict:
    """Read native tensors afresh rather than cached robot.data or USD poses."""
    state = {}
    for name, view in sorted(env._task_body_views.items()):
        state[f"task/{name}/transform_xyzw"] = numeric(view.get_transforms())
        state[f"task/{name}/velocity"] = numeric(view.get_velocities())
    for side, robot in sorted(env._robots.items()):
        view = robot.root_physx_view
        for key, getter in (
            ("root_transform_xyzw", view.get_root_transforms),
            ("root_velocity", view.get_root_velocities),
            ("link_transform_xyzw", view.get_link_transforms),
            ("link_velocity", view.get_link_velocities),
            ("joint_position", view.get_dof_positions),
            ("joint_velocity", view.get_dof_velocities),
        ):
            state[f"robot/{side}/{key}"] = numeric(getter())
    view = env._lw_articulation_view
    if view is None or tuple(env._lw_scene_spec.dof_names) != ("lid_hinge",):
        raise RuntimeError("Direct physical state did not include GEN3 articulation")
    for key, getter in (
        ("root_transform_xyzw", view.get_root_transforms),
        ("root_velocity", view.get_root_velocities),
        ("link_transform_xyzw", view.get_link_transforms),
        ("link_velocity", view.get_link_velocities),
        ("joint_position", view.get_dof_positions),
        ("joint_velocity", view.get_dof_velocities),
    ):
        state[f"task/articulation/{key}"] = numeric(getter())
    return state


def compare_state(before, after) -> dict:
    if before.keys() != after.keys():
        return {"identical": False, "reason": "physical entity/field set differs"}
    deltas = {}
    for key in before:
        left, right = np.asarray(before[key]), np.asarray(after[key])
        if left.shape != right.shape:
            return {"identical": False, "reason": f"tensor shape differs: {key}"}
        deltas[key] = float(np.max(np.abs(left - right))) if left.size else 0.0
    maximum = max(deltas.values(), default=0.0)
    return {"identical": maximum == 0.0, "max_absolute_delta": maximum, "field_deltas": deltas}


def camera_properties(stage) -> dict:
    from pxr import Usd, UsdGeom
    return {
        str(prim.GetPath()): {
            attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
            if attr.HasAuthoredValueOpinion()
        }
        for prim in Usd.PrimRange(stage.GetPseudoRoot()) if prim.IsA(UsdGeom.Camera)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("source", "real_v12"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", type=Path, help="The source run's render_audit.json")
    args = parser.parse_args()
    from validation_runtime import validate_runtime, render_preview_frames
    identity = validate_runtime()
    assert os.environ.get("TASKDEV_PHYSICAL_GPU") == os.environ["GEN3_GPU_INDEX"]
    assert os.environ.get("TASKDEV_LOGICAL_GPU") == "0"
    assert os.environ["HANDSON_ISAACSIM_RENDERER_GPU"] == "0"
    assert os.environ["HANDSON_ISAACSIM_TASK_VISUAL_PROFILE"] == args.profile
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
    sys.argv.extend([
        "--portable-root", str((args.out.parent / "runtime").resolve()),
        "--/log/file=" + str((args.out.parent / "kit.log").resolve()),
        "--/app/settings/persistent=false",
    ])
    result = {
        "task_name": "eval_gen02_open_trash_bin", "profile": args.profile,
        "seed": args.seed, "runtime": identity,
        "host": os.environ["GEN3_HOST"], "gpu_uuid": os.environ["GEN3_GPU_UUID"],
        "physics_steps_during_render": 0, "errors": [], "passed": False,
    }
    env = None
    try:
        from handson.sim.isaaclab.lw_native_articulation_env import HandsOnLWNativeArticulationIsaacLabEnv
        env = HandsOnLWNativeArticulationIsaacLabEnv(
            "eval_gen02_open_trash_bin", seed=args.seed, randomize=True,
            render_mode="rgb_array", image_obs=True, launch_isaac=True, headless=True,
            arm_type="rokae", hand_type="wuji1", physics_dt=1 / 120,
            hz=30, isaac_device="cuda:0",
        )
        env._fast_shutdown_on_close = True
        # The constructor completes its ordinary seeded reset. From here the
        # timeline stays paused; rendering is forbidden to advance physics.
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.commit()
        if timeline.is_playing():
            raise RuntimeError("Timeline did not pause")

        def forbidden_step(*_args, **_kwargs):
            result["physics_steps_during_render"] += 1
            raise RuntimeError("Appearance render attempted to step simulation")

        env._sim.step = forbidden_step
        env._step_isaaclab = forbidden_step
        result["task_visual_appearance_report"] = getattr(env, "task_visual_appearance_report", None)
        if args.profile == "real_v12" and result["task_visual_appearance_report"] is None:
            raise RuntimeError("Shared appearance integration report is absent")
        from handson.sim.isaaclab.blender_visual_layer import physics_signature
        result["physics_attributes_before"] = physics_signature(env._stage)
        result["physical_before"] = direct_physics(env)
        result["sim_time_before"] = float(env._sim.current_time)
        result["timeline_time_before"] = float(timeline.get_current_time())
        frames = render_preview_frames(env)
        result["physical_after"] = direct_physics(env)
        result["physics_attributes_after"] = physics_signature(env._stage)
        assert result["physics_attributes_before"] == result["physics_attributes_after"]
        result["sim_time_after"] = float(env._sim.current_time)
        result["timeline_time_after"] = float(timeline.get_current_time())
        result["physical_render_comparison"] = compare_state(result["physical_before"], result["physical_after"])
        result["cameras"] = camera_properties(env._stage)
        assert not timeline.is_playing()
        assert result["sim_time_before"] == result["sim_time_after"]
        assert result["timeline_time_before"] == result["timeline_time_after"]
        assert result["physical_render_comparison"]["identical"]
        from PIL import Image
        result["images"] = {}
        for name, frame in zip(("main", "wrist_left", "wrist_right"), frames):
            pixels = np.asarray(frame)
            wrist_width, wrist_height = env._wrist_recording_camera_output_size
            expected = (env._render_height, env._render_width, 3) if name == "main" else (wrist_height, wrist_width, 3)
            assert pixels.shape == expected, (name, pixels.shape)
            assert pixels.dtype == np.uint8 and float(pixels.std()) > 1.0
            path = args.out.parent / f"{name}.png"
            Image.fromarray(pixels).save(path)
            result["images"][name] = {
                "shape": list(pixels.shape), "std": float(pixels.std()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fallback_rejected": True,
                "near_black_fraction": float(np.mean(np.all(pixels < 5, axis=2))),
                "near_white_fraction": float(np.mean(np.all(pixels > 250, axis=2))),
            }
        if args.reference:
            source = json.loads(args.reference.read_text())
            assert source["profile"] == "source" and source["passed"]
            assert source["seed"] == args.seed
            result["reference"] = str(args.reference)
            result["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
            result["cross_process_physics"] = compare_state(source["physical_before"], result["physical_before"])
            result["cross_process_cameras_identical"] = source["cameras"] == result["cameras"]
            result["cross_process_physics_attributes_identical"] = source["physics_attributes_before"] == result["physics_attributes_before"]
            assert result["cross_process_physics_attributes_identical"]
            assert result["cross_process_physics"]["identical"], "Cold-start physical states differ; no exact-state A/B claim"
            assert result["cross_process_cameras_identical"], "Cold-start cameras differ"
        result["passed"] = True
    except Exception as exc:
        result["errors"].append(str(exc))
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()
    print(json.dumps({"profile": args.profile, "passed": result["passed"], "errors": result["errors"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
