"""Fresh-process GEN1 three-camera appearance check at a paused physical state.

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
    if not any(k.startswith("task/gen1_soap_bottle/") for k in state):
        raise RuntimeError("Direct physical state did not include GEN1 lid")
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


def camera_tensors(env) -> dict:
    return {name: {field: numeric(getattr(camera.data, field))
                   for field in ("intrinsic_matrices", "pos_w", "quat_w_world")}
            for name, camera in sorted(env._cameras.items())}


def final_task_materials(stage, profile):
    from pxr import Usd, UsdGeom, UsdShade
    from handson.tasks.eval_gen03_orient_bottle_upright import visual_appearance as source
    rows = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        part = source._source_part(str(prim.GetPath()))
        if part is None or not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shader = material.ComputeSurfaceSource()[0]
        if not shader:
            shader = material.ComputeSurfaceSource("mdl")[0]
        row = {"mesh": str(prim.GetPath()), "source_node": part,
               "material": str(material.GetPath()), "shader": shader.GetIdAttr().Get() if shader else None,
               "inputs": {i.GetBaseName(): repr(i.Get()) for i in shader.GetInputs()} if shader else {}}
        if profile == "real_v12":
            expected = source.SOURCE_MATERIALS[source.PART_MATERIALS[part]]
            assert row["shader"] == "UsdPreviewSurface"
            for name in ("metallic", "roughness"):
                assert abs(shader.GetInput(name).Get() - expected[name]) < 1e-6
        rows.append(row)
    assert len(rows) == 10
    return rows


def existing_contact_trajectory(raw):
    """Repeat only the previously used half-grip approach, stopping at contact."""
    from scipy.spatial.transform import Rotation
    from native_smoke import serial
    from handson.tasks.policy_wrappers import SingleArmPolicyWrapper
    from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin, stable_recording_origin_kwargs_for_task,
    )
    policy = SingleArmPolicyWrapper(raw)
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    obs, info = policy.reset(seed=7)
    obs, info = stabilize_recording_origin(
        policy, obs, info, **stable_recording_origin_kwargs_for_task(gen1.TASK_NAME))
    neutral = np.r_[np.zeros(7), raw._right_hand].astype(np.float32)
    for _ in range(3):
        _, _, terminated, truncated, _ = policy.step(neutral)
        assert not terminated and not truncated
    palm = raw._robot_body_pose_matrix(raw._control_robot("right"), raw._ee_body_indices["right"])
    rotation = Rotation.from_matrix(palm[:3, :3])
    bottle = raw._pose_for_role("target").copy()
    center = gen1.rotation_matrix(bottle[3:]) @ np.array([0., 0., -.035]) + bottle[:3]
    grasp = center - rotation.apply([.045, 0., .11])
    hand = raw._right_hand.copy()
    closed = .5 * raw._tool_power_grip_pose(hand)
    position = palm[:3, 3].copy()
    stages = []
    for name, target, frames, end_hand in (
        ("approach", grasp + [0., 0., .065], 90, hand.copy()),
        ("descend", grasp, 90, hand.copy()),
        ("close", grasp, 75, closed),
        ("grasp_hold", grasp, 45, closed),
    ):
        start_position, start_hand = position.copy(), hand.copy()
        for alpha in np.linspace(0., 1., frames):
            position = (1 - alpha) * start_position + alpha * target
            hand = (1 - alpha) * start_hand + alpha * end_hand
            action = np.r_[position, rotation.as_quat(scalar_first=True), hand].astype(np.float32)
            _, _, terminated, truncated, _ = policy.step(action)
            assert not terminated and not truncated, f"Unexpected terminal during {name}"
        metrics = {k: (v.item() if isinstance(v, np.generic) else v)
                   for k, v in gen1.current_metrics(raw).items()}
        stages.append({"stage": name, "frames": frames, "metrics": metrics,
                       "bottle_pose": numeric(raw._pose_for_role("target"))})
    return serial({"source": "native_supervisor_validation.py documented half-grip trial",
            "public_policy_action_steps": 303, "grip_scale": .5,
            "local_palm_offset_m": [.045, 0., .11],
            "task_success_claimed": False,
            "origin": info["physical_recording_origin"], "stages": stages})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("source", "real_v12"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference", type=Path, help="The source run's render_audit.json")
    parser.add_argument("--phase", choices=("reset", "contact"), default="reset")
    args = parser.parse_args()
    from validation_runtime import validate_runtime, render_preview_frames
    identity = validate_runtime()
    assert os.environ.get("TASKDEV_PHYSICAL_GPU") == os.environ["GEN1_GPU_INDEX"]
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
        "task_name": "eval_gen03_orient_bottle_upright", "profile": args.profile,
        "phase": args.phase,
        "seed": args.seed, "runtime": identity,
        "host": os.environ["GEN1_HOST"], "gpu_uuid": os.environ["GEN1_GPU_UUID"],
        "physics_steps_during_render": 0, "errors": [], "passed": False,
        "complete": False,
    }
    # Constructor cleanup can exit the interpreter directly. Leave a negative
    # pending report in that case; only the completed checks can turn it green.
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    env = None
    try:
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        env = HandsOnABCIsaacLabEnv(
            "eval_gen03_orient_bottle_upright", seed=args.seed, randomize=False,
            render_mode="rgb_array", image_obs=True, launch_isaac=True, headless=True,
            arm_type="rokae", hand_type="wuji1", physics_dt=1 / 120,
            hz=30, isaac_device="cuda:0",
        )
        env._fast_shutdown_on_close = True
        if args.phase == "contact":
            result["contact_trajectory"] = existing_contact_trajectory(env)
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
        from handson.sim.isaaclab.task_visual_appearance import selected_task_visual_profile
        from handson.sim.isaaclab.blender_visual_layer import physics_signature
        assert selected_task_visual_profile(env) == args.profile
        result["task_visual_appearance_report"] = getattr(env, "task_visual_appearance_report", None)
        if args.profile == "real_v12":
            appearance = result["task_visual_appearance_report"]
            if appearance is None:
                raise RuntimeError("Shared appearance integration report is absent")
            assert appearance["robot"]["links"] == 60
            assert len(appearance["objects"]["matched_prims"]) == 10
            assert appearance["objects"]["material_count"] == 3
            assert appearance["objects"]["status"] == "applied"
            assert not appearance["objects"]["binding_errors"]
            assert not appearance["objects"]["missing_parts"]
            assert len(appearance["lighting"]["lights"]) >= 2
            assert appearance["physics_before"] == appearance["physics_after"]
        else:
            assert result["task_visual_appearance_report"] is None
        result["physical_scene_signature"] = physics_signature(env._stage)
        result["physical_before"] = direct_physics(env)
        result["sim_time_before"] = float(env._sim.current_time)
        result["timeline_time_before"] = float(timeline.get_current_time())
        frames = render_preview_frames(env)
        result["physical_after"] = direct_physics(env)
        result["sim_time_after"] = float(env._sim.current_time)
        result["timeline_time_after"] = float(timeline.get_current_time())
        result["physical_render_comparison"] = compare_state(result["physical_before"], result["physical_after"])
        result["cameras"] = camera_properties(env._stage)
        result["camera_tensors"] = camera_tensors(env)
        result["composed_task_materials"] = final_task_materials(env._stage, args.profile)
        assert not timeline.is_playing()
        assert result["sim_time_before"] == result["sim_time_after"]
        assert result["timeline_time_before"] == result["timeline_time_after"]
        assert result["physical_render_comparison"]["identical"]
        from PIL import Image
        result["images"] = {}
        for name, frame in zip(("main", "wrist_left", "wrist_right"), frames):
            pixels = np.asarray(frame)
            expected = (640, 640, 3) if name == "main" else (480, 848, 3)
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
            assert source.get("phase", "reset") == args.phase
            result["reference"] = str(args.reference)
            result["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
            result["cross_process_physics"] = compare_state(source["physical_before"], result["physical_before"])
            result["cross_process_cameras_identical"] = source["cameras"] == result["cameras"]
            result["cross_process_camera_tensors_identical"] = source["camera_tensors"] == result["camera_tensors"]
            result["cross_process_physical_scene_identical"] = source["physical_scene_signature"] == result["physical_scene_signature"]
            assert result["cross_process_physics"]["identical"], "Cold-start physical states differ; no exact-state A/B claim"
            assert result["cross_process_cameras_identical"], "Cold-start cameras differ"
            assert result["cross_process_camera_tensors_identical"], "Cold-start camera K/extrinsic tensors differ"
            assert result["cross_process_physical_scene_identical"], "Cold-start physical scene properties differ"
        result["passed"] = True
    except Exception as exc:
        result["errors"].append(str(exc))
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        result["complete"] = True
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()
    print(json.dumps({"profile": args.profile, "passed": result["passed"], "errors": result["errors"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
