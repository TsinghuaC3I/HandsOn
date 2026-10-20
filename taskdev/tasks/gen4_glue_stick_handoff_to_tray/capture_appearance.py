#!/usr/bin/env python3
"""Cold-process GEN4 three-camera A/B; never restore physical state or calibration."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value).copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite direct native state")
    return result.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visual-profile", choices=("source", "real_v12"), required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    assert metadata.version("isaacsim") == "5.1.0.0"
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    sys.argv = [sys.argv[0], "--portable-root", str(root / ".gen4-local/kit" / args.output.name),
                "--/log/file=" + str((args.output / "kit.log").resolve())]
    os.environ.update(HANDSON_ISAACSIM_TASK_VISUAL_PROFILE=args.visual_profile,
                      HANDSON_ENABLE_ISAACSIM_NEWTON="0", HANDSON_ISAACLAB_DEFER_CAMERA_RENDER="0",
                      HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY="0")
    from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
    from handson.tasks.eval_gen04_pass_glue_stick import contract as c
    result = {"completed": False, "passed": False, "profile": args.visual_profile, "seed": args.seed,
              "isaacsim": metadata.version("isaacsim"), "scope": "diagnostic preview; no new actions or physics-state writes",
              "exceptions": []}
    env = None
    try:
        env = HandsOnGEN4IsaacLabEnv(seed=args.seed, randomize=True, image_obs=True, headless=True,
                                   render_mode="rgb_array", isaac_device="cuda:0", physics_backend="physx",
                                   arm_type="rokae", hand_type="wuji1", render_resolution=(576, 1024))
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdPhysics
        from handson.sim.isaaclab.blender_visual_layer import physics_signature
        def reject_fallback(*args, **kwargs):
            raise RuntimeError("GEN4 appearance diagnostic rejects fallback camera frames")
        env._fallback_camera_frame = reject_fallback
        assert set(env._cameras) == {"head_camera", "wrist_left_camera", "wrist_right_camera"}
        names = (c.GLUE_BODY, c.TRAY_BODY, *c.DISTRACTORS)
        task_paths = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(env._stage.GetPrimAtPath("/World/Task"))
                      if p.GetName() in names and p.HasAPI(UsdPhysics.RigidBodyAPI)}
        assert set(task_paths) == set(names)
        first_robot = next(iter(env._robots.values()))
        rigid_view = first_robot._physics_sim_view.create_rigid_body_view([task_paths[n] for n in names])
        assert rigid_view.count == len(names)
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.commit()
        assert not timeline.is_playing()

        def direct_state():
            robots = {}
            for side, robot in env._robots.items():
                view = robot.root_physx_view
                robots[side] = {"joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
                                "joint_positions": array(view.get_dof_positions()),
                                "joint_velocities": array(view.get_dof_velocities()),
                                "link_transforms": array(view.get_link_transforms()),
                                "link_velocities": array(view.get_link_velocities())}
            return {"time": float(env._sim.current_time), "robots": robots,
                    "task_paths": list(rigid_view.prim_paths), "task_transforms": array(rigid_view.get_transforms()),
                    "task_velocities": array(rigid_view.get_velocities())}

        def cameras():
            return {name: {"K": array(camera.data.intrinsic_matrices),
                           "position_world": array(camera.data.pos_w),
                           "quaternion_world": array(camera.data.quat_w_world),
                           "calibration": {"width": camera.cfg.width, "height": camera.cfg.height,
                                           "offset_position": list(camera.cfg.offset.pos),
                                           "offset_rotation": list(camera.cfg.offset.rot),
                                           "offset_convention": camera.cfg.offset.convention}}
                    for name, camera in env._cameras.items()}

        for _ in range(4):
            env.render()
        before = direct_state()
        camera_before = cameras()
        physics_before = physics_signature(env._stage)
        for _ in range(4):
            frames = env.render()
        after = direct_state()
        camera_after = cameras()
        result.update(direct_before=before, direct_after=after, cameras=camera_after,
                      physics_signature=physics_before,
                      appearance=getattr(env, "task_visual_appearance_report", None))
        assert before == after, "Rendering advanced or changed direct PhysX state"
        assert camera_before == camera_after, "Rendering changed camera K or extrinsics"
        assert physics_before == physics_signature(env._stage)
        if args.visual_profile == "real_v12":
            assert result["appearance"]["profile"] == "real_v12"
            assert result["appearance"]["robot"]["links"] == 60
            assert result["appearance"]["objects"]["changed_attributes"] == 0
        else:
            assert result["appearance"] is None
            assert not env._stage.GetPrimAtPath("/World/TaskVisualAppearance")
            assert not any(p.GetName() == "BlenderVisuals" for p in env._stage.Traverse())
        result["frames"] = {}
        for name, frame in zip(("main", "wrist_left", "wrist_right"), frames):
            rgb = np.asarray(frame, dtype=np.uint8)
            assert rgb.ndim == 3 and rgb.shape[-1] == 3 and float(rgb.std()) > 3
            Image.fromarray(rgb).save(args.output / (name + ".png"))
            result["frames"][name] = {"shape": list(rgb.shape), "mean": float(rgb.mean()), "std": float(rgb.std())}
        assert len(result["frames"]) == 3
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            assert baseline["completed"] and baseline["profile"] == "source" and baseline["seed"] == args.seed
            assert baseline["physics_signature"] == result["physics_signature"], "A/B physics contract differs"
            assert baseline["direct_before"] == before, "Cold A/B initial direct PhysX states differ"
            for name, camera in camera_after.items():
                assert camera["K"] == baseline["cameras"][name]["K"]
                assert camera["calibration"] == baseline["cameras"][name]["calibration"]
                for field in ("position_world", "quaternion_world"):
                    np.testing.assert_allclose(camera[field], baseline["cameras"][name][field], rtol=0, atol=1e-6)
            result["baseline_comparison"] = {"path": str(args.baseline), "physics_contract_equal": True,
                                             "direct_PhysX_state_equal": True,
                                             "K_and_calibration_equal": True, "world_camera_tolerance": 1e-6}
        result.update(render_preserves_direct_state=True, render_preserves_cameras=True,
                      warnings=list(env._isaac_warnings), completed=True)
        assert not any("RGB unavailable" in w for w in result["warnings"])
        result["passed"] = True
    except BaseException:
        result["exceptions"].append(traceback.format_exc())
        raise
    finally:
        (args.output / "appearance.json").write_text(json.dumps(result, indent=2) + "\n")
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()


if __name__ == "__main__":
    main()
