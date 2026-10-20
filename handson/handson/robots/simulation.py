"""Standalone Isaac Sim gravity/drive diagnostic for estimated robot profiles.

Run in the documented Isaac Sim environment. This is not benchmark evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=("kuka_kr6_r700", "abb_irb120", "fanuc_lrmate200id"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args._worker:
        if output.exists():
            parser.error("--output must be a new directory")
        child = subprocess.run([sys.executable, "-m", __spec__.name, *sys.argv[1:], "--_worker"])
        # Kit shutdown can exit with zero while unwinding a Python exception.
        result = output / "result.json"
        passed = result.is_file() and json.loads(result.read_text()).get("passed") is True
        if child.returncode or not passed or (output / "ERROR.json").exists():
            code = child.returncode or 1
            raise SystemExit(code if code > 0 else 128 - code)
        return
    output.mkdir(parents=True, exist_ok=False)
    from .assets import get_robot_asset
    from .dynamics import load_simulation_profile
    asset = get_robot_asset(args.robot)
    profile = load_simulation_profile(asset)
    urdf = asset.prepare_urdf(output / "robot.urdf", simulation=True)
    # Importers may write mesh conversion caches; confine them to this run.
    tree = ET.parse(urdf)
    for mesh in tree.getroot().iter("mesh"):
        source = Path(mesh.get("filename"))
        private = output / "meshes" / ("mesh_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16] + source.suffix)
        private.parent.mkdir(exist_ok=True)
        if not private.exists():
            shutil.copy2(source, private)
        mesh.set("filename", str(private))
    tree.write(urdf, encoding="utf-8", xml_declaration=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 960,
                         "renderer": "RayTracedLighting", "active_gpu": 0, "physics_gpu": 0,
                         "multi_gpu": False})
    try:
        import numpy as np
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from PIL import Image

        enable_extension("isaacsim.asset.importer.urdf")
        world = World(stage_units_in_meters=1.0, physics_dt=profile["physics_dt"], rendering_dt=1/30)
        stage = omni.usd.get_context().get_stage()
        ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not ok:
            raise RuntimeError("URDF importer configuration failed")
        config.set_fix_base(True)
        config.set_merge_fixed_joints(True)
        config.set_import_inertia_tensor(True)
        config.set_self_collision(False)
        config.set_default_drive_strength(2000)
        config.set_default_position_drive_damping(100)
        ok, root_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf),
                                                 import_config=config, get_articulation_root=True)
        if not ok:
            raise RuntimeError("URDF import failed")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in profile["joint_drives"]:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                spec = profile["joint_drives"][prim.GetName()]
                drive.CreateTypeAttr(profile["drive_type"])
                drive.CreateMaxForceAttr(spec["effort_limit_nm"])
                # USD angular gains use degrees; PhysX tensor/controller APIs use radians.
                drive.CreateStiffnessAttr(spec["stiffness_nm_rad"] * np.pi / 180)
                drive.CreateDampingAttr(spec["damping_nm_s_rad"] * np.pi / 180)
        robot = world.scene.add(SingleArticulation(prim_path=root_path, name=args.robot))
        world.scene.add_default_ground_plane(z_position=-0.02)
        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(700)
        light = UsdLux.DistantLight.Define(stage, "/World/Sun")
        light.CreateIntensityAttr(1800)
        UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 30))
        camera = rep.create.camera(position=(1.8, -1.8, 1.4), look_at=(0, 0, .45), focal_length=35)
        product = rep.create.render_product(camera, (1280, 960))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        world.reset()
        names = list(robot.dof_names)
        if set(names) != set(profile["joint_drives"]):
            raise ValueError(f"Unexpected imported joints: {names}")
        drives = [profile["joint_drives"][n] for n in names]
        home = np.array([d["home_rad"] for d in drives])
        source_limits = {j.get("name"): j.find("limit") for j in tree.getroot().findall("joint")}
        limits = np.array([[float(source_limits[n].get("lower")), float(source_limits[n].get("upper"))] for n in names])
        controller = robot.get_articulation_controller()
        actual_kp, actual_kd = controller.get_gains()
        if not (np.allclose(actual_kp, [d["stiffness_nm_rad"] for d in drives], rtol=1e-4)
                and np.allclose(actual_kd, [d["damping_nm_s_rad"] for d in drives], rtol=1e-4)):
            raise ValueError("Imported joint gains differ from the simulation profile")
        # Initialization only. All subsequent motion is driven by joint targets.
        robot.set_joint_positions(home)
        robot.set_joint_velocities(np.zeros(len(names)))
        traces = []
        dt = profile["physics_dt"]
        snapshots = {0: "initial", int(3/dt)-1: "held", int(6/dt)-1: "motion", int(9/dt)-1: "final"}
        start_time = world.current_time
        for step in range(int(9/dt)):
            t = step * dt
            target = home.copy()
            if t >= 3:
                target += np.array([d["test_amplitude_rad"] for d in drives]) * np.sin(2*np.pi*(t-3)/4)
            target = np.clip(target, limits[:, 0] + .02, limits[:, 1] - .02)
            if step % round(profile["control_dt"] / dt) == 0:
                controller.apply_action(ArticulationAction(joint_positions=target))
            before_time = world.current_time
            world.step(render=False)
            if not np.isclose(world.current_time - before_time, dt, atol=1e-8):
                raise ValueError("Physical action clock differs from configured timestep")
            if step % 8 == 0:
                world.render()
            q = robot.get_joint_positions()
            v = robot.get_joint_velocities()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise ValueError("Nonfinite articulation state")
            traces.append({"time_s": world.current_time-start_time, "target": target.tolist(), "q": q.tolist(), "dq": v.tolist()})
            if step in snapshots:
                capture_time = world.current_time
                for _ in range(8):
                    world.render()
                if world.current_time != capture_time:
                    raise ValueError("Render-only capture advanced physics")
                pixels = np.asarray(rgb.get_data())
                if pixels.size == 0:
                    raise RuntimeError("Empty rendered image")
                Image.fromarray(pixels[:, :, :3].astype(np.uint8)).save(output / (snapshots[step]+".png"))
        q = np.array([r["q"] for r in traces])
        goals = np.array([r["target"] for r in traces])
        hold_error = float(np.max(np.abs(q[int(2/dt):int(3/dt)] - goals[int(2/dt):int(3/dt)])))
        motion_error = float(np.max(np.abs(q[int(4/dt):] - goals[int(4/dt):])))
        motion_range = np.ptp(q[int(3/dt):], axis=0)
        limit_violation = float(max(0, np.max(limits[:, 0]-q), np.max(q-limits[:, 1])))
        stage.Export(str(output / "scene.usda"))
        (output / "trajectory.json").write_text(json.dumps(traces)+"\n")
        report = {"robot": args.robot, "scope": "Standalone fixed-base gravity hold and bounded joint motion; no task or payload qualification",
                  "profile_status": profile["status"], "drive_type": profile["drive_type"],
                  "actual_stiffness_nm_rad": np.asarray(actual_kp).tolist(), "actual_damping_nm_s_rad": np.asarray(actual_kd).tolist(), "profile_source_sha256": profile["source_sha256"],
                  "physics_steps": len(traces), "physics_dt": dt, "control_dt": profile["control_dt"], "physical_duration_s": world.current_time-start_time, "clock_verified": True,
                  "dof_names": names, "gravity_hold_max_error_rad": hold_error,
                  "motion_max_error_rad": motion_error, "joint_motion_range_rad": motion_range.tolist(),
                  "joint_limit_violation_rad": limit_violation, "self_collision_enabled": False,
                  "passed": bool(hold_error < .08 and motion_error < .10 and np.all(motion_range > .15) and limit_violation < .01)}
        (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        rgb.detach()
        if not report["passed"]:
            raise RuntimeError("Standalone robot diagnostic failed; see result.json")
    except Exception as exc:
        (output / "ERROR.json").write_text(json.dumps({"error": str(exc)}, indent=2)+"\n")
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
