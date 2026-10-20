"""Real XY2 startup, composed-shape queries, and bounded tracking diagnostics.

This runner never fabricates task success or replaces operator acceptance.
It must be launched in the authorized single-GPU runtime namespace.
"""

import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from handson.sim.isaaclab.xy2_collection_env import HandsOnXY2IsaacLabEnv


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def shapes(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        # MJCF's authored primitive extents already include local scaling;
        # BBoxCache can apply it twice. Compute from shape geometry instead.
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
        bound = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1])),
                         cache.GetLocalToWorldTransform(prim)).ComputeAlignedRange()
        result.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                       "owner": str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else None,
                       "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
                       "bounds": [list(bound.GetMin()), list(bound.GetMax())]})
    return result


def overlaps(stage, inventory):
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf
    query = get_physx_scene_query_interface()
    pairs = set()
    errors = []
    controls = []
    for row in inventory:
        if row["type"] == "Plane":
            continue
        found = []
        center = np.mean(row["bounds"], axis=0).tolist()
        query.overlap_sphere(.01, center, lambda hit: found.append(str(hit.collision)) is None, False)
        if row["path"] in found:
            controls.append({"source": row["path"], "hits": found})
            break
    for row in inventory:
        source = row["path"]
        if row["type"] == "Plane":
            continue
        try:
            encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(source))
            def hit(value):
                target = str(value.collision)
                if target != source:
                    pairs.add(tuple(sorted((source, target))))
                return True
            query.overlap_shape(encoded[0], encoded[1], hit, False)
        except Exception as exc:
            errors.append({"path": source, "error": str(exc)})
    return {"pairs": sorted(pairs), "errors": errors, "positive_controls": controls,
            "positive_control_passed": bool(controls),
            "excluded_query_sources": [r["path"] for r in inventory if r["type"] == "Plane"],
            "query": "PhysX overlap_shape; depth requires contact readback"}


def physical_snapshot(env, obs):
    keys = ("physx_task_rigid_body_state", "physx_task_rigid_body_valid",
            "physx_task_articulation_root_state", "physx_task_articulation_root_valid",
            "physx_task_articulation_dof_position", "physx_task_articulation_dof_valid")
    result = {k: obs[k] for k in keys if k in obs}
    result["poses"] = {name: env._physics_body_pose(name) for name in ("mouse", "mousepad", "table")}
    result["predicate_poses"] = {role: env._pose_for_role(role).copy() for role in ("target", "receptacle")}
    if "physx_contact_header_count" in obs:
        from pxr import PhysicsSchemaTools
        n = int(np.asarray(obs["physx_contact_header_count"]).flat[0])
        contacts = []
        for ids, meta in zip(obs["physx_contact_header_ids"][:n], obs["physx_contact_header_meta"][:n]):
            start, count = int(meta[1]), int(meta[2])
            points = obs["physx_contact_point_data"][start:start+count]
            contacts.append({"paths": [str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in ids[:4]],
                             "points": points})
        result["contacts"] = contacts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--targets", default="60")
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--open-pose-audit", action="store_true",
                        help="Also reset directly to target hinge poses for static geometry/retention; not a success trajectory")
    parser.add_argument("--render-preview", action="store_true",
                        help="Separate diagnostic process with live canonical cameras, never formal online capture")
    parser.add_argument("--platform-servo", action="store_true",
                        help="Use the ordinary collection UI's existing realtime control environment")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.render_preview:
        os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    if args.platform_servo:
        from collection_ui.server import ISAAC_COLLECTION_REALTIME_ENV
        os.environ.update(ISAAC_COLLECTION_REALTIME_ENV)
    version = importlib.metadata.version("isaacsim")
    assert version.startswith("5.1."), version
    report = {"isaacsim": version, "scope": "startup/reset/geometry/tracking only",
              "full_task_success_verified": False, "platform_servo": args.platform_servo, "rows": []}
    write(output / "report.json", report)

    class AuditedXY2(HandsOnXY2IsaacLabEnv):
        _diagnostic_open_reset = False

        def _initial_laptop_reset(self, seed):
            sample = super()._initial_laptop_reset(seed)
            if self._diagnostic_open_reset:
                sample = replace(sample, hinge_angle_deg=-sample.target_open_deg)
                self._xy2_layout = replace(self._xy2_layout, laptop=sample)
            return sample

        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            self._sim.get_physics_context().set_enable_scene_query_support(True)
            inventory = shapes(self._stage)
            self._stage.GetRootLayer().Export(str(output / "authored-stage.usda"))
            before = {"inventory": inventory, "before_first_sim_reset": True}
            try:
                from omni.physx import get_physx_interface
                get_physx_interface().force_load_physics_from_usd()
                before["overlap"] = overlaps(self._stage, inventory)
            except Exception as exc:
                before["query_error"] = str(exc)
            write(output / "before-first-physics.json", before)

    env = None
    code = 1
    try:
        env = AuditedXY2(
            launch_isaac=True, image_obs=args.render_preview, headless=True, render_mode="rgb_array",
            seed=int(args.seeds.split(",")[0]), target_object=args.targets.split(",")[0],
            randomize=True, isaac_device="cuda:0", hand_type="wuji1", arm_type="rokae",
        )
        assert env._fu015_contact_contract_ready, "Native laptop/contact runtime not ready"
        assert set(env._robots) == {"left", "right"}, "Native robot pair missing"
        assert env._fu015_articulation_view is not None, "Laptop articulation missing"
        report["initialized"] = True
        report["timing"] = {"physics_dt": env._physics_dt, "control_dt": env._control_dt}
        report["cameras_online"] = list(env._cameras)
        for mode, angle in [(mode, int(angle)) for mode in ([False, True] if args.open_pose_audit else [False])
                            for angle in args.targets.split(",")]:
            env._diagnostic_open_reset = mode
            for seed in map(int, args.seeds.split(",")):
                obs, info = env.reset(seed=seed, target_label=str(angle))
                row = {"seed": seed, "target": angle, "static_open_pose": mode, "layout": env._xy2_layout.to_dict(),
                       "reset_info": info, "reset_physics": physical_snapshot(env, obs), "samples": []}
                inventory = shapes(env._stage)
                row["reset_shapes"] = inventory
                row["reset_overlap"] = overlaps(env._stage, inventory)
                command = np.concatenate([env._right_tcp_pose, env._left_tcp_pose,
                                          env._right_hand, env._left_hand])
                origin = command.copy()
                for step in range(args.steps):
                    if args.tracking:
                        scale = min(1., step / 12.) if step < 30 else max(0., 1. - (step - 30) / 12.)
                        command[2] = origin[2] + .03 * scale
                        command[9] = origin[9] + .03 * scale
                        command[14:] = origin[14:] + .15 * scale
                    begin = time.monotonic()
                    obs, reward, terminated, truncated, info = env.step(command)
                    row["samples"].append({
                        "step": step, "wall_seconds": time.monotonic() - begin,
                        "terminated": terminated, "truncated": truncated,
                        "right_tcp": env._fu015_tcp_pose_diagnostic("right"),
                        "left_tcp": env._fu015_tcp_pose_diagnostic("left"),
                        "right_hand": env._fu015_hand_joint_diagnostic("right"),
                        "left_hand": env._fu015_hand_joint_diagnostic("left"),
                        "laptop": env._fu015_last_metrics,
                        "physics": physical_snapshot(env, obs),
                    })
                    if terminated or truncated:
                        row["terminal_info"] = info
                        break
                row["settled_overlap"] = overlaps(env._stage, shapes(env._stage))
                row["final_info"] = info
                if args.render_preview:
                    from PIL import Image
                    assert not env._deferred_camera_render
                    for _ in range(3):
                        frames = env.render()
                    row["preview_frames"] = []
                    for key, frame in zip(("main", "wrist_left", "wrist_right"), frames):
                        frame = np.asarray(frame)
                        assert frame.ndim == 3 and float(frame.std()) > 5, key
                        name = f"seed{seed}-angle{angle}-open{int(mode)}-{key}.png"
                        Image.fromarray(frame.astype(np.uint8)).save(output / name)
                        row["preview_frames"].append({"path": name, "shape": list(frame.shape), "std": float(frame.std())})
                report["rows"].append(row)
                write(output / "report.json", report)
                print("XY2_SAMPLE " + json.dumps({"seed": seed, "target": angle,
                      "steps": len(row["samples"]), "terminated": terminated,
                      "truncated": truncated}), flush=True)
        schema = env.get_recording_schema()
        write(output / "environment-schema.json", schema)
        report["isaac_warnings"] = env._isaac_warnings
        report["completed_diagnostic"] = True
        code = 0
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        write(output / "report.json", report)
        # FU015's existing disposable-worker close skips the known Kit teardown
        # hang. The parent guard observes this process and owns only its group.
        if env is not None:
            env.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
