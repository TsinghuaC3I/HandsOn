#!/usr/bin/env python3
"""XY3 live reset/geometry and small control-response diagnostics.

No scripted task trajectory or synthetic successful episode is created.
Composed-shape helpers reuse the independently inspected XY2 diagnostic.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "handson")]
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.tasks.eval_comb04_place_fruit_and_photograph_logo import semantics as xy3
from handson.tasks.eval_comb04_place_fruit_and_photograph_logo import runtime as xy3_runtime

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

def snapshot(env, obs):
    values = {"objects": {}, "robots": {}}
    for name in xy3.DYNAMIC_NAMES:
        values["objects"][name] = {"pose": env._physics_body_pose(name), "velocity": env._physics_body_velocity(name)}
    for side, robot in env._robots.items():
        q = robot.data.joint_pos[0].detach().cpu().numpy()
        qd = robot.data.joint_vel[0].detach().cpu().numpy()
        target = robot.data.joint_pos_target[0].detach().cpu().numpy()
        values["robots"][side] = {"names": robot.joint_names, "q": q, "qd": qd,
            "drive_target": target, "max_target_error": float(np.max(np.abs(q-target))),
            "tcp_actual": robot.data.body_pose_w[0, robot.find_bodies(f"{side}_palm_link")[0][0]].detach().cpu().numpy(), "tcp_command": getattr(env, f"_{side}_tcp_pose")}
    if "physx_contact_header_count" in obs:
        from pxr import PhysicsSchemaTools
        n = int(np.asarray(obs["physx_contact_header_count"]).flat[0])
        contacts = []
        for ids, meta in zip(obs["physx_contact_header_ids"][:n], obs["physx_contact_header_meta"][:n]):
            start, count = int(meta[1]), int(meta[2])
            contacts.append({"paths": [str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in ids[:4]],
                             "points": obs["physx_contact_point_data"][start:start+count]})
        values["contacts"] = contacts
    values["hidden"] = obs.get("xy3_hidden_state")
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--targets", default="apple")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--tracking", action="store_true")
    p.add_argument("--matrix", action="store_true", help="4 fruits x 6 canonical/seeded/boundary layouts with repeated resets")
    p.add_argument("--render-preview", action="store_true")
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    from collection_ui.server import ISAAC_COLLECTION_REALTIME_ENV
    os.environ.update(ISAAC_COLLECTION_REALTIME_ENV)
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"]="1"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"]="0" if args.render_preview else "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"]="0"
    version=importlib.metadata.version("isaacsim");assert version.startswith("5.1.")
    report={"isaacsim":version,"python":sys.executable,"source":str(ROOT),
        "scope":"actual geometry/reset and bounded control response; operator task completion pending", "rows": []}
    class AuditedXY3(HandsOnABCIsaacLabEnv):
        validation_layout = None
        def _apply_task_pose_randomization(self):
            if self.validation_layout is None:
                return super()._apply_task_pose_randomization()
            original_sampler=xy3.sample_layout
            xy3.sample_layout=lambda *a,**kw: self.validation_layout
            try:
                return super()._apply_task_pose_randomization()
            finally:
                xy3.sample_layout=original_sampler

        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            self._sim.get_physics_context().set_enable_scene_query_support(True)
            inventory=shapes(self._stage)
            self._stage.GetRootLayer().Export(str(output/"authored-stage.usda"))
            from omni.physx import get_physx_interface
            get_physx_interface().force_load_physics_from_usd()
            write(output/"before-first-physics.json", {"inventory":inventory,"overlap":overlaps(self._stage,inventory)})
    env=None;code=1
    try:
        print("XY3_CONSTRUCT",flush=True)
        env=AuditedXY3(xy3.TASK_NAME, hz=30, time_limit=240., launch_isaac=True,
            image_obs=args.render_preview, headless=True, render_mode="rgb_array",
            randomize=True, seed=int(args.seeds.split(',')[0]), target_object=args.targets.split(',')[0],
            isaac_device="cuda:0", hand_type="wuji1", arm_type="rokae")
        report["timing"]={"physics_dt":env._physics_dt,"control_dt":env._control_dt,"substeps":env._n_substeps}
        report["online_cameras"]=list(env._cameras)
        assert set(env._robots)=={"right","left"}
        assert np.isclose(env._physics_dt,1/120) and np.isclose(env._control_dt,1/30) and env._n_substeps==4
        layouts=[(f"seed_{seed}",int(seed),None) for seed in args.seeds.split(',')]
        if args.matrix:
            layouts=[]
            for label in ("canonical","seed_101","seed_202","bucket_x_min","camera_x_max","fruit_y_min"):
                positions={n:p.copy() for n,p in xy3.DEFAULT_XY.items()}
                yaw={n:0. for n in xy3.DYNAMIC_NAMES};seed=0
                if label.startswith("seed_"):
                    seed=int(label.split('_')[1])
                    positions,yaw=xy3.sample_layout(np.random.RandomState(seed),randomize=True,
                        is_clear=lambda p,q:xy3_runtime.layout_is_clear(env,p,q))
                elif label=="bucket_x_min":positions["bucket"][0]=xy3.XY_BOUNDS["bucket"][0,0]
                elif label=="camera_x_max":
                    positions["camera"][0]=xy3.XY_BOUNDS["camera"][1,0];yaw["camera"]=5.
                elif label=="fruit_y_min":
                    for n in xy3.FRUIT_NAMES:positions[n][1]=xy3.XY_BOUNDS[n][0,1]
                assert xy3_runtime.layout_is_clear(env,positions,yaw),label
                layouts.append((label,seed,(positions,yaw)))
        for fruit in (xy3.FRUIT_NAMES if args.matrix else args.targets.split(',')):
            for layout_name,seed,layout in layouts:
                env.validation_layout=layout
                obs,info=env.reset(seed=seed,target_label=fruit)
                first=snapshot(env,obs)
                if args.matrix:
                    first_xy={n:p.copy() for n,p in env._xy3_reset_xy.items()}
                    obs,info=env.reset(seed=seed,target_label=fruit)
                    assert all(np.array_equal(first_xy[n],env._xy3_reset_xy[n]) for n in xy3.DYNAMIC_NAMES)
                row={"fruit":fruit,"seed":seed,"layout":layout_name,"repeated_reset":bool(args.matrix),
                     "first_reset":first,"reset_info":info,"reset":snapshot(env,obs),
                     "sampled_xy":env._xy3_reset_xy,"sampled_yaw":env._xy3_reset_yaw,"samples":[]}
                inventory=shapes(env._stage)
                row["reset_shapes"]=inventory;row["reset_overlap"]=overlaps(env._stage,inventory)
                origin=np.r_[env._right_tcp_pose,env._left_tcp_pose,env._right_hand,env._left_hand]
                for step in range(args.steps):
                    command=origin.copy()
                    if args.tracking:
                        scale=min(1.,step/12.) if step<36 else max(0.,1.-(step-36)/12.)
                        command[2]+=.025*scale;command[9]+=.025*scale;command[14:]+=.12*scale
                    begin=time.monotonic();obs,reward,terminated,truncated,info=env.step(command)
                    row["samples"].append({"step":step,"wall_seconds":time.monotonic()-begin,
                        "terminated":terminated,"truncated":truncated,"info":info,"physics":snapshot(env,obs)})
                    if terminated or truncated:raise RuntimeError("Empty/small response diagnostic ended the episode")
                row["settled_overlap"]=overlaps(env._stage,shapes(env._stage))
                if args.render_preview:
                    from PIL import Image
                    for _ in range(3):frames=env.render()
                    row["preview_frames"]=[]
                    for name,frame in zip(("main","wrist_left","wrist_right"),frames):
                        frame=np.asarray(frame);assert frame.ndim==3 and frame.std()>5,name
                        file=f"{fruit}-{layout_name}-{name}.png";Image.fromarray(frame.astype(np.uint8)).save(output/file)
                        row["preview_frames"].append({"path":file,"shape":list(frame.shape),"std":float(frame.std())})
                report["rows"].append(row);write(output/"report.json",report)
                print("XY3_SAMPLE "+json.dumps({"fruit":fruit,"seed":seed,"steps":args.steps}),flush=True)
        write(output/"environment-schema.json",env.get_recording_schema())
        report["isaac_warnings"]=env._isaac_warnings
        report["completed_diagnostic"]=True;code=0
    except BaseException:
        report["error"]=traceback.format_exc();traceback.print_exc()
    finally:
        write(output/"report.json",report)
        if env is not None:
            env._fast_shutdown_on_close=True
            env.close()
    return code

if __name__=="__main__":raise SystemExit(main())
