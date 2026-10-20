#!/usr/bin/env python3
"""XY1 real Isaac 5.1 diagnostics; reports evidence, never fabricates task completion."""
from __future__ import annotations
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "handson"), str(ROOT)]
from handson.sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv
from handson.tasks.eval_comb03_pass_block_and_close_box import contract as c


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def collision_snapshot(stage):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    query = omni.physx.get_physx_scene_query_interface()
    rows = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        body_path = str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else None
        hits = []
        def callback(hit):
            hits.append({"collision": str(hit.collision), "rigid_body": str(hit.rigid_body)})
            return True
        encoded = PhysicsSchemaTools.encodeSdfPath(prim.GetPath())
        queried = str(prim.GetTypeName()) in {"Cube", "Sphere", "Capsule", "Cylinder", "Mesh"}
        count = query.overlap_shape(encoded[0], encoded[1], callback, False) if queried else 0
        mesh_mode = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_mode = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        rows.append({
            "path": str(prim.GetPath()), "body": body_path,
            "type": str(prim.GetTypeName()), "approximation": mesh_mode,
            "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
            "hit_count": int(count), "hits": hits, "queried": queried,
        })
    return {"scope": "PhysX scene queries over final collider prims; mesh inputs use convex query approximation; pairs require geometric/contact review",
            "dynamic_query_limit": "GPU overlap queries retained authored poses in diagnostics; use cooked shapes at measured PhysX poses for moving-body audits",
            "colliders": rows, "collider_count": len(rows),
            "total_hits": sum(r["hit_count"] for r in rows)}


def collision_geometry(stage):
    """Export the final enabled geometry and exact authored joint filters."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdUtils, PhysicsSchemaTools
    import omni.physx
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    cache = UsdGeom.XformCache()
    shapes, joints, articulations = [], [], []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append({"path": str(prim.GetPath()), "body0": [str(x) for x in joint.GetBody0Rel().GetTargets()],
                           "body1": [str(x) for x in joint.GetBody1Rel().GetTargets()],
                           "collision_enabled": joint.GetCollisionEnabledAttr().Get()})
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            articulations.append({"path": str(prim.GetPath()), "self_collisions": PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()})
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        body = str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else None
        world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        body_world = np.asarray(cache.GetLocalToWorldTransform(owner), dtype=float).T if body else np.eye(4)
        attributes = {}
        for name in ("size", "radius", "height", "axis", "physics:approximation", "physxCollision:contactOffset", "physxCollision:restOffset", "physxConvexHullCollision:hullVertexLimit"):
            attr = prim.GetAttribute(name)
            if attr and attr.Get() is not None:
                value = attr.Get()
                attributes[name] = np.asarray(value).tolist() if not isinstance(value, (str, float, int)) else value
        if prim.IsA(UsdGeom.Mesh):
            from scipy.spatial import ConvexHull
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            attributes["source_vertex_count"] = len(points)
            attributes["source_vertices_sha256"] = hashlib.sha256(points.tobytes()).hexdigest()
            if attributes.get("physics:approximation") != "convexHull":
                raise RuntimeError(f"Audit needs an exact geometry exporter for {prim.GetPath()}")
            # Keep every extremal point; interior/tessellation vertices do not
            # alter a convex hull. This is geometry, not a screenshot proxy.
            attributes["points"] = points[ConvexHull(points).vertices].tolist()
            cooked = {}
            def on_cooked(result, convexes):
                cooked["valid"] = result == PhysxCollisionRepresentationResult.RESULT_VALID
                cooked["result"] = str(result)
                cooked["hulls"] = [{"vertices": [[float(x), float(y), float(z)] for x, y, z in convex.vertices],
                                    "indices": list(convex.indices),
                                    "polygons": [{"index_base": polygon.index_base, "num_vertices": polygon.num_vertices,
                                                  "plane": [float(v) for v in polygon.plane]} for polygon in convex.polygons]}
                                   for convex in convexes]
            omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                run_asynchronously=False, on_result=on_cooked)
            if not cooked.get("valid") or not cooked.get("hulls"):
                raise RuntimeError(f"No final cooked geometry for {prim.GetPath()}: {cooked}")
            attributes["cooked"] = cooked
        shapes.append({"path": str(prim.GetPath()), "body": body, "type": str(prim.GetTypeName()),
                       "world": world.tolist(), "body_world": body_world.tolist(),
                       "local": (np.linalg.inv(body_world) @ world).tolist(), "attributes": attributes})
    return {"shapes": shapes, "joints": joints, "articulations": articulations,
            "geometry_source": "final composed USD, before first PhysX step; each mesh uses its recorded PhysX approximation"}


class AuditEnvironment(HandsOnDIsaacLabEnv):
    def __init__(self, output, task_name=c.TASK_NAME, **kwargs):
        assert task_name == c.TASK_NAME
        self.audit_output = Path(output)
        self.audit_pre_steps = 0
        self.audit_first = None
        self.audit_first_error = None
        self.audit_subscription = None
        self.audit_resets = 0
        super().__init__(c.TASK_NAME, **kwargs)

    def _in_process_scene_rebuild_kwargs(self, seed):
        kwargs = super()._in_process_scene_rebuild_kwargs(seed)
        generation = int(self._physx_scene_generation) + 1
        kwargs["output"] = self.audit_output.parent / f"scene_{generation:03d}"
        kwargs["output"].mkdir(parents=True, exist_ok=True)
        return kwargs

    def _reset_scene_state(self):
        super()._reset_scene_state()
        if self._sim is not None:
            snapshot = physical_snapshot(self)
            snapshot["poses"] = {name: self._physics_body_pose(name) for name in snapshot["poses"]}
            self.write_json(f"authored_reset_{self.audit_resets}.json", snapshot)
            self.audit_resets += 1

    def _configure_task_simulation_settings(self):
        super()._configure_task_simulation_settings()
        import omni.physx
        self._sim.get_physics_context().set_enable_scene_query_support(True)
        self.audit_timeline_playing_before_reset = bool(self._sim.is_playing())
        self.audit_subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(
            self._audit_pre_step, True, 0)
        print("XY1_AUDIT pre-step callback installed", flush=True)

    def _audit_pre_step(self, dt):
        if self.audit_pre_steps == 0:
            try:
                self.audit_first = collision_snapshot(self._stage)
                self.audit_first["physics_steps_observed_before_capture"] = self.audit_pre_steps
                self.audit_first["timeline_playing_when_installed"] = self.audit_timeline_playing_before_reset
                self.write_json("first_pre_step.json", self.audit_first)
                self.write_json("collision_geometry.json", collision_geometry(self._stage))
                print(f"XY1_AUDIT first-pre-step colliders={self.audit_first['collider_count']} hits={self.audit_first['total_hits']}", flush=True)
            except Exception:
                self.audit_first_error = traceback.format_exc()
                self.write_json("first_pre_step_error.json", {"error": self.audit_first_error})
                print(self.audit_first_error, flush=True)
        self.audit_pre_steps += 1

    def write_json(self, name, value):
        (self.audit_output / name).write_text(json.dumps(json_safe(value), indent=2) + "\n")


def physical_snapshot(env):
    robots = {}
    for side, robot in env._robots.items():
        def array(x):
            return x.detach().cpu().numpy()
        robots[side] = {
            "joint_names": list(robot.joint_names),
            "joint_pos": array(robot.data.joint_pos[0]),
            "joint_vel": array(robot.data.joint_vel[0]),
            "body_names": list(robot.body_names),
            "body_pose": array(robot.data.body_pose_w[0]),
        }
    return {"poses": {name: np.asarray(pose).copy() for name, pose in env._poses.items()},
            "sampled_poses": {name: np.asarray(pose).copy() for name, pose in env._xy1_sampled_poses.items()},
            "velocities": {n: env._physics_body_velocity(n) for n in c.NOMINAL_POSITIONS},
            "task_info": env._task_info(), "robots": robots,
            "physics_steps": env.audit_pre_steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--in-place-resets", action="store_true", help="Diagnostic only; the platform rebuilds its PhysX scene on later resets")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("isaacsim")
    assert version.startswith("5.1."), version
    print(f"XY1_RUNTIME python={sys.executable} isaacsim={version} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    env = None
    try:
        seeds = list(map(int, args.seeds.split(",")))
        scene_output = args.output / "scene_000"
        scene_output.mkdir(parents=True, exist_ok=True)
        env = AuditEnvironment(scene_output, seed=seeds[0], launch_isaac=True, image_obs=False,
                               render_mode="rgb_array", headless=True, randomize=True,
                               hz=30, physics_dt=1/120, control_dt=1/30,
                               time_limit=300, isaac_device="cuda:0", physics_backend="physx")
        for ordinal, seed in enumerate(seeds):
            print(f"XY1_AUDIT reset seed={seed}", flush=True)
            if ordinal and not args.in_place_resets:
                env.audit_subscription = None
                report = env.hard_reset_physx_scene(seed=seed)
                env.write_json("scene_rebuild.json", report)
            obs, info = env.reset(seed=seed)
            env.write_json(f"reset_{ordinal}_seed_{seed}.json", physical_snapshot(env))
            env.write_json(f"reset_{ordinal}_overlaps.json", collision_snapshot(env._stage))
            for _ in range(args.hold_frames):
                obs, reward, terminated, truncated, info = env.step_with_current_joint_targets()
                if terminated or truncated:
                    break
            env.write_json(f"settled_{ordinal}_seed_{seed}.json", physical_snapshot(env))
            env.write_json(f"settled_{ordinal}_overlaps.json", collision_snapshot(env._stage))
            print(f"XY1_AUDIT settled seed={seed} failed={info.get('failed',False)} steps={env.env_step}", flush=True)
        env._stage.GetRootLayer().Export(str(args.output / "final_stage.usdc"))
        summary = {"loaded": True, "isaacsim": version, "first_pre_step_captured": env.audit_first is not None,
                   "first_pre_step_error": env.audit_first_error, "physics_steps": env.audit_pre_steps,
                   "recording_schema": env.get_recording_schema(), "trajectory_validated": False}
        summary["scene_reset_mode"] = "in_place_diagnostic" if args.in_place_resets else "platform_scene_rebuild"
        summary["seeds"] = seeds
        (args.output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2)+"\n")
        return int(env.audit_first is None)
    finally:
        if env is not None:
            env.audit_subscription = None
            env._fast_shutdown_on_close = True
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
