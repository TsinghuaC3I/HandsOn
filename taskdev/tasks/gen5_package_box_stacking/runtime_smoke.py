"""First live GEN5 load/reset/hold probe; does not certify the full live gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "handson")]


def jsonable(value):
    import numpy as np
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def capture_shape_overlap(env, phase):
    if getattr(env, "_snapshot_audit", False):
        from collision_snapshots import capture
        return capture(env, phase)
    import carb
    import omni.physx
    from pxr import Usd, UsdPhysics, UsdGeom, PhysicsSchemaTools, Sdf
    stage = env._stage
    # Fabric leaves USD at its authored pose. Publish the current PhysX
    # transforms to the readback stage before USD-based shape queries. This
    # copies simulation output only; it does not advance or restore physics.
    omni.physx.get_physx_interface().update_transformations(False, True)
    inventory = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False: continue
        rigid = prim
        while rigid and not rigid.IsPseudoRoot() and not rigid.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = rigid.GetParent()
        inventory.append({"path": str(prim.GetPath()),
            "type": str(prim.GetTypeName()),
            "rigid_body": str(rigid.GetPath()) if rigid and not rigid.IsPseudoRoot() else None,
            "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility())})
    query = omni.physx.get_physx_scene_query_interface()
    control_hits = []
    def control_report(hit):
        control_hits.append(str(hit.collision))
        return True
    query.overlap_sphere(3.0, carb.Float3(0, 0, 1), control_report, False)
    body_controls = {}
    for name in getattr(env, "_task_body_views", {}):
        pose = env._physics_body_pose(name)
        if pose is None or not name.startswith("gen5_"):
            continue
        hits = []
        def body_report(hit):
            hits.append(str(hit.rigid_body))
            return True
        query.overlap_sphere(0.002, carb.Float3(*map(float, pose[:3])), body_report, False)
        body_controls[name] = {"physical_center": pose[:3].tolist(), "hits": hits,
                               "own_body_found": any(name in path for path in hits)}
    pairs, errors, self_hits = [], [], {}
    for source in inventory:
        # Infinite planes are supported targets but unsupported input shapes.
        # Every finite shape is still queried against the floor/supports.
        if source["type"] == "Plane":
            continue
        hits = []
        def report(hit):
            hits.append({"collision": str(hit.collision), "rigid_body": str(hit.rigid_body)})
            return True
        try:
            path = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(source["path"]))
            count = query.overlap_shape(path[0], path[1], report, False)
            self_hits[source["path"]] = any(hit["collision"] == source["path"] for hit in hits)
            for hit in hits:
                if hit["collision"] != source["path"]:
                    pairs.append({"source": source["path"], "source_body": source["rigid_body"],
                                  "hit": hit, "reported_count": int(count)})
        except Exception as exc:
            errors.append({"source": source["path"], "error": repr(exc)})
    return {"phase": phase, "inventory": inventory, "pairs": pairs, "errors": errors,
            "physics_interface": {"running": omni.physx.get_physx_interface().is_running(),
                "readback_suppressed": omni.physx.get_physx_interface().is_readback_suppressed(),
                "suppress_readback_setting": carb.settings.get_settings().get("/physics/suppressReadback"),
                "update_to_usd_setting": carb.settings.get_settings().get("/physics/updateToUsd")},
            "positive_control_hits": control_hits,
            "self_hits": self_hits,
            "physical_center_controls": body_controls,
            "query_enabled": bool(env._sim.cfg.enable_scene_query_support),
            "scope": "Raw live shape overlap inventory; pair classification, contact depth and all multipose gates remain separate."}


def main():
    from isaaclab.app import AppLauncher
    from handson.sim.isaaclab.runtime_display import isaac_kit_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--audit-suite", choices=("smoke", "multipose", "query", "snapshot", "edge"), default="smoke")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    args.kit_args = isaac_kit_args(args.kit_args, headless=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"task_id": "GEN5", "scope": "First live load/reset/hold smoke only", "passed": False}
    launcher = None
    env = None
    try:
        launcher = AppLauncher(args)
        import numpy as np
        import torch
        from handson.sim.isaaclab import eval_gen07_stack_boxes as gen5
        from handson.sim.isaaclab import abc_collection_env as abc
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        # Headless IsaacLab disables scene queries by default. Enable the
        # read-only query manager for this diagnostic, without changing task
        # dynamics or the ordinary collection configuration.
        original_cfg = abc.make_simulation_cfg
        def audit_cfg(*values, **kwargs):
            cfg = original_cfg(*values, **kwargs)
            if args.audit_suite not in ("snapshot", "edge"):
                cfg.enable_scene_query_support = True
                cfg.use_fabric = False
            return cfg
        abc.make_simulation_cfg = audit_cfg

        class AuditedEnv(HandsOnABCIsaacLabEnv):
            _snapshot_audit = args.audit_suite in ("snapshot", "edge")
            def _reset_scene_state(self):
                super()._reset_scene_state()
                query = capture_shape_overlap(self, "sampled_reset_before_physics")
                query["poses"] = {n: self._physics_body_pose(n).tolist() for n in gen5.BODY_NAMES}
                result.setdefault("sampled_reset_queries", []).append(query)
                self._audit_reset_rows = result["sampled_reset_queries"]

            def _configure_task_simulation_settings(self):
                super()._configure_task_simulation_settings()
                if self._snapshot_audit:
                    self._stage.Flatten().Export(str(args.output.parent / "composed_scene.usdc"))
                    result["authored_first_state"] = capture_shape_overlap(self, "authored_before_first_sim_reset_or_step")
                    return
                if args.audit_suite != "smoke":
                    return
                import omni.physx
                # Cook the final authored shapes without advancing time.
                # This hook precedes the constructor's first SimContext.reset.
                before_time = float(self._sim.current_time)
                omni.physx.get_physx_interface().force_load_physics_from_usd()
                result["before_first_reset_shapes"] = capture_shape_overlap(self, "cooked_before_first_sim_reset_or_step")
                result["before_first_reset_shapes"]["time_before"] = before_time
                result["before_first_reset_shapes"]["time_after"] = float(self._sim.current_time)
                print("[GEN5] Captured authored collision shapes before the first reset/step.", flush=True)

            def _init_task_specific_runtime(self):
                super()._init_task_specific_runtime()
                result["earliest_cooked_shapes"] = capture_shape_overlap(self, "after_initial_sim_reset_before_task_pose_reset_and_settle")

        env = AuditedEnv(gen5.TASK_NAME, seed=args.seed, control_dt=1/30,
            physics_dt=1/120, render_mode="rgb_array", image_obs=False,
            randomize=True, randomize_dynamics=False, hand_type="wuji1", arm_type="rokae",
            launch_isaac=True, headless=True, isaac_device=args.device, physics_backend="physx",
            existing_simulation_app=launcher.app, existing_app_launcher=launcher)
        result["device"] = {"torch_device_count": torch.cuda.device_count(), "torch_device_name": torch.cuda.get_device_name(0)}
        result["post_constructor_shapes"] = capture_shape_overlap(env, "after_constructor_reset_and_settle")
        observation, info = env.reset(seed=args.seed)
        result["schema"] = env.get_recording_schema()
        result["observation_shapes"] = {k: {sk: list(np.shape(sv)) for sk, sv in v.items()} if isinstance(v, dict)
                                         else list(np.shape(v)) for k, v in observation.items()}
        rows = []
        for step in range(args.steps):
            # None preserves the established hand and pose targets. This probe
            # does not fabricate a teleoperation trajectory or successful demo.
            observation, reward, terminated, truncated, info = env.step(None)
            rows.append({"step": step, "poses": {n: env._pose_for_role(n) for n in gen5.BODY_NAMES},
                         "velocities": {n: env._physics_body_velocity(n) for n in gen5.BODY_NAMES},
                         "terminated": bool(terminated), "truncated": bool(truncated),
                         "succeed": bool(info.get("succeed", False))})
        result["steps"] = jsonable(rows)
        result["post_hold_shapes"] = capture_shape_overlap(env, "after_control_hold")
        result["conditions"] = env._success_condition_rows()
        result["final_info"] = jsonable(info)
        result["body_paths"] = {n: str(env._find_rigid_body_prim(n).GetPath()) for n in gen5.BODY_NAMES}
        result["checks"] = {
            "one_visible_cuda_gpu": torch.cuda.device_count() == 1,
            "four_independent_bodies": len(set(result["body_paths"].values())) == 4,
            "physics_120_hz": abs(env._physics_dt - 1/120) < 1e-10,
            "control_30_hz": abs(env._control_dt - 1/30) < 1e-10,
            "deferred_cameras": env._deferred_camera_render,
            "finite_readback": all(np.isfinite(np.asarray(row[key][name])).all() for row in rows for key in ("poses", "velocities") for name in gen5.BODY_NAMES),
            "no_success_at_reset": not any(row["succeed"] for row in rows),
            "no_terminal_during_hold": not any(row["terminated"] or row["truncated"] for row in rows),
            "english_conditions": all(row["label"].isascii() for row in result["conditions"]),
            "hidden_state_schema": "gen5_hidden_state" in (result["schema"] or {}).get("fields", {}),
            "collision_capture_available": all(bool(result[key]["inventory"]) for key in ("earliest_cooked_shapes", "post_constructor_shapes", "post_hold_shapes")),
        }
        result["passed"] = all(result["checks"].values())
        if "before_first_reset_shapes" in result:
            result["pre_first_step_time_unchanged"] = result["before_first_reset_shapes"]["time_before"] == result["before_first_reset_shapes"]["time_after"] == 0.0
        if args.audit_suite in ("multipose", "query", "snapshot", "edge"):
            from live_layout_audit import run
            run(env, capture_shape_overlap, args.output.parent / "layouts.json",
                seed_count=0 if args.audit_suite in ("query", "edge") else 20,
                layout_indices=(7,) if args.audit_suite == "query" else (9,) if args.audit_suite == "edge" else range(10),
                layout_steps=300 if args.audit_suite == "edge" else 90)
            result["multipose_completed"] = True
    except Exception:
        result["exception"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(jsonable(result), indent=2) + "\n")
        print("[GEN5] Probe result saved; closing the diagnostic child.", flush=True)
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()
        if launcher is not None:
            launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
