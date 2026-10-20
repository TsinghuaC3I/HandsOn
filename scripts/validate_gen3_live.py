#!/usr/bin/env python3
"""GEN3 composed-scene and stable-origin audit in the collection runtime.

Run only after the host/GPU allocation gate. This script does not launch the
collection UI, certify recordings, or count scripted poses as teleoperation.
PhysX overlap_shape uses a convex approximation of an input mesh; positive
results remain review candidates until their actual geometry is checked.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handson"))
from validate_lw_native_articulation_live import _json_ready, _reset_and_accept_origin


def shape_inventory(stage):
    from pxr import Usd, UsdPhysics
    from handson.sim.isaaclab.lw_native_articulation_scene import LW_TABLE_PATH, LW_TASK_SPECS
    spec = LW_TASK_SPECS["eval_gen02_open_trash_bin"]
    rows = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        path = str(prim.GetPath())
        if "/HandsOnRobots/LeftRobot/" in path:
            group = "left_robot"
        elif "/HandsOnRobots/RightRobot/" in path:
            group = "right_robot"
        elif path.startswith(spec.root_body_path + "/"):
            group = "bin_base"
        elif path.startswith(spec.right_body_path + "/"):
            group = "lid"
        elif path.startswith(LW_TABLE_PATH + "/"):
            group = "table"
        elif path == "/World/Task/worldBody/floor/collisions/CollisionPlane":
            group = "ground_support"
        else:
            continue
        body = prim
        while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        rows[path] = {
            "group": group, "body": str(body.GetPath()) if body and not body.IsPseudoRoot() else path,
            "source_mesh": prim.GetCustomDataByKey("handson:sourceNode"),
            "mesh_approximation": str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"),
        }
    return rows


def audit_shapes(raw, phase, output):
    import omni.physx
    from pxr import PhysicsSchemaTools, Sdf, UsdPhysics
    # Keep SDK queries as supplementary evidence. Publishing USD transforms
    # did not prevent their observed false negatives after tensor resets;
    # the authoritative intersection audit below uses live body tensors.
    omni.physx.get_physx_interface().update_transformations(
        updateToFastCache=False, updateToUsd=True, updateVelocitiesToUsd=False)
    inventory = shape_inventory(raw._stage)
    support_review = json.loads((ROOT / "taskdev/tasks/eval_gen02_open_trash_bin/evidence/reference_robot_support_review.json").read_text())
    reference_support = {tuple(row["paths"]): row for row in support_review["pairs"]}
    fixed_mounts = {}
    for prim in raw._stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint) or "/HandsOnRobots/" not in str(prim.GetPath()):
            continue
        joint = UsdPhysics.Joint(prim)
        first, second = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if first and second:
            a, b = str(first[0]), str(second[0])
            fixed_mounts.setdefault(a, []).append((b, str(prim.GetPath())))
            fixed_mounts.setdefault(b, []).append((a, str(prim.GetPath())))

    def fixed_mount_chain(first, second):
        seen = {first}
        queue = [(first, [])]
        for body, chain in queue:
            for neighbor, joint in fixed_mounts.get(body, []):
                if neighbor == second:
                    return chain + [joint]
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, chain + [joint]))
        return []
    counts = {key: sum(row["group"] == key for row in inventory.values())
              for key in ("left_robot", "right_robot", "bin_base", "lid", "table")}
    if not all(counts.values()):
        raise RuntimeError(f"Incomplete composed collision inventory: {counts}")
    query = omni.physx.get_physx_scene_query_interface()
    pairs = {}
    total_hits = 0
    def classify(pair):
        first, second = (inventory[path] for path in pair)
        same_body = first["body"] == second["body"]
        hinge = ({first["group"], second["group"]} == {"bin_base", "lid"}
                 and "SourceMesh06" in {first["source_mesh"], second["source_mesh"]})
        support = reference_support.get(pair)
        fixed_mount = fixed_mount_chain(first["body"], second["body"])
        if support:
            side = "Right" if first["group"] == "right_robot" else "Left"
            root_path = f"/World/HandsOnRobots/{side}Robot/base_{side.lower()}"
            enabled = raw._stage.GetPrimAtPath(root_path).GetAttribute("physxArticulation:enabledSelfCollisions").Get()
            if enabled is not False:
                raise RuntimeError(f"Reference robot collision settings changed: {root_path}")
        return {"paths": pair, "same_compound_rigid_body": same_body,
            "existing_source_hinge_assembly": hinge,
            "existing_rigid_robot_mount_joint_chain": fixed_mount,
            "measured_reference_mechanical_support": support,
            "requires_geometry_review": not (same_body or hinge or support or fixed_mount)}

    for source in inventory:
        def hit_callback(hit):
            target = str(hit.collision)
            if target in inventory and target != source:
                pair = tuple(sorted((source, target)))
                pairs[pair] = classify(pair)
            return True
        if inventory[source]["group"] != "ground_support":
            # The plane is a target of every other shape query; it has no
            # bounded volume with which to make an overlap_shape input.
            encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(source))
            total_hits += int(query.overlap_shape(encoded[0], encoded[1], hit_callback, False))
    # Calibrate query readiness against the existing bin/table. A clear
    # scene may legitimately have zero overlap hits; that alone is not a
    # readiness test, especially in a headless IsaacLab scene.
    probe = query.raycast_closest((-.28, 0., 2.), (0., 0., -1.), 3.)
    query_ready = bool(probe.get("hit"))
    exact = raw._gen3_composed_audit.run(raw, before_physics=phase == "before_first_physics")
    exact_pairs = [{**classify(tuple(row["paths"])), **row} for row in exact["overlaps"]]
    exact["overlaps"] = exact_pairs
    unreviewed = [row for row in exact_pairs if row["requires_geometry_review"]]
    if exact["minimum_world_shape_z_m"] < -1e-7:
        unreviewed.append({"ground_penetration_m": -exact["minimum_world_shape_z_m"]})
    report = {"phase": phase, "inventory": inventory, "counts": counts,
        "query_api": "omni.physx.get_physx_scene_query_interface().overlap_shape",
        "mesh_query_semantics": "Input mesh is convex-approximated; positive pairs require exact source/cooked-shape review.",
        "reported_hits": total_hits, "pairs": list(pairs.values()),
        "query_readiness_ray_hit": dict(probe), "query_readiness_passed": query_ready,
        "exact_cooked_shape_audit": exact,
        "exact_overlaps_missing_from_sdk_query": [row["paths"] for row in exact_pairs if tuple(row["paths"]) not in pairs],
        "unreviewed_pairs": unreviewed}
    (output / f"{phase}.json").write_text(json.dumps(_json_ready(report), indent=2, default=str) + "\n")
    print(json.dumps({"GEN3_phase": phase, "shape_counts": counts,
                      "actual_overlaps": len(exact_pairs),
                      "unreviewed_pairs": len(report["unreviewed_pairs"])}), flush=True)
    if not query_ready:
        raise RuntimeError("The existing bin/table query calibration failed; an unavailable query cannot pass")
    if report["unreviewed_pairs"]:
        raise RuntimeError(f"Composed collision candidates require inspection: {phase}")
    return report


def export_cooked_shapes(stage, output, physics_steps):
    """Retain actual PhysX convex data, including any cooking adjustments."""
    import omni.physx
    from pxr import UsdGeom, UsdUtils, PhysicsSchemaTools
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    cooking = omni.physx.get_physx_cooking_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    shapes = {}
    for path, row in shape_inventory(stage).items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pieces = []
        status = []
        def ready(result, convexes):
            status.append(result)
            for data in convexes:
                pieces.append({"vertices": [[float(v.x), float(v.y), float(v.z)] for v in data.vertices],
                    "indices": list(data.indices),
                    "polygons": [{"index_base": int(p.index_base), "num_vertices": int(p.num_vertices)}
                                 for p in data.polygons]})
        # Isaac 5.1's own ConvexMeshDataDemo uses this synchronous public API.
        # It returns cooked collision data while the timeline is stopped.
        cooking.request_convex_collision_representation(stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            run_asynchronously=False, on_result=ready)
        if status != [PhysxCollisionRepresentationResult.RESULT_VALID] or not pieces:
            raise RuntimeError(f"Cooked convex representation unavailable: {path}: {status}")
        shapes[path] = {**row, "cooked_piece_count": len(pieces), "pieces": pieces,
            "world_transform_row_matrix": [list(row) for row in UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)]}
    report = {"physics_steps": physics_steps, "shapes": shapes}
    (output / "cooked_shapes_before_first_physics.json").write_text(json.dumps(report) + "\n")
    print(json.dumps({"GEN3_cooked_mesh_count": len(shapes),
                      "available_mesh_count": sum(bool(r["pieces"]) for r in shapes.values()),
                      "physics_steps": physics_steps}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "taskdev/tasks/eval_gen02_open_trash_bin/evidence/live")
    parser.add_argument("--probe-steps", type=int, default=0,
        help="Diagnostic-only idle control frames; never counts as origin or task acceptance")
    parser.add_argument("--contact-probe", action="store_true",
        help="Normal-control diagnostic contact trial with actual preview cameras; not a recording or human trial")
    parser.add_argument("--contact-batch", action="store_true",
        help="Repeat the normal-control contact diagnostic at five representative task poses")
    args = parser.parse_args()
    args.contact_probe = args.contact_probe or args.contact_batch
    output = args.out.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("GEN3 evidence must stay in its dedicated checkout")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0" if args.contact_probe else "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    device = os.environ.get("HANDSON_ISAACLAB_DEVICE", "")
    renderer = os.environ.get("HANDSON_ISAACSIM_RENDERER_GPU", "")
    if not device.startswith("cuda:") or not renderer.isdigit():
        raise ValueError("Set the audited CUDA device and physical Vulkan renderer before launching GEN3")
    from isaaclab.app import AppLauncher
    from handson.sim.isaaclab.compat import create_app_launcher
    from handson.sim.isaaclab.runtime_display import isaac_kit_args
    launcher = create_app_launcher(AppLauncher, {"headless": True, "enable_cameras": args.contact_probe,
        "device": device, "kit_args": isaac_kit_args(
            existing=f"--portable-root {output / 'kit'}", headless=True)})
    app = launcher.app
    env = None
    physics_subscription = None
    physics_steps = 0
    result = {"task_id": "GEN3", "random_seed_count_required": 20,
        "boundary_pose_count_required": 27, "stable_origins": [],
        "dynamic_five_pose_contact_gate": "pending",
        "three_real_teleoperation_successes": "pending",
        "normal_UI_start_stop_reset_recording": "pending", "three_camera_post_render": "pending"}
    try:
        import omni.physx
        def on_physics_step(_dt):
            nonlocal physics_steps
            physics_steps += 1
        physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)
        from handson.sim.isaaclab import lw_native_articulation_env as runtime
        canonical_class = runtime.HandsOnLWNativeArticulationIsaacLabEnv

        class AuditedGEN3(canonical_class):
            boundary_pose = None

            def _init_recording_cameras(self):
                # Diagnostic preview only: use the canonical camera models,
                # capture explicitly at milestones, and keep observations small.
                previous = self.image_obs
                try:
                    if args.contact_probe:
                        self.image_obs = True
                    return super()._init_recording_cameras()
                finally:
                    self.image_obs = previous

            def _configure_task_simulation_settings(self):
                super()._configure_task_simulation_settings()
                self._gen3_audit_index = 0
                from pxr import PhysxSchema, UsdPhysics
                # Headless IsaacLab disables scene-query support by default.
                # Enable this observation feature before cooking any shapes.
                for prim in self._stage.Traverse():
                    if prim.IsA(UsdPhysics.Scene):
                        PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableSceneQuerySupportAttr().Set(True)
                for side in ("left", "right"):
                    path = f"/World/HandsOnRobots/{side.title()}Robot/base_{side}"
                    PhysxSchema.PhysxContactReportAPI.Apply(self._stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)
                self._stage.Flatten().Export(str(output / "composed_before_first_physics.usda"))
                # Public API cooks the already composed scene while the
                # timeline is stopped; do not simulate or advance a frame.
                omni.physx.get_physx_interface().force_load_physics_from_usd()
                result["physics_steps_before_first_geometry_query"] = physics_steps
                (output / "live_validation.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
                if physics_steps:
                    raise RuntimeError("The required pre-physics query ran after a physics step")
                from gen3_composed_collision import ComposedCollisionAudit
                cooked = export_cooked_shapes(self._stage, output, physics_steps)
                self._gen3_composed_audit = ComposedCollisionAudit(self._stage, shape_inventory(self._stage), cooked)
                audit_shapes(self, "before_first_physics", output)

            def _sample_reset_state(self, *, clear_manual_request):
                super()._sample_reset_state(clear_manual_request=clear_manual_request)
                if self.boundary_pose is not None:
                    x, y, yaw = self.boundary_pose
                    self._lw_reset_sample = replace(self._lw_reset_sample,
                        root_x_m=x, root_y_m=y, root_yaw_deg=yaw)
                    self._lw_state.reset(self._lw_reset_sample)

            def _reset_scene_state(self):
                super()._reset_scene_state()
                index = self._gen3_audit_index
                self._gen3_audit_index += 1
                audit_shapes(self, f"reset_{index:03d}_before_settle", output)

        runtime.HandsOnLWNativeArticulationIsaacLabEnv = AuditedGEN3
        from handson.tasks.eval_gen02_open_trash_bin.config import TaskConfig
        try:
            env = TaskConfig().get_native_environment(policy_mode=True, seed=0,
                render_mode="rgb_array", randomize=True,
                launch_isaac=True, headless=True, image_obs=False, isaac_device=device,
                existing_simulation_app=app)
        finally:
            runtime.HandsOnLWNativeArticulationIsaacLabEnv = canonical_class
        raw = env.unwrapped
        if args.contact_probe:
            from probe_gen3_lid_contact import run_contact_probe
            if args.contact_batch:
                result['dynamic_contact_trials'] = []
                poses = [(-.28,0.,0.),(-.36,-.08,-20.),(-.36,.08,20.),(-.20,-.08,20.),(-.20,.08,-20.)]
                for index, pose in enumerate(poses):
                    trial = {'pose': pose}
                    result['dynamic_contact_trials'].append(trial)
                    folder = output / f'contact_pose_{index:02d}'
                    folder.mkdir()
                    try:
                        run_contact_probe(env, raw, folder, trial, pose=pose)
                    except RuntimeError as exc:
                        if str(exc) != 'The physical contact probe did not reach the unchanged 75-degree success threshold':
                            raise
                        trial['diagnostic_failure'] = str(exc)
                    (output / 'live_validation.json').write_text(json.dumps(_json_ready(result), indent=2)+'\n')
                if not all(trial['contact_probe']['physical_success_observed'] for trial in result['dynamic_contact_trials']):
                    raise RuntimeError('Not all five physical contact poses passed; retained failures remain acceptance failures')
            else:
                run_contact_probe(env, raw, output, result)
            return
        if args.probe_steps:
            import numpy as np
            result["diagnostic_only"] = True
            result["robot_probe"] = []
            for step in range(args.probe_steps + 1):
                if step % 10 == 0 or step == args.probe_steps:
                    row = {"control_step": step, "task_truth": raw._read_lw_truth(), "robots": {}}
                    from pxr import PhysicsSchemaTools
                    fields = raw._physx_ground_truth_trace.latest_substep_fields((
                        "physx_contact_report_valid", "physx_contact_capture_source",
                        "physx_contact_header_count", "physx_contact_header_ids",
                        "physx_contact_header_meta", "physx_contact_point_data"))
                    row["contacts"] = []
                    row["contact_capture_valid"] = fields.get("physx_contact_report_valid")
                    row["contact_capture_source"] = fields.get("physx_contact_capture_source")
                    for index in range(int(fields.get("physx_contact_header_count", [0])[0])):
                        ids = fields["physx_contact_header_ids"][index]
                        meta = fields["physx_contact_header_meta"][index]
                        sample = fields["physx_contact_point_data"][int(meta[1]):int(meta[1]+meta[2])]
                        row["contacts"].append({"actor_paths": [str(PhysicsSchemaTools.intToSdfPath(int(x))) for x in ids[:2]],
                            "collider_paths": [str(PhysicsSchemaTools.intToSdfPath(int(x))) for x in ids[2:4]],
                            "points": [{"position": p[:3], "separation_m": float(p[9]), "impulse": p[6:9]} for p in sample]})
                    for side, robot in raw._robots.items():
                        robot.update(0.)
                        row["robots"][side] = {"joint_names": robot.joint_names,
                            "q": robot.data.joint_pos.detach().cpu().numpy(),
                            "qd": robot.data.joint_vel.detach().cpu().numpy(),
                            "target": raw._joint_targets[side].detach().cpu().numpy(),
                            "limits": robot.data.soft_joint_pos_limits.detach().cpu().numpy(),
                            "body_names": robot.body_names,
                            "body_state": robot.data.body_state_w.detach().cpu().numpy()}
                    result["robot_probe"].append(row)
                if step < args.probe_steps:
                    raw.step(None)
            audit_shapes(raw, "diagnostic_idle_after_motion", output)
            return
        for seed in range(20):
            result["stable_origins"].append(_reset_and_accept_origin(env, raw, seed))
            audit_shapes(raw, f"seed_{seed:02d}_after_stable_origin", output)
            (output / "live_validation.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        for index, pose in enumerate(itertools.product([-.4, -.28, -.16], [-.12, 0., .12], [-30., 0., 30.])):
            raw.boundary_pose = pose
            row = _reset_and_accept_origin(env, raw, index)
            row["boundary_pose"] = pose
            result["stable_origins"].append(row)
            audit_shapes(raw, f"boundary_{index:02d}_after_stable_origin", output)
            (output / "live_validation.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        raw._stage.Flatten().Export(str(output / "composed_after_stable_origin.usda"))
        result["composed_reset_and_origin_checks_passed"] = True
    except BaseException as exc:
        result["exception"] = traceback.format_exc()
        result["exception_diagnostics"] = getattr(exc, "diagnostics", None)
        print(result["exception"], flush=True)
        raise
    finally:
        (output / "live_validation.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        physics_subscription = None
        # IsaacLab's standalone STOP callback otherwise waits forever for
        # playback to resume during app.close(). Public clear_instance()
        # unsubscribes it before the owned diagnostic process shuts down.
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        if env is not None:
            env.unwrapped._fast_shutdown_on_close = True
            env.close()
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
