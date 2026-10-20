#!/usr/bin/env python3
"""Review retained GEN3 live motion shapes and all recorded PhysX substeps."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import fcl
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]


def audit(stage_path, inventory_path, probe_path):
    stage = Usd.Stage.Open(str(stage_path))
    inventory = json.loads(inventory_path.read_text())["inventory"]
    support = {tuple(p["paths"]) for p in json.loads((ROOT / "taskdev/tasks/eval_gen02_open_trash_bin/evidence/reference_robot_support_review.json").read_text())["pairs"]}
    fixed = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint) or "/HandsOnRobots/" not in str(prim.GetPath()):
            continue
        joint = UsdPhysics.Joint(prim)
        first, second = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if first and second:
            a, b = str(first[0]), str(second[0])
            fixed.setdefault(a, set()).add(b)
            fixed.setdefault(b, set()).add(a)

    def connected(a, b):
        queue, seen = [a], {a}
        for current in queue:
            for neighbor in fixed.get(current, ()):
                if neighbor == b:
                    return True
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return False

    source_shapes = {}

    def source_shape(path, matrices):
        body = inventory[path]["body"]
        if path not in source_shapes:
            prim = stage.GetPrimAtPath(path)
            source = UsdGeom.Mesh(prim)
            visual = UsdGeom.Mesh(stage.GetPrimAtPath(path.replace("/collisions/", "/visuals/")))
            vertices = np.asarray(source.GetPointsAttr().Get(), dtype=float)
            faces = np.asarray(source.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            if not np.array_equal(vertices, np.asarray(visual.GetPointsAttr().Get())) or not np.array_equal(faces, np.asarray(visual.GetFaceVertexIndicesAttr().Get())):
                raise ValueError(f"Source collision mesh differs from original visible mesh: {path}")
            body_world = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(0)).T
            mesh_world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            relative = np.linalg.inv(body_world) @ mesh_world
            vertices = vertices @ relative[:3, :3].T + relative[:3, 3]
            triangles = faces.reshape(-1, 3)
            mesh = fcl.BVHModel()
            mesh.beginModel(len(vertices), len(triangles))
            mesh.addSubModel(vertices, triangles)
            mesh.endModel()
            source_shapes[path] = mesh
        world = np.asarray(matrices[body])
        return fcl.CollisionObject(source_shapes[path], fcl.Transform(world[:3, :3], world[:3, 3]))

    probe = json.loads(gzip.decompress(probe_path.read_bytes()) if probe_path.suffix == '.gz' else probe_path.read_text())
    unknown, manipulation, source_reviews = [], [], []
    samples = list(probe["composed_motion_samples"])
    if 'final_composed_geometry' in probe:
        samples.append({'step': probe['frames'][-1]['step'], 'phase': 'final_physical_pose', 'geometry': probe['final_composed_geometry']})
    for sample in samples:
        geometry = sample["geometry"]
        for pair in geometry["overlaps"]:
            paths = tuple(pair["paths"])
            a, b = [inventory[path] for path in paths]
            groups = {a["group"], b["group"]}
            row = {"step": sample["step"], "phase": sample["phase"], **pair}
            if paths in support or connected(a["body"], b["body"]) or (groups == {"bin_base", "lid"} and "SourceMesh06" in {a["source_mesh"], b["source_mesh"]}):
                continue
            if "lid" in groups and groups & {"right_robot", "left_robot"} and any("finger" in path for path in paths):
                manipulation.append(row)
                continue
            # The nominal lift exposed this precise reference robot proxy
            # pair. Re-measure its ORIGINAL surfaces at every new pose; do
            # not exempt other self-pairs or reuse an old clearance value.
            body_names = {a["body"].rsplit("/", 1)[-1], b["body"].rsplit("/", 1)[-1]}
            side = 'right' if groups == {'right_robot'} else 'left' if groups == {'left_robot'} else None
            if side and body_names == {f"link0_{side}", f"link2_{side}"}:
                shapes = [source_shape(path, geometry["body_world_matrices"]) for path in paths]
                result = fcl.CollisionResult()
                fcl.collide(*shapes, fcl.CollisionRequest(enable_contact=True), result)
                distance = fcl.DistanceResult()
                separation = fcl.distance(*shapes, fcl.DistanceRequest(enable_nearest_points=True), distance)
                row.update(source_visual_surfaces_intersect=bool(result.is_collision), source_surface_clearance_m=float(separation), source_nearest_points=np.asarray(distance.nearest_points).tolist())
                source_reviews.append(row)
                if not result.is_collision and separation > 1e-7:
                    continue
            unknown.append(row)
    body_groups = {row["body"]: row["group"] for row in inventory.values()}
    unexpected_contacts, missing_substeps = [], []
    max_depth = 0.
    for frame in probe["frames"]:
        substeps = frame["contacts"].get("all_substeps", [])
        if len(substeps) != 4 or not all(s["valid"] for s in substeps):
            missing_substeps.append(frame["step"])
        for substep in substeps:
            max_depth = max(max_depth, substep["max_penetration_depth_m"])
            points = np.asarray(substep["points"])
            for paths, meta in zip(substep["actor_paths"], substep["header_meta"]):
                groups = {body_groups.get(path, "unknown") for path in paths}
                contact_points = points[int(meta[1]):int(meta[1] + meta[2])]
                if groups & {"right_robot", "left_robot"} and groups & {"bin_base", "table", "ground_support"} and len(contact_points) and np.any(np.linalg.norm(contact_points[:, 6:9], axis=1) > 1e-6):
                    unexpected_contacts.append({"step": frame["step"], "actor_paths": paths, "points": contact_points.tolist()})
    return {"probe": str(probe_path), "scope": "Actual tensor-pose cooked-shape snapshots and all four recorded PhysX substeps per control frame; scripted diagnostic only",
        "motion_sample_count": len(samples), "control_frame_count": len(probe["frames"]),
        "unapproved_overlaps": unknown, "intended_finger_lid_contacts": manipulation,
        "individually_measured_original_robot_proxy_pairs": source_reviews,
        "unexpected_robot_support_contacts": unexpected_contacts, "incomplete_contact_frames": missing_substeps,
        "maximum_full_substep_penetration_m": max_depth,
        "passed": not (unknown or unexpected_contacts or missing_substeps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.stage, args.inventory, args.probe)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
