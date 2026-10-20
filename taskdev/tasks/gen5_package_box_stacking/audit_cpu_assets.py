"""CPU source/scene preflight. This cannot certify live PhysX or teleoperation."""
from pathlib import Path
from itertools import combinations, permutations
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from handson.sim.isaaclab import eval_gen07_stack_boxes as gen5
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv, _RAISED_ARM_HOME

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).parent / "evidence"


def main():
    manifest = gen5.geometry()
    report = {"scope": "CPU source geometry, MJCF compile, seeded layout and exact convex mesh distance preflight; not live PhysX acceptance",
              "objects": {}, "layouts": [], "stack_orders": []}
    source_scene = trimesh.load_scene(gen5.ASSET_DIR / "source/cardboard_boxes/scene.gltf", process=False)
    orientations = [Rotation.from_euler("xyz", xyz, degrees=True) for xyz in
                    [(0, 0, 0), (0, 0, 45), (0, 0, 90), (90, 0, 0), (-90, 0, 0),
                     (0, 90, 0), (0, -90, 0), (180, 0, 0), (8, -7, 45), (-10, 5, 90)]]
    for name, spec in manifest["objects"].items():
        visual = trimesh.load(gen5.ASSET_DIR / spec["visual_mesh"], process=False)
        collision = trimesh.load(gen5.ASSET_DIR / spec["collision_mesh"], process=False)
        welded = visual.copy(); welded.merge_vertices(merge_tex=True, merge_norm=True)
        assert welded.is_watertight and welded.is_winding_consistent and welded.is_convex
        assert collision.is_watertight and collision.is_convex
        original = source_scene.geometry[spec["source_component"]]
        expected = (original.vertices - np.asarray(spec["source_center"])) * spec["uniform_scale_m_per_source_unit"]
        source_error = max(cKDTree(expected).query(visual.vertices)[0].max(), cKDTree(visual.vertices).query(expected)[0].max())
        assert source_error < 1e-10
        assert visual.visual.uv is not None and np.isfinite(visual.visual.uv).all()
        np.testing.assert_allclose(visual.extents, spec["dimensions_m"], atol=1e-10)
        max_surface_error = 0.0
        for rotation in orientations:
            vv, cv = rotation.apply(visual.vertices), rotation.apply(collision.vertices)
            error = max(cKDTree(vv).query(cv)[0].max(), cKDTree(cv).query(vv)[0].max())
            max_surface_error = max(max_surface_error, float(error))
        assert max_surface_error < 1e-10
        inertia = np.asarray(spec["inertia_kg_m2"])
        eigenvalues = np.linalg.eigvalsh(inertia)
        assert np.all(eigenvalues > 0) and eigenvalues[-1] <= eigenvalues[:2].sum()
        report["objects"][name] = {"source_vertex_error_m": float(source_error),
            "ten_pose_visual_collider_vertex_error_m": max_surface_error,
            "watertight": True, "convex": True, "dimensions_m": visual.extents.tolist(),
            "mass_kg": spec["mass_kg"], "inertia_eigenvalues": eigenvalues.tolist(),
            "uv_range": [visual.visual.uv.min(axis=0).tolist(), visual.visual.uv.max(axis=0).tolist()]}
    scene_path = gen5.ASSET_DIR.parent / gen5.SCENE_XML
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    report["compiled_scene"] = {"body_count": model.nbody, "geom_count": model.ngeom,
        "mesh_count": model.nmesh, "texture_count": model.ntex}
    data = mujoco.MjData(model)
    right_home = [-1.5707963, 1.5707963, 1.5707963, 1.5707963, -1.5707963, 0, 0]
    for side, values in (("right", right_home), ("left", _RAISED_ARM_HOME["left"])):
        for i, value in enumerate(values, 1):
            data.qpos[model.joint(f"joint{i}_{side}").qposadr[0]] = value
        data.qpos[model.joint(f"{side}_finger1_joint1").qposadr[0]] = 0.1
    homes = data.qpos.copy()
    object_geoms = [model.geom(name + "_collision").id for name in gen5.BODY_NAMES]
    colliders = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    table_id = model.geom("table_collision").id
    spans = {name: [] for name in gen5.BODY_NAMES}
    for seed in range(24):
        env = HandsOnABCIsaacLabEnv(gen5.TASK_NAME, launch_isaac=False, image_obs=False, randomize=True, seed=seed)
        try:
            env.reset(seed=seed)
            layout = env._task_specific_reset_xy_positions()
            kind = "seeded"
            if seed >= 20:
                kind = f"boundary_{seed - 20}"
                layout = {name: np.asarray([bounds[(seed-20)//2, 0], bounds[(seed-20)%2, 1]])
                          for name, bounds in gen5.RESET_XY_BOUNDS.items()}
            data.qpos[:] = homes
            table_z = env._current_table_top_z()
            model.body("table").pos[2] = table_z - 0.89
            poses = {}
            for name in gen5.BODY_NAMES:
                pose_xyzw = np.asarray([[*layout[name], table_z + gen5.half_extents()[name][2], 0, 0, 0, 1]], dtype=float)
                env._customize_task_reset_pose(name, pose_xyzw, applied_height=table_z-0.89)
                pose = pose_xyzw[0, [0, 1, 2, 6, 3, 4, 5]]
                qid = model.joint(name + "_joint").qposadr[0]
                data.qpos[qid:qid + 7] = pose
                poses[name] = pose.tolist()
                if seed < 20: spans[name].append(layout[name])
            mujoco.mj_forward(model, data)
            distances = []
            for g1 in object_geoms:
                for g2 in colliders:
                    if g1 == g2 or (g2 in object_geoms and g2 < g1): continue
                    distance = float(mujoco.mj_geomDistance(model, data, g1, g2, 0.3, None))
                    if distance >= 0.3: continue
                    separation = None
                    # MuJoCo's mesh GJK occasionally returns exactly zero for
                    # well-separated meshes. Record that raw result and require
                    # a positive separating-plane witness for such pairs.
                    if distance == 0 and g2 != table_id:
                        bounds = []
                        for gid in (g1, g2):
                            mid = model.geom_dataid[gid]
                            assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
                            start = model.mesh_vertadr[mid]
                            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
                            vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                            bounds.append(np.asarray([vertices.min(axis=0), vertices.max(axis=0)]))
                        gaps = np.maximum(bounds[0][0] - bounds[1][1], bounds[1][0] - bounds[0][1])
                        assert np.max(gaps) > 0, (seed, g1, g2, "ambiguous zero-distance pair")
                        separation = {"method": "world-axis projection of actual mesh vertices",
                                      "axis": int(np.argmax(gaps)), "gap_m": float(np.max(gaps))}
                    distances.append({"pair": [model.geom(g1).name, model.geom(g2).name],
                                      "geom_ids": [g1, g2],
                                      "bodies": [model.body(model.geom_bodyid[g1]).name, model.body(model.geom_bodyid[g2]).name],
                                      "geom_types": [int(model.geom_type[g1]), int(model.geom_type[g2])], "distance_m": distance,
                                      "intended_support": g2 == table_id, "zero_query_separation_witness": separation})
            unexpected = [row for row in distances if row["distance_m"] < -1e-6 and not row["intended_support"]]
            assert not unexpected, (seed, unexpected)
            report["layouts"].append({"seed": seed, "kind": kind, "poses_wxyz": poses,
                "table_top_z": table_z, "mesh_distances": distances, "unexpected_object_intersections": unexpected})
        finally:
            env.close()
    report["twenty_seed_xy_span_m"] = {name: np.ptp(values, axis=0).tolist() for name, values in spans.items()}
    assert all(min(span) > 0.01 for span in report["twenty_seed_xy_span_m"].values())
    # Run the inherited predicates with measured assets, not numerical fixtures.
    env = HandsOnABCIsaacLabEnv(gen5.TASK_NAME, launch_isaac=False, image_obs=False, randomize=False)
    try:
        env._right_tcp_pose = np.asarray([0.3, -0.4, 1.3, 1, 0, 0, 0])
        env._physics_body_velocity = lambda name: np.zeros(6)
        for upper in permutations(gen5.SMALL_NAMES):
            env._b04_velocity_samples = [];env._b04_velocity_stable_latched = False
            z = env._current_table_top_z()
            for name in (gen5.BASE_NAME, *upper):
                h = gen5.half_extents()[name][2];z += h
                env._poses[name] = np.asarray([-0.29, 0.10, z, 1, 0, 0, 0]);z += h
            assert env._block_stack_stable()
            report["stack_orders"].append(list(upper))
        env._poses[gen5.SMALL_NAMES[-1]][0] += 0.10
        assert not env._block_stack_stable()
    finally:
        env.close()
    report["status"] = "passed_cpu_preflight_only"
    (EVIDENCE / "cpu_asset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "layouts": len(report["layouts"]),
                      "source_poses_per_object": len(orientations), "upper_orders": len(report["stack_orders"])}))


if __name__ == "__main__":
    main()
