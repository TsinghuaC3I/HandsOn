#!/usr/bin/env python3
"""Reproducible XY6 source-asset and reset evidence; never claims PhysX QA."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handson"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/xy6/cpu_audit.json")
    args = parser.parse_args()
    from pxr import Usd, UsdGeom, UsdPhysics
    from handson.sim.isaaclab.b320_bimanual_scene import B320_TASK_SPECS, B320_SOURCE_BUNDLE_SHA256, _source_bundle_sha256
    from handson.sim.isaaclab.eval_comb06_season_and_stir_scene import TASK_NAME, OBJECT_NAMES, TRANSLATION_HALF_RANGES_M, validate_asset_files
    from handson.sim.isaaclab.eval_comb06_season_and_stir_env import HandsOnXY6IsaacLabEnv

    spec = B320_TASK_SPECS[TASK_NAME]
    validate_asset_files(spec)
    report = {"task_id": "XY6", "scope": "CPU source geometry, interface and seeded reset only",
              "live_collision_verified": False, "live_collection_verified": False,
              "assets": [], "source_bundle_hashes": {}, "resets": []}
    for key in spec.source_asset_keys:
        if key in B320_SOURCE_BUNDLE_SHA256:
            report["source_bundle_hashes"][key] = _source_bundle_sha256(key)
    for obj in spec.object_specs:
        stage = Usd.Stage.Open(str(obj.source_path))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        meshes = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                meshes.append({"path": str(prim.GetPath()),
                               "bounds_before_spawn_scale_rotation": [list(box.GetMin()), list(box.GetMax())],
                               "collision_authored": prim.HasAPI(UsdPhysics.CollisionAPI),
                               "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                               "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility())})
        report["assets"].append({"name": obj.name, "source": str(obj.source_path.relative_to(ROOT)),
                                 "sha256": hashlib.sha256(obj.source_path.read_bytes()).hexdigest(),
                                 "mass_kg": obj.mass_kg, "scale": obj.scale,
                                 "static_friction": obj.static_friction, "dynamic_friction": obj.dynamic_friction,
                                 "meshes": meshes})
    env = HandsOnXY6IsaacLabEnv(launch_isaac=False, image_obs=False, randomize=True)
    samples = []
    try:
        for seed in range(20):
            obs, info = env.reset(seed=seed)
            first = obs["xy6_object_poses"].copy()
            again, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(first, again["xy6_object_poses"])
            assert env.observation_space.contains(obs)
            samples.append(first[:, :3])
            report["resets"].append({"seed": seed, **info["xy6_reset"]})
        span = np.ptp(np.stack(samples), axis=0)
        assert np.all(span[:, :2] > .01)
        report["translation_span_m"] = dict(zip(OBJECT_NAMES, span.tolist()))
        report["boundary_layouts_for_live_validation"] = [
            {"name": f"corner_{sx}_{sy}", "poses_wxyz": {
                obj.name: [obj.home_translation[0] + sx*TRANSLATION_HALF_RANGES_M[obj.name][0],
                           obj.home_translation[1] + sy*TRANSLATION_HALF_RANGES_M[obj.name][1],
                           obj.home_translation[2], *obj.home_orientation_wxyz]
                for obj in spec.object_specs}}
            for sx in (-1, 1) for sy in (-1, 1)
        ]
        report["passed"] = True
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"CPU audit passed: 5 original assets, 20 reproducible seeds; {args.output}")


if __name__ == "__main__":
    main()
