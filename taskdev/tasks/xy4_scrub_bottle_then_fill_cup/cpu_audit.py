#!/usr/bin/env python3
"""Reproducible layout/source audit; never a PhysX or robot-reach certificate."""
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'handson'), str(ROOT)]
import numpy as np
from handson.sim.isaaclab.xy4_env import HandsOnXY4IsaacLabEnv
from handson.sim.isaaclab.xy4_scene import layout_clearances, validate_xy4_asset_files, PAIR_CLEARANCE_M
from handson.sim.isaaclab.b320_bimanual_env import _ka021_cup_reset_pose_fits_basin, _yaw_quaternion


def layout_cases():
    env = HandsOnXY4IsaacLabEnv(launch_isaac=False, image_obs=False, randomize=False)
    cases = []
    try:
        env.reset(seed=0)
        base = {k: v.copy() for k, v in env._xy4_layout.items()}
        def add(name, seed, poses):
            cases.append({
                'name': name, 'seed': seed,
                'poses_wxyz': {k: v.tolist() for k, v in poses.items()},
                'clearance_m': layout_clearances(poses),
                'mug_full_footprint_inside_basin': bool(_ka021_cup_reset_pose_fits_basin(poses['ceramic_mug'])),
                'cup_park_position_w': env._ka021_cup_park_position_w.tolist(),
            })
        add('standard', 0, base)
        env.randomize = True
        for seed in range(12):
            env.reset(seed=seed)
            add(f'seed_{seed:02}', seed, env._xy4_layout)
        env.randomize = False
        env.reset(seed=0)
        for index, signs in enumerate(product((-1., 1.), repeat=3)):
            poses = {k: v.copy() for k, v in base.items()}
            poses['hs005_bottle'][:2] += np.array([signs[0], signs[1]]) * .020
            poses['hs005_brush'][:2] += np.array([signs[1], signs[2]]) * .015
            poses['ceramic_mug'][:2] += np.array([signs[2], signs[0]]) * .010
            for name, angle in (('hs005_bottle', signs[0] * 8),
                                ('hs005_brush', signs[1] * 10),
                                ('ceramic_mug', signs[2] * 25)):
                poses[name][3:] = _yaw_quaternion(angle)
            add(f'boundary_{index:02}', None, poses)
        return cases
    finally:
        env.close()


def source_geometry():
    from pxr import Usd, UsdGeom, UsdPhysics
    from handson.sim.isaaclab.b320_bimanual_scene import KA021_HS005_SCENE_PATH
    stage = Usd.Stage.Open(str(KA021_HS005_SCENE_PATH))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
    rows = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            rows.append({'path': str(prim.GetPath()), 'shape': prim.GetTypeName(),
                         'min': list(bounds.GetMin()), 'max': list(bounds.GetMax()),
                         'computed_visibility': str(UsdGeom.Imageable(prim).ComputeVisibility())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = layout_cases()
    report = {
        'scope': 'CPU source identity, USD helper visibility and conservative XY layout screening only',
        'live_collision_or_reachability_pass': False,
        'boundary_definition': 'All 8 sign combinations at source bottle XY +/-20mm and brush XY +/-15mm; mug +/-10mm about XY4 default. Source yaw limits: bottle +/-8deg, brush +/-10deg, mug +/-25deg.',
        'layout_screen_pass': all(min(row['clearance_m'].values()) >= PAIR_CLEARANCE_M and row['mug_full_footprint_inside_basin'] for row in cases),
        'assets_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in validate_xy4_asset_files()},
        'layouts': cases,
        'hs005_source_colliders': source_geometry(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'layout_cases': len(cases), 'asset_files': len(report['assets_sha256']), 'layout_screen_pass': report['layout_screen_pass']}))
    return 0 if report['layout_screen_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
