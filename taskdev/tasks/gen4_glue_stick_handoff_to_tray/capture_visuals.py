#!/usr/bin/env python3
"""Capture actual GEN4 RTX views in an explicitly separate preview process."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    sys.argv = [sys.argv[0], '--portable-root', str(root/'.gen4-local/kit'),
        '--/log/file='+str((args.output/'kit.log').resolve())]
    os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON'] = '0'
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '0'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] = '0'
    from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
    from handson.tasks.eval_gen04_pass_glue_stick import contract as c
    result = {'task':'GEN4','seed':args.seed,'mode':'separate RTX preview',
        'camera_frames':{},'exceptions':[],'not_collection_or_contact_acceptance':True}
    env = None
    try:
        env = HandsOnGEN4IsaacLabEnv(seed=args.seed, randomize=True, image_obs=True,
            headless=True, render_mode='rgb_array', physics_backend='physx',
            isaac_device='cuda:0', arm_type='rokae', hand_type='wuji1',
            render_resolution=(576,1024))
        for _ in range(60):
            _,_,terminated,truncated,_=env.step(np.zeros(54))
            assert not terminated and not truncated
        assert set(env._cameras)=={'head_camera','wrist_left_camera','wrist_right_camera'}
        for _ in range(4): frames=env.render()
        from PIL import Image
        for name,frame in zip(('main','wrist_left','wrist_right'),frames):
            rgb=np.ascontiguousarray(frame,dtype=np.uint8)
            assert rgb.ndim==3 and rgb.shape[-1]==3 and float(rgb.std())>3
            path=args.output/(name+'.png')
            Image.fromarray(rgb).save(path)
            result['camera_frames'][name]={'file':path.name,'shape':list(rgb.shape),
                'mean':float(rgb.mean()),'standard_deviation':float(rgb.std()),
                'black_fraction':float(np.all(rgb==0,axis=-1).mean())}
        assert env.observation_space['images'].contains(env._compute_observation()['images'])
        from pxr import Usd, UsdGeom, UsdPhysics
        rows=[];visible_proxies=[]
        for prim in Usd.PrimRange(env._stage.GetPseudoRoot(),Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim): continue
            path=str(prim.GetPath()); collision=prim.HasAPI(UsdPhysics.CollisionAPI)
            visibility=str(UsdGeom.Imageable(prim).ComputeVisibility())
            task_body=next((n for n in (c.GLUE_BODY,c.TRAY_BODY,*c.DISTRACTORS) if '/'+n+'/' in path),None)
            row={'path':path,'type':prim.GetTypeName(),'visibility':visibility,
                'collision':collision,'task_body':task_body}
            row['category']='physical collider' if collision else 'render geometry'
            if collision and visibility!='invisible':row['category']='render geometry and physical collider'
            rows.append(row)
            if task_body and collision and visibility!='invisible':visible_proxies.append(path)
        result['composed_geometry']=rows
        result['visible_task_collision_proxies']=visible_proxies
        assert not visible_proxies,visible_proxies
        result['reset_layout']=env._gen4_reset_layout
        result['poses']={n:env._poses[n].tolist() for n in (c.GLUE_BODY,c.TRAY_BODY,*c.DISTRACTORS)}
        result['warnings']=list(env._isaac_warnings)
        assert not any('RGB unavailable' in w for w in result['warnings'])
        result['completed']=True
    except BaseException:
        result['exceptions'].append(traceback.format_exc())
        raise
    finally:
        (args.output/'visuals.json').write_text(json.dumps(result,indent=2)+'\n')
        if env is not None:
            env._fast_shutdown_on_close=True
            env.close()


if __name__=='__main__':main()
