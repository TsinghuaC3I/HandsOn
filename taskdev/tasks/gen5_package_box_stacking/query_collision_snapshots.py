"""Query the exact composed colliders at immutable authored/live GPU snapshots.

The live GPU pipeline suppresses CPU readback. These snapshots preserve its
actual tensor transforms and original collider geometry, then use a separate
non-stepping PhysX query scene. This script never writes to the live scene.
"""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'handson')]


def query_one(source_layer,snapshot):
    import carb,omni.physx
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Sdf,Gf,PhysicsSchemaTools
    stage=Usd.Stage.Open(source_layer,Sdf.Layer.CreateAnonymous('collision_audit_session.usda'))
    stage.SetEditTarget(stage.GetSessionLayer())
    while True:
        instances=[p for p in stage.Traverse() if p.IsInstance()]
        if not instances:break
        for p in instances:p.SetInstanceable(False)
    # Queries need collision geometry only. Disabling motion/constraints in
    # this private copy prevents joint initialization from changing a measured
    # link pose. The original collision approximation/materials are preserved.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):prim.SetActive(False);continue
        for api in (UsdPhysics.RigidBodyAPI,UsdPhysics.ArticulationRootAPI,PhysxSchema.PhysxArticulationAPI,PhysxSchema.PhysxRigidBodyAPI):
            if prim.HasAPI(api):prim.RemoveAPI(api)
        if prim.IsA(UsdPhysics.Scene):
            settings=PhysxSchema.PhysxSceneAPI.Apply(prim)
            settings.CreateEnableGPUDynamicsAttr().Set(False)
            settings.CreateBroadphaseTypeAttr().Set('MBP')
            settings.CreateEnableSceneQuerySupportAttr().Set(True)
    max_pose_error=0.0
    for path,row in sorted(snapshot['bodies'].items(),key=lambda item:item[0].count('/')):
        prim=stage.GetPrimAtPath(path)
        parent=np.asarray(UsdGeom.XformCache().GetParentToWorldTransform(prim)).T
        world=np.asarray(row['matrix']);local=np.linalg.inv(parent).dot(world)
        x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.SetResetXformStack(False)
        x.AddTransformOp().Set(Gf.Matrix4d(tuple(tuple(float(v) for v in line) for line in local.T)))
        actual=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        max_pose_error=max(max_pose_error,float(np.abs(actual-world).max()))
    assert max_pose_error<1e-7,max_pose_error
    cache=UsdUtils.StageCache.Get();stage_id=cache.Insert(stage)
    interface=omni.physx.get_physx_simulation_interface()
    query=omni.physx.get_physx_scene_query_interface()
    pairs=[];self_hits={};controls={};errors=[]
    package_bounds={}
    for row in snapshot['inventory']:
        if '/gen5_package_' not in row['path']:continue
        prim=stage.GetPrimAtPath(row['path']);mesh=UsdGeom.Mesh(prim)
        points=np.asarray(mesh.GetPointsAttr().Get());matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
        world=np.c_[points,np.ones(len(points))].dot(matrix)[:,:3]
        package_bounds[row['owner']]={'min':world.min(0).tolist(),'max':world.max(0).tolist()}
    owner={row['path']:row['owner'] for row in snapshot['inventory']}
    try:
        interface.attach_stage(stage_id.ToLongInt())
        for source in snapshot['inventory']:
            if source['type']=='Plane':continue
            hits=[]
            def report(hit):hits.append(str(hit.collision));return True
            encoded=PhysicsSchemaTools.encodeSdfPath(Sdf.Path(source['path']))
            try:
                query.overlap_shape(encoded[0],encoded[1],report,False)
                self_hits[source['path']]=source['path'] in hits
                for target in hits:
                    if target!=source['path']:
                        pairs.append({'source':source['path'],'source_body':source['owner'],
                            'target':target,'target_body':owner.get(target)})
            except Exception as exc:errors.append({'path':source['path'],'error':repr(exc)})
        for path,row in snapshot['bodies'].items():
            if '/gen5_package_' not in path:continue
            hits=[]
            def control_report(hit):hits.append(str(hit.collision));return True
            position=np.asarray(row['matrix'])[:3,3]
            query.overlap_sphere(.002,carb.Float3(*map(float,position)),control_report,False)
            controls[path]={'center':position.tolist(),'hits':hits,'own_body_found':any(owner.get(p)==path for p in hits)}
    finally:
        interface.detach_stage();cache.Erase(stage_id)
    return {'phase':snapshot['phase'],'max_snapshot_matrix_error':max_pose_error,'pairs':pairs,
        'self_hits':self_hits,'physical_center_controls':controls,'errors':errors,'package_bounds':package_bounds,
        'functional':bool(self_hits) and all(self_hits.values()) and bool(controls) and all(r['own_body_found'] for r in controls.values()) and not errors}


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=Path(os.environ.get('GEN5_SNAPSHOT_SOURCE_DIR', str(Path(__file__).parent/'evidence/live_snapshot_v1'))))
    parser.add_argument('--seed');parser.add_argument('--steps');parser.add_argument('--audit-suite')
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=False;args.headless=True
    launcher=AppLauncher(args)
    result={'scope':'Independent non-stepping PhysX queries on exact composed colliders and authored/live GPU tensor snapshots','queries':[],'passed':False}
    try:
        from pxr import Sdf
        main=json.loads((args.source_dir/'result.json').read_text());layouts=json.loads((args.source_dir/'layouts.json').read_text())
        source=Sdf.Layer.FindOrOpen(str(args.source_dir/'composed_scene.usdc'))
        from pxr import Usd
        original=Usd.Stage.Open(source)
        result['original_self_collision_settings']={str(p.GetPath()):{a.GetName():a.Get() for a in p.GetAttributes() if 'selfcollis' in a.GetName().lower()} for p in original.Traverse() if any('selfcollis' in a.GetName().lower() for a in p.GetAttributes())}
        result['source_directory']=str(args.source_dir)
        snapshots=[main['authored_first_state'],*main['sampled_reset_queries'],main['post_hold_shapes']]
        for row in layouts['seeds']:
            snapshots.extend([row['shape_query'],row['after_hold_query']])
        for row in layouts['layouts']:
            snapshots.extend([row['before_rollout_query'],row['after_hold_query']])
        for i,snapshot in enumerate(snapshots):
            row=query_one(source,snapshot);result['queries'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(f"[GEN5] Snapshot query {i+1}/{len(snapshots)}: functional={row['functional']}, pairs={len(row['pairs'])}",flush=True)
        result['passed']=all(row['functional'] for row in result['queries'])
    except Exception:
        import traceback
        result['exception']=traceback.format_exc();traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        launcher.app.close(wait_for_replicator=False,skip_cleanup=True)
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
