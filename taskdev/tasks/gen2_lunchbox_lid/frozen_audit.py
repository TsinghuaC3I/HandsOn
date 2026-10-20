"""Query unchanged final collision geometry at recorded GPU poses; no simulation."""
from argparse import Namespace
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np


def run(directory,out):
    from isaaclab.app import AppLauncher
    from handson.sim.isaaclab.compat import create_app_launcher
    from handson.sim.isaaclab.runtime_display import isaac_kit_args
    launcher=create_app_launcher(AppLauncher,Namespace(headless=True,enable_cameras=False,device='cuda:0',
        kit_args=isaac_kit_args(headless=True)))
    app=launcher.app
    import omni.usd,omni.physx
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,PhysicsSchemaTools,Gf,Sdf
    out=Path(out);cache=out/'frozen_cache';cache.mkdir(exist_ok=True)
    results=[]
    for path in sorted(Path(directory).glob('*.json')):
        data=json.loads(path.read_text())
        if 'actors' not in data:continue
        base=path.parent/data['geometry_file']
        assert hashlib.sha256(base.read_bytes()).hexdigest()==data['geometry_sha256']
        layer=Sdf.Layer.CreateAnonymous();stage=Usd.Stage.Open(Sdf.Layer.FindOrOpen(str(base)),layer)
        stage.SetEditTarget(layer)
        pred=Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        prims=[p for p in Usd.PrimRange(stage.GetPseudoRoot(),pred) if p.IsActive()]
        joint_pairs=set();fixed={};excluded={}
        def root(p):
            while p in fixed:p=fixed[p]
            return p
        for side,source in data['robot_sources'].items():
            prefix=f'/World/HandsOnRobots/{side.capitalize()}Robot/base_{side}/'
            for x in ET.parse(source).findall('./contact/exclude'):
                excluded[tuple(sorted((prefix+x.get('body1'),prefix+x.get('body2'))))]=source
        for prim in prims:
            if prim.IsA(UsdPhysics.Joint):
                joint=UsdPhysics.Joint(prim);a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
                if a and b and joint.GetJointEnabledAttr().Get() is not False:
                    pair=tuple(sorted((str(a[0]),str(b[0]))));joint_pairs.add(pair)
                    if prim.IsA(UsdPhysics.FixedJoint):
                        ra,rb=root(pair[0]),root(pair[1])
                        if ra!=rb:fixed[rb]=ra
                prim.SetActive(False)  # Override root-layer joints in this session layer.
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if prim.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableGPUDynamicsAttr().Set(False)
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableSceneQuerySupportAttr().Set(True)
        for actor,record in data['actors'].items():
            pose=np.array(record['pose_wxyz']);prim=stage.GetPrimAtPath(actor)
            matrix=Gf.Matrix4d(1);matrix.SetRotate(Gf.Quatd(float(pose[3]),Gf.Vec3d(*map(float,pose[4:]))))
            matrix.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
            xf=UsdGeom.Xformable(prim);xf.MakeMatrixXform().Set(matrix);xf.SetResetXformStack(True)
        check_cache=UsdGeom.XformCache()
        max_pose_error=max(float(np.linalg.norm(np.array(check_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(actor)).ExtractTranslation())-np.array(record['pose_wxyz'][:3]))) for actor,record in data['actors'].items())
        assert max_pose_error<1e-8,max_pose_error
        assert not any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
        frozen=cache/'current_frozen.usdc';stage.Flatten().Export(str(frozen))
        omni.usd.get_context().open_stage(str(frozen))
        app.update()
        omni.physx.get_physx_interface().force_load_physics_from_usd()
        query=omni.physx.get_physx_scene_query_interface();hits=set();counts={}
        for collider,info in data['colliders'].items():
            assert info['visibility']=='invisible',(collider,info)
            if info['type']=='Plane':continue
            def report(hit):
                other=str(hit.collision)
                if collider!=other:hits.add(tuple(sorted((collider,other))))
                return True
            counts[collider]=int(query.overlap_shape(*PhysicsSchemaTools.encodeSdfPath(Sdf.Path(collider)),report,False))
        contacts={tuple(c['actors']):c for c in data.get('contact_pairs',[])}
        pairs=[];bad=[]
        for a,b in sorted(hits):
            aa=data['colliders'][a]['owner'];bb=data['colliders'][b]['owner'];body_pair=tuple(sorted((aa,bb)))
            reason=('same source rigid body' if aa==bb else 'authored adjacent joint' if body_pair in joint_pairs else
                    'authored fixed assembly' if root(aa)==root(bb) else 'explicit source collision exclusion' if body_pair in excluded else None)
            contact=contacts.get(body_pair)
            support=any('/World/Task/table/' in x for x in body_pair) and any('/World/Task/gen2_' in x for x in body_pair)
            if reason is None and support and contact and contact['min_separation_m'] is not None and contact['min_separation_m']>=-.001:
                reason='intended table support; live GPU depth within 1 mm contact shell'
            row=dict(pair=[a,b],actors=[aa,bb],reason=reason,contact=contact);pairs.append(row)
            if reason is None:bad.append(row)
        lid=next(v['pose_wxyz'] for k,v in data['actors'].items() if k.endswith('/gen2_lid'))
        ray_hits=[]
        def report_ray(hit):
            ray_hits.append(dict(collision=str(hit.collision),position=list(hit.position),distance=float(hit.distance)))
            return True
        query.raycast_all(tuple(np.array(lid[:3])+[0,0,.2]),(0.,0.,-1.),.3,report_ray)
        lid_hits=[h for h in ray_hits if '/gen2_lid/' in h['collision']]
        error=min((abs(float(h['position'][2])-float(lid[2])-.008) for h in lid_hits),default=1e9)
        row=dict(snapshot=path.name,geometry_sha256=data['geometry_sha256'],actors=len(data['actors']),
                 collider_count=len(data['colliders']),query_counts=counts,overlaps=pairs,unapproved=bad,
                 live_gpu_contacts=list(contacts.values()),lid_surface_ray_hits=ray_hits,lid_surface_error_m=error,
                 passed=not bad and error<.0001 and max(counts.values())>0,
                 simulation_steps=0,max_actor_position_error_m=max_pose_error,method='Static PhysX query of unchanged final shapes at live GPU actor poses')
        results.append(row)
        (out/'frozen_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('[GEN2] Frozen query',path.name,'unapproved',len(bad),'ray error',error,flush=True)
        omni.physx.get_physx_interface().release_physics_objects()
        omni.usd.get_context().close_stage()
    return app,dict(frozen_audits=results,frozen_geometry_queries_passed=bool(results) and all(r['passed'] for r in results))
