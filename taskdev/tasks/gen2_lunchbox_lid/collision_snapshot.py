"""Capture final composed colliders and actual GPU actor poses for static queries.

GPU direct-API scene queries retain initial actor transforms in Isaac 5.1.
The frozen audit uses the unchanged cooked geometry at recorded GPU poses;
it never writes poses to the live environment or simulates a CPU episode.
"""
from pathlib import Path
import json
import hashlib
import numpy as np


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def capture(env, directory, tag):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysicsSchemaTools
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    stage=env._stage
    base=directory/'composed_scene.usdc'
    if not base.exists():stage.Flatten().Export(str(base))
    pred=Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prims=[p for p in Usd.PrimRange(stage.GetPseudoRoot(),pred) if p.IsActive()]
    actors={};colliders={};xforms=UsdGeom.XformCache()
    robot_states={}
    for side,robot in env._robots.items():
        pos=array(robot.data.body_pos_w)[0];quat=array(robot.data.body_quat_w)[0]
        robot_states[side]={name:np.r_[pos[i],quat[i]].tolist() for i,name in enumerate(robot.body_names)}
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        path=str(prim.GetPath());name=prim.GetName();pose=None;source=None
        for side,poses in robot_states.items():
            if f'/{side.capitalize()}Robot/' in path and name in poses:
                pose=poses[name];source='IsaacLab GPU articulation body state'
        if pose is None:
            physical=env._physics_body_pose(name)
            if physical is not None:pose=physical.tolist();source='PhysX GPU rigid-body tensor'
        if pose is None:
            # The fixed center camera support is a physical fixture, separate
            # from task entities. Read its actual pose for geometric auditing.
            views=getattr(env,'_gen2_audit_fixture_views',{})
            if path not in views:
                views[path]=env._task_physics_sim_view.create_rigid_body_view(path)
                env._gen2_audit_fixture_views=views
            view=views[path]
            if view.count:
                v=array(view.get_transforms())[0]
                pose=np.r_[v[:3],v[6],v[3:6]].tolist();source='PhysX GPU auxiliary fixture tensor'
        rigid=UsdPhysics.RigidBodyAPI(prim)
        kinematic=bool(rigid.GetKinematicEnabledAttr().Get())
        if pose is None:
            if not kinematic and rigid.GetRigidBodyEnabledAttr().Get() is not False:
                raise RuntimeError('Uncaptured dynamic actor: '+path)
            matrix=xforms.GetLocalToWorldTransform(prim);q=matrix.ExtractRotationQuat()
            pose=[*matrix.ExtractTranslation(),q.GetReal(),*q.GetImaginary()]
            source='Unchanged static/kinematic USD transform'
        actors[path]=dict(pose_wxyz=pose,source=source,kinematic=kinematic)
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        path=str(prim.GetPath());owner=prim
        while owner and owner.IsValid() and str(owner.GetPath()) not in actors:owner=owner.GetParent()
        owner_path=str(owner.GetPath()) if owner and owner.IsValid() else path
        colliders[path]=dict(owner=owner_path,type=prim.GetTypeName(),visibility=str(UsdGeom.Imageable(prim).ComputeVisibility()))
    contact={};contact_pairs=[]
    trace=getattr(env,'_physx_ground_truth_trace',None)
    if trace is not None:
        fields=trace._sample_tensor_contact_fields()
        if fields is not None:
            np.savez_compressed(directory/(tag+'_contacts.npz'),**fields)
            contact={k:array(v).tolist() for k,v in fields.items() if 'count' in k or 'valid' in k or 'source' in k}
            for i in range(int(fields['physx_contact_header_count'][0])):
                ids=fields['physx_contact_header_ids'][i];meta=fields['physx_contact_header_meta'][i]
                names=sorted(str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in ids[:2])
                points=fields['physx_contact_point_data'][int(meta[1]):int(meta[1]+meta[2])]
                contact_pairs.append(dict(actors=names,points=points.tolist(),
                    min_separation_m=float(points[:,9].min()) if len(points) else None))
    data=dict(tag=tag,geometry_file=base.name,geometry_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
              actors=actors,colliders=colliders,contact_summary=contact,contact_pairs=contact_pairs,
              robot_sources={side:str(robot.cfg.spawn.asset_path) for side,robot in env._robots.items()},
              sim_time=float(env._sim.current_time),physics_device=env._isaac_device,
              note='Frozen geometry audit only; all actor poses were read from the live GPU episode.')
    (directory/(tag+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    return dict(file=str(directory/(tag+'.json')),actors=len(actors),colliders=len(colliders),contact_summary=contact)
