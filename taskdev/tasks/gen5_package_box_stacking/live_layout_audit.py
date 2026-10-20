"""Live GEN5 multipose diagnostics; never a substitute for operator recordings."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from handson.sim.isaaclab import eval_gen07_stack_boxes as gen5


def array(value):
    if hasattr(value, 'detach'): value=value.detach().cpu().numpy()
    return np.asarray(value)


def robot_readback(env):
    rows={}
    for side, robot in env._robots.items():
        data=robot.data
        q, qd=array(data.joint_pos)[0], array(data.joint_vel)[0]
        target=array(data.joint_pos_target)[0]
        rows[side]={'joint_names':list(robot.joint_names),'position':q.tolist(),
                    'velocity':qd.tolist(),'target':target.tolist(),
                    'max_joint_speed_rad_s':float(np.max(np.abs(qd))),
                    'max_target_error_rad':float(np.max(np.abs(q-target)))}
    return rows


def contact_readback(env, obs):
    table=env.get_recording_schema()['physx']['contact_report']['path_table']
    paths=sorted(row['path'] for row in table)
    n=int(obs['physx_contact_signature_pair_count'][0])
    pairs=[]
    for indices, counts, metrics in zip(obs['physx_contact_signature_pair_path_indices'][:n],
            obs['physx_contact_signature_pair_counts'][:n],obs['physx_contact_signature_pair_metrics'][:n]):
        pairs.append({'paths':[paths[int(i)-1] for i in indices],
            'contact_points':int(counts[0]),'friction_anchors':int(counts[1]),
            'max_penetration_m':float(metrics[0]),'normal_impulse_ns':float(metrics[1]),
            'friction_impulse_ns':float(metrics[2])})
    return {'valid':bool(obs['physx_contact_signature_valid'][0]),'pairs':pairs}


def frame(env,obs,info):
    poses={name:env._physics_body_pose(name).tolist() for name in gen5.BODY_NAMES}
    velocities={name:env._physics_body_velocity(name).tolist() for name in gen5.BODY_NAMES}
    return {'poses':poses,'velocities':velocities,'robots':robot_readback(env),
            'contacts':contact_readback(env,obs),'succeed':bool(info.get('succeed')),
            'info':{k:v for k,v in info.items() if k.startswith('stack_') or k in ('manual_reset','isaac_warnings')}}


def mesh_points(stage,prim):
    from pxr import UsdGeom
    mesh=UsdGeom.Mesh(prim)
    transform=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    matrix=np.asarray(transform,dtype=float)
    points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    return np.c_[points,np.ones(len(points))].dot(matrix)[:,:3]


def composed_geometry(env):
    from pxr import Usd,UsdGeom,UsdPhysics
    report={'inventory':[], 'packages':{}}
    for prim in Usd.PrimRange(env._stage.GetPseudoRoot(),Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.IsActive():continue
        if not prim.IsA(UsdGeom.Imageable):continue
        visible=str(UsdGeom.Imageable(prim).ComputeVisibility())
        collider=prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        geometric=prim.IsA(UsdGeom.Gprim)
        role=('physical_collider' if collider else 'render_geometry' if geometric and visible!='invisible' else 'hidden_helper' if geometric else 'scene_structure')
        report['inventory'].append({'path':str(prim.GetPath()),'type':str(prim.GetTypeName()),'role':role,'visibility':visible})
    for name in gen5.BODY_NAMES:
        body=env._find_rigid_body_prim(name); visuals=[];colliders=[]
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsA(UsdGeom.Mesh):continue
            points=mesh_points(env._stage,prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI):colliders.append((prim,points))
            elif str(UsdGeom.Imageable(prim).ComputeVisibility())!='invisible':visuals.append((prim,points))
        assert visuals and colliders,(name,'missing visible mesh or collider')
        v=np.concatenate([p for _,p in visuals]); c=np.concatenate([p for _,p in colliders])
        eq=ConvexHull(v).equations
        protrusion=float(np.maximum(c.dot(eq[:,:3].T)+eq[:,3],0).max())
        error=max(float(cKDTree(v).query(c)[0].max()),float(cKDTree(c).query(v)[0].max()))
        report['packages'][name]={'visual_paths':[str(p.GetPath()) for p,_ in visuals],
            'collider_paths':[str(p.GetPath()) for p,_ in colliders],
            'collider_visibility':[str(UsdGeom.Imageable(p).ComputeVisibility()) for p,_ in colliders],
            'visual_min':v.min(0).tolist(),'visual_max':v.max(0).tolist(),
            'collision_min':c.min(0).tolist(),'collision_max':c.max(0).tolist(),
            'max_outward_protrusion_m':protrusion,'vertex_surface_difference_m':error,
            'mass_kg':float(UsdPhysics.MassAPI(body).GetMassAttr().Get()),
            'diagonal_inertia':list(UsdPhysics.MassAPI(body).GetDiagonalInertiaAttr().Get())}
    return report


def set_layout(env,poses):
    # Diagnostic scenario initialization only. No writes occur during the
    # subsequent gravity/contact-driven rollouts or any replay.
    for name,pose in poses.items():
        assert env._set_task_body_world_pose(env._task_body_views[name],np.asarray(pose))
    env._sim.forward()
    env._refresh_object_poses()


def make_layout(env,index):
    specs=gen5.geometry()['objects']; top=env._current_table_top_z()
    names=list(gen5.BODY_NAMES); xy={n:list(gen5.RESET_XY[n]) for n in names}
    if index<4:
        corner=[(0,0),(1,1),(0,1),(1,0)][index]
        xy={n:[gen5.RESET_XY_BOUNDS[n][corner[0]][0],gen5.RESET_XY_BOUNDS[n][corner[1]][1]] for n in names}
    poses={}
    if index<7:
        angles=[(0,0,45),(0,0,90),(90,0,0),(-90,0,0),(0,90,0),(0,-90,0),(180,0,0)]
        for i,name in enumerate(names):
            angle=angles[index] if name!=gen5.BASE_NAME else (0,0,45 if index%2==0 else 90)
            if index==6 and name in (gen5.SMALL_NAMES[0],gen5.SMALL_NAMES[2]):
                angle=(8,-7,45) if name==gen5.SMALL_NAMES[0] else (-10,5,90)
            rot=Rotation.from_euler('xyz',angle,degrees=True)
            # Exact source vertices determine support for each resting face.
            import trimesh
            mesh=trimesh.load(gen5.ASSET_DIR/specs[name]['collision_mesh'],process=False)
            z=top-float(rot.apply(mesh.vertices)[:,2].min())+0.0001
            q=rot.as_quat();poses[name]=[*xy[name],z,q[3],*q[:3]]
        return f'face_or_yaw_{index}',poses,False
    z=top
    for i,name in enumerate(names):
        rot=Rotation.from_euler('xyz',(0,0,45 if index==8 else 0),degrees=True)
        h=gen5.half_extents()[name][2]
        # Offset stack retains broad overlap; edge stack is an intentional
        # near-miss test and is allowed to topple without terminal/reset.
        x=-0.29+(0.003*i if index==8 else 0.013*i if index==9 else 0)
        z+=h
        q=rot.as_quat();poses[name]=[x,0.10,z+0.0001*i,q[3],*q[:3]]
        z+=h
    return ('center_stack','offset_stack','edge_stack')[index-7],poses,index in (7,8)


def run(env,capture_shape_overlap,out,*,seed_count=20,layout_indices=range(10),layout_steps=90):
    result={'scope':'Live physical layouts, separate from ordinary collection/operator acceptance',
            'seeds':[],'layouts':[],'geometry':composed_geometry(env)}
    def save():out.write_text(json.dumps(result,indent=2,default=lambda x:array(x).tolist())+'\n')
    for seed in range(seed_count):
        obs,info=env.reset(seed=seed)
        first_authored=env._audit_reset_rows[-1]['poses']
        obs,info=env.reset(seed=seed)
        repeated_authored=env._audit_reset_rows[-1]['poses']
        repeated_equal=all(np.array_equal(first_authored[n],repeated_authored[n]) for n in gen5.BODY_NAMES)
        start=frame(env,obs,info)
        before=capture_shape_overlap(env,f'seed_{seed}_after_reset')
        samples=[]
        for _ in range(30):
            obs,reward,terminated,truncated,info=env.step(None)
            row=frame(env,obs,info);row.update(terminated=bool(terminated),truncated=bool(truncated));samples.append(row)
        result['seeds'].append({'seed':seed,'first_authored_poses':first_authored,'repeated_authored_poses':repeated_authored,
            'same_seed_exact_repeatability':repeated_equal,'after_reset':start,'shape_query':before,
            'after_hold_query':capture_shape_overlap(env,f'seed_{seed}_after_hold'),
            'frames':samples})
        save();print(f'[GEN5] Live seed {seed+1}/20 complete.',flush=True)
    for index in layout_indices:
        env.reset(seed=7)
        name,poses,expected_success=make_layout(env,index)
        set_layout(env,poses)
        before=capture_shape_overlap(env,f'{name}_before_rollout')
        samples=[]
        for _ in range(layout_steps):
            obs,reward,terminated,truncated,info=env.step(None)
            row=frame(env,obs,info);row.update(terminated=bool(terminated),truncated=bool(truncated));samples.append(row)
        result['layouts'].append({'name':name,'authored_poses':poses,'expected_success':expected_success,
            'before_rollout_query':before,'after_hold_query':capture_shape_overlap(env,f'{name}_after_hold'),
            'frames':samples,'geometry':composed_geometry(env)})
        save();print(f'[GEN5] Live layout {index+1}/10 ({name}) complete.',flush=True)
    return result
