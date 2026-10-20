"""Inspect the ordinary GEN7 environment on its selected Isaac 5.1 device.

This probe records evidence. It does not replace the platform/operator,
multi-layout reach, saved-episode or physical replay acceptance gates.
"""
from pathlib import Path
import argparse
import json
import traceback

import numpy as np


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str)+"\n")


def colliders(stage):
    from pxr import Usd, UsdPhysics
    return [prim for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]


def stage_inventory(stage):
    from pxr import UsdGeom, UsdPhysics, PhysxSchema
    rows=[]
    for prim in colliders(stage):
        api=UsdPhysics.MeshCollisionAPI(prim)
        imageable=UsdGeom.Imageable(prim)
        rows.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),
                     'visibility':str(imageable.ComputeVisibility()) if imageable else None,
                     'mesh_approximation':api.GetApproximationAttr().Get() if api else None,
                     'world_transform':np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).tolist(),
                     'contact_offset':PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(),
                     'rest_offset':PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get()})
    return rows


def physics_inventory(raw):
    from pxr import Usd,UsdPhysics,PhysxSchema
    from handson.sim.isaaclab.gen7_physics_geometry import physical_body_matrices
    matrices=physical_body_matrices(raw)
    rows=[]
    for prim in Usd.PrimRange(raw._stage.GetPrimAtPath('/World/Task')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            row={'path':str(prim.GetPath()),'gpu_world_matrix':matrices[str(prim.GetPath())].tolist() if str(prim.GetPath()) in matrices else None}
            row['attributes']={a.GetName():a.Get() for a in prim.GetAttributes() if any(key in a.GetName().lower() for key in ['mass','inertia','damping','solver','kinematic','rigidbodyenabled','sleep','stabilization'])}
            rows.append(row)
    return {'control_dt':raw._control_dt,'physics_dt':raw._physics_dt,'bodies':rows,
            'scenes':[{a.GetName():str(a.Get()) for a in prim.GetAttributes()} for prim in Usd.PrimRange.Stage(raw._stage) if prim.IsA(UsdPhysics.Scene)],
            'right_tcp_target':raw._right_tcp_pose.tolist(),'right_hand_target':raw._right_hand.tolist()}


def actual_geometry_capture(raw,output,label):
    from handson.sim.isaaclab.gen7_physics_geometry import PhysicsGeometry
    geometry=getattr(raw,'_gen7_physics_geometry',None)
    if geometry is None:geometry=raw._gen7_physics_geometry=PhysicsGeometry(raw)
    snapshot=geometry.prepare(raw)
    rows=[];arrays={}
    for i,(shape,inverse) in enumerate(snapshot.shapes):
        rows.append({'index':i,'path':shape['path'],'body':shape['body'],'kind':shape['kind'],
                     'world_matrix':np.linalg.inv(inverse).tolist()})
        rows[-1].update({key:shape[key] for key in ('axis','radius','height') if key in shape})
        if 'vertices' in shape:
            arrays[f'vertices_{i}']=shape['vertices'];arrays[f'faces_{i}']=shape['faces']
    emit(output/f'{label}_gpu_shapes.json',rows)
    if not (output/'cooked_shapes.npz').exists():np.savez_compressed(output/'cooked_shapes.npz',**arrays)


def preview_images(raw,output,label):
    from PIL import Image
    for _ in range(8):images=raw.render()
    for name,frame in zip(('head_camera','wrist_left_camera','wrist_right_camera'),images):
        if isinstance(frame,np.ndarray) and frame.ndim==3:Image.fromarray(frame).save(output/f'{label}_{name}.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=1);parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--stabilize',action='store_true')
    parser.add_argument('--boundaries',action='store_true')
    parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    import os
    import torch
    torch.cuda.set_device(os.environ['HANDSON_ISAACLAB_DEVICE'])
    if args.preview:os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER']='0'
    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
    from handson.tasks.eval_gen06_spray_shoe.config import TaskConfig
    from handson.sim.isaaclab import eval_gen06_spray_shoe_scene as scene
    contacts=[]
    phase='authored_startup'
    original_sample_reset=HandsOnABCIsaacLabEnv._sample_reset_state
    def sample_with_boundary_height(raw,**kwargs):
        result=original_sample_reset(raw,**kwargs)
        height=getattr(raw,'_gen7_probe_boundary_height',None)
        if height is not None:raw.delta_h[:]=height
        return result
    HandsOnABCIsaacLabEnv._sample_reset_state=sample_with_boundary_height
    # Keep Kit's writable files inside this task's isolated checkout.
    import shlex
    original_kit_args=HandsOnABCIsaacLabEnv._task_specific_isaac_kit_args
    def isolated_kit_args(raw):
        return ' '.join(filter(None,[original_kit_args(raw),
            '--portable-root '+shlex.quote(str(args.output/'kit')),
            '--/log/file='+shlex.quote(str(args.output/'kit.log'))]))
    HandsOnABCIsaacLabEnv._task_specific_isaac_kit_args=isolated_kit_args
    original_hook=HandsOnABCIsaacLabEnv._configure_task_simulation_settings
    def observe_setup(raw):
        original_hook(raw)
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        for prim in Usd.PrimRange.Stage(raw._stage):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        def on_contact(headers,data):
            for header in headers:
                points=[]
                for i in range(int(header.contact_data_offset),int(header.contact_data_offset+header.num_contact_data)):
                    point=data[i]
                    points.append({'separation_m':float(point.separation),'position':list(point.position),
                                   'normal':list(point.normal),'impulse':list(point.impulse)})
                contacts.append({'phase':phase,'collider0':str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                                 'collider1':str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                                 'actor0':str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                                 'actor1':str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                                 'type':int(header.type),'points':points})
        raw._gen7_probe_contact_subscription=get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
        emit(args.output/'pre_first_reset_colliders.json',stage_inventory(raw._stage))
        emit(args.output/'collision_filters.json',[
            {'path':str(prim.GetPath()),'attributes':{a.GetName():str(a.Get()) for a in prim.GetAttributes()
                 if 'selfcollision' in a.GetName().lower() or 'collisionenabled' in a.GetName().lower()},
             'relationships':{rel.GetName():[str(x) for x in rel.GetTargets()] for rel in prim.GetRelationships()
                 if 'filtered' in rel.GetName().lower() or (prim.IsA(UsdPhysics.Joint) and 'body' in rel.GetName().lower())}}
            for prim in Usd.PrimRange.Stage(raw._stage,Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.FilteredPairsAPI)])
        original_reset=raw._sim.reset
        def observe_reset(*a,**kw):
            result=original_reset(*a,**kw)
            actual_geometry_capture(raw,args.output,'after_initial_physx_reset')
            emit(args.output/'after_initial_physx_reset_colliders.json',stage_inventory(raw._stage))
            return result
        raw._sim.reset=observe_reset
        original_step=raw._sim.step
        count=0;phase_counts={}
        def observe_step(*a,**kw):
            nonlocal count
            phase_counts[phase]=phase_counts.get(phase,0)+1
            phase_count=phase_counts[phase]
            if phase.endswith('_reset') and phase_count==1:
                actual_geometry_capture(raw,args.output,phase+'_authored')
                trace=getattr(raw,'_physx_ground_truth_trace',None)
                if trace is not None:np.savez_compressed(args.output/f'{phase}_authored_snapshot.npz',**trace.snapshot())
            result=original_step(*a,**kw)
            if phase.endswith('_reset') and phase_count in (1,2,3,4,5,10,30,120):
                actual_geometry_capture(raw,args.output,phase+f'_step_{phase_count}')
                trace=getattr(raw,'_physx_ground_truth_trace',None)
                if trace is not None:np.savez_compressed(args.output/f'{phase}_step_{phase_count}_snapshot.npz',**trace.snapshot())
            count+=1
            if count==1:
                actual_geometry_capture(raw,args.output,'first_physics_step')
                emit(args.output/'first_physics_step_colliders.json',stage_inventory(raw._stage))
                print('[GEN7 probe] First Physics step captured.',flush=True)
            return result
        raw._sim.step=observe_step
    HandsOnABCIsaacLabEnv._configure_task_simulation_settings=observe_setup
    env=None
    try:
        env=TaskConfig().get_environment(policy_mode=True,sim_backend='isaacsim',
            render_mode='rgb_array',headless=True,image_obs=args.preview,randomize=True)
        raw=env.unwrapped
        raw._fast_shutdown_on_close=True
        print('[GEN7 probe] Ordinary task factory constructed.',flush=True)
        runs=[]
        layouts=[(f'seed_{seed}',seed,None) for seed in range(args.seeds)]
        if args.boundaries:
            import itertools
            layouts.insert(0,('nominal',0,(0,0,0)))
            for i,values in enumerate(itertools.product([-1,1],repeat=3)):
                layouts.insert(i+1,(f'boundary_{i}',0,values))
        original_sampler=scene.sample_layout
        for label,seed,boundary in layouts:
            phase=label+'_reset'
            raw._gen7_probe_boundary_height=(None if boundary is None else 0. if label=='nominal' or int(label.split('_')[-1])%2==0 else .02)
            if boundary is None:
                scene.sample_layout=original_sampler
            else:
                def boundary_sampler(raw):
                    sx,sy,angle=boundary
                    xy={'spray':scene.BOTTLE_XY+scene.XY_JITTER*np.array([sx,sy]),
                        scene.SHOE_NAME:scene.SHOE_XY-scene.XY_JITTER*np.array([sx,sy])}
                    yaw={'spray':angle*scene.YAW_DEGREES,scene.SHOE_NAME:-angle*scene.YAW_DEGREES}
                    if scene.layout_margin(raw,xy,yaw)<0:raise RuntimeError('Boundary layout fails production clearance sampler')
                    raw._gen7_reset_yaws=yaw
                    raw._gen7_reset_parameters=np.r_[xy['spray'],yaw['spray'],xy[scene.SHOE_NAME],yaw[scene.SHOE_NAME]]
                    return xy,yaw['spray']
                scene.sample_layout=boundary_sampler
            obs,info=env.reset(seed=seed)
            print('[GEN7 probe] Reset',label,'state shape',obs['state'].shape,'info',info,flush=True)
            actual_geometry_capture(raw,args.output,label+'_reset')
            emit(args.output/f'{label}_physics.json',physics_inventory(raw))
            emit(args.output/'recording_schema.json',raw.get_recording_schema())
            if args.preview:preview_images(raw,args.output,label+'_before_origin')
            if args.stabilize:
                from handson.sim.isaaclab.recording_origin import stabilize_recording_origin,stable_recording_origin_kwargs_for_task
                obs,info=stabilize_recording_origin(env,obs,info,
                    **stable_recording_origin_kwargs_for_task(raw.task_name),
                    progress_callback=lambda row:print('[GEN7 origin]',json.dumps(row,default=str),flush=True))
                emit(args.output/f'{label}_origin.json',info)
            phase=label+'_settled'
            frames=[]
            for step in range(args.steps):
                action=np.r_[raw._right_tcp_pose,raw._right_hand].astype(np.float32)
                obs,reward,terminated,truncated,info=env.step(action)
                poses={name:np.asarray(pose).tolist() for name,pose in raw._poses.items() if name in ('spray','spray_bottle','gen7_shoe')}
                frames.append({'step':step,'poses':poses,'trigger':raw._physics_dof_position('joint_0'),
                               'terminated':bool(terminated),'truncated':bool(truncated),'info':info})
                if terminated or truncated:raise RuntimeError('GEN7 terminated during untouched initialization probe')
            actual_geometry_capture(raw,args.output,label+'_settled')
            emit(args.output/f'{label}_settled_colliders.json',stage_inventory(raw._stage))
            emit(args.output/f'{label}_frames.json',frames)
            trace=raw._physx_ground_truth_trace
            np.savez_compressed(args.output/f'{label}_physical_snapshot.npz',**trace.snapshot())
            if args.preview:
                preview_images(raw,args.output,label)
            runs.append({'layout':label,'seed':seed,'steps':args.steps,'reset_parameters':raw._gen7_reset_parameters.tolist()})
        emit(args.output/'contacts.json',contacts)
        raw._stage.Export(str(args.output/'composed_scene.usdc'))
        emit(args.output/'recording_schema.json',raw.get_recording_schema())
        emit(args.output/'probe_result.json',{'factory_load_reset_step_pass':True,'runs':runs,
             'warnings':raw._isaac_warnings,'full_acceptance_pass':False})
        print('[GEN7 probe] Finished load/reset/step probe; other acceptance gates remain pending.',flush=True)
    except BaseException:
        emit(args.output/'contacts.json',contacts)
        emit(args.output/'probe_error.json',{'traceback':traceback.format_exc()})
        if env is not None:
            emit(args.output/'error_physics.json',physics_inventory(env.unwrapped))
            actual_geometry_capture(env.unwrapped,args.output,'error')
            trace=getattr(env.unwrapped,'_physx_ground_truth_trace',None)
            if trace is not None:np.savez_compressed(args.output/'error_physics_snapshot.npz',**trace.snapshot())
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
