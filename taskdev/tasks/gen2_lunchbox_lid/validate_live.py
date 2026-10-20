"""GEN2 GPU phase-3 audit through the ordinary ABC collection constructor.

Requires the user's GPU allocation policy to have been satisfied externally.
This script never selects an unapproved GPU, changes object poses during actions,
or substitutes for the five real manipulations / UI episode / human acceptance.
"""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import itertools
import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'handson'),str(ROOT)]
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--boundary',type=int,choices=range(8))
parser.add_argument('--frozen-dir',type=Path)
parser.add_argument('--suite',action='store_true')
parser.add_argument('--preview',action='store_true')
parser.add_argument('--motion',action='store_true')
parser.add_argument('--predicate-fixtures',action='store_true')
args=parser.parse_args()
args.out.parent.mkdir(parents=True,exist_ok=True)
runtime_dir=args.out.parent/'runtime'
runtime_dir.mkdir(parents=True,exist_ok=True)
# Keep Kit configuration, caches, logs and Python bytecode in the GEN2 checkout.
sys.argv.extend(['--portable-root',str(runtime_dir.resolve()),
                 '--/log/file='+str((args.out.parent/'kit.log').resolve()),
                 '--/app/settings/persistent=false'])
version=metadata.version('isaacsim')
assert version.startswith('5.1.'),version
from validation_runtime import validate_runtime
runtime_identity = validate_runtime()
assert os.environ.get('GEN2_GPU_INDEX'), 'Use the allocation-checked task launcher'
assert os.environ.get('TASKDEV_PHYSICAL_GPU') == os.environ['GEN2_GPU_INDEX']
assert os.environ.get('TASKDEV_LOGICAL_GPU') == '0', 'Private single-GPU namespace required'
assert os.environ['HANDSON_ISAACSIM_RENDERER_GPU'] == '0'
os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON']='0'
os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER']='0' if args.preview else '1'
os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY']='0'
os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH']='1'
result=dict(task='GEN2',canonical_name='eval_gen01_cover_lunchbox',isaacsim_version=version,
            seed=args.seed,boundary=args.boundary,phases=[],errors=[],passed=False)
result['runtime']={'python':sys.executable,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'gpu_namespace':str(args.out.parent/'gpu_namespace.json'),
                   'renderer_gpu':os.environ['HANDSON_ISAACSIM_RENDERER_GPU'],
                   'runtime_family':'isaac_sim_5_1','newton_enabled':False, **runtime_identity,
                   'host':os.environ['GEN2_HOST'], 'gpu_uuid':os.environ['GEN2_GPU_UUID']}
print('[GEN2] Isaac Sim version:',version,flush=True)


def save():
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


contact_stats={}
physics_step=0

def install_contacts(env):
    import omni.physx
    from pxr import PhysxSchema,Usd,UsdPhysics,PhysicsSchemaTools
    for prim in Usd.PrimRange(env._stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    def on_contacts(headers,data):
        for header in headers:
            pair=tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                               str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
            for i in range(int(header.contact_data_offset),int(header.contact_data_offset+header.num_contact_data)):
                point=data[i];sep=float(point.separation)
                row=contact_stats.setdefault(pair,dict(pair=list(pair),min_separation_m=sep,
                    max_impulse_ns=0.,physics_steps=[]))
                row['min_separation_m']=min(row['min_separation_m'],sep)
                row['max_impulse_ns']=max(row['max_impulse_ns'],float(np.linalg.norm(np.asarray(point.impulse))))
                if physics_step not in row['physics_steps']:row['physics_steps'].append(physics_step)
    env._gen2_audit_contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)


def snapshot(env,phase):
    import omni.physx
    from pxr import PhysicsSchemaTools,Sdf,Usd,UsdGeom,UsdPhysics
    stage=env._stage
    pred=Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prims=[p for p in Usd.PrimRange(stage.GetPseudoRoot(),pred) if p.IsActive()]
    colliders=[];owners={};inventory=[];joint_pairs=set();planes=[];fixed_parent={};source_excludes={}
    def fixed_root(path):
        while path in fixed_parent:path=fixed_parent[path]
        return path
    for side,robot in env._robots.items():
        source=Path(robot.cfg.spawn.asset_path)
        tree=ET.parse(source)
        prefix=f'/World/HandsOnRobots/{side.capitalize()}Robot/base_{side}/'
        for exclude in tree.findall('./contact/exclude'):
            pair=tuple(sorted((prefix+exclude.attrib['body1'],prefix+exclude.attrib['body2'])))
            source_excludes[pair]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    for prim in prims:
        path=str(prim.GetPath())
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is False:continue
            a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
            if a and b:
                joint_pairs.add(tuple(sorted((str(a[0]),str(b[0])))))
                if prim.IsA(UsdPhysics.FixedJoint):
                    ra,rb=fixed_root(str(a[0])),fixed_root(str(b[0]))
                    if ra!=rb:fixed_parent[rb]=ra
        if prim.IsA(UsdGeom.Gprim):
            imageable=UsdGeom.Imageable(prim)
            collision=prim.HasAPI(UsdPhysics.CollisionAPI)
            inventory.append(dict(path=path,type=prim.GetTypeName(),collision=collision,
                                  visibility=str(imageable.ComputeVisibility())))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        body=prim
        while body and body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        owners[path]=str(body.GetPath()) if body and body.IsValid() else path
        colliders.append(path)
        if prim.GetTypeName()=='Plane':planes.append(path)
    query=omni.physx.get_physx_scene_query_interface()
    hits=set();counts={}
    for path in colliders:
        if path in planes:continue  # Queried as targets of every finite shape.
        encoded=PhysicsSchemaTools.encodeSdfPath(Sdf.Path(path))
        def report(hit):
            other=str(hit.collision)
            if other!=path:hits.add(tuple(sorted((path,other))))
            return True
        counts[path]=int(query.overlap_shape(*encoded,report,False))
    # No broad robot/body allowlist: only source components of one rigid body
    # and links explicitly connected by an authored USD joint are expected.
    pairs=[];unapproved=[]
    for a,b in sorted(hits):
        aa,bb=owners.get(a,a),owners.get(b,b)
        reason=('same rigid-body components; separate visual audit required' if aa==bb else
                'authored adjacent joint links' if tuple(sorted((aa,bb))) in joint_pairs else
                'authored fixed-joint assembly' if fixed_root(aa)==fixed_root(bb) else
                'explicit source MJCF collision exclusion' if tuple(sorted((aa,bb))) in source_excludes else None)
        contact=contact_stats.get((a,b))
        table_contact=('/World/Task/table/' in aa and any('/World/Task/'+n+'/' in bb for n in ('gen2_lid','gen2_lunchbox'))) or ('/World/Task/table/' in bb and any('/World/Task/'+n+'/' in aa for n in ('gen2_lid','gen2_lunchbox')))
        if reason is None and table_contact and contact and contact['min_separation_m']>=-.001:
            reason='intended task/table support; measured penetration <= 1 mm contact shell'
        row=dict(pair=[a,b],bodies=[aa,bb],allow_reason=reason,
                 depth_m=max(0.,-contact['min_separation_m']) if contact else None,
                 source_exclusion=source_excludes.get(tuple(sorted((aa,bb)))))
        pairs.append(row)
        if reason is None:unapproved.append(row)
    evidence=dict(phase=phase,collider_count=len(colliders),query_counts=counts,
                  planes_queried_as_targets=planes,
                  inventory=inventory,overlap_pairs=pairs,unapproved=unapproved,contacts=list(contact_stats.values()),
                  note='Overlap queries provide pair identities; contact reports provide depth after stepping.')
    # A nonzero self-hit alone does not prove dynamic query transforms are current.
    if phase!='authored_before_first_simulation_reset_or_step':
        lid=env._physics_body_pose('gen2_lid')
        hit=query.raycast_closest(tuple(lid[:3]+np.array([0,0,.2])),(0.,0.,-1.),.3)
        evidence['lid_surface_ray']={k:(list(v) if k in ('position','normal') else v) for k,v in dict(hit).items()}
        if not hit.get('hit') or 'gen2_lid' not in str(hit.get('collision','')):
            raise RuntimeError('Dynamic query canary failed to hit the physical lid')
        error=abs(float(hit['position'][2])-(float(lid[2])+.008))
        evidence['lid_surface_ray_error_m']=error
        if error>.0001:raise RuntimeError(f'Dynamic query transform is stale by {error} m')
    result['phases'].append(evidence);save()
    print('[GEN2] shape audit',phase,'colliders',len(colliders),'unapproved',len(unapproved),flush=True)
    if not counts or max(counts.values())==0:
        raise RuntimeError('PhysX scene query canary failed: every finite shape returned zero hits')
    if not colliders or not any('gen2_lid' in p for p in colliders) or not any('finger' in p for p in colliders):
        raise RuntimeError('GEN2 shape query did not cover the complete scene')
    if unapproved:
        result['errors'].append(f'GEN2 {phase}: {len(unapproved)} unapproved overlap pairs');save()
        raise RuntimeError(result['errors'][-1])
    return evidence


def motion(env):
    from handson.tasks.eval_gen01_cover_lunchbox.geometry import BODY_NAMES
    bodies={}
    for name in BODY_NAMES:
        pose=env._physics_body_pose(name);vel=env._physics_body_velocity(name)
        if pose is None or vel is None:raise RuntimeError('GEN2 missing physical state: '+name)
        if not np.all(np.isfinite(np.r_[pose,vel])):raise RuntimeError('GEN2 nonfinite physical state: '+name)
        bodies[name]=dict(pose=np.asarray(pose).tolist(),velocity=np.asarray(vel).tolist())
    robots={}
    for side,robot in env._robots.items():
        def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
        velocity=array(robot.data.joint_vel)
        error=array(robot.data.joint_pos_target)-array(robot.data.joint_pos)
        robots[side]=dict(max_joint_speed=float(np.max(np.abs(velocity))),max_target_error=float(np.max(np.abs(error))))
    return dict(bodies=bodies,robots=robots)


if args.frozen_dir is not None:
    from frozen_audit import run
    app,audits=run(args.frozen_dir,args.out.parent)
    result.update(audits);save()
    app.close(wait_for_replicator=False,skip_cleanup=True)
    raise SystemExit(0 if result['frozen_geometry_queries_passed'] else 1)

env=None
try:
    from handson.sim.isaaclab import abc_collection_env as abc
    HandsOnABCIsaacLabEnv=abc.HandsOnABCIsaacLabEnv
    original_cfg=abc.make_simulation_cfg
    def audit_cfg(*a,**kw):
        cfg=original_cfg(*a,**kw)
        cfg.enable_scene_query_support=True
        return cfg
    abc.make_simulation_cfg=audit_cfg
    from handson.tasks.eval_gen01_cover_lunchbox import geometry as g
    original=HandsOnABCIsaacLabEnv._configure_task_simulation_settings
    def before_first_reset(self):
        original(self)
        self._fast_shutdown_on_close=True  # Existing recorder shutdown path.
        import omni.physx
        install_contacts(self)
        # Documented PhysX API loads/cooks shapes without simulating a step.
        omni.physx.get_physx_interface().force_load_physics_from_usd()
        snapshot(self,'authored_before_first_simulation_reset_or_step')
    HandsOnABCIsaacLabEnv._configure_task_simulation_settings=before_first_reset
    active_boundary=args.boundary
    sampler=HandsOnABCIsaacLabEnv._task_specific_reset_xy_positions
    pose_customizer=HandsOnABCIsaacLabEnv._customize_task_reset_pose
    def boundary_sampler(self):
        if active_boundary is None:return sampler(self)
        sx,sy,opposed=list(itertools.product((-1,1),repeat=3))[active_boundary]
        return {g.BOX:g.DEFAULT_XY[g.BOX]+np.array([sx,sy])*.025,
                g.LID:g.DEFAULT_XY[g.LID]+np.array([opposed*sx,-sy])*.025}
    def boundary_pose(self,name,pose_xyzw,**kwargs):
        pose=pose_customizer(self,name,pose_xyzw,**kwargs)
        if active_boundary is not None and name in g.BODY_NAMES:
            sx,sy,opposed=list(itertools.product((-1,1),repeat=3))[active_boundary]
            angle=np.deg2rad(5*(sx if name==g.BOX else sy*opposed))
            pose=np.array(pose,copy=True);pose[...,3:]=[0,0,np.sin(angle/2),np.cos(angle/2)]
        return pose
    HandsOnABCIsaacLabEnv._task_specific_reset_xy_positions=boundary_sampler
    HandsOnABCIsaacLabEnv._customize_task_reset_pose=boundary_pose
    env=HandsOnABCIsaacLabEnv(g.TASK_NAME,seed=args.seed,randomize=True,
        render_mode='rgb_array',image_obs=args.preview,launch_isaac=True,headless=True,
        arm_type='rokae',hand_type='wuji1',physics_dt=1/120,hz=30,isaac_device='cuda:0')
    result['post_constructor_motion']=motion(env)
    from collision_snapshot import capture
    result['frozen_snapshots']=[capture(env,args.out.parent/'snapshots','after_constructor_reset')]
    result['settle_frames']=[]
    for physics_step in range(1,121):
        env._step_isaaclab()
        if physics_step%10==0:result['settle_frames'].append(dict(step=physics_step,**motion(env)))
    result['settled_motion']=motion(env)
    result['frozen_snapshots'].append(capture(env,args.out.parent/'snapshots','after_one_second_settle'))
    # Record every measured value; these conservative stability observations do
    # not replace the approved semantic thresholds or the normal origin gate.
    for data in result['settled_motion']['bodies'].values():
        velocity=np.asarray(data['velocity'])
        assert np.linalg.norm(velocity[:3])<=.025 and np.linalg.norm(velocity[3:])<=.30
    if args.predicate_fixtures:
        from validate_predicate_fixtures import run as run_predicates
        result['predicate_fixtures']=run_predicates(env,args.out.parent/'predicate_fixtures.json')
        save()
    if args.motion:
        from robot_motion import run as run_motion
        result['robot_motion']=run_motion(env,args.out.parent/'robot_motion',motion)
        save()
    if args.preview:
        from PIL import Image
        from collision_snapshot import array
        from validation_runtime import render_preview_frames
        result['preview']={}
        frames=render_preview_frames(env)
        for name,frame in zip(('main','wrist_left','wrist_right'),frames):
            pixels=np.asarray(frame)
            Image.fromarray(pixels.astype(np.uint8)).save(args.out.parent/(name+'.png'))
            result['preview'][name]=dict(shape=list(pixels.shape),std=float(pixels.std()),fallback_rejected=True)
            assert pixels.std()>1
        result['robot_state']={side:dict(body_names=robot.body_names,
            body_pos=array(robot.data.body_pos_w).tolist(),body_quat=array(robot.data.body_quat_w).tolist(),
            joint_names=robot.joint_names,joint_pos=array(robot.data.joint_pos).tolist()) for side,robot in env._robots.items()}
        result['action_origin']=dict(right_tcp=env._right_tcp_pose.tolist(),right_hand=env._right_hand.tolist())
        save()
    if args.suite:
        result['layout_suite']=[]
        cases=[('seed_'+str(i),i,None) for i in range(20)]+[('boundary_'+str(i),100+i,i) for i in range(8)]+[('repeat_seed_0_'+str(i),0,None) for i in range(2)]
        for tag,seed,boundary in cases:
            active_boundary=boundary;contact_stats.clear();physics_step=0
            observation,info=env.reset(seed=seed)
            record=dict(tag=tag,seed=seed,boundary=boundary,post_reset=motion(env),snapshots=[])
            record['snapshots'].append(capture(env,args.out.parent/'snapshots',tag+'_reset'))
            record['frames']=[]
            for physics_step in range(1,121):
                env._step_isaaclab()
                if physics_step%10==0:record['frames'].append(dict(step=physics_step,**motion(env)))
            record['settled']=motion(env)
            for body in record['settled']['bodies'].values():
                velocity=np.asarray(body['velocity'])
                assert np.linalg.norm(velocity[:3])<=.025 and np.linalg.norm(velocity[3:])<=.30
            record['snapshots'].append(capture(env,args.out.parent/'snapshots',tag+'_settled'))
            result['layout_suite'].append(record);save()
            print('[GEN2] Captured layout',tag,flush=True)
    result['phase3_partial_pass']=False
    result['shape_and_motion_checks_completed']=True
    result['remaining']=['multi-seed aggregation','contact/depth review','source collider visual protrusion review',
                         'canonical cameras and viewport review','five robot manipulations','ordinary UI recording and replay','human teleoperation']
except Exception as exc:
    result['errors'].append(str(exc));result['traceback']=traceback.format_exc()
    traceback.print_exc()
finally:
    save()
    if env is not None:
        result['normal_work_completed_before_close']=True;save()
        env._fast_shutdown_on_close=True
        env.close()
if result['errors']:raise SystemExit(1)
