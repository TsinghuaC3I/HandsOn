"""Observe GEN1 final PhysX shapes before integration and after reset settling.

Instrumentation only: normal config, native reset, unchanged dynamics and success.
Use GEN1_AUDIT_CASE_LIMIT=1 for inspecting the auditor; only all 32 cases count.
"""
from pathlib import Path
import json,os,traceback
import numpy as np
from native_smoke import serial

TASK=Path(__file__).resolve().parent
QUERY_READBACK=os.environ.get('GEN1_AUDIT_QUERY_READBACK','1')=='1'
REPORT=TASK/('evidence/native_physics_audit.json' if QUERY_READBACK else 'evidence/native_stability_audit.json')

class Observer:
 def __init__(self,raw):
  self.raw=raw;self.snapshots=[];self.pending='cold_start_before_first_physics';self.contacts=[];self.tensor_contacts=[]
  import omni.physx
  self.query=omni.physx.get_physx_scene_query_interface()
  self.subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)
  # Read contacts after each actual _step_isaaclab call; no event-order assumption.
  self.authored_bottle=self.usd_pose(raw._find_rigid_body_prim('gen1_soap_bottle'))
 def usd_pose(self,prim):
  from pxr import UsdGeom
  T=UsdGeom.XformCache().GetLocalToWorldTransform(prim);q=T.ExtractRotationQuat()
  return [*T.ExtractTranslation(),q.GetReal(),*q.GetImaginary()]
 def on_contacts(self,headers,data):
  from pxr import PhysicsSchemaTools
  for h in headers:
   values=[]
   for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
    values.append({'separation_m':float(d.separation),'position':list(d.position),'normal':list(d.normal),'impulse':float(d.impulse) if np.isscalar(d.impulse) else list(d.impulse)})
   self.contacts.append({'actor0':str(PhysicsSchemaTools.intToSdfPath(h.actor0)),'actor1':str(PhysicsSchemaTools.intToSdfPath(h.actor1)),'collider0':str(PhysicsSchemaTools.intToSdfPath(h.collider0)),'collider1':str(PhysicsSchemaTools.intToSdfPath(h.collider1)),'points':values})
 def on_step(self,dt):
  for role,view in getattr(self.raw,'_gen1_contact_views',{}).items():
   values=view.get_contact_data(float(dt))
   values=[v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v) for v in values]
   forces,points,normals,separations,counts,starts=values
   forces=forces.reshape(-1);points=points.reshape(-1,3);normals=normals.reshape(-1,3);separations=separations.reshape(-1)
   for index,(count,start) in enumerate(zip(counts.reshape(-1),starts.reshape(-1))):
    count=int(count);start=int(start)
    if count:self.tensor_contacts.append({'role':role,'filter_index':index,'normal_forces_n':forces[start:start+count].tolist(),'positions':points[start:start+count].tolist(),'normals':normals[start:start+count].tolist(),'separations_m':separations[start:start+count].tolist()})
  if False and self.pending is not None:
   label=self.pending;self.pending=None
   try:self.snapshots.append(self.snapshot(label,float(dt)))
   except Exception:self.snapshots.append({'label':label,'error':traceback.format_exc()})
 def snapshot(self,label,dt=0.):
  from pxr import Usd,UsdPhysics,UsdGeom,PhysicsSchemaTools,Sdf
  import omni.physx
  if QUERY_READBACK:
   sim_views=[getattr(self.raw,'_task_physics_sim_view',None),*[getattr(robot,'_physics_sim_view',None) for robot in self.raw._robots.values()]]
   for view in sim_views:
    if view is not None:view.flush();view.update_articulations_kinematic()
   omni.physx.get_physx_interface().update_transformations(False,True,True,False)
  stage=self.raw._stage
  prims=list(Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))
  def owner(prim):
   while prim and prim.IsValid():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):return str(prim.GetPath())
    prim=prim.GetParent()
   return ''
  paths={str(p.GetPath()):owner(p) or str(p.GetPath()) for p in prims if p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False}
  adjacent=set();filtered=set();fixed={}
  for p in prims:
   if p.HasAPI(UsdPhysics.FilteredPairsAPI):
    for target in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
     a,b=str(p.GetPath()),str(target)
     if '/HandsOnRobots/' in a and '/HandsOnRobots/' in b:filtered.add(tuple(sorted((a,b))))
  for p in prims:
   if p.IsActive() and p.IsA(UsdPhysics.Joint):
    joint=UsdPhysics.Joint(p);a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
    if a and b and '/HandsOnRobots/' in str(a[0]) and '/HandsOnRobots/' in str(b[0]):
     adjacent.add(tuple(sorted((str(a[0]),str(b[0])))))
     if p.IsA(UsdPhysics.FixedJoint):
      fixed.setdefault(str(a[0]),set()).add(str(b[0]));fixed.setdefault(str(b[0]),set()).add(str(a[0]))
  fixed_pairs=set()
  for root in fixed:
   seen={root};todo=[root]
   while todo:
    for item in fixed.get(todo.pop(),set())-seen:seen.add(item);todo.append(item)
   fixed_pairs.update(tuple(sorted((root,item))) for item in seen if item!=root)
  pairs={};counts={};errors=[]
  for path,body in (paths.items() if QUERY_READBACK else []):
   if stage.GetPrimAtPath(path).GetTypeName()=='Plane':continue # Every finite shape is still queried against this plane.
   enc=PhysicsSchemaTools.encodeSdfPath(Sdf.Path(path));hits=[]
   def receive(hit):
    hits.append((str(hit.collision),str(hit.rigid_body)));return True
   try:counts[path]=int(self.query.overlap_shape(enc[0],enc[1],receive,False))
   except Exception as e:errors.append({'path':path,'error':str(e)});continue
   for other,other_body in hits:
    if other==path:continue
    other_body=paths.get(other) or other_body or other
    if body==other_body:continue # Compound shapes on the same rigid body form one union.
    key=tuple(sorted((path,other)))
    bodies=tuple(sorted((body,other_body)))
    bottle_table=any('gen1_soap_bottle' in p for p in bodies) and any(p.endswith('/table') for p in bodies)
    kind='table_support' if bottle_table else 'source_adjacent_robot_joint' if bodies in adjacent else 'source_explicit_filtered_joint_interface' if bodies in filtered else 'source_fixed_mount_assembly' if bodies in fixed_pairs else 'unapproved'
    pairs[key]={'colliders':key,'bodies':bodies,'classification':kind}
  bottle=self.raw._find_rigid_body_prim('gen1_soap_bottle')
  result={'label':label,'before_physics_callback_dt':dt,'collider_count':len(paths),'query_hit_count':sum(counts.values()),'query_errors':errors,'pairs':list(pairs.values()),'unapproved_count':sum(p['classification']=='unapproved' for p in pairs.values()),'explicit_robot_joint_interfaces':sorted(adjacent),'bottle_usd_pose_wxyz':self.usd_pose(bottle)}
  for path,view,indices,*_ in getattr(self.raw,'_task_reset_body_states',[]):
   if 'gen1_soap_bottle' in str(path):
    values=view.get_transforms()
    if hasattr(values,'detach'):values=values.detach().cpu().numpy()
    index=int(indices[0].item()) if hasattr(indices[0],'item') else int(indices[0])
    xyz_xyzw=np.asarray(values)[index]
    result['bottle_physics_pose_wxyz']=np.r_[xyz_xyzw[:3],xyz_xyzw[6],xyz_xyzw[3:6]]
  result['plane_query_coverage']='Finite shapes query against the ground plane; overlap_shape cannot use Plane as its source geometry.'
  result['task_body_states']={name:{'transforms':serial(v.get_transforms().detach().cpu().numpy().copy()),'velocities':serial(v.get_velocities().detach().cpu().numpy().copy())} for name,v in self.raw._task_body_views.items()}
  result['filtered_pairs']={str(p.GetPath()):[str(t) for t in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets()] for p in prims if p.HasAPI(UsdPhysics.FilteredPairsAPI)}
  return result

def main():
 from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv as Env
 from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 import handson.sim.isaaclab.abc_collection_env as runtime_module
 original_make_cfg=runtime_module.make_simulation_cfg
 def make_audit_cfg(*args,**kwargs):
  cfg=original_make_cfg(*args,**kwargs)
  if not QUERY_READBACK:return cfg
  cfg.use_fabric=False # Diagnostic USD readback, with the same GPU dynamics and source shapes.
  cfg.enable_scene_query_support=True # Observation infrastructure only; no dynamics or geometry changes.
  return cfg
 runtime_module.make_simulation_cfg=make_audit_cfg
 original_configure=Env._configure_task_simulation_settings
 original_step=Env._step_isaaclab
 original_reset=Env._reset_scene_state
 original_sample=gen1.sample_reset_pose
 observers=[];environment=None
 from datetime import datetime,timezone
 report={'passed':False,'scope':'Actual composed collection shapes and GPU dynamics, diagnostic CPU readback for native scene queries','cases':[],'started_at_utc':datetime.now(timezone.utc).isoformat()}
 REPORT.write_text(json.dumps(report,indent=2)+'\n')
 def configured(raw):
  original_configure(raw)
  if raw.task_name==gen1.TASK_NAME:
   import carb,omni.physx
   if not QUERY_READBACK:
    raw._gen1_audit_observer=Observer(raw);observers.append(raw._gen1_audit_observer);return
   raw._isaac_device='cpu' # Native tensors follow enabled CPU readback; GPU dynamics remain on.
   carb.settings.get_settings().set_bool('/physics/suppressReadback',False)
   carb.settings.get_settings().set_bool('/physics/updateToUsd',True)
   raw._gen1_audit_observer=Observer(raw);observers.append(raw._gen1_audit_observer)
   omni.physx.get_physx_interface().force_load_physics_from_usd()
   raw._gen1_audit_observer.snapshots.append(serial(raw._gen1_audit_observer.snapshot('cold_start_authored_before_integration')))
   raw._gen1_audit_observer.pending=None
   report['initial_snapshots']=serial(raw._gen1_audit_observer.snapshots)
   REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n')
 def reset(raw):
  original_reset(raw)
  if hasattr(raw,'_gen1_audit_observer'):
   raw._gen1_audit_observer.pending=getattr(raw,'_gen1_audit_label','constructor_reset')+'_before_physics'
 def step(raw,*args,**kwargs):
  observer=getattr(raw,'_gen1_audit_observer',None)
  if observer is not None and observer.pending is not None:
   label=observer.pending;observer.pending=None
   observer.snapshots.append(serial(observer.snapshot(label)))
  result=original_step(raw,*args,**kwargs)
  if observer is not None:observer.on_step(raw._sim_dt)
  return result
 Env._configure_task_simulation_settings=configured;Env._reset_scene_state=reset;Env._step_isaaclab=step
 try:
  environment=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=True,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=False,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  raw=environment.unwrapped;observer=observers[0]
  cases=[(f'boundary_{i}',i,xy,yaw) for i,(xy,yaw) in enumerate((([x,y],yaw) for x in gen1.XY_BOUNDS[:,0] for y in gen1.XY_BOUNDS[:,1] for yaw in (-30.,0.,30.)))]
  cases += [(f'seed_{i}',i,None,None) for i in range(20)]
  limit=int(os.environ.get('GEN1_AUDIT_CASE_LIMIT','32'))
  offset=int(os.environ.get('GEN1_AUDIT_CASE_OFFSET','0'))
  settling_steps=int(os.environ.get('GEN1_AUDIT_SETTLING_CONTROL_STEPS','120'))
  report['case_offset']=offset;report['settling_control_steps']=settling_steps
  for label,seed,xy,yaw in cases[offset:offset+limit]:
   raw._gen1_audit_label=label
   if xy is None:gen1.sample_reset_pose=original_sample
   else:
    gen1.sample_reset_pose=lambda rng,shape,table_z,randomize,xy=xy,yaw=yaw:gen1.pose_at_support(shape,np.asarray(xy),yaw,table_z)
   observer.contacts.clear();observer.tensor_contacts.clear()
   observation,info=environment.reset(seed=seed)
   before=raw._gen1_sampled_reset_pose.copy()
   action=np.r_[raw._right_tcp_pose,raw._right_hand].astype(np.float32)
   reset_return_pose=raw._pose_for_role('target').copy()
   trajectory=[];body_trajectory=[];substep_states=[]
   for step in range(settling_steps):
    observation,reward,terminated,truncated,info=environment.step(action)
    trajectory.append(raw._pose_for_role('target').copy())
    substep_states.extend(np.asarray(observation['physx_substep_task_rigid_body_state']).copy())
    if step<20 or step%10==0:
     body_trajectory.append({'step':step,'states':serial(observation['physx_task_rigid_body_state'])})
    if terminated or truncated:raise RuntimeError(f'{label}: unexpected terminal while resting')
   tail=np.asarray(substep_states[-61:],dtype=np.float64)
   positional_delta=np.linalg.norm(np.diff(tail[:,:,:3],axis=0),axis=-1)
   q=tail[:,:,3:7];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
   dots=np.abs(np.sum(q[1:]*q[:-1],axis=-1));angular_delta=2*np.arccos(np.clip(dots,-1,1))
   full=np.asarray(substep_states,dtype=np.float64)
   full_q=full[:,:,3:7];full_q=full_q/np.linalg.norm(full_q,axis=-1,keepdims=True)
   delta_p=np.linalg.norm(np.diff(full[:,:,:3],axis=0),axis=-1).max(axis=1)
   delta_r=(2*np.arccos(np.clip(np.abs(np.sum(full_q[1:]*full_q[:-1],axis=-1)),-1,1))).max(axis=1)
   windows=[{'first_substep':i,'last_substep':i+60,'max_translation_m':float(delta_p[i:i+60].max()),'max_rotation_rad':float(delta_r[i:i+60].max())} for i in range(120,len(delta_p)-59) if np.all(delta_p[i:i+60]<=1e-6) and np.all(delta_r[i:i+60]<=1e-5)]
   snapshot=observer.snapshot(label+'_settled')
   states=observation['physx_task_rigid_body_state'];velocity=np.asarray(states)[-1,7:]
   case={'label':label,'seed':seed,'sampled_reset_pose_wxyz':before,'reset_return_pose_wxyz':reset_return_pose,'body_trajectory':body_trajectory,'last_60_substep_states':tail,'a06_static_pose_windows':windows[:1],'max_last_60_substep_translation_m':float(positional_delta.max()),'max_last_60_substep_rotation_rad':float(angular_delta.max()),'joint_velocity_max_rad_s':float(np.max(np.abs(observation['physx_robot_joint_velocity']))),'joint_target_error_max_rad':float(np.max(np.abs(observation['physx_robot_joint_position']-observation['physx_robot_joint_target']))),'settled_pose_wxyz':trajectory[-1],'settled_snapshot':snapshot,'last_20_position_span_m':np.ptp(np.asarray(trajectory)[-20:,:3],axis=0),'metrics':gen1.current_metrics(raw),'physical_body_states':states,'contacts':observer.contacts,'tensor_contact_depths':observer.tensor_contacts,'success_steps_required':raw._success_steps_required()}
   report['cases'].append(serial(case));observer.contacts=[];observer.tensor_contacts=[]
   body_names=[entry.name for entry in raw._physx_ground_truth_trace.rigid_bodies]
   bottle_index=next(i for i,name in enumerate(body_names) if name.rsplit('/',1)[-1]==gen1.BODY_NAME)
   report['rigid_body_names']=body_names
   bottle_poses=full[:,bottle_index,:7]
   initial_q=np.asarray(before[3:7]);initial_q=initial_q/np.linalg.norm(initial_q)
   q_bottle=bottle_poses[:,3:7]/np.linalg.norm(bottle_poses[:,3:7],axis=1,keepdims=True)
   report['cases'][-1]['max_reset_rotation_drift_deg']=float(np.degrees(2*np.arccos(np.clip(np.abs(q_bottle@initial_q),-1,1))).max())
   report['cases'][-1]['max_reset_translation_drift_m']=float(np.linalg.norm(bottle_poses[:,:3]-before[:3],axis=1).max())
   report['initial_snapshots']=serial(observer.snapshots)
   REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n')
   print('GEN1_CASE',label,'unapproved',snapshot['unapproved_count'],'position_span',case['last_20_position_span_m'],flush=True)
  snapshots=[*observer.snapshots,*[x['settled_snapshot'] for x in report['cases']]]
  report['query_readback']=QUERY_READBACK
  report['passed']=QUERY_READBACK and len(report['cases'])==32 and all(not x.get('error') and not x.get('query_errors') and x.get('unapproved_count',1)==0 and x.get('query_hit_count',0)>0 for x in snapshots)
  if not QUERY_READBACK:
   report['scope']='Ordinary GPU tensor collection path: support, drift, velocities, contact depths; no scene-query claim'
   report['passed']=len(report['cases'])==32 and all(c['metrics']['gen1_on_table'] and c['metrics']['gen1_hand_released'] and np.linalg.norm(np.asarray(c['settled_pose_wxyz'])[:2]-np.asarray(c['sampled_reset_pose_wxyz'])[:2])<0.01 and c['joint_velocity_max_rad_s']<0.2 and c['joint_target_error_max_rad']<0.1 for c in report['cases'])
  report['initial_authored_bottle']=observer.authored_bottle
  report['timing']={'physics_dt':raw._sim_dt,'physics_substeps':raw._n_substeps,'nominal_control_dt':raw._control_dt,'hold_step_seconds':gen1.simulated_step_seconds(raw)}
 except BaseException:
  report['error']=traceback.format_exc();raise
 finally:
  gen1.sample_reset_pose=original_sample;Env._configure_task_simulation_settings=original_configure;Env._reset_scene_state=original_reset;Env._step_isaaclab=original_step;runtime_module.make_simulation_cfg=original_make_cfg
  if observers:report['initial_snapshots']=serial(observers[0].snapshots)
  REPORT.write_text(json.dumps(serial(report),indent=2,default=str)+'\n')
  print('GEN1_PHYSICS_REPORT',REPORT,flush=True)
  if environment is not None:
   environment.unwrapped._fast_shutdown_on_close=True
   environment.close()

if __name__=='__main__':main()
