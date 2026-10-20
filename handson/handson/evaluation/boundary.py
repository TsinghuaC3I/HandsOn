"""Prime native task checkers at a measured reset boundary, before policy action 0.

This uses the current environment's task methods. It neither replays collection
states nor imports a historical runtime or changes a success threshold.
"""
import math
import numpy as np
from .contracts import require,write_new
from .readback import capture,plain
TASKS={'t15_open_drawers','t01_align_blocks_in_midair','t38_set_dual_knobs'}

def prepare(env,job,output):
 if job['task'] not in TASKS:return None
 parameters=job['origin_parameters'];dt=env._physics_dt
 minimum=math.ceil(parameters['initial_settle_seconds']/dt)
 window=math.ceil(parameters['stable_seconds_required']/dt)
 maximum=minimum+math.ceil(parameters['timeout_seconds']/dt)
 initial=capture(env);stable=0;history=[];previous=None
 for index in range(maximum):
  env._step_isaaclab()
  ready=plain(env.stable_recording_origin_task_readiness())
  poses={k:np.asarray(plain(v.get_transforms())).copy() for k,v in env._task_body_views.items()}
  velocities={k:np.asarray(plain(v.get_velocities())).copy() for k,v in env._task_body_views.items()}
  linear=max([float(np.linalg.norm(v[:,:3],axis=-1).max()) for v in velocities.values()]+[0.])
  angular=max([float(np.linalg.norm(v[:,3:],axis=-1).max()) for v in velocities.values()]+[0.])
  joint=max([float(np.abs(np.asarray(plain(v.get_dof_velocities()))).max()) for _,v,*_ in env._task_reset_articulation_states]+[0.])
  finite=all(np.isfinite(x).all() for x in [*poses.values(),*velocities.values()])
  quiet=linear<=parameters['linear_velocity_limit'] and angular<=parameters['angular_velocity_limit'] and joint<=parameters['joint_velocity_limit']
  # BC021's approved reset accepts a static pose despite residual contact velocity.
  static=False
  if previous is not None and parameters['stability_metric']=='velocity_or_static_pose':
   from scipy.spatial.transform import Rotation
   static=all(np.max(np.linalg.norm(poses[k][:,:3]-previous[k][:,:3],axis=-1))<=parameters['pose_translation_limit'] and
    np.max((Rotation.from_quat(poses[k][:,3:]).inv()*Rotation.from_quat(previous[k][:,3:])).magnitude())<=parameters['pose_rotation_limit'] for k in poses)
  stable=stable+1 if index+1>=minimum and ready['ready'] and finite and (quiet or static) else 0
  history.append(dict(step=index+1,ready=ready['ready'],linear=linear,angular=angular,joint=joint,static_pose=static,stable_steps=stable))
  previous=poses
  if stable>=window:break
 write_new(output/'TASK_BOUNDARY_SETTLE.json',dict(parameters=parameters,samples=history,initial=initial,final_readiness=ready,passed=stable>=window,scope='native task checker boundary; not collection certification'))
 require(stable>=window,'Task checker boundary did not become ready: '+str(ready))
 # The approved origins explicitly permit one velocity-only transaction after
 # settling. Keep poses and joint coordinates fixed; never clamp policy steps.
 require(parameters['post_settle_velocity_zeroing'] and not parameters['stabilization_velocity_clamp'],'Unsupported task boundary protocol')
 trace=env._physx_ground_truth_trace
 before=capture(env)
 if job['task']=='t01_align_blocks_in_midair':
  transaction=trace.write_zero_velocity_boundary(trace.snapshot(),trace.schema(),sync=True)
 else:
  # Drawer/knob scenes may have no standalone rigid-body rows. Use the
  # existing task-only transaction, which supports articulation-only scenes.
  transaction=trace.zero_task_velocities()
  trace._sync_after_restore()
  transaction['scope']='task velocities only; robot state untouched'
 after=capture(env)
 for name,row in before['objects'].items():require(row['pose_xyzw']==after['objects'][name]['pose_xyzw'],'Boundary changed an object pose')
 for name,row in before['articulations'].items():require(row['joint_pos']==after['articulations'][name]['joint_pos'],'Boundary changed task DOF positions')
 for name,row in before['robots'].items():require(row['joint_pos']==after['robots'][name]['joint_pos'],'Boundary changed robot joint positions')
 ready=plain(env.stable_recording_origin_task_readiness());require(ready['ready'],'Task truth unavailable at boundary')
 accepted=plain(env.accept_stable_recording_origin(dict(task_readiness_required=True,final_task_readiness=ready)))
 require(accepted and accepted.get('accepted') and accepted.get('physics_state_writes') is False,'Native task checker was not primed')
 result=dict(accepted=accepted,readiness=ready,velocity_only_transaction=transaction,measured=capture(env),continuous_readiness_seconds=stable*dt,environment_step=env.env_step)
 write_new(output/'TASK_BOUNDARY.json',result)
 return result
