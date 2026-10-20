"""GEN1 ordinary collection environment smoke, using only the assigned GPU."""
from pathlib import Path
import json
import os
import traceback
import numpy as np

TASK=Path(__file__).resolve().parent
OUT=TASK/'evidence/native_smoke.json'

def serial(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [serial(v) for v in value]
 return value

def main():
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
 env=None;report={'task':gen1.TASK_NAME,'passed':False}
 try:
  env=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=True,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=False,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  obs,info=env.reset(seed=7);raw=env.unwrapped
  report['control_dt']=raw._control_dt;report['physics_dt']=raw._sim_dt
  report['physics_substeps']=raw._n_substeps
  report['simulated_seconds_per_control_step']=gen1.simulated_step_seconds(raw)
  report['success_steps_required']=raw._success_steps_required()
  report['deferred_camera_render']=raw._deferred_camera_render
  report['observation_shapes']={k:list(np.asarray(v).shape) for k,v in obs.items()}
  report['reset_info']=info
  report['initial_pose_wxyz']=raw._pose_for_role('target')
  report['reset_pose_wxyz']=raw._gen1_sampled_reset_pose
  from pxr import Usd,UsdGeom,UsdPhysics
  bottle=raw._find_rigid_body_prim(gen1.BODY_NAME)
  colliders=[p for p in Usd.PrimRange(bottle) if p.HasAPI(UsdPhysics.CollisionAPI)]
  report['bottle_path']=str(bottle.GetPath());report['collider_count']=len(colliders)
  report['collider_visibility']={str(p.GetPath()):str(UsdGeom.Imageable(p).ComputeVisibility()) for p in colliders}
  assert colliders and all(v=='invisible' for v in report['collider_visibility'].values())
  assert abs(raw._sim_dt-1/120)<1e-10
  assert raw._deferred_camera_render
  action=np.r_[raw._right_tcp_pose,raw._right_hand].astype(np.float32)
  rows=[]
  for i in range(150):
   obs,reward,terminated,truncated,info=env.step(action)
   if i%10==0:rows.append({'step':i,'pose':raw._pose_for_role('target'),'metrics':gen1.current_metrics(raw),'terminated':terminated,'truncated':truncated})
   if terminated or truncated:raise RuntimeError('Unexpected termination during side-rest hold')
  report['samples']=rows
  report['final_metrics']=gen1.current_metrics(raw)
  assert report['final_metrics']['gen1_contacts_available']
  assert not report['final_metrics']['gen1_success_now']
  report['passed']=True
 except BaseException:
  report['error']=traceback.format_exc();raise
 finally:
  OUT.write_text(json.dumps(serial(report),indent=2,default=str)+'\n')
  print('GEN1_REPORT',OUT,flush=True)
  if env is not None:
   env.unwrapped._fast_shutdown_on_close=True  # Existing ordinary recorder shutdown contract.
   env.close()

if __name__=='__main__':main()
