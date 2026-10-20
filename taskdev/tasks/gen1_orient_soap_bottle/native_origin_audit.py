"""Apply the unmodified ordinary recording-origin gate to all 32 GEN1 layouts."""
from pathlib import Path
import json,os,traceback
import numpy as np
from native_smoke import serial

TASK=Path(__file__).resolve().parent

def main():
 os.environ['HANDSON_STABLE_RECORDING_ORIGIN']='1'
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
 from handson.sim.isaaclab.recording_origin import stabilize_recording_origin,stable_recording_origin_kwargs_for_task
 original_sample=gen1.sample_reset_pose
 env=None;report={'passed':False,'scope':'Every layout uses the ordinary full-frame physical recording-origin gate; no tolerance overrides','cases':[]}
 try:
  env=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=True,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=False,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  raw=env.unwrapped
  cases=[(f'boundary_{i}',i,xy,yaw) for i,(xy,yaw) in enumerate((([x,y],yaw) for x in gen1.XY_BOUNDS[:,0] for y in gen1.XY_BOUNDS[:,1] for yaw in (-30.,0.,30.)))]
  cases += [(f'seed_{i}',i,None,None) for i in range(20)]
  for label,seed,xy,yaw in cases:
   if xy is None:gen1.sample_reset_pose=original_sample
   else:gen1.sample_reset_pose=lambda rng,shape,table_z,randomize,xy=xy,yaw=yaw:gen1.pose_at_support(shape,np.asarray(xy),yaw,table_z)
   obs,info=env.reset(seed=seed)
   case={'label':label,'seed':seed,'sampled_reset_pose_wxyz':raw._gen1_sampled_reset_pose.copy(),'passed':False}
   try:
    obs,info=stabilize_recording_origin(env,obs,info,**stable_recording_origin_kwargs_for_task(gen1.TASK_NAME))
    case['physical_recording_origin']=info['physical_recording_origin'];case['passed']=True
   except Exception as exc:
    case['error']=traceback.format_exc();case['diagnostics']=getattr(exc,'diagnostics',None)
   report['cases'].append(serial(case));(TASK/'evidence/native_origin_audit.json').write_text(json.dumps(report,indent=2)+'\n')
   print('GEN1_ORIGIN_CASE',label,case['passed'],flush=True)
  report['passed']=len(report['cases'])==32 and all(c['passed'] for c in report['cases'])
 except BaseException:report['error']=traceback.format_exc();raise
 finally:
  gen1.sample_reset_pose=original_sample
  (TASK/'evidence/native_origin_audit.json').write_text(json.dumps(serial(report),indent=2)+'\n')
  if env is not None:env.unwrapped._fast_shutdown_on_close=True;env.close()

if __name__=='__main__':main()
