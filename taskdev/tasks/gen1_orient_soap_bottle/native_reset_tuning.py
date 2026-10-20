"""Compare authored support clearances with unchanged dynamics and origin limits.

Diagnostic reset fixtures only. No trajectory state is overwritten after reset.
"""
from pathlib import Path
from dataclasses import replace
import json, traceback
import numpy as np
from scipy.spatial.transform import Rotation
from native_smoke import serial

TASK=Path(__file__).resolve().parent

def main():
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
 original_sample=gen1.sample_reset_pose
 original_clearance=gen1.SUPPORT_CLEARANCE_M
 environment=None;report={'scope':'Reset fixture comparison only; no acceptance threshold changes','cases':[]}
 try:
  environment=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=True,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=False,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  raw=environment.unwrapped
  layouts=[('center',[-.3,-.06],0),('low_low',[-.35,-.12],-30),('low_high',[-.35,0],30),('high_low',[-.25,-.12],30),('high_high',[-.25,0],-30)]
  for roll_deg,clearance in [(0,.00005),(0,.0001),(-3.6,.0001),(-3.6,.0005)]:
   metadata=json.loads((gen1.ASSET_DIR/'geometry.json').read_text())
   source_q=metadata.get('source_facet_quat_wxyz',gen1.geometry().side_rest_quat_wxyz)
   q=(Rotation.from_quat(source_q,scalar_first=True)*Rotation.from_euler('z',roll_deg,degrees=True)).as_quat(scalar_first=True)
   shape=replace(gen1.geometry(),side_rest_quat_wxyz=q)
   gen1.SUPPORT_CLEARANCE_M=clearance
   for i,(label,xy,yaw) in enumerate(layouts):
    gen1.sample_reset_pose=lambda rng,_shape,table_z,randomize:gen1.pose_at_support(shape,np.asarray(xy),yaw,table_z)
    environment.reset(seed=i)
    initial=raw._gen1_sampled_reset_pose.copy();poses=[]
    for _ in range(360):
     raw._step_isaaclab();raw._refresh_object_poses()
     poses.append(raw._pose_for_role('target').copy())
    poses=np.asarray(poses)
    rotations=Rotation.from_quat(initial[3:],scalar_first=True).inv()*Rotation.from_quat(poses[:,3:],scalar_first=True)
    entry={'roll_deg':roll_deg,'clearance_m':clearance,'layout':label,'initial_pose_wxyz':initial,'settled_pose_wxyz':poses[-1],'max_rotation_drift_deg':float(np.degrees(rotations.magnitude()).max()),'max_translation_drift_m':float(np.linalg.norm(poses[:,:3]-initial[:3],axis=1).max()),'metrics':gen1.current_metrics(raw)}
    report['cases'].append(serial(entry))
    (TASK/'evidence/native_reset_tuning.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GEN1_RESET_TUNING',roll_deg,clearance,label,entry['max_rotation_drift_deg'],flush=True)
 except BaseException:report['error']=traceback.format_exc();raise
 finally:
  gen1.sample_reset_pose=original_sample;gen1.SUPPORT_CLEARANCE_M=original_clearance
  (TASK/'evidence/native_reset_tuning.json').write_text(json.dumps(report,indent=2)+'\n')
  if environment is not None:environment.unwrapped._fast_shutdown_on_close=True;environment.close()

if __name__=='__main__':main()
