"""Five-layout public-action reach/touch probe; not operator or grasp completion.

The bottle is written only by the ordinary reset. Robot motion uses the public
27-value action interface. An explicit preview renders three cameras at contact.
"""
from pathlib import Path
import json,os,traceback
import numpy as np
from native_smoke import serial

TASK=Path(__file__).resolve().parent

def main():
 os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER']='0'
 os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY']='0'
 os.environ['HANDSON_STABLE_RECORDING_ORIGIN']='1'
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
 from handson.sim.isaaclab.recording_origin import stabilize_recording_origin,stable_recording_origin_kwargs_for_task
 from PIL import Image
 from scipy.spatial.transform import Rotation,Slerp
 original_sample=gen1.sample_reset_pose
 env=None;report={'passed':False,'scope':'Scripted five-layout physical reach and one-second intended touch; full grasp/reorientation and operator assessment remain separate','cases':[]}
 try:
  env=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=True,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=True,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  raw=env.unwrapped
  layouts=[('center',[-.3,-.06],0),('low_low',[-.35,-.12],-30),('low_high',[-.35,0],30),('high_low',[-.25,-.12],30),('high_high',[-.25,0],-30)]
  for seed,(label,xy,yaw) in enumerate(layouts[:int(os.environ.get('GEN1_CONTACT_CASE_LIMIT','5'))]):
   gen1.sample_reset_pose=lambda rng,shape,table_z,randomize:gen1.pose_at_support(shape,np.asarray(xy),yaw,table_z)
   obs,info=env.reset(seed=seed)
   obs,info=stabilize_recording_origin(env,obs,info,**stable_recording_origin_kwargs_for_task(gen1.TASK_NAME))
   origin=info['physical_recording_origin']
   robot=raw._control_robot('right')
   palm=raw._robot_body_pose_matrix(robot,raw._ee_body_indices['right'])
   rest_rotation=Rotation.from_matrix(palm[:3,:3])
   grasp_rotation=Rotation.from_euler('z',yaw,degrees=True)*rest_rotation
   pose=raw._pose_for_role('target').copy()
   center=gen1.rotation_matrix(pose[3:])@np.array([0.,0.,-.035])+pose[:3]
   # Present the barrel to proximal fingers, preserving the reachable source
   # palm pitch. A distal-tip descent requires an unsuitable low wrist pose.
   grasp_position=center-grasp_rotation.apply([.045,0.,.11])
   above_position=grasp_position+np.array([0.,0.,.065])
   rest_position=palm[:3,3].copy()
   rotation_path=Slerp([0.,1.],Rotation.concatenate([rest_rotation,grasp_rotation]))
   start=np.r_[rest_position,rest_rotation.as_quat(scalar_first=True)]
   hand=raw._right_hand.copy();rows=[];unexpected=[]
   def step(command,stage):
    obs,reward,terminated,truncated,info=env.step(np.r_[command,hand].astype(np.float32))
    metrics=gen1.current_metrics(raw)
    physical_palm=raw._robot_body_pose_matrix(robot,raw._ee_body_indices['right'])
    contact_depths={}
    for role,view in raw._gen1_contact_views.items():
     data=view.get_contact_data(raw._sim_dt)
     separation=np.asarray(data[3].detach().cpu()).reshape(-1)
     counts=np.asarray(data[4].detach().cpu()).reshape(-1)
     starts=np.asarray(data[5].detach().cpu()).reshape(-1)
     active=[float(v) for count,start in zip(counts,starts) for v in separation[int(start):int(start)+int(count)]]
     contact_depths[role]=min(active) if active else None
    rows.append({'stage':stage,'command':command.copy(),'hand_targets':hand.copy(),'physical_palm_matrix':physical_palm,'pose':raw._pose_for_role('target').copy(),'metrics':metrics,'minimum_contact_separation_m':contact_depths,'terminated':bool(terminated),'truncated':bool(truncated),'info':info})
    if terminated or truncated:unexpected.append(len(rows)-1)
    return metrics
   for a in np.linspace(0,1,90):
    goal=np.r_[(1-a)*rest_position+a*above_position,rotation_path(a).as_quat(scalar_first=True)]
    step(goal,'approach')
   for _ in range(45):step(goal,'approach_hold')
   for a in np.linspace(0,1,90):
    goal=np.r_[(1-a)*above_position+a*grasp_position,grasp_rotation.as_quat(scalar_first=True)]
    step(goal,'descend')
   open_hand=hand.copy();closed_hand=.50*raw._tool_power_grip_pose(open_hand)
   for a in np.linspace(0,1,75):
    hand=(1-a)*open_hand+a*closed_hand;step(goal,'close_fingers')
   contact_forces=[]
   for _ in range(75):contact_forces.append(step(goal,'touch_hold')['gen1_hand_force_n'])
   touched=bool(max(contact_forces,default=0)>.02)
   frame_paths={}
   for _ in range(3):frames=raw.render()
   for name,frame in zip(('main','wrist_left','wrist_right'),frames,strict=True):
    p=TASK/f'evidence/contact_{label}_{name}.png';Image.fromarray(np.asarray(frame,dtype=np.uint8)).save(p);frame_paths[name]=str(p)
   for a in np.linspace(0,1,75):
    hand=(1-a)*closed_hand+a*open_hand;step(goal,'open_fingers')
   released_position=goal[:3]+np.array([0.,0.,.10])
   for a in np.linspace(0,1,90):
    retreat=goal.copy();retreat[:3]=(1-a)*goal[:3]+a*released_position;step(retreat,'retreat')
   for _ in range(30):release_metrics=step(retreat,'released_hold')
   sustained=0;longest=0
   for force in contact_forces:
    sustained=sustained+1 if force>gen1.CONTACT_FORCE_EPS_N else 0;longest=max(longest,sustained)
   case={'layout':label,'seed':seed,'initial_physical_palm_wxyz':start,'physical_recording_origin':origin,'touched':touched,'longest_contact_seconds':longest*gen1.simulated_step_seconds(raw),'release_metrics':release_metrics,'unexpected_terminals':unexpected,'camera_paths':frame_paths,'rows':rows}
   case['passed']=bool(touched and case['longest_contact_seconds']>=1 and release_metrics['gen1_hand_released'] and not unexpected)
   report['cases'].append(serial(case));(TASK/'evidence/native_contact_probe.json').write_text(json.dumps(report,indent=2)+'\n')
   print('GEN1_CONTACT_PROBE',label,case['passed'],case['longest_contact_seconds'],flush=True)
  report['passed']=len(report['cases'])==5 and all(c['passed'] for c in report['cases'])
 except BaseException:report['error']=traceback.format_exc();raise
 finally:
  gen1.sample_reset_pose=original_sample
  (TASK/'evidence/native_contact_probe.json').write_text(json.dumps(serial(report),indent=2)+'\n')
  if env is not None:env.unwrapped._fast_shutdown_on_close=True;env.close()

if __name__=='__main__':main()
