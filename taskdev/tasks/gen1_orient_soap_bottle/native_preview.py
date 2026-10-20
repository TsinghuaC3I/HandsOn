"""Render the actual native GEN1 scene; this is an explicit preview process."""
from pathlib import Path
import os,json,traceback
import numpy as np
from native_smoke import serial
TASK=Path(__file__).resolve().parent

def main():
 os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER']='0'
 os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY']='0'
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 from PIL import Image
 report={'passed':False,'scope':'Native three-camera preview; formal collection remains deferred'};env=None
 try:
  env=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=False,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=True,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  obs,info=env.reset(seed=7);raw=env.unwrapped
  action=np.r_[raw._right_tcp_pose,raw._right_hand].astype(np.float32)
  for _ in range(120):raw._step_isaaclab()
  raw._refresh_object_poses()
  # Keep the uncommanded arm at its physical home while refreshing telemetry.
  obs,_,_,_,_=env.step(np.r_[np.zeros(7),raw._right_hand].astype(np.float32))
  from handson.data.recording_fields import resolve_environment_recording_schema,declared_recording_field_names,select_next_observation_for_recording
  schema=resolve_environment_recording_schema(env)
  required=declared_recording_field_names(schema)
  selected=select_next_observation_for_recording(obs,required_fields=required)
  assert {'gen1_task_state','gen1_reset_pose_wxyz'}<=set(selected)
  report['recording_fields']={name:schema['fields'][name] for name in ('gen1_task_state','gen1_reset_pose_wxyz')}
  report['next_recording_field_shapes']={name:list(np.asarray(selected[name]).shape) for name in report['recording_fields']}
  for _ in range(2):frames=raw.render()
  frames=raw.render();report['cameras']={}
  for name,frame in zip(('main','wrist_left','wrist_right'),frames,strict=True):
   frame=np.asarray(frame);assert frame.ndim==3 and frame.shape[-1]==3 and frame.std()>5
   path=TASK/f'evidence/native_preview_{name}.png';Image.fromarray(frame.astype(np.uint8)).save(path)
   report['cameras'][name]={'path':str(path),'shape':list(frame.shape),'std':float(frame.std())}
  report['bottle_pose_wxyz']=raw._pose_for_role('target');report['passed']=True
 except BaseException:report['error']=traceback.format_exc();raise
 finally:
  path=TASK/'evidence/native_preview.json';path.write_text(json.dumps(serial(report),indent=2)+'\n');print('GEN1_PREVIEW_REPORT',path,flush=True)
  if env is not None:env.unwrapped._fast_shutdown_on_close=True;env.close()
if __name__=='__main__':main()
