"""Native GEN1 predicate fixtures; these are not manipulation demonstrations.

Body writes establish independent test fixtures. Ordinary control and gravity
then run unchanged. Robot manipulation/contact evidence is a separate gate.
"""
from pathlib import Path
import json,traceback
import numpy as np
from native_smoke import serial

TASK=Path(__file__).resolve().parent

def main():
 from handson.tasks.eval_gen03_orient_bottle_upright.config import TaskConfig
 from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1
 report={'passed':False,'scope':'Native predicate fixtures, not robot manipulation or saved demonstrations','cases':[]};env=None
 try:
  env=TaskConfig().get_environment(policy_mode=True,render_mode='rgb_array',randomize=True,sim_backend='isaacsim',physics_backend='physx',launch_isaac=True,headless=True,image_obs=False,seed=7,isaac_device='cuda:0',arm_type='rokae',hand_type='wuji1')
  obs,info=env.reset(seed=7);raw=env.unwrapped
  action=np.r_[raw._right_tcp_pose,raw._right_hand].astype(np.float32)
  shape=gen1.geometry();table_z=raw._current_table_top_z()
  upright=np.r_[gen1.RESET_CENTER_XY,table_z-shape.bottom_vertices[:,2].min()+.0005,1.,0.,0.,0.]
  def fixture(pose):
   assert raw._set_task_body_world_pose(raw._task_body_views[gen1.BODY_NAME],pose)
   raw._refresh_object_poses()
  def run(label,count):
   rows=[]
   for i in range(count):
    obs,reward,terminated,truncated,info=env.step(action)
    rows.append({'index':i,'metrics':gen1.current_metrics(raw),'counter':raw._success_counter,'succeed':bool(info.get('succeed')),'terminated':bool(terminated),'truncated':bool(truncated),'sim_time':serial(obs['physx_sim_time'])})
    if terminated or truncated:break
   report['cases'].append({'label':label,'rows':rows});return rows
  rows=run('side_rest_false',10)
  assert not any(r['succeed'] or r['terminated'] or r['counter'] for r in rows)
  suspended=upright.copy();suspended[2]+=.15;fixture(suspended)
  rows=run('upright_in_air_false',5)
  assert all(not r['metrics']['gen1_on_table'] and not r['succeed'] for r in rows)
  fixture(upright);rows=run('upright_partial_hold',15)
  assert 0<rows[-1]['counter']<raw._success_steps_required() and not rows[-1]['succeed']
  fixture(gen1.pose_at_support(shape,gen1.RESET_CENTER_XY,0,table_z))
  rows=run('tilt_interrupts_hold_without_terminal',3)
  assert all(r['counter']==0 and not r['terminated'] and not r['truncated'] for r in rows)
  fixture(upright);rows=run('upright_recovery_full_hold',45)
  assert rows[-1]['succeed'] and rows[-1]['terminated'] and not rows[-1]['truncated']
  successful_rows=[r for r in rows if r['metrics']['gen1_success_now']]
  assert len(successful_rows)==raw._success_steps_required()==30
  assert all(not r['succeed'] for r in rows[:-1])
  assert len(successful_rows)*gen1.simulated_step_seconds(raw)>=.5
  obs,info=env.reset(seed=7)
  assert raw._success_counter==0 and not info.get('succeed')
  report.update(passed=True,hold_steps=30,hold_simulated_seconds=.5,manual_reset_counter=raw._success_counter,held_by_hand_case='Pending real robot contact sweep')
 except BaseException:
  report['error']=traceback.format_exc();raise
 finally:
  path=TASK/'evidence/native_semantics.json';path.write_text(json.dumps(serial(report),indent=2)+'\n');print('GEN1_SEMANTICS_REPORT',path,flush=True)
  if env is not None:env.unwrapped._fast_shutdown_on_close=True;env.close()
if __name__=='__main__':main()
