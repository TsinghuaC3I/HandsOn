"""Exercise GEN1 through the ordinary CollectionState start/stop path."""
from pathlib import Path
import json,os,sys,time,traceback

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
TRIAL=int(os.environ.get('GEN1_UI_TRIAL','1'))
OUT=TASK/'evidence'/f'platform_cold_start_{TRIAL}'
OUT.mkdir(parents=True,exist_ok=True)
for key,value in {
 'HANDSON_COLLECTION_TELEOP_STACK':'disabled',
 'HANDSON_COLLECTION_REQUIRE_TELEOP_READY':'0',
 'HANDSON_COLLECTION_CLEAN_STALE_ON_START':'0',
 'HANDSON_COLLECTION_RUNTIME_DIR':str(TASK/'evidence/platform_runtime'),
 'HANDSON_MODE_A_LOG_DIR':str(TASK/'evidence/platform_runtime/device_logs'),
 'HANDSON_REMOTE_TASK_ROOT':str(ROOT),
 'HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY':'0',
}.items():os.environ[key]=value

def main():
 from collection_ui.server import CollectionState
 report={'passed':False,'trial':TRIAL,'scope':'Ordinary platform start/reset-ready/stop; physical teleoperation awaits operator'}
 state=None
 try:
  state=CollectionState(ROOT,sys.executable,TASK/'evidence/dataset',os.environ['HANDSON_ISAACSIM_PYTHON'])
  report['runtimes']=state.status()['runtimes']
  assert report['runtimes']['isaacsim']['runtime_family']=='isaac_sim_5_1'
  assert 'isaacsim_v6' not in report['runtimes']
  assert not report['runtimes']['isaacsim_newton']['available']
  report['catalog_item']=next(item for item in state.tasks() if item['name']=='eval_gen03_orient_bottle_upright')
  assert report['catalog_item']['runtime_family']=='isaac_sim_5_1'
  assert report['catalog_item']['sim_backends']==['isaacsim']
  start=state.start_collection({'task_name':'eval_gen03_orient_bottle_upright','sim_backend':'isaacsim','isaac_arm_render_mode':'live','arm_type':'rokae','hand_type':'wuji1','collector_id':'gen1_validation','successes_needed':1,'randomize':True,'show_sim_cameras':False,'render_mode':'rgb_array','save_depth':False,'out_dir':str(TASK/'evidence/dataset')})
  report['start']=start
  (OUT/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
  assert start.get('success'),start
  session=start['session'];log=Path(session['log_path'])
  runtime=Path(session['runtime_status_path']) if session.get('runtime_status_path') else None
  deadline=time.monotonic()+600;ready=False
  while time.monotonic()<deadline:
   if runtime is not None and runtime.is_file():
    try:report['runtime']=json.loads(runtime.read_text())
    except json.JSONDecodeError:pass
   text=log.read_text(errors='replace') if log.is_file() else ''
   if 'Environment reset complete for eval_gen03_orient_bottle_upright.' in text and state.collection_process is not None and state.collection_process.poll() is None:
    ready=True;break
   if state.collection_process is None or state.collection_process.poll() is not None:break
   time.sleep(1)
  report['ready']=ready;report['log_path']=str(log)
  report['log_tail']=log.read_text(errors='replace')[-12000:] if log.is_file() else ''
  assert ready,'Ordinary recorder did not reach reset-ready state'
  report['stop']=state.stop_collection()
  deadline=time.monotonic()+45
  while state.collection_process is not None and state.collection_process.poll() is None and time.monotonic()<deadline:time.sleep(.5)
  report['stopped']=state.collection_process is None or state.collection_process.poll() is not None
  report['passed']=bool(report['ready'] and report['stopped'])
 except BaseException:report['error']=traceback.format_exc();raise
 finally:
  if state is not None and state.collection_process is not None and state.collection_process.poll() is None:report['cleanup_stop']=state.stop_collection()
  (OUT/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
  print('GEN1_PLATFORM_REPORT',OUT/'report.json',flush=True)
if __name__=='__main__':main()
