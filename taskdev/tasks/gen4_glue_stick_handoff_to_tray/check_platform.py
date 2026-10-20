#!/usr/bin/env python3
"""Exercise ordinary GEN4 collection HTTP start/reset/stop in a private UI.

This does not fabricate a successful episode. Save/video/replay acceptance is
reported separately and requires an actually successful physical trajectory.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[3]
    from handson.tasks.eval_gen04_pass_glue_stick.contract import TASK_NAME,PROMPT
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    env=dict(os.environ)
    overrides={
        'HANDSON_COLLECTION_CLEAN_STALE_ON_START':'0','HANDSON_IDLE_MAINTENANCE':'0',
        'HANDSON_ISAACSIM_CACHE_WARMUP':'0','HANDSON_BACKGROUND_V3_UPLOAD':'0',
        'HANDSON_REVIEW_AUTO_UPLOAD':'0','HANDSON_COLLECTION_TELEOP_STACK':'disabled',
        'HANDSON_COLLECTION_REQUIRE_TELEOP_READY':'0','HANDSON_AUTO_START_TELEOP':'1',
        'HANDSON_ISAAC_PRESERVE_CUDA_VISIBLE_DEVICES':'1','HANDSON_GPU_DEVICE':'0',
        'HANDSON_ISAACSIM_PORTABLE_ROOT':str(root/'.gen4-local/kit'),
        'HANDSON_GEN4_EXCLUSIVE_GPU_UUID':env['CUDA_VISIBLE_DEVICES'],
        'HANDSON_GRAPHICS_LIBRARY_PATH':str(root.parent/'host-tools/usr/lib/x86_64-linux-gnu'),
    }
    env.update(overrides)
    out=root/'.gen4-local/platform-episodes';out.mkdir(exist_ok=True)
    command=[sys.executable,'-B',str(root/'collection_ui/server.py'),'--host','127.0.0.1',
        '--port',str(port),'--root',str(root),'--python',sys.executable,
        '--isaacsim-python',env['HANDSON_ISAACSIM_PYTHON'],'--out-dir',str(out)]
    report={'task':'GEN4','url':url,'server_command':command,'environment_overrides':overrides,
        'cycles':[],'exceptions':[],'completed':False,'successful_save_and_replay':'pending physical completion'}
    server=None;active=False
    def write():
        (args.output/'platform.json').write_text(json.dumps(report,indent=2)+'\n')
    def request(path,payload=None,timeout=20):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    try:
        with (args.output/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['server_pid']=server.pid;write()
        deadline=time.monotonic()+45
        while True:
            if server.poll() is not None:raise RuntimeError('The isolated collection server exited; inspect server.log')
            try:
                report['health']=request('/api/health',timeout=2);break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(1)
        catalog=request('/api/tasks')
        (args.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
        matches=[x for x in catalog.get('items',[]) if x.get('name')==TASK_NAME]
        assert len(matches)==1 and matches[0]['prompt']==PROMPT
        report['catalog_entry']=matches[0]
        report['initial_status']=request('/api/status');write()
        for index in range(3):
            row={'cycle':index+1};report['cycles'].append(row);write()
            payload={'task_name':TASK_NAME,'task_prompt':PROMPT,'sim_backend':'isaacsim',
                'arm_type':'rokae','hand_type':'wuji1','robot_type':'ROKAE+Wuji',
                'isaac_arm_render_mode':'live','collector_id':'gen4_isolated_validation',
                'collection_note':'GEN4 ordinary launch/reset/stop verification',
                'out_dir':str(out),'randomize':True,'render_mode':'rgb_array',
                'show_sim_cameras':False,'save_depth':False,'successes_needed':1}
            row['start']=request('/api/collection/start',payload,timeout=45);write()
            assert row['start'].get('success'),row['start']
            active=True
            session=row['start'].get('session',{})
            logpath=Path(row['start'].get('log_path') or session.get('log_path',''))
            assert logpath.is_relative_to(root)
            deadline=time.monotonic()+180
            while True:
                content=logpath.read_text(errors='replace') if logpath.exists() else ''
                if f'Environment reset complete for {TASK_NAME}.' in content:break
                if 'Traceback (most recent call last)' in content:raise RuntimeError('The ordinary recorder raised an error; inspect its child log')
                if time.monotonic()>deadline:raise TimeoutError('Ordinary recorder did not reach its accepted initial state')
                time.sleep(2)
            row['reset_ready']=True
            row['status_running']=request('/api/status');write()
            # Allow shader/first-action startup to finish and retain several
            # complete steady timing windows from the ordinary recorder.
            row['steady_observation_seconds']=10;write()
            time.sleep(10)
            if index==0:
                candidates=[]
                # The ordinary entry is a supervisor. Signal only its owned
                # real recorder descendant with the installed reset handler.
                pending=[int(row['start']['pid'])];seen=set()
                while pending:
                    pid=pending.pop()
                    if pid in seen:continue
                    seen.add(pid);proc=Path('/proc')/str(pid)
                    try:
                        pending.extend(int(x) for x in (proc/'task'/str(pid)/'children').read_text().split())
                        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                        flags=(proc/'environ').read_bytes().split(b'\0')
                        status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
                        catches_reset=int(status['SigCgt'].strip(),16)&(1<<(signal.SIGUSR1-1))
                    except (FileNotFoundError,PermissionError):continue
                    if (str(root/'scripts/record_demos_zarr.py') in cmd and TASK_NAME in cmd
                        and b'HANDSON_ISAAC_DEFERRED_ARM_WORKER=1' in flags and catches_reset):
                        candidates.append(pid)
                candidates=sorted(set(candidates));row['manual_reset_worker_candidates']=candidates
                if len(candidates)==1:
                    before=logpath.read_text(errors='replace').count('[Keyboard] R reset triggered from the live camera')
                    before_ready=logpath.read_text(errors='replace').count('[Recorder] Stable physical frame 0:')
                    os.kill(candidates[0],signal.SIGUSR1)
                    deadline=time.monotonic()+120
                    while time.monotonic()<deadline:
                        content=logpath.read_text(errors='replace')
                        if content.count('[Keyboard] R reset triggered from the live camera')>before:
                            row['manual_reset_requested']=True
                            if content.count('[Recorder] Stable physical frame 0:')>before_ready:
                                row['manual_reset_completed']=True;break
                        if 'Traceback (most recent call last)' in content:raise RuntimeError('Manual reset failed; inspect the ordinary recorder log')
                        time.sleep(2)
                write()
            row['stop']=request('/api/collection/stop',{},timeout=60);active=False
            assert row['stop'].get('success'),row['stop']
            (args.output/f'cycle_{index+1}_recorder.log').write_text(logpath.read_text(errors='replace'))
            row['completed_start_stop']=True;write()
            time.sleep(3)
        report['completed']=True
    except BaseException:
        report['exceptions'].append(traceback.format_exc());raise
    finally:
        if active:
            try:report['cleanup_stop']=request('/api/collection/stop',{},timeout=60)
            except Exception as exc:report['cleanup_stop_error']=str(exc)
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:report['server_cleanup_pending_pid']=server.pid
        write()


if __name__=='__main__':main()
