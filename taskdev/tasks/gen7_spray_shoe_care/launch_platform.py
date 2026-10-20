"""Launch an isolated ordinary UI without uploads, notifications or GPU warmup."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18077)
    p.add_argument('--probe-run',default='probe_15');args=p.parse_args()
    task=Path(__file__).resolve().parent;root=task.parents[2]
    previous=json.loads((task/'evidence/runtime'/args.probe_run/'launch.json').read_text())
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        check.bind(('127.0.0.1',args.port))
    output=task/'evidence/platform';output.mkdir(parents=True,exist_ok=True)
    cache=task/'cache/runtime'
    for name in ('xdg-runtime','xdg-config','xdg-state'):(cache/name).mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(previous['environment_overrides'])
    env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1',
        'PATH':str(task/'cache/bootstrap/git/usr/bin')+':'+env['PATH'],
        'GIT_EXEC_PATH':str(task/'cache/bootstrap/git/usr/lib/git-core'),
        'OMNI_KIT_ACCEPT_EULA':'YES','HEADLESS':'1',
        'TMPDIR':str(cache/'tmp'),'XDG_CACHE_HOME':str(cache/'xdg'),
        'XDG_RUNTIME_DIR':str(cache/'xdg-runtime'),'XDG_CONFIG_HOME':str(cache/'xdg-config'),
        'XDG_STATE_HOME':str(cache/'xdg-state'),
        'CUDA_CACHE_PATH':str(cache/'cuda'),'TORCH_HOME':str(cache/'torch'),
        'HANDSON_COLLECTION_RUNTIME_DIR':str(cache/'collection_ui'),
        'HANDSON_MODE_A_LOG_DIR':str(cache/'mode_a_logs'),
        'HANDSON_COLLECTION_SIM_BACKEND':'isaacsim','HANDSON_COLLECTION_ARM_TYPE':'rokae',
        'HANDSON_COLLECTION_HAND_TYPE':'wuji1',
        'HANDSON_BACKGROUND_V3_UPLOAD':'0','HANDSON_REVIEW_AUTO_UPLOAD':'0',
        'HANDSON_IDLE_MAINTENANCE':'0','HANDSON_WORK_PLAN_DESKTOP_NOTIFICATIONS':'0',
        'HANDSON_ISAACSIM_CACHE_WARMUP':'0'})
    command=[previous['runtime'],str(root/'collection_ui/server.py'),'--host','127.0.0.1',
             '--port',str(args.port),'--root',str(root),'--python',previous['runtime'],
             '--isaacsim-python',previous['runtime'],'--out-dir',str(output/'episodes')]
    with (output/'server.log').open('w') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':child.pid,'command':command,'url':f'http://127.0.0.1:{args.port}',
            'runtime_probe':args.probe_run,'background_uploads_notifications_warmup':False}
    (output/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
