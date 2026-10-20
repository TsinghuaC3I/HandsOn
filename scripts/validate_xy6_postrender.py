"""Run the shared camera CLI on a genuine, unsuccessful XY6 short capture."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'handson')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'replay_manifest.json').read_text())
    recording=manifest['recording']
    assert recording['recording_purpose']=='xy6_camera_diagnostic'
    assert recording['successful_episode'] is False
    from scripts.isaac_deferred_arm_replay_worker import _render_command,_render_env
    command=_render_command(source,'eval_comb06_season_and_stir',30)
    report={'source':str(source),'command':command,'scope':'shared camera post-render of a short diagnostic',
            'successful_episode_verified':False,'physical_replay_verified':False}
    try:
        render_env=_render_env()
        # The isolated GPU namespace rejects NVENC on this host. Reuse the
        # shared writer's supported CPU H.264 option; collection is unchanged.
        render_env['HANDSON_VIDEO_ENCODER']='cpu'
        report['video_encoder']='cpu'
        with (output/'renderer.log').open('w') as log:
            result=subprocess.run(command,env=render_env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        report['returncode']=result.returncode
        assert result.returncode==0,'Inspect renderer.log'
        log_text=(output/'renderer.log').read_text(errors='replace')
        errors=[float(x) for x in re.findall(r'ee_err=([\d.]+)m',log_text)]
        assert errors and max(errors)<.03,errors
        report['max_logged_palm_target_error_m']=max(errors)
        import cv2
        import numpy as np
        report['videos']=[]
        for camera,shape in (('main',(640,640)),('wrist_left',(480,848)),('wrist_right',(480,848))):
            file=source/'videos'/(camera+'.mp4');reader=cv2.VideoCapture(str(file))
            assert reader.isOpened(),str(file)
            frames=[]
            try:
                fps=reader.get(cv2.CAP_PROP_FPS)
                while True:
                    ok,frame=reader.read()
                    if not ok:break
                    assert frame.shape[:2]==shape,frame.shape
                    frames.append(frame)
            finally:reader.release()
            assert len(frames)==recording['step_count'] and abs(fps-30)<.001
            assert all(float(f.std())>5 for f in frames)
            change=max(float(np.abs(f.astype(float)-frames[0]).mean()) for f in frames)
            assert change>.01,'Motion capture must not produce a constant video'
            cv2.imwrite(str(output/(camera+'-first.png')),frames[0])
            cv2.imwrite(str(output/(camera+'-hold.png')),frames[min(35,len(frames)-1)])
            report['videos'].append({'camera':camera,'frames':len(frames),'fps':fps,
                'shape':list(frames[0].shape),'mean_pixel_change_max':change,
                'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'path':str(file)})
        report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc();traceback.print_exc()
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report.get('passed') else 1


if __name__=='__main__':
    raise SystemExit(main())
