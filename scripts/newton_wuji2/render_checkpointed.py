"""Canonical full replay with atomic frame checkpoints on shared storage.

The initial run uses every original action/state unchanged. Recovery continues
from the next recorded state, with the same frozen V16 profile and visual-only
pose-driven glove. No interpolation or sparse frame substitution is used.
"""
from pathlib import Path
import gzip,hashlib,json,os,pickle,sys
import numpy as np
from PIL import Image

root=Path(os.environ['HANDSON_WUJI2_RUNTIME'])
sys.path.insert(0,str(root/'source'))
from scripts import isaac_render_arm_replay as renderer

sample=Path(os.environ['HANDSON_WUJI2_EPISODE'])
checkpoint=sample/'frame_checkpoints';checkpoint.mkdir(exist_ok=True)
original=renderer._replay_single_demo
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def replay(actions,initial_state,recorded_states,*args,**kwargs):
    assert len(actions)==len(recorded_states)
    maximum=int(os.environ.get('HANDSON_RENDER_MAX_FRAMES','0'))
    if maximum:
        actions,recorded_states=actions[:maximum],recorded_states[:maximum]
    frames=len(actions)
    assert frames > 0
    contract={'source_archive_sha256':sha(sample/'replay.zarr.zip'),
        'actions_sha256':hashlib.sha256(actions.tobytes()).hexdigest(),
        'states_sha256':hashlib.sha256(recorded_states.tobytes()).hexdigest(),
        'manifest_sha256':sha(root/'assets'/os.environ['WUJI2_VISUAL_BUNDLE']/'manifest.json'),
        'cloth_profile_sha256':sha(root/os.environ['WUJI2_CLOTH_PROFILE']),
        'cloth_module_sha256':sha(root/(os.environ['WUJI2_CLOTH_MODULE']+'.py')),
        'checkpoint_code_sha256':sha(__file__),'V16_spp':int(os.environ['DEXJOCO_V16_PREVIEW_SPP']),'frames':frames,'fps':30}
    identity=checkpoint/'CONTRACT.json'
    if identity.exists():assert json.loads(identity.read_text())==contract
    else:atomic_json(identity,contract)
    previous=[]
    for index in range(frames):
        path=checkpoint/f'{index:06d}.pkl.gz';receipt=checkpoint/f'{index:06d}.json'
        if not path.exists() or not receipt.exists():break
        row=json.loads(receipt.read_text());assert row['source_index']==index and sha(path)==row['sha256']
        with gzip.open(path,'rb') as stream:previous.append(pickle.load(stream))
    start=len(previous)
    if start==frames:
        # A process may finish its final capture and be interrupted before MP4
        # publication. The canonical writer needs only the saved transitions.
        sources=json.loads((checkpoint/'000000.json').read_text())['camera_sources']
        return False,previous,{},sources,{}
    callback=kwargs.pop('transition_callback',None)
    completed=start
    def save(transition,camera_sources):
        nonlocal completed
        if callback:callback(transition,camera_sources)
        index=completed
        assert np.array_equal(transition['actions'],actions[index])
        path=checkpoint/f'{index:06d}.pkl.gz';temp=checkpoint/f'{index:06d}.tmp'
        with gzip.open(temp,'wb',compresslevel=1) as stream:pickle.dump(transition,stream,protocol=5)
        temp.replace(path)
        atomic_json(checkpoint/f'{index:06d}.json',{'source_index':index,'source_time_s':index/30,
            'source_state_sha256':hashlib.sha256(recorded_states[index].tobytes()).hexdigest(),
            'sha256':sha(path),'camera_sources':camera_sources})
        if index in (0,frames//2,frames-1):
            stills=sample/'progress_stills';stills.mkdir(exist_ok=True)
            for name in renderer.CANONICAL_CAMERA_KEYS:
                key=camera_sources[name];raw=renderer.observation_image(transition['observations'],key)
                frame=renderer.postprocess_canonical_image(renderer._safe_rgb(raw),name,key)
                Image.fromarray(frame).save(stills/f'{name}_{index:06d}.png')
        completed+=1
        atomic_json(checkpoint/'PROGRESS.json',{'completed_frames':completed,'expected_frames':frames,'last_source_index':index,'resume_start':start})
    kwargs['transition_callback']=save
    if start:
        kwargs['desc']+=f' (resuming source frame {start})'
        result=original(actions[start:],recorded_states[start].ravel(),recorded_states[start:],*args,**kwargs)
    else:
        result=original(actions,initial_state,recorded_states,*args,**kwargs)
    result=list(result);result[1]=previous+result[1]
    assert completed==len(result[1])==frames
    return tuple(result)

renderer._replay_single_demo=replay
renderer.app.run(renderer.main)
