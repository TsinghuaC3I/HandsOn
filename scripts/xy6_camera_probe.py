"""Save genuine short XY6 state/action captures for a camera-only diagnostic.

These artifacts are not accepted benchmark episodes or physical replay proof.
The source is a real camera-deferred control run, with success left unchanged.
The normal shared camera CLI consumes the same Zarr and manifest fields below.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def save_camera_input(path: Path, rows, env, seed: int):
    import zarr
    from handson.data.episode_store import ZarrEpisodeStore
    from handson.data.recording_fields import write_recording_schema

    assert rows and not any(r['succeed'] for r in rows)
    assert not env._cameras and env._deferred_camera_render
    path.mkdir(parents=True,exist_ok=False)
    data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    data['timestamp']=np.arange(len(rows),dtype=np.float64)*env._control_dt
    assert data['action'].shape==(len(rows),54)
    assert data['state'].shape==(len(rows),123)
    assert np.array_equal(data['state'][1:],data['next_state'][:-1])
    assert all(np.isfinite(v).all() for v in data.values())
    store=ZarrEpisodeStore.create_empty(storage=zarr.DirectoryStore(str(path/'replay.zarr')))
    store.append_episode(data)
    schema=write_recording_schema(path,store.root,data)
    manifest={'format':'handson.replay_manifest','version':1,
        'task':{'name':env.task_name},
        'recording':{'recording_purpose':'xy6_camera_diagnostic',
            'recorder':'validate_xy6_live.py','successful_episode':False,
            'sim_backend':'isaacsim','randomize':True,'collection_run_seed':0,
            'episode_reset_seed':int(seed),'step_count':len(rows),
            'action_shape_per_step':[54],'action_semantics':'exact_raw_environment_action_executed',
            'action_storage_layout':'right_pose_left_pose_right_hand_left_hand',
            'arm_type':'rokae','hand_type':'wuji1',
            'recording_camera_profile':env.recording_camera_profile,
            'online_camera_capture_scope':'all_canonical_postrender'},
        'environment':{'variables':{}},
        'assets':{'recording_schema':{'path':schema.name,
            'sha256':hashlib.sha256(schema.read_bytes()).hexdigest()}}}
    (path/'replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'path':str(path),'steps':len(rows),'online_camera_count':0,
            'all_success_false':True,'scope':'short camera input; no accepted episode or physical-replay claim'}
