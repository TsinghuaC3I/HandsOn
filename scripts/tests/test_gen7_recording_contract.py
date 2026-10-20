"""Exercise the shared real recording and replay schema paths on CPU."""
import numpy as np
import pytest
import zarr

from handson.data.recording_fields import build_recording_schema, stack_transition_numeric_observation_fields
from handson.sim.isaaclab import eval_gen06_spray_shoe_scene as g
from scripts.certify_physx_replay import CertificationError, _validated_task_state_schema


def recorded_fields():
    schema=g.recording_schema({'fields':{}})
    observations=[]
    for frame in range(4):
        observations.append({'state':np.zeros(49,dtype=np.float32),
            'gen7_hidden_state':np.r_[np.zeros(12),float(frame)],
            'gen7_reset_parameters':np.array([-.085,-.22,0,-.26,.12,0],dtype=np.float64)})
    trajectory=[{'actions':np.zeros(27,dtype=np.float32),'observations':observations[i],
                 'next_observations':observations[i+1]} for i in range(3)]
    fields=stack_transition_numeric_observation_fields(trajectory,environment_schema=schema)
    return schema,fields


def test_shared_writer_records_both_hidden_boundaries():
    schema,fields=recorded_fields()
    for name,layout in g.RECORDING_LAYOUTS.items():
        assert fields[name].shape==(3,len(layout))
        assert fields[f'next_{name}'].shape==(3,len(layout))
        np.testing.assert_array_equal(fields[name][1:],fields[f'next_{name}'][:-1])
    stored=build_recording_schema(fields,environment_schema=schema)
    result=_validated_task_state_schema(g.TASK_NAME,schema,schema['fields'],stored['fields'])
    assert set(result)==set(g.RECORDING_LAYOUTS)
    stored['fields']['next_gen7_hidden_state']['shape_per_step']=[12]
    with pytest.raises(CertificationError):
        _validated_task_state_schema(g.TASK_NAME,schema,schema['fields'],stored['fields'])


def test_visual_replay_reads_the_real_recording_extras(tmp_path):
    from scripts.replay_demos_zarr import _load_gen7_recording_state_extras
    _,fields=recorded_fields()
    root=zarr.open(str(tmp_path/'episode.zarr'),mode='w');data=root.create_group('data')
    for name,values in fields.items():data.create_dataset(name,data=values)
    result=_load_gen7_recording_state_extras(tmp_path/'episode.zarr',expected_steps=3)
    for name in g.RECORDING_LAYOUTS:np.testing.assert_array_equal(result[name],fields[name])
    del data['gen7_hidden_state']
    with pytest.raises(ValueError,match='missing gen7_hidden_state'):
        _load_gen7_recording_state_extras(tmp_path/'episode.zarr',expected_steps=3)
