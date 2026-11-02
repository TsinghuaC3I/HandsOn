"""Renaming a task must preserve the paper's appearance and camera draws."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from handson.evaluation.camera_randomization import sample
from handson.sim.isaaclab.room_appearance import resolve

ORACLE = json.loads((Path(__file__).parent/'fixtures/paper_visual_sampling.json').read_text())


@pytest.mark.parametrize('row', ORACLE['rows'], ids=lambda r:r['task'])
def test_public_task_names_preserve_recorded_visual_draws(row):
    profile = resolve(dict(mode='randomized',spp=32,task=row['task'],seed=row['seed']))
    assert profile['sampling_task'] == row['source_task']
    assert profile['sampling_version'] == row['appearance']['version']
    for key,value in row['appearance'].items():
        if key not in {'task','version'}:
            assert profile[key] == value, key
    cameras = sample(row['task'],'independent-camera-probe',row['seed'])
    assert cameras['sampling_task'] == row['source_task']
    for key,value in row['cameras'].items():
        if key != 'task':
            assert cameras[key] == value, key


def test_explicit_recorded_appearance_is_not_resampled():
    recorded = ORACLE['rows'][0]['appearance']
    actual = resolve(recorded)
    assert all(actual[key] == value for key,value in recorded.items())


def test_archived_setting_uses_appearance_stream_instead_of_reset(monkeypatch):
    from handson.evaluation.paper_trials import load, select_setting
    from handson.sim.isaaclab import b_eval_setting
    from handson.sim.isaaclab.default_appearance import episode_appearance_seed
    task='t46_store_charger'
    row=load('pi05',task,0)
    selection=select_setting(row)
    # Actual paper attempt a1-3813fd4f4eab's appearance_000.json.
    assert row['reset_seed'] == 710520073
    assert selection['seeds']['appearance'] == 1649711851
    monkeypatch.setattr(b_eval_setting,'selected',lambda:selection)
    env=SimpleNamespace(task_name=task,_seed=row['reset_seed'])
    assert episode_appearance_seed(env) == 1649711851
    env.task_name='eval_flex03_pack_pillow'
    assert episode_appearance_seed(env) == 710520073
    monkeypatch.setattr(b_eval_setting,'selected',lambda:None)
    assert episode_appearance_seed(env) == 710520073
