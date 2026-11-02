"""Hash-bound inputs of the accepted Core-50 paper trials.

This restores trial inputs, including the accepted reset candidate. It does not
rerun the historical scheduler, substitute seeds, or certify simulator parity.
"""
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

from .contracts import require

ROOT = Path(__file__).parent
MANIFEST_SHA256 = 'd2634f3f9fb2844d9c47d5105c6af14f73d58821dd858dfc2f2a9aa285a42c4d'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def manifest():
    raw = (ROOT/'paper_trials.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, 'Paper trial manifest changed')
    value = json.loads(raw)
    require(value['schema'] == 'handson_accepted_paper_trials_v1' and
            value['trials'] == 45000 and not value['includes_outcomes'], 'Unknown paper trial schema')
    return value


@lru_cache(maxsize=16)
def _group(model, task):
    catalog = manifest()
    require(model in catalog['models'] and task in catalog['tasks'], 'No archived Core paper trials for this model/task')
    name = f'{model}/{task}.json.gz'
    raw = (ROOT/'paper_trial_data'/name).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == catalog['files'][name], 'Paper trial inputs changed')
    rows = json.loads(gzip.decompress(raw))
    require([r['index'] for r in rows] == list(range(100)), 'Incomplete paper trial group')
    require(all((r['model'], r['task']) == (model, task) for r in rows), 'Wrong paper trial group')
    return rows


def load(model, task, index):
    require(type(index) is int and 0 <= index < 100, 'Paper trial index must be 0–99')
    row = deepcopy(_group(model, task)[index])
    row['input_manifest_sha256'] = MANIFEST_SHA256
    return row


def validate(row, task, seed, physics_backend, paper_variant):
    require(row == load(row['model'], task, row['index']), 'Paper trial inputs were modified')
    require(row['task'] == task and row['seed'] == seed, 'Task/seed conflicts with the paper trial')
    require(physics_backend == 'physx', 'Archived paper trials require PhysX')
    require(paper_variant == row['paper_variant'], 'Source variant conflicts with the paper trial')
    require(abs(row['control_dt'] - 1/30) < 1e-12, 'Paper trial has a different action clock')


def select_setting(row):
    reference = row['setting']
    if reference is None:
        return None
    from .settings import select_setting as select
    value = select(reference['master_seed'], row['task'], reference['episode_index'],
                   forced_setting=reference['setting_id'] if reference['forced_for_validation'] else None)
    for key in reference.keys() - {'profile_sha256', 'physics_sha256'}:
        require(value[key] == reference[key], f'Historical setting field differs: {key}')
    require(digest(value['profile']) == reference['profile_sha256'], 'Historical setting profile differs')
    require(digest(value['physics']) == reference['physics_sha256'], 'Historical setting physics differs')
    require(value['seeds']['reset'] == row['reset_seed'], 'Historical reset stream differs')
    return value


def configure_environment(row, contract):
    """Only recorded control keys; never accept paths or arbitrary overrides."""
    import os
    reference = row['control_environment']
    base = contract['control_environment']
    extra = {'HANDSON_PHYSX_MAX_CONTACT_HEADERS', 'HANDSON_PHYSX_MAX_CONTACT_POINTS',
             'HANDSON_PHYSX_MAX_FRICTION_ANCHORS'}
    require(all(reference.get(k) == v for k,v in base.items()), 'Historical control contract differs')
    require(set(reference) - set(base) <= extra, 'Unrecognized historical control override')
    for key in extra:
        os.environ.pop(key, None)
    os.environ.update(reference)


def bind_instruction(row, goal_readback):
    require(goal_readback['goal'] == row['goal'], 'Frozen prompt and native goal disagree')
    return dict(row['language'] or {}, prompt=row['prompt'], canonical_prompt=row['canonical_prompt'],
                goal=row['goal'], native_goal_readback=goal_readback['native_goal_readback'],
                source='hash-verified accepted paper trial', model=row['model'], trial_index=row['index'],
                input_manifest_sha256=MANIFEST_SHA256, paper_variant=row['paper_variant'],
                historical_language_draw_present=row['language'] is not None,
                constant_within_episode=True, heldout_in_production=False)
