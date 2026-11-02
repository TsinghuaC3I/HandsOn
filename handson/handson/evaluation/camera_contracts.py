"""Select the renderer inputs of the paper's task/source cohort.

The compact selector was checked against all 45,000 accepted Core trial jobs.
It changes no reset-filter geometry and contains no policy outcomes.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .contracts import require
from .paper_trials import MANIFEST_SHA256 as TRIAL_MANIFEST_SHA256, digest

MANIFEST_SHA256 = 'fdf1da4cb13df402ef7b7256d9447d13ebf6dfb33f20c6ebfcf0f088c8aa76bc'
_active = ContextVar('handson_evaluation_camera_contract', default=None)


@lru_cache(maxsize=1)
def _catalog():
    raw = Path(__file__).with_name('paper_camera_contracts.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, 'Paper camera contracts changed')
    value = json.loads(raw)
    require(value['schema'] == 'handson_paper_camera_contracts_v1' and
            value['paper_trial_manifest_sha256'] == TRIAL_MANIFEST_SHA256 and
            value['verified_trials'] == 45000, 'Unknown paper camera contracts')
    require(all(digest(v) == k for k, v in value['profiles'].items()), 'Camera profile digest differs')
    default = json.loads(Path(__file__).with_name('camera_contract.json').read_text())
    require(digest(default) == value['default_profile'], 'Default camera contract differs')
    return value


def select(task, paper_variant=None, physics_backend='physx'):
    catalog = _catalog()
    profile = catalog['default_profile']
    if physics_backend == 'physx':
        for row in catalog['overrides']:
            if (row['task'], row['paper_variant']) == (task, paper_variant):
                profile = row['profile']
                break
    return deepcopy(catalog['profiles'][profile]), dict(
        task=task, paper_variant=paper_variant, physics_backend=physics_backend,
        profile_sha256=profile, manifest_sha256=MANIFEST_SHA256,
        paper_trial_manifest_sha256=TRIAL_MANIFEST_SHA256)


def current():
    contract = _active.get()
    if contract is None:
        catalog = _catalog()
        contract = catalog['profiles'][catalog['default_profile']]
    return deepcopy(contract)


@contextmanager
def use(task, paper_variant=None, physics_backend='physx'):
    contract, receipt = select(task, paper_variant, physics_backend)
    token = _active.set(contract)
    try:
        yield receipt
    finally:
        _active.reset(token)
