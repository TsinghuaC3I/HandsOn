"""Compare with calls/states from the unchanged historical USD helper."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from handson.sim.isaaclab import paper_reset_visual_rng as module


class RecordedRandom:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.calls = []

    def __getattr__(self, name):
        operation = getattr(self.rng, name)
        def call(*args, **kwargs):
            value = operation(*args, **kwargs)
            self.calls.append(dict(method=name, args=list(args), kwargs=kwargs,
                                   value=np.asarray(value).tolist()))
            return value
        return call


def test_draws_and_entire_rng_state_match_original_helper():
    fixture = json.loads((Path(__file__).parent / 'fixtures/paper_reset_visual_rng.json').read_text())
    for case in fixture['cases']:
        random = RecordedRandom(case['seed'])
        module.sample(random, key_light_radius=True)
        assert random.calls == case['calls']
        state = random.rng.get_state()
        assert dict(keys=state[1].tolist(), position=state[2], has_gauss=state[3],
                    cached_gaussian=state[4]) == case['state']


def test_missing_key_radius_omits_only_conditional_last_draw():
    with_radius, without_radius = RecordedRandom(7), RecordedRandom(7)
    module.sample(with_radius, key_light_radius=True)
    module.sample(without_radius, key_light_radius=False)
    assert without_radius.calls == with_radius.calls[:-1]


def test_later_sources_newton_and_nonrandom_resets_do_not_touch_the_rng():
    base = dict(task_name='t03_cover_box', _physics_backend='physx', randomize=True,
                _randomize_task_scene_visuals=object(), _random=object())
    with patch.object(module, 'sample', side_effect=AssertionError('unexpected draw')):
        for replacement in [dict(task_name='t08_hammer_nail'), dict(_physics_backend='newton'),
                            dict(randomize=False), dict(_randomize_task_scene_visuals=None)]:
            module.preserve(SimpleNamespace(**dict(base, **replacement)))
