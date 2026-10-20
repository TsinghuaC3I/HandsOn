import json
from pathlib import Path
import random

from handson.tasks import language
from handson.tasks.registry import ASSETS


def test_every_published_goal_has_the_original_language_split():
    goals = json.loads((ASSETS / 'goals.json').read_text())
    bank = language.bank()
    assert set(bank['tasks']) == set(goals)
    for task, entries in goals.items():
        for entry in entries:
            matches = [row for row in bank['tasks'][task]['goals'] if row['values'] == entry['values']]
            assert len(matches) == 1
            row = matches[0]
            assert row['canonical'] == entry['canonical']
            assert len(row['train']) == 32 and len(row['heldout']) == 8
            assert not set(row['train']) & set(row['heldout'])


def test_language_draw_is_repeatable_and_does_not_consume_environment_rng():
    task = 't26_place_fruit_in_bucket'
    goal = {'fruit': 'apple'}
    before = random.getstate()
    draws = [language.sample(task, goal, seed) for seed in range(100)]
    assert random.getstate() == before
    assert draws == [language.sample(task, goal, seed) for seed in range(100)]
    assert any(row['prompt_variant'] == 'canonical' for row in draws)
    assert any(row['prompt_variant'].startswith('train_') for row in draws)
    assert all(not row['heldout_in_production'] for row in draws)
