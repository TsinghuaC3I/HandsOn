"""Check complete reset tuple preservation and attempt-local hook cleanup."""
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from handson.evaluation.frozen_goals import (
    FrozenGoal, condition_knife, condition_laptop, condition_timer,
)


@dataclass
class LaptopReset:
    target_open_deg: float
    pose: tuple
    contact_gate: int


def test_laptop_conditions_only_the_angle_after_one_complete_native_draw():
    calls = []
    def native(seed):
        calls.append(seed)
        return LaptopReset(90., (0.1, -0.2, 0.9), 17)
    events = []
    result = condition_laptop(native, 30, events)(123)
    assert result == LaptopReset(30., (0.1, -0.2, 0.9), 17)
    assert calls == [123]
    assert events[0]['original_target'] == 90.


def test_knife_rejection_preserves_the_entire_accepted_native_tuple_and_rng():
    rejected = (np.array([.1, .2]), 20., .7, -1)
    accepted = (np.array([-.1, .3]), 100., .2, 1)
    draws = iter([rejected, accepted])
    def native(rng, **kwargs):
        rng.uniform(size=4)
        return next(draws)
    rng, reference = np.random.RandomState(42), np.random.RandomState(42)
    events = []
    actual = condition_knife(native, 'extend', events)(rng, randomize=True)
    reference.uniform(size=8)
    assert actual is accepted
    assert rng.uniform() == reference.uniform()
    assert events[0]['native_draws'] == 2


def test_impossible_goals_fail_with_bounded_draws():
    with pytest.raises(ValueError, match='knife sampler'):
        condition_knife(lambda rng, **kw: (None, 0, .8, -1), 'extend', [], limit=3)(
            None, randomize=False)
    calls = []
    def timer(rng):
        calls.append(1)
        return (1, 1)
    with pytest.raises(ValueError, match='timer sampler'):
        condition_timer(timer, [2, 2], [], limit=3)(None)
    assert len(calls) == 3


def test_hook_restored_after_reset_error_and_old_native_route_untouched():
    original = lambda seed: LaptopReset(90., (0, 0, 0), 0)
    module = SimpleNamespace(sample_reset_state=original)
    with patch('handson.evaluation.frozen_goals.importlib.import_module', return_value=module):
        with pytest.raises(RuntimeError):
            with FrozenGoal('t16_open_laptop', {'target_angle_degrees':30}):
                assert module.sample_reset_state(0).target_open_deg == 30
                raise RuntimeError('native reset failed')
        assert module.sample_reset_state is original
        with FrozenGoal('t16_open_laptop', {'target_angle_degrees':90}, condition_native=False):
            assert module.sample_reset_state is original


def test_instruction_goal_cannot_be_bound_to_a_different_native_reset():
    goal = FrozenGoal('t24_place_ball_in_bowl', {'ball':'golf ball'})
    env = SimpleNamespace(task_name='t24_place_ball_in_bowl', _target_object_name='tennis_ball')
    with pytest.raises(ValueError, match='Native goal differs'):
        goal.verify(env)
