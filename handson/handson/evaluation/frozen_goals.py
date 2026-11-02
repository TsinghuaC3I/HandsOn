"""Attempt-scoped goal conditioning used by the historical Core evaluator.

Install before construction, read the goal back after reset, and restore hooks
when the attempt ends. No object state, action, or success predicate is written.
The source campaign adapters are recorded in frozen_goal_provenance.json.
"""
from contextlib import ExitStack
import dataclasses
import importlib
import inspect
import json

from .contracts import require
from ..tasks.registry import ASSETS, TASKS


def condition_laptop(native, wanted, events):
    require(wanted in (30, 60, 90, 120), 'Unsupported laptop goal')

    def sample(seed):
        original = native(seed)
        require(dataclasses.is_dataclass(original), 'Laptop reset schema changed')
        result = dataclasses.replace(original, target_open_deg=float(wanted))
        before, after = dataclasses.asdict(original), dataclasses.asdict(result)
        require({k:v for k,v in before.items() if k != 'target_open_deg'} ==
                {k:v for k,v in after.items() if k != 'target_open_deg'},
                'Laptop conditioning changed a non-goal field')
        events.append(dict(kind='laptop_goal', seed=int(seed),
                           original_target=original.target_open_deg,
                           conditioned_target=result.target_open_deg))
        return result

    return sample


def condition_knife(native, wanted, events, limit=4096):
    require(wanted in ('extend', 'retract'), 'Unsupported knife goal')
    direction = 1 if wanted == 'extend' else -1

    def sample(random_state, **kwargs):
        for count in range(1, limit + 1):
            result = native(random_state, **kwargs)
            if result[3] == direction:
                events.append(dict(kind='knife_complete_native_tuple', native_draws=count,
                                   target_direction=direction, reset_progress=float(result[2])))
                return result
            if not kwargs['randomize']:
                break
        raise ValueError('Native knife sampler did not produce the frozen motion')

    return sample


def condition_timer(native, wanted, events, limit=128):
    require(len(wanted) == 2 and all(x in (1, 2) for x in wanted), 'Unsupported timer goal')

    def sample(rng):
        for count in range(1, limit + 1):
            result = native(rng)
            if list(result) == list(wanted):
                events.append(dict(kind='timer_native_goal', native_draws=count, target=list(result)))
                return result
        raise ValueError('Native timer sampler did not produce the frozen goal')

    return sample


def condition_cup(native, variants, wanted, events):
    """Consume the original variant draw, then bind only its selected index."""
    text = inspect.getsource(native)
    require('int(self._random.choice(len(ycb02.VARIANTS)))' in text and
            'variant = ycb02.VARIANTS[variant_index]' in text, 'Cup goal source changed')
    matches = [i for i,v in enumerate(variants) if v.cup == wanted]
    require(len(matches) == 1, 'Cup does not identify one native variant')

    class GoalRng:
        def __init__(self, rng):
            self.rng, self.calls = rng, 0

        def __getattr__(self, name):
            return getattr(self.rng, name)

        def choice(self, *args, **kwargs):
            value = self.rng.choice(*args, **kwargs)
            self.calls += 1
            if self.calls == 1:
                require(args == (len(variants),) and not kwargs, 'Native cup goal draw changed')
                events.append(dict(kind='cup_variant', original=int(value), conditioned=matches[0]))
                return matches[0]
            return value

    def reset(env):
        if env.task_name != 't27_place_largest_fitting_ball':
            return native(env)
        require(env.randomize, 'Frozen cup goal requires a randomized native reset')
        previous, proxy = env._random, GoalRng(env._random)
        env._random = proxy
        try:
            result = native(env)
            require(proxy.calls > 0, 'Native cup reset consumed no goal draw')
            return result
        finally:
            env._random = previous

    return reset


class FrozenGoal:
    """Bind a recorded goal route; callers supply the frozen trial's settings.

    ``condition_native=False`` preserves the older laptop/knife native draws.
    It must come from the actual campaign route, not from a requested outcome.
    """
    def __init__(self, task, values, *, condition_native=True):
        require(task in TASKS and TASKS[task].split == 'core', 'Frozen goals require a Core task')
        entries = json.loads((ASSETS/'goals.json').read_text())[task]
        require(sum(e['values'] == values for e in entries) == 1, 'Unknown frozen task goal')
        self.task, self.values = task, dict(values)
        self.condition_native = condition_native
        self.events = []
        self.constructor = {}
        self.stack = ExitStack()
        self.entered = False
        if task == 't26_place_fruit_in_bucket':
            self.constructor['target_object'] = {'apple':'apple', 'banana':'011_banana',
                                                 'orange':'orange', 'pear':'pear'}[values['fruit']]
        elif task == 't24_place_ball_in_bowl':
            self.constructor['target_object'] = values['ball'].replace(' ', '_')
        elif task == 't28_place_tableware':
            self.constructor['target_object'] = values['requested_tableware']
        elif task == 't27_place_largest_fitting_ball':
            self.constructor['target_object'] = 'tennis_ball'
        elif task == 't48_unfold_fan':
            self.constructor.update(target_object='p001_fan_center',
                                    target_open_deg=values['target_angle_degrees'])
        self.reset_parameters = dict(self.constructor)

    def _patch(self, target, name, replacement):
        original = getattr(target, name)
        self.stack.callback(setattr, target, name, original)
        setattr(target, name, replacement)

    def __enter__(self):
        require(not self.entered, 'A frozen goal context is single-use')
        self.entered = True
        try:
            if self.task == 't16_open_laptop' and self.condition_native:
                module = importlib.import_module('handson.sim.isaaclab.fu015_collection_env')
                self._patch(module, 'sample_reset_state', condition_laptop(
                    module.sample_reset_state, self.values['target_angle_degrees'], self.events))
            elif self.task == 't40_slide_knife_blade' and self.condition_native:
                module = importlib.import_module('handson.sim.isaaclab.as013_collection_env')
                self._patch(module, 'sample_reset_contract', condition_knife(
                    module.sample_reset_contract, self.values['motion'], self.events))
            elif self.task == 't39_set_timer':
                module = importlib.import_module('handson.sim.isaaclab.abc_collection_env')
                self._patch(module, '_ycb07_seeded_target', condition_timer(
                    module._ycb07_seeded_target,
                    [self.values['minutes'], self.values['seconds']], self.events))
            elif self.task == 't27_place_largest_fitting_ball':
                module = importlib.import_module('handson.sim.isaaclab.abc_collection_env')
                cls = module.HandsOnABCIsaacLabEnv
                self._patch(cls, '_reset_ycb02_episode_state', condition_cup(
                    cls._reset_ycb02_episode_state, module.ycb02.VARIANTS,
                    self.values['requested_cup'].lower()+'_cups', self.events))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)

    def verify(self, env):
        require(env.task_name == self.task, 'Frozen goal task mismatch')
        values, readback = TASKS[self.task].goal(env)
        require(values == self.values, f'Native goal differs from frozen trial: {values} != {self.values}')
        return dict(goal=values, native_goal_readback=readback, native_goal_events=list(self.events),
                    condition_native=self.condition_native, passed=True,
                    writes_recorded_state=False, success_checker_changed=False)
