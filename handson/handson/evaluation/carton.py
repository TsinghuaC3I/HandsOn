"""Carton force is diagnostic; native support, lift and hold still score."""
import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

from .force_observer import RawForce, SOURCE_SHA, TASK, instrument_read_contacts

REVISION = 'carton_force_observe_only_v2'


class ObservedForce(RawForce):
    def summary(self):
        value = super().summary()
        value.update(revision=REVISION, scoring_force_limit_n=None,
                     force_affects_success=False, force_affects_termination=False,
                     diagnostic_reference_limit_n=12.0,
                     prompt_force_clause_waived_for_scoring=True)
        return value


def without_force_failure(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
             and n.test.func.id == 'any' and '_b320_contact_peak' in ast.unparse(n.test)]
    assert len(nodes) == 1, 'Unreviewed native failure predicate'
    nodes[0].test = ast.Constant(value=False)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<carton-force-observe-only-v2>', 'exec'), namespace)
    return namespace[original.__name__]


def no_force_state(native_state):
    value = dict(native_state)
    value['force_ok'] = True
    value['hold_valid'] = bool(value['airborne'] and value['support'])
    return value


def install(env, folder):
    assert env.task_name == TASK
    original = env._read_contacts.__func__
    assert hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest() == SOURCE_SHA
    failure = env._compute_failure.__func__
    assert hashlib.sha256(Path(inspect.getsourcefile(failure)).read_bytes()).hexdigest() == SOURCE_SHA
    tracker = ObservedForce(env, folder)
    tracker.replace('_read_contacts', instrument_read_contacts(original, tracker.observe))
    state, metrics = env._carton_instantaneous_state, env._carton_metrics
    substep = env._handson_post_physics_substep_callback

    def observed_state(self):
        return no_force_state(state())

    def observed_metrics(self):
        values, flags = metrics()
        values = dict(values)
        values.update(force_scoring_revision=REVISION,
                      left_side_raw_force_peak_n=tracker.peak['left'],
                      right_side_raw_force_peak_n=tracker.peak['right'],
                      raw_force_diagnostic_reference_n=12.0,
                      raw_force_within_diagnostic_reference=tracker.force_ok(),
                      force_limit_enforced=False,
                      controller_force_feedback_saturation_n=10.0)
        return values, flags

    def observed_substep(self, index, count):
        previous = tracker.substep
        tracker.substep = dict(index=int(index), count=int(count))
        try:
            return substep(index, count)
        finally:
            tracker.substep = previous

    tracker.replace('_carton_instantaneous_state', observed_state)
    tracker.replace('_compute_failure', without_force_failure(failure))
    tracker.replace('_carton_metrics', observed_metrics)
    tracker.replace('_handson_post_physics_substep_callback', observed_substep)
    return tracker


def run_with_force(original_run, env, job, output, number, seed, capture):
    assert job['carton_force_scoring']['revision'] == REVISION
    tracker = install(env, Path(output)/f'CARTON_RAW_FORCE_{number:03d}')
    try:
        result = original_run(env, job, output, number, seed, capture)
        assert all(tracker.counts[s] > 0 for s in ['left', 'right'])
        return result
    finally:
        tracker.close()
