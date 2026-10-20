"""BC004 raw-force scoring; preserve native controller's saturated feedback."""
import ast
import hashlib
import inspect
import json
import math
import textwrap
import types
from pathlib import Path

TASK = 't10_lift_carton'
REVISION = 'carton_raw_force_12n_v1'
SOURCE_SHA = '91a1bd42e67df088bcc755261ba92d1f38043ed15acada073d860328bd3934ec'
LIMIT_N = 12.0
SIDES = ('left', 'right')

class RawForce:
    def __init__(self, env, folder):
        self.env, self.folder = env, Path(folder)
        self.peak = dict.fromkeys(SIDES, 0.0)
        self.latest = dict.fromkeys(SIDES, None)
        self.counts = dict.fromkeys(SIDES, 0)
        self.first_exceedance = None
        self.substep = None
        self.events = self.folder.with_suffix('.jsonl').open('x')
        self.saved = {}
        self.samples = 0
        self.invalid = None
    def observe(self, env, side, force, readable):
        assert env is self.env and side in SIDES
        self.samples += 1
        row = dict(sample=self.samples, side=side, control_step=int(env.env_step),
                   physics_substep=self.substep, physics_dt=float(env._sim_dt),
                   raw_force_n=float(force) if math.isfinite(float(force)) else None,
                   readable=bool(readable))
        valid = readable and math.isfinite(float(force)) and float(force) >= 0
        if valid:
            self.latest[side] = float(force)
            self.peak[side] = max(self.peak[side], float(force))
            self.counts[side] += 1
            if float(force) > LIMIT_N and self.first_exceedance is None:
                self.first_exceedance = dict(row)
        else:
            self.invalid = dict(row)
        self.events.write(json.dumps(row, allow_nan=False)+'\n')
        if not valid:
            self.events.flush()
            raise RuntimeError('CARTON_RAW_FORCE_SENSOR_INVALID: '+json.dumps(row))
    def force_ok(self):
        return all(self.counts[s] > 0 and self.peak[s] <= LIMIT_N for s in SIDES)
    def replace(self, name, function):
        self.saved[name] = (name in self.env.__dict__, self.env.__dict__.get(name))
        setattr(self.env, name, types.MethodType(function, self.env))
    def summary(self):
        return dict(revision=REVISION, source_sha256=SOURCE_SHA, limit_n=LIMIT_N,
                    raw_peak_n=self.peak, raw_latest_n=self.latest, samples_by_side=self.counts,
                    raw_force_ok=self.force_ok(), first_exceedance=self.first_exceedance,
                    invalid_sample=self.invalid, controller_feedback_unchanged=True,
                    extra_contact_reads=0, physics_changed=False,
                    scope='policy phase only; native reset retained; every existing side-contact read observed')
    def close(self):
        for name,(present,value) in self.saved.items():
            if present:setattr(self.env,name,value)
            else:delattr(self.env,name)
        self.events.close()
        self.folder.with_suffix('.summary.json').write_text(json.dumps(self.summary(),indent=2)+'\n')


def instrument_read_contacts(original, observe):
    """One inserted observer call, before saturation, after the existing read."""
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
           and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_read_b320_contact_force_sample'
           and isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts]==['force','readable']]
    assert len(calls)==1, 'Unexpected native contact read structure'
    target=calls[0]
    parents=[n for n in ast.walk(tree) if hasattr(n,'body') and isinstance(n.body,list) and target in n.body]
    assert len(parents)==1
    body=parents[0].body;body.insert(body.index(target)+1,ast.parse('_observe_carton_raw(self, side, force, readable)').body[0])
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,_observe_carton_raw=observe)
    exec(compile(tree,'<carton-raw-force-observer-v1>','exec'),namespace)
    return namespace[original.__name__]


def install(env, folder):
    assert env.task_name==TASK
    cls=type(env)
    original=getattr(env,'_read_contacts').__func__
    path=Path(inspect.getsourcefile(original))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA, 'Unreviewed native force source'
    tracker=RawForce(env,folder)
    tracker.replace('_read_contacts',instrument_read_contacts(original,tracker.observe))
    state=env._carton_instantaneous_state
    failure=env._compute_failure
    metrics=env._carton_metrics
    substep=env._handson_post_physics_substep_callback
    def raw_state(self):
        value=dict(state())
        value['force_ok']=tracker.force_ok()
        value['hold_valid']=bool(value['airborne'] and value['support'] and value['force_ok'])
        return value
    def raw_failure(self):
        native=failure()
        if tracker.first_exceedance is not None:
            self._b320_failure_reason='carton_side_force_exceeded'
            self._success_counter=0
            return True
        return native
    def raw_metrics(self):
        values,flags=metrics()
        values=dict(values)
        values.update(force_scoring_revision=REVISION,
                      left_side_raw_force_peak_n=tracker.peak['left'],
                      right_side_raw_force_peak_n=tracker.peak['right'],
                      raw_force_limit_n=LIMIT_N,raw_force_ok=tracker.force_ok(),
                      controller_force_feedback_saturation_n=10.0)
        return values,flags
    def raw_substep(self,index,count):
        previous=tracker.substep
        tracker.substep=dict(index=int(index),count=int(count))
        try:return substep(index,count)
        finally:tracker.substep=previous
    tracker.replace('_carton_instantaneous_state',raw_state)
    tracker.replace('_compute_failure',raw_failure)
    tracker.replace('_carton_metrics',raw_metrics)
    tracker.replace('_handson_post_physics_substep_callback',raw_substep)
    return tracker


def run_with_force(original_run,env,job,output,number,seed,capture):
    assert job['carton_force_scoring']['revision']==REVISION
    tracker=install(env,Path(output)/f'CARTON_RAW_FORCE_{number:03d}')
    try:
        result=original_run(env,job,output,number,seed,capture)
        assert all(tracker.counts[s]>0 for s in SIDES), 'No raw force samples during policy'
        return result
    finally:tracker.close()
