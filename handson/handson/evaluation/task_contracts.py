"""Frozen native task parameters consumed by the paper evaluation entry."""
import json
import os
from pathlib import Path

from .contracts import require, write_new
from ..tasks.registry import TASKS, native_contracts


def load(task):
    return TASKS[task].native_contract()


def configure(task):
    contract = load(task)
    # The original worker starts with a clean task-control environment.
    records = native_contracts()
    keys = {key for value in records.values() for key in value['control_environment']}
    for key in keys:
        os.environ.pop(key, None)
    if contract is not None:
        os.environ.update(contract['control_environment'])
    return contract


def initialize(env, wrapped, observation, reset_info, contract, output):
    import hashlib
    import importlib
    implementations = json.loads(Path(__file__).with_name('origin_implementations.json').read_text())
    implementation = implementations.get(env.task_name)
    if implementation is None:
        from ..sim.isaaclab import recording_origin
    else:
        recording_origin = importlib.import_module(implementation['module'])
        actual = hashlib.sha256(Path(recording_origin.__file__).read_bytes()).hexdigest()
        require(actual == implementation['packaged_sha256'], 'Native origin implementation changed')
        write_new(output / 'ORIGIN_IMPLEMENTATION.json', implementation)
        from .contact_numeric import CHECKED_SOURCE_PAIRS, install_checked_copy_layout
        traces = json.loads(Path(__file__).with_name('trace_implementations.json').read_text())
        producer = traces[env.task_name]
        pair = (implementation['source_sha256'], producer['source_sha256'])
        if pair in CHECKED_SOURCE_PAIRS:
            receipt = install_checked_copy_layout(recording_origin, implementation, producer)
            write_new(output / 'CHECKER_LAYOUT_PATCH.json', receipt)
    if contract is not None and contract['origin_mode'] == 'approved_b_boundary':
        from .boundary import prepare
        prepare(env, dict(task=env.task_name,
                         origin_parameters=contract['origin_parameters']), output)
        return observation, reset_info
    if contract is None:
        kwargs = recording_origin.stable_recording_origin_kwargs_for_task(env.task_name)
    else:
        require(contract['origin_mode'] == 'native_recording', 'Unknown task origin')
        kwargs = recording_origin.stable_recording_origin_kwargs_from_report(contract['origin_parameters'])
    write_new(output / 'ORIGIN_PARAMETERS.json', kwargs)
    return recording_origin.stabilize_recording_origin(env, observation, reset_info, **kwargs)


def install_native_trace(task, output):
    """Pair each frozen origin gate with its original contact serializer."""
    import hashlib
    import importlib
    implementations = json.loads(Path(__file__).with_name('trace_implementations.json').read_text())
    implementation = implementations.get(task)
    if implementation is None:
        return
    module = importlib.import_module(implementation['module'])
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    require(actual == implementation['packaged_sha256'], 'Native trace implementation changed')
    from ..sim.isaaclab import bimanual_assembly_env
    source_type = module.PhysxGroundTruthTrace
    if not hasattr(source_type, 'snapshot_from_latest_substep'):
        # Older recorders always perform a fresh native read at this boundary.
        # Keep that behavior when called by the current shared environment.
        class NativeTrace(source_type):
            snapshot_from_latest_substep = source_type.snapshot
        trace_type = NativeTrace
    else:
        trace_type = source_type
    bimanual_assembly_env.PhysxGroundTruthTrace = trace_type
    paired_base = TASKS[task].trace_environment_module
    if paired_base:
        base = importlib.import_module("handson.sim.isaaclab." + paired_base)
        base.PhysxGroundTruthTrace = trace_type
    write_new(output / 'TRACE_IMPLEMENTATION.json', implementation)
