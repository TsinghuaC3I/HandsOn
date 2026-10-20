#!/usr/bin/env python3
"""Compare canonical contact outputs and CPU cost on real GEN4 journal inputs."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from handson.data.physx_session_history import load_discarded_trajectory
from handson.sim.isaaclab import physx_ground_truth as module


def main():
    root = Path(__file__).resolve().parents[3]
    evidence = Path(__file__).with_name('evidence')
    platform = json.loads((evidence / 'live_platform_attempt6/history_summary.json').read_text())
    session = root / platform['sessions'][0]['path']
    data = load_discarded_trajectory(session, 1)
    schema_path = session / 'events/00000001/recording_schema.json'
    schema = json.loads(schema_path.read_text())['environment_schema']['physx']['contact_report']
    source_path = 'handson/handson/sim/isaaclab/physx_ground_truth.py'
    base = '282390f0fec08daca53e2d6b03f6ad8772b609f4'
    baseline_source = subprocess.check_output(['git', 'show', f'{base}:{source_path}'], cwd=root, text=True)
    tree = ast.parse(baseline_source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PhysxGroundTruthTrace')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_canonicalize_contact_fields')
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<release-contact-canonicalizer>', 'exec'), namespace)
    baseline = namespace['_canonicalize_contact_fields']
    current = module.PhysxGroundTruthTrace._canonicalize_contact_fields
    trace = object.__new__(module.PhysxGroundTruthTrace)
    trace.max_contact_headers = schema['max_headers']
    trace.max_contact_points = schema['max_contact_points']
    trace.max_friction_anchors = schema['max_friction_anchors']
    trace._contact_report_paths = tuple(schema['report_enabled_task_body_paths'])
    trace._tensor_contact_sensor_paths = tuple(schema['tensor_sensor_paths'])
    trace._tensor_contact_filter_paths = tuple(schema['tensor_filter_paths'])
    trace._contact_report_path_table = tuple(schema['path_table'])
    contact_names = tuple(trace._empty_contact_fields())
    samples = []
    for step, count in enumerate(data['next_physx_substep_count'][:, 0]):
        for substep in range(int(count)):
            samples.append({name: data[name.replace('physx_', 'next_physx_substep_', 1)][step, substep].copy()
                            for name in contact_names})
    assert samples
    for fields in samples:
        expected = baseline(trace, {k: v.copy() for k, v in fields.items()})
        actual = current(trace, {k: v.copy() for k, v in fields.items()})
        assert set(actual) == set(expected)
        for name in expected:
            assert actual[name].dtype == expected[name].dtype
            assert actual[name].tobytes() == expected[name].tobytes(), name
        assert actual['physx_contact_signature_valid'][0] == 1
    timings = {'baseline': [], 'current': []}
    for _ in range(5):
        for label, function in [('baseline', baseline), ('current', current)]:
            start = time.perf_counter()
            for sample in samples:
                function(trace, sample)
            timings[label].append(time.perf_counter() - start)
    medians = {label: float(np.median(values)) for label, values in timings.items()}
    result = {'task': 'GEN4', 'scope': 'CPU canonicalization only; not end-to-end wall throughput',
              'actual_physics_substep_samples': len(samples), 'all_fields_byte_identical': True,
              'baseline_commit': base, 'baseline_source_sha256': hashlib.sha256(baseline_source.encode()).hexdigest(),
              'current_source_sha256': hashlib.sha256((root / source_path).read_bytes()).hexdigest(),
              'recording_schema_sha256': hashlib.sha256(schema_path.read_bytes()).hexdigest(),
              'elapsed_seconds': timings, 'median_seconds': medians,
              'cpu_speedup': medians['baseline'] / medians['current']}
    (evidence / 'contact_readback_cpu_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
