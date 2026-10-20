"""Offline reproduction using pinned producer/checker and captured live arrays.

Never imports Isaac, steps physics, changes contact data, or relaxes a threshold.
"""
import argparse
import ast
from collections.abc import Mapping
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .contracts import digest, read, require, write_new


CHECKED_CANDIDATE_SHA256 = '9e2db4f97d6749fe6a0a198218f2afb35f9ef7e9ea46165758d2ee1fb6e7f6b6'
CHECKED_SOURCE_PAIRS = {
    ('49a8111f235e1d1872044d47e720e5186b1a794614ad1910ac4e97faa60a9464',
     'f033481c143cb9d716ebb1895587984d126916172e8a5bac9f4544649d87976b'):
        dict(candidate_sha256=CHECKED_CANDIDATE_SHA256,
             evidence='Reviewed C delivery contact_numeric_audit.py; D02_D06_CONTACT_REPRO.json; C10 same exact producer/checker files'),
    ('e9b53f5ee3e32a1d2022944300877c483365cd67a4c0dee4b057c77f8eaee2a5',
     '77bf602bf1f3ac1c55b23f8128bf4ccd7a788e1c196a01c9b4cc4ca82763b156'):
        dict(candidate_sha256=CHECKED_CANDIDATE_SHA256,
             evidence='A03 captured live arrays; contact_numeric_001/RESULT.json'),
    ('8933925eaaa2e7c7325ceefa1d5891d65d3b21764f3457892aa8d900f463d39c',
     '66b97e001d4c915d49e07ccf12c9125e2a6a6ea76aa50d90e09e650652b9ff82'):
        dict(candidate_sha256=CHECKED_CANDIDATE_SHA256,
             evidence='C06 pinned functions on A03 fixtures; contact_numeric_001/C06_FUNCTION_RECHECK.json'),
    ('4a733bfaef3d33cb4d0ff1deab0a19ac6fb771bb72394f16c22aa521af323161',
     '77bf602bf1f3ac1c55b23f8128bf4ccd7a788e1c196a01c9b4cc4ca82763b156'):
        dict(candidate_sha256='85d18aedd6853aea60d5218ab3ba8b987f04584b51bb2ad197780cb8357c298f',
             evidence='B01 captured live arrays; contact_numeric_001/B01_FUNCTION_RECHECK_009.json'),
}


def checked_source_pair(origin_sha256, producer_sha256):
    pair = (origin_sha256, producer_sha256)
    require(pair in CHECKED_SOURCE_PAIRS, 'Unreviewed origin/producer source pair')
    return CHECKED_SOURCE_PAIRS[pair]


def extract_function(path, name):
    text = Path(path).read_text()
    nodes = [n for n in ast.walk(ast.parse(text))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, f'Expected one {name}')
    return ast.get_source_segment(text, nodes[0])


def compile_function(text, name):
    namespace = dict(np=np, math=math, Mapping=Mapping, Any=Any)
    exec(compile(text, '<pinned-contact-diagnostic>', 'exec'), namespace)
    return namespace[name]


def contiguous_checker_source(text):
    """Match producer per-vector dtype/copy; retain exact array_equal checks."""
    replacements = {
        'normal = row[3:6]': 'normal = np.array(row[3:6], dtype=np.float64, copy=True)',
        'impulse = row[6:9]': 'impulse = np.array(row[6:9], dtype=np.float64, copy=True)',
        'float(np.linalg.norm(row[3:6]))':
            'float(np.linalg.norm(np.array(row[3:6], dtype=np.float64, copy=True)))',
    }
    for old, new in replacements.items():
        require(text.count(old) == 1, f'Unreviewed checker: {old}')
        text = text.replace(old, new)
    return text


def install_checked_copy_layout(module, origin_record, producer_record):
    """Scoped correction for explicitly tested source pairs; no file edits.

    C06's offline fixtures originate from A03: this grants a diagnostic run,
    not task-wide runtime acceptance. Unknown combinations remain rejected.
    """
    origin_path = Path(module.__file__)
    import importlib
    producer = importlib.import_module(producer_record['module'])
    producer_path = Path(producer.__file__)
    require(digest(origin_path) == origin_record['packaged_sha256'], 'Packaged origin changed')
    require(digest(producer_path) == producer_record['packaged_sha256'], 'Packaged producer changed')
    origin_sha256 = origin_record['source_sha256']
    producer_sha256 = producer_record['source_sha256']
    checked = checked_source_pair(origin_sha256, producer_sha256)
    text = inspect.getsource(module._contact_penetration_summary).rstrip()
    candidate = contiguous_checker_source(text)
    candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
    require(candidate_hash == checked['candidate_sha256'], 'Candidate differs from tested function')
    module._contact_penetration_summary = compile_function(candidate, '_contact_penetration_summary')
    return dict(mode='producer_copy', original_file_sha256=origin_sha256,
        producer_file_sha256=producer_sha256, offline_evidence=checked['evidence'],
        original_function_sha256=hashlib.sha256(text.encode()).hexdigest(),
        candidate_function_sha256=candidate_hash, exact_array_equal_preserved=True,
        physical_thresholds_changed=False, task_success_changed=False,
        source_files_modified=False, writes_physics=False)


def run_case(source_dir, fixture):
    producer_path = source_dir/'physx_ground_truth.py'
    checker_path = source_dir/'recording_origin.py'
    producer = compile_function(extract_function(producer_path,
                                '_canonicalize_contact_fields'), '_canonicalize_contact_fields')
    checker_source = extract_function(checker_path, '_contact_penetration_summary')
    original = compile_function(checker_source, '_contact_penetration_summary')
    candidate_source = contiguous_checker_source(checker_source)
    candidate = compile_function(candidate_source, '_contact_penetration_summary')
    schema = read(fixture.with_suffix('.json'))['schema']
    with np.load(fixture.with_suffix('.npz'), allow_pickle=False) as archive:
        fields = {k: archive[k].copy() for k in archive.files if not k.startswith('checker_')}
    config = schema['physx']['contact_report']
    trace = SimpleNamespace(max_contact_headers=config['max_headers'],
        _contact_report_paths=config['report_enabled_task_body_paths'],
        _contact_report_path_table=config['path_table'],
        _tensor_contact_sensor_paths=config['tensor_sensor_paths'],
        _tensor_contact_filter_paths=config['tensor_filter_paths'])
    recomputed = producer(trace, {k: v.copy() for k, v in fields.items()})
    metric = 'physx_contact_signature_pair_metrics'
    baseline = original(fields, schema)
    fixed = candidate(fields, schema)
    corrupt = {k: v.copy() for k, v in fields.items()}
    corrupt[metric][0, 1] += 0.001
    corrupt_result = candidate(corrupt, schema)
    bad_count = {k: v.copy() for k, v in fields.items()}
    bad_count['physx_contact_signature_pair_counts'][0, 0] += 1
    count_result = candidate(bad_count, schema)
    count = int(fields['physx_contact_signature_pair_count'][0])
    return dict(fixture=str(fixture), fixture_sha256=digest(fixture.with_suffix('.npz')),
        numpy_version=np.__version__, producer_sha256=digest(producer_path),
        checker_sha256=digest(checker_path),
        candidate_function_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(),
        producer_recomputed_exact=bool(np.array_equal(recomputed[metric], fields[metric])),
        producer_max_abs_diff=float(np.max(np.abs(recomputed[metric]-fields[metric]))),
        original_result=baseline, contiguous_candidate_result=fixed,
        corrupt_metrics_result=corrupt_result, corrupt_counts_result=count_result,
        candidate_accepts_same_payload=fixed['valid'] is True,
        still_rejects_corruption=not corrupt_result['valid'] and not count_result['valid'],
        pair_count=count, physics_or_thresholds_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--fixture', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = [run_case(args.source_dir, fixture) for fixture in args.fixture]
    write_new(args.output, dict(cases=rows, all_candidate_pass=all(
        r['candidate_accepts_same_payload'] and r['still_rejects_corruption'] for r in rows)))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
