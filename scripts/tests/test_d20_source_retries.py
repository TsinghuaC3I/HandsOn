"""Execute recorder integration blocks without launching or importing Kit."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from handson.data import d20_stage_saves as saves
from scripts import record_demos_zarr as recorder


def main_ast():
    tree = ast.parse(Path(recorder.__file__).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def harness(nodes, namespace, *, arguments=''):
    function = ast.parse(f'def run({arguments}):\n    pass').body[0]
    function.body = nodes
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 recorder.__file__, 'exec'), namespace)
    return namespace['run']


def test_recorder_retry_wrapper_updates_source_before_every_full_reset(monkeypatch):
    bad, good = SimpleNamespace(path=Path('bad')), SimpleNamespace(path=Path('good'))
    tried, tags, statuses, published, modes = [], [], [], [], []
    monkeypatch.setattr(saves, 'mark_continuation_rejected', lambda source, *a: tags.append(source))
    monkeypatch.setattr(saves, 'select_source', lambda *a, **kw: (modes.append(kw['source_mode']), good)[1])
    def reset_once():
        tried.append(current_source())
        if current_source() is bad:
            raise saves.ContinuationRejected('source-state rejection')
        return {'obs': True}, {'lineage': current_source()}
    wrapper = next(n for n in main_ast().body if isinstance(n, ast.FunctionDef) and n.name == '_reset_environment')
    namespace = dict(with_source_retries=saves.with_source_retries,
                     d20_source_mode='deferred',
                     _publish_d20_source=lambda task, source, phase: published.append((source, phase)),
                     _reset_environment_once=reset_once, FLAGS=SimpleNamespace(out_dir='/unused'),
                     task_id='d20_2_retrieve_and_plate',
                     _write_isaac_worker_status=lambda *a, **kw: statuses.append((a, kw)),
                     _write_collection_runtime_status=lambda *a, **kw: statuses.append((a, kw)))
    run = harness([wrapper, ast.parse('return _reset_environment, lambda: d20_source').body[0]],
                  namespace, arguments='d20_source')
    reset, current_source = run(bad)
    assert reset() == ({'obs': True}, {'lineage': good})
    assert tried == [bad, good] and tags == [bad] and current_source() is good
    assert any(a[0] == 'recovering' for a, _ in statuses)
    assert modes == ['deferred']
    assert published == [(bad, 'attempting'), (good, 'attempting'), (good, 'ready')]
    # Ordinary tasks keep the same one-reset path.
    reset, current_source = run(None)
    assert reset() == ({'obs': True}, {'lineage': None})


def test_initial_source_exhaustion_returns_before_simulator_creation():
    block = next(n for n in main_ast().body if isinstance(n, ast.Try)
                 and 'd20_source = select_source' in ast.unparse(n.body))
    reports, launched = [], []
    def exhausted(*a, **kw):
        raise saves.NoUsableSource('all rejected')
    namespace = dict(NoUsableSource=saves.NoUsableSource, select_source=exhausted, d20_source_mode='deferred',
                     _report_d20_sources_exhausted=lambda *a: reports.append(a),
                     FLAGS=SimpleNamespace(out_dir='/unused'), task_id='d20_2_retrieve_and_plate',
                     os=SimpleNamespace(environ={}), launch=lambda: launched.append(True))
    harness([block, *ast.parse('launch()').body], namespace)()
    assert len(reports) == 1 and launched == []


@pytest.mark.parametrize('journal_error', [False, True])
def test_startup_runtime_exhaustion_preserves_evidence_then_closes(journal_error):
    block = next(n for n in main_ast().body if isinstance(n, ast.Try)
                 and ast.unparse(n.body) == 'obs, info = _reset_environment()')
    events = []
    def exhausted():
        raise saves.NoUsableSource('all runtime-rejected')
    def finalize(paths):
        assert paths == []
        events.append('journal')
        if journal_error:
            raise RuntimeError('unsealed startup evidence')
    namespace = dict(NoUsableSource=saves.NoUsableSource, _reset_environment=exhausted,
                     _report_d20_sources_exhausted=lambda *a: events.append('status'),
                     physx_session_journal=SimpleNamespace(finalize=finalize),
                     _enable_isaac_fast_shutdown=lambda e: events.append('fast_shutdown'),
                     env=SimpleNamespace(close=lambda: events.append('close')),
                     FLAGS=SimpleNamespace(sim_backend='isaacsim'), task_id='d20_2_retrieve_and_plate')
    harness([block], namespace)()
    assert events == ['status', 'journal', 'fast_shutdown', 'close']


def test_exhaustion_status_is_stopped_not_failed(monkeypatch):
    events = []
    monkeypatch.setattr(recorder, '_write_isaac_worker_status', lambda *a, **kw: events.append((a, kw)))
    monkeypatch.setattr(recorder, '_write_collection_runtime_status', lambda *a, **kw: events.append((a, kw)))
    recorder._report_d20_sources_exhausted('d20_3_finish_table_setting', saves.NoUsableSource('none left'))
    assert len(events) == 2
    assert all(a[0] == 'stopped' and kw['code'] == 'd20_sources_exhausted' for a, kw in events)


def test_worker_status_is_dedicated_and_bound_to_session_mode(tmp_path, monkeypatch):
    path = tmp_path / 'worker-source.json'
    monkeypatch.setenv('HANDSON_D20_SOURCE_STATUS_FILE', str(path))
    monkeypatch.setenv('HANDSON_COLLECTION_SESSION_ID', 'live-session')
    monkeypatch.setenv('HANDSON_D20_SOURCE_MODE', 'deferred')
    source = SimpleNamespace(reference=lambda: {'episode_id': 'actual-source', 'manifest_sha256': 'hash'})
    recorder._publish_d20_source('d20_2_retrieve_and_plate', source, 'ready')
    status = saves.read_source_status(path, session_id='live-session', task_name='d20_2_retrieve_and_plate')
    assert status['source'] == source.reference()
    assert status['worker_pid'] == recorder.os.getpid() and status['source_mode'] == 'deferred'
    monkeypatch.setattr(recorder, '_write_isaac_worker_status', lambda *a, **kw: None)
    monkeypatch.setattr(recorder, '_write_collection_runtime_status', lambda *a, **kw: None)
    recorder._report_d20_sources_exhausted('d20_2_retrieve_and_plate', saves.NoUsableSource('empty'))
    status = saves.read_source_status(path, session_id='live-session', task_name='d20_2_retrieve_and_plate')
    assert status['source'] is None and status['phase'] == 'exhausted'


def test_post_success_selection_and_consumption_preserve_queue_mode():
    tree = main_ast()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    selections = [n for n in calls if n.func.id in {'select_source', 'with_source_retries'}]
    assert len(selections) == 3
    assert all(any(kw.arg == 'source_mode' and ast.unparse(kw.value) == 'd20_source_mode'
                   for kw in n.keywords) for n in selections)
    consumed = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'd20_source is not None'
                    and 'mark_stitched' in ast.unparse(n))
    events = []
    harness([consumed], dict(d20_source='current', task_id='task', demo_dir='success',
                            mark_stitched=lambda *a, **kw: events.append('claim'),
                            _publish_d20_source=lambda *a: events.append(a[-1])))()
    assert events == ['claim', 'consumed']
