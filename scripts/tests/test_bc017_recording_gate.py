from __future__ import annotations

import inspect

from scripts import record_demos_zarr as recorder


def test_bc017_recorder_has_no_task_specific_gate() -> None:
    source = inspect.getsource(recorder.main)
    assert "evaluate_bc017_collection_gate" not in source
    assert "BC017_ACCEPTANCE_CANDIDATE" not in source
    assert "acceptance_candidate" not in source


def test_bc017_recorder_does_not_import_acceptance_gate_helpers() -> None:
    assert not hasattr(recorder, "evaluate_bc017_collection_gate")
    assert not hasattr(recorder, "_BC017_ACCEPTANCE_CANDIDATE_CONTEXT")
