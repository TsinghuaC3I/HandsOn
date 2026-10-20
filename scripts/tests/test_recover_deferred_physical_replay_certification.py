from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import record_demos_zarr as recorder
from scripts import recover_deferred_physical_replay_certification as recovery


def _pending_demo(
    tmp_path: Path,
    *,
    task: str = "t09_insert_peg",
    recorder_pid: int = 4321,
) -> Path:
    demo = (
        recorder._physical_replay_pending_parent(tmp_path) / f"{task}_demo_1__physx_v3"
    )
    (demo / "replay.zarr").mkdir(parents=True)
    (demo / recorder.PHYSX_V3_MARKER_FILE).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (demo / "collection_meta.json").write_text(
        json.dumps({"collection_session_id": "session-1"}),
        encoding="utf-8",
    )
    (demo / "replay_manifest.json").write_text(
        json.dumps({"recording": {"recorder_pid": recorder_pid}}),
        encoding="utf-8",
    )
    return demo.resolve()


def test_recovery_ignores_stable_v2_pending_data(tmp_path: Path) -> None:
    pending = recorder._physical_replay_pending_parent(tmp_path)
    stable = pending / "bench320_fu006_demo_1__stable_v2"
    (stable / "replay.zarr").mkdir(parents=True)
    (stable / recorder.STABLE_V2_MARKER_FILE).write_text(
        "{}\n",
        encoding="utf-8",
    )

    result = recovery.recover(tmp_path)

    assert result["task_count"] == 0
    assert result["launched_count"] == 0


def test_recovery_cli_skips_pending_from_a_live_direct_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pending_demo(tmp_path, recorder_pid=4321)
    monkeypatch.setattr(
        recorder,
        "_process_command",
        lambda pid: (
            ("python", "/repo/scripts/record_demos_zarr.py") if pid == 4321 else ()
        ),
    )
    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda *_args, **_kwargs: pytest.fail("recovery raced a live collector"),
    )

    result = recovery.recover(tmp_path)

    assert result["launched_count"] == 0
    assert result["live_source_skipped_count"] == 1
    assert result["recovery_exhausted_count"] == 0


def test_one_orphan_recovery_is_allowed_then_becomes_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_demo(tmp_path, recorder_pid=4321)
    recorder._write_json_atomic(
        recorder._deferred_physical_worker_state_path(demo),
        {
            "format": "handson.deferred_physical_replay_worker_state",
            "version": 1,
            "status": "running",
            "pid": 8001,
            "launch_attempt_count": 1,
            "last_launch_at": "2026-08-02T09:00:00+08:00",
        },
    )
    monkeypatch.setattr(recorder, "_process_command", lambda _pid: ())
    monkeypatch.setattr(
        recorder.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=8002),
    )

    first = recovery.recover(tmp_path)
    second = recovery.recover(tmp_path)

    assert first["launched_count"] == 1
    assert first["recovery_exhausted_count"] == 0
    assert second["launched_count"] == 0
    assert second["recovery_exhausted_count"] == 1
    state = json.loads(
        recorder._deferred_physical_worker_state_path(demo).read_text(encoding="utf-8")
    )
    assert state["status"] == "recovery_exhausted"
    assert state["terminal"] is True
    assert state["review_status"] == "review_required"
    assert state["physical_replay_outcome_code"] == (
        recorder.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
    )
    assert state["launch_attempt_count"] == 2
    assert state["recovery_attempt_count"] == 1


def test_legacy_terminal_failure_is_archived_and_gets_one_current_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_demo(tmp_path, recorder_pid=4321)
    state_path = recorder._deferred_physical_worker_state_path(demo)
    old_state = {
        "format": "handson.deferred_physical_replay_worker_state",
        "version": 1,
        "status": "invalid",
        "terminal": True,
        "pid": None,
        "launch_attempt_count": 17,
        "error": "legacy preflight failure",
    }
    recorder._write_json_atomic(state_path, old_state)
    acceptance_path = demo / recorder.FRESH_PROCESS_ACCEPTANCE_NAME
    acceptance_path.write_text(
        json.dumps({"status": "failed", "reason": "legacy failure"}),
        encoding="utf-8",
    )
    report_path = demo / recorder.PHYSICAL_REPLAY_REPORT_NAME
    report_path.write_text(
        json.dumps({"status": "failed", "reason": "old report"}),
        encoding="utf-8",
    )
    runs_dir = demo / recorder.PHYSICAL_REPLAY_RUN_DIRECTORY
    runs_dir.mkdir()
    (runs_dir / "run_001.json").write_text(
        json.dumps({"status": "failed", "reason": "old run"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(recorder, "_process_command", lambda _pid: ())
    monkeypatch.setattr(
        recorder.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=9001),
    )

    result = recovery.recover(tmp_path)

    assert result["launched_count"] == 1
    assert result["legacy_review_migrated_count"] == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "running"
    assert state["review_status"] == "review_required"
    assert state["launch_attempt_count"] == 1
    assert state["previous_launch_attempt_count"] == 17
    assert state["legacy_review_migration"] is True
    archive = state["previous_review_archive"]
    archive_path = Path(archive["archive_path"])
    assert archive_path.is_dir()
    assert (
        json.loads((archive_path / "worker_state.json").read_text(encoding="utf-8"))
        == old_state
    )
    assert (
        json.loads(
            (archive_path / recorder.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(
                encoding="utf-8"
            )
        )["reason"]
        == "legacy failure"
    )
    assert (
        json.loads(
            (
                archive_path / recorder.PHYSICAL_REPLAY_RUN_DIRECTORY / "run_001.json"
            ).read_text(encoding="utf-8")
        )["reason"]
        == "old run"
    )
    manifest = json.loads(
        (archive_path / "history_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["legacy_review_migration"] is True
    assert manifest["previous_launch_attempt_count"] == 17
    assert len(manifest["documents"]) == 4


def test_only_versioned_evidenced_invalid_acceptance_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_demo(tmp_path, recorder_pid=4321)
    proof_dir = demo / recorder.PHYSICAL_REPLAY_RUN_DIRECTORY
    proof_dir.mkdir()
    proof_path = proof_dir / "run_001.json"
    proof_path.write_text(
        json.dumps({"status": "failed", "reason": "physical_state_diverged"}),
        encoding="utf-8",
    )
    proof_sha256 = recorder._sha256_regular_file(proof_path)
    (demo / recorder.FRESH_PROCESS_ACCEPTANCE_NAME).write_text(
        json.dumps(
            {
                "format": recorder.DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT,
                "version": recorder.DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION,
                "mode": recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
                "status": "failed",
                "physical_replay_outcome_version": 1,
                "physical_replay_outcome_code": (
                    recorder.PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
                ),
                "review_status": "invalid",
                "certification_runs": [
                    {
                        "run": 1,
                        "report": ("physical_replay_certification_runs/run_001.json"),
                        "report_sha256": proof_sha256,
                    }
                ],
                "outcome_evidence": {
                    "format": ("handson.physical_replay_invalid_outcome_evidence"),
                    "version": 1,
                    "run": 1,
                    "report": ("physical_replay_certification_runs/run_001.json"),
                    "report_sha256": proof_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "versioned invalid outcome must not be requeued"
        ),
    )

    assert recorder._has_final_failed_deferred_physical_proof(demo)

    result = recovery.recover(tmp_path)

    assert result["launched_count"] == 0
    assert result["legacy_review_migrated_count"] == 0
    recorder._write_json_atomic(
        recorder._deferred_physical_worker_state_path(demo),
        {
            "status": "invalid",
            "terminal": True,
            "launch_attempt_count": 2,
            "physical_replay_outcome_version": 1,
            "physical_replay_outcome_code": (
                recorder.PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
            ),
            "review_status": "invalid",
            "outcome_evidence": {"report_sha256": proof_sha256},
        },
    )
    proof_path.write_text("{}", encoding="utf-8")
    assert not recorder._has_final_failed_deferred_physical_proof(demo)
    assert recorder._deferred_physical_launch_attempt_count(demo) == 0


@pytest.mark.parametrize(
    "partial_state",
    [
        {
            "status": "invalid",
            "terminal": True,
            "physical_replay_outcome_version": 1,
        },
        {
            "status": "invalid",
            "terminal": True,
            "physical_replay_outcome_version": 1,
            "physical_replay_outcome_code": "physical_replay_invalid_v1",
            "review_status": "invalid",
        },
    ],
)
def test_partial_new_invalid_state_is_still_a_legacy_review(
    partial_state: dict[str, object],
) -> None:
    assert recorder._legacy_deferred_physical_state_requires_review(partial_state)

    current_review = {
        **partial_state,
        "physical_replay_outcome_code": (
            recorder.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        ),
        "review_status": "review_required",
    }
    assert not recorder._legacy_deferred_physical_state_requires_review(current_review)


def test_absolute_reviewer_worker_override_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pending_demo(tmp_path, recorder_pid=4321)
    worker_script = (
        tmp_path
        / "reviewer-worktree"
        / "scripts"
        / "deferred_physical_replay_certification_worker.py"
    )
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text("# reviewer worker\n", encoding="utf-8")
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV,
        str(worker_script),
    )
    monkeypatch.setattr(recorder, "_process_command", lambda _pid: ())
    commands: list[list[str]] = []

    def popen(command: list[str], **_kwargs):
        commands.append(command)
        return SimpleNamespace(pid=9010)

    monkeypatch.setattr(recorder.subprocess, "Popen", popen)

    result = recovery.recover(tmp_path)

    assert result["launched_count"] == 1
    assert commands[0][1] == str(worker_script.resolve())


def test_reviewer_worker_override_rejects_unsafe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = recorder.DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV
    monkeypatch.setenv(
        env_name,
        "scripts/deferred_physical_replay_certification_worker.py",
    )
    with pytest.raises(RuntimeError, match="must be absolute"):
        recorder._deferred_physical_replay_worker_script()

    wrong_name = tmp_path / "worker.py"
    wrong_name.write_text("# wrong name\n", encoding="utf-8")
    monkeypatch.setenv(env_name, str(wrong_name))
    with pytest.raises(RuntimeError, match="basename must be"):
        recorder._deferred_physical_replay_worker_script()

    real_worker = tmp_path / "real" / "deferred_physical_replay_certification_worker.py"
    real_worker.parent.mkdir()
    real_worker.write_text("# real\n", encoding="utf-8")
    linked_worker = (
        tmp_path / "linked" / "deferred_physical_replay_certification_worker.py"
    )
    linked_worker.parent.mkdir()
    linked_worker.symlink_to(real_worker)
    monkeypatch.setenv(env_name, str(linked_worker))
    with pytest.raises(RuntimeError, match="must not traverse a symlink"):
        recorder._deferred_physical_replay_worker_script()


def test_recovery_cli_stdout_is_one_machine_readable_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        recovery,
        "recover",
        lambda *_args, **_kwargs: {
            "format": "handson.deferred_physical_replay_recovery",
            "version": 1,
            "status": "complete",
            "launched_count": 2,
        },
    )

    assert recovery.main([f"--out-root={tmp_path}"]) == 0

    output = capsys.readouterr().out
    assert json.loads(output)["launched_count"] == 2
    assert output.count("\n") == 1
