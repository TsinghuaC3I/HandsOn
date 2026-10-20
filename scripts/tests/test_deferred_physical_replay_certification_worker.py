from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from scripts import deferred_physical_replay_certification_worker as worker


FAKE_CERTIFIER = r"""#!/usr/bin/env python3
import argparse
import json
import os
import signal
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--demo-dir", action="append", required=True, type=Path)
args = parser.parse_args()
for demo_dir in args.demo_dir:
    payload = {
        "status": "passed",
        "reason": "",
        "certified_inputs": {"certifier_pid": os.getpid()},
        "observed_environment": {
            "recorded": os.environ.get("HANDSON_RECORDED"),
            "stale": os.environ.get("HANDSON_STALE"),
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "fresh_worker": os.environ.get(
                "HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER"
            ),
        },
    }
    (demo_dir / "physical_replay_certification.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
if os.environ.get("FAKE_CERTIFIER_SIGNAL") == "1":
    os.kill(os.getpid(), signal.SIGTERM)
raise SystemExit(int(os.environ.get("FAKE_CERTIFIER_EXIT_CODE", "0")))
"""

NO_REPORT_CERTIFIER = r"""#!/usr/bin/env python3
raise SystemExit(86)
"""

CUDA_700_THEN_PASS_CERTIFIER = r"""#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--demo-dir", action="append", required=True, type=Path)
args = parser.parse_args()
counter_path = Path(os.environ["HANDSON_TEST_LAUNCH_COUNTER"])
attempt = int(counter_path.read_text(encoding="utf-8") or "0") + 1
counter_path.write_text(str(attempt), encoding="utf-8")
if attempt == 1:
    print("PhysX error: Fetching GPU Narrowphase failed! 700", flush=True)
    raise SystemExit(0)
for demo_dir in args.demo_dir:
    payload = {
        "status": "passed",
        "reason": "",
        "certified_inputs": {
            "certifier_pid": os.getpid(),
            "device": os.environ.get("HANDSON_ISAACLAB_DEVICE"),
        },
    }
    (demo_dir / "physical_replay_certification.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
"""

CUDA_700_ALWAYS_CERTIFIER = r"""#!/usr/bin/env python3
import os
from pathlib import Path

counter_path = Path(os.environ["HANDSON_TEST_LAUNCH_COUNTER"])
attempt = int(counter_path.read_text(encoding="utf-8") or "0") + 1
counter_path.write_text(str(attempt), encoding="utf-8")
print("CUDA error 700: cudaErrorIllegalAddress", flush=True)
raise SystemExit(0)
"""


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _demo_dir(
    tmp_path: Path,
    *,
    name: str = "b03_demo",
    session_id: str = "session-1",
    episode_index: int = 1,
    reset_ordinal: int = 0,
    variables: dict[str, str] | None = None,
    recording: dict[str, object] | None = None,
) -> Path:
    demo = tmp_path / worker.PENDING_DIRECTORY_NAME / name
    (demo / "replay.zarr").mkdir(parents=True)
    recording_payload: dict[str, object] = {
        "episode_index": episode_index,
        "episode_reset_ordinal": reset_ordinal,
        "collection_run_seed": 100,
        "recorder_lifecycle": "persistent_process",
        "recorder_pid": os.getpid(),
        "source_process_reuse_enabled": True,
        "fresh_scene_replay_required": True,
        "fresh_certifier_process_per_run_required": True,
        "physical_replay_certification_scope": (worker.SESSION_CERTIFICATION_SCOPE),
        "physical_replay_session_history_pending": False,
        "physical_replay_session_history_id": session_id,
        "physical_replay_session_history_root_relative_path": (
            ".physical_replay_pending/.physx_session_history/session-1"
        ),
        "physical_replay_session_history_sha256": "a" * 64,
        "physical_replay_session_history_event_count": 16,
        "physical_replay_session_history_accepted_event_ordinal": (reset_ordinal + 1),
        "physical_replay_session_history_certified_through_event_ordinal": 15,
    }
    recording_payload.update(recording or {})
    (demo / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": "t09_insert_peg"},
                "recording": recording_payload,
                "environment": {"variables": variables or {}},
            }
        ),
        encoding="utf-8",
    )
    (demo / "collection_meta.json").write_text(
        json.dumps(
            {
                "collection_session_id": session_id,
                **{
                    key: value
                    for key, value in recording_payload.items()
                    if key.startswith("physical_replay_")
                },
            }
        ),
        encoding="utf-8",
    )
    return demo


def test_each_fresh_certifier_rebuilds_the_recorded_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(
        tmp_path,
        variables={
            "HANDSON_RECORDED": "manifest-value",
            "OMP_NUM_THREADS": "7",
        },
    )
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "certification_report_blocker",
        lambda *_a, **_k: None,
    )
    monkeypatch.setenv("HANDSON_RECORDED", "wrong-current-value")
    monkeypatch.setenv("HANDSON_STALE", "must-not-leak")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "must-not-leak")
    monkeypatch.setenv("HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER", "1")

    result = worker._run_single_certification(
        demo,
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "passed"
    assert result["timed_out"] is False
    archived_path = demo / result["report"]
    archived_bytes = archived_path.read_bytes()
    assert result["report_sha256"] == hashlib.sha256(archived_bytes).hexdigest()
    archived = json.loads(archived_bytes)
    assert archived["observed_environment"] == {
        "recorded": "manifest-value",
        "stale": None,
        "omp": "7",
        "cublas": None,
        "fresh_worker": None,
    }


def test_certifier_environment_drops_bc011_collection_angle_guidance(
    tmp_path: Path,
) -> None:
    demo = _demo_dir(
        tmp_path,
        variables={worker.BC011_COLLECTION_ANGLE_GUIDANCE_ENV: "1"},
    )

    child_env = worker._certifier_environment(demo)

    assert worker.BC011_COLLECTION_ANGLE_GUIDANCE_ENV not in child_env


def test_crashed_single_certifier_cannot_reuse_previous_run_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(tmp_path)
    stale_report = demo / worker.REPORT_NAME
    stale_report.write_text('{"status":"passed","old":true}', encoding="utf-8")
    stale_proof = (
        demo / worker.FRESH_PROCESS_REPORT_DIRECTORY / "run_001.json"
    )
    stale_proof.parent.mkdir()
    stale_proof.write_text('{"status":"passed","old":true}', encoding="utf-8")
    certifier = _write_executable(
        tmp_path / "no_report_certifier.py",
        NO_REPORT_CERTIFIER,
    )
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "certification_report_blocker",
        lambda *_a, **_k: None,
    )

    result = worker._run_single_certification(
        demo,
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 86
    assert result["report"] == ""
    assert result["report_sha256"] == ""
    assert not stale_report.exists()
    assert not stale_proof.exists()
    assert result["fresh_process_launch_attempts"] == 1
    assert result["discarded_gpu_startup_attempts"] == []


def test_single_cuda_700_before_report_retries_in_fresh_gpu_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = tmp_path / "single-launch-counter.txt"
    counter.write_text("0", encoding="utf-8")
    demo = _demo_dir(
        tmp_path,
        variables={
            "HANDSON_TEST_LAUNCH_COUNTER": str(counter),
            "HANDSON_ISAACLAB_DEVICE": "cuda:0",
        },
    )
    certifier = _write_executable(
        tmp_path / "cuda_700_then_pass.py",
        CUDA_700_THEN_PASS_CERTIFIER,
    )
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "certification_report_blocker",
        lambda *_a, **_k: None,
    )

    result = worker._run_single_certification(
        demo,
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "passed"
    assert counter.read_text(encoding="utf-8") == "2"
    assert result["fresh_process_launch_attempts"] == 2
    assert len(result["discarded_gpu_startup_attempts"]) == 1
    discarded = result["discarded_gpu_startup_attempts"][0]
    assert discarded["classification"] == (
        "explicit_gpu_context_poisoned_before_report"
    )
    assert discarded["pid"] != result["pid"]
    archived = json.loads((demo / result["report"]).read_text(encoding="utf-8"))
    assert archived["certified_inputs"] == {
        "certifier_pid": result["pid"],
        "device": "cuda:0",
    }


def test_single_cuda_700_retry_exhaustion_stays_review_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = tmp_path / "exhausted-launch-counter.txt"
    counter.write_text("0", encoding="utf-8")
    demo = _demo_dir(
        tmp_path,
        variables={
            "HANDSON_TEST_LAUNCH_COUNTER": str(counter),
            "HANDSON_ISAACLAB_DEVICE": "cuda:0",
        },
    )
    certifier = _write_executable(
        tmp_path / "cuda_700_always.py",
        CUDA_700_ALWAYS_CERTIFIER,
    )
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "certification_report_blocker",
        lambda *_a, **_k: None,
    )

    result = worker._run_single_certification(
        demo,
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "failed"
    assert "exhausted 2 GPU-only fresh-process attempts" in result["reason"]
    assert counter.read_text(encoding="utf-8") == "2"
    assert result["fresh_process_launch_attempts"] == 2
    assert len(result["discarded_gpu_startup_attempts"]) == 1
    assert result["report"] == ""
    assert result["report_sha256"] == ""
    outcome = worker._review_outcome_from_runs(
        [result],
        physically_valid=False,
    )
    assert outcome["physical_replay_outcome_code"] == (
        worker.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
    )
    assert outcome["review_status"] == worker.PHYSICAL_REPLAY_REVIEW_REQUIRED


def test_three_runs_use_distinct_fresh_pids_and_mark_process_lifecycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_recorder_pid = os.getpid()
    demo = _demo_dir(
        tmp_path,
        recording={
            "recorder_lifecycle": "persistent_process",
            "recorder_pid": source_recorder_pid,
            "source_process_reuse_enabled": True,
            "fresh_scene_replay_required": True,
            "fresh_certifier_process_per_run_required": True,
        },
    )
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "certification_report_blocker",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        worker,
        "capture_certified_inputs",
        lambda _path: {"tree": "sha"},
    )
    monkeypatch.setattr(
        worker,
        "deferred_physical_replay_acceptance_upload_blocker",
        lambda _path, **_kwargs: None,
    )

    result = worker.certify_demo_dir(
        demo,
        exp_name="t09_insert_peg",
        certification_runs_required=3,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "passed"
    pids = [run["pid"] for run in result["runs"]]
    assert len(pids) == 3
    assert len(set(pids)) == 3
    published_demo = tmp_path / demo.name
    assert result["published_demo_dir"] == str(published_demo)
    assert published_demo.is_dir()
    assert not demo.exists()
    acceptance = json.loads(
        (published_demo / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["certifier_pids"] == pids
    assert acceptance["persistent_source_process_for_collection_batch"] is True
    assert acceptance["fresh_source_process_per_episode"] is False
    assert acceptance["fresh_certifier_process_per_run"] is True
    assert acceptance["recorder_lifecycle"] == "persistent_process"
    assert acceptance["fresh_scene_replay_required"] is True
    assert acceptance["source_recorder_pid"] == source_recorder_pid
    assert acceptance["recorder_pid"] == source_recorder_pid
    assert source_recorder_pid not in pids
    assert "persistent_source_process_across_episodes" not in acceptance
    assert "source_process_reused_across_episodes" not in acceptance
    assert acceptance["fresh_certifier_process_per_run_required"] is True


def test_one_fresh_subprocess_receives_every_demo_in_session_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=4,
    )
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "local_physical_replay_report_blocker",
        lambda *_a, **_k: None,
    )

    result = worker._run_session_certification(
        [first, second],
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "passed"
    assert [row["demo_name"] for row in result["episodes"]] == [
        first.name,
        second.name,
    ]
    assert (
        len({row["certified_inputs"]["certifier_pid"] for row in result["episodes"]})
        == 1
    )
    assert all(
        (demo / "physical_replay_certification_runs" / "run_001.json").is_file()
        for demo in (first, second)
    )


def test_crashed_session_certifier_cannot_reuse_previous_run_proofs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=4,
    )
    stale_paths: list[Path] = []
    for demo in (first, second):
        latest = demo / worker.REPORT_NAME
        latest.write_text('{"status":"passed","old":true}', encoding="utf-8")
        proof = demo / worker.FRESH_PROCESS_REPORT_DIRECTORY / "run_001.json"
        proof.parent.mkdir()
        proof.write_text('{"status":"passed","old":true}', encoding="utf-8")
        stale_paths.extend((latest, proof))
    certifier = _write_executable(
        tmp_path / "no_report_certifier.py",
        NO_REPORT_CERTIFIER,
    )
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "local_physical_replay_report_blocker",
        lambda *_a, **_k: None,
    )

    result = worker._run_session_certification(
        [first, second],
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 86
    assert all(row["report"] == "" for row in result["episodes"])
    assert all(row["report_sha256"] == "" for row in result["episodes"])
    assert all(not path.exists() for path in stale_paths)
    assert result["fresh_process_launch_attempts"] == 1
    assert result["discarded_gpu_startup_attempts"] == []


def test_session_cuda_700_before_reports_retries_one_shared_gpu_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = tmp_path / "session-launch-counter.txt"
    counter.write_text("0", encoding="utf-8")
    variables = {
        "HANDSON_TEST_LAUNCH_COUNTER": str(counter),
        "HANDSON_ISAACLAB_DEVICE": "cuda:0",
    }
    first = _demo_dir(
        tmp_path,
        episode_index=1,
        reset_ordinal=0,
        variables=variables,
    )
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=4,
        variables=variables,
    )
    certifier = _write_executable(
        tmp_path / "session_cuda_700_then_pass.py",
        CUDA_700_THEN_PASS_CERTIFIER,
    )
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "local_physical_replay_report_blocker",
        lambda *_a, **_k: None,
    )

    result = worker._run_session_certification(
        [first, second],
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "passed"
    assert counter.read_text(encoding="utf-8") == "2"
    assert result["fresh_process_launch_attempts"] == 2
    assert len(result["discarded_gpu_startup_attempts"]) == 1
    assert result["discarded_gpu_startup_attempts"][0]["pid"] != result["pid"]
    assert len(result["episodes"]) == 2
    assert all(
        row["certified_inputs"] == {
            "certifier_pid": result["pid"],
            "device": "cuda:0",
        }
        for row in result["episodes"]
    )


def test_session_report_blocker_reason_overrides_passing_report_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(tmp_path)
    certifier = _write_executable(
        tmp_path / "fake_certifier.py",
        FAKE_CERTIFIER.replace(
            '"reason": "",',
            '"reason": "continuous_physx_replay_matched",',
        ),
    )
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "local_physical_replay_report_blocker",
        lambda *_a, **_k: {
            "code": "local_report_invalid",
            "detail": "synthetic local blocker",
        },
    )

    result = worker._run_session_certification(
        [demo],
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "failed"
    assert result["reason"] == f"{demo.name}: synthetic local blocker"
    assert result["episodes"][0]["reason"] == "synthetic local blocker"


def test_session_uses_three_processes_total_and_publishes_all_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=4,
    )
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "local_physical_replay_report_blocker",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        worker,
        "capture_certified_inputs",
        lambda path: {"tree": Path(path).name},
    )
    monkeypatch.setattr(
        worker,
        "deferred_physical_replay_acceptance_upload_blocker",
        lambda _path, **_kwargs: None,
    )

    result = worker.certify_session(
        [second, first],
        exp_name="t09_insert_peg",
        certification_runs_required=3,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == "passed"
    assert result["physical_certification_status"] == "passed"
    assert result["physical_certification_runs_passed"] == 3
    assert result["publication_status"] == "published"
    assert result["failure_stage"] is None
    assert len(result["runs"]) == 3
    pids = [run["pid"] for run in result["runs"]]
    assert len(set(pids)) == 3
    assert all(len(run["episodes"]) == 2 for run in result["runs"])
    published = [tmp_path / first.name, tmp_path / second.name]
    assert result["published_demo_dirs"] == [str(path) for path in published]
    assert all(path.is_dir() for path in published)
    acceptances = [
        json.loads(
            (path / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
        )
        for path in published
    ]
    assert [acceptance["certifier_pids"] for acceptance in acceptances] == [
        pids,
        pids,
    ]
    assert [acceptance["session_episode_index"] for acceptance in acceptances] == [1, 2]
    assert all(
        acceptance["certification_scope"] == worker.SESSION_CERTIFICATION_SCOPE
        for acceptance in acceptances
    )
    assert all(
        acceptance["session_episode_reset_ordinals"] == [0, 4]
        for acceptance in acceptances
    )
    assert all(
        acceptance["session_publication_all_targets_prevalidated"] is True
        and acceptance["session_publication_rollback_on_catchable_error"] is True
        for acceptance in acceptances
    )
    assert all(
        acceptance["publication_commit"]
        == "ordered_directory_renames_then_atomic_session_marker"
        and acceptance["publication_status"] == "committed_by_atomic_session_marker"
        for acceptance in acceptances
    )
    commit_paths = {
        acceptance["session_publication_commit_root_relative_path"]
        for acceptance in acceptances
    }
    assert len(commit_paths) == 1
    commit = json.loads((tmp_path / commit_paths.pop()).read_text(encoding="utf-8"))
    assert commit["format"] == worker.SESSION_PUBLICATION_COMMIT_FORMAT
    assert commit["status"] == "committed"
    assert commit["session_demo_names"] == [first.name, second.name]
    assert commit["session_episode_reset_ordinals"] == [0, 4]
    assert [member["demo_name"] for member in commit["members"]] == [
        first.name,
        second.name,
    ]
    assert [member["acceptance_sha256"] for member in commit["members"]] == [
        worker._sha256_file(path / worker.FRESH_PROCESS_ACCEPTANCE_NAME)
        for path in published
    ]


def test_session_validation_uses_reset_ordinal_and_rejects_mixed_sessions(
    tmp_path: Path,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=1)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=5,
    )

    context = worker._validated_session_context(
        [second, first],
        exp_name="t09_insert_peg",
    )

    assert context["demo_dirs"] == [first.resolve(), second.resolve()]
    assert context["episode_reset_ordinals"] == [1, 5]

    mixed = _demo_dir(
        tmp_path,
        name="b03_demo_3",
        session_id="another-session",
        episode_index=3,
        reset_ordinal=6,
    )
    with pytest.raises(RuntimeError, match="mixes collection sessions"):
        worker._validated_session_context(
            [first, second, mixed],
            exp_name="t09_insert_peg",
        )


def test_one_failed_session_run_marks_every_member_failed_and_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=2,
    )

    def fail_session(demos: list[Path], run_index: int, **_kwargs):
        return {
            "run": run_index,
            "pid": os.getpid() + 100,
            "returncode": 1,
            "started_at": "2026-08-02T01:00:00+00:00",
            "finished_at": "2026-08-02T01:00:01+00:00",
            "status": "failed",
            "reason": "second episode diverged",
            "timed_out": False,
            "episodes": [
                {
                    "demo_name": demo.name,
                    "status": "passed" if index == 1 else "failed",
                    "reason": "" if index == 1 else "diverged",
                    "report": (
                        f"physical_replay_certification_runs/run_{run_index:03d}.json"
                    ),
                    "report_sha256": "a" * 64,
                    "certified_inputs": {"tree": demo.name},
                }
                for index, demo in enumerate(demos, start=1)
            ],
        }

    monkeypatch.setattr(worker, "_run_session_certification", fail_session)
    monkeypatch.setattr(
        worker,
        "capture_certified_inputs",
        lambda path: {"tree": Path(path).name},
    )

    result = worker.certify_session(
        [first, second],
        exp_name="t09_insert_peg",
        certification_runs_required=3,
    )

    assert result["status"] == "failed"
    assert result["physical_certification_status"] == "failed"
    assert result["physical_certification_runs_passed"] == 0
    assert result["publication_status"] == "not_attempted"
    assert result["failure_stage"] == "physical_certification"
    assert len(result["runs"]) == 1
    assert first.is_dir() and second.is_dir()
    assert not (tmp_path / first.name).exists()
    assert not (tmp_path / second.name).exists()
    for demo in (first, second):
        acceptance = json.loads(
            (demo / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
        )
        assert acceptance["status"] == "failed"
        assert acceptance["published_after_post_collection_certification"] is False
        assert acceptance["reason"] == "second episode diverged"
        assert acceptance["review_status"] == "review_required"
        assert acceptance["physical_replay_outcome_code"] == (
            worker.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        )
        assert "outcome_evidence" not in acceptance


def _completed_invalid_report_projection(
    *,
    reason: str = "physical_state_diverged",
) -> dict[str, object]:
    projection: dict[str, object] = {
        "run": 1,
        "report": "physical_replay_certification_runs/run_001.json",
        "report_sha256": "a" * 64,
        "report_format": worker.REPORT_FORMAT,
        "report_version": worker.REPORT_VERSION,
        "physical_report_status": "failed",
        "physical_report_reason": reason,
        "error_type": None,
        "certification_mode": worker.CERTIFICATION_MODE,
        "execution_mode": worker.PHYSICS_ONLY_EXECUTION_MODE,
        "steps_expected": 24,
        "steps_executed": 24,
        "task_success_replay": False,
        "final_succeed": False,
        "trace_match": False,
        "transition_outcomes_match": True,
        "first_divergence": {"action_index": 7, "field": "object_qpos"},
        "first_outcome_mismatch": None,
        "portable": True,
        "source_identity": "clean_commit_match",
        "source_snapshot_commit": None,
        "source_reviewer_upgrade_paths": None,
        worker.CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD: None,
        "timed_out": False,
    }
    if reason == "transition_outcome_diverged":
        projection.update(
            {
                "trace_match": True,
                "transition_outcomes_match": False,
                "first_divergence": None,
                "first_outcome_mismatch": {
                    "action_index": 11,
                    "expected_succeed": True,
                    "actual_succeed": False,
                },
            }
        )
    elif reason == "final_info_succeed_is_false":
        projection.update(
            {
                "trace_match": True,
                "transition_outcomes_match": True,
                "first_divergence": None,
                "first_outcome_mismatch": None,
            }
        )
    return projection


@pytest.mark.parametrize(
    "reason",
    [
        "physical_state_diverged",
        "transition_outcome_diverged",
        "final_info_succeed_is_false",
    ],
)
def test_only_completed_replay_with_explicit_evidence_is_invalid(
    reason: str,
) -> None:
    run = _completed_invalid_report_projection(reason=reason)

    outcome = worker._review_outcome_from_runs([run], physically_valid=False)

    assert outcome["review_status"] == "invalid"
    assert outcome["physical_replay_outcome_code"] == (
        worker.PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
    )
    assert outcome["outcome_evidence"]["report_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps_executed", 23),
        ("error_type", "CertificationError"),
        ("timed_out", True),
        ("report_format", "legacy.report"),
        ("physical_report_reason", "replay_contract_is_diagnostic_only"),
        ("first_divergence", {"action_index": -1}),
        ("portable", False),
        ("source_identity", "dirty_content_match"),
    ],
)
def test_preflight_incomplete_or_contract_failure_requires_review(
    field: str,
    value: object,
) -> None:
    run = _completed_invalid_report_projection()
    run[field] = value

    outcome = worker._review_outcome_from_runs([run], physically_valid=False)

    assert outcome == {
        "physical_replay_outcome_version": 1,
        "physical_replay_outcome_code": (
            worker.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        ),
        "review_status": "review_required",
    }


def test_invalid_evidence_binds_read_only_contact_overlay_digest() -> None:
    run = _completed_invalid_report_projection()
    overlay_payload = {
        "format": "handson.physx_contact_signature_read_only_overlay",
        "version": 1,
        "algorithm": "rigid_contact_tensor_exact_zero_padding_v1",
        "source_open_mode": "read_only",
        "source_files_mutated": False,
        "segments": [{"segment_id": "episode-1"}],
        "segment_count": 1,
        "signature_frames_reconstructed": 3,
        "exact_zero_padding_rows_removed": 3,
    }
    overlay = {
        **overlay_payload,
        "digest_sha256": hashlib.sha256(
            json.dumps(
                overlay_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    run.update(
        {
            "source_identity": "clean_recording_reviewer_upgrade",
            "source_snapshot_commit": "a" * 40,
            "source_reviewer_upgrade_paths": [
                "scripts/certify_physx_replay.py",
                "handson/handson/sim/isaaclab/physx_ground_truth.py",
            ],
            worker.CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD: overlay,
        }
    )

    outcome = worker._review_outcome_from_runs([run], physically_valid=False)

    evidence = outcome["outcome_evidence"]
    assert (
        evidence["contact_signature_overlay_digest_sha256"]
        == (overlay["digest_sha256"])
    )
    assert evidence[worker.CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD] == overlay

    overlay["digest_sha256"] = "0" * 64
    blocked = worker._review_outcome_from_runs([run], physically_valid=False)
    assert blocked["review_status"] == "review_required"


def test_session_marks_only_evidenced_episode_invalid_in_acceptance_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=2,
    )
    invalid_projection = _completed_invalid_report_projection()

    def fail_second(demos: list[Path], run_index: int, **_kwargs):
        return {
            "run": run_index,
            "pid": os.getpid() + 100,
            "returncode": 1,
            "started_at": "2026-08-02T01:00:00+00:00",
            "finished_at": "2026-08-02T01:00:01+00:00",
            "status": "failed",
            "reason": "second episode diverged",
            "timed_out": False,
            "episodes": [
                {
                    "demo_name": first.name,
                    "status": "passed",
                    "reason": "",
                    "report": "physical_replay_certification_runs/run_001.json",
                    "report_sha256": "b" * 64,
                },
                {
                    "demo_name": second.name,
                    "status": "failed",
                    "reason": "physical_state_diverged",
                    **invalid_projection,
                },
            ],
        }

    monkeypatch.setattr(worker, "_run_session_certification", fail_second)
    monkeypatch.setattr(
        worker,
        "capture_certified_inputs",
        lambda path: {"tree": Path(path).name},
    )

    result = worker.certify_session(
        [first, second],
        exp_name="t09_insert_peg",
        certification_runs_required=3,
    )

    outcomes = {row["demo_name"]: row for row in result["episode_outcomes"]}
    assert outcomes[first.name]["review_status"] == "review_required"
    assert outcomes[second.name]["review_status"] == "invalid"
    for demo, expected in ((first, "review_required"), (second, "invalid")):
        acceptance = json.loads(
            (demo / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
        )
        assert acceptance["review_status"] == expected

    worker._write_worker_terminal_states(
        [first, second], status="failed", result=result, error=""
    )
    first_state = json.loads(
        worker._worker_state_path(first).read_text(encoding="utf-8")
    )
    second_state = json.loads(
        worker._worker_state_path(second).read_text(encoding="utf-8")
    )
    assert first_state["status"] == "review_required"
    assert first_state["terminal"] is False
    assert second_state["status"] == "invalid"
    assert second_state["terminal"] is True
    assert second_state["physical_replay_outcome_code"] == (
        worker.PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
    )
    assert second_state["outcome_evidence"]["steps_executed"] == 24


def test_publication_gate_block_preserves_three_pass_physical_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(tmp_path)
    pids = iter((os.getpid() + 101, os.getpid() + 102, os.getpid() + 103))

    def pass_session(demos: list[Path], run_index: int, **_kwargs):
        pid = next(pids)
        return {
            "run": run_index,
            "pid": pid,
            "returncode": 0,
            "started_at": "2026-08-02T01:00:00+00:00",
            "finished_at": "2026-08-02T01:00:01+00:00",
            "status": "passed",
            "reason": "",
            "timed_out": False,
            "episodes": [
                {
                    "demo_name": demos[0].name,
                    "status": "passed",
                    "reason": "",
                    "report": (
                        f"physical_replay_certification_runs/run_{run_index:03d}.json"
                    ),
                    "report_sha256": "a" * 64,
                    "certified_inputs": {"tree": demos[0].name},
                }
            ],
        }

    monkeypatch.setattr(worker, "_run_session_certification", pass_session)
    monkeypatch.setattr(
        worker,
        "capture_certified_inputs",
        lambda path: {"tree": Path(path).name},
    )
    monkeypatch.setattr(
        worker,
        "deferred_physical_replay_acceptance_upload_blocker",
        lambda _path, **_kwargs: {
            "code": "physical_replay_non_portable",
            "detail": "portable must be true",
        },
    )

    result = worker.certify_session(
        [demo],
        exp_name="t09_insert_peg",
        certification_runs_required=3,
    )

    assert result["status"] == "failed"
    assert result["physical_certification_status"] == "passed"
    assert result["physical_certification_runs_passed"] == 3
    assert result["publication_status"] == "blocked"
    assert result["failure_stage"] == "publication_gate"
    assert result["reason"] == "portable must be true"
    assert demo.is_dir()
    assert not (tmp_path / demo.name).exists()
    acceptance = json.loads(
        (demo / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
    )
    assert acceptance["review_status"] == "review_required"
    assert acceptance["physical_replay_outcome_code"] == (
        worker.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
    )
    assert "outcome_evidence" not in acceptance


@pytest.mark.parametrize(
    ("exit_code", "terminated", "expected_status"),
    [(17, False, "failed"), (0, True, "failed")],
)
def test_valid_report_still_requires_a_clean_certifier_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    terminated: bool,
    expected_status: str,
) -> None:
    demo = _demo_dir(tmp_path)
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    monkeypatch.setattr(worker, "CERTIFIER", certifier)
    monkeypatch.setattr(
        worker,
        "certification_report_blocker",
        lambda *_a, **_k: None,
    )
    monkeypatch.setenv("FAKE_CERTIFIER_EXIT_CODE", str(exit_code))
    if terminated:
        monkeypatch.setenv("FAKE_CERTIFIER_SIGNAL", "1")

    result = worker._run_single_certification(
        demo,
        1,
        timeout_seconds=5.0,
        termination_grace_seconds=0.2,
    )

    assert result["status"] == expected_status
    if terminated:
        assert result["returncode"] == -signal.SIGTERM
        assert "signal" in result["reason"]
    else:
        assert result["returncode"] == exit_code


def test_first_failed_certifier_stops_remaining_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(tmp_path)
    calls: list[int] = []

    def fail_once(_demo: Path, run_index: int, **_kwargs):
        calls.append(run_index)
        return {
            "run": run_index,
            "pid": 1234,
            "returncode": 9,
            "status": "failed",
            "reason": "synthetic failure",
        }

    monkeypatch.setattr(worker, "_run_single_certification", fail_once)
    monkeypatch.setattr(worker, "capture_certified_inputs", lambda _path: {})

    result = worker.certify_demo_dir(
        demo,
        exp_name="t09_insert_peg",
        certification_runs_required=3,
    )

    assert result["status"] == "failed"
    assert calls == [1]
    acceptance = json.loads(
        (demo / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
    )
    assert acceptance["certification_runs_passed"] == 0
    assert acceptance["certifier_pids"] == [1234]
    assert demo.is_dir()
    assert not (tmp_path / demo.name).exists()


@pytest.mark.parametrize(
    "failure",
    ["acceptance", "existing_target"],
)
def test_validation_or_publication_failure_never_exposes_pending_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    demo = _demo_dir(
        tmp_path,
        recording={
            "recorder_lifecycle": "persistent_process",
            "recorder_pid": 9000,
            "fresh_scene_replay_required": True,
        },
    )
    pids = iter((9001, 9002, 9003))

    def pass_once(_demo: Path, run_index: int, **_kwargs):
        return {
            "run": run_index,
            "pid": next(pids),
            "returncode": 0,
            "status": "passed",
            "reason": "",
        }

    monkeypatch.setattr(worker, "_run_single_certification", pass_once)
    monkeypatch.setattr(worker, "capture_certified_inputs", lambda _path: {})
    monkeypatch.setattr(
        worker,
        "deferred_physical_replay_acceptance_upload_blocker",
        lambda _path, **_kwargs: (
            {"code": "invalid_acceptance", "detail": "synthetic rejection"}
            if failure == "acceptance"
            else None
        ),
    )
    target = tmp_path / demo.name
    if failure == "existing_target":
        target.mkdir()
        (target / "sentinel.txt").write_text("do not replace", encoding="utf-8")
    result = worker.certify_demo_dir(
        demo,
        exp_name="t09_insert_peg",
        certification_runs_required=3,
    )

    assert result["status"] == "failed"
    assert result["published_demo_dir"] is None
    assert demo.is_dir()
    if failure == "acceptance":
        assert not target.exists()
    else:
        assert (target / "sentinel.txt").read_text(encoding="utf-8") == (
            "do not replace"
        )
    acceptance = json.loads(
        (demo / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
    )
    assert acceptance["status"] == "failed"
    assert acceptance["published_after_post_collection_certification"] is False


def test_atomic_rename_is_the_only_publication_commit_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(
        tmp_path,
        recording={
            "recorder_lifecycle": "persistent_process",
            "recorder_pid": 9000,
            "fresh_scene_replay_required": True,
        },
    )
    pids = iter((9001, 9002, 9003))

    def pass_once(_demo: Path, run_index: int, **_kwargs):
        return {
            "run": run_index,
            "pid": next(pids),
            "returncode": 0,
            "status": "passed",
            "reason": "",
        }

    class SimulatedPowerLoss(BaseException):
        pass

    def rename_then_lose_power(source: Path) -> Path:
        target = worker._pending_publication_target(source)
        os.rename(source, target)
        raise SimulatedPowerLoss

    monkeypatch.setattr(worker, "_run_single_certification", pass_once)
    monkeypatch.setattr(worker, "capture_certified_inputs", lambda _path: {})
    monkeypatch.setattr(
        worker,
        "deferred_physical_replay_acceptance_upload_blocker",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr(worker, "_publish_pending_demo", rename_then_lose_power)

    with pytest.raises(SimulatedPowerLoss):
        worker.certify_demo_dir(
            demo,
            exp_name="t09_insert_peg",
            certification_runs_required=3,
        )

    published = tmp_path / demo.name
    assert published.is_dir()
    assert not demo.exists()
    acceptance = json.loads(
        (published / worker.FRESH_PROCESS_ACCEPTANCE_NAME).read_text(encoding="utf-8")
    )
    assert acceptance["status"] == "passed"
    assert acceptance["published_after_post_collection_certification"] is True
    assert acceptance["publication_commit"] == (
        "same_filesystem_atomic_directory_rename"
    )


def test_session_publication_rolls_back_an_intermediate_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=2,
    )
    real_rename = worker.os.rename
    forward_renames = 0

    def fail_second_forward_rename(source: Path, target: Path) -> None:
        nonlocal forward_renames
        if Path(source).parent.name == worker.PENDING_DIRECTORY_NAME:
            forward_renames += 1
            if forward_renames == 2:
                raise OSError("synthetic second-member failure")
        real_rename(source, target)

    monkeypatch.setattr(worker.os, "rename", fail_second_forward_rename)
    session = worker._validated_session_context(
        [first, second],
        exp_name="t09_insert_peg",
    )

    with pytest.raises(OSError, match="second-member failure"):
        worker._publish_pending_session(
            [first, second],
            session_context=session,
        )

    assert first.is_dir() and second.is_dir()
    assert not (tmp_path / first.name).exists()
    assert not (tmp_path / second.name).exists()


def test_failed_session_is_reported_as_one_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=3,
    )
    calls: list[list[Path]] = []

    def certify(demos: list[Path], **_kwargs):
        calls.append(list(demos))
        return {
            "status": "failed",
            "reason": "synthetic session failure",
            "collection_session_id": "session-1",
            "demo_dirs": [str(path) for path in demos],
        }

    monkeypatch.setattr(worker, "certify_session", certify)

    with pytest.raises(RuntimeError, match="Deferred PhysX certification failed"):
        worker.main(
            [
                "--exp_name=t09_insert_peg",
                f"--log_path={tmp_path / 'worker.log'}",
                str(first),
                str(second),
            ]
        )

    assert calls == [[first.resolve(), second.resolve()]]


@pytest.mark.parametrize(
    (
        "result_status",
        "physical_status",
        "publication_status",
        "failure_stage",
        "expected_exit_code",
    ),
    [
        ("passed", "passed", "published", None, 0),
        ("failed", "passed", "blocked", "publication_gate", 1),
    ],
)
def test_main_always_replaces_running_state_with_terminal_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_status: str,
    physical_status: str,
    publication_status: str,
    failure_stage: str | None,
    expected_exit_code: int,
) -> None:
    demo = _demo_dir(tmp_path)
    state_path = worker._worker_state_path(demo)
    worker._write_json_atomic(
        state_path,
        {
            "format": worker.WORKER_STATE_FORMAT,
            "version": 1,
            "status": "running",
            "pid": 4321,
            "terminal": False,
            "launch_attempt_count": 1,
        },
    )
    monkeypatch.delenv(worker.SOURCE_RECORDER_PID_ENV, raising=False)
    monkeypatch.setattr(
        worker,
        "BATCH_EXECUTION_LOCK_PATH",
        tmp_path / "locks" / "physical.lock",
    )
    monkeypatch.setattr(
        worker,
        "_wait_for_postprocess_inputs",
        lambda *_args, **_kwargs: None,
    )
    runs = [
        {"run": index, "status": "passed", "pid": 5000 + index} for index in range(1, 4)
    ]
    result = {
        "status": result_status,
        "reason": "synthetic publication blocker" if result_status == "failed" else "",
        "collection_session_id": "session-1",
        "demo_dirs": [str(demo)],
        "physical_certification_status": physical_status,
        "physical_certification_runs_passed": 3,
        "publication_status": publication_status,
        "failure_stage": failure_stage,
        "runs": runs,
    }
    monkeypatch.setattr(worker, "certify_session", lambda *_a, **_k: result)

    argv = [
        "--exp_name=t09_insert_peg",
        f"--log_path={tmp_path / 'worker.log'}",
        str(demo),
    ]
    if result_status == "passed":
        assert worker.main(argv) == 0
    else:
        with pytest.raises(RuntimeError, match="synthetic publication blocker"):
            worker.main(argv)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == (
        "passed" if result_status == "passed" else "review_required"
    )
    assert state["terminal"] is (result_status == "passed")
    assert state["pid"] is None
    assert state["worker_exit_code"] == expected_exit_code
    assert state["physical_certification_status"] == physical_status
    assert state["physical_certification_runs_passed"] == 3
    assert state["physical_certification_runs_completed"] == 3
    assert state["publication_status"] == publication_status
    assert state["failure_stage"] == failure_stage
    assert state["launch_attempt_count"] == 1
    assert "completed_at" in state
    if result_status == "failed":
        assert "synthetic publication blocker" in state["error"]
        assert state["review_status"] == "review_required"
        assert state["physical_replay_outcome_code"] == (
            worker.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        )
    else:
        assert "error" not in state
        assert state["review_status"] == "valid"
        assert state["physical_replay_outcome_code"] == (
            worker.PHYSICAL_REPLAY_VALID_OUTCOME_CODE
        )


def test_worker_waits_for_ui_output_scoped_gpu_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(tmp_path)
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("HANDSON_COLLECTION_RUNTIME_DIR", str(runtime_root))
    monkeypatch.setattr(
        worker,
        "BATCH_EXECUTION_LOCK_PATH",
        tmp_path / "locks" / "physical.lock",
    )
    monkeypatch.setattr(
        worker,
        "_wait_for_postprocess_inputs",
        lambda *_args, **_kwargs: None,
    )
    started = threading.Event()

    def certify(demos: list[Path], **_kwargs):
        started.set()
        return {
            "status": "passed",
            "reason": "",
            "collection_session_id": "session-1",
            "demo_dirs": [str(path) for path in demos],
            "physical_certification_status": "passed",
            "physical_certification_runs_passed": 3,
            "publication_status": "published",
            "failure_stage": None,
            "runs": [
                {"run": index, "status": "passed", "pid": 6000 + index}
                for index in range(1, 4)
            ],
        }

    monkeypatch.setattr(worker, "certify_session", certify)
    gate_path = worker._output_scoped_gpu_gate_path(tmp_path.resolve())
    expected_scope = hashlib.sha256(
        str(tmp_path.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    assert gate_path == (
        runtime_root / expected_scope / ".deferred_physical_replay_gpu_launch.lock"
    )
    gate_path.parent.mkdir(parents=True)
    holder = gate_path.open("a+", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            worker.main(
                [
                    "--exp_name=t09_insert_peg",
                    f"--log_path={tmp_path / 'worker.log'}",
                    str(demo),
                ]
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    time.sleep(0.05)
    assert not started.is_set()
    fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    holder.close()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert not errors
    assert started.is_set()


def test_worker_waits_for_every_postprocess_before_first_certifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _demo_dir(tmp_path, episode_index=1, reset_ordinal=0)
    second = _demo_dir(
        tmp_path,
        name="b03_demo_2",
        episode_index=2,
        reset_ordinal=3,
    )
    events: list[tuple[str, Path]] = []

    monkeypatch.delenv(worker.SOURCE_RECORDER_PID_ENV, raising=False)
    monkeypatch.setattr(
        worker,
        "_wait_for_postprocess_inputs",
        lambda demo, **_kwargs: events.append(("wait", demo)),
    )

    def certify(demos: list[Path], **_kwargs):
        events.append(("certify", demos[0]))
        return {
            "status": "passed",
            "collection_session_id": "session-1",
            "demo_dirs": [str(path) for path in demos],
        }

    monkeypatch.setattr(worker, "certify_session", certify)

    assert (
        worker.main(
            [
                "--exp_name=t09_insert_peg",
                f"--log_path={tmp_path / 'worker.log'}",
                str(first),
                str(second),
            ]
        )
        == 0
    )
    assert events == [
        ("wait", first.resolve()),
        ("wait", second.resolve()),
        ("certify", first.resolve()),
    ]


def test_repo_execution_flock_serializes_different_output_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "out_a" / worker.PENDING_DIRECTORY_NAME / "a01_demo"
    second = tmp_path / "out_b" / worker.PENDING_DIRECTORY_NAME / "b03_demo"
    for demo_dir in (first, second):
        (demo_dir / "replay.zarr").mkdir(parents=True)

    monkeypatch.delenv(worker.SOURCE_RECORDER_PID_ENV, raising=False)
    monkeypatch.setattr(
        worker,
        "BATCH_EXECUTION_LOCK_PATH",
        tmp_path / "repo_sessions" / "physical_execution.lock",
    )
    monkeypatch.setattr(
        worker,
        "_wait_for_postprocess_inputs",
        lambda *_args, **_kwargs: None,
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    order: list[str] = []

    def certify(demo_dirs: list[Path], **_kwargs):
        demo_dir = demo_dirs[0]
        if demo_dir == first.resolve():
            order.append("first_enter")
            first_entered.set()
            assert release_first.wait(timeout=2.0)
            order.append("first_exit")
        else:
            order.append("second_enter")
            second_entered.set()
        return {
            "status": "passed",
            "collection_session_id": "session-1",
            "demo_dirs": [str(demo_dir)],
        }

    monkeypatch.setattr(worker, "certify_session", certify)
    results: dict[str, int] = {}

    def run(name: str, demo_dir: Path) -> None:
        results[name] = worker.main(
            [
                "--exp_name=t09_insert_peg",
                f"--log_path={tmp_path / f'{name}.log'}",
                str(demo_dir),
            ]
        )

    first_thread = threading.Thread(target=run, args=("first", first))
    second_thread = threading.Thread(target=run, args=("second", second))
    first_thread.start()
    assert first_entered.wait(timeout=1.0)
    second_thread.start()
    try:
        assert second_entered.wait(timeout=0.2) is False
    finally:
        release_first.set()
        first_thread.join(timeout=2.0)
        second_thread.join(timeout=2.0)

    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False
    assert results == {"first": 0, "second": 0}
    assert order == ["first_enter", "first_exit", "second_enter"]


def test_timeout_terminates_then_kills_the_whole_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HungProcess:
        pid = 4321
        returncode = None

        def __init__(self) -> None:
            self.communications = 0

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications <= 2:
                raise subprocess.TimeoutExpired(["certifier"], timeout)
            self.returncode = -signal.SIGKILL
            return "partial output", None

    process = HungProcess()
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pgid, signum: signals.append((pgid, signum)),
    )

    output, timed_out = worker._communicate_with_timeout(
        process,  # type: ignore[arg-type]
        timeout_seconds=0.1,
        termination_grace_seconds=0.1,
    )

    assert timed_out is True
    assert output == "partial output"
    assert signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]


@pytest.mark.parametrize(
    "variables",
    [None, {"HANDSON_SEED": 7}],
)
def test_invalid_recorded_environment_fails_before_launch(
    tmp_path: Path,
    variables: object,
) -> None:
    demo = _demo_dir(tmp_path)
    manifest = {"environment": {}}
    if variables is not None:
        manifest["environment"]["variables"] = variables
    (demo / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid recorded environment"):
        worker._certifier_environment(demo)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recorder_lifecycle", "fresh_process", "recorder_lifecycle"),
        ("fresh_scene_replay_required", False, "fresh_scene_replay_required"),
        ("recorder_pid", "not-a-pid", "source recorder pid"),
    ],
)
def test_manifest_process_evidence_must_match_persistent_source_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    demo = _demo_dir(tmp_path)
    manifest = json.loads((demo / "replay_manifest.json").read_text(encoding="utf-8"))
    manifest["recording"] = {field: value}
    (demo / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        worker._recording_process_contract(demo)


@pytest.mark.parametrize("status", ["complete", "not_requested"])
def test_completed_postprocess_allows_physical_certification(
    tmp_path: Path,
    status: str,
) -> None:
    demo = _demo_dir(tmp_path)
    (demo / worker.E01_POSTPROCESS_STATUS_NAME).write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )

    worker._wait_for_postprocess_inputs(demo, timeout_seconds=1.0)


@pytest.mark.parametrize("status", ["failed", "stopped", "corrupt"])
def test_incomplete_or_failed_postprocess_blocks_physical_certification(
    tmp_path: Path,
    status: str,
) -> None:
    demo = _demo_dir(tmp_path)
    (demo / worker.E01_POSTPROCESS_STATUS_NAME).write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="postprocess"):
        worker._wait_for_postprocess_inputs(demo, timeout_seconds=1.0)


def test_queued_postprocess_wait_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _demo_dir(tmp_path)
    (demo / worker.E01_POSTPROCESS_STATUS_NAME).write_text(
        json.dumps({"status": "queued"}),
        encoding="utf-8",
    )
    clock = iter((10.0, 10.0, 12.0))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="waiting for E/F postprocess"):
        worker._wait_for_postprocess_inputs(
            demo,
            timeout_seconds=1.0,
            poll_seconds=0.1,
        )


def test_worker_waits_for_source_recorder_to_exit_before_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(worker.SOURCE_RECORDER_PID_ENV, "4321")
    source_checks = iter((True, True, False))
    sleeps: list[float] = []
    clock = iter((10.0, 10.0, 10.1, 10.2))

    def check_source(pid: int, signal_number: int) -> None:
        assert pid == 4321
        assert signal_number == 0
        if not next(source_checks):
            raise ProcessLookupError

    monkeypatch.setattr(worker.os, "kill", check_source)
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        worker.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    worker._wait_for_launching_recorder_exit(
        timeout_seconds=1.0,
        poll_seconds=0.05,
    )

    assert sleeps == [0.05, 0.05]


def test_worker_rejects_invalid_source_recorder_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(worker.SOURCE_RECORDER_PID_ENV, "not-a-pid")

    with pytest.raises(RuntimeError, match=worker.SOURCE_RECORDER_PID_ENV):
        worker._wait_for_launching_recorder_exit(timeout_seconds=1.0)
