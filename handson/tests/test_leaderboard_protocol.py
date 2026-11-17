"""CPU checks for the handson-eval-v1 leaderboard protocol, suite runner and submission validator."""
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from handson.evaluation import leaderboard as lb
from handson.evaluation import suite

REPO = Path(__file__).resolve().parents[2]
IDENTITY = dict(protocol=lb.PROTOCOL, git_commit="0" * 40, git_dirty=False,
                protocol_files=dict(lb.PROTOCOL_FILE_SHA256))
EXECUTION = dict(transport="openpi_websocket", action_horizon=30, history_frames=1, history_stride=30)


def test_public_trials_are_frozen():
    rows = lb.catalog()
    assert len(rows) == 66
    assert sum(split == "core" for _, split in rows) == 50
    assert lb.public_universe_sha256() == lb.PUBLIC_UNIVERSE_SHA256
    assert {track: len(lb.manifest(track)) for track in lb.TRACKS} == dict(
        multi_task=5800, single_task=5000, zero_shot=660)
    assert lb.public_seed("t01_align_blocks_in_midair", 0) == 229608639
    # Track lists are prefixes of one public list per task.
    small = {(r["task"], r["trial"]): r["seed"] for r in lb.manifest("zero_shot")}
    large = {(r["task"], r["trial"]): r["seed"] for r in lb.manifest("multi_task")}
    assert all(large[key] == seed for key, seed in small.items())
    assert all(0 <= seed <= lb.SEED_MASK for seed in large.values())


def test_protocol_files_match_the_frozen_hashes():
    assert lb.protocol_file_hashes() == lb.PROTOCOL_FILE_SHA256


def test_hidden_seeds_need_the_key_and_avoid_public_seeds():
    task = "t24_place_ball_in_bowl"
    first = lb.hidden_seeds(b"key-one", task, 10)
    assert first == lb.hidden_seeds(b"key-one", task, 10)
    assert first != lb.hidden_seeds(b"key-two", task, 10)
    assert not set(first) & {lb.public_seed(task, i) for i in range(lb.MAX_PUBLIC_TRIALS)}
    rows = lb.manifest("zero_shot", hidden_key=b"key-one")
    assert {row["set"] for row in rows} == {"hidden"} and len(rows) == 660


def _episode(folder, planned, success, *, formal=True):
    folder.mkdir(parents=True)
    result = dict(task=planned["task"], seed=planned["seed"], steps=300, success=success,
                  stop_reason="native_terminated" if success else "native_horizon",
                  complete_evaluation_episode=True, counts_as_formal_trial=formal, physics_backend="physx",
                  paper_physics_protocol=True, protocol="handson_native_eef52", diagnostic_only=False,
                  paper_variant=None)
    (folder / "RESULT.json").write_text(json.dumps(result))
    (folder / "JOB.json").write_text(json.dumps(dict(policy_probe=EXECUTION, benchmark=dict(render_stride=30))))
    (folder / "PROMPT.json").write_text(json.dumps(dict(goal_id="g0", prompt_variant="canonical")))


def _run(tmp_path, track="zero_shot", outcome=lambda row: row["trial"] % 3 == 0):
    run = tmp_path / "run"
    rows = lb.manifest(track)
    run.mkdir()
    (run / "SUITE.json").write_text(json.dumps(dict(
        schema="handson_suite_run_v1", protocol=lb.PROTOCOL, track=track, set="public",
        manifest_sha256=lb.manifest_sha256(rows), evaluator=IDENTITY, evaluator_args=[])))
    (run / "MANIFEST.jsonl").write_text("".join(lb._canonical(row) + "\n" for row in rows))
    for row in rows:
        folder = lb.trial_dir(run, row["task"], row["trial"])
        _episode(folder / "attempt_00" / "episode", row, outcome(row))
    return run, rows


def _submission(tmp_path, run, track="zero_shot"):
    out = tmp_path / "20261201_example-lab_example-model"
    results, summary = suite._summarize([run])
    out.mkdir()
    (out / "results.jsonl").write_text(lb.results_text(results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    meta = yaml.safe_load((REPO / "leaderboard" / "templates" / "metadata.yaml").read_text())
    meta["track"] = track
    if track == "zero_shot":
        meta["training"]["handson_data"] = "none"
    (out / "metadata.yaml").write_text(yaml.safe_dump(meta))
    return out, summary


def test_summary_and_validation_round_trip(tmp_path):
    run, rows = _run(tmp_path)
    # One trial needed a retry after an infrastructure error.
    retried = rows[5]
    folder = lb.trial_dir(run, retried["task"], retried["trial"])
    shutil.move(folder / "attempt_00", folder / "attempt_01")
    (folder / "attempt_00" / "episode").mkdir(parents=True)
    (folder / "attempt_00" / "episode" / "ERROR.json").write_text("{}")
    out, summary = _submission(tmp_path, run)
    assert summary["counts"] == dict(planned=660, scored=660, invalid=0, missing=0,
                                     successes=sum(r["trial"] % 3 == 0 for r in rows),
                                     retried_trials=1, attempts=661)
    assert summary["execution"] == dict(EXECUTION, render_stride=30) and summary["consistent_execution"]
    # Each task scores trials 0, 3, 6, 9 of ten.
    assert summary["suites"]["core"]["percent"] == 40.0 and summary["overall"]["percent"] == 40.0
    row = next(json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()
               if json.loads(line)["trial"] == retried["trial"] and json.loads(line)["task"] == retried["task"])
    assert row["scored_attempt"] == "attempt_01" and row["attempts"] == 2
    report = lb.validate_submission(out)
    assert report["errors"] == []
    assert report["warnings"] == []


def test_validator_rejects_tampering(tmp_path):
    run, _ = _run(tmp_path)
    out, summary = _submission(tmp_path, run)
    lines = (out / "results.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["success"] = not first["success"]
    (out / "results.jsonl").write_text("\n".join([lb._canonical(first)] + lines[1:]) + "\n")
    assert any("does not match" in e for e in lb.validate_submission(out)["errors"])
    (out / "results.jsonl").write_text("\n".join(lines[1:]) + "\n")
    errors = lb.validate_submission(out)["errors"]
    assert any("planned trials have no row" in e for e in errors)
    (out / "results.jsonl").write_text("\n".join(lines) + "\n")
    meta = yaml.safe_load((out / "metadata.yaml").read_text())
    meta["interface"]["uses_task_identifier"] = True
    meta["training"]["handson_data"] = "Core-50"
    (out / "metadata.yaml").write_text(yaml.safe_dump(meta))
    errors = lb.validate_submission(out)["errors"]
    assert any("task identifier" in e for e in errors) and any("zero_shot" in e for e in errors)
    dirty = dict(summary, evaluator=dict(IDENTITY, git_dirty=True))
    (out / "summary.json").write_text(json.dumps(dirty))
    assert any("local changes" in e for e in lb.validate_submission(out)["errors"])


def test_non_formal_and_missing_trials_are_not_scored(tmp_path):
    run, rows = _run(tmp_path, outcome=lambda row: True)
    bounded = lb.trial_dir(run, rows[0]["task"], rows[0]["trial"]) / "attempt_00"
    shutil.rmtree(bounded)
    _episode(bounded / "episode", rows[0], True, formal=False)
    shutil.rmtree(lb.trial_dir(run, rows[1]["task"], rows[1]["trial"]))
    results, summary = suite._summarize([run])
    assert results[0]["status"] == "invalid" and "formal" in results[0]["reason"]
    assert results[1]["status"] == "missing"
    assert summary["counts"]["scored"] == 658
    # Unscored trials count as failures in the task's rate.
    task = next(t for t in summary["tasks"] if t["task"] == rows[0]["task"])
    assert task["successes"] == 8 and task["rate"] == 0.8


def test_consistency_decision(tmp_path):
    run, _ = _run(tmp_path)
    _, public = suite._summarize([run])
    assert lb.consistency(public, public)["consistent"]
    shutil.rmtree(tmp_path / "run")
    worse, _ = _run(tmp_path, outcome=lambda row: False)
    _, hidden = suite._summarize([worse])
    decision = lb.consistency(public, hidden)
    assert not decision["consistent"] and decision["metric"] == "overall"


def test_suite_rejects_reserved_evaluator_arguments():
    with pytest.raises(SystemExit):
        suite._passthrough(["--seed", "3"])
    with pytest.raises(SystemExit):
        suite._passthrough(["--output=elsewhere"])
    assert suite._passthrough(["--host", "10.0.0.2", "--port", "9000"]) == ["--host", "10.0.0.2", "--port", "9000"]


FAKE_EVALUATOR = '''
import json, sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = Path(args["--output"])
flag = out.parent.parent / "fail-once"
if args["--task"] == "t01_align_blocks_in_midair" and not flag.exists():
    flag.write_text("")
    out.mkdir(parents=True)
    (out / "ERROR.json").write_text("{}")
    sys.exit(1)
out.mkdir(parents=True)
(out / "scene-cache").mkdir()
result = dict(task=args["--task"], seed=int(args["--seed"]), steps=90, success=True, stop_reason="native_terminated",
              complete_evaluation_episode=True, counts_as_formal_trial=True, physics_backend="physx",
              paper_physics_protocol=True, protocol="handson_native_eef52", diagnostic_only=False, paper_variant=None)
(out / "RESULT.json").write_text(json.dumps(result))
(out / "JOB.json").write_text(json.dumps(dict(policy_probe=dict(transport="openpi_websocket", action_horizon=30,
    history_frames=1, history_stride=30), benchmark=dict(render_stride=30))))
'''


def test_runner_retries_infrastructure_errors_and_resumes(tmp_path, monkeypatch):
    fake = tmp_path / "fake_evaluator.py"
    fake.write_text(FAKE_EVALUATOR)
    monkeypatch.setattr(lb, "evaluator_identity", lambda: dict(IDENTITY))
    monkeypatch.setattr(suite.Runner, "command", lambda self, row, attempt: [
        sys.executable, str(fake), "--task", row["task"], "--seed", str(row["seed"]),
        "--output", str(attempt / "episode")])
    output = tmp_path / "run"
    argv = ["run", "--track", "zero_shot", "--output", str(output), "--tasks",
            "t01_align_blocks_in_midair,t24_place_ball_in_bowl", "--gpus", "0,1", "--per-gpu", "2"]
    suite.main(argv)
    # Every t01 trial fails once, then succeeds on its second attempt.
    entries = sorted(p.name for p in lb.trial_dir(output, "t01_align_blocks_in_midair", 0).iterdir())
    assert entries == ["attempt_00", "attempt_01", "fail-once"]
    dirs = [p for p in (output / "trials").rglob("attempt_[0-9][0-9]") if p.is_dir()]
    complete = [p for p in dirs if lb.attempt_complete(p)]
    assert len(dirs) == 30 and len(complete) == 20
    assert not any((p / "episode" / "scene-cache").exists() for p in complete)
    assert json.loads((dirs[0] / "ATTEMPT.json").read_text())["gpu"] in ("0", "1")
    # Resuming runs nothing new; a different evaluator configuration is refused.
    suite.main(argv)
    assert len([p for p in (output / "trials").rglob("attempt_[0-9][0-9]") if p.is_dir()]) == 30
    with pytest.raises(SystemExit):
        suite.main(argv + ["--", "--port", "9001"])
