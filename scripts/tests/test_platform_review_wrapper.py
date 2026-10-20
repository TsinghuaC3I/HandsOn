from __future__ import annotations

import json
from pathlib import Path

from scripts import platform_physical_replay_review as review


def _demo(parent: Path, name: str) -> Path:
    demo_dir = parent / name
    demo_dir.mkdir(parents=True)
    (demo_dir / "collection_meta.json").write_text(
        json.dumps({"task_name": "t44_stack_blocks"}),
        encoding="utf-8",
    )
    return demo_dir


def _passed_report() -> dict:
    return {
        "status": "passed",
        "task": "t44_stack_blocks",
        "steps_expected": 12,
        "steps_executed": 12,
        "task_success_replay": True,
        "final_succeed": True,
        "trace_match": True,
        "transition_outcomes_match": True,
        "reason": "matched",
    }


def test_review_resume_restores_complete_session_order(tmp_path: Path) -> None:
    pending = tmp_path / review.PENDING_DIRECTORY_NAME
    history = pending / ".physx_session_history" / "session-a"
    history.mkdir(parents=True)
    first = _demo(tmp_path, "first")
    second = _demo(pending, "second")
    relative_history = history.relative_to(tmp_path).as_posix()
    for demo_dir in (first, second):
        (demo_dir / "replay_manifest.json").write_text(
            json.dumps(
                {
                    "recording": {
                        "physical_replay_session_history_root_relative_path": (
                            relative_history
                        )
                    }
                }
            ),
            encoding="utf-8",
        )
    (history / review.SESSION_FINALIZATION_NAME).write_text(
        json.dumps(
            {
                "accepted_demos": [
                    {"demo_id": first.name},
                    {"demo_id": second.name},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert review._complete_session_paths([second]) == [
        first.resolve(),
        second.resolve(),
    ]


def test_review_publishes_each_pass_and_continues_same_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pending = tmp_path / review.PENDING_DIRECTORY_NAME
    pending.mkdir()
    first = _demo(pending, "first")
    second = _demo(pending, "second")
    calls: list[Path] = []

    def certify(paths, *, platform_review, on_episode_completed):
        assert platform_review is True
        assert list(paths) == [first.resolve(), second.resolve()]
        calls.append(first)
        on_episode_completed(first.resolve(), _passed_report())
        assert not first.exists()
        assert (tmp_path / first.name).is_dir()
        calls.append(second)
        on_episode_completed(
            second.resolve(),
            {
                **_passed_report(),
                "status": "failed",
                "task_success_replay": False,
                "final_succeed": False,
                "trace_match": False,
                "transition_outcomes_match": False,
                "reason": "physical trajectory diverged",
            },
        )
        return {}

    monkeypatch.setattr(review, "certify_demo_session", certify)

    result = review.run_review([first, second], review_run_id="review-1")

    assert result == 0
    assert calls == [first, second]
    passed = json.loads(
        (tmp_path / first.name / review.STATUS_NAME).read_text(encoding="utf-8")
    )
    invalid = json.loads(
        (second / review.STATUS_NAME).read_text(encoding="utf-8")
    )
    assert passed["status"] == "passed"
    assert passed["published_path"] == str(tmp_path / first.name)
    assert invalid["status"] == "invalid"
    assert invalid["physics_executed"] is True
    assert invalid["reason"] == "physical trajectory diverged"
    assert not any("hash" in key or "sha256" in key for key in passed)
    assert not any("hash" in key or "sha256" in key for key in invalid)


def test_review_keeps_infrastructure_failure_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pending = tmp_path / review.PENDING_DIRECTORY_NAME
    pending.mkdir()
    demo_dir = _demo(pending, "failed")

    def certify(paths, *, platform_review, on_episode_completed):
        on_episode_completed(
            Path(paths[0]),
            {
                "status": "failed",
                "steps_expected": None,
                "steps_executed": None,
                "task_success_replay": False,
                "trace_match": False,
                "transition_outcomes_match": False,
                "reason": "CUDA out of memory",
                "error_type": "RuntimeError",
            },
        )
        return {}

    monkeypatch.setattr(review, "certify_demo_session", certify)

    result = review.run_review([demo_dir], review_run_id="review-2")

    status = json.loads(
        (demo_dir / review.STATUS_NAME).read_text(encoding="utf-8")
    )
    assert result == 1
    assert demo_dir.is_dir()
    assert status["status"] == "failed"
    assert status["physics_executed"] is False
    assert status["error_type"] == "RuntimeError"
