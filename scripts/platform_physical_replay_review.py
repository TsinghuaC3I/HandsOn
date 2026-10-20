#!/usr/bin/env python3
"""Run one real PhysX replay per episode for the collection platform."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "handson"
for entry in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts.certify_physx_replay import certify_demo_session  # noqa: E402


STATUS_NAME = "physical_replay_review.json"
STATUS_FORMAT = "handson.platform_physical_replay_review"
PENDING_DIRECTORY_NAME = ".physical_replay_pending"
SESSION_FINALIZATION_NAME = "FINALIZED.json"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _task_name(demo_dir: Path) -> str:
    for name in ("collection_meta.json", "replay_manifest.json"):
        try:
            document = json.loads((demo_dir / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        task = document.get("task")
        task = task if isinstance(task, dict) else {}
        value = str(document.get("task_name") or task.get("name") or "").strip()
        if value:
            return value
    return ""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _session_history_path(demo_dir: Path) -> Path | None:
    manifest = _json_object(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    value = str(
        recording.get("physical_replay_session_history_root_relative_path")
        or ""
    ).strip()
    if not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"复核数据的物理会话路径无效: {demo_dir.name}"
        )
    for parent in demo_dir.parents:
        candidate = parent / relative
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(f"找不到物理会话记录: {demo_dir.name}")


def _complete_session_paths(demo_dirs: Sequence[Path]) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in demo_dirs]
    if not paths:
        return []
    history_paths = {
        history
        for path in paths
        if (history := _session_history_path(path)) is not None
    }
    if not history_paths:
        return paths
    if len(history_paths) != 1:
        raise RuntimeError("一次物理复核批次包含多个会话记录。")

    history_path = next(iter(history_paths))
    finalization = _json_object(history_path / SESSION_FINALIZATION_NAME)
    accepted = finalization.get("accepted_demos")
    if not isinstance(accepted, list) or not accepted:
        raise RuntimeError("物理会话记录缺少已接受数据顺序。")

    pending_root = next(
        (
            parent
            for parent in history_path.parents
            if parent.name == PENDING_DIRECTORY_NAME
        ),
        None,
    )
    search_roots: list[Path] = []
    if pending_root is not None:
        search_roots.extend((pending_root.parent, pending_root))
    for path in paths:
        search_roots.append(path.parent)
        if path.parent.name == PENDING_DIRECTORY_NAME:
            search_roots.append(path.parent.parent)
        else:
            search_roots.append(path.parent / PENDING_DIRECTORY_NAME)

    ordered: list[Path] = []
    accepted_names: set[str] = set()
    for index, row in enumerate(accepted):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"物理会话第 {index + 1} 条顺序记录无效。")
        demo_id = str(row.get("demo_id") or "").strip()
        if not demo_id or demo_id in accepted_names or Path(demo_id).name != demo_id:
            raise RuntimeError(f"物理会话第 {index + 1} 条数据名称无效。")
        accepted_names.add(demo_id)
        match = next(
            (
                root / demo_id
                for root in search_roots
                if (root / demo_id).is_dir()
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"物理会话数据缺失: {demo_id}")
        ordered.append(match.resolve())

    unexpected = sorted(path.name for path in paths if path.name not in accepted_names)
    if unexpected:
        raise RuntimeError("复核批次包含不属于该物理会话的数据: " + ", ".join(unexpected))
    return ordered


def _review_status(report: Mapping[str, Any]) -> str:
    passed = bool(
        report.get("status") == "passed"
        and report.get("task_success_replay") is True
        and report.get("trace_match") is True
        and report.get("transition_outcomes_match") is True
        and type(report.get("steps_expected")) is int
        and report.get("steps_executed") == report.get("steps_expected")
    )
    if passed:
        return "passed"
    if not report.get("error_type") and type(report.get("steps_expected")) is int:
        return "invalid"
    return "failed"


def run_review(demo_dirs: Sequence[Path], *, review_run_id: str) -> int:
    paths = _complete_session_paths(demo_dirs)
    positions = {path: index for index, path in enumerate(paths, start=1)}
    completed = 0
    infrastructure_failures = 0
    started_at = _now_iso()
    for path in paths:
        _write_json_atomic(
            path / STATUS_NAME,
            {
                "format": STATUS_FORMAT,
                "version": 1,
                "review_run_id": review_run_id,
                "status": "queued",
                "task_name": _task_name(path),
                "episode": path.name,
                "episode_index": positions[path],
                "episode_count": len(paths),
                "started_at": started_at,
                "updated_at": _now_iso(),
            },
        )

    def completed_episode(path: Path, report: Mapping[str, Any]) -> None:
        nonlocal completed, infrastructure_failures
        completed += 1
        status = _review_status(report)
        if status == "failed":
            infrastructure_failures += 1
        finished_at = _now_iso()
        payload = {
            "format": STATUS_FORMAT,
            "version": 1,
            "review_run_id": review_run_id,
            "status": status,
            "task_name": str(report.get("task") or _task_name(path)),
            "episode": path.name,
            "episode_index": positions.get(path, completed),
            "episode_count": len(paths),
            "physics_executed": status in {"passed", "invalid"},
            "steps_expected": report.get("steps_expected"),
            "steps_executed": report.get("steps_executed"),
            "task_success_replay": report.get("task_success_replay"),
            "final_succeed": report.get("final_succeed"),
            "trace_match": report.get("trace_match"),
            "transition_outcomes_match": report.get(
                "transition_outcomes_match"
            ),
            "reason": str(report.get("reason") or ""),
            "error_type": str(report.get("error_type") or ""),
            "report": "physical_replay_certification.json",
            "started_at": started_at,
            "finished_at": finished_at,
            "updated_at": finished_at,
        }
        _write_json_atomic(path / STATUS_NAME, payload)
        published_path = path
        if status == "passed" and path.parent.name == PENDING_DIRECTORY_NAME:
            target = path.parent.parent / path.name
            try:
                if os.path.lexists(target):
                    raise FileExistsError(f"target already exists: {target}")
                os.rename(path, target)
                published_path = target
                payload["published_path"] = str(target)
                payload["updated_at"] = _now_iso()
                _write_json_atomic(target / STATUS_NAME, payload)
            except OSError as exc:
                infrastructure_failures += 1
                payload.update(
                    {
                        "status": "failed",
                        "physics_executed": True,
                        "reason": f"physical replay passed but publication failed: {exc}",
                        "error_type": type(exc).__name__,
                        "updated_at": _now_iso(),
                    }
                )
                _write_json_atomic(path / STATUS_NAME, payload)
                published_path = path
        print(
            json.dumps(
                {
                    "event": "episode_completed",
                    "completed": completed,
                    "total": len(paths),
                    "status": payload["status"],
                    "episode": published_path.name,
                    "path": str(published_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:
        certify_demo_session(
            paths,
            platform_review=True,
            on_episode_completed=completed_episode,
        )
    except BaseException as exc:
        finished_at = _now_iso()
        for path in paths:
            status_path = path / STATUS_NAME
            if not status_path.exists():
                target = path.parent.parent / path.name
                status_path = target / STATUS_NAME
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                current = {}
            if isinstance(current, dict) and current.get("status") in {
                "passed",
                "invalid",
                "failed",
            }:
                continue
            payload = {
                **(current if isinstance(current, dict) else {}),
                "format": STATUS_FORMAT,
                "version": 1,
                "review_run_id": review_run_id,
                "status": "failed",
                "physics_executed": False,
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "finished_at": finished_at,
                "updated_at": finished_at,
            }
            _write_json_atomic(status_path, payload)
        raise
    return 1 if infrastructure_failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run-id", required=True)
    parser.add_argument("demo_dirs", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    return run_review(
        arguments.demo_dirs,
        review_run_id=str(arguments.review_run_id),
    )


if __name__ == "__main__":
    raise SystemExit(main())
