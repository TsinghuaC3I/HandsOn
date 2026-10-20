#!/usr/bin/env python3
"""Render recent cloud ``__physx_v3`` replay packages and publish videos.

The command is a dry run unless ``--apply`` is supplied.  It never touches
immutable ``__verified_v3`` publications. Existing video writes use the ETags
observed during discovery; queued raw packages create missing videos with
``If-None-Match: *`` so concurrent cloud changes fail closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zoneinfo import ZoneInfo

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "handson"
for _path in (_REPO_ROOT, _PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from collection_ui.review_tool.ks3_review_source import Ks3Client  # noqa: E402
from collection_ui.server import (  # noqa: E402
    CLOUD_RENDER_REQUEST_FILENAME,
    Ks3Uploader,
    find_isaacsim_python,
)
from handson.data.deferred_arm_replay_artifacts import (  # noqa: E402
    CANONICAL_ARM_VIDEO_PATHS,
    capture_deferred_arm_replay_artifacts,
)


PHYSX_V3_SUFFIX = "__physx_v3"
REPLAY_ARCHIVE = "replay.zarr.zip"
STATUS_FILE = "deferred_arm_replay_status.json"
CLOUD_META_FILE = "cloud_upload_meta.json"
CLOUD_RENDER_COMMIT_FILE = "CLOUD_RENDER_COMMIT.v1.json"
MAX_ATTEMPTS = 3


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recent_dates(days: int, *, now: dt.datetime | None = None) -> list[str]:
    timezone = ZoneInfo("Asia/Shanghai")
    current = now or dt.datetime.now(timezone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone)
    current = current.astimezone(timezone)
    return [
        (current.date() - dt.timedelta(days=offset)).isoformat()
        for offset in range(max(1, int(days)))
    ]


def _root_prefix(configured_prefix: str, override: str) -> str:
    if override.strip():
        root = override.strip().strip("/")
    else:
        parts = configured_prefix.strip("/").split("/")
        root = "/".join(parts[:-1]) if parts and _is_date(parts[-1]) else "/".join(parts)
    return (root + "/") if root else ""


def _is_date(value: str) -> bool:
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return len(value) == 10


def _iter_pages(
    client: Ks3Client,
    *,
    prefix: str,
    delimiter: str,
) -> Iterable[dict[str, Any]]:
    marker = ""
    seen: set[str] = set()
    while True:
        page = client.list_objects(
            prefix=prefix,
            delimiter=delimiter,
            marker=marker,
            max_keys=100,
        )
        yield page
        if not page.get("is_truncated"):
            return
        next_marker = str(page.get("next_marker") or "")
        if not next_marker or next_marker in seen:
            raise RuntimeError(f"KS3 pagination stalled below {prefix}")
        seen.add(next_marker)
        marker = next_marker


def discover_episode_prefixes(
    client: Ks3Client,
    *,
    root_prefix: str,
    dates: Iterable[str],
) -> list[str]:
    result: list[str] = []
    for date in dates:
        date_prefix = f"{root_prefix}{date}/"
        for page in _iter_pages(client, prefix=date_prefix, delimiter="/"):
            for prefix in page.get("prefixes") or []:
                name = str(prefix).rstrip("/").rsplit("/", 1)[-1]
                if name.endswith(PHYSX_V3_SUFFIX):
                    result.append(str(prefix))
    return sorted(set(result))


def list_episode_objects(client: Ks3Client, episode_prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for page in _iter_pages(client, prefix=episode_prefix, delimiter=""):
        objects.extend(
            row
            for row in page.get("contents") or []
            if str(row.get("key") or "").startswith(episode_prefix)
        )
    return objects


def _safe_relative_key(key: str, episode_prefix: str) -> Path:
    relative = key.removeprefix(episode_prefix)
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"unsafe episode object key: {key}")
    return Path(*posix.parts)


def _download_object(
    client: Ks3Client,
    key: str,
    destination: Path,
    *,
    attempts: int = MAX_ATTEMPTS,
) -> str:
    last_error = ""
    for attempt in range(1, attempts + 1):
        connection = None
        try:
            connection, response = client.open_object(key, timeout=120)
            if response.status != 200:
                body = response.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status}: {body}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.part")
            with temporary.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            os.replace(temporary, destination)
            return str(response.getheader("ETag", "")).strip().strip('"')
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}"
            destination.with_name(f".{destination.name}.part").unlink(
                missing_ok=True
            )
        finally:
            if connection is not None:
                connection.close()
    raise RuntimeError(f"download failed for {key}: {last_error}")


def safe_extract_replay_archive(archive_path: Path, replay_dir: Path) -> None:
    replay_dir.mkdir(parents=True, exist_ok=False)
    resolved_root = replay_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"unsafe replay archive member: {member.filename}")
            target = (replay_dir / Path(*relative.parts)).resolve()
            if resolved_root not in target.parents and target != resolved_root:
                raise ValueError(f"replay archive escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _run_render(
    demo_dir: Path,
    *,
    isaac_python: Path,
    task_name: str,
    video_fps: int,
    attempts: int = MAX_ATTEMPTS,
) -> int:
    command = [
        str(isaac_python),
        str(_REPO_ROOT / "scripts" / "isaac_deferred_arm_replay_worker.py"),
        str(demo_dir),
        "--exp_name",
        task_name,
        "--video_fps",
        str(max(1, int(video_fps))),
        "--force",
    ]
    for attempt in range(1, attempts + 1):
        print(
            f"[render {attempt}/{attempts}] {' '.join(command)}",
            flush=True,
        )
        result = subprocess.run(command, cwd=_REPO_ROOT, check=False)
        if result.returncode == 0:
            capture_deferred_arm_replay_artifacts(demo_dir)
            return attempt
    raise RuntimeError(f"arm render failed after {attempts} attempts: {demo_dir.name}")


def _put_with_retry(
    uploader: Ks3Uploader,
    local_path: Path,
    remote_key: str,
    *,
    if_match: str = "",
    if_none_match: bool = False,
    attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        result = uploader.put_file(
            local_path,
            remote_key,
            if_match=if_match,
            if_none_match=if_none_match,
        )
        if result.get("success"):
            result["attempts"] = attempt
            return result
        if int(result.get("status") or 0) in {409, 412}:
            break
    raise RuntimeError(
        f"PUT failed for {remote_key} after {attempts} attempts: "
        f"HTTP {result.get('status')} {result.get('detail')}"
    )


def update_cloud_receipt(
    receipt: dict[str, Any],
    replacements: dict[str, dict[str, Any]],
    *,
    render_attempts: int,
    request_id: str = "",
    render_commit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(receipt)
    rows = receipt.get("files")
    rows = [dict(row) for row in rows] if isinstance(rows, list) else []
    by_path = {
        str(row.get("path") or ""): row
        for row in rows
        if isinstance(row, dict)
    }
    for relative, replacement in replacements.items():
        row = by_path.get(relative)
        if row is None:
            row = {"path": relative, "key": replacement["key"]}
            rows.append(row)
            by_path[relative] = row
        row.update(
            {
                "key": replacement["key"],
                "bytes": int(replacement["bytes"]),
                "etag": str(replacement.get("etag") or ""),
                "sha256": str(replacement["sha256"]),
            }
        )
    updated.update(
        {
            "status": "uploaded",
            "files": rows,
            "file_count": len(rows) + 1,
            "uploaded_files": len(rows) + 1,
            "bytes": sum(int(row.get("bytes") or 0) for row in rows),
            "cloud_ack": "ok",
            "cloud_ack_at": _now_iso(),
            "arm_rerender_replacement": {
                "format": "handson.cloud_physx_v3_arm_rerender",
                "version": 1,
                "completed_at": _now_iso(),
                "render_attempts": int(render_attempts),
                "upload_max_attempts": MAX_ATTEMPTS,
                "replaced_paths": sorted(replacements),
                "conditional_write": "source_etag_or_create_only",
            },
            "cloud_render": {
                "status": "complete",
                "request_id": str(request_id),
                "completed_at": _now_iso(),
                "local_render_performed": False,
                **(
                    {"render_commit": dict(render_commit)}
                    if isinstance(render_commit, dict)
                    else {}
                ),
            },
        }
    )
    return updated


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def process_episode(
    prefix: str,
    *,
    client: Ks3Client,
    uploader: Ks3Uploader,
    work_root: Path,
    isaac_python: Path,
    force: bool = False,
    require_request: bool = True,
) -> None:
    name = prefix.rstrip("/").rsplit("/", 1)[-1]
    existing_receipt, _existing_receipt_etag = client.get_json(
        prefix + CLOUD_META_FILE
    )
    previous_replacement = existing_receipt.get("arm_rerender_replacement")
    if (
        not force
        and isinstance(previous_replacement, dict)
        and previous_replacement.get("format")
        == "handson.cloud_physx_v3_arm_rerender"
        and previous_replacement.get("version") == 1
    ):
        print(f"[skip already replaced] {prefix}", flush=True)
        return
    objects = list_episode_objects(client, prefix)
    by_relative = {
        _safe_relative_key(str(row["key"]), prefix).as_posix(): row
        for row in objects
    }
    if REPLAY_ARCHIVE not in by_relative:
        raise RuntimeError(f"{prefix} has no {REPLAY_ARCHIVE}")
    if CLOUD_META_FILE not in by_relative:
        raise RuntimeError(f"{prefix} has no {CLOUD_META_FILE}")
    has_render_request = CLOUD_RENDER_REQUEST_FILENAME in by_relative
    if require_request and not has_render_request:
        print(f"[skip no cloud render request] {prefix}", flush=True)
        return
    if not has_render_request and any(
        video not in by_relative for video in CANONICAL_ARM_VIDEO_PATHS
    ):
        raise RuntimeError(
            f"{prefix} has missing videos but no {CLOUD_RENDER_REQUEST_FILENAME}"
        )

    demo_dir = work_root / name
    demo_dir.mkdir(parents=True, exist_ok=False)
    # The renderer needs replay plus episode sidecars, but not old videos.
    for relative, row in by_relative.items():
        if relative.startswith("videos/") or relative == CLOUD_META_FILE:
            continue
        _download_object(
            client,
            str(row["key"]),
            demo_dir / relative,
        )
    safe_extract_replay_archive(
        demo_dir / REPLAY_ARCHIVE,
        demo_dir / "replay.zarr",
    )
    collection_meta = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    render_request = (
        json.loads(
            (demo_dir / CLOUD_RENDER_REQUEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        if has_render_request
        else {}
    )
    task_name = str(
        collection_meta.get("task_name")
        or name.split("_demo_", 1)[0]
    )
    video_fps = int(collection_meta.get("video_fps") or 30)
    render_attempts = _run_render(
        demo_dir,
        isaac_python=isaac_python,
        task_name=task_name,
        video_fps=video_fps,
    )

    replacements: dict[str, dict[str, Any]] = {}
    for relative in CANONICAL_ARM_VIDEO_PATHS:
        local_path = demo_dir / relative
        source = by_relative.get(relative)
        source = source if isinstance(source, dict) else {}
        remote_key = str(source.get("key") or (prefix + relative))
        result = _put_with_retry(
            uploader,
            local_path,
            remote_key,
            if_match=str(source.get("etag") or ""),
            if_none_match=not bool(source),
        )
        replacements[relative] = {
            "key": remote_key,
            "bytes": int(local_path.stat().st_size),
            "etag": str(result.get("etag") or "").strip().strip('"'),
            "sha256": _sha256(local_path),
        }

    status_path = demo_dir / STATUS_FILE
    if status_path.is_file():
        status_key = prefix + STATUS_FILE
        status_source = by_relative.get(STATUS_FILE, {})
        status_result = _put_with_retry(
            uploader,
            status_path,
            status_key,
            if_match=str(status_source.get("etag") or ""),
        )
        replacements[STATUS_FILE] = {
            "key": status_key,
            "bytes": int(status_path.stat().st_size),
            "etag": str(status_result.get("etag") or "").strip().strip('"'),
            "sha256": _sha256(status_path),
        }

    receipt, receipt_etag = client.get_json(prefix + CLOUD_META_FILE)
    receipt_rows = receipt.get("files")
    receipt_rows = receipt_rows if isinstance(receipt_rows, list) else []
    replay_row = next(
        (
            dict(row)
            for row in receipt_rows
            if isinstance(row, dict)
            and row.get("path") == REPLAY_ARCHIVE
        ),
        {},
    )
    render_commit = {
        "format": "handson.cloud_render_commit",
        "version": 1,
        "request_id": str(render_request.get("request_id") or ""),
        "source_episode_id": str(
            render_request.get("source_episode_id") or name
        ),
        "remote_prefix": prefix.rstrip("/"),
        "replay": {
            "key": str(replay_row.get("key") or (prefix + REPLAY_ARCHIVE)),
            "bytes": int(replay_row.get("bytes") or 0),
            "sha256": str(replay_row.get("sha256") or ""),
        },
        "outputs": [
            {
                "path": path,
                "key": str(row.get("key") or ""),
                "bytes": int(row.get("bytes") or 0),
                "sha256": str(row.get("sha256") or ""),
            }
            for path, row in sorted(replacements.items())
            if path in CANONICAL_ARM_VIDEO_PATHS
        ],
        "completed_at": _now_iso(),
    }
    commit_path = demo_dir / CLOUD_RENDER_COMMIT_FILE
    _write_json(commit_path, render_commit)
    commit_source = by_relative.get(CLOUD_RENDER_COMMIT_FILE)
    commit_source = commit_source if isinstance(commit_source, dict) else {}
    commit_key = prefix + CLOUD_RENDER_COMMIT_FILE
    commit_result = _put_with_retry(
        uploader,
        commit_path,
        commit_key,
        if_match=str(commit_source.get("etag") or ""),
        if_none_match=not bool(commit_source),
    )
    commit_receipt = {
        "path": CLOUD_RENDER_COMMIT_FILE,
        "key": commit_key,
        "bytes": int(commit_path.stat().st_size),
        "etag": str(commit_result.get("etag") or "").strip().strip('"'),
        "sha256": _sha256(commit_path),
    }
    replacements[CLOUD_RENDER_COMMIT_FILE] = commit_receipt
    updated_receipt = update_cloud_receipt(
        receipt,
        replacements,
        render_attempts=render_attempts,
        request_id=str(render_request.get("request_id") or ""),
        render_commit=commit_receipt,
    )
    receipt_path = demo_dir / CLOUD_META_FILE
    _write_json(receipt_path, updated_receipt)
    _put_with_retry(
        uploader,
        receipt_path,
        prefix + CLOUD_META_FILE,
        if_match=receipt_etag,
    )
    print(f"[replaced] {prefix} ({task_name}, {video_fps} fps)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3, help="Recent Shanghai calendar days.")
    parser.add_argument("--date", action="append", default=[], help="Explicit YYYY-MM-DD; repeatable.")
    parser.add_argument("--root-prefix", default="", help="KS3 root before date, e.g. handson/.")
    parser.add_argument("--task", default="", help="Only episode names starting with this task.")
    parser.add_argument("--episode-contains", default="", help="Substring filter for episode names.")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="Maximum episode prefixes printed during inventory; 0 prints all.",
    )
    parser.add_argument("--isaac-python", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even when the cloud receipt already records this replacement.",
    )
    parser.add_argument(
        "--include-unrequested",
        action="store_true",
        help=(
            "Also process legacy prefixes without CLOUD_RENDER_REQUEST.v1.json; "
            "the default consumes only explicitly queued cloud-render jobs."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Perform downloads, renders and conditional replacements.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.days < 1:
        raise SystemExit("--days must be >= 1")
    dates = list(dict.fromkeys(args.date or _recent_dates(args.days)))
    if any(not _is_date(value) for value in dates):
        raise SystemExit("every --date must be YYYY-MM-DD")

    uploader = Ks3Uploader.from_env()
    if not uploader.configured():
        raise SystemExit(
            "KS3 config missing: " + ", ".join(uploader.missing_config())
        )
    client = Ks3Client(
        {
            "endpoint": f"{uploader.scheme}://{uploader.endpoint_host}",
            "bucket": uploader.bucket,
            "bucket_domain": f"{uploader.scheme}://{uploader.host}",
            "access_key_id": uploader.access_key_id,
            "access_key_secret": uploader.access_key_secret,
        }
    )
    root = _root_prefix(uploader.prefix, args.root_prefix)
    prefixes = discover_episode_prefixes(
        client,
        root_prefix=root,
        dates=dates,
    )
    if args.task:
        prefixes = [
            prefix
            for prefix in prefixes
            if prefix.rstrip("/").rsplit("/", 1)[-1].startswith(args.task)
        ]
    if args.episode_contains:
        prefixes = [
            prefix for prefix in prefixes if args.episode_contains in prefix
        ]
    if args.max_episodes > 0:
        prefixes = prefixes[: args.max_episodes]
    print(
        f"dates={','.join(dates)} root={root} physx_v3_episodes={len(prefixes)}",
        flush=True,
    )
    printed_prefixes = (
        prefixes
        if args.list_limit == 0
        else prefixes[: max(0, args.list_limit)]
    )
    for prefix in printed_prefixes:
        print(f"  {prefix}")
    if len(printed_prefixes) < len(prefixes):
        print(f"  ... {len(prefixes) - len(printed_prefixes)} more (use --list-limit 0)")
    if not args.apply or not prefixes:
        print("dry-run: add --apply to render and replace these exact prefixes")
        return 0

    isaac_python_text = args.isaac_python or find_isaacsim_python()
    isaac_python = Path(isaac_python_text).expanduser()
    if not isaac_python.is_file():
        raise SystemExit(f"Isaac Sim Python not found: {isaac_python}")
    owned_temp = not bool(args.work_dir)
    work_root = (
        Path(tempfile.mkdtemp(prefix="handson-cloud-arm-rerender-"))
        if owned_temp
        else Path(args.work_dir).expanduser().resolve()
    )
    work_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    try:
        for index, prefix in enumerate(prefixes, start=1):
            print(f"[{index}/{len(prefixes)}] {prefix}", flush=True)
            episode_work_dir = work_root / prefix.rstrip("/").rsplit("/", 1)[-1]
            try:
                process_episode(
                    prefix,
                    client=client,
                    uploader=uploader,
                    work_root=work_root,
                    isaac_python=isaac_python,
                    force=bool(args.force),
                    require_request=not bool(args.include_unrequested),
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((prefix, f"{type(exc).__name__}: {exc}"))
                print(f"[failed] {prefix}: {failures[-1][1]}", flush=True)
            finally:
                if not args.keep_work_dir:
                    shutil.rmtree(episode_work_dir, ignore_errors=True)
    finally:
        if owned_temp and not args.keep_work_dir:
            shutil.rmtree(work_root, ignore_errors=True)
        else:
            print(f"work_dir={work_root}")
    if failures:
        print(f"failed={len(failures)}", flush=True)
        for prefix, detail in failures:
            print(f"  {prefix}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
