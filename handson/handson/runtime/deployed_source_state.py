"""Verify a Git-free deployed worktree against a hash-bound source receipt."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


RECEIPT_NAME = "handson_deployed_source_state.json"
RECEIPT_FORMAT = "handson.deployed_source_state"
RECEIPT_VERSION = 1
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_VERIFICATION_CACHE: dict[
    str,
    tuple[str, str, dict[str, Any]],
] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_git_dir(repo_root: Path) -> Path | None:
    marker = Path(repo_root).resolve() / ".git"
    if marker.is_dir():
        return marker
    try:
        line = marker.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    if not line.startswith("gitdir:"):
        return None
    candidate = Path(line.split(":", 1)[1].strip()).expanduser()
    if not candidate.is_absolute():
        candidate = marker.parent / candidate
    try:
        return candidate.resolve()
    except OSError:
        return None


def deployed_source_receipt_path(repo_root: Path) -> Path | None:
    git_dir = repository_git_dir(repo_root)
    return None if git_dir is None else git_dir / RECEIPT_NAME


def _safe_worktree_path(repo_root: Path, relative: str) -> Path | None:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        return None
    root = Path(repo_root).resolve()
    path = root / candidate
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, ValueError):
        return None
    return path


def verify_deployed_source_state(
    repo_root: Path,
    *,
    use_cache: bool = False,
) -> dict[str, Any] | None:
    """Return a Git-shaped identity only after every receipt entry verifies."""

    root = Path(repo_root).resolve()
    receipt_path = deployed_source_receipt_path(root)
    if receipt_path is None:
        return None
    try:
        receipt_bytes = receipt_path.read_bytes()
        payload = json.loads(receipt_bytes.decode("utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    commit = str(payload.get("commit") or "").strip().lower()
    files = payload.get("files")
    ancestors = payload.get("ancestors")
    if (
        payload.get("format") != RECEIPT_FORMAT
        or payload.get("version") != RECEIPT_VERSION
        or not _HEX40.fullmatch(commit)
        or not isinstance(files, list)
        or not files
        or not isinstance(ancestors, list)
        or commit not in ancestors
    ):
        return None

    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    normalized: list[tuple[str, str, str, Path, str, str, int, str]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            return None
        relative = str(item.get("path") or "")
        kind = str(item.get("kind") or "")
        expected = str(item.get("sha256") or "").lower()
        git_oid = str(item.get("git_oid") or "").lower()
        gitlink_state = str(item.get("state") or "")
        lfs_size_value = item.get("size")
        lfs_size = (
            int(lfs_size_value)
            if type(lfs_size_value) is int and lfs_size_value >= 0
            else -1
        )
        lfs_pointer_sha256 = str(item.get("pointer_sha256") or "").lower()
        path = _safe_worktree_path(root, relative)
        if (
            path is None
            or relative in seen
            or kind not in {"file", "symlink", "gitlink", "lfs"}
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or (
                kind == "lfs"
                and (
                    lfs_size < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", lfs_pointer_sha256)
                )
            )
            or (
                kind == "gitlink"
                and (
                    not _HEX40.fullmatch(git_oid)
                    or gitlink_state != "empty_directory"
                    or expected
                    != hashlib.sha256(
                        b"gitlink\0" + git_oid.encode("ascii")
                    ).hexdigest()
                )
            )
        ):
            return None
        seen.add(relative)
        normalized.append(
            (
                relative,
                kind,
                expected,
                path,
                git_oid,
                gitlink_state,
                lfs_size,
                lfs_pointer_sha256,
            )
        )

    stat_rows: list[tuple[str, str, int, int, int, str]] = []
    for (
        relative,
        kind,
        _expected,
        path,
        _git_oid,
        _gitlink_state,
        _lfs_size,
        _lfs_pointer_sha256,
    ) in normalized:
        try:
            stat = path.lstat()
            target = os.readlink(path) if path.is_symlink() else ""
            stat_rows.append(
                (
                    relative,
                    kind,
                    int(stat.st_mode),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    target,
                )
            )
        except OSError:
            stat_rows.append((relative, kind, 0, -1, -1, "missing"))
    stat_fingerprint = hashlib.sha256(
        json.dumps(
            stat_rows,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cached = _VERIFICATION_CACHE.get(str(root))
    if (
        use_cache
        and cached
        and cached[0] == receipt_sha256
        and cached[1] == stat_fingerprint
    ):
        return dict(cached[2])

    changed_files: list[str] = []
    verified_rows: list[dict[str, str]] = []
    for (
        relative,
        kind,
        expected,
        path,
        git_oid,
        gitlink_state,
        lfs_size,
        lfs_pointer_sha256,
    ) in normalized:
        actual = ""
        try:
            if kind == "symlink" and path.is_symlink():
                actual = hashlib.sha256(
                    os.readlink(path).encode("utf-8", errors="surrogateescape")
                ).hexdigest()
            elif kind == "file" and path.is_file() and not path.is_symlink():
                actual = _sha256_file(path)
            elif (
                kind == "lfs"
                and path.is_file()
                and not path.is_symlink()
            ):
                file_sha256 = _sha256_file(path)
                if (
                    (
                        path.stat().st_size == lfs_size
                        and file_sha256 == expected
                    )
                    or file_sha256 == lfs_pointer_sha256
                ):
                    # Pointer and materialized forms represent the same
                    # tracked LFS object, so the content identity is its OID.
                    actual = expected
            elif (
                kind == "gitlink"
                and gitlink_state == "empty_directory"
                and path.is_dir()
                and not path.is_symlink()
                and not any(path.iterdir())
            ):
                actual = hashlib.sha256(
                    b"gitlink\0" + git_oid.encode("ascii")
                ).hexdigest()
        except OSError:
            actual = ""
        if actual != expected:
            changed_files.append(relative)
        verified_rows.append(
            {"path": relative, "kind": kind, "sha256": actual or "missing"}
        )

    identity_bytes = json.dumps(
        verified_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dirty = bool(changed_files)
    result = {
        "commit": commit,
        "dirty": dirty,
        "status": "dirty" if dirty else "clean",
        "changed_files": sorted(changed_files),
        "diff_binary_sha256": receipt_sha256,
        "worktree_content_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "source": "verified_deployment_receipt",
        "receipt_path": str(receipt_path),
        "ancestors": tuple(
            value
            for value in (str(item).strip().lower() for item in ancestors)
            if _HEX40.fullmatch(value)
        ),
    }
    if use_cache:
        _VERIFICATION_CACHE[str(root)] = (
            receipt_sha256,
            stat_fingerprint,
            dict(result),
        )
    return result


__all__ = [
    "RECEIPT_FORMAT",
    "RECEIPT_NAME",
    "RECEIPT_VERSION",
    "deployed_source_receipt_path",
    "repository_git_dir",
    "verify_deployed_source_state",
]
