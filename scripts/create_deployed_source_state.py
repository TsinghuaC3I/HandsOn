#!/usr/bin/env python3
"""Create the hash receipt used by Git-free HandsOn deployments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def _git(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_lfs_pointer(path: Path) -> tuple[str, int, str] | None:
    """Return the object hash/size encoded by a canonical Git LFS pointer."""

    try:
        if path.stat().st_size > 1024:
            return None
        pointer_bytes = path.read_bytes()
        lines = pointer_bytes.splitlines()
    except OSError:
        return None
    if len(lines) != 3 or lines[0] != b"version https://git-lfs.github.com/spec/v1":
        return None
    oid_match = re.fullmatch(rb"oid sha256:([0-9a-f]{64})", lines[1])
    size_match = re.fullmatch(rb"size ([0-9]+)", lines[2])
    if oid_match is None or size_match is None:
        return None
    return (
        oid_match.group(1).decode("ascii"),
        int(size_match.group(1)),
        hashlib.sha256(pointer_bytes).hexdigest(),
    )


def build_receipt(repo_root: Path) -> dict[str, object]:
    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if status.strip():
        raise RuntimeError("tracked worktree files must be clean before deployment")
    tracked_entries: list[tuple[str, str, str]] = []
    for item in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not item:
            continue
        try:
            metadata, relative_bytes = item.split(b"\t", 1)
            mode_bytes, object_bytes, stage_bytes = metadata.split()
        except ValueError as exc:
            raise RuntimeError("unable to parse tracked Git index entry") from exc
        if stage_bytes != b"0":
            raise RuntimeError("unmerged Git index entries cannot be deployed")
        tracked_entries.append(
            (
                mode_bytes.decode("ascii"),
                object_bytes.decode("ascii"),
                relative_bytes.decode("utf-8", errors="surrogateescape"),
            )
        )
    files: list[dict[str, object]] = []
    for mode, object_id, relative in tracked_entries:
        path = root / relative
        if mode == "160000":
            if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
                raise RuntimeError(
                    "gitlink must be represented by an empty directory in the "
                    f"deployment: {relative}"
                )
            kind = "gitlink"
            sha256 = hashlib.sha256(
                b"gitlink\0" + object_id.encode("ascii")
            ).hexdigest()
            files.append(
                {
                    "path": relative,
                    "kind": kind,
                    "git_oid": object_id,
                    "state": "empty_directory",
                    "sha256": sha256,
                }
            )
            continue
        if path.is_symlink():
            kind = "symlink"
            content = os.readlink(path).encode("utf-8", errors="surrogateescape")
            sha256 = hashlib.sha256(content).hexdigest()
        elif path.is_file():
            lfs_pointer = _git_lfs_pointer(path)
            if lfs_pointer is not None:
                sha256, size, pointer_sha256 = lfs_pointer
                files.append(
                    {
                        "path": relative,
                        "kind": "lfs",
                        "sha256": sha256,
                        "size": size,
                        "pointer_sha256": pointer_sha256,
                    }
                )
                continue
            kind = "file"
            sha256 = _sha256_file(path)
        else:
            raise RuntimeError(f"tracked path is missing or unsupported: {relative}")
        files.append({"path": relative, "kind": kind, "sha256": sha256})
    ancestors = [
        line.strip()
        for line in _git(root, "rev-list", "HEAD").decode("ascii").splitlines()
        if line.strip()
    ]
    return {
        "format": "handson.deployed_source_state",
        "version": 1,
        "commit": commit,
        "ancestors": ancestors,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_receipt(args.repo_root)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        f"wrote {output}: commit={payload['commit']} files={len(payload['files'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
