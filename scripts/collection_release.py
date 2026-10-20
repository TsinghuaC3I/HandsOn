#!/usr/bin/env python3
"""Safe release publishing and authoritative updates for collection machines."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "release/collection"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_HARDWARE_PATHS = frozenset(
    {
        "teleoperation/manus/vendor/wuji-hand-teleop/src/input_devices/openvr_input/config/openvr_input.yaml",
        "teleoperation/manus/vendor/wuji-hand-teleop/src/input_devices/pico_input/config/pico_input.yaml",
        "teleoperation/manus/vendor/wuji-hand-teleop/src/input_devices/manus_input/manus_input_py/manus_input_py/config/manus_input.yaml",
        "teleoperation/manus/vendor/wuji-hand-teleop/src/camera/config/camera_config.yaml",
        "teleoperation/manus/vendor/wuji-hand-teleop/src/camera/config/stereo_head/stereo_head_config.yaml",
        "teleoperation/manus/vendor/wuji-hand-teleop/src/output_devices/tianji_world_output/tianji_world_output/config/tianji_robot.yaml",
        "teleoperation/manus/vendor/wuji-hand-teleop/src/output_devices/tianji_output/tianji_output/config/tianji_output.yaml",
        "teleoperation/manus/config/haorui_ik.env",
    }
)

EXPLICIT_PROTECTED_PATHS = frozenset({".env", ".handson_collection.env"})
REMOTE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ReleaseError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class UsageError(ReleaseError):
    def __init__(self, detail: str) -> None:
        super().__init__("invalid_arguments", detail)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class WorktreeEntry:
    status: str
    path: str
    original_path: str | None = None
    tracked: bool = True

    @property
    def affected_paths(self) -> tuple[str, ...]:
        if self.original_path is None:
            return (self.path,)
        return (self.original_path, self.path)

    @property
    def allowed_for_publish(self) -> bool:
        return self.tracked and all(path in LOCAL_HARDWARE_PATHS for path in self.affected_paths)

    def as_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "path": self.path,
            "tracked": self.tracked,
            "allowed_for_publish": self.allowed_for_publish,
        }
        if self.original_path is not None:
            result["original_path"] = self.original_path
        return result


class RepositoryLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None

    def __enter__(self) -> "RepositoryLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise ReleaseError("busy", "another release operation holds the repository lock") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


class CollectionRelease:
    def __init__(self, root: Path, remote: str, branch: str) -> None:
        if not root.exists() or not root.is_dir():
            raise ReleaseError("invalid_root", f"repository root does not exist: {root}")
        if not REMOTE_NAME_RE.fullmatch(remote):
            raise ReleaseError("invalid_remote", f"remote must be a configured name, got: {remote!r}")
        if branch.startswith("-") or any(char.isspace() for char in branch):
            raise ReleaseError("invalid_branch", f"invalid release branch: {branch!r}")

        self.root = root.resolve()
        self.remote = remote
        self.branch = branch

        top_level = self._git(["rev-parse", "--show-toplevel"], check=False)
        if top_level.returncode != 0:
            raise ReleaseError("not_a_repository", self._failure_detail(top_level, "not a Git worktree"))
        self.root = Path(top_level.stdout.strip()).resolve()

        valid_branch = self._git(["check-ref-format", f"refs/heads/{branch}"], check=False)
        if valid_branch.returncode != 0:
            raise ReleaseError("invalid_branch", f"invalid release branch: {branch!r}")
        configured_remote = self._git(["remote", "get-url", remote], check=False)
        if configured_remote.returncode != 0:
            raise ReleaseError("unknown_remote", f"Git remote is not configured: {remote}")

        git_dir_result = self._git(["rev-parse", "--git-dir"])
        git_dir = Path(git_dir_result.stdout.strip())
        if not git_dir.is_absolute():
            git_dir = self.root / git_dir
        self.lock_path = git_dir.resolve() / "collection-release.lock"
        progress_path = str(os.environ.get("HANDSON_RELEASE_PROGRESS_PATH") or "").strip()
        self.progress_path = Path(progress_path).expanduser() if progress_path else None

    def _write_progress(self, **changes: object) -> None:
        if self.progress_path is None:
            return
        payload: dict[str, object] = {}
        try:
            current = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if isinstance(current, dict):
                payload.update(current)
        except (OSError, json.JSONDecodeError):
            pass
        payload.update(changes)
        payload["updated_at"] = time.time()
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_name(
            f".{self.progress_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.progress_path)

    @staticmethod
    def _progress_bytes(value: str, unit: str) -> int:
        factors = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
        return int(float(value) * factors[unit])

    def _record_git_progress(self, message: str) -> None:
        receiving = re.search(
            r"Receiving objects:\s*(\d+)%.*?,\s*([0-9.]+)\s*(KiB|MiB|GiB)",
            message,
        )
        if receiving:
            percent = max(0, min(100, int(receiving.group(1))))
            downloaded = self._progress_bytes(receiving.group(2), receiving.group(3))
            total = downloaded if percent >= 100 else int(downloaded * 100 / max(1, percent))
            self._write_progress(
                active=True,
                phase="updating",
                percent=percent,
                downloaded_bytes=downloaded,
                total_bytes=max(downloaded, total),
                detail="正在下载发布版本。",
            )
        elif "Resolving deltas:" in message:
            self._write_progress(
                active=True,
                phase="applying",
                percent=100,
                detail="下载完成，正在应用更新。",
            )

    def finish_progress(self, *, success: bool, detail: str) -> None:
        current: dict[str, object] = {}
        if self.progress_path is not None:
            try:
                value = json.loads(self.progress_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    current = value
            except (OSError, json.JSONDecodeError):
                pass
        downloaded = int(current.get("downloaded_bytes") or 0)
        total = int(current.get("total_bytes") or downloaded)
        self._write_progress(
            active=False,
            phase="complete" if success else "failed",
            percent=100 if success else int(current.get("percent") or 0),
            downloaded_bytes=downloaded,
            total_bytes=max(downloaded, total),
            detail=detail,
        )

    def _git_with_progress(
        self,
        command: list[str],
        *,
        environment: dict[str, str],
        input_text: str | None,
    ) -> GitResult:
        stderr_parts: list[str] = []
        progress_line = ""
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_text is not None else None,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
            )
            if input_text is not None and process.stdin is not None:
                process.stdin.write(input_text)
                process.stdin.close()
            assert process.stderr is not None
            while True:
                character = process.stderr.read(1)
                if character == "":
                    break
                stderr_parts.append(character)
                if character in "\r\n":
                    self._record_git_progress(progress_line)
                    progress_line = ""
                else:
                    progress_line += character
            if progress_line:
                self._record_git_progress(progress_line)
            returncode = process.wait()
            stdout_file.seek(0)
            stdout = stdout_file.read()
        return GitResult(returncode, stdout, "".join(stderr_parts))

    def _git(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> GitResult:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        command = [
            "git",
            "-c",
            "submodule.recurse=false",
            "-C",
            str(self.root),
            *args,
        ]
        if args and args[0] == "fetch" and self.progress_path is not None:
            result = self._git_with_progress(
                command,
                environment=environment,
                input_text=input_text,
            )
        else:
            completed = subprocess.run(
                command,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            check=False,
            )
            result = GitResult(completed.returncode, completed.stdout, completed.stderr)
        if check and result.returncode != 0:
            raise ReleaseError("git_failed", self._failure_detail(result, f"git {args[0]} failed"))
        return result

    @staticmethod
    def _failure_detail(result: GitResult, fallback: str) -> str:
        detail = (result.stderr or result.stdout).strip()
        return (detail or fallback)[:2000]

    def lock(self) -> RepositoryLock:
        return RepositoryLock(self.lock_path)

    def _head(self) -> str:
        head = self._git(["rev-parse", "--verify", "HEAD^{commit}"], check=False)
        if head.returncode != 0:
            raise ReleaseError("no_head", "repository has no committed HEAD")
        return head.stdout.strip()

    def _current_branch(self) -> str | None:
        branch = self._git(["symbolic-ref", "--quiet", "--short", "HEAD"], check=False)
        return branch.stdout.strip() if branch.returncode == 0 else None

    def _operation_in_progress(self) -> str | None:
        candidates = (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "rebase-apply",
            "rebase-merge",
            "BISECT_LOG",
        )
        for candidate in candidates:
            path_result = self._git(["rev-parse", "--git-path", candidate])
            path = Path(path_result.stdout.strip())
            if not path.is_absolute():
                path = self.root / path
            if path.exists():
                return candidate
        return None

    def _tracking_ref(self) -> str:
        return f"refs/remotes/{self.remote}/{self.branch}"

    def _cached_remote_head(self) -> str | None:
        cached = self._git(
            ["rev-parse", "--verify", f"{self._tracking_ref()}^{{commit}}"],
            check=False,
        )
        return cached.stdout.strip() if cached.returncode == 0 else None

    def _remote_head(self) -> str | None:
        ref = f"refs/heads/{self.branch}"
        self._write_progress(
            active=True,
            phase="checking",
            percent=0,
            downloaded_bytes=0,
            total_bytes=0,
            detail="正在检查远端发布版本。",
        )
        advertised = self._git(
            ["ls-remote", "--exit-code", "--heads", self.remote, ref],
            check=False,
        )
        if advertised.returncode == 2 and not advertised.stdout.strip():
            self._git(["update-ref", "-d", self._tracking_ref()], check=False)
            return None
        if advertised.returncode != 0:
            raise ReleaseError(
                "remote_unavailable",
                self._failure_detail(advertised, "could not inspect remote release branch"),
            )

        fetched = self._git(
            [
                "fetch",
                "--progress",
                "--no-tags",
                "--no-recurse-submodules",
                self.remote,
                f"+{ref}:{self._tracking_ref()}",
            ],
            check=False,
        )
        if fetched.returncode != 0:
            raise ReleaseError(
                "fetch_failed",
                self._failure_detail(fetched, "could not fetch remote release branch"),
            )
        remote_head = self._git(
            ["rev-parse", "--verify", f"{self._tracking_ref()}^{{commit}}"]
        )
        self._write_progress(
            active=True,
            phase="applying",
            percent=100,
            detail="下载完成，正在核对并应用版本。",
        )
        return remote_head.stdout.strip()

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self._git(["merge-base", "--is-ancestor", ancestor, descendant], check=False)
        if result.returncode not in (0, 1):
            raise ReleaseError("history_check_failed", self._failure_detail(result, "could not compare history"))
        return result.returncode == 0

    def _relation(self, head: str, remote_head: str | None) -> str:
        if remote_head is None:
            return "remote_missing"
        if head == remote_head:
            return "up_to_date"
        if self._is_ancestor(remote_head, head):
            return "ahead"
        if self._is_ancestor(head, remote_head):
            return "behind"
        return "diverged"

    def _worktree(self) -> list[WorktreeEntry]:
        result = self._git(
            [
                "status",
                "--porcelain=v2",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ]
        )
        records = result.stdout.split("\0")
        entries: list[WorktreeEntry] = []
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if not record:
                continue
            if record.startswith("1 "):
                fields = record.split(" ", 8)
                if len(fields) != 9:
                    raise ReleaseError("status_parse_failed", "unexpected ordinary status record")
                entries.append(WorktreeEntry(fields[1], fields[8]))
            elif record.startswith("2 "):
                fields = record.split(" ", 9)
                if len(fields) != 10 or index >= len(records):
                    raise ReleaseError("status_parse_failed", "unexpected rename status record")
                original_path = records[index]
                index += 1
                entries.append(WorktreeEntry(fields[1], fields[9], original_path))
            elif record.startswith("u "):
                fields = record.split(" ", 10)
                if len(fields) != 11:
                    raise ReleaseError("status_parse_failed", "unexpected unmerged status record")
                entries.append(WorktreeEntry("UU", fields[10]))
            elif record.startswith("? "):
                entries.append(WorktreeEntry("??", record[2:], tracked=False))
            else:
                raise ReleaseError("status_parse_failed", f"unexpected status record: {record[:40]!r}")
        return entries

    def _incoming_paths(self, remote_head: str) -> list[str]:
        result = self._git(
            ["diff", "--name-only", "-z", "--no-renames", "HEAD", remote_head]
        )
        return sorted(path for path in result.stdout.split("\0") if path)

    @staticmethod
    def _paths_overlap(first: str, second: str) -> bool:
        return first == second or first.startswith(second + "/") or second.startswith(first + "/")

    def _local_conflicts(
        self,
        entries: Iterable[WorktreeEntry],
        incoming_paths: Iterable[str],
    ) -> list[str]:
        local_paths = {
            path
            for entry in entries
            for path in entry.affected_paths
        }
        return sorted(
            incoming
            for incoming in incoming_paths
            if any(self._paths_overlap(incoming, local) for local in local_paths)
        )

    @staticmethod
    def _is_local_machine_path(path: str) -> bool:
        return bool(
            path in LOCAL_HARDWARE_PATHS
            or path in EXPLICIT_PROTECTED_PATHS
            or path.endswith("/.env")
            or path == "demos"
            or path.startswith("demos/")
        )

    def _discard_local_code_changes(
        self,
        entries: Iterable[WorktreeEntry],
    ) -> list[str]:
        """Restore repository-authoritative code while retaining machine data."""
        selected = [
            entry
            for entry in entries
            if not any(
                self._is_local_machine_path(path) for path in entry.affected_paths
            )
        ]
        if not selected:
            return []

        tracked_paths = sorted(
            {
                path
                for entry in selected
                if entry.tracked
                for path in entry.affected_paths
            }
        )
        untracked_paths = {
            path
            for entry in selected
            if not entry.tracked
            for path in entry.affected_paths
        }

        if tracked_paths:
            reset = self._git(
                ["reset", "--quiet", "HEAD", "--", *tracked_paths],
                check=False,
            )
            if reset.returncode != 0:
                raise ReleaseError(
                    "local_cleanup_failed",
                    self._failure_detail(reset, "could not reset local code changes"),
                )

            restore_paths = []
            for path in tracked_paths:
                exists_at_head = self._git(
                    ["cat-file", "-e", f"HEAD:{path}"],
                    check=False,
                )
                if exists_at_head.returncode == 0:
                    restore_paths.append(path)
                else:
                    untracked_paths.add(path)
            if restore_paths:
                restored = self._git(
                    [
                        "restore",
                        "--source=HEAD",
                        "--worktree",
                        "--",
                        *restore_paths,
                    ],
                    check=False,
                )
                if restored.returncode != 0:
                    raise ReleaseError(
                        "local_cleanup_failed",
                        self._failure_detail(
                            restored,
                            "could not restore tracked code",
                        ),
                    )

        if untracked_paths:
            cleaned = self._git(
                ["clean", "--force", "-d", "--", *sorted(untracked_paths)],
                check=False,
            )
            if cleaned.returncode != 0:
                raise ReleaseError(
                    "local_cleanup_failed",
                    self._failure_detail(
                        cleaned,
                        "could not remove untracked local code",
                    ),
                )

        remaining = [
            path
            for entry in self._worktree()
            if not any(
                self._is_local_machine_path(path) for path in entry.affected_paths
            )
            for path in entry.affected_paths
        ]
        if remaining:
            raise ReleaseError(
                "local_cleanup_failed",
                "local code paths remain after cleanup: " + ", ".join(remaining),
            )
        return sorted(
            {
                path
                for entry in selected
                for path in entry.affected_paths
            }
        )

    def _prepare_hardware_overrides(
        self,
        entries: Iterable[WorktreeEntry],
        incoming_paths: Sequence[str],
    ) -> dict[str, bytes | None]:
        """Temporarily restore conflicting hardware files for a fast-forward."""
        incoming = tuple(incoming_paths)
        selected = [
            entry
            for entry in entries
            if any(path in LOCAL_HARDWARE_PATHS for path in entry.affected_paths)
            and any(
                self._paths_overlap(path, remote_path)
                for path in entry.affected_paths
                for remote_path in incoming
            )
        ]
        if not selected:
            return {}

        hardware_paths = sorted(
            {
                path
                for entry in selected
                for path in entry.affected_paths
                if path in LOCAL_HARDWARE_PATHS
            }
        )
        overrides: dict[str, bytes | None] = {}
        for path in hardware_paths:
            absolute = self.root / path
            overrides[path] = absolute.read_bytes() if absolute.is_file() else None

        cleanup_paths = sorted(
            {path for entry in selected for path in entry.affected_paths}
        )
        reset = self._git(
            ["reset", "--quiet", "HEAD", "--", *cleanup_paths],
            check=False,
        )
        if reset.returncode != 0:
            raise ReleaseError(
                "hardware_overlay_failed",
                self._failure_detail(reset, "could not prepare local hardware overlay"),
            )

        tracked_paths = []
        untracked_paths = []
        for path in cleanup_paths:
            exists_at_head = self._git(
                ["cat-file", "-e", f"HEAD:{path}"],
                check=False,
            )
            if exists_at_head.returncode == 0:
                tracked_paths.append(path)
            else:
                untracked_paths.append(path)
        if tracked_paths:
            restored = self._git(
                ["restore", "--source=HEAD", "--worktree", "--", *tracked_paths],
                check=False,
            )
            if restored.returncode != 0:
                raise ReleaseError(
                    "hardware_overlay_failed",
                    self._failure_detail(restored, "could not restore hardware files"),
                )
        if untracked_paths:
            cleaned = self._git(
                ["clean", "--force", "-d", "--", *untracked_paths],
                check=False,
            )
            if cleaned.returncode != 0:
                raise ReleaseError(
                    "hardware_overlay_failed",
                    self._failure_detail(cleaned, "could not prepare untracked hardware files"),
                )
        return overrides

    def _restore_hardware_overrides(
        self,
        overrides: dict[str, bytes | None],
    ) -> None:
        for path, contents in overrides.items():
            absolute = self.root / path
            if contents is None:
                if absolute.exists():
                    absolute.unlink()
                continue
            absolute.parent.mkdir(parents=True, exist_ok=True)
            absolute.write_bytes(contents)

    def _protected_incoming(self, incoming_paths: Sequence[str]) -> list[str]:
        protected = {
            path
            for path in incoming_paths
            if path in EXPLICIT_PROTECTED_PATHS
            or path.endswith("/.env")
            or path == "demos"
            or path.startswith("demos/")
        }
        if incoming_paths:
            ignored = self._git(
                ["check-ignore", "--no-index", "-z", "--stdin"],
                check=False,
                input_text="\0".join(incoming_paths) + "\0",
            )
            if ignored.returncode not in (0, 1):
                raise ReleaseError(
                    "ignore_check_failed",
                    self._failure_detail(ignored, "could not check protected ignored paths"),
                )
            protected.update(path for path in ignored.stdout.split("\0") if path)
        return sorted(protected)

    def _branch_head(self, branch: str) -> str | None:
        result = self._git(
            ["rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}"],
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _changed_paths(self, old_head: str | None, new_head: str) -> list[str]:
        if old_head is None:
            result = self._git(
                ["ls-tree", "-r", "--name-only", "-z", new_head],
            )
        else:
            result = self._git(
                ["diff", "--name-only", "-z", "--no-renames", old_head, new_head],
            )
        return sorted(path for path in result.stdout.split("\0") if path)

    def _create_pin_backup(self, head: str, label: str) -> str:
        backup_ref = (
            "refs/heads/backup/collection/"
            f"{time.time_ns()}-{label}-{head[:12]}"
        )
        created = self._git(
            ["update-ref", backup_ref, head],
            check=False,
        )
        if created.returncode != 0:
            raise ReleaseError(
                "backup_failed",
                self._failure_detail(
                    created,
                    "could not preserve local history before pinning the release branch",
                ),
            )
        return backup_ref

    def _stash_for_pin(self, head: str) -> tuple[str, str]:
        message = f"handson collection pin {time.time_ns()}"
        stashed = self._git(
            ["stash", "push", "--include-untracked", "--message", message],
            check=False,
        )
        if stashed.returncode != 0:
            raise ReleaseError(
                "stash_failed",
                self._failure_detail(
                    stashed,
                    "could not preserve local changes before pinning the release branch",
                ),
            )
        stash = self._git(
            ["rev-parse", "--verify", "refs/stash^{commit}"],
            check=False,
        )
        if stash.returncode != 0:
            raise ReleaseError(
                "stash_failed",
                "Git reported local changes but did not create a recoverable stash",
            )
        stash_head = stash.stdout.strip()
        stash_ref = (
            "refs/handson/collection-stashes/"
            f"{time.time_ns()}-{head[:12]}"
        )
        preserved = self._git(
            ["update-ref", stash_ref, stash_head],
            check=False,
        )
        if preserved.returncode != 0:
            raise ReleaseError(
                "stash_backup_failed",
                self._failure_detail(
                    preserved,
                    "could not create a durable ref for stashed local changes",
                ),
            )
        return stash_head, stash_ref

    def pin(self) -> dict[str, object]:
        """Pin this worktree to the configured collection branch without data loss."""
        operation = self._operation_in_progress()
        previous_head = self._head()
        initial_branch = self._current_branch()
        entries = self._worktree()
        preserved_local_paths = sorted(
            {
                path
                for entry in entries
                for path in entry.affected_paths
            }
        )
        worktree_json = [entry.as_json() for entry in entries]

        if operation:
            return {
                "ok": False,
                "command": "pin",
                "code": "operation_in_progress",
                "detail": f"Git operation is in progress: {operation}",
                "root": str(self.root),
                "remote": self.remote,
                "branch": self.branch,
                "current_branch": initial_branch,
                "head": previous_head,
                "remote_head": None,
                "relation": "unknown",
                "operation_in_progress": operation,
                "worktree": worktree_json,
                "updated": False,
                "branch_changed": False,
                "backup_ref": None,
                "backup_refs": [],
                "stash_ref": None,
                "preserved_local_paths": preserved_local_paths,
                "publish_prompt_required": False,
                "can_collect": False,
            }

        remote_head = self._remote_head()
        target_head = self._branch_head(self.branch)
        target_relation = (
            "remote_missing"
            if remote_head is None
            else "local_missing"
            if target_head is None
            else self._relation(target_head, remote_head)
        )
        branch_mismatch = initial_branch != self.branch
        base: dict[str, object] = {
            "root": str(self.root),
            "remote": self.remote,
            "branch": self.branch,
            "current_branch": initial_branch,
            "head": previous_head,
            "remote_head": remote_head,
            "target_local_head": target_head,
            "relation": target_relation,
            "branch_mismatch": branch_mismatch,
            "operation_in_progress": None,
            "worktree": worktree_json,
            "updated": False,
            "branch_changed": False,
            "backup_ref": None,
            "backup_refs": [],
            "stash_ref": None,
            "preserved_local_paths": preserved_local_paths,
            "publish_prompt_required": False,
            "can_collect": False,
        }
        if remote_head is None:
            return {
                "ok": False,
                "command": "pin",
                "code": "remote_missing",
                "detail": "remote release branch does not exist",
                **base,
            }

        incoming_paths = (
            self._changed_paths(target_head, remote_head)
            if target_relation in {"local_missing", "behind", "diverged"}
            else []
        )
        protected = self._protected_incoming(incoming_paths)
        base.update(
            {
                "incoming_paths": incoming_paths,
                "protected_incoming_paths": protected,
            }
        )
        if protected:
            return {
                "ok": False,
                "command": "pin",
                "code": "protected_path",
                "detail": "release pin would touch protected ignored data",
                **base,
            }

        if (
            not branch_mismatch
            and (
                target_relation == "ahead"
                or (target_relation == "up_to_date" and bool(entries))
            )
        ):
            snapshot = self._snapshot(refresh_remote=False)
            return {
                "ok": True,
                "command": "pin",
                "code": "publish_prompt_required",
                "detail": (
                    "local committed or uncommitted changes may be published; "
                    "collection may continue without publishing"
                ),
                **snapshot,
                "previous_head": previous_head,
                "target_local_head": target_head,
                "branch_mismatch": False,
                "updated": False,
                "branch_changed": False,
                "backup_ref": None,
                "backup_refs": [],
                "stash_ref": None,
                "preserved_local_paths": preserved_local_paths,
                "publish_prompt_required": True,
                "can_collect": True,
            }

        if (
            not branch_mismatch
            and target_relation == "up_to_date"
            and not entries
        ):
            snapshot = self._snapshot(refresh_remote=False)
            return {
                "ok": True,
                "command": "pin",
                "code": "pinned",
                "detail": "worktree is already pinned to the current release",
                **snapshot,
                "previous_head": previous_head,
                "target_local_head": target_head,
                "branch_mismatch": False,
                "updated": False,
                "branch_changed": False,
                "backup_ref": None,
                "backup_refs": [],
                "stash_ref": None,
                "preserved_local_paths": [],
                "publish_prompt_required": False,
                "can_collect": True,
            }

        backup_refs: list[str] = []
        if target_relation == "diverged" and target_head is not None:
            backup_refs.append(
                self._create_pin_backup(target_head, "diverged")
            )
        if branch_mismatch and initial_branch is None:
            backup_refs.append(
                self._create_pin_backup(previous_head, "detached")
            )

        stash_head: str | None = None
        stash_ref: str | None = None
        if entries:
            stash_head, stash_ref = self._stash_for_pin(previous_head)
            remaining = self._worktree()
            if remaining:
                return {
                    "ok": False,
                    "command": "pin",
                    "code": "stash_incomplete",
                    "detail": (
                        "some local changes could not be stashed safely; "
                        f"recoverable stash retained at {stash_ref}"
                    ),
                    **base,
                    "backup_ref": backup_refs[0] if backup_refs else None,
                    "backup_refs": backup_refs,
                    "stash_ref": stash_ref,
                    "worktree": [entry.as_json() for entry in remaining],
                }

        branch_changed = False
        pin_result: GitResult | None = None
        if branch_mismatch:
            if target_relation == "local_missing":
                pin_result = self._git(
                    [
                        "switch",
                        "--create",
                        self.branch,
                        "--track",
                        f"{self.remote}/{self.branch}",
                    ],
                    check=False,
                )
            elif target_relation == "diverged":
                pin_result = self._git(
                    ["switch", "--force-create", self.branch, remote_head],
                    check=False,
                )
            else:
                pin_result = self._git(
                    ["switch", self.branch],
                    check=False,
                )
            branch_changed = pin_result.returncode == 0
        elif target_relation == "diverged":
            pin_result = self._git(
                ["switch", "--force-create", self.branch, remote_head],
                check=False,
            )

        if pin_result is not None and pin_result.returncode != 0:
            return {
                "ok": False,
                "command": "pin",
                "code": "branch_pin_failed",
                "detail": self._failure_detail(
                    pin_result,
                    "could not switch to the pinned collection branch",
                ),
                **base,
                "backup_ref": backup_refs[0] if backup_refs else None,
                "backup_refs": backup_refs,
                "stash_ref": stash_ref,
            }

        if target_relation == "behind":
            forwarded = self._git(
                ["merge", "--ff-only", "--no-edit", remote_head],
                check=False,
            )
            if forwarded.returncode != 0:
                return {
                    "ok": False,
                    "command": "pin",
                    "code": "fast_forward_failed",
                    "detail": self._failure_detail(
                        forwarded,
                        "could not fast-forward the pinned collection branch",
                    ),
                    **base,
                    "branch_changed": branch_changed,
                    "backup_ref": backup_refs[0] if backup_refs else None,
                    "backup_refs": backup_refs,
                    "stash_ref": stash_ref,
                }

        upstream = self._git(
            [
                "branch",
                "--set-upstream-to",
                f"{self.remote}/{self.branch}",
                self.branch,
            ],
            check=False,
        )
        if upstream.returncode != 0:
            return {
                "ok": False,
                "command": "pin",
                "code": "upstream_pin_failed",
                "detail": self._failure_detail(
                    upstream,
                    "could not pin the collection branch upstream",
                ),
                **base,
                "branch_changed": branch_changed,
                "backup_ref": backup_refs[0] if backup_refs else None,
                "backup_refs": backup_refs,
                "stash_ref": stash_ref,
            }

        if stash_head is not None:
            replayed = self._git(
                ["stash", "apply", "--index", stash_head],
                check=False,
            )
            if replayed.returncode != 0:
                current_head = self._head()
                return {
                    "ok": False,
                    "command": "pin",
                    "code": "stash_replay_conflict",
                    "detail": (
                        "release branch was pinned, but local changes conflict; "
                        f"resolve them or recover from {stash_ref}"
                    ),
                    **base,
                    "current_branch": self._current_branch(),
                    "head": current_head,
                    "relation": self._relation(current_head, remote_head),
                    "worktree": [
                        entry.as_json() for entry in self._worktree()
                    ],
                    "updated": current_head != previous_head,
                    "branch_changed": branch_changed,
                    "backup_ref": backup_refs[0] if backup_refs else None,
                    "backup_refs": backup_refs,
                    "stash_ref": stash_ref,
                    "publish_prompt_required": False,
                    "can_collect": False,
                }
            current_stash = self._git(
                ["rev-parse", "--verify", "refs/stash^{commit}"],
                check=False,
            )
            if (
                current_stash.returncode == 0
                and current_stash.stdout.strip() == stash_head
            ):
                self._git(
                    ["stash", "drop", "--quiet", "stash@{0}"],
                    check=False,
                )

        final_snapshot = self._snapshot(refresh_remote=False)
        final_head = str(final_snapshot["head"])
        final_relation = str(final_snapshot["relation"])
        final_worktree = self._worktree()
        publish_prompt_required = (
            final_relation == "ahead" or bool(final_worktree)
        )
        return {
            "ok": True,
            "command": "pin",
            "code": (
                "publish_prompt_required"
                if publish_prompt_required
                else "pinned"
            ),
            "detail": (
                "local changes may be published; collection may continue without publishing"
                if publish_prompt_required
                else "worktree pinned to the current collection release"
            ),
            **final_snapshot,
            "previous_head": previous_head,
            "target_local_head": final_head,
            "branch_mismatch": False,
            "incoming_paths": [],
            "updated": final_head != previous_head,
            "branch_changed": branch_changed,
            "backup_ref": backup_refs[0] if backup_refs else None,
            "backup_refs": backup_refs,
            "stash_ref": stash_ref,
            "preserved_local_paths": preserved_local_paths,
            "publish_prompt_required": publish_prompt_required,
            "can_collect": True,
        }

    def _snapshot(
        self,
        *,
        refresh_remote: bool = True,
        tolerate_remote_failure: bool = False,
    ) -> dict[str, object]:
        operation = self._operation_in_progress()
        head = self._head()
        remote_error: ReleaseError | None = None
        remote_refreshed = refresh_remote
        try:
            remote_head = (
                self._remote_head() if refresh_remote else self._cached_remote_head()
            )
            if remote_head is None and not refresh_remote:
                remote_head = self._remote_head()
                remote_refreshed = True
        except ReleaseError as exc:
            if not tolerate_remote_failure:
                raise
            remote_error = exc
            remote_head = self._cached_remote_head()
        relation = (
            "unknown"
            if remote_error and remote_head is None
            else self._relation(head, remote_head)
        )
        worktree = self._worktree()
        incoming = self._incoming_paths(remote_head) if relation == "behind" and remote_head else []
        conflicts = self._local_conflicts(worktree, incoming)
        protected = self._protected_incoming(incoming)
        blocking_publish = [entry.path for entry in worktree if not entry.allowed_for_publish]

        can_publish = (
            remote_error is None
            and operation is None
            and not blocking_publish
            and relation in {"remote_missing", "up_to_date", "ahead"}
        )
        can_update = (
            remote_error is None
            and operation is None
            and relation in {"up_to_date", "behind"}
            and not protected
        )
        return {
            "root": str(self.root),
            "remote": self.remote,
            "branch": self.branch,
            "current_branch": self._current_branch(),
            "head": head,
            "remote_head": remote_head,
            "relation": relation,
            "available": True,
            "remote_available": remote_error is None,
            "remote_refreshed": remote_refreshed and remote_error is None,
            "remote_error_code": remote_error.code if remote_error else None,
            "remote_error_detail": remote_error.detail if remote_error else None,
            "operation_in_progress": operation,
            "worktree": [entry.as_json() for entry in worktree],
            "blocking_publish_paths": blocking_publish,
            "incoming_paths": incoming,
            "conflicting_paths": conflicts,
            "protected_incoming_paths": protected,
            "can_publish": can_publish,
            "can_update": can_update,
        }

    def status(self, *, refresh: bool = False) -> dict[str, object]:
        snapshot = self._snapshot(
            refresh_remote=refresh,
            tolerate_remote_failure=True,
        )
        if snapshot["remote_available"]:
            code = str(snapshot["relation"])
            detail = "repository and release branch inspected"
        else:
            code = "remote_status_unavailable"
            remote_detail = str(snapshot.get("remote_error_detail") or "remote check failed")
            detail = (
                "local version is available; remote release status is temporarily "
                f"unknown: {remote_detail}"
            )
        return {
            "ok": True,
            "command": "status",
            "code": code,
            "detail": detail,
            **snapshot,
        }

    def check(self, action: str) -> dict[str, object]:
        snapshot = self._snapshot()
        ready = bool(snapshot[f"can_{action}"])
        if ready:
            code = "ready"
            detail = f"repository is ready to {action}"
        else:
            code, detail = self._blocked_reason(action, snapshot)
        return {
            "ok": ready,
            "command": "check",
            "action": action,
            "code": code,
            "detail": detail,
            **snapshot,
        }

    @staticmethod
    def _blocked_reason(action: str, snapshot: dict[str, object]) -> tuple[str, str]:
        operation = snapshot["operation_in_progress"]
        if operation:
            return "operation_in_progress", f"Git operation is in progress: {operation}"
        if action == "publish":
            blocked = snapshot["blocking_publish_paths"]
            if blocked:
                return "dirty_worktree", "uncommitted non-hardware paths block publishing"
            relation = snapshot["relation"]
            if relation == "behind":
                return "remote_ahead", "remote release contains commits not present in HEAD"
            if relation == "diverged":
                return "diverged", "remote release is not an ancestor of HEAD"
        else:
            protected = snapshot["protected_incoming_paths"]
            if protected:
                return "protected_path", "incoming release would touch protected ignored data"
            relation = snapshot["relation"]
            if relation == "remote_missing":
                return "remote_missing", "remote release branch does not exist"
            if relation == "ahead":
                return "local_ahead", "local HEAD contains commits not in the release branch"
            if relation == "diverged":
                return "diverged", "local HEAD cannot fast-forward to the release branch"
        return "not_ready", f"repository is not ready to {action}"

    def publish(self) -> dict[str, object]:
        snapshot = self._snapshot()
        if not snapshot["can_publish"]:
            code, detail = self._blocked_reason("publish", snapshot)
            return {
                "ok": False,
                "command": "publish",
                "code": code,
                "detail": detail,
                **snapshot,
            }
        if snapshot["relation"] == "up_to_date":
            return {
                "ok": True,
                "command": "publish",
                "code": "already_published",
                "detail": "remote release already points to HEAD",
                **snapshot,
            }

        refspec = f"HEAD:refs/heads/{self.branch}"
        pushed = self._git(
            [
                "push",
                "--porcelain",
                "--recurse-submodules=no",
                self.remote,
                refspec,
            ],
            check=False,
        )
        if pushed.returncode != 0:
            raise ReleaseError(
                "push_rejected",
                self._failure_detail(pushed, "remote rejected release push"),
            )
        remote_after = self._git(
            ["ls-remote", "--exit-code", "--heads", self.remote, f"refs/heads/{self.branch}"]
        )
        advertised_head = remote_after.stdout.split(None, 1)[0] if remote_after.stdout.strip() else ""
        if advertised_head != snapshot["head"]:
            raise ReleaseError("publish_verification_failed", "remote release does not point to published HEAD")
        return {
            "ok": True,
            "command": "publish",
            "code": "published",
            "detail": "committed HEAD published to the release branch",
            **snapshot,
            "remote_head": advertised_head,
            "relation": "up_to_date",
            "can_update": True,
        }

    def update(self) -> dict[str, object]:
        snapshot = self._snapshot()
        if not snapshot["can_update"]:
            code, detail = self._blocked_reason("update", snapshot)
            return {
                "ok": False,
                "command": "update",
                "code": code,
                "detail": detail,
                **snapshot,
            }
        previous_head = str(snapshot["head"])
        discarded_local_paths = self._discard_local_code_changes(self._worktree())
        remaining_worktree = self._worktree()
        remaining_blockers = [
            entry.path for entry in remaining_worktree if not entry.allowed_for_publish
        ]

        if snapshot["relation"] == "up_to_date" and not discarded_local_paths:
            return {
                "ok": True,
                "command": "update",
                "code": "up_to_date",
                "detail": "local HEAD already matches the release branch",
                **snapshot,
            }
        if snapshot["relation"] == "up_to_date":
            return {
                "ok": True,
                "command": "update",
                "code": "updated",
                "detail": "local code restored to the authoritative release",
                **snapshot,
                "previous_head": previous_head,
                "discarded_local_paths": discarded_local_paths,
                "applied_paths": discarded_local_paths,
                "blocking_publish_paths": remaining_blockers,
                "can_publish": not remaining_blockers,
                "can_update": True,
            }

        remote_head = str(snapshot["remote_head"])
        hardware_overrides = self._prepare_hardware_overrides(
            self._worktree(),
            list(snapshot["incoming_paths"]),
        )
        try:
            merged = self._git(
                ["merge", "--ff-only", "--no-edit", remote_head],
                check=False,
            )
        finally:
            self._restore_hardware_overrides(hardware_overrides)
        if merged.returncode != 0:
            raise ReleaseError(
                "update_failed",
                self._failure_detail(merged, "fast-forward update failed"),
            )
        updated_head = self._head()
        if updated_head != remote_head:
            raise ReleaseError("update_verification_failed", "HEAD does not match fetched release after update")
        remaining_worktree = self._worktree()
        remaining_blockers = [
            entry.path for entry in remaining_worktree if not entry.allowed_for_publish
        ]
        return {
            "ok": True,
            "command": "update",
            "code": "updated",
            "detail": "local worktree fast-forwarded to the release branch",
            **snapshot,
            "previous_head": previous_head,
            "head": updated_head,
            "relation": "up_to_date",
            "incoming_paths": [],
            "discarded_local_paths": discarded_local_paths,
            "applied_paths": sorted(
                set(snapshot["incoming_paths"]) | set(discarded_local_paths)
            ),
            "preserved_local_paths": sorted(hardware_overrides),
            "blocking_publish_paths": remaining_blockers,
            "can_publish": not remaining_blockers,
            "can_update": True,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("HANDSON_RELEASE_ROOT", str(DEFAULT_ROOT)),
        help="fixed Git worktree root (env: HANDSON_RELEASE_ROOT)",
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get("HANDSON_RELEASE_REMOTE", DEFAULT_REMOTE),
        help="configured Git remote name (env: HANDSON_RELEASE_REMOTE)",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("HANDSON_RELEASE_BRANCH", DEFAULT_BRANCH),
        help="fixed release branch (env: HANDSON_RELEASE_BRANCH)",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="refresh remote state (accepted for collection-server compatibility)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="inspect local and remote release state")
    check = subparsers.add_parser("check", help="run publish or update preflight checks")
    check.add_argument("--action", choices=("publish", "update"), default="publish")
    subparsers.add_parser("publish", help="publish committed HEAD to the release branch")
    subparsers.add_parser(
        "pin",
        help="safely pin this worktree to the configured collection branch",
    )
    subparsers.add_parser(
        "update",
        help="restore authoritative code and fast-forward to the release branch",
    )
    return parser


def emit(payload: dict[str, object]) -> None:
    relation = payload.get("relation")
    payload.setdefault("success", bool(payload.get("ok")))
    payload.setdefault("updated", payload.get("code") == "updated")
    payload.setdefault("local_sha", payload.get("head"))
    payload.setdefault("remote_sha", payload.get("remote_head"))
    payload.setdefault("available", bool(payload.get("local_sha")))
    payload.setdefault("update_available", relation == "behind")
    payload.setdefault("diverged", relation == "diverged")
    payload.setdefault("can_publish", False)
    payload.setdefault("can_update", False)
    payload.setdefault("branch_changed", False)
    payload.setdefault("backup_ref", None)
    payload.setdefault("preserved_local_paths", [])
    payload.setdefault("publish_prompt_required", False)
    payload.setdefault("can_collect", False)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    command: str | None = None
    release: CollectionRelease | None = None
    try:
        args = build_parser().parse_args(argv)
        command = args.command
        release = CollectionRelease(Path(args.root), args.remote, args.branch)
        with release.lock():
            if args.command == "status":
                payload = release.status(refresh=args.fetch)
            elif args.command == "check":
                payload = release.check(args.action)
            elif args.command == "publish":
                payload = release.publish()
            elif args.command == "pin":
                payload = release.pin()
            else:
                payload = release.update()
    except ReleaseError as exc:
        payload = {
            "ok": False,
            "command": command,
            "code": exc.code,
            "detail": exc.detail,
        }
    except KeyboardInterrupt:
        payload = {
            "ok": False,
            "command": command,
            "code": "interrupted",
            "detail": "operation interrupted",
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "command": command,
            "code": "internal_error",
            "detail": str(exc)[:2000],
        }
    if release is not None:
        release.finish_progress(
            success=bool(payload.get("ok")),
            detail=str(payload.get("detail") or payload.get("code") or ""),
        )
    emit(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
