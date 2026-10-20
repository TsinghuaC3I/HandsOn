"""Cross-process serialization for one episode's publishable contents."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator


LOCK_DIRECTORY_NAME = ".episode_publication_locks"
INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV = (
    "HANDSON_INHERITED_EPISODE_PUBLICATION_LOCK_FD"
)
# Compatibility name for callers that imported the first implementation.  Its
# value now names the inherited-FD capability, never a PID assertion.
VERIFIED_PARENT_EPISODE_PUBLICATION_LOCK_ENV = (
    INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV
)


class EpisodePublicationLockBusy(RuntimeError):
    """Raised when another process is mutating or uploading an episode."""

    def __init__(self, demo_dir: Path, lock_path: Path):
        self.demo_dir = demo_dir
        self.lock_path = lock_path
        super().__init__(
            f"episode publication is busy: {demo_dir} (lock: {lock_path})"
        )


@dataclass(frozen=True)
class EpisodePublicationLockLease:
    """A held lock whose descriptor may be inherited by one child process."""

    demo_dir: Path
    lock_path: Path
    handle: IO[str]

    def fileno(self) -> int:
        return self.handle.fileno()


def episode_publication_lock_path(demo_dir: str | Path) -> Path:
    """Return a stable lock path outside the directory that gets uploaded."""

    resolved = Path(demo_dir).expanduser().resolve()
    digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:20]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved.name).strip("._")
    safe_name = safe_name[:80] or "episode"
    return resolved.parent / LOCK_DIRECTORY_NAME / f"{safe_name}-{digest}.lock"


@contextmanager
def episode_publication_lock(
    demo_dir: str | Path,
    *,
    blocking: bool = False,
) -> Iterator[EpisodePublicationLockLease]:
    """Hold one exclusive episode lock across gate, mutation, and cloud PUTs.

    The lock file is a sibling-side hidden control file, never a child of the
    episode, so an upload directory walk cannot accidentally publish it.
    """

    resolved = Path(demo_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"episode directory not found: {resolved}")
    lock_path = episode_publication_lock_path(resolved)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle: IO[str] = lock_path.open("a+", encoding="utf-8")
    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        try:
            fcntl.flock(handle.fileno(), operation)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise EpisodePublicationLockBusy(resolved, lock_path) from exc
        try:
            yield EpisodePublicationLockLease(
                demo_dir=resolved,
                lock_path=lock_path,
                handle=handle,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    except BaseException:
        if not handle.closed:
            handle.close()
        raise


def inherited_episode_publication_lock_fd(
    demo_dir: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    inherited_fd: int | None = None,
) -> int | None:
    """Return a verified inherited descriptor for this episode's held lock.

    The descriptor is a capability passed by ``subprocess.Popen(pass_fds=...)``.
    It must identify the exact current lock-file inode.  Reasserting an
    exclusive non-blocking flock on that same open-file description succeeds
    for a genuine parent-held inherited lock, but fails when the environment
    points at an unlocked descriptor while a foreign process owns the lock.
    Rechecking the inode after ``flock`` also closes a lock-path replacement
    race.  An invalid capability is ignored and the caller acquires its own
    ordinary lock instead.
    """

    environment = os.environ if environ is None else environ
    if inherited_fd is None:
        raw_fd = environment.get(
            INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV,
            "",
        )
        try:
            inherited_fd = int(raw_fd)
        except (TypeError, ValueError):
            return None
    if type(inherited_fd) is not int or inherited_fd < 0:
        return None

    lock_path = episode_publication_lock_path(demo_dir)

    def descriptor_matches_current_path() -> bool:
        try:
            descriptor_stat = os.fstat(inherited_fd)
            path_stat = os.lstat(lock_path)
        except OSError:
            return False
        return bool(
            stat.S_ISREG(descriptor_stat.st_mode)
            and stat.S_ISREG(path_stat.st_mode)
            and not stat.S_ISLNK(path_stat.st_mode)
            and descriptor_stat.st_dev == path_stat.st_dev
            and descriptor_stat.st_ino == path_stat.st_ino
        )

    if not descriptor_matches_current_path():
        return None
    try:
        fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return None
    if not descriptor_matches_current_path():
        return None
    return inherited_fd


def verified_parent_holds_episode_publication_lock(
    demo_dir: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    parent_pid: int | None = None,
) -> bool:
    """Compatibility wrapper for the inherited-FD ownership proof."""

    del parent_pid
    return inherited_episode_publication_lock_fd(
        demo_dir,
        environ=environ,
    ) is not None


@contextmanager
def episode_publication_scope(
    demo_dir: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    parent_pid: int | None = None,
) -> Iterator[None]:
    """Inherit a verified parent lock or acquire one for a direct caller."""

    del parent_pid
    if inherited_episode_publication_lock_fd(
        demo_dir,
        environ=environ,
    ) is not None:
        yield
        return
    with episode_publication_lock(demo_dir, blocking=True):
        yield


__all__ = [
    "EpisodePublicationLockBusy",
    "EpisodePublicationLockLease",
    "INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV",
    "LOCK_DIRECTORY_NAME",
    "VERIFIED_PARENT_EPISODE_PUBLICATION_LOCK_ENV",
    "episode_publication_lock",
    "episode_publication_lock_path",
    "episode_publication_scope",
    "inherited_episode_publication_lock_fd",
    "verified_parent_holds_episode_publication_lock",
]
