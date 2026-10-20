"""Deterministic cloud transport for a replay.zarr directory.

The on-disk episode keeps ``replay.zarr/`` as its canonical, writable form.
Cloud uploads replace the directory's many small objects with one derived
``replay.zarr.zip`` object.  Other episode files remain independent objects.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPLAY_DIRECTORY_NAME = "replay.zarr"
REPLAY_ARCHIVE_NAME = "replay.zarr.zip"
REPLAY_ARCHIVE_FORMAT = "handson.replay_zarr_zip"
REPLAY_ARCHIVE_VERSION = 1
REPLAY_TREE_HASH_ALGORITHM = "sha256_tree_v1"
_COPY_BUFFER_BYTES = 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class ReplayArchiveError(RuntimeError):
    """Raised when a replay archive cannot be produced without ambiguity."""


@dataclass(frozen=True)
class UploadEntry:
    """One local file and the episode-relative object path used in the cloud."""

    local_path: Path
    relative_path: str


@dataclass(frozen=True)
class EpisodeUploadPlan:
    """Cloud objects and replay transport metadata for one episode."""

    files: tuple[UploadEntry, ...]
    replay_transport: dict[str, Any]
    snapshot_root: Path
    snapshot_episode_root: Path
    source_root: Path
    _guard: "_EpisodeSnapshotGuard" = field(repr=False, compare=False)

    def verify_source_unchanged(self) -> None:
        """Fail unless the frozen files and their source remain approved.

        Uploaders call this immediately before their final remote commit.  The
        context manager repeats the check on normal exit, so a caller cannot
        accidentally turn an early check into a lasting authorization.
        """

        self._guard.verify()


@dataclass(frozen=True)
class _TreeEntryIdentity:
    relative_path: str
    kind: str
    stat_signature: tuple[int, ...]
    sha256: str = ""


@dataclass(frozen=True)
class _FileIdentity:
    stat_signature: tuple[int, ...]
    sha256: str


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ReplayArchiveError(
            "Secure episode snapshots require O_DIRECTORY and O_NOFOLLOW"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(
        os, "O_CLOEXEC", 0
    )


def _file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReplayArchiveError(
            "Secure episode snapshots require O_NOFOLLOW"
        )
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_source_root(path: Path) -> int:
    try:
        descriptor = os.open(path, _directory_open_flags())
    except OSError as exc:
        raise ReplayArchiveError(
            f"Cannot securely open episode directory {path}: {exc}"
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ReplayArchiveError(f"Episode directory not found: {path}")
    return descriptor


def _require_named_identity(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> os.stat_result:
    try:
        current = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ReplayArchiveError(f"{label} changed while snapshotting: {exc}") from exc
    if _stat_signature(current) != _stat_signature(expected):
        raise ReplayArchiveError(f"{label} changed while snapshotting")
    return current


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while creating private upload snapshot")
        view = view[written:]


def _capture_regular_file_from_fd(
    source_directory_fd: int,
    name: str,
    initial_named: os.stat_result,
    *,
    label: str,
    destination: Path | None,
) -> _TreeEntryIdentity:
    try:
        source_fd = os.open(
            name,
            _file_open_flags(),
            dir_fd=source_directory_fd,
        )
    except OSError as exc:
        raise ReplayArchiveError(
            f"Cannot open {label} without following links: {exc}"
        ) from exc
    destination_fd = -1
    try:
        initial_open = os.fstat(source_fd)
        if (
            not stat.S_ISREG(initial_open.st_mode)
            or _stat_signature(initial_open) != _stat_signature(initial_named)
        ):
            raise ReplayArchiveError(f"{label} changed before snapshot copy")
        if destination is not None:
            destination_fd = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(source_fd, _COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if destination_fd >= 0:
                _write_all(destination_fd, chunk)
        after_open = os.fstat(source_fd)
        if (
            byte_count != int(initial_open.st_size)
            or _stat_signature(after_open) != _stat_signature(initial_open)
        ):
            raise ReplayArchiveError(f"{label} changed during snapshot copy")
        _require_named_identity(
            source_directory_fd,
            name,
            initial_open,
            label=label,
        )
        return _TreeEntryIdentity(
            relative_path=label,
            kind="file",
            stat_signature=_stat_signature(initial_open),
            sha256=digest.hexdigest(),
        )
    except OSError as exc:
        raise ReplayArchiveError(
            f"Cannot capture episode file {label}: {exc}"
        ) from exc
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)


def _capture_tree_from_directory_fd(
    source_fd: int,
    *,
    destination: Path | None = None,
    relative_parts: tuple[str, ...] = (),
) -> tuple[_TreeEntryIdentity, ...]:
    initial_root = os.fstat(source_fd)
    if not stat.S_ISDIR(initial_root.st_mode):
        raise ReplayArchiveError("Episode tree contains a non-directory root")
    root_label = "/".join(relative_parts) or "."
    entries: list[_TreeEntryIdentity] = [
        _TreeEntryIdentity(
            relative_path=root_label,
            kind="directory",
            stat_signature=_stat_signature(initial_root),
        )
    ]
    try:
        names = sorted(os.listdir(source_fd))
    except OSError as exc:
        raise ReplayArchiveError(f"Cannot enumerate episode tree: {exc}") from exc
    for name in names:
        if name in {"", ".", ".."} or "/" in name:
            raise ReplayArchiveError("Episode contains an unsafe path component")
        relative = (*relative_parts, name)
        label = "/".join(relative)
        try:
            named = os.stat(
                name,
                dir_fd=source_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ReplayArchiveError(
                f"Cannot inspect episode entry {label}: {exc}"
            ) from exc
        if stat.S_ISLNK(named.st_mode):
            raise ReplayArchiveError(
                f"Episode contains a symlink, which is not allowed: {label}"
            )
        if stat.S_ISDIR(named.st_mode):
            try:
                child_fd = os.open(
                    name,
                    _directory_open_flags(),
                    dir_fd=source_fd,
                )
            except OSError as exc:
                raise ReplayArchiveError(
                    f"Cannot open episode directory {label} without following "
                    f"links: {exc}"
                ) from exc
            try:
                if _stat_signature(os.fstat(child_fd)) != _stat_signature(named):
                    raise ReplayArchiveError(
                        f"Episode directory {label} changed before snapshot copy"
                    )
                child_destination = (
                    destination / name if destination is not None else None
                )
                if child_destination is not None:
                    child_destination.mkdir(mode=0o700)
                entries.extend(
                    _capture_tree_from_directory_fd(
                        child_fd,
                        destination=child_destination,
                        relative_parts=relative,
                    )
                )
                if _stat_signature(os.fstat(child_fd)) != _stat_signature(named):
                    raise ReplayArchiveError(
                        f"Episode directory {label} changed during snapshot copy"
                    )
                _require_named_identity(
                    source_fd,
                    name,
                    named,
                    label=f"Episode directory {label}",
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(named.st_mode):
            entries.append(
                _capture_regular_file_from_fd(
                    source_fd,
                    name,
                    named,
                    label=label,
                    destination=(
                        destination / name
                        if destination is not None
                        else None
                    ),
                )
            )
        else:
            raise ReplayArchiveError(
                f"Episode contains a non-regular entry: {label}"
            )
    if _stat_signature(os.fstat(source_fd)) != _stat_signature(initial_root):
        raise ReplayArchiveError(
            f"Episode directory {root_label} changed during snapshot copy"
        )
    return tuple(entries)


def _file_identity(path: Path) -> _FileIdentity:
    try:
        parent_fd = os.open(path.parent, _directory_open_flags())
    except OSError as exc:
        raise ReplayArchiveError(
            f"Cannot securely inspect snapshot file {path}: {exc}"
        ) from exc
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(named.st_mode):
            raise ReplayArchiveError(
                f"Snapshot upload entry is not regular: {path}"
            )
        file_fd = os.open(path.name, _file_open_flags(), dir_fd=parent_fd)
        try:
            opened = os.fstat(file_fd)
            if _stat_signature(opened) != _stat_signature(named):
                raise ReplayArchiveError(f"Snapshot upload entry changed: {path}")
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(file_fd, _COPY_BUFFER_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(file_fd)
            if (
                size != int(opened.st_size)
                or _stat_signature(after) != _stat_signature(opened)
            ):
                raise ReplayArchiveError(f"Snapshot upload entry changed: {path}")
            _require_named_identity(
                parent_fd,
                path.name,
                opened,
                label=f"Snapshot upload entry {path}",
            )
            return _FileIdentity(_stat_signature(opened), digest.hexdigest())
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)


def _root_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReplayArchiveError(f"Snapshot/source root is not a directory: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def _freeze_snapshot_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ReplayArchiveError(
                    f"Private snapshot contains a non-regular file: {path}"
                )
            path.chmod(0o400)
        for name in directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise ReplayArchiveError(
                    f"Private snapshot contains a non-directory: {path}"
                )
            path.chmod(0o500)
    root.chmod(0o500)


def _thaw_snapshot_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, _files in os.walk(root, topdown=True, followlinks=False):
        Path(current).chmod(0o700)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(0o700)


@contextmanager
def _private_upload_snapshot_directory(
    episode_name: str,
) -> Iterator[tuple[Path, Path]]:
    """Create a private root and make cleanup robust after read-only freeze."""

    with tempfile.TemporaryDirectory(
        prefix="handson-cloud-upload-snapshot-"
    ) as temp_dir:
        snapshot_root = Path(temp_dir)
        snapshot_episode = snapshot_root / episode_name
        snapshot_episode.mkdir(mode=0o700)
        try:
            yield snapshot_root, snapshot_episode
        finally:
            _thaw_snapshot_tree(snapshot_episode)


class _EpisodeSnapshotGuard:
    __slots__ = (
        "source_fd",
        "source_identity",
        "source_root",
        "source_root_identity",
        "snapshot_episode_root",
        "snapshot_root_identity",
        "upload_identities",
    )

    def __init__(
        self,
        *,
        source_fd: int,
        source_identity: tuple[_TreeEntryIdentity, ...],
        source_root: Path,
        snapshot_episode_root: Path,
        upload_entries: tuple[UploadEntry, ...],
    ) -> None:
        self.source_fd = source_fd
        self.source_identity = source_identity
        self.source_root = source_root
        source_stat = os.fstat(source_fd)
        self.source_root_identity = (
            int(source_stat.st_dev),
            int(source_stat.st_ino),
        )
        self.snapshot_episode_root = snapshot_episode_root
        self.snapshot_root_identity = _root_identity(snapshot_episode_root)
        self.upload_identities = tuple(
            (entry.local_path, _file_identity(entry.local_path))
            for entry in upload_entries
        )

    def _verify_root_name(self) -> None:
        try:
            current = self.source_root.lstat()
        except OSError as exc:
            raise ReplayArchiveError(
                f"Source episode root changed after upload snapshot: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or (int(current.st_dev), int(current.st_ino))
            != self.source_root_identity
        ):
            raise ReplayArchiveError(
                "Source episode root inode changed after upload snapshot"
            )

    def verify(self) -> None:
        if _root_identity(self.snapshot_episode_root) != self.snapshot_root_identity:
            raise ReplayArchiveError("Private upload snapshot root changed")
        for path, expected in self.upload_identities:
            if _file_identity(path) != expected:
                raise ReplayArchiveError(
                    f"Private upload snapshot file changed: {path.name}"
                )

        self._verify_root_name()
        current_identity = _capture_tree_from_directory_fd(self.source_fd)
        if current_identity != self.source_identity:
            raise ReplayArchiveError(
                "Source episode changed after the private upload snapshot was created"
            )
        self._verify_root_name()

    def close(self) -> None:
        if self.source_fd >= 0:
            os.close(self.source_fd)
            self.source_fd = -1


def _regular_replay_files(replay_dir: Path) -> list[Path]:
    replay_dir = replay_dir.expanduser()
    if replay_dir.is_symlink() or not replay_dir.is_dir():
        raise ReplayArchiveError(
            f"Missing regular {REPLAY_DIRECTORY_NAME} directory: {replay_dir}"
        )
    try:
        candidates = sorted(replay_dir.rglob("*"))
    except OSError as exc:
        raise ReplayArchiveError(
            f"Cannot enumerate {REPLAY_DIRECTORY_NAME}: {exc}"
        ) from exc
    symlinks = [path for path in candidates if path.is_symlink()]
    if symlinks:
        relative = symlinks[0].relative_to(replay_dir).as_posix()
        raise ReplayArchiveError(
            f"{REPLAY_DIRECTORY_NAME} contains a symlink, which is not allowed: "
            f"{relative}"
        )
    unsupported = [
        path
        for path in candidates
        if not path.is_dir() and not path.is_file()
    ]
    if unsupported:
        relative = unsupported[0].relative_to(replay_dir).as_posix()
        raise ReplayArchiveError(
            f"{REPLAY_DIRECTORY_NAME} contains a non-regular entry: {relative}"
        )
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise ReplayArchiveError(f"{REPLAY_DIRECTORY_NAME} contains no files")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_tree_identity(replay_dir: str | Path) -> dict[str, Any]:
    """Hash a Zarr tree using the certification-compatible v1 framing."""

    root = Path(replay_dir)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    try:
        for path in _regular_replay_files(root):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            size = int(path.stat().st_size)
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(
                    lambda: stream.read(_COPY_BUFFER_BYTES),
                    b"",
                ):
                    digest.update(chunk)
            file_count += 1
            total_bytes += size
    except OSError as exc:
        raise ReplayArchiveError(
            f"Cannot hash {REPLAY_DIRECTORY_NAME} tree {root}: {exc}"
        ) from exc
    return {
        "algorithm": REPLAY_TREE_HASH_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "bytes": total_bytes,
    }


def create_replay_archive(
    replay_dir: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    """Create a deterministic, uncompressed ZipStore-compatible archive."""

    source = Path(replay_dir).expanduser()
    destination = Path(archive_path).expanduser()
    try:
        source_resolved = source.resolve()
        destination_resolved = destination.resolve()
    except OSError as exc:
        raise ReplayArchiveError(f"Cannot resolve replay archive path: {exc}") from exc
    if source_resolved in destination_resolved.parents:
        raise ReplayArchiveError(
            f"{REPLAY_ARCHIVE_NAME} must be created outside {REPLAY_DIRECTORY_NAME}"
        )

    before = replay_tree_identity(source)
    files = _regular_replay_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            destination,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for path in files:
                relative = path.relative_to(source).as_posix()
                info = zipfile.ZipInfo(relative, date_time=_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                with path.open("rb") as source_stream, archive.open(
                    info,
                    mode="w",
                    force_zip64=True,
                ) as archive_stream:
                    shutil.copyfileobj(
                        source_stream,
                        archive_stream,
                        length=_COPY_BUFFER_BYTES,
                    )
        after = replay_tree_identity(source)
        if before != after:
            destination.unlink(missing_ok=True)
            raise ReplayArchiveError(
                f"{REPLAY_DIRECTORY_NAME} changed while it was being archived"
            )
        archive_bytes = int(destination.stat().st_size)
        archive_sha256 = _sha256_file(destination)
    except ReplayArchiveError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        destination.unlink(missing_ok=True)
        raise ReplayArchiveError(
            f"Cannot create {REPLAY_ARCHIVE_NAME}: {exc}"
        ) from exc

    return {
        "format": REPLAY_ARCHIVE_FORMAT,
        "version": REPLAY_ARCHIVE_VERSION,
        "archive_path": REPLAY_ARCHIVE_NAME,
        "source_path": REPLAY_DIRECTORY_NAME,
        "compression": "stored",
        "entry_root": "/",
        "source_tree": before,
        "archive": {
            "sha256": archive_sha256,
            "bytes": archive_bytes,
        },
    }


def _loose_episode_files(
    demo_dir: Path,
    *,
    excluded_names: frozenset[str],
    excluded_name_fragments: tuple[str, ...],
) -> list[UploadEntry]:
    entries: list[UploadEntry] = []
    for path in sorted(demo_dir.rglob("*")):
        if path.is_symlink():
            relative = path.relative_to(demo_dir).as_posix()
            raise ReplayArchiveError(
                f"Episode contains a symlink, which is not allowed: {relative}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(demo_dir)
        if relative.parts[0] == REPLAY_DIRECTORY_NAME:
            continue
        if relative.as_posix() == REPLAY_ARCHIVE_NAME:
            continue
        if path.name in excluded_names:
            continue
        if any(fragment in path.name for fragment in excluded_name_fragments):
            continue
        entries.append(UploadEntry(path, relative.as_posix()))
    return entries


@contextmanager
def prepare_episode_upload(
    demo_dir: str | Path,
    *,
    excluded_names: frozenset[str] | set[str] = frozenset(
        {"cloud_upload_meta.json"}
    ),
    excluded_name_fragments: tuple[str, ...] = (),
) -> Iterator[EpisodeUploadPlan]:
    """Yield immutable snapshot objects and retain the source identity.

    The caller never receives paths into the mutable source episode.  A
    complete private copy is taken through no-follow descriptors, publication
    blockers are rerun on that copy, and both source and snapshot are checked
    again before a normal context exit.
    """

    requested_episode = Path(demo_dir).expanduser()
    source_fd = _open_source_root(requested_episode)
    try:
        episode = requested_episode.resolve(strict=True)
        opened_root = os.fstat(source_fd)
        named_root = episode.lstat()
        if (
            not stat.S_ISDIR(named_root.st_mode)
            or (int(named_root.st_dev), int(named_root.st_ino))
            != (int(opened_root.st_dev), int(opened_root.st_ino))
        ):
            raise ReplayArchiveError(
                "Episode root changed while it was being opened"
            )
    except ReplayArchiveError:
        os.close(source_fd)
        raise
    except OSError as exc:
        os.close(source_fd)
        raise ReplayArchiveError(
            f"Episode root changed while it was being opened: {exc}"
        ) from exc
    guard: _EpisodeSnapshotGuard | None = None
    snapshot_episode: Path | None = None
    excluded = frozenset(excluded_names)
    try:
        with _private_upload_snapshot_directory(episode.name) as (
            snapshot_root,
            snapshot_episode,
        ):
            snapshot_metadata = snapshot_root.lstat()
            if (
                not stat.S_ISDIR(snapshot_metadata.st_mode)
                or snapshot_metadata.st_uid != os.getuid()
                or stat.S_IMODE(snapshot_metadata.st_mode) != 0o700
            ):
                raise ReplayArchiveError(
                    "Private upload snapshot root must be owner-only mode 0700"
                )
            source_resolved = episode
            snapshot_resolved = snapshot_root.resolve(strict=True)
            if (
                source_resolved == snapshot_resolved
                or source_resolved in snapshot_resolved.parents
                or snapshot_resolved in source_resolved.parents
            ):
                raise ReplayArchiveError(
                    "Private upload snapshot must be disjoint from its source"
                )

            source_identity = _capture_tree_from_directory_fd(
                source_fd,
                destination=snapshot_episode,
            )
            if _capture_tree_from_directory_fd(source_fd) != source_identity:
                raise ReplayArchiveError(
                    "Source episode changed while its private upload snapshot "
                    "was created"
                )
            _freeze_snapshot_tree(snapshot_episode)
            archive_path = snapshot_root / REPLAY_ARCHIVE_NAME
            replay_transport = create_replay_archive(
                snapshot_episode / REPLAY_DIRECTORY_NAME,
                archive_path,
            )
            archive_path.chmod(0o400)
            files = _loose_episode_files(
                snapshot_episode,
                excluded_names=excluded,
                excluded_name_fragments=excluded_name_fragments,
            )
            files.append(UploadEntry(archive_path, REPLAY_ARCHIVE_NAME))
            files.sort(key=lambda entry: entry.relative_path)
            upload_entries = tuple(files)
            guard = _EpisodeSnapshotGuard(
                source_fd=source_fd,
                source_identity=source_identity,
                source_root=episode,
                snapshot_episode_root=snapshot_episode,
                upload_entries=upload_entries,
            )
            source_fd = -1
            # Close the initial gate-to-snapshot window before exposing any
            # upload path.  The same check runs again at final commit/exit.
            guard.verify()
            plan = EpisodeUploadPlan(
                files=upload_entries,
                replay_transport=replay_transport,
                snapshot_root=snapshot_root,
                snapshot_episode_root=snapshot_episode,
                source_root=episode,
                _guard=guard,
            )
            try:
                yield plan
            except BaseException:
                raise
            else:
                guard.verify()
            finally:
                guard.close()
                _thaw_snapshot_tree(snapshot_episode)
    finally:
        if guard is not None:
            guard.close()
        if source_fd >= 0:
            os.close(source_fd)
        if snapshot_episode is not None:
            _thaw_snapshot_tree(snapshot_episode)


def resolve_replay_path(demo_dir: str | Path) -> Path:
    """Prefer canonical local Zarr, then fall back to its cloud archive."""

    episode = Path(demo_dir).expanduser()
    replay_dir = episode / REPLAY_DIRECTORY_NAME
    if replay_dir.is_dir():
        return replay_dir
    archive_path = episode / REPLAY_ARCHIVE_NAME
    if archive_path.is_file():
        return archive_path
    raise FileNotFoundError(
        f"Neither {REPLAY_DIRECTORY_NAME} nor {REPLAY_ARCHIVE_NAME} exists "
        f"under {episode}"
    )


@contextmanager
def open_replay_group(replay_path: str | Path) -> Iterator[Any]:
    """Open either a Zarr directory or ``replay.zarr.zip`` read-only."""

    import zarr

    path = Path(replay_path).expanduser()
    if path.is_dir():
        store = zarr.storage.DirectoryStore(str(path))
    elif path.is_file() and path.name.endswith(".zip"):
        store = zarr.storage.ZipStore(str(path), mode="r")
    else:
        raise FileNotFoundError(f"Replay store not found: {path}")
    try:
        yield zarr.open_group(store=store, mode="r")
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()
