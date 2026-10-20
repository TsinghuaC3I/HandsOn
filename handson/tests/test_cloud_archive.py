from __future__ import annotations

import hashlib
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from handson.runtime.replay_archive import (
    REPLAY_ARCHIVE_FORMAT,
    ReplayArchiveError,
    create_replay_archive,
    open_replay_group,
    prepare_episode_upload,
    resolve_replay_path,
)


def _write_replay(path: Path) -> None:
    root = zarr.group(store=zarr.DirectoryStore(str(path)))
    data = root.require_group("data")
    data.array(
        "action",
        np.arange(12, dtype=np.float64).reshape(3, 4),
        chunks=(2, 4),
    )
    data.array(
        "state",
        np.arange(9, dtype=np.float64).reshape(3, 3),
        chunks=(2, 3),
    )
    root.require_group("meta").array(
        "episode_ends",
        np.asarray([3], dtype=np.int64),
        chunks=(1,),
    )


def test_episode_upload_replaces_zarr_tree_with_readable_archive(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    replay = episode / "replay.zarr"
    _write_replay(replay)
    (episode / "videos").mkdir()
    (episode / "videos" / "main.mp4").write_bytes(b"video")
    (episode / "collection_meta.json").write_text("{}", encoding="utf-8")
    (episode / "cloud_upload_meta.json").write_text(
        '{"status":"old"}',
        encoding="utf-8",
    )

    with prepare_episode_upload(episode) as plan:
        paths = [entry.relative_path for entry in plan.files]
        assert paths == [
            "collection_meta.json",
            "replay.zarr.zip",
            "videos/main.mp4",
        ]
        assert plan.source_root == episode
        assert plan.snapshot_episode_root.name == episode.name
        assert stat.S_IMODE(plan.snapshot_root.stat().st_mode) == 0o700
        assert (
            stat.S_IMODE(plan.snapshot_episode_root.stat().st_mode) == 0o500
        )
        assert all(
            entry.local_path.is_relative_to(plan.snapshot_root)
            for entry in plan.files
        )
        assert all(
            not entry.local_path.is_relative_to(episode)
            for entry in plan.files
        )
        assert all(
            stat.S_IMODE(entry.local_path.stat().st_mode) == 0o400
            for entry in plan.files
        )
        archive_entry = next(
            entry
            for entry in plan.files
            if entry.relative_path == "replay.zarr.zip"
        )
        archive_path = archive_entry.local_path
        assert archive_path.is_file()
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert ".zgroup" in names
            assert "data/action/.zarray" in names
            assert all(
                not name.startswith("replay.zarr/")
                for name in names
            )
            assert all(
                info.compress_type == zipfile.ZIP_STORED
                for info in archive.infolist()
            )
        with open_replay_group(archive_path) as root:
            np.testing.assert_array_equal(
                root["data"]["action"][:],
                np.arange(12, dtype=np.float64).reshape(3, 4),
            )

        transport = plan.replay_transport
        assert transport["format"] == REPLAY_ARCHIVE_FORMAT
        assert transport["compression"] == "stored"
        assert transport["source_tree"]["file_count"] > 1
        assert transport["archive"]["bytes"] == archive_path.stat().st_size
        assert transport["archive"]["sha256"] == hashlib.sha256(
            archive_path.read_bytes()
        ).hexdigest()

    assert not archive_path.exists()
    assert replay.is_dir()
    assert not (episode / "replay.zarr.zip").exists()


def test_archive_is_deterministic_and_resolver_prefers_directory(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    replay = episode / "replay.zarr"
    _write_replay(replay)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_meta = create_replay_archive(replay, first)
    second_meta = create_replay_archive(replay, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_meta == second_meta
    assert resolve_replay_path(episode) == replay
    replay.rename(tmp_path / "saved_replay.zarr")
    first.rename(episode / "replay.zarr.zip")
    assert resolve_replay_path(episode) == episode / "replay.zarr.zip"


def test_upload_rejects_symlink_in_episode(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    _write_replay(episode / "replay.zarr")
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":"must not be followed"}', encoding="utf-8")
    (episode / "linked.json").symlink_to(outside)

    with pytest.raises(ReplayArchiveError, match="symlink"):
        with prepare_episode_upload(episode):
            pass


def test_upload_snapshot_rejects_unmarked_source_change(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    _write_replay(episode / "replay.zarr")
    metadata = episode / "collection_meta.json"
    metadata.write_text('{"approved":true}\n', encoding="utf-8")

    with pytest.raises(ReplayArchiveError, match="Source episode changed"):
        with prepare_episode_upload(episode) as plan:
            metadata.write_text('{"approved":false}\n', encoding="utf-8")
            plan.verify_source_unchanged()


def test_upload_snapshot_rejects_source_root_inode_replacement(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    _write_replay(episode / "replay.zarr")
    (episode / "collection_meta.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ReplayArchiveError, match="root inode changed"):
        with prepare_episode_upload(episode) as plan:
            episode.rename(tmp_path / "original-episode")
            episode.mkdir()
            plan.verify_source_unchanged()
