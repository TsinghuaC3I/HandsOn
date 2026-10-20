from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rerender_wrist_cameras import (
    Episode,
    _cloud_remote_prefix,
    _filter_episodes,
    _frame_is_housing_obstruction,
    _join_remote_key,
    _scan_candidates,
    _staged_result_is_resumable,
    _upload_cloud_replacement,
    _upsert_file_entry,
)


def _episode(tmp_path: Path, **overrides) -> Episode:
    values = {
        "path": tmp_path / "b01_demo_1",
        "task": "t04_cover_pitcher",
        "collected_at": "2026-07-20T09:30:00+08:00",
        "collector_id": "pengjiashuo",
        "video_fps": 30,
        "expected_frames": 100,
        "apply_fisheye_distortion": False,
        "fisheye_intrinsics_yaml": "",
    }
    values.update(overrides)
    return Episode(**values)


def test_housing_obstruction_classifier_rejects_uniform_gray_frame() -> None:
    frame = np.full((480, 640, 3), [136, 136, 152], dtype=np.uint8)
    noise = np.random.default_rng(4).integers(-5, 6, size=frame.shape)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    assert _frame_is_housing_obstruction(frame)


def test_housing_obstruction_classifier_keeps_normal_scene() -> None:
    y, x = np.mgrid[:240, :320]
    frame = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)

    assert not _frame_is_housing_obstruction(frame)


def test_all_mode_selects_every_episode_without_decoding_video(
    tmp_path: Path, monkeypatch
) -> None:
    episodes = [_episode(tmp_path), _episode(tmp_path, path=tmp_path / "b01_demo_2")]
    args = SimpleNamespace(
        all=True,
        upload_cloud=False,
        scan_stride=1,
        max_episodes=0,
    )
    monkeypatch.setattr(
        "scripts.rerender_wrist_cameras.inspect_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decoded video")),
    )

    selected, inspections = _scan_candidates(episodes, args)

    assert selected == episodes
    assert inspections == {str(episode.path): [] for episode in episodes}


def test_completed_staged_result_can_resume_without_rerender(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    for rel_path in ("videos/wrist_left.mp4", "videos/wrist_right.mp4"):
        path = stage_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    item = {"stage_dir": str(stage_dir), "expected_frames": 123}
    result = {"success": True, "stage_dir": str(stage_dir), "frame_count": 123}

    assert _staged_result_is_resumable(item, result)
    result["frame_count"] = 122
    assert not _staged_result_is_resumable(item, result)


def test_episode_filters_use_shanghai_time_and_exact_identity(tmp_path: Path) -> None:
    episodes = [
        _episode(tmp_path),
        _episode(
            tmp_path,
            path=tmp_path / "a01_demo_2",
            task="t26_place_fruit_in_bucket",
            collector_id="someone_else",
            collected_at="2026-07-19T18:00:00+08:00",
        ),
    ]
    args = SimpleNamespace(
        task=["t04_cover_pitcher"],
        episode=["b01_demo_1"],
        collector="pengjiashuo",
        from_time="2026-07-20",
        to_time="2026-07-20",
    )

    assert _filter_episodes(episodes, args) == [episodes[0]]
    assert dt.datetime.fromisoformat(episodes[0].collected_at).utcoffset() == dt.timedelta(hours=8)


def test_cloud_prefix_reuses_old_key_instead_of_current_date(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    uploader = SimpleNamespace(prefix="handson/2026-07-20")
    old = {"remote_prefix": "handson/2026-07-18/original_episode"}

    prefix = _cloud_remote_prefix(episode, tmp_path, old, uploader)

    assert prefix == "handson/2026-07-18/original_episode"
    assert _join_remote_key(prefix, "videos/wrist_right.mp4") == (
        "handson/2026-07-18/original_episode/videos/wrist_right.mp4"
    )


def test_cloud_prefix_adds_never_uploaded_episode_under_configured_prefix(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    uploader = SimpleNamespace(prefix="handson/2026-07-20")

    assert _cloud_remote_prefix(episode, tmp_path, {}, uploader) == (
        "handson/2026-07-20/b01_demo_1"
    )


def test_cloud_file_entry_is_replaced_by_relative_path() -> None:
    entries = [
        {
            "path": "videos/wrist_right.mp4",
            "key": "old/key",
            "bytes": 1,
            "etag": "old",
        }
    ]
    uploaded = {
        "key": "same/prefix/videos/wrist_right.mp4",
        "bytes": 99,
        "etag": '"abc"',
        "sha256": "123",
    }

    _upsert_file_entry(entries, uploaded, "videos/wrist_right.mp4")

    assert entries == [
        {
            "path": "videos/wrist_right.mp4",
            "key": "same/prefix/videos/wrist_right.mp4",
            "bytes": 99,
            "etag": '"abc"',
            "sha256": "123",
        }
    ]


def test_cloud_repair_overwrites_old_keys_and_commits_metadata_last(
    tmp_path: Path, monkeypatch
) -> None:
    episode = _episode(tmp_path)
    episode.path.mkdir()
    videos = episode.path / "videos"
    videos.mkdir()
    rerender_videos = []
    for name, content in (("wrist_left.mp4", b"left-fixed"), ("wrist_right.mp4", b"right-fixed")):
        path = videos / name
        path.write_bytes(content)
        rerender_videos.append(
            {
                "path": f"videos/{name}",
                "new_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (episode.path / "wrist_rerender_meta.json").write_text(
        json.dumps(
            {
                "status": "replaced_locally",
                "method": "isaaclab_state_guided_exact",
                "videos": rerender_videos,
            }
        ),
        encoding="utf-8",
    )
    old_prefix = "handson/2026-07-18/original_episode"
    (episode.path / "cloud_upload_meta.json").write_text(
        json.dumps(
            {
                "status": "uploaded",
                "bucket": "bucket",
                "remote_prefix": old_prefix,
                "files": [
                    {
                        "path": "videos/wrist_left.mp4",
                        "key": f"{old_prefix}/videos/wrist_left.mp4",
                        "bytes": 1,
                        "etag": "old",
                    },
                    {
                        "path": "videos/wrist_right.mp4",
                        "key": f"{old_prefix}/videos/wrist_right.mp4",
                        "bytes": 1,
                        "etag": "old",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    class FakeUploader:
        bucket = "bucket"
        endpoint_host = "endpoint"
        host = "bucket.endpoint"
        region = "NINGXIA"
        prefix = "handson/2026-07-20"

        @classmethod
        def from_env(cls):
            return cls()

        def configured(self):
            return True

        def missing_config(self):
            return []

        def put_file(self, path, key):
            content = Path(path).read_bytes()
            etag = hashlib.md5(content).hexdigest()
            calls.append(("file", key))
            return {
                "success": True,
                "status": 200,
                "etag": f'"{etag}"',
                "bytes": len(content),
            }

        def put_json(self, payload, key):
            calls.append(("json", key, payload))
            return {"success": True, "status": 200, "etag": '"meta"', "bytes": 10}

    import collection_ui.server as server

    monkeypatch.setattr(server, "Ks3Uploader", FakeUploader)
    monkeypatch.setattr(server, "cloud_upload_confirmed", lambda meta: True)
    monkeypatch.setattr(server, "postprocess_upload_blocker", lambda _path: None)

    result = _upload_cloud_replacement(episode, tmp_path, "batch-1")

    assert calls[:3] == [
        ("file", f"{old_prefix}/videos/wrist_left.mp4"),
        ("file", f"{old_prefix}/videos/wrist_right.mp4"),
        ("file", f"{old_prefix}/wrist_rerender_meta.json"),
    ]
    assert calls[-1][0:2] == ("json", f"{old_prefix}/cloud_upload_meta.json")
    assert result["status"] == "uploaded"
    assert result["source"] == "wrist_rerender_direct_ks3"
    assert result["wrist_rerender"]["mode"] == "same_key_wrist_overwrite"


def test_legacy_cloud_repair_gate_blocks_before_uploader_or_put(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode = _episode(tmp_path)
    episode.path.mkdir()
    videos = episode.path / "videos"
    videos.mkdir()
    rows = []
    for name, content in (
        ("wrist_left.mp4", b"left-fixed"),
        ("wrist_right.mp4", b"right-fixed"),
    ):
        (videos / name).write_bytes(content)
        rows.append(
            {
                "path": f"videos/{name}",
                "new_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (episode.path / "wrist_rerender_meta.json").write_text(
        json.dumps({"status": "replaced_locally", "videos": rows}),
        encoding="utf-8",
    )

    import collection_ui.server as server

    uploader_attempts = []

    class ForbiddenUploader:
        @classmethod
        def from_env(cls):
            uploader_attempts.append(True)
            raise AssertionError("uploader must not be created")

    monkeypatch.setattr(server, "Ks3Uploader", ForbiddenUploader)
    monkeypatch.setattr(
        server,
        "postprocess_upload_blocker",
        lambda _path: {
            "code": "physical_replay_pending",
            "detail": "physical replay is still pending",
        },
    )

    with pytest.raises(RuntimeError, match="physical replay is still pending"):
        _upload_cloud_replacement(episode, tmp_path, "batch")

    assert uploader_attempts == []


def test_rerender_cloud_marker_is_trusted_by_collection_and_review_tools() -> None:
    from collection_ui import server

    review_tool_dir = Path(__file__).resolve().parents[2] / "collection_ui" / "review_tool"
    sys.path.insert(0, str(review_tool_dir))
    try:
        from collection_ui.review_tool import local_review_tool
    finally:
        sys.path.remove(str(review_tool_dir))

    meta = {
        "status": "uploaded",
        "uploaded_at": "2026-07-20T20:00:00+08:00",
        "source": "wrist_rerender_direct_ks3",
        "upload_protocol": "ks3_put_object_v2",
        "upload_verified_by": "ks3_put_object_2xx",
        "cloud_ack": "ok",
        "file_count": 3,
        "files": [
            {"path": "videos/wrist_left.mp4", "key": "prefix/left", "bytes": 10},
            {"path": "videos/wrist_right.mp4", "key": "prefix/right", "bytes": 10},
        ],
    }

    assert server.cloud_upload_confirmed(meta)
    assert local_review_tool.cloud_upload_confirmed(meta)
