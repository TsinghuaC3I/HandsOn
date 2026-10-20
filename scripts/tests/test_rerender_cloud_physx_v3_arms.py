from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import pytest

from scripts.rerender_cloud_physx_v3_arms import (
    _put_with_retry,
    _recent_dates,
    _root_prefix,
    build_parser,
    safe_extract_replay_archive,
    update_cloud_receipt,
)


def test_recent_dates_use_shanghai_calendar_and_configured_root() -> None:
    now = dt.datetime(2026, 8, 6, 0, 30, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    assert _recent_dates(3, now=now) == [
        "2026-08-06",
        "2026-08-05",
        "2026-08-04",
    ]
    assert _root_prefix("handson/2026-08-06", "") == "handson/"
    assert build_parser().parse_args([]).include_unrequested is False


def test_safe_extract_replay_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "replay.zarr.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="unsafe replay archive member"):
        safe_extract_replay_archive(archive, tmp_path / "replay.zarr")


def test_cloud_receipt_tracks_replaced_video_hashes() -> None:
    receipt = {
        "status": "uploaded",
        "files": [
            {
                "path": "videos/main.mp4",
                "key": "old/main",
                "bytes": 4,
                "etag": "old",
                "sha256": "0" * 64,
            },
            {"path": "replay.zarr.zip", "key": "replay", "bytes": 10},
        ],
    }
    replacements = {
        "videos/main.mp4": {
            "key": "new/main",
            "bytes": 8,
            "etag": "new",
            "sha256": "1" * 64,
        }
    }
    updated = update_cloud_receipt(
        receipt,
        replacements,
        render_attempts=2,
    )
    main = next(
        row for row in updated["files"] if row["path"] == "videos/main.mp4"
    )
    assert main["etag"] == "new"
    assert main["sha256"] == "1" * 64
    assert updated["bytes"] == 18
    assert updated["arm_rerender_replacement"]["render_attempts"] == 2
    assert updated["cloud_render"]["status"] == "complete"


def test_cloud_renderer_creates_missing_video_with_if_none_match(
    tmp_path: Path,
) -> None:
    video = tmp_path / "main.mp4"
    video.write_bytes(b"rendered")
    calls: list[dict[str, object]] = []

    class _Uploader:
        def put_file(self, path, key, **kwargs):
            calls.append({"path": path, "key": key, **kwargs})
            return {"success": True, "status": 200, "etag": "new"}

    result = _put_with_retry(
        _Uploader(),
        video,
        "handson/date/episode/videos/main.mp4",
        if_none_match=True,
    )

    assert result["success"] is True
    assert calls == [
        {
            "path": video,
            "key": "handson/date/episode/videos/main.mp4",
            "if_match": "",
            "if_none_match": True,
        }
    ]
