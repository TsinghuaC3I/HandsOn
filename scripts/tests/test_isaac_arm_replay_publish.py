from __future__ import annotations

from pathlib import Path

from scripts.isaac_arm_replay_publish import publish_arm_video_directory


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_arm_replay_replaces_canonical_videos_and_removes_legacy_dir(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a02_demo"
    _write_file(demo_dir / "videos" / "main.mp4", b"original")
    _write_file(demo_dir / "videos_arm" / "main.mp4", b"stale")
    rendered = demo_dir / ".videos_arm-new"
    _write_file(rendered / "main.mp4", b"arm")
    _write_file(rendered / "camera_info.json", b"{}")

    output = publish_arm_video_directory(demo_dir, rendered)

    assert output == demo_dir / "videos"
    assert (demo_dir / "videos" / "main.mp4").read_bytes() == b"arm"
    assert (demo_dir / "videos" / "camera_info.json").read_bytes() == b"{}"
    assert not (demo_dir / "videos_arm").exists()
    assert not rendered.exists()
    assert not list(demo_dir.glob(".videos.pre-arm-replay-*"))


def test_arm_replay_restores_original_videos_if_publish_fails(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a02_demo"
    _write_file(demo_dir / "videos" / "main.mp4", b"original")
    rendered = demo_dir / ".missing-videos-arm"

    try:
        publish_arm_video_directory(demo_dir, rendered)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected publish failure")

    assert (demo_dir / "videos" / "main.mp4").read_bytes() == b"original"
    assert not rendered.exists()


def test_arm_replay_marks_replacement_before_moving_existing_videos(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a02_demo"
    _write_file(demo_dir / "videos" / "main.mp4", b"original")
    rendered = demo_dir / ".videos_arm-new"
    _write_file(rendered / "main.mp4", b"arm")
    callback_observations: list[tuple[bool, bool]] = []

    publish_arm_video_directory(
        demo_dir,
        rendered,
        before_replace=lambda: callback_observations.append(
            (
                (demo_dir / "videos" / "main.mp4").exists(),
                (rendered / "main.mp4").exists(),
            )
        ),
    )

    assert callback_observations == [(True, True)]
    assert (demo_dir / "videos" / "main.mp4").read_bytes() == b"arm"


def test_arm_replay_preserves_original_videos_when_pre_replace_marker_fails(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a02_demo"
    _write_file(demo_dir / "videos" / "main.mp4", b"original")
    rendered = demo_dir / ".videos_arm-new"
    _write_file(rendered / "main.mp4", b"arm")

    def fail_before_replace() -> None:
        raise OSError("marker write failed")

    try:
        publish_arm_video_directory(
            demo_dir,
            rendered,
            before_replace=fail_before_replace,
        )
    except OSError as exc:
        assert str(exc) == "marker write failed"
    else:
        raise AssertionError("expected pre-replacement marker failure")

    assert (demo_dir / "videos" / "main.mp4").read_bytes() == b"original"
    assert (rendered / "main.mp4").read_bytes() == b"arm"
    assert not list(demo_dir.glob(".videos.pre-arm-replay-*"))
