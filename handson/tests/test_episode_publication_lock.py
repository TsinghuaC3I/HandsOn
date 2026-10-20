from __future__ import annotations

import multiprocessing
from pathlib import Path

from handson.data.episode_publication_lock import (
    EpisodePublicationLockBusy,
    episode_publication_lock,
    episode_publication_lock_path,
)


def _try_lock_in_child(demo_dir: str, queue) -> None:
    try:
        with episode_publication_lock(demo_dir, blocking=False):
            queue.put("acquired")
    except EpisodePublicationLockBusy:
        queue.put("busy")


def test_episode_publication_lock_is_cross_process_and_outside_episode(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a01_demo"
    demo_dir.mkdir()
    lock_path = episode_publication_lock_path(demo_dir)
    assert lock_path.parent.parent == demo_dir.parent
    assert demo_dir not in lock_path.parents

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    with episode_publication_lock(demo_dir, blocking=False):
        process = context.Process(
            target=_try_lock_in_child,
            args=(str(demo_dir), queue),
        )
        process.start()
        process.join(timeout=5)
        assert process.exitcode == 0
        assert queue.get(timeout=1) == "busy"

    process = context.Process(
        target=_try_lock_in_child,
        args=(str(demo_dir), queue),
    )
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0
    assert queue.get(timeout=1) == "acquired"
