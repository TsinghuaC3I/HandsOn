from __future__ import annotations

import os
from pathlib import Path
import shutil
from collections.abc import Callable


LEGACY_ARM_VIDEOS_DIR = "videos_arm"
CANONICAL_VIDEOS_DIR = "videos"


def publish_arm_video_directory(
    demo_dir: Path,
    rendered_dir: Path,
    *,
    before_replace: Callable[[], None] | None = None,
) -> Path:
    """Replace the canonical videos directory with a completed arm replay."""
    output = demo_dir / CANONICAL_VIDEOS_DIR
    old_output = demo_dir / f".{CANONICAL_VIDEOS_DIR}.pre-arm-replay-{os.getpid()}"
    legacy_output = demo_dir / LEGACY_ARM_VIDEOS_DIR
    if old_output.exists():
        shutil.rmtree(old_output)

    moved_existing = False
    installed_rendered = False
    try:
        if before_replace is not None:
            before_replace()
        if output.exists():
            output.replace(old_output)
            moved_existing = True
        rendered_dir.replace(output)
        installed_rendered = True
    except BaseException:
        # A callback or the first rename can fail before the original
        # canonical directory has moved.  Never treat that original as the
        # partially installed render and delete it.
        if installed_rendered and output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if moved_existing and old_output.exists() and not output.exists():
            old_output.replace(output)
        raise

    shutil.rmtree(old_output, ignore_errors=True)
    if legacy_output.exists() and legacy_output.resolve() != output.resolve():
        shutil.rmtree(legacy_output, ignore_errors=True)
    return output
