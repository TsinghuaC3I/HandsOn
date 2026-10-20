"""Shared visual setup used by Isaac replay and rerender entry points."""

from __future__ import annotations

from typing import Any


def reapply_reference_robot_appearance_for_replay(environment: Any) -> int:
    """Refresh episode appearance without replacing live robot references."""

    raw_environment = getattr(environment, "unwrapped", environment)
    reapply = getattr(
        raw_environment,
        "reapply_reference_robot_appearance",
        None,
    )
    if not callable(reapply):
        return 0
    configured = int(reapply())
    if configured < 0:
        raise RuntimeError("Replay robot appearance count cannot be negative")
    if configured:
        print(
            "[HandsOn/IsaacLab] Replay robot appearance refreshed: "
            f"{configured} verified visual links",
            flush=True,
        )
    return configured
