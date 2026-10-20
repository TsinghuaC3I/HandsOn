"""HandsOn data utilities with optional heavy dependencies loaded lazily.

The ROS2 MANUS bridge shares the lightweight raw-message codec in this
package, but its device environment intentionally does not install Zarr or
OpenCV.  Import those storage/video helpers only when they are requested.
"""

from typing import Any


__all__ = ["ZarrEpisodeStore", "Mp4VideoWriter"]


def __getattr__(name: str) -> Any:
    if name == "ZarrEpisodeStore":
        from .episode_store import ZarrEpisodeStore

        return ZarrEpisodeStore
    if name == "Mp4VideoWriter":
        from .video_writer import Mp4VideoWriter

        return Mp4VideoWriter
    raise AttributeError(name)
