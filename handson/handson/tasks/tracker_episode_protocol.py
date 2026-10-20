"""Local control protocol for episode and immediate tracker rebasing."""

from __future__ import annotations

import os
import socket
import time
import uuid

import numpy as np


# Keep this separate from the legacy GeoRT 5013/5015 keypoint inputs.
DEFAULT_TRACKER_EPISODE_CONTROL_PORT = 5512
TRACKER_EPISODE_BASELINE_REQUEST = b"HANDSON_TRACKER_EPISODE_BASELINE_V1"
TRACKER_EPISODE_BASELINE_ACK = b"HANDSON_TRACKER_EPISODE_BASELINE_V1 OK\n"
TRACKER_EPISODE_BASELINE_REJECT = b"HANDSON_TRACKER_EPISODE_BASELINE_V1 REJECT\n"
TRACKER_EPISODE_GUARD_RELEASE_REQUEST = (
    b"HANDSON_TRACKER_EPISODE_GUARD_RELEASE_V1"
)
TRACKER_EPISODE_GUARD_RELEASE_ACK = (
    b"HANDSON_TRACKER_EPISODE_GUARD_RELEASE_V1 OK\n"
)
TRACKER_AUTO_BASELINE_REQUEST = b"HANDSON_TRACKER_AUTO_BASELINE_V1 "
TRACKER_AUTO_BASELINE_ACK = b"HANDSON_TRACKER_AUTO_BASELINE_V1 OK\n"
TRACKER_AUTO_BASELINE_CANCEL = b"HANDSON_TRACKER_AUTO_BASELINE_V1 CANCEL "


def tracker_pose_valid(pose: np.ndarray) -> bool:
    return bool(
        pose.shape == (4, 4)
        and np.all(np.isfinite(pose))
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
        and np.linalg.det(pose[:3, :3]) > 0.0
    )


class TrackerAutoBaselineClient:
    """Poll the bridge without blocking physics; one token per freeze."""

    def __init__(self, *, host: str = "127.0.0.1", port: int | None = None):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.connect((host, tracker_episode_control_port() if port is None else port))
        self._socket.setblocking(False)
        self._token = uuid.uuid4().bytes
        self._sent_at = None
        self._polled_at = time.monotonic()

    def poll(self) -> bytes | None:
        now = time.monotonic()
        # A long simulation stall can leave an old acknowledgement queued.
        # Request a current sample rather than assigning it a fresh timestamp.
        stale = now - self._polled_at > 0.25
        self._polled_at = now
        prefix = TRACKER_AUTO_BASELINE_ACK + self._token
        for _ in range(16):
            try:
                response = self._socket.recv(4096)
            except (BlockingIOError, OSError):
                break
            if not stale and response.startswith(prefix):
                packet = response[len(prefix):]
                if len(packet) in {12 * 8, 24 * 8}:
                    return packet
        if stale or self._sent_at is None or now - self._sent_at >= 0.5:
            try:
                self._socket.send(TRACKER_AUTO_BASELINE_REQUEST + self._token)
                self._sent_at = now
            except OSError:
                pass
        return None

    def close(self):
        try:
            self._socket.send(TRACKER_AUTO_BASELINE_CANCEL + self._token)
        except OSError:
            pass
        self._socket.close()


def tracker_episode_control_port() -> int:
    """Return the machine-local bridge control port with a safe fallback."""

    raw = os.environ.get("HANDSON_TRACKER_EPISODE_CONTROL_PORT", "").strip()
    if not raw:
        return DEFAULT_TRACKER_EPISODE_CONTROL_PORT
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_TRACKER_EPISODE_CONTROL_PORT
    return port if 1 <= port <= 65535 else DEFAULT_TRACKER_EPISODE_CONTROL_PORT


def request_tracker_episode_baseline(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_s: float = 1.5,
) -> bytes | None:
    """Ask Mode A to adopt its next complete tracker pair as a new baseline.

    The acknowledgement carries the exact pose packet that was also published
    on port 5012.  The simulation can therefore establish its episode-relative
    baseline without racing an older queued UDP datagram.
    """

    target_port = tracker_episode_control_port() if port is None else int(port)
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        control.settimeout(max(0.05, float(timeout_s)))
        control.sendto(TRACKER_EPISODE_BASELINE_REQUEST, (host, target_port))
        response, _address = control.recvfrom(4096)
    except (OSError, socket.timeout):
        return None
    finally:
        control.close()

    if not response.startswith(TRACKER_EPISODE_BASELINE_ACK):
        return None
    packet = response[len(TRACKER_EPISODE_BASELINE_ACK) :]
    if len(packet) not in {12 * 8, 24 * 8}:
        return None
    return packet


def release_tracker_episode_guard(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_s: float = 0.5,
) -> bool:
    """Tell Mode A that no collection episode is currently consuming 5012.

    The bridge keeps publishing tracker packets while idle, but accepts large
    repositioning and reconnects without entering the strict in-episode hold.
    A later baseline request arms the guard again at the exact packet used by
    the next collection episode.
    """

    target_port = tracker_episode_control_port() if port is None else int(port)
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        control.settimeout(max(0.05, float(timeout_s)))
        control.sendto(
            TRACKER_EPISODE_GUARD_RELEASE_REQUEST,
            (host, target_port),
        )
        response, _address = control.recvfrom(512)
    except (OSError, socket.timeout):
        return False
    finally:
        control.close()
    return response == TRACKER_EPISODE_GUARD_RELEASE_ACK
