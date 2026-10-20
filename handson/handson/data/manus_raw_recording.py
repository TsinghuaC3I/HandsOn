"""Loss-detecting capture and Zarr storage for raw MANUS glove messages."""

from __future__ import annotations

import copy
import json
import socket
import threading
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


MANUS_RAW_FORMAT = "handson.manus_raw_events"
MANUS_RAW_VERSION = 1
MANUS_RAW_DEFAULT_PORT = 5018
MANUS_RAW_PACKET_MAGIC = b"DJMR1\0"
MANUS_RAW_MAX_PACKET_BYTES = 65_507
MANUS_RAW_SCHEMA_FILE = "manus_raw_schema.json"


def _pose_xyzw(pose: Any) -> list[float]:
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def _quaternion_xyzw(quaternion: Any) -> list[float]:
    return [
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    ]


def manus_message_to_event(
    message: Any,
    *,
    topic: str,
    source_sequence: int,
    bridge_unix_time_ns: int | None = None,
    bridge_monotonic_time_ns: int | None = None,
) -> dict[str, Any]:
    """Convert every field in ``manus_ros2_msgs/ManusGlove`` to primitives."""

    raw_nodes = [
        {
            "node_id": int(node.node_id),
            "parent_node_id": int(node.parent_node_id),
            "joint_type": str(node.joint_type),
            "chain_type": str(node.chain_type),
            "pose_xyzw": _pose_xyzw(node.pose),
        }
        for node in message.raw_nodes
    ]
    ergonomics = [
        {"type": str(item.type), "value": float(item.value)}
        for item in message.ergonomics
    ]
    raw_sensor = [_pose_xyzw(pose) for pose in message.raw_sensor]
    event = {
        "format": MANUS_RAW_FORMAT,
        "version": MANUS_RAW_VERSION,
        "source_topic": str(topic),
        "source_sequence": int(source_sequence),
        "bridge_unix_time_ns": int(
            time.time_ns()
            if bridge_unix_time_ns is None
            else bridge_unix_time_ns
        ),
        "bridge_monotonic_time_ns": int(
            time.monotonic_ns()
            if bridge_monotonic_time_ns is None
            else bridge_monotonic_time_ns
        ),
        "glove_id": int(message.glove_id),
        "side": str(message.side),
        "raw_node_count": int(message.raw_node_count),
        "raw_nodes": raw_nodes,
        "ergonomics_count": int(message.ergonomics_count),
        "ergonomics": ergonomics,
        "raw_sensor_orientation_xyzw": _quaternion_xyzw(
            message.raw_sensor_orientation
        ),
        "raw_sensor_count": int(message.raw_sensor_count),
        "raw_sensor_pose_xyzw": raw_sensor,
    }
    validate_manus_raw_event(event)
    return event


def validate_manus_raw_event(event: Mapping[str, Any]) -> None:
    if event.get("format") != MANUS_RAW_FORMAT:
        raise ValueError("MANUS raw event has an unknown format")
    if event.get("version") != MANUS_RAW_VERSION:
        raise ValueError("MANUS raw event has an unsupported version")
    if not str(event.get("source_topic") or ""):
        raise ValueError("MANUS raw event has no source topic")
    sequence = event.get("source_sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("MANUS raw event has an invalid source sequence")
    for count_name, rows_name in (
        ("raw_node_count", "raw_nodes"),
        ("ergonomics_count", "ergonomics"),
        ("raw_sensor_count", "raw_sensor_pose_xyzw"),
    ):
        declared = event.get(count_name)
        rows = event.get(rows_name)
        if type(declared) is not int or declared < 0:
            raise ValueError(f"MANUS raw event has invalid {count_name}")
        if not isinstance(rows, list) or declared != len(rows):
            raise ValueError(
                f"MANUS {count_name}={declared!r} does not match "
                f"{rows_name} length"
            )
    orientation = np.asarray(
        event.get("raw_sensor_orientation_xyzw"), dtype=np.float64
    )
    if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
        raise ValueError("MANUS raw sensor orientation must be finite xyzw[4]")
    for node in event["raw_nodes"]:
        pose = np.asarray(node.get("pose_xyzw"), dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("MANUS raw node pose must be finite xyz+xyzw[7]")
    sensor_poses = np.asarray(
        event["raw_sensor_pose_xyzw"], dtype=np.float64
    )
    if sensor_poses.size and (
        sensor_poses.ndim != 2
        or sensor_poses.shape[1] != 7
        or not np.all(np.isfinite(sensor_poses))
    ):
        raise ValueError("MANUS raw sensor poses must be finite xyz+xyzw[N,7]")


def encode_manus_raw_event(event: Mapping[str, Any]) -> bytes:
    validate_manus_raw_event(event)
    document = json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    packet = MANUS_RAW_PACKET_MAGIC + zlib.compress(document, level=1)
    if len(packet) > MANUS_RAW_MAX_PACKET_BYTES:
        raise ValueError(
            f"compressed MANUS raw packet is too large: {len(packet)} bytes"
        )
    return packet


def decode_manus_raw_event(packet: bytes) -> dict[str, Any]:
    if not packet.startswith(MANUS_RAW_PACKET_MAGIC):
        raise ValueError("MANUS raw packet magic is invalid")
    event = json.loads(zlib.decompress(packet[len(MANUS_RAW_PACKET_MAGIC) :]))
    if not isinstance(event, dict):
        raise ValueError("MANUS raw packet root must be an object")
    validate_manus_raw_event(event)
    return event


def summarize_manus_raw_events(
    events: Iterable[Mapping[str, Any]],
    *,
    receiver_errors: Iterable[str] = (),
) -> dict[str, Any]:
    """Describe stream integrity without deciding whether an episode may save.

    Raw MANUS capture is an audit sidecar.  A missing or incomplete sidecar is
    reported explicitly, but it must never turn a valid robot episode into a
    failed collection attempt.
    """

    rows = list(events)
    sequences: dict[str, list[int]] = defaultdict(list)
    sides: set[str] = set()
    for event in rows:
        validate_manus_raw_event(event)
        topic = str(event["source_topic"])
        sequences[topic].append(int(event["source_sequence"]))
        sides.add(str(event.get("side") or ""))
    gaps: list[dict[str, int | str]] = []
    for topic, topic_sequences in sorted(sequences.items()):
        for previous, current in zip(topic_sequences, topic_sequences[1:]):
            if current != previous + 1:
                gaps.append(
                    {
                        "source_topic": topic,
                        "previous": previous,
                        "current": current,
                    }
                )
    errors = [str(error) for error in receiver_errors]
    if not rows:
        status = "no_events"
    elif errors:
        status = "receiver_errors"
    elif gaps:
        status = "source_sequence_gaps"
    else:
        status = "complete"
    return {
        "format": MANUS_RAW_FORMAT,
        "version": MANUS_RAW_VERSION,
        "event_count": len(rows),
        "topics": sorted(sequences),
        "sides": sorted(side for side in sides if side),
        "sequence_gap_count": len(gaps),
        "sequence_gaps": gaps[:32],
        "receiver_error_count": len(errors),
        "receiver_errors": errors[:32],
        "complete": status == "complete",
        "status": status,
    }


def validate_manus_raw_events(
    events: Iterable[Mapping[str, Any]],
    *,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    rows = list(events)
    summary = summarize_manus_raw_events(rows)
    if require_nonempty and not rows:
        raise ValueError("no raw MANUS messages were captured for this episode")
    if summary["sequence_gap_count"]:
        raise ValueError(
            "raw MANUS source sequence is incomplete: "
            f"{summary['sequence_gaps'][:8]}"
        )
    return summary


class ManusRawUdpRecorder:
    """Receive bridge packets and atomically delimit them by episode."""

    def __init__(self, host: str = "127.0.0.1", port: int = MANUS_RAW_DEFAULT_PORT):
        self.host = str(host)
        self.port = int(port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(0.2)
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="manus-raw-udp-recorder",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet, _address = self._socket.recvfrom(
                    MANUS_RAW_MAX_PACKET_BYTES
                )
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self._errors.append(f"{type(exc).__name__}: {exc}")
                return
            try:
                event = decode_manus_raw_event(packet)
                event["recorder_unix_time_ns"] = time.time_ns()
                event["recorder_monotonic_time_ns"] = time.monotonic_ns()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._errors.append(f"{type(exc).__name__}: {exc}")
                continue
            with self._lock:
                self._events.append(event)

    def begin_episode(self) -> None:
        with self._lock:
            self._events.clear()
            self._errors.clear()

    def snapshot(self, *, require_complete: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            events = copy.deepcopy(self._events)
            errors = list(self._errors)
        if errors:
            raise ValueError(f"raw MANUS receiver errors: {errors[:8]}")
        validate_manus_raw_events(events, require_nonempty=require_complete)
        return events

    def snapshot_report(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return received events and diagnostics without raising."""

        with self._lock:
            events = copy.deepcopy(self._events)
            errors = list(self._errors)
        return events, summarize_manus_raw_events(
            events,
            receiver_errors=errors,
        )

    def close(self) -> None:
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=1.0)


def _string_array(values: list[str]) -> np.ndarray:
    width = max([1, *(len(value) for value in values)])
    return np.asarray(values, dtype=f"<U{width}")


def _offsets(lengths: Iterable[int]) -> np.ndarray:
    result = [0]
    for length in lengths:
        result.append(result[-1] + int(length))
    return np.asarray(result, dtype=np.int64)


def write_manus_raw_zarr(
    root_group: Any,
    events: list[dict[str, Any]],
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the variable-length MANUS event stream below ``manus_raw``."""

    resolved_summary = dict(
        summary if summary is not None else summarize_manus_raw_events(events)
    )
    group = root_group.require_group("manus_raw")
    for key, value in resolved_summary.items():
        group.attrs[key] = value
    group.attrs["pose_layout"] = "position_xyz,orientation_xyzw"
    group.attrs["transport"] = "ROS2 SensorDataQoS -> compressed UDP loopback"

    raw_nodes = [node for event in events for node in event["raw_nodes"]]
    ergonomics = [item for event in events for item in event["ergonomics"]]
    sensors = [
        pose
        for event in events
        for pose in event["raw_sensor_pose_xyzw"]
    ]
    arrays: dict[str, np.ndarray] = {
        "bridge_unix_time_ns": np.asarray(
            [event["bridge_unix_time_ns"] for event in events], dtype=np.int64
        ),
        "bridge_monotonic_time_ns": np.asarray(
            [event["bridge_monotonic_time_ns"] for event in events], dtype=np.int64
        ),
        "recorder_unix_time_ns": np.asarray(
            [event["recorder_unix_time_ns"] for event in events], dtype=np.int64
        ),
        "recorder_monotonic_time_ns": np.asarray(
            [event["recorder_monotonic_time_ns"] for event in events], dtype=np.int64
        ),
        "source_sequence": np.asarray(
            [event["source_sequence"] for event in events], dtype=np.int64
        ),
        "source_topic": _string_array(
            [str(event["source_topic"]) for event in events]
        ),
        "glove_id": np.asarray(
            [event["glove_id"] for event in events], dtype=np.int32
        ),
        "side": _string_array([str(event["side"]) for event in events]),
        "raw_node_count": np.asarray(
            [event["raw_node_count"] for event in events], dtype=np.int32
        ),
        "raw_node_offsets": _offsets(
            len(event["raw_nodes"]) for event in events
        ),
        "raw_node_id": np.asarray(
            [node["node_id"] for node in raw_nodes], dtype=np.int32
        ),
        "raw_node_parent_node_id": np.asarray(
            [node["parent_node_id"] for node in raw_nodes], dtype=np.int32
        ),
        "raw_node_joint_type": _string_array(
            [str(node["joint_type"]) for node in raw_nodes]
        ),
        "raw_node_chain_type": _string_array(
            [str(node["chain_type"]) for node in raw_nodes]
        ),
        "raw_node_pose_xyzw": np.asarray(
            [node["pose_xyzw"] for node in raw_nodes], dtype=np.float64
        ).reshape(-1, 7),
        "ergonomics_count": np.asarray(
            [event["ergonomics_count"] for event in events], dtype=np.int32
        ),
        "ergonomics_offsets": _offsets(
            len(event["ergonomics"]) for event in events
        ),
        "ergonomics_type": _string_array(
            [str(item["type"]) for item in ergonomics]
        ),
        "ergonomics_value": np.asarray(
            [item["value"] for item in ergonomics], dtype=np.float32
        ),
        "raw_sensor_orientation_xyzw": np.asarray(
            [event["raw_sensor_orientation_xyzw"] for event in events],
            dtype=np.float64,
        ).reshape(-1, 4),
        "raw_sensor_count": np.asarray(
            [event["raw_sensor_count"] for event in events], dtype=np.int32
        ),
        "raw_sensor_offsets": _offsets(
            len(event["raw_sensor_pose_xyzw"]) for event in events
        ),
        "raw_sensor_pose_xyzw": np.asarray(
            sensors, dtype=np.float64
        ).reshape(-1, 7),
    }
    for name, array in arrays.items():
        group.array(name, array, overwrite=True)
    return resolved_summary


def write_manus_raw_schema(demo_dir: Path, summary: Mapping[str, Any]) -> Path:
    schema = {
        "format": MANUS_RAW_FORMAT,
        "version": MANUS_RAW_VERSION,
        "source_ros_type": "manus_ros2_msgs/msg/ManusGlove",
        "source_topics": ["/manus_glove_0", "/manus_glove_1"],
        "zarr_group": "replay.zarr/manus_raw",
        "event_axis": "independent MANUS callback events; not resampled to control steps",
        "pose_layout": "position_xyz,orientation_xyzw",
        "timebases": {
            "bridge_unix_time_ns": "wall clock when the bridge received the ROS callback",
            "bridge_monotonic_time_ns": "bridge-process monotonic callback time",
            "recorder_unix_time_ns": "wall clock when the recorder decoded the sidecar packet",
            "recorder_monotonic_time_ns": "recorder-process monotonic receive time",
        },
        "event_fields": [
            "source_topic",
            "source_sequence",
            "glove_id",
            "side",
            "raw_node_count",
            "ergonomics_count",
            "raw_sensor_orientation_xyzw",
            "raw_sensor_count",
        ],
        "variable_length_fields": {
            "raw_nodes": [
                "raw_node_id",
                "raw_node_parent_node_id",
                "raw_node_joint_type",
                "raw_node_chain_type",
                "raw_node_pose_xyzw",
            ],
            "ergonomics": ["ergonomics_type", "ergonomics_value"],
            "raw_sensor": ["raw_sensor_pose_xyzw"],
        },
        "variable_length_layout": (
            "raw_node_offsets, ergonomics_offsets and raw_sensor_offsets are "
            "CSR-style E+1 offsets into their flattened arrays"
        ),
        "completeness": dict(summary),
    }
    path = Path(demo_dir) / MANUS_RAW_SCHEMA_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
