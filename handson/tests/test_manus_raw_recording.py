from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from handson.data.manus_raw_recording import (
    MANUS_RAW_FORMAT,
    MANUS_RAW_VERSION,
    decode_manus_raw_event,
    encode_manus_raw_event,
    manus_message_to_event,
    summarize_manus_raw_events,
    validate_manus_raw_events,
    write_manus_raw_schema,
    write_manus_raw_zarr,
)


def _event(sequence: int, topic: str = "/manus_glove_0") -> dict:
    return {
        "format": MANUS_RAW_FORMAT,
        "version": MANUS_RAW_VERSION,
        "source_topic": topic,
        "source_sequence": sequence,
        "bridge_unix_time_ns": 1000 + sequence,
        "bridge_monotonic_time_ns": 2000 + sequence,
        "recorder_unix_time_ns": 3000 + sequence,
        "recorder_monotonic_time_ns": 4000 + sequence,
        "glove_id": 17,
        "side": "left",
        "raw_node_count": 1,
        "raw_nodes": [
            {
                "node_id": 2,
                "parent_node_id": 1,
                "joint_type": "Distal",
                "chain_type": "Index",
                "pose_xyzw": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            }
        ],
        "ergonomics_count": 1,
        "ergonomics": [{"type": "IndexSpread", "value": 0.25}],
        "raw_sensor_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "raw_sensor_count": 1,
        "raw_sensor_pose_xyzw": [
            [4.0, 5.0, 6.0, 0.1, 0.2, 0.3, 0.9]
        ],
    }


def _pose(values: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=values[0], y=values[1], z=values[2]),
        orientation=SimpleNamespace(
            x=values[3], y=values[4], z=values[5], w=values[6]
        ),
    )


def test_manus_ros_message_conversion_keeps_nodes_ergonomics_and_sensors() -> None:
    message = SimpleNamespace(
        glove_id=23,
        side="right",
        raw_node_count=1,
        raw_nodes=[
            SimpleNamespace(
                node_id=8,
                parent_node_id=7,
                joint_type="Metacarpal",
                chain_type="Thumb",
                pose=_pose((1, 2, 3, 0.1, 0.2, 0.3, 0.9)),
            )
        ],
        ergonomics_count=1,
        ergonomics=[SimpleNamespace(type="ThumbMCP", value=12.5)],
        raw_sensor_orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
        raw_sensor_count=1,
        raw_sensor=[_pose((4, 5, 6, 0.4, 0.5, 0.6, 0.7))],
    )

    event = manus_message_to_event(
        message,
        topic="/manus_glove_1",
        source_sequence=41,
        bridge_unix_time_ns=10,
        bridge_monotonic_time_ns=20,
    )

    assert event["glove_id"] == 23
    assert event["raw_nodes"][0]["pose_xyzw"] == [
        1.0,
        2.0,
        3.0,
        0.1,
        0.2,
        0.3,
        0.9,
    ]
    assert event["ergonomics"] == [{"type": "ThumbMCP", "value": 12.5}]
    assert event["raw_sensor_pose_xyzw"][0][:3] == [4.0, 5.0, 6.0]


def test_raw_manus_packet_round_trip_preserves_every_field() -> None:
    event = _event(9)
    decoded = decode_manus_raw_event(encode_manus_raw_event(event))
    assert decoded == event


def test_raw_manus_sequence_gap_fails_closed() -> None:
    with pytest.raises(ValueError, match="sequence is incomplete"):
        validate_manus_raw_events([_event(10), _event(12)])


def test_raw_manus_sidecar_reports_gaps_without_rejecting_episode() -> None:
    summary = summarize_manus_raw_events(
        [_event(10), _event(12)],
        receiver_errors=["example receive warning"],
    )

    assert summary["complete"] is False
    assert summary["sequence_gap_count"] == 1
    assert summary["receiver_error_count"] == 1


def test_empty_raw_manus_sidecar_can_be_written_for_diagnostics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = zarr.group(
            store=zarr.DirectoryStore(str(Path(directory) / "replay.zarr"))
        )
        summary = write_manus_raw_zarr(root, [])

        assert summary["status"] == "no_events"
        assert summary["complete"] is False
        assert root["manus_raw/source_sequence"].shape == (0,)


def test_raw_manus_zarr_uses_offsets_and_preserves_source_precision() -> None:
    events = [_event(5), _event(6)]
    with tempfile.TemporaryDirectory() as directory:
        root = zarr.group(
            store=zarr.DirectoryStore(str(Path(directory) / "replay.zarr"))
        )
        summary = write_manus_raw_zarr(root, events)
        group = root["manus_raw"]

        assert summary["complete"] is True
        assert summary["event_count"] == 2
        np.testing.assert_array_equal(group["raw_node_offsets"][:], [0, 1, 2])
        np.testing.assert_array_equal(group["raw_sensor_offsets"][:], [0, 1, 2])
        np.testing.assert_allclose(
            group["raw_node_pose_xyzw"][:],
            [events[0]["raw_nodes"][0]["pose_xyzw"]] * 2,
        )
        assert group["raw_node_joint_type"][:].tolist() == [
            "Distal",
            "Distal",
        ]


def test_raw_manus_schema_is_self_describing_for_replay() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = write_manus_raw_schema(
            Path(directory),
            validate_manus_raw_events([_event(1)]),
        )
        document = json.loads(path.read_text(encoding="utf-8"))

    assert document["source_ros_type"] == "manus_ros2_msgs/msg/ManusGlove"
    assert "source_sequence" in document["event_fields"]
    assert "raw_node_pose_xyzw" in document["variable_length_fields"][
        "raw_nodes"
    ]
    assert document["completeness"]["complete"] is True
