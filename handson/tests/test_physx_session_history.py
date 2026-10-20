from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from handson.data import physx_session_history as history


def _stable_origin(physics_steps: int = 120) -> dict[str, object]:
    return {
        "physics_steps": physics_steps,
        "initial_settle_steps": 60,
        "post_settle_velocity_zeroing": True,
    }


def _compact_episode(
    *,
    start_time: float,
    steps: int = 2,
) -> dict[str, np.ndarray]:
    times = start_time + np.arange(steps, dtype=np.float64) * 0.02
    next_times = times + 0.02
    targets = np.arange(steps * 6, dtype=np.float64).reshape(steps, 2, 3)
    valid = np.ones_like(targets, dtype=np.uint8)
    return {
        "action": np.arange(steps * 3, dtype=np.float64).reshape(steps, 3),
        "terminal": np.asarray([False] * (steps - 1) + [True], dtype=np.uint8),
        "succeed": np.zeros((steps,), dtype=np.uint8),
        "physx_sim_time": times[:, None],
        "next_physx_sim_time": next_times[:, None],
        "physx_robot_joint_target": targets[:, 0],
        "physx_robot_joint_target_valid": valid[:, 0],
        "next_physx_robot_joint_target": targets[:, 1],
        "next_physx_robot_joint_target_valid": valid[:, 1],
        "physx_substep_count": np.full((steps, 1), 2, dtype=np.int64),
        "physx_substep_robot_joint_target": targets,
        "physx_substep_robot_joint_target_valid": valid,
        "next_physx_substep_count": np.full((steps, 1), 2, dtype=np.int64),
        "next_physx_substep_robot_joint_target": targets,
        "next_physx_substep_robot_joint_target_valid": valid,
        # Camera-looking fields must never enter the compact history segment.
        "images": np.zeros((steps, 8, 8, 3), dtype=np.uint8),
    }


def _journal(tmp_path: Path) -> history.PhysxSessionHistoryJournal:
    return history.PhysxSessionHistoryJournal.create(
        tmp_path / "session-history",
        session_id="session-001",
        task_name="a01_fruit_to_basket",
        collection_run_seed=1234,
        metadata={"collector_id": "collector-1"},
    )


def test_hard_scene_rebuild_policy_seals_generation_and_clock_boundaries(
    tmp_path: Path,
) -> None:
    journal = history.PhysxSessionHistoryJournal.create(
        tmp_path / "hard-reset-session-history",
        session_id="session-hard-reset",
        task_name="t26_place_fruit_in_bucket",
        collection_run_seed=900,
        metadata={"in_process_physx_scene_rebuild_per_reset": True},
    )
    journal.record_reset_attempt(
        seed=900,
        ordinal=0,
        outcome="rejected_unstable_origin",
        stable_kwargs={"timeout_seconds": 3.0},
        scene_rebuilt_before_reset=False,
        scene_generation=0,
        error="still moving",
        start_sim_time=0.0,
        end_sim_time=2.0,
    )
    journal.record_reset_attempt(
        seed=901,
        ordinal=1,
        outcome="rejected_unstable_origin",
        stable_kwargs={"timeout_seconds": 3.0},
        scene_rebuilt_before_reset=True,
        scene_generation=1,
        scene_rebuild_report={
            "scene_rebuilt": True,
            "scene_generation": 1,
            "process_restarted": False,
        },
        error="still moving",
        # A fresh PxScene starts its own simulation clock at zero.
        start_sim_time=0.0,
        end_sim_time=1.5,
    )

    finalization = journal.finalize([])
    loaded = history.load_session_history(journal.path)

    assert finalization["sim_time_continuity"] == "complete"
    assert [
        event["payload"]["scene_generation"]
        for event in loaded["events"]
    ] == [0, 1]
    assert loaded["events"][1]["payload"]["scene_rebuild_report"] == {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "process_restarted": False,
    }


def test_hard_scene_rebuild_policy_rejects_missing_generation_marker(
    tmp_path: Path,
) -> None:
    journal = history.PhysxSessionHistoryJournal.create(
        tmp_path / "invalid-hard-reset-session-history",
        session_id="session-invalid-hard-reset",
        task_name="t26_place_fruit_in_bucket",
        collection_run_seed=900,
        metadata={"in_process_physx_scene_rebuild_per_reset": True},
    )
    journal.record_reset_attempt(
        seed=900,
        ordinal=0,
        outcome="rejected_unstable_origin",
        stable_kwargs={},
        error="still moving",
    )
    journal.record_reset_attempt(
        seed=901,
        ordinal=1,
        outcome="rejected_unstable_origin",
        stable_kwargs={},
        error="still moving",
    )

    with pytest.raises(ValueError, match="scene-rebuild marker"):
        journal.finalize([])


def test_records_complete_ordered_chain_and_loads_compact_discard(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    rejected = journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="rejected_unstable",
        stable_kwargs={"timeout_seconds": 3.0},
        error="still moving",
        physics_steps=420,
        start_sim_time=0.0,
        end_sim_time=1.0,
    )
    accepted_reset = journal.record_reset_attempt(
        seed=1235,
        ordinal=1,
        outcome="accepted",
        stable_kwargs={"timeout_seconds": 3.0},
        stable_origin=_stable_origin(),
        start_sim_time=1.0,
        end_sim_time=2.0,
    )
    discarded = journal.record_trajectory(
        _compact_episode(start_time=2.0),
        reason="teleoperation_manual_reset",
        reset_ordinal=1,
    )
    journal.record_reset_attempt(
        seed=1236,
        ordinal=2,
        outcome="accepted",
        stable_kwargs={"timeout_seconds": 3.0},
        stable_origin=_stable_origin(180),
        start_sim_time=2.04,
        end_sim_time=3.0,
    )

    demo_dir = tmp_path / "a01_demo_1__stable_v2"
    (demo_dir / "replay.zarr").mkdir(parents=True)
    accepted = journal.record_trajectory(
        _compact_episode(start_time=3.0),
        reason="successful_terminal",
        reset_ordinal=2,
        accepted_demo_dir=demo_dir,
    )
    finalization = journal.finalize([demo_dir])

    assert rejected["ordinal"] == 0
    assert accepted_reset["previous_event_sha256"] == rejected["event_sha256"]
    assert discarded["ordinal"] == 2
    assert accepted["ordinal"] == 4
    assert finalization["event_count"] == 5
    assert finalization["certified_through_event_ordinal"] == 4
    assert finalization["uncertified_tail"] is None
    assert finalization["sim_time_continuity"] == "complete"
    assert finalization["accepted_demos"] == [
        {
            "event_ordinal": 4,
            "reset_ordinal": 2,
            "demo_id": demo_dir.name,
            "step_count": 2,
        }
    ]
    assert len(finalization["history_identity"]["sha256"]) == 64

    loaded = history.load_session_history(journal.path)
    assert [event["kind"] for event in loaded["events"]] == [
        "reset_attempt",
        "reset_attempt",
        "trajectory",
        "reset_attempt",
        "trajectory",
    ]
    compact = history.load_discarded_trajectory(journal.path, 2)
    assert "images" not in compact
    np.testing.assert_array_equal(
        compact["action"],
        _compact_episode(start_time=2.0)["action"],
    )
    assert set(history._RECORDED_DRIVE_FIELDS).issubset(compact)
    discarded_dir = journal.path / discarded["event_directory"]
    assert (discarded_dir / history.DISCARDED_SCHEMA_NAME).is_file()
    schema = json.loads(
        (discarded_dir / history.DISCARDED_SCHEMA_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert set(compact) == set(schema["fields"])

    with pytest.raises(RuntimeError, match="already finalized"):
        journal.record_reset_attempt(
            seed=1237,
            ordinal=3,
            outcome="rejected_unstable",
            stable_kwargs={},
        )


def test_failed_segment_write_does_not_consume_an_event_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
    )
    original_writer = history._write_compact_segment

    def fail_write(*_args, **_kwargs) -> None:
        raise RuntimeError("synthetic zarr failure")

    monkeypatch.setattr(history, "_write_compact_segment", fail_write)
    with pytest.raises(RuntimeError, match="synthetic zarr failure"):
        journal.record_trajectory(
            _compact_episode(start_time=1.0),
            reason="manual_reset",
            reset_ordinal=0,
        )

    partial = history.load_session_history(
        journal.path,
        require_finalized=False,
    )
    assert [event["ordinal"] for event in partial["events"]] == [0]
    assert [child.name for child in journal.events_path.iterdir()] == [
        "00000000"
    ]

    monkeypatch.setattr(history, "_write_compact_segment", original_writer)
    committed = journal.record_trajectory(
        _compact_episode(start_time=1.0),
        reason="manual_reset",
        reset_ordinal=0,
    )
    assert committed["ordinal"] == 1


def test_recorder_transition_sequence_keeps_required_physx_drive_fields(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        1234,
        0,
        "accepted",
        {},
        stable_origin=_stable_origin(),
    )
    arrays = _compact_episode(start_time=1.0)
    semantic_fields = {
        name
        for name in arrays
        if name.startswith("physx_")
    }
    transitions = []
    for index in range(2):
        current = (
            {name: arrays[name][index] for name in semantic_fields}
            if index == 0
            else {}
        )
        next_observation = {
            name: arrays[f"next_{name}"][index]
            for name in semantic_fields
        }
        transitions.append(
            {
                "observations": current,
                "next_observations": next_observation,
                "actions": arrays["action"][index],
                "dones": bool(arrays["terminal"][index]),
                "infos": {"succeed": bool(arrays["succeed"][index])},
            }
        )

    event = journal.record_trajectory(
        transitions,
        reason="unsuccessful_terminal_transition",
        reset_ordinal=0,
        required_fields=semantic_fields,
    )
    journal.finalize()

    loaded = history.load_discarded_trajectory(
        journal.path,
        event["ordinal"],
        require_finalized=False,
    )
    for field in history._RECORDED_DRIVE_FIELDS:
        np.testing.assert_array_equal(loaded[field], arrays[field])


def test_discarded_numerical_guard_journals_partial_terminal_substep(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        1234,
        0,
        "accepted",
        {},
        stable_origin=_stable_origin(),
    )
    shape = (2, 1, 2)
    zeros = np.zeros(shape, dtype=np.float64)
    active = zeros.copy()
    active[0] = 1.0
    valid = np.zeros(shape, dtype=np.uint8)
    valid[0] = 1
    field_names = {
        "physx_robot_joint_target",
        "physx_robot_joint_target_valid",
        "physx_substep_count",
        "physx_substep_robot_joint_position",
        "physx_substep_robot_joint_velocity",
        "physx_substep_robot_joint_target",
        "physx_substep_robot_joint_valid",
        "physx_substep_robot_joint_target_valid",
    }
    initial = {
        "physx_robot_joint_target": np.zeros((1, 2), dtype=np.float64),
        "physx_robot_joint_target_valid": np.zeros((1, 2), dtype=np.uint8),
        "physx_substep_count": np.asarray([0], dtype=np.int64),
        "physx_substep_robot_joint_position": zeros.copy(),
        "physx_substep_robot_joint_velocity": zeros.copy(),
        "physx_substep_robot_joint_target": zeros.copy(),
        "physx_substep_robot_joint_valid": np.zeros(shape, dtype=np.uint8),
        "physx_substep_robot_joint_target_valid": np.zeros(
            shape,
            dtype=np.uint8,
        ),
    }
    partial = {
        "physx_robot_joint_target": np.ones((1, 2), dtype=np.float64),
        "physx_robot_joint_target_valid": np.ones((1, 2), dtype=np.uint8),
        "physx_substep_count": np.asarray([1], dtype=np.int64),
        "physx_substep_robot_joint_position": active.copy(),
        "physx_substep_robot_joint_velocity": active.copy(),
        "physx_substep_robot_joint_target": active.copy(),
        "physx_substep_robot_joint_valid": valid.copy(),
        "physx_substep_robot_joint_target_valid": valid.copy(),
    }
    fields = {
        "physx_robot_joint_target": {
            "shape": [1, 2],
            "dtype": "float64",
        },
        "physx_robot_joint_target_valid": {
            "shape": [1, 2],
            "dtype": "uint8",
        },
        "physx_substep_count": {"shape": [1], "dtype": "int64"},
    }
    for name in field_names - {
        "physx_robot_joint_target",
        "physx_robot_joint_target_valid",
        "physx_substep_count",
    }:
        fields[name] = {
            "shape": list(shape),
            "dtype": "uint8" if name.endswith("valid") else "float64",
        }
    schema = {"version": 4, "fields": fields}
    transition = {
        "observations": initial,
        "next_observations": partial,
        "actions": np.zeros(3, dtype=np.float64),
        "dones": False,
        "infos": {
            "manual_reset": True,
            "numerical_guard_event": {
                "body": "task_body_alpha",
                "reason": "rigid_motion_spike",
            },
        },
    }

    with pytest.raises(ValueError, match="Incomplete PhysX v4"):
        journal.record_trajectory(
            [transition],
            reason="d04_numerical_guard_abort",
            reset_ordinal=0,
            environment_schema=schema,
            required_fields=field_names,
        )

    event = journal.record_trajectory(
        [transition],
        reason="d04_numerical_guard_abort",
        reset_ordinal=0,
        environment_schema=schema,
        required_fields=field_names,
        allow_partial_terminal_guard=True,
    )
    assert event["payload"]["partial_terminal_guard"] is True
    loaded = history.load_discarded_trajectory(
        journal.path,
        event["ordinal"],
        require_finalized=False,
    )
    assert loaded["next_physx_substep_count"][0, 0] == 1


def test_sequence_rules_and_unfinished_final_reset_becomes_tail(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    with pytest.raises(ValueError, match="next contiguous value 0"):
        journal.record_reset_attempt(
            seed=1235,
            ordinal=1,
            outcome="rejected_unstable",
            stable_kwargs={},
        )

    journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="rejected_unstable",
        stable_kwargs={},
    )
    with pytest.raises(ValueError, match="accepted reset"):
        journal.record_trajectory(
            {},
            reason="manual_reset",
            reset_ordinal=0,
        )

    journal.record_reset_attempt(
        seed=1235,
        ordinal=1,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
    )
    with pytest.raises(ValueError, match="trajectory before the next reset"):
        journal.record_reset_attempt(
            seed=1236,
            ordinal=2,
            outcome="accepted",
            stable_kwargs={},
            stable_origin=_stable_origin(),
        )
    journal.record_trajectory(
        {},
        reason="immediate_keyboard_reset",
        reset_ordinal=1,
    )
    dangling = journal.record_reset_attempt(
        seed=1236,
        ordinal=2,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
    )

    finalization = journal.finalize()

    assert finalization["certified_through_event_ordinal"] is None
    assert finalization["uncertified_tail"] == {
        "kind": "accepted_reset_without_trajectory",
        "reset_ordinal": 2,
        "event_ordinal": dangling["ordinal"],
    }


def test_finalized_identity_detects_event_tampering(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
    )
    journal.record_trajectory(
        {},
        reason="manual_reset",
        reset_ordinal=0,
    )
    journal.finalize()

    event_path = journal.events_path / "00000001" / history.EVENT_DOCUMENT_NAME
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["payload"]["reason"] = "tampered"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(ValueError, match="event hashes changed|identity no longer matches"):
        history.load_session_history(journal.path)


def test_explicit_accepted_demo_order_must_match_events(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
    )
    demo_dir = tmp_path / "a01_demo_1__stable_v2"
    (demo_dir / "replay.zarr").mkdir(parents=True)
    journal.record_accepted_demo(
        _compact_episode(start_time=1.0),
        demo_dir=demo_dir,
        reset_ordinal=0,
    )

    with pytest.raises(ValueError, match="does not match journal events"):
        journal.finalize([tmp_path / "different_demo"])


def test_accepted_session_fails_closed_without_complete_sim_time(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
    )
    demo_dir = tmp_path / "a01_demo_1__stable_v2"
    (demo_dir / "replay.zarr").mkdir(parents=True)
    journal.record_accepted_demo(
        _compact_episode(start_time=1.0),
        demo_dir=demo_dir,
        reset_ordinal=0,
    )

    with pytest.raises(ValueError, match="simulation-time boundary"):
        journal.finalize([demo_dir])


def test_events_after_last_accepted_demo_are_an_uncertified_tail(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="accepted",
        stable_kwargs={},
        stable_origin=_stable_origin(),
        start_sim_time=0.0,
        end_sim_time=1.0,
    )
    demo_dir = tmp_path / "a01_demo_1__stable_v2"
    (demo_dir / "replay.zarr").mkdir(parents=True)
    accepted = journal.record_accepted_demo(
        _compact_episode(start_time=1.0),
        demo_dir=demo_dir,
        reset_ordinal=0,
    )
    rejected = journal.record_reset_attempt(
        seed=1235,
        ordinal=1,
        outcome="rejected_unstable_origin",
        stable_kwargs={"timeout_seconds": 3.0},
        error="still moving",
        physics_steps=240,
        start_sim_time=1.04,
        end_sim_time=5.04,
    )

    finalization = journal.finalize([demo_dir])

    assert finalization["certified_through_event_ordinal"] == accepted[
        "ordinal"
    ]
    assert finalization["uncertified_tail"] == {
        "kind": "events_after_certified_prefix",
        "first_event_ordinal": rejected["ordinal"],
        "last_event_ordinal": rejected["ordinal"],
        "event_count": 1,
    }


def test_stale_owned_event_temporary_is_cleaned_after_writer_crash(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    stale = journal.events_path / (
        ".00000000.tmp-12345-0123456789abcdef0123456789abcdef"
    )
    stale.mkdir()
    (stale / "partial").write_text("not committed", encoding="utf-8")

    event = journal.record_reset_attempt(
        seed=1234,
        ordinal=0,
        outcome="rejected_unstable_origin",
        stable_kwargs={},
        physics_steps=1,
        start_sim_time=0.0,
        end_sim_time=0.01,
    )

    assert event["ordinal"] == 0
    assert not stale.exists()
