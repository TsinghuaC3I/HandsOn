#!/usr/bin/env python3
"""Run the production camera diagnostic plus a live YCB10 replay-sensor probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import validate_isaac_deferred_camera_roundtrip as roundtrip


def collect_and_probe(env, config, args, report):
    roundtrip_collect(env, config, args, report)
    # This probe runs after the complete diagnostic episode is saved.  It
    # restores that same semantic boundary, without writing the marker pose,
    # and verifies that a missing live sensor defeats recorded valid evidence.
    raw = env.unwrapped
    state = raw._compute_observation()["state"]
    parts = {
        key: np.asarray(state[key]).copy()
        for key in (
            "ycb10_hidden_state",
            "finger_contact_forces",
            "ycb10_contact_validity",
        )
    }
    assert np.all(parts["ycb10_contact_validity"] == 1)
    assert not raw._ycb10_state.success
    # Emulate a fresh semantic cache at this physical boundary, so this check
    # cannot pass if the native hook merely leaves the old cache unchanged.
    # Neither the source nor restored state is a successful task state.
    raw._ycb10_state.reset()
    raw._ycb10_last_finger_forces.fill(0.0)
    raw._ycb10_last_contact_validity.fill(0.0)
    assert not np.array_equal(
        raw._ycb10_state.hidden_vector(), parts["ycb10_hidden_state"]
    )
    removed = raw._ycb10_finger_contact_views.pop(2)
    try:
        restored = raw.restore_recorded_initial_state(parts)["state"]
        for key, value in parts.items():
            np.testing.assert_array_equal(restored[key], value)
        _, _, terminated, _, info = raw.step_with_current_joint_targets()
        assert terminated and not info.get("succeed", False), info
        assert raw._ycb10_last_contact_validity[1] == 0
        assert raw._ycb10_state.first_terminal_reason == "invalid_contact_state"
        assert raw._ycb10_state.earned_axial == 0
    finally:
        raw._ycb10_finger_contact_views[2] = removed
    assert raw._ycb10_state.first_terminal_reason == "invalid_contact_state"
    env.reset(seed=args.seed)
    assert np.all(raw._ycb10_last_contact_validity == 1)
    assert not raw._ycb10_state.first_terminal_reason
    report["native_semantic_restore_live_probe"] = {
        "pass": True,
        "scope": "after_saved_non_successful_diagnostic",
        "restored_hidden_values": 200,
        "semantic_cache_cleared_before_restore": True,
        "marker_state_writes": 0,
        "missing_live_sensor_overrides_recorded_validity": True,
        "terminal_reason": "invalid_contact_state",
        "reset_recovers": True,
    }


if __name__ == "__main__":
    roundtrip_collect = roundtrip.collect
    roundtrip.collect = collect_and_probe
    for path in (
        str(Path(__file__).resolve().relative_to(ROOT)),
        "handson/handson/sim/isaaclab/recording_origin.py",
        "handson/handson/data/recording_fields.py",
        "handson/handson/data/episode_store.py",
        "handson/handson/evaluation/native.py",
        "configs/multi_task/t33_reposition_marker_grasp.yaml",
    ):
        sys.argv.extend(["--source-path", path])
    roundtrip.main()
