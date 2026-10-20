from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

import scripts.c01_compute_arm_plan_qpos as arm_plan
from scripts.e01_compute_arm_plan_qpos import _E01_TABLE_DELTA_INDEX


def test_e01_table_delta_is_zero_or_explicit_state_101_only() -> None:
    state = np.zeros(102, dtype=np.float64)
    state[68] = 0.75
    state[_E01_TABLE_DELTA_INDEX] = 0.012

    assert arm_plan._table_delta_from_state(
        state,
        table_delta_index=None,
    ) == 0.0
    assert arm_plan._table_delta_from_state(
        state,
        table_delta_index=_E01_TABLE_DELTA_INDEX,
    ) == pytest.approx(0.012)
    assert arm_plan._table_delta_from_state(
        state[:14],
        table_delta_index=_E01_TABLE_DELTA_INDEX,
    ) == 0.0


class _FakeController:
    def __init__(self) -> None:
        self.panda_right_dof_ids = np.arange(7, dtype=np.int64)
        self.panda_left_dof_ids = np.arange(7, 14, dtype=np.int64)
        self.panda_right_home = np.zeros(7, dtype=np.float64)
        self.panda_left_home = np.zeros(7, dtype=np.float64)
        self.model = type(
            "FakeModel",
            (),
            {
                "jnt_range": np.tile(
                    np.asarray([[-2.0, 2.0]], dtype=np.float64),
                    (14, 1),
                )
            },
        )()


class _FakePlan1Env:
    def __init__(self, **_kwargs) -> None:
        self._c01_background_arm = _FakeController()

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


def _make_replay(path: Path, frame_count: int = 3) -> Path:
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    states = np.zeros((frame_count, 102), dtype=np.float64)
    states[:, 68] = 0.9
    states[:, _E01_TABLE_DELTA_INDEX] = 0.0
    data.create_dataset(
        "state",
        data=states,
        shape=states.shape,
        dtype=states.dtype,
    )
    next_states = states.copy()
    next_states[:, 0] = np.arange(1, frame_count + 1, dtype=np.float64)
    data.create_dataset(
        "next_state",
        data=next_states,
        shape=next_states.shape,
        dtype=next_states.dtype,
    )
    return path


def _patch_lightweight_solver(monkeypatch, *, audit_valid: bool) -> list[int | None]:
    import handson.sim.envs.panda_plan1_hammer_nail_env as env_module

    monkeypatch.setattr(env_module, "PandaPlan1HammerNailGymEnv", _FakePlan1Env)
    monkeypatch.setattr(arm_plan, "_configure_env", lambda: None)

    def solve(_env, states, *, reverse):
        del reverse
        frame_count = states.shape[0]
        return (
            np.zeros((frame_count, 14), dtype=np.float64),
            np.ones((frame_count, 1), dtype=np.float64),
            np.zeros((frame_count, 1), dtype=np.float64),
        )

    def smooth(raw_qpos, _errors, _ranges, _fallback, **_kwargs):
        return raw_qpos.copy(), np.zeros(raw_qpos.shape[0], dtype=bool)

    observed_table_indices: list[int | None] = []

    def audit(_controller, states, qpos, *, table_delta_index, **_kwargs):
        observed_table_indices.append(table_delta_index)
        frame_count = states.shape[0]
        valid = np.full(frame_count, audit_valid, dtype=bool)
        return {
            "position_error": np.zeros((frame_count, 2), dtype=np.float64),
            "orientation_error": np.zeros((frame_count, 2), dtype=np.float64),
            "fk_valid": valid.copy(),
            "arm_table_min_distance": np.full(frame_count, np.inf),
            "inter_arm_min_distance": np.full(frame_count, np.inf),
            "self_collision_min_distance": np.full(frame_count, np.inf),
            "collision_valid": valid.copy(),
            "valid": valid.copy(),
        }

    monkeypatch.setattr(arm_plan, "_solve_recorded_arm_track", solve)
    monkeypatch.setattr(arm_plan, "_global_smooth_arm_path", smooth)
    monkeypatch.setattr(arm_plan, "_audit_smoothed_arm_path", audit)
    monkeypatch.setattr(arm_plan, "_bidirectional_stitch_candidates", lambda *a, **k: [])
    return observed_table_indices


def _compute_e01(
    replay_path: Path,
    *,
    table_delta_index: int | None = None,
) -> None:
    arm_plan._compute_for_replay(
        replay_path,
        overwrite=False,
        hz=30,
        velocity_weight=0.05,
        acceleration_weight=0.2,
        max_joint_step=0.4,
        max_ik_error=0.08,
        max_fk_position_error=0.02,
        max_fk_orientation_error=0.20,
        collision_penetration_tolerance=0.002,
        collision_sweep_joint_step=0.02,
        output_prefix="e01_arm_plan",
        log_label="E01 arm plan",
        table_delta_index=table_delta_index,
        state_dataset="next_state",
    )


def test_e01_compute_writes_only_e01_prefixed_validated_arm_fields(
    monkeypatch,
    tmp_path,
) -> None:
    replay_path = _make_replay(tmp_path / "replay.zarr")
    observed_table_indices = _patch_lightweight_solver(
        monkeypatch,
        audit_valid=True,
    )

    _compute_e01(
        replay_path,
        table_delta_index=_E01_TABLE_DELTA_INDEX,
    )

    data = zarr.open(str(replay_path), mode="r")["data"]
    assert observed_table_indices == [_E01_TABLE_DELTA_INDEX]
    assert data["e01_arm_plan_qpos"].shape == (3, 14)
    for suffix in (
        "raw_qpos",
        "success",
        "error",
        "joint_step",
        "rejected_frame",
        "fk_position_error",
        "fk_orientation_error",
        "fk_valid_frame",
        "arm_table_min_distance",
        "inter_arm_min_distance",
        "self_collision_min_distance",
        "collision_valid_frame",
        "valid_frame",
    ):
        assert f"e01_arm_plan_{suffix}" in data
    assert not any(name.startswith("c01_arm_plan_") for name in data.keys())
    assert data.attrs["e01_arm_plan_table_delta_index"] == (
        _E01_TABLE_DELTA_INDEX
    )
    assert data.attrs["e01_arm_plan_state_dataset"] == "next_state"


def test_e01_postcheck_failure_is_saved_as_invalid_and_raised(
    monkeypatch,
    tmp_path,
) -> None:
    replay_path = _make_replay(tmp_path / "replay.zarr")
    _patch_lightweight_solver(monkeypatch, audit_valid=False)

    with pytest.raises(RuntimeError, match="rendering is blocked"):
        _compute_e01(replay_path)

    data = zarr.open(str(replay_path), mode="r")["data"]
    np.testing.assert_array_equal(data["e01_arm_plan_valid_frame"][:], 0)
    # Raw IK status remains independently visible; the invalid postcheck is
    # never rewritten as a successful final plan.
    np.testing.assert_array_equal(data["e01_arm_plan_success"][:], 1.0)
