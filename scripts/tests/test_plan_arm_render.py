from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from handson.sim.envs.hand_specs import HandController
from handson.sim.envs.plan_mode import build_plan_mode_model
from scripts.plan_render_arm_replay import (
    _require_valid_plan_frames,
    _restore_recorded_hand,
    _show_completed_arm,
)


def _plan_data(frame_count: int = 3) -> dict[str, np.ndarray]:
    return {
        "state": np.zeros((frame_count, 8), dtype=np.float64),
        "plan_arm_qpos": np.zeros((frame_count, 7), dtype=np.float64),
        "plan_arm_success": np.ones((frame_count, 1), dtype=np.float64),
    }


def test_plan_arm_render_requires_a_complete_valid_trajectory() -> None:
    data = _plan_data()

    _require_valid_plan_frames(data, 3)

    data["plan_arm_success"][1, 0] = 0.0
    with pytest.raises(RuntimeError, match="1 invalid frame"):
        _require_valid_plan_frames(data, 3)


def test_plan_arm_render_rejects_mismatched_frame_counts() -> None:
    data = _plan_data()
    data["plan_arm_qpos"] = np.zeros((2, 7), dtype=np.float64)

    with pytest.raises(RuntimeError, match="different frame counts"):
        _require_valid_plan_frames(data, 3)


def test_plan_arm_render_validates_optional_native_hand_frames() -> None:
    data = _plan_data()
    data["plan_hand_qpos"] = np.zeros((3, 20), dtype=np.float64)

    _require_valid_plan_frames(data, 3)

    data["plan_hand_qpos"] = np.zeros((3, 19), dtype=np.float64)
    with pytest.raises(RuntimeError, match="plan_hand_qpos shape"):
        _require_valid_plan_frames(data, 3)


def test_show_completed_arm_only_changes_plan_arm_alpha() -> None:
    rgba = np.zeros((6, 4), dtype=np.float64)
    controller = SimpleNamespace(arm_geom_ids={1, 4})
    raw_env = SimpleNamespace(
        _plan_mode_controller=controller,
        _model=SimpleNamespace(geom_rgba=rgba),
    )

    actual = _show_completed_arm(raw_env)

    assert actual is controller
    np.testing.assert_allclose(rgba[[1, 4], 3], 1.0)
    np.testing.assert_allclose(rgba[[0, 2, 3, 5], 3], 0.0)


def test_restore_recorded_hand_sets_joint_state_and_actuator_target() -> None:
    qpos = np.zeros(12, dtype=np.float64)
    ctrl = np.zeros(12, dtype=np.float64)
    calls = []
    hand = SimpleNamespace(
        qpos_ids=np.asarray([2, 4, 7]),
        policy_to_native=lambda values: np.asarray(values) + 0.5,
        set_native_ctrl=lambda data, native: (
            calls.append(np.asarray(native).copy()),
            data.ctrl.__setitem__([2, 4, 7], native),
        ),
    )
    raw_env = SimpleNamespace(
        _hand=hand,
        _data=SimpleNamespace(qpos=qpos, ctrl=ctrl),
    )

    restored = _restore_recorded_hand(
        raw_env,
        np.asarray([0.1, 0.2, 0.3]),
    )

    np.testing.assert_allclose(restored, [0.6, 0.7, 0.8])
    np.testing.assert_allclose(qpos[[2, 4, 7]], restored)
    np.testing.assert_allclose(ctrl[[2, 4, 7]], restored)
    np.testing.assert_allclose(calls, [restored])


def test_restore_recorded_hand_prefers_recorded_native_joint_state() -> None:
    qpos = np.zeros(12, dtype=np.float64)
    ctrl = np.zeros(12, dtype=np.float64)
    calls = []

    def fail_policy_conversion(_values):
        raise AssertionError("policy conversion must not run for native qpos")

    hand = SimpleNamespace(
        qpos_ids=np.asarray([2, 4, 7]),
        policy_to_native=fail_policy_conversion,
        set_native_ctrl=lambda data, native: (
            calls.append(np.asarray(native).copy()),
            data.ctrl.__setitem__([1, 5, 9], native),
        ),
    )
    raw_env = SimpleNamespace(
        _hand=hand,
        _data=SimpleNamespace(qpos=qpos, ctrl=ctrl),
    )
    native = np.asarray([0.4, -0.2, 1.1], dtype=np.float64)

    restored = _restore_recorded_hand(
        raw_env,
        np.asarray([9.0, 9.0, 9.0]),
        native_qpos=native,
    )

    np.testing.assert_allclose(restored, native)
    np.testing.assert_allclose(qpos[[2, 4, 7]], native)
    np.testing.assert_allclose(ctrl[[1, 5, 9]], native)
    np.testing.assert_allclose(calls, [native])


@pytest.mark.parametrize(
    "native_qpos",
    (
        np.asarray([0.1, 0.2], dtype=np.float64),
        np.asarray([0.1, np.nan, 0.3], dtype=np.float64),
    ),
)
def test_restore_recorded_hand_rejects_invalid_native_joint_state(
    native_qpos: np.ndarray,
) -> None:
    hand = SimpleNamespace(
        qpos_ids=np.asarray([2, 4, 7]),
        policy_to_native=lambda values: np.asarray(values),
        set_native_ctrl=lambda _data, _native: pytest.fail(
            "invalid native qpos must not reach the actuator target"
        ),
    )
    raw_env = SimpleNamespace(
        _hand=hand,
        _data=SimpleNamespace(
            qpos=np.zeros(12, dtype=np.float64),
            ctrl=np.zeros(12, dtype=np.float64),
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid recorded hand qpos"):
        _restore_recorded_hand(
            raw_env,
            np.zeros(3, dtype=np.float64),
            native_qpos=native_qpos,
        )


def test_restore_recorded_hand_changes_real_plan_model_qpos_across_frames() -> None:
    xml_path = (
        Path(__file__).resolve().parents[2]
        / "handson"
        / "handson"
        / "sim"
        / "envs"
        / "xmls"
        / "arena_arm_hand_ycb_tasks_rokae_wuji1.xml"
    )
    model = build_plan_mode_model(xml_path)
    data = mujoco.MjData(model)
    hand = HandController(
        model,
        hand_type="wuji1",
        sides=("right",),
        dual_arm=False,
    )
    raw_env = SimpleNamespace(_hand=hand, _data=data)
    frames = (
        np.linspace(0.05, 0.35, 16, dtype=np.float64),
        np.linspace(0.45, 0.75, 16, dtype=np.float64),
    )

    restored_frames = []
    for policy_qpos in frames:
        target = _restore_recorded_hand(raw_env, policy_qpos)
        mujoco.mj_forward(model, data)
        actual = data.qpos[hand.qpos_ids].copy()
        np.testing.assert_allclose(actual, target, atol=0.0, rtol=0.0)
        restored_frames.append(actual)

    assert np.max(np.abs(restored_frames[1] - restored_frames[0])) > 0.1
