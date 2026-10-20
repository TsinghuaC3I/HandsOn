import json

import numpy as np
import pytest

from scripts import record_demos_zarr as recorder


def _carton_failure_info() -> dict[str, object]:
    return {
        "succeed": False,
        "failed": np.asarray(True),
        "b320_failure_reason": "carton_side_force_exceeded",
        "b320_contact_available": {"left": True, "right": True},
        "b320_contact_readable": {"left": True, "right": True},
        "b320_contact_force_n": {
            "left": np.float64(10.25),
            "right": np.float64(1.75),
        },
        "b320_contact_peak_n": {
            "left": np.float64(12.48),
            "right": np.float64(2.25),
        },
        "b320_carton_zone_valid": {"left": True, "right": False},
        "b320_carton_transport_started": True,
        "b320_carton_transport_support_lost": False,
        "b320_metrics": {
            "tilt_deg": np.float64(1.53),
            "side_force_peak_n": np.float64(12.48),
            "unavailable_metric": float("inf"),
        },
    }


def test_b320_terminal_summary_preserves_physical_failure_evidence() -> None:
    summary = recorder._b320_terminal_summary(
        "t10_lift_carton",
        done=np.asarray(True),
        truncated=False,
        info=_carton_failure_info(),
        episode_steps=34,
        episode_sim_time_seconds=1.1,
    )

    assert summary is not None
    assert summary["outcome"] == "failure"
    assert summary["terminal_reason"] == "carton_side_force_exceeded"
    assert summary["failure_reason"] == "carton_side_force_exceeded"
    assert summary["truncated_reason"] is None
    assert summary["succeed"] is False
    assert summary["failed"] is True
    assert summary["terminated"] is True
    assert summary["truncated"] is False
    assert summary["episode_steps"] == 34
    assert summary["episode_sim_time_seconds"] == pytest.approx(1.1)
    assert summary["b320_contact_peak_n"] == {
        "left": pytest.approx(12.48),
        "right": pytest.approx(2.25),
    }
    assert summary["b320_carton_zone_valid"] == {
        "left": True,
        "right": False,
    }
    assert summary["b320_metrics"]["unavailable_metric"] is None


def test_b320_terminal_summary_keeps_timeout_distinct_from_failure() -> None:
    summary = recorder._b320_terminal_summary(
        "t10_lift_carton",
        done=False,
        truncated=np.asarray(True),
        info={
            "succeed": False,
            "time_limit_reached": True,
            "truncated_reason": "time_limit",
        },
        episode_steps=660,
        episode_sim_time_seconds=22.0,
    )

    assert summary is not None
    assert summary["outcome"] == "truncated"
    assert summary["terminal_reason"] == "time_limit"
    assert summary["failure_reason"] is None
    assert summary["truncated_reason"] == "time_limit"
    assert summary["time_limit_reached"] is True
    assert summary["succeed"] is False
    assert summary["failed"] is False
    assert summary["terminated"] is False
    assert summary["truncated"] is True


def test_bc005_canonical_runtime_key_preserves_failure_reason() -> None:
    summary = recorder._b320_terminal_summary(
        "loaded_tray_through_narrow_gate",
        done=True,
        truncated=False,
        info={
            "succeed": False,
            "failed": True,
            "failure_reason": "sustained_excessive_tilt",
            "tray_tilt_deg": np.float64(40.5),
        },
        episode_steps=117,
        episode_sim_time_seconds=3.9,
    )

    assert summary is not None
    assert summary["outcome"] == "failure"
    assert summary["terminal_reason"] == "sustained_excessive_tilt"
    assert summary["failure_reason"] == "sustained_excessive_tilt"
    assert summary["episode_steps"] == 117
    assert recorder._as013_unsuccessful_terminal_reason(
        "loaded_tray_through_narrow_gate",
        {
            "failure_reason": "sustained_excessive_tilt",
            "termination_reason": "sustained_excessive_tilt",
        },
    ) == (
        "unsuccessful_terminal_transition:"
        "bc005_first_failure=sustained_excessive_tilt"
    )


def test_bc011_terminal_summary_preserves_handover_and_table_truth() -> None:
    summary = recorder._b320_terminal_summary(
        "t20_pass_and_orient_pen",
        done=True,
        truncated=False,
        info={
            "succeed": True,
            "b320_handover_completed": True,
            "b320_pen_table_contact_available": True,
            "b320_pen_table_contact_readable": True,
            "b320_pen_table_contact_force_n": np.float64(0.0),
            "bc011_pen_variant": "cc_by_black_pen_high_quality/pen.usda",
            "bc011_pen_tip_axis_local": [-1.0, 0.0, 0.0],
        },
        episode_steps=71,
        episode_sim_time_seconds=71.0 / 30.0,
    )

    assert summary is not None
    assert summary["outcome"] == "success"
    assert summary["b320_handover_completed"] is True
    assert summary["b320_pen_table_contact_available"] is True
    assert summary["b320_pen_table_contact_readable"] is True
    assert summary["b320_pen_table_contact_force_n"] == pytest.approx(0.0)
    assert summary["bc011_pen_variant"] == "cc_by_black_pen_high_quality/pen.usda"
    assert summary["bc011_pen_tip_axis_local"] == [-1.0, 0.0, 0.0]


def test_b320_terminal_summary_prints_strict_machine_readable_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    returned = recorder._print_b320_terminal_summary(
        "t10_lift_carton",
        done=True,
        truncated=False,
        info=_carton_failure_info(),
        episode_steps=50,
        episode_sim_time_seconds=1.633,
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("[B320 episode] ")
    printed = json.loads(line.removeprefix("[B320 episode] "))
    assert printed == returned
    assert printed["terminal_reason"] == "carton_side_force_exceeded"
    assert printed["b320_contact_peak_n"]["left"] == pytest.approx(12.48)


def test_b320_terminal_status_detail_is_explicitly_not_success() -> None:
    summary = recorder._b320_terminal_summary(
        "t10_lift_carton",
        done=True,
        truncated=False,
        info=_carton_failure_info(),
        episode_steps=34,
        episode_sim_time_seconds=1.1,
    )
    assert summary is not None

    before = recorder._b320_terminal_status_detail(
        summary,
        reset_complete=False,
        hold_seconds=1.5,
    )
    after = recorder._b320_terminal_status_detail(
        summary,
        reset_complete=True,
    )

    assert "纸箱侧面受力超限" in before
    assert "非任务成功" in before
    assert "左 12.48 N / 右 2.25 N" in before
    assert "step=34" in before
    assert "sim_time=1.100 s" in before
    assert "保留 1.5 秒" in before
    assert "轨迹已丢弃" in after
    assert "场景已完成复位" in after


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, recorder.B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT),
        ("invalid", recorder.B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT),
        ("-3", 0.0),
        ("0.25", 0.25),
        ("99", recorder.B320_TERMINAL_RESET_HOLD_SECONDS_MAX),
    ],
)
def test_b320_terminal_reset_hold_is_configurable_and_bounded(
    raw_value: str | None,
    expected: float,
) -> None:
    environment = {}
    if raw_value is not None:
        environment[recorder.B320_TERMINAL_RESET_HOLD_SECONDS_ENV] = raw_value

    assert recorder._b320_terminal_reset_hold_seconds(environment) == expected


def test_b320_terminal_hold_uses_wall_time_only_for_unsuccessful_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(recorder.time, "sleep", slept.append)
    failed_summary = {"outcome": "failure"}

    duration = recorder._hold_b320_unsuccessful_terminal_frame(
        "t10_lift_carton",
        failed_summary,
        seconds=0.4,
    )
    success_duration = recorder._hold_b320_unsuccessful_terminal_frame(
        "t10_lift_carton",
        {"outcome": "success"},
        seconds=0.4,
    )
    other_duration = recorder._hold_b320_unsuccessful_terminal_frame(
        "t26_place_fruit_in_bucket",
        failed_summary,
        seconds=0.4,
    )

    assert duration == pytest.approx(0.4)
    assert success_duration == 0.0
    assert other_duration == 0.0
    assert slept == [pytest.approx(0.4)]
