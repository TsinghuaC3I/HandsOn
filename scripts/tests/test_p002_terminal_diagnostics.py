from scripts import record_demos_zarr as recorder


P002_TASK = "t17_open_latched_door"
HL015_TASK = "bench320_hl015_bimanual_cable_route_dual_plug"


def test_p002_first_physical_failure_is_persisted_in_journal_reason() -> None:
    reason = recorder._as013_unsuccessful_terminal_reason(
        P002_TASK,
        {"p002_terminal_reason": "forbidden_rear_handle_contact"},
    )

    assert reason == (
        "unsuccessful_terminal_transition:"
        "p002_first_terminal=forbidden_rear_handle_contact"
    )


def test_p002_timeout_and_missing_reason_remain_distinct() -> None:
    timeout = recorder._as013_unsuccessful_terminal_reason(
        P002_TASK,
        {"p002_terminal_reason": "time_limit"},
    )
    missing = recorder._as013_unsuccessful_terminal_reason(P002_TASK, {})

    assert timeout == (
        "unsuccessful_terminal_transition:p002_first_terminal=time_limit"
    )
    assert missing == "unsuccessful_terminal_transition"


def test_p002_visible_preview_requires_production_stable_origin() -> None:
    assert recorder._stable_recording_origin_required_for_run(
        P002_TASK,
        "isaacsim",
        True,
    )
    assert recorder._stable_recording_origin_required_for_run(
        P002_TASK,
        "isaacsim",
        False,
    )
    assert not recorder._stable_recording_origin_required_for_run(
        P002_TASK,
        "mujoco",
        True,
    )
    assert not recorder._stable_recording_origin_required_for_run(
        "t26_place_fruit_in_bucket",
        "isaacsim",
        True,
    )


def test_lw_visible_previews_require_stable_recording_origin() -> None:
    """LW previews must not publish a ready state before physical settling."""

    for task_name in (
        "g05_folding_fan_bimanual_open",
        "t18_open_micropipette_box",
        "t49_unfold_ruler",
    ):
        assert recorder._stable_recording_origin_required_for_run(
            task_name,
            "isaacsim",
            True,
        )
        assert recorder._stable_recording_origin_required_for_run(
            task_name,
            "isaacsim",
            False,
        )
        assert not recorder._stable_recording_origin_required_for_run(
            task_name,
            "mujoco",
            True,
        )


def test_hl015_visible_preview_requires_stable_recording_origin() -> None:
    """Native bimanual previews must not skip the reset/stability gate."""

    for task_name in (HL015_TASK,):
        assert recorder._stable_recording_origin_required_for_run(
            task_name,
            "isaacsim",
            True,
        )
        assert recorder._stable_recording_origin_required_for_run(
            task_name,
            "isaacsim",
            False,
        )
        assert not recorder._stable_recording_origin_required_for_run(
            task_name,
            "mujoco",
            True,
        )


def test_existing_as013_journal_reason_is_unchanged() -> None:
    reason = recorder._as013_unsuccessful_terminal_reason(
        "t40_slide_knife_blade",
        {"as013_failure_reason": "blade_force_exceeded"},
    )

    assert reason == (
        "unsuccessful_terminal_transition:"
        "as013_first_failure=blade_force_exceeded"
    )
