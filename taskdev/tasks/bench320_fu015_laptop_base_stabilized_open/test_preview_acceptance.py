from __future__ import annotations

import json
from pathlib import Path

from taskdev.tasks.t16_open_laptop.preview_acceptance import (
    MARKER_PREFIX,
    evaluate_bounded_preview_log,
    main,
)


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]


def _marker(payload: dict[str, object]) -> str:
    return MARKER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _passing_log() -> str:
    counts = {"left_hand": 21, "right_hand": 21, "robot_table": 66}
    return "\n".join(
        [
            "[HandsOn/FU015] Exact contact contract ready: left_hand=21, right_hand=21, robot_table_filters=66",
            _marker(
                {
                    "event": "transit_clearance",
                    "task_id": "B320-FU015",
                    "env_step": 45,
                    "filter_counts": counts,
                    "left_robot_table_peak_force_n": 0.0,
                    "right_robot_table_peak_force_n": 0.0,
                    "robot_table_clear": True,
                }
            ),
            _marker(
                {
                    "event": "terminal",
                    "task_id": "B320-FU015",
                    "env_step": 344,
                    "filter_counts": counts,
                    "terminated": True,
                    "succeed": True,
                    "failed": False,
                    "success_counter": 15,
                    "success_steps_required": 15,
                    "left_base_contact": True,
                    "right_display_contact": True,
                    "robot_table_clear": True,
                }
            ),
            "Reached max_steps=420; stopping run.",
            "Preview ended after 420 steps.",
        ]
    )


def test_bounded_preview_acceptance_requires_exact_physical_proof(tmp_path: Path) -> None:
    report = evaluate_bounded_preview_log(_passing_log())
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["startup_filter_counts"] == [
        {"left_hand": 21, "right_hand": 21, "robot_table": 66}
    ]
    stdout_log = tmp_path / "stdout.log"
    output = tmp_path / "acceptance.json"
    stdout_log.write_text(_passing_log())
    assert main([str(stdout_log), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["passed"] is True


def test_bounded_preview_acceptance_fails_closed_for_each_scoped_gate() -> None:
    passing = _passing_log()
    mutations = {
        "filter_count": passing.replace("robot_table_filters=66", "robot_table_filters=65"),
        "transit": passing.replace(
            '"left_robot_table_peak_force_n":0.0',
            '"left_robot_table_peak_force_n":0.03',
        ),
        "role_contact": passing.replace('"right_display_contact":true', '"right_display_contact":false'),
        "terminal": passing.replace('"succeed":true', '"succeed":false'),
        "physical_failure": passing.replace(
            "Reached max_steps=420; stopping run.",
            "[HandsOn/FU015] Failure latched at action step 52: {}\nReached max_steps=420; stopping run.",
        ),
        "collection": passing + "\n[Saved replay.zarr] forbidden",
    }
    for name, text in mutations.items():
        report = evaluate_bounded_preview_log(text)
        assert report["passed"] is False, name
        assert report["errors"], name


def test_live_runtime_wires_transit_and_terminal_markers_without_state_writes() -> None:
    source = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/fu015_collection_env.py"
    ).read_text()
    hook = source.split("def _emit_fu015_preview_acceptance_marker", 1)[1].split(
        "def _task_info", 1
    )[0]
    assert '"event": "transit_clearance"' in hook
    assert '"event": "terminal"' in hook
    assert '"filter_counts": filter_counts' in hook
    assert '"left_base_contact"' in hook
    assert '"right_display_contact"' in hook
    assert "set_root_transforms" not in hook
    assert "set_dof_positions" not in hook
