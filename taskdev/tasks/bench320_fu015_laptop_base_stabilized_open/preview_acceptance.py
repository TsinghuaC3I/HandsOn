"""Fail-closed acceptance parser for the FU015 bounded 420-step preview.

This runner reads an already-finished scheduler stdout log.  It never starts
Isaac Sim, writes a trajectory, or invokes replay/certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Sequence


TASK_ID = "B320-FU015"
DEFAULT_MAX_STEPS = 420
EXPECTED_FILTER_COUNTS = {
    "left_hand": 21,
    "right_hand": 21,
    "robot_table": 66,
}
MARKER_PREFIX = "[HandsOn/FU015][PREVIEW_ACCEPTANCE] "
CONTACT_READY_RE = re.compile(
    r"Exact contact contract ready:\s*"
    r"left_hand=(?P<left>\d+),\s*"
    r"right_hand=(?P<right>\d+),\s*"
    r"robot_table_filters=(?P<table>\d+)"
)
FAILURE_RE = re.compile(r"\[HandsOn/FU015\] Failure latched at action step (\d+):")


def _filter_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {
            "left_hand": int(value["left_hand"]),
            "right_hand": int(value["right_hand"]),
            "robot_table": int(value["robot_table"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_bounded_preview_log(
    text: str,
    *,
    expected_max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Return a deterministic report; any missing physical proof fails closed."""

    errors: list[str] = []
    startup_counts = [
        {
            "left_hand": int(match.group("left")),
            "right_hand": int(match.group("right")),
            "robot_table": int(match.group("table")),
        }
        for match in CONTACT_READY_RE.finditer(text)
    ]
    if not startup_counts:
        errors.append("missing exact contact-contract startup evidence")
    elif any(counts != EXPECTED_FILTER_COUNTS for counts in startup_counts):
        errors.append("contact filters must remain exactly 21/21/66")

    markers: list[dict[str, Any]] = []
    malformed_markers = 0
    for line in text.splitlines():
        if MARKER_PREFIX not in line:
            continue
        raw = line.split(MARKER_PREFIX, 1)[1].strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            malformed_markers += 1
            continue
        if isinstance(value, dict):
            markers.append(value)
        else:
            malformed_markers += 1
    if malformed_markers:
        errors.append("malformed FU015 preview-acceptance marker")

    transit_events = [item for item in markers if item.get("event") == "transit_clearance"]
    if not transit_events:
        errors.append("missing completed-transit clearance evidence")
    for event in transit_events:
        if event.get("task_id") != TASK_ID:
            errors.append("transit marker task identity mismatch")
        try:
            transit_step = int(event.get("env_step", -1))
        except (TypeError, ValueError):
            transit_step = -1
        if transit_step != 45:
            errors.append("transit marker must describe the completed step-44 action")
        if _filter_counts(event.get("filter_counts")) != EXPECTED_FILTER_COUNTS:
            errors.append("transit marker does not bind exact 21/21/66 filters")
        if event.get("robot_table_clear") is not True:
            errors.append("robot/table contact occurred during transit")
        for side in ("left", "right"):
            try:
                force = float(event[f"{side}_robot_table_peak_force_n"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"missing {side} transit peak table-force scalar")
                continue
            if not 0.0 <= force <= 0.02:
                errors.append(f"{side} transit table force exceeds 0.02 N")

    terminal_events = [item for item in markers if item.get("event") == "terminal"]
    successful_terminals = [
        item
        for item in terminal_events
        if item.get("terminated") is True
        and item.get("succeed") is True
        and item.get("failed") is False
    ]
    if not successful_terminals:
        errors.append("missing physically successful terminal event")
    for event in terminal_events:
        if event.get("task_id") != TASK_ID:
            errors.append("terminal marker task identity mismatch")
        if _filter_counts(event.get("filter_counts")) != EXPECTED_FILTER_COUNTS:
            errors.append("terminal marker does not bind exact 21/21/66 filters")
        if event.get("terminated") is not True:
            errors.append("terminal marker is not terminated")
        if event.get("succeed") is not True or event.get("failed") is not False:
            errors.append("terminal semantics are not succeed=true/failed=false")
        if event.get("left_base_contact") is not True:
            errors.append("terminal event lacks left-base role contact")
        if event.get("right_display_contact") is not True:
            errors.append("terminal event lacks right-display role contact")
        if event.get("robot_table_clear") is not True:
            errors.append("terminal event has forbidden robot/table contact")
        try:
            observed = int(event["success_counter"])
            required = int(event["success_steps_required"])
        except (KeyError, TypeError, ValueError):
            errors.append("terminal event lacks success-hold counters")
        else:
            if required != 15 or observed < required:
                errors.append("terminal event did not satisfy the 15-step hold")

    failure_steps = [int(value) for value in FAILURE_RE.findall(text)]
    if failure_steps:
        errors.append("FU015 latched a physical failure during the bounded preview")

    reached_line = f"Reached max_steps={int(expected_max_steps)}; stopping run."
    ended_line = f"Preview ended after {int(expected_max_steps)} steps."
    if reached_line not in text or ended_line not in text:
        errors.append(f"preview did not complete the exact {int(expected_max_steps)}-step bound")

    forbidden_artifact_markers = [
        token
        for token in (
            "[Saved replay.zarr]",
            "fresh_process_acceptance.json",
            "task_success_replay",
            "[Saved] success #",
        )
        if token in text
    ]
    if forbidden_artifact_markers:
        errors.append("preview-only log contains collection/replay artifact markers")

    return {
        "format": "handson.fu015.bounded_preview_acceptance.v1",
        "task_id": TASK_ID,
        "expected_max_steps": int(expected_max_steps),
        "passed": not errors,
        "errors": errors,
        "startup_filter_counts": startup_counts,
        "transit_events": transit_events,
        "terminal_events": terminal_events,
        "failure_steps": failure_steps,
        "forbidden_artifact_markers": forbidden_artifact_markers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stdout_log", type=Path)
    parser.add_argument("--expected-max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_max_steps <= 0:
        raise SystemExit("--expected-max-steps must be positive")
    report = evaluate_bounded_preview_log(
        args.stdout_log.read_text(encoding="utf-8", errors="replace"),
        expected_max_steps=args.expected_max_steps,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
