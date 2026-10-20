from __future__ import annotations

from pathlib import Path

import yaml

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
)
from handson.sim.isaaclab.b320_bc017_contract import (
    OPEN_THRESHOLD,
    PHYSICS_HZ,
    SYNC_WINDOW_SECONDS,
    SYNC_WINDOW_STEPS,
    TASK_NAME,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKDEV_ROOT = (
    REPO_ROOT / "taskdev/tasks/bench320_bc017_dual_drawer_synchronized_open"
)
CANONICAL_PROMPT = (
    "Use each assigned hand to pull the highlighted drawer on its side beyond "
    "60 percent travel; each side independently selects one of three drawers "
    "per episode, and both thresholds must be crossed within 1.2 seconds."
)


def _yaml(name: str) -> dict[str, object]:
    return yaml.safe_load((TASKDEV_ROOT / name).read_text(encoding="utf-8"))


def test_bc017_keeps_literal_success_semantics() -> None:
    contract = _yaml("task.yaml")
    assert contract["task"]["name"] == TASK_NAME
    assert contract["task"]["description"] == CANONICAL_PROMPT
    assert contract["runtime"]["physics_hz"] == PHYSICS_HZ == 120
    assert OPEN_THRESHOLD == 0.60
    assert SYNC_WINDOW_SECONDS == 1.2
    assert SYNC_WINDOW_STEPS == 144
    assert contract["success"]["threshold"] == "> 0.60"
    assert contract["success"]["release_required"] is False


def test_bc017_taskdev_locks_independent_selection_and_replay_metadata() -> None:
    contract = _yaml("task.yaml")
    workflow = _yaml("workflow_state.yaml")

    selection = contract["target_selection"]
    assert selection["candidates_per_side"] == ["top", "middle", "bottom"]
    assert selection["independently_randomized_between_sides"] is True
    assert selection["resampled"] == "every_episode"
    assert selection["fixed_within_episode"] is True
    assert selection["left_marker"] == "emissive_cyan_outline"
    assert selection["right_marker"] == "emissive_orange_outline"
    assert selection["success_scope"] == "selected_pair_only"
    assert selection["nonselected_drawer_motion_terminal"] is False

    assert contract["observations"]["required"][:3] == [
        "two_selected_target_indices",
        "six_normalized_drawer_progress_values",
        "twelve_hand_drawer_contact_channels",
    ]
    assert contract["collection"]["episode_metadata_required"] == [
        "selected_left_drawer_level",
        "selected_right_drawer_level",
    ]
    assert contract["success"]["nonselected_drawers_count_toward_success"] is False
    assert contract["success"]["nonselected_drawer_motion_terminal"] is False

    assert workflow["episode_target_selection"] == {
        "candidates_per_side": ["top", "middle", "bottom"],
        "independently_randomized_between_sides": True,
        "resampled": "every_episode",
        "fixed_within_episode": True,
        "left_marker": "emissive_cyan_outline",
        "right_marker": "emissive_orange_outline",
        "only_selected_pair_counts_toward_success": True,
        "nonselected_drawer_motion_terminal": False,
    }
    assert workflow["recording_metadata"] == {
        "fixed_cardinality_drawer_progress_values": 6,
        "fixed_cardinality_hand_drawer_contact_channels": 12,
        "selected_target_indices": ["left", "right"],
    }


def test_bc017_is_collection_ready_without_task_specific_gates() -> None:
    contract = _yaml("task.yaml")
    workflow = _yaml("workflow_state.yaml")
    assert contract["asset"]["selection_status"] == "integrated"
    assert contract["collection"]["collection_ready"] is True
    assert workflow["state"] == "collection_ready"
    assert workflow["preview_ready"] is True
    assert workflow["collection_ready"] is True
    assert workflow["blocked"] is False
    assert workflow["task_specific_gates"] == []
    assert TASK_NAME not in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    assert TASK_NAME in ISAACSIM_FORMAL_COLLECTION_TASKS
