import ast
import json
from pathlib import Path

import pytest

from scripts import canary_loaded_tray_platform as canary
from handson.sim.isaaclab.loaded_tray_gate_scene import (
    GATE_CLEAR_WIDTH,
    GATE_PLANE_X,
    GATE_POST_HALF_X,
    GATE_SOURCE_CLEAR_WIDTH,
    LEFT_GRIP_LOCAL_POSITION,
    RIGHT_GRIP_LOCAL_POSITION,
    TRAY_GATE_CLEARANCE_M,
    TRAY_FORWARD_HALF_EXTENT,
    TRAY_HALF_EXTENTS,
    TRAY_PLACED_CENTER_X,
    TRAY_SCRIPTED_PLACED_CENTER_X,
    TRAY_START_POSITION,
    tray_local_position_to_world,
)


SCRIPTED_TRAJECTORY = (
    Path(__file__).resolve().parents[1]
    / "probe_loaded_tray_gate_scripted_trajectory.py"
)


def _scripted_probe_argument_defaults() -> dict[str, object]:
    tree = ast.parse(SCRIPTED_TRAJECTORY.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "add_argument"
            or not node.args
            or not isinstance(node.args[0], ast.Constant)
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            if isinstance(keyword.value, ast.Constant):
                defaults[str(node.args[0].value)] = keyword.value.value
            else:
                defaults[str(node.args[0].value)] = ast.unparse(keyword.value)
    return defaults


def _make_pending_episode(root: Path, name: str, task_name: str) -> Path:
    episode = root / ".physical_replay_pending" / name
    (episode / "replay.zarr").mkdir(parents=True)
    (episode / "DATASET_PHYSX_V3.json").write_text("{}\n", encoding="utf-8")
    (episode / "replay_manifest.json").write_text(
        json.dumps({"task": {"name": task_name}}) + "\n",
        encoding="utf-8",
    )
    return episode


def test_pending_episode_items_accepts_only_complete_matching_save(tmp_path: Path) -> None:
    valid = _make_pending_episode(
        tmp_path,
        f"{canary.TASK_NAME}_demo_1__physx_v3",
        canary.TASK_NAME,
    )
    _make_pending_episode(
        tmp_path,
        f"{canary.TASK_NAME}_demo_2__physx_v3",
        "wrong_task",
    )
    incomplete = (
        tmp_path
        / ".physical_replay_pending"
        / f"{canary.TASK_NAME}_demo_3__physx_v3"
    )
    incomplete.mkdir(parents=True)

    assert canary._pending_episode_items(tmp_path) == [
        {
            "episode_id": valid.name,
            "path": str(valid.resolve()),
            "publication_state": "physical_replay_pending",
            "task_name": canary.TASK_NAME,
        }
    ]


def test_saved_episode_items_keeps_api_and_pending_items_unique(tmp_path: Path) -> None:
    valid = _make_pending_episode(
        tmp_path,
        f"{canary.TASK_NAME}_demo_1__physx_v3",
        canary.TASK_NAME,
    )
    response = {
        "items": [
            {
                "episode_id": valid.name,
                "path": "/published/path",
                "task_name": canary.TASK_NAME,
            }
        ]
    }

    assert canary._saved_episode_items(response, tmp_path) == response["items"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (canary.EXPECTED_COMPLETION_PREVIEW_ERROR, True),
        ("TimeoutError: trajectory did not finish before its deadline", False),
        ("RuntimeError: UDP pump failed", False),
    ],
)
def test_expected_completion_preview_error_is_exact(
    message: str, expected: bool
) -> None:
    assert canary._is_expected_completion_preview_error(message) is expected


def test_canary_waypoints_follow_selected_asset_scene_metadata() -> None:
    assert canary.GATE_PLANE_X == pytest.approx(GATE_PLANE_X)
    assert canary.FINAL_TRAY_CENTER_X == pytest.approx(
        TRAY_SCRIPTED_PLACED_CENTER_X
    )
    for reset, approach, local_grip in (
        (
            canary.RESET_RIGHT_POSITION,
            canary.APPROACH_RIGHT,
            RIGHT_GRIP_LOCAL_POSITION,
        ),
        (
            canary.RESET_LEFT_POSITION,
            canary.APPROACH_LEFT,
            LEFT_GRIP_LOCAL_POSITION,
        ),
    ):
        expected = list(
            tray_local_position_to_world(TRAY_START_POSITION, local_grip)
        )
        expected[2] += canary.PALM_ABOVE_GRIP_M
        assert reset + approach == pytest.approx(expected)

    for reset, crossing, final, local_grip in (
        (
            canary.RESET_RIGHT_POSITION,
            canary.CROSSING_RIGHT,
            canary.FINAL_RIGHT,
            RIGHT_GRIP_LOCAL_POSITION,
        ),
        (
            canary.RESET_LEFT_POSITION,
            canary.CROSSING_LEFT,
            canary.FINAL_LEFT,
            LEFT_GRIP_LOCAL_POSITION,
        ),
    ):
        crossing_expected = list(
            tray_local_position_to_world(
                (GATE_PLANE_X, 0.0, TRAY_START_POSITION[2] + canary.LIFT_HEIGHT_M),
                local_grip,
            )
        )
        crossing_expected[2] += canary.PALM_ABOVE_GRIP_M
        final_expected = list(
            tray_local_position_to_world(
                (
                    TRAY_SCRIPTED_PLACED_CENTER_X,
                    0.0,
                    TRAY_START_POSITION[2] + canary.LIFT_HEIGHT_M,
                ),
                local_grip,
            )
        )
        final_expected[2] += canary.PALM_ABOVE_GRIP_M
        assert reset + crossing == pytest.approx(crossing_expected)
        assert reset + final == pytest.approx(final_expected)


def test_scripted_probe_defaults_stay_on_selected_side_rim_grip() -> None:
    defaults = _scripted_probe_argument_defaults()
    assert defaults["--left-command-bias-x"] == pytest.approx(0.0)
    assert defaults["--left-command-bias-y"] == pytest.approx(0.0)
    assert defaults["--left-command-bias-z"] == pytest.approx(0.0)
    assert LEFT_GRIP_LOCAL_POSITION[0] > 0.0
    assert RIGHT_GRIP_LOCAL_POSITION[0] < 0.0
    assert defaults["--transport-final-center-x"] == (
        "float(TRAY_SCRIPTED_PLACED_CENTER_X)"
    )
    assert TRAY_PLACED_CENTER_X == pytest.approx(
        GATE_PLANE_X
        + GATE_POST_HALF_X
        + TRAY_GATE_CLEARANCE_M
        + TRAY_FORWARD_HALF_EXTENT
    )
    assert TRAY_SCRIPTED_PLACED_CENTER_X == pytest.approx(
        TRAY_PLACED_CENTER_X + 0.010
    )
    assert GATE_SOURCE_CLEAR_WIDTH == pytest.approx(2.0 * TRAY_HALF_EXTENTS[1])
    assert GATE_CLEAR_WIDTH < GATE_SOURCE_CLEAR_WIDTH
