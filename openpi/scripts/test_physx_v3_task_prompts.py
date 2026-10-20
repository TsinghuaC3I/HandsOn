import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("build_physx_v3_actual_eef52_lerobot_v21.py")
SPEC = importlib.util.spec_from_file_location("physx_v3_converter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("task", "target", "expected"),
    [
        ("t26_place_fruit_in_bucket", "apple", "Pick the apple and place it into the bucket."),
        ("t26_place_fruit_in_bucket", "011_banana", "Pick the banana and place it into the bucket."),
        ("t24_place_ball_in_bowl", "tennis_ball", "Pick the tennis ball and place it into the bowl."),
        ("t24_place_ball_in_bowl", "mini_soccer_ball", "Pick the mini soccer ball and place it into the bowl."),
        ("t25_place_cup_on_plate", "025_mug", "Pick the mug and place it into the tray."),
        ("t19_orient_cup_upright", "a_cup", "Pick up the cup and place it upright on the table."),
        ("t14_move_and_click_mouse", "", "Move the mouse and click it."),
    ],
)
def test_task_prompt(task: str, target: str, expected: str) -> None:
    assert MODULE.task_prompt(task, target) == expected


@pytest.mark.parametrize(
    ("task", "target"),
    [
        ("t26_place_fruit_in_bucket", ""),
        ("t24_place_ball_in_bowl", "basketball"),
        ("t19_orient_cup_upright", "rubiks_cube"),
        ("unknown_task", ""),
    ],
)
def test_task_prompt_rejects_unknown_semantics(task: str, target: str) -> None:
    with pytest.raises(ValueError):
        MODULE.task_prompt(task, target)
