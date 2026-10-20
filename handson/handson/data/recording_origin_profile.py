"""Lightweight task-family classification for recording-origin contracts."""

from __future__ import annotations
from ..runtime.task_families import ABC_PREFIX_TASKS


ABC_STYLE_COLLECTION_TASKS = frozenset(
    {
        "z01_pipette",
        "z02_bottle_cap",
        # These tasks use the native ABC robot/PhysX collection scene but have
        # descriptive runtime names rather than A/B/C-prefixed task names.
        "eval_gen03_orient_bottle_upright",
        "eval_gen01_cover_lunchbox",
        "eval_gen02_open_trash_bin",
        "eval_gen04_pass_glue_stick",
        "eval_gen07_stack_boxes",
        "eval_gen05_set_mixer_knobs",
        "eval_gen06_spray_shoe",
        "eval_comb04_place_fruit_and_photograph_logo",
        "eval_comb02_open_laptop_and_click_mouse",
        "loaded_tray_through_narrow_gate",
        "eval_comb06_season_and_stir",
        "eval_comb01_cover_pitcher_and_box",
        # The YCB collection tasks all run through the awake ABC PhysX
        # recorder contract, even though their public identifiers use the
        # YCB catalogue prefix rather than A/B/C.
        "t34_roll_ball_through_gate",
        "t27_place_largest_fitting_ball",
        "t35_rotate_propeller",
        "t47_trigger_drill_on_block",
        "t29_place_wine_glass",
        "t33_reposition_marker_grasp",
        "t41_sort_nested_cups",
        "t28_place_tableware",
        "t39_set_timer",
        "t07_flip_sponge",
        "eval_flex01_compress_sponge_into_mug",
    }
)


def is_abc_style_collection_task(task_name: object) -> bool:
    """Return whether a task uses the awake ABC recording-origin contract."""

    normalized_task_name = str(task_name or "").strip().lower()
    return (
        normalized_task_name in ABC_PREFIX_TASKS
        or normalized_task_name[:1] in {"a", "b", "c"}
        or normalized_task_name in ABC_STYLE_COLLECTION_TASKS
    )
