"""Authoritative task capabilities shared by the UI and demo recorder."""

from __future__ import annotations
from .task_families import ABC_PREFIX_TASKS, ABCD_PREFIX_TASKS, ORDINARY_ABC_TASKS, YCB_TASKS


BC011_COLLECTION_ANGLE_GUIDANCE_ENV = (
    "HANDSON_BC011_COLLECTION_ANGLE_GUIDANCE"
)


ISAACSIM_COLLECTION_TASKS = frozenset(
    {
        "eval_comb03_pass_block_and_close_box",
        "eval_comb02_open_laptop_and_click_mouse",
        "eval_comb04_place_fruit_and_photograph_logo",
        "t26_place_fruit_in_bucket",
        "t25_place_cup_on_plate",
        "t24_place_ball_in_bowl",
        "t34_roll_ball_through_gate",
        "t27_place_largest_fitting_ball",
        "t41_sort_nested_cups",
        "t28_place_tableware",
        "t39_set_timer",
        "t07_flip_sponge",
        "eval_flex01_compress_sponge_into_mug",
        "t19_orient_cup_upright",
        "eval_gen03_orient_bottle_upright",
        "eval_gen01_cover_lunchbox",
        "t04_cover_pitcher",
        "t09_insert_peg",
        "t44_stack_blocks",
        "t03_cover_box",
        "eval_comb01_cover_pitcher_and_box",
        "t40_slide_knife_blade",
        "t22_pass_screwdriver",
        "t16_open_laptop",
        "t37_scrub_bottle",
        "eval_comb05_scrub_bottle_then_fill_cup",
        "t48_unfold_fan",
        "t17_open_latched_door",
        "t08_hammer_nail",
        "t36_scoop_marbles",
        "t50_water_plant",
        "t14_move_and_click_mouse",
        "t23_photograph_logo",
        "t02_connect_charger_and_cable",
        "t46_store_charger",
        "t11_line_up_blocks",
        "t13_load_cans",
        "t12_load_and_start_microwave",
        "t21_pass_block",
        "eval_comb03_pass_block_and_close_box",
        "t43_squeeze_scissors",
        "t42_sort_objects",
        "t31_prepare_coffee",
        "t35_rotate_propeller",
        "eval_flex02_fold_towel_twice",
        "eval_flex03_pack_pillow",
        "t06_fit_device_cover",
        "t18_open_micropipette_box",
        "eval_gen02_open_trash_bin",
        "eval_gen04_pass_glue_stick",
        "eval_gen07_stack_boxes",
        "eval_gen05_set_mixer_knobs",
        "eval_gen06_spray_shoe",
        "t49_unfold_ruler",
        "t10_lift_carton",
        "t20_pass_and_orient_pen",
        "t30_pour_seasonings",
        "eval_comb06_season_and_stir",
        "t15_open_drawers",
        "t01_align_blocks_in_midair",
        "t45_stir_pot",
        "t32_press_soap_pump",
        "t05_fill_cup",
        "t38_set_dual_knobs",
        "t47_trigger_drill_on_block",
        "t29_place_wine_glass",
        "t33_reposition_marker_grasp",
    }
)

# No registered Isaac Sim task is currently preview-only.  Keep this explicit
# capability set so the UI and recorder can continue to fail closed if a
# diagnostic-only scene is introduced later.
ISAACSIM_PREVIEW_TASKS = frozenset()

# No registered task is denied formal collection.  Keep the empty capability
# set as a compatibility surface for the UI, recorder, and older callers.
ISAACSIM_COLLECTION_UNVERIFIED_TASKS = frozenset()

# Effective formal capability used by status and downstream certification
# consumers. It is intentionally separate from the collection-start gate.
ISAACSIM_FORMAL_COLLECTION_TASKS = frozenset(
    ISAACSIM_COLLECTION_TASKS
)

# Canonical RGB capture is never allowed to compete with formal Isaac PhysX
# collection.  Every registered formal task records state/actions online and
# reconstructs exactly ``main``, ``wrist_left`` and ``wrist_right`` in the
# shared offline renderer after the simulator child exits.  Keep this set
# derived from the formal capability instead of maintaining another task-name
# allowlist which can drift when a task is added.
ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS = frozenset(
    ISAACSIM_FORMAL_COLLECTION_TASKS
)

YCB02_COLLECTION_TASK = "t27_place_largest_fitting_ball"
_AWAKE_CONTACT_COLLECTION_TASKS = frozenset(
    {
        "eval_comb01_cover_pitcher_and_box",
        "t34_roll_ball_through_gate",
        YCB02_COLLECTION_TASK,
        "t28_place_tableware",
        "t39_set_timer",
        "t07_flip_sponge",
    }
)

def _normalized_task_name(task_name: object) -> str:
    return str(task_name or "").strip().lower()


def is_abc_collection_task_name(task_name: object) -> bool:
    """Return whether a task uses the awake-contact ABC collection contract."""

    normalized = _normalized_task_name(task_name)
    return bool(
        (normalized in ABC_PREFIX_TASKS or normalized[:1] in {"a", "b", "c"})
        or normalized == YCB02_COLLECTION_TASK
        or normalized == "eval_gen03_orient_bottle_upright"
    )


def uses_awake_contact_collection_origin(task_name: object) -> bool:
    """Return whether reset must preserve the awake PhysX contact manifold."""

    normalized = _normalized_task_name(task_name)
    return bool(
        (normalized in ABC_PREFIX_TASKS or normalized[:1] in {"a", "b", "c"})
        or normalized in _AWAKE_CONTACT_COLLECTION_TASKS
        or normalized == "eval_gen03_orient_bottle_upright"
    )


def uses_deferred_arm_collection_pipeline(task_name: object) -> bool:
    """Return whether collection records hands before rendering full arms."""

    normalized = _normalized_task_name(task_name)
    return bool(
        normalized != "bimanual_assembly"
        and (
            (normalized in ABCD_PREFIX_TASKS or normalized[:1] in {"a", "b", "c", "d"})
            or (normalized in YCB_TASKS or normalized.startswith("ycb"))
            or normalized in {"eval_gen03_orient_bottle_upright", "eval_gen04_pass_glue_stick", "eval_gen07_stack_boxes"}
        )
    )


def is_deferred_arm_collection_task_name(task_name: object) -> bool:
    """Compatibility alias for callers using the older helper name."""

    return uses_deferred_arm_collection_pipeline(task_name)


def uses_wrist_camera_rerender_pipeline(task_name: object) -> bool:
    """Return whether an episode is eligible for the legacy wrist rerender."""

    normalized = _normalized_task_name(task_name)
    ordinary_abc_task = bool(
        len(normalized) >= 4
        and normalized[0] in {"a", "b", "c"}
        and normalized[1:3].isdigit()
        and normalized[3] == "_"
    )
    return bool(normalized in ORDINARY_ABC_TASKS or ordinary_abc_task or (normalized in YCB_TASKS or normalized.startswith("ycb"))
                or normalized == "eval_gen03_orient_bottle_upright")


def supports_isaacsim_deferred_camera_render(
    sim_backend: str,
    task_name: str,
) -> bool:
    """Return whether a formal run must use canonical three-camera replay."""

    return bool(
        str(sim_backend).strip().lower() == "isaacsim"
        and str(task_name).strip().lower()
        in ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS
    )

# Task names outside the ordinary ``a01_...`` through ``d19_...`` naming
# scheme that support the detached-arm reconstruction path.  Canonical camera
# deferral is intentionally broader and is derived from every formal Isaac
# task above; do not use this arm-specific set as a camera allowlist.
ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS = frozenset(
    {
        "eval_gen03_orient_bottle_upright",
        "t10_lift_carton",
        "t15_open_drawers",
        "t01_align_blocks_in_midair",
        # The paper evaluator restores this recorded source capability.
        "t13_load_cans",
        "eval_flex02_fold_towel_twice",
    }
)

__all__ = [
    "BC011_COLLECTION_ANGLE_GUIDANCE_ENV",
    "ISAACSIM_COLLECTION_TASKS",
    "ISAACSIM_PREVIEW_TASKS",
    "ISAACSIM_COLLECTION_UNVERIFIED_TASKS",
    "ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS",
    "ISAACSIM_FORMAL_COLLECTION_TASKS",
    "ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS",
    "YCB02_COLLECTION_TASK",
    "is_abc_collection_task_name",
    "is_deferred_arm_collection_task_name",
    "uses_awake_contact_collection_origin",
    "uses_deferred_arm_collection_pipeline",
    "uses_wrist_camera_rerender_pipeline",
    "supports_isaacsim_deferred_camera_render",
]
