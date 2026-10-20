"""Task catalog shared by factories, evaluation and model services.

Imports stay lightweight: native task implementations load only on demand.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

ASSETS = Path(__file__).with_name("assets")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    config_module: str
    split: str
    checkpoint_name: str
    goal_reader: str
    camera_environment_module: str = ""
    trace_environment_module: str = ""
    bridge: str = ""
    force_scoring: bool = False
    terminal_contact_flag: str = ""

    def native_contract(self):
        return native_contracts().get(self.name)

    def goal(self, env):
        from . import goal_readers
        return getattr(goal_readers, self.goal_reader)(env)

    def evaluation_adapter(self):
        from ..evaluation.task_adapters import NativeTaskAdapter
        return NativeTaskAdapter(self)


_CONFIG_MODULES = {
    "eval_comb02_open_laptop_and_click_mouse": ".eval_comb02_open_laptop_and_click_mouse.config",
    "eval_comb04_place_fruit_and_photograph_logo": ".eval_comb04_place_fruit_and_photograph_logo.config",
    "t26_place_fruit_in_bucket": ".t26_place_fruit_in_bucket.config",
    "t25_place_cup_on_plate": ".t25_place_cup_on_plate.config",
    "t24_place_ball_in_bowl": ".t24_place_ball_in_bowl.config",
    "t34_roll_ball_through_gate": ".t34_roll_ball_through_gate.config",
    "t27_place_largest_fitting_ball": ".t27_place_largest_fitting_ball.config",
    "t41_sort_nested_cups": ".t41_sort_nested_cups.config",
    "t28_place_tableware": ".t28_place_tableware.config",
    "t39_set_timer": ".t39_set_timer.config",
    "t07_flip_sponge": ".t07_flip_sponge.config",
    "eval_flex01_compress_sponge_into_mug": ".eval_flex01_compress_sponge_into_mug.config",
    "t19_orient_cup_upright": ".t19_orient_cup_upright.config",
    "eval_gen03_orient_bottle_upright": ".eval_gen03_orient_bottle_upright.config",
    "eval_gen01_cover_lunchbox": ".eval_gen01_cover_lunchbox.config",
    "t04_cover_pitcher": ".t04_cover_pitcher.config",
    "t09_insert_peg": ".t09_insert_peg.config",
    "t44_stack_blocks": ".t44_stack_blocks.config",
    "eval_gen07_stack_boxes": ".eval_gen07_stack_boxes.config",
    "t03_cover_box": ".t03_cover_box.config",
    "eval_comb01_cover_pitcher_and_box": ".eval_comb01_cover_pitcher_and_box.config",
    "t40_slide_knife_blade": ".t40_slide_knife_blade.config",
    "t22_pass_screwdriver": ".t22_pass_screwdriver.config",
    "t16_open_laptop": ".t16_open_laptop.config",
    "eval_comb05_scrub_bottle_then_fill_cup": ".eval_comb05_scrub_bottle_then_fill_cup.config",
    "t37_scrub_bottle": ".t37_scrub_bottle.config",
    "t48_unfold_fan": ".t48_unfold_fan.config",
    "t17_open_latched_door": ".t17_open_latched_door.config",
    "t08_hammer_nail": ".t08_hammer_nail.config",
    "t36_scoop_marbles": ".t36_scoop_marbles.config",
    "t50_water_plant": ".t50_water_plant.config",
    "eval_gen06_spray_shoe": ".eval_gen06_spray_shoe.config",
    "t14_move_and_click_mouse": ".t14_move_and_click_mouse.config",
    "t23_photograph_logo": ".t23_photograph_logo.config",
    "t02_connect_charger_and_cable": ".t02_connect_charger_and_cable.config",
    "t46_store_charger": ".t46_store_charger.config",
    "t11_line_up_blocks": ".t11_line_up_blocks.config",
    "t13_load_cans": ".t13_load_cans.config",
    "t12_load_and_start_microwave": ".t12_load_and_start_microwave.config",
    "eval_comb03_pass_block_and_close_box": ".eval_comb03_pass_block_and_close_box.config",
    "t21_pass_block": ".t21_pass_block.config",
    "eval_gen04_pass_glue_stick": ".eval_gen04_pass_glue_stick.config",
    "t43_squeeze_scissors": ".t43_squeeze_scissors.config",
    "t42_sort_objects": ".t42_sort_objects.config",
    "t31_prepare_coffee": ".t31_prepare_coffee.config",
    "t35_rotate_propeller": ".t35_rotate_propeller.config",
    "eval_flex02_fold_towel_twice": ".eval_flex02_fold_towel_twice.config",
    "eval_flex03_pack_pillow": ".eval_flex03_pack_pillow.config",
    "t06_fit_device_cover": ".t06_fit_device_cover.config",
    "t18_open_micropipette_box": ".t18_open_micropipette_box.config",
    "eval_gen02_open_trash_bin": ".eval_gen02_open_trash_bin.config",
    "t49_unfold_ruler": ".t49_unfold_ruler.config",
    "t10_lift_carton": ".t10_lift_carton.config",
    "t20_pass_and_orient_pen": ".t20_pass_and_orient_pen.config",
    "t30_pour_seasonings": ".t30_pour_seasonings.config",
    "eval_comb06_season_and_stir": ".eval_comb06_season_and_stir.config",
    "t38_set_dual_knobs": ".t38_set_dual_knobs.config",
    "eval_gen05_set_mixer_knobs": ".eval_gen05_set_mixer_knobs.config",
    "t15_open_drawers": ".t15_open_drawers.config",
    "t01_align_blocks_in_midair": ".t01_align_blocks_in_midair.config",
    "t45_stir_pot": ".t45_stir_pot.config",
    "t32_press_soap_pump": ".t32_press_soap_pump.config",
    "t05_fill_cup": ".t05_fill_cup.config",
    "t47_trigger_drill_on_block": ".t47_trigger_drill_on_block.config",
    "t29_place_wine_glass": ".t29_place_wine_glass.config",
    "t33_reposition_marker_grasp": ".t33_reposition_marker_grasp.config",
}

# Task-specific behavior is registered here, outside the shared rollout.
_OVERRIDES = {
    "t03_cover_box": dict(camera_environment_module="core_box_env", trace_environment_module="core_box_base"),
    "t09_insert_peg": dict(camera_environment_module="core_peg_env", trace_environment_module="core_peg_base"),
    "t22_pass_screwdriver": dict(bridge="as019_bridge"),
    "t10_lift_carton": dict(force_scoring=True),
    "t11_line_up_blocks": dict(terminal_contact_flag="c12_direct_hand_contact"),
    "t26_place_fruit_in_bucket": dict(goal_reader="fruit"),
    "t24_place_ball_in_bowl": dict(goal_reader="ball"),
    "t27_place_largest_fitting_ball": dict(goal_reader="fitting_cup"),
    "t28_place_tableware": dict(goal_reader="tableware"),
    "t39_set_timer": dict(goal_reader="timer"),
    "t48_unfold_fan": dict(goal_reader="fan"),
    "t16_open_laptop": dict(goal_reader="laptop"),
    "t40_slide_knife_blade": dict(goal_reader="knife"),
    "t38_set_dual_knobs": dict(goal_reader="dual_knobs"),
    "eval_comb04_place_fruit_and_photograph_logo": dict(goal_reader="fruit_logo"),
    "eval_comb02_open_laptop_and_click_mouse": dict(goal_reader="laptop_mouse"),
    "eval_gen05_set_mixer_knobs": dict(goal_reader="mixer_knobs"),
}
_NAMES = json.loads((ASSETS / "task_checkpoint_names.json").read_text())
if set(_NAMES) != set(_CONFIG_MODULES):
    raise ValueError("Task catalog and checkpoint names disagree")
TASKS = MappingProxyType({
    name: TaskSpec(name=name, config_module=module,
                   split="transfer" if name.startswith("eval_") else "core",
                   checkpoint_name=_NAMES[name],
                   **dict(dict(goal_reader="empty" if name.startswith("eval_") else "fixed"),
                          **_OVERRIDES.get(name, {})))
    for name, module in _CONFIG_MODULES.items()
})


def native_contracts():
    return json.loads((ASSETS / "native_contracts.json").read_text())
