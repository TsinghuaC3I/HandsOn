"""Generic native-Isaac preview support for HandsOn MuJoCo task scenes."""

from __future__ import annotations

from . import d20_stages as _d20_stages
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import xml.etree.ElementTree as ET

from ...tasks.eval_comb04_place_fruit_and_photograph_logo import semantics as xy3
from ...tasks.eval_comb04_place_fruit_and_photograph_logo import runtime as xy3_runtime

import isaaclab.sim as sim_utils

from ...tasks.eval_comb01_cover_pitcher_and_box import layout as xy5

from .bimanual_assembly_scene import (
    _CACHE_DIR,
    _XML_DIR,
    _prepare_robot_mjcf,
    _repair_imported_visual_materials,
    spawn_center_rig,
)
from .b320_bimanual_scene import (
    B320_SUCCESS_CONDITION_LABELS,
    B320_TASK_NAMES,
    B320_TASK_SCENES,
)
from .compat import (
    is_lab3,
    make_mjcf_file_cfg,
    mjcf_full_scene_fix_base,
)
from .bc007_scene import (
    SUCCESS_CONDITION_LABELS as BC007_SUCCESS_CONDITION_LABELS,
)
from .d_collection_scene import (
    D_SUCCESS_CONDITION_LABELS,
    D_TASK_FIRST_PHYSX_POSITION_OVERRIDES,
    D_TASK_BODY_POSITION_OVERRIDES,
    D_TASK_BODY_QUATERNION_OVERRIDES,
    D_TASK_SCENES,
)
from .d_long_horizon_scene import (
    D_LONG_SUCCESS_CONDITION_LABELS,
    D_LONG_TASK_NAMES,
    D_LONG_TASK_SCENES,
    apply_d_long_task_filter,
)
from .e_collection_scene import (
    E01_TASK_NAME,
    E_SUCCESS_CONDITION_LABELS,
    E_TASK_SCENES,
    resolve_garmentlab_asset_path,
)
from .e02_rope_knot_scene import (
    E02_SCENE_XML,
    E02_SUCCESS_CONDITION_LABELS,
    E02_TASK_NAME,
)
from .e03_untie_rope_scene import (
    E03_SCENE_XML,
    E03_SUCCESS_CONDITION_LABELS,
    E03_TASK_NAME,
)
from .e04_cats_cradle_scene import (
    E04_SCENE_XML,
    E04_SUCCESS_CONDITION_LABELS,
    E04_TASK_NAME,
)
from .e05_fold_wipe_scene import (
    E05_SCENE_XML,
    E05_SUCCESS_CONDITION_LABELS,
    E05_TASK_NAME,
)
from .loaded_tray_gate_scene import (
    SUCCESS_CONDITION_LABELS as LOADED_TRAY_SUCCESS_CONDITION_LABELS,
    TASK_SCENES as LOADED_TRAY_TASK_SCENES,
)
from .g01_deformable_asset_scene import (
    G01_3_SCENE_XML,
    G01_3_SUCCESS_CONDITION_LABELS,
    G01_5_TASK_NAME,
    G01_5_SUCCESS_CONDITION_LABELS,
    G01_CARTON_TASK_NAMES,
    G01_SCENE_XML,
    G01_SUCCESS_CONDITION_LABELS,
    G01_TASK_NAMES,
)
from .t06_fit_device_cover_scene import (
    G02_SCENE_XML,
    G02_SUCCESS_CONDITION_LABELS,
    G02_TASK_NAME,
)
from .flexible_towel_double_fold_scene import (
    FLEXIBLE_TOWEL_SCENE_XML,
    FLEXIBLE_TOWEL_SUCCESS_CONDITION_LABELS,
    FLEXIBLE_TOWEL_TASK_NAME,
)
from .lw_native_articulation_scene import (
    LW_SCENE_XML,
    LW_SUCCESS_CONDITION_LABELS,
    LW_TASK_NAMES,
)
from .ycb11_scene import (
    YCB11_SCENE_XML,
    YCB11_SUCCESS_CONDITION_LABELS,
    YCB11_TASK_NAME,
)
from .recording_cameras import (
    D09_OPERATOR_VIEWPORT_CAMERA_EYE,
    D09_OPERATOR_VIEWPORT_CAMERA_TARGET,
    OPERATOR_VIEWPORT_CAMERA_EYE,
    OPERATOR_VIEWPORT_CAMERA_TARGET,
)


_OPTIONAL_DUAL_KNOB_TASK_NAME = "synchronized_dual_knob_setting"
_YCB07_TASK_NAME = "t39_set_timer"


_PACKAGED_XML_DIR = Path(__file__).resolve().parents[1] / "envs" / "xmls"


@dataclass(frozen=True)
class IsaacTaskSpec:
    task_name: str
    scene_xml: str
    camera_eye: tuple[float, float, float] = OPERATOR_VIEWPORT_CAMERA_EYE
    camera_target: tuple[float, float, float] = OPERATOR_VIEWPORT_CAMERA_TARGET

    @property
    def source_path(self) -> Path:
        return _XML_DIR / self.scene_xml

    def source_path_for_hand(self, hand_type: str = "wuji1") -> Path:
        normalized = str(hand_type or "wuji1").strip().lower()
        if normalized == "wuji1":
            return self.source_path
        if normalized != "wuji2":
            raise ValueError(
                "Isaac task scenes support only hand_type=wuji1 or wuji2; "
                f"got {hand_type!r}."
            )
        return _XML_DIR / self.scene_xml.replace("_wuji1.xml", "_wuji2.xml")


_TASK_SCENES = {
    "z01_pipette": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "z02_bottle_cap": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "p003_bimanual_folding_ruler_shape_configuration": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "eval_comb04_place_fruit_and_photograph_logo": "arena_arm_hand_eval_comb04_place_fruit_and_photograph_logo_rokae_wuji1.xml",
    "t26_place_fruit_in_bucket": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "t25_place_cup_on_plate": "arena_arm_hand_t25_place_cup_on_plate_rokae_wuji1.xml",
    "t24_place_ball_in_bowl": "arena_arm_hand_ball_to_bowl_rokae_wuji1.xml",
    # YCB01 uses the shared YCB arena; the task-specific layout/physical-truth
    # passes below remove the unrelated objects and keep the two gate boxes
    # plus tennis ball at source scale.
    "t34_roll_ball_through_gate": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "t27_place_largest_fitting_ball": "arena_arm_hand_t27_place_largest_fitting_ball_rokae_wuji1.xml",
    "t41_sort_nested_cups": "arena_arm_hand_t41_sort_nested_cups_rokae_wuji1.xml",
    "t28_place_tableware": "arena_arm_hand_t28_place_tableware_rokae_wuji1.xml",
    "t39_set_timer": "arena_arm_hand_ycb07_timer_rokae_wuji1.xml",
    "t07_flip_sponge": "arena_arm_hand_t07_flip_sponge_rokae_wuji1.xml",
    # YCB11 imports the mug from MJCF and mounts the exact tracked sponge as a
    # native PhysX volume deformable in the preview runner.
    YCB11_TASK_NAME: YCB11_SCENE_XML,
    "t29_place_wine_glass": "arena_arm_hand_t29_place_wine_glass_rokae_wuji1.xml",
    "t19_orient_cup_upright": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "eval_gen03_orient_bottle_upright": "arena_arm_hand_eval_gen03_orient_bottle_upright_rokae_wuji1.xml",
    "t04_cover_pitcher": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "bench320_hs014_bottle_stabilized_cap_opening": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "b02_nut_bolt_assembly": "arena_arm_hand_b02_nut_bolt_assembly_rokae_wuji1.xml",
    "t09_insert_peg": "arena_arm_hand_t09_insert_peg_rokae_wuji1.xml",
    "t44_stack_blocks": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "t03_cover_box": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    # GEN2/GEN5/GEN7 ship task-owned MJCF scenes.  Keep them on the generic
    # Isaac preview/collection path so the UI does not launch a child that
    # immediately fails its task registry lookup.
    "eval_gen01_cover_lunchbox": "arena_eval_gen01_cover_lunchbox_rokae_wuji1.xml",
    "eval_gen07_stack_boxes": "arena_arm_hand_eval_gen07_stack_boxes_rokae_wuji1.xml",
    "eval_gen06_spray_shoe": "arena_arm_hand_eval_gen06_spray_shoe_rokae_wuji1.xml",
    "eval_gen04_pass_glue_stick": "arena_arm_hand_gen4_glue_stick_handoff_rokae_wuji1.xml",
    "eval_comb01_cover_pitcher_and_box": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "t40_slide_knife_blade": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "b320_towel_corner_synchronized_spread": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t16_open_laptop": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t37_scrub_bottle": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t48_unfold_fan": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t17_open_latched_door": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    # AS019 has its own exact USD asset and does not need the generic peg/tray
    # assembly scene.  That scene used to be a fallback here, but its extra
    # articulations can poison the first PhysX GPU step before the AS019 task
    # even gets a chance to render.  Use the already-stable C-tool bootstrap
    # and compose the screwdriver/zone below instead.
    "t22_pass_screwdriver": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "p006_side_release_buckle_align_press": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t08_hammer_nail": "arena_arm_hand_t08_hammer_nail_rokae_wuji1.xml",
    "t36_scoop_marbles": "arena_arm_hand_ycb_tasks_rokae_wuji1.xml",
    "t50_water_plant": "arena_arm_hand_plant_rokae_wuji1.xml",
    "t14_move_and_click_mouse": "arena_arm_hand_monitor_mouse_rokae_wuji1.xml",
    "eval_comb02_open_laptop_and_click_mouse": "arena_arm_hand_monitor_mouse_rokae_wuji1.xml",
    "t23_photograph_logo": "arena_arm_hand_bimanual_photograph_rokae_wuji1.xml",
    "c07_pitcher_pour_cup": "arena_arm_hand_c07_ycb_pour_rokae_wuji1.xml",
    "c08_tongs_ice_cubes": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "c09_staple_paper_sheets": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t02_connect_charger_and_cable": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "bench320_hl015_bimanual_cable_route_dual_plug": "arena_arm_hand_hl015_cable_route_rokae_wuji1.xml",
    "t46_store_charger": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "c11_xylophone_sequence": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t11_line_up_blocks": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "t35_rotate_propeller": "arena_arm_hand_t35_rotate_propeller_rokae_wuji1.xml",
    "t13_load_cans": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "p004_hold_spring_clamp_open_insert_card": "arena_arm_hand_p004_hold_spring_clamp_open_insert_card_rokae_wuji1.xml",

    "t47_trigger_drill_on_block": "arena_arm_hand_ycb05_drill_point_trigger_rokae_wuji1.xml",

    "t33_reposition_marker_grasp": "arena_arm_hand_t33_reposition_marker_grasp_rokae_wuji1.xml",
    "water_plant": "arena_arm_hand_plant_rokae_wuji1.xml",
    "hammer_nail": "arena_arm_hand_hammer_nail_rokae_wuji1.xml",
    "fold_glasses": "arena_arm_hand_glass_v2_rokae_wuji1.xml",
    "pinch_tongs": "arena_arm_hand_table_tongs_rokae_wuji1.xml",
    "click_mouse": "arena_arm_hand_monitor_mouse_rokae_wuji1.xml",
    "pick_bucket": "arena_arm_hand_bucket_pick_rokae_wuji1.xml",
    "bimanual_hanoi": "arena_arm_hand_bimanual_hanoi_rokae_wuji1.xml",
    "bimanual_assembly": "arena_arm_hand_bimanual_assembly_rokae_wuji1.xml",
    "bimanual_microwave_cook": "arena_arm_hand_microwave_cook_rokae_wuji1.xml",
    "bimanual_photograph": "arena_arm_hand_bimanual_photograph_rokae_wuji1.xml",
    "bimanual_unlock_ipad": "arena_arm_hand_ipad_rokae_wuji1.xml",
    "user_make_pizza_long_horizon": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml",
    "p005_key_insert_turn_unlock_open_shackle":
        "arena_arm_hand_p005_key_unlock_padlock_rokae_wuji1.xml",
    **D_TASK_SCENES,
    **D_LONG_TASK_SCENES,
    **LOADED_TRAY_TASK_SCENES,
    **E_TASK_SCENES,
    "bench320_bc007_bimanual_long_rod_yaw_reorientation":
        "arena_arm_hand_bc007_long_rod_rokae_wuji1.xml",
    E02_TASK_NAME: E02_SCENE_XML,
    E03_TASK_NAME: E03_SCENE_XML,
    E04_TASK_NAME: E04_SCENE_XML,
    E05_TASK_NAME: E05_SCENE_XML,
    **{
        task_name: (
            G01_3_SCENE_XML
            if task_name in G01_CARTON_TASK_NAMES
            else G01_SCENE_XML
        )
        for task_name in G01_TASK_NAMES
    },
    G02_TASK_NAME: G02_SCENE_XML,
    FLEXIBLE_TOWEL_TASK_NAME: FLEXIBLE_TOWEL_SCENE_XML,
    **{task_name: LW_SCENE_XML for task_name in LW_TASK_NAMES},
    **B320_TASK_SCENES,
}

TASK_SPECS = {
    task_name: IsaacTaskSpec(task_name=task_name, scene_xml=scene_xml)
    for task_name, scene_xml in _TASK_SCENES.items()
}
TASK_SPECS["t31_prepare_coffee"] = IsaacTaskSpec(
    task_name="t31_prepare_coffee",
    scene_xml=_TASK_SCENES["t31_prepare_coffee"],
    camera_eye=D09_OPERATOR_VIEWPORT_CAMERA_EYE,
    camera_target=D09_OPERATOR_VIEWPORT_CAMERA_TARGET,
)
ISAACLAB_TASKS = frozenset({*TASK_SPECS, _OPTIONAL_DUAL_KNOB_TASK_NAME})
# These formal collection scenes spawn their controlled robot pair separately.
# Re-importing the source MJCF's second pair makes PhysX allocate GPU/pinned
# state for inactive articulations during the first reset.  Keep the complete
# D06--D19 family on the task-only path; leaving a long-horizon scene's pair in
# the MJCF and disabling it only after USD import can corrupt the first GPU
# broadphase (and makes the first-PhysX cache differ from the collection path).
COLLECTION_TASK_ONLY_ROBOT_TASKS = frozenset(
    {
        "eval_comb02_open_laptop_and_click_mouse",
        # C10 also creates its canonical arm pair in the collection env; do
        # not import a second pair just to disable it after USD conversion.
        "t02_connect_charger_and_cable",
        "bench320_hl015_bimanual_cable_route_dual_plug",
        "t12_load_and_start_microwave",
        "t17_open_latched_door",
        "p006_side_release_buckle_align_press",
        "p003_bimanual_folding_ruler_shape_configuration",
        # BC005's collection environment owns the only controlled ROKAE/Wuji
        # pair and composes the selected tray/frame beneath converter bodies.
        # Keeping the source MJCF robot pair would also route collection
        # through the generic visual-preview asset composer, removing the
        # placeholder scopes that the production composition audit replaces.
        "loaded_tray_through_narrow_gate",
        "user_make_pizza_long_horizon",
        *D_LONG_TASK_NAMES,
        *G01_TASK_NAMES,
        G02_TASK_NAME,
        FLEXIBLE_TOWEL_TASK_NAME,
        *LW_TASK_NAMES,
        *B320_TASK_NAMES,
    }
)
# Bump the scene cache namespace whenever task MJCF geometry or collision
# topology changes.  Reusing the previous USD here can keep stale prim paths
# (for example the D09 button body/geom name collision) and make Isaac crash
# before the first reset even though the source XML is valid.
# D12's moving drawer, transported knife, raised board grip, and randomized
# reset geometry must not reuse USDs generated from the old task-only MJCF.
# v192 invalidated v191 USDs for a partial asset import that has since been
# removed.  v193 adds D08's slot guides and articulated high-low-high toast
# rack.
# v194 replaces that nested rack articulation with collision-gated fixed high
# supports because Isaac 5.1's GPU solver crashes on the imported extra link.
# v195 limits rack toggling to importer-authored collider children; applying a
# CollisionAPI to their parent Xforms leaves Fabric with invalid shape indices.
# v196 mounts D08's visible rack pads on the existing lever link so the rack
# physically follows its stable high-low-high motion without another joint.
# v197 raises the D08 rack and guide tops so popped toast remains upright and
# exposes enough of each slice for a human operator to grasp it safely.
# v198 keeps only the D08 rack visuals on the lever link and uses fixed high
# and low collision stops, avoiding Isaac 5.1 moving-shape CUDA failures.
# v199 replaces the ineffective fixed stops with one lever-linked invisible
# shelf and removes the scripted bread impulse so the rack carries the toast.
# v200 places D08's released lever at the top of its travel while preserving
# the existing high and low toast-rack support heights.
# v201 replaces D09's primitive cabinet, coffee machine, mug, and tray with
# licensed appearance meshes and collision geometry derived for those assets.
# v214 gives D08's lever-linked rack a preloaded physical return and removes
# the former pose/velocity pop path from the collection runtime.
# The converter's own ``.asset_hash`` describes only its config, so an old USD
# can otherwise survive even when source mesh contents materially change.
# v235 adds the strip-mounted HL015 route gates/cover target and the dedicated
# HL015 cable-route scene include.  Invalidate prior C-tool USDs so the new
# visual markers cannot be masked by a stale converter cache.
# v236 moves the HL015 cover out of the cable-tip reset envelope.  The source
# XML change must invalidate both the prepared MJCF and converted USD so a
# seeded reset cannot reuse the old overlapping cover pose.
# v238 gave the HL015 cover a task-specific staging lane; the generic
# camera-centered x clamp otherwise folded that pose back into the cable/plug
# envelope for some randomized group-yaw seeds.
# v239 authors the deterministic midpoint of that lane in both the standalone
# HL015 scene and the shared C-tool include.  Invalidate converted USD/MJCF
# artifacts so a preview cannot briefly restore the old overlapping source pose
# before the seeded reset sampler runs.
_TASK_MJCF_CACHE_VERSION = 277
# YCB02 changed its imported cup/ball topology.  The shared converter cache
# does not hash included MJCF contents, so keep the four-ball C/G/J scene out
# of caches produced for the old G/I/J and five-ball layouts.  v2 replaces
# C-cup Collider27's PhysX-solid convex hull with a genuinely hollow ring.
_YCB02_FOUR_BALL_ROWS_USD_CACHE_REVISION = "cgj_four_ball_rows_v2_hollow_c_rim"
_YCB02_RIM_SEGMENT_COUNT = 12
_YCB02_C_RIM_SOURCE_MASS = 0.006124959294401715
_YCB02_C_CUP_COLLISION_SPEC = {
    "inner_radius": 0.02640,
    "outer_radius": 0.03245,
    "center_z": 0.0301575,
    # Exact values from the source-compiled YCB02 c_cups body.  Replacing one
    # collision mesh must not change the cup's original 17 g dynamics.
    "mass": 0.01699999037744309,
    "center_of_mass": (
        -0.00011543438601543666,
        -0.00009401538968950294,
        0.009121513216168709,
    ),
    "principal_axes": (
        0.9182948365478597,
        -0.0012165524787352592,
        0.0034572918885640066,
        0.3958802347963396,
    ),
    "diagonal_inertia": (
        1.3219285251596239e-5,
        1.3093159344158062e-5,
        1.028734060100456e-5,
    ),
}
# YCB04 composes its three dedicated cup bodies on top of the shared YCB
# bootstrap.  Isolate the filtered three-cup USD from older caches that still
# contain every unrelated shared YCB prop.  The source Collider27 mesh is an
# annulus, but PhysX imports its convex hull as a solid disk across the cup
# mouth.  Hollow analytic rings keep the nested cups out of that impossible
# collision volume; v3 also seats them on the resulting physical support.
_YCB04_ISOLATED_CUPS_USD_CACHE_REVISION = "isolated_hollow_rims_seated_v3"
# Root-height offsets from the large cup, for the prepared hollow geometry.
# The source medium cup is 20.18 mm above the large cup and falls about 12 mm
# after the filled-mouth hull is removed.  Cross-section support is at about
# 8.4 mm (large/medium) and 6.4 mm (medium/small); start each pair 0.1 mm
# above that estimate. Preserve the large cup's table support and let PhysX
# form contacts.
_YCB04_NESTED_ROOT_Z_OFFSETS_M = {"medium": 0.0085, "small": 0.0150}
_YCB04_RIM_SEGMENT_COUNT = 12
_YCB04_SOURCE_RIM_MASS = 0.0022222222
_YCB04_CUP_COLLISION_SPECS = {
    "large": {
        "inner_radius": 0.02640,
        "outer_radius": 0.03245,
        "center_z": 0.0301575,
        "mass": 0.0599999994,
        "center_of_mass": (-0.000159325995, -0.000133346523, -0.011766480073),
        "principal_axes": (
            0.918255750458,
            0.000179588618,
            -0.000771286593,
            0.395987057383,
        ),
        "diagonal_inertia": (
            4.1647246441e-5,
            4.1058790678e-5,
            3.3224446236e-5,
        ),
    },
    "medium": {
        "inner_radius": 0.02440,
        "outer_radius": 0.02995,
        "center_z": 0.029215,
        "mass": 0.0599999994,
        "center_of_mass": (-0.000146789558, -0.000122633357, -0.011682128384),
        "principal_axes": (
            0.918264975298,
            0.000083984753,
            -0.000627598609,
            0.395965950819,
        ),
        "diagonal_inertia": (
            3.7030486108e-5,
            3.6531924630e-5,
            2.8108354485e-5,
        ),
    },
    "small": {
        "inner_radius": 0.02235,
        "outer_radius": 0.02745,
        "center_z": 0.0282715,
        "mass": 0.0599999994,
        "center_of_mass": (0.000135670655, 0.000110843873, -0.011304766343),
        "principal_axes": (
            0.920333712866,
            -0.000063750898,
            0.000525214086,
            0.391133707379,
        ),
        "diagonal_inertia": (
            3.3298207697e-5,
            3.2879111280e-5,
            2.3614132574e-5,
        ),
    },
}
# The YCB07 source is an included MJCF model whose wrapper is promoted to a
# free articulation below. Keep its converted USD in a task-local namespace
# so an interrupted/old cache cannot silently restore the pre-promotion tree.
_YCB07_TIMER_USD_CACHE_REVISION = "timer_source_v2"
_TASK_USD_CONTENT_SCHEMA = b"handson-b03-task-usd-content-v1"
_TASK_USD_MIN_ROOT_BYTES = 768
_TASK_USD_MIN_BASE_BYTES = 4096
_MOCAP_TARGET_HELPER_GEOMS = frozenset({"target", "target_right", "target_left"})
_B04_WOOD_BLOCK_VISUAL_COLOR = (0.58, 0.39, 0.20)
_B04_WOOD_BLOCK_TEXTURE = (
    _XML_DIR
    / "lightwheel_ycb"
    / "wood_block"
    / "visuals"
    / "T_wood_block_BC001.png"
)
_B04_SLEEP_THRESHOLD = 0.005
_B04_SMALL_BLOCK_HALF_EXTENTS = {
    "BuildingBlock017": 0.014925,
    "BuildingBlock018": 0.015,
    "BuildingBlock019": 0.015,
}
_BC004_D08_ALIGNED_USD_CACHE_REVISION = "d08_aligned_v1"
_P004_REAL_YCB_CLAMP_USD_CACHE_REVISION = (
    "real_ycb_extra_large_clamp_card_support_v2"
)
_P005_BOTTOM_KEYWAY_FREE_PADLOCK_USD_CACHE_REVISION = (
    # v5 invalidates the v4 converter output after the baked 1.2x colliders
    # and reset-pose contract were updated.  Isaac's MJCF converter does not
    # include referenced OBJ bytes in its cache key, so reusing v4 can leave a
    # physically different key/padlock scene behind the authored reset.
    "scaled_baked_collision_lock_key_ycb1x_v5"
)
# BC005 now composes its production tray and selected wooden-frame USD below
# the converter-owned rigid bodies.  Keep that topology and the selected-asset
# layout out of caches produced by either placeholder-only predecessor.
_BC005_FORWARD_LAYOUT_USD_CACHE_REVISION = "forward_assets_v7"
_D09_CABINET_USD_CACHE_REVISION = "dual_door_model_matched_bay_v7"
_A01_ANALYTIC_FRUIT_COLLISIONS = {
    # The six scanned banana hulls share an almost identical lower support
    # plane.  GPU PhysX can alternate the active manifold across those seams,
    # which leaves a persistent correction torque even after translation has
    # settled.  One flat-bottomed primitive uses the exact compound bounds and
    # mass, so reset support remains unchanged and contact has one manifold.
    "011_banana": {
        "type": "box",
        "size": "0.0333585 0.0973900 0.019722",
        "mass": "0.07",
    },
    "apple": {"type": "sphere", "size": "0.038", "mass": "0.10"},
    "orange": {"type": "sphere", "size": "0.037", "mass": "0.11"},
    # Keep the pear's existing footprint and support height, but give its
    # flattened base a non-axisymmetric contact manifold.  A cylindrical base
    # can continuously rotate its active edge on GPU PhysX and never quite
    # satisfy the strict angular-velocity recording gate.
    "pear": {"type": "box", "size": "0.028 0.028 0.052", "mass": "0.11"},
}
_A01_ANALYTIC_FRUIT_INERTIALS = {
    # Exact MuJoCo-compiled rigid-body inertials for every A01 free body whose
    # per-geom mass is not preserved by the Isaac MJCF importer.  The importer
    # otherwise emits sentinels and lets PhysX infer density-dependent values.
    "011_banana": {
        "pos": "0 0 0",
        "quat": "1 0 0 0",
        "mass": "0.07",
        "diaginertia": (
            "0.00023038795229333336 0.0000350407588125 "
            "0.0002472773711858334"
        ),
    },
    "apple": {
        "pos": "0 0 0",
        "quat": "1 0 0 0",
        "mass": "0.10",
        "diaginertia": "0.00005776 0.00005776 0.00005776",
    },
    "orange": {
        "pos": "0 0 0",
        "quat": "1 0 0 0",
        "mass": "0.11",
        "diaginertia": "0.000060236 0.000060236 0.000060236",
    },
    "pear": {
        "pos": "0 0 0",
        "quat": "1 0 0 0",
        "mass": "0.11",
        "diaginertia": (
            "0.0001278933333333333 0.0001278933333333333 "
            "0.00005749333333333334"
        ),
    },
    # The two bucket side colliders each declare mass=0.05 in the source.  The
    # importer drops those opinions and infers 0.09216 kg from density instead,
    # changing the root mass by exactly 0.08432 kg unless the compiled source
    # inertial is authored explicitly here and again on the runtime USD prim.
    "bucket": {
        "pos": (
            "-0.0091849759261078915 -0.0026685663751596701 "
            "-0.0046551046477350906"
        ),
        "quat": (
            "0.50092912967801029 0.50074393305003673 "
            "0.49932365587470223 0.49900040805347629"
        ),
        "mass": "1.6283199999999998",
        "diaginertia": (
            "0.043445027619892711 0.029461449518300258 "
            "0.025951661242985445"
        ),
    },
}
_B01_ISAAC_LID_INSERT_RADIUS = 0.043
_B01_ISAAC_LID_INSERT_HALF_HEIGHT = 0.010
_B01_ISAAC_LID_INSERT_CENTER_Z = -0.0135045
_C04_PLANT_TEXTURES = frozenset({"image0", "image1", "image2", "image3"})
_C04_PLANT_ALPHA_TEXTURES = frozenset({"image0", "image1"})
_C04_PLANT_ALPHA_MATERIALS = frozenset({"blinn1SG", "blinn2SG"})
_C04_PLANT_ALPHA_TEXTURE_SUFFIX = "_isaac_alpha.png"
_C04_PLANT_MODEL_MATERIALS = {
    "model_0": "blinn2SG",
    "model_1": "blinn1SG",
    "model_2": "blinn4SG",
    "model_3": "blinn3SG",
}
_C06_CAMERA_DEBUG_GEOMS = frozenset(
    {"camera_axis_x", "camera_axis_y", "camera_axis_z"}
)
_C06_ISAAC_ONLY_REMOVED_CAMERA_GEOMS = _C06_CAMERA_DEBUG_GEOMS | frozenset(
    {"camera_live_screen"}
)
_C04_SPRAY_BODY_INERTIALS = {
    # Values compiled by MuJoCo from spray.xml.  Author them explicitly because
    # the Isaac MJCF importer otherwise leaves these links with zero/negative
    # mass and the bottle pitches forward while resting on the table.
    "link_2": {
        "pos": "0 -0.04 -0.008",
        "quat": "0.707106781186548 0.707106781186548 0 0",
        "mass": "0.106864",
        "diaginertia": "0.0002484944213333334 0.0002484944213333334 0.000041035776",
    },
    "cone_vis": {
        "pos": "0 0 0",
        "mass": "1e-6",
        "diaginertia": "1e-12 1e-12 1e-12",
    },
    "link_1_helper": {
        "pos": "0 0 0",
        "mass": "1e-6",
        "diaginertia": "1e-12 1e-12 1e-12",
    },
    "link_1": {
        "pos": "0.0005636160335810302 0.06839649818902302 0.023815934615581607",
        "quat": "0.9850915965781937 -0.1717202968940305 0.002757215612013255 -0.009954081962701622",
        "mass": "0.012",
        "diaginertia": "0.00001835580195682238 0.000015192044174056295 0.000004678091126283619",
    },
    "link_0": {
        "pos": "-0.003539504703114288 0.005979149596654326 0.030218858140560537",
        "quat": "0.7458932352184467 0.6465968232047076 0.10495125804734028 -0.12058633136148196",
        "mass": "0.0128",
        "diaginertia": "0.000007094904046617121 0.00000708687187485088 0.0000005629653410813747",
    },
}

_SUCCESS_CONDITION_LABELS = {
    "p003_bimanual_folding_ruler_shape_configuration": (
        "Endpoint roles acquired: right S0, left S2",
        "Ruler remains supported by tabletop; no airborne shaping",
        "Two joint angles match the sampled target",
        "All three segment directions match the sampled target",
        "All three segments table-supported and centered",
        "Both hands released",
        "Ruler low velocity for 30 control frames",
        "Both hands clear of table on final success frame",
        "Ordered tabletop sequence complete",
    ),
    "t26_place_fruit_in_bucket": (
        "Right hand touching bucket",
    ),
    "t25_place_cup_on_plate": (
        "Target object is the brown mug",
        "Mug centered on tray",
        "Mug bottom at tray height",
        "Mug upright",
        "Mug stable and not dropped",
    ),
    "t24_place_ball_in_bowl": (
        "Ball inside bowl opening",
        "Ball height inside bowl",
    ),
    "t29_place_wine_glass": (
        'Upright wine glass released and stable at the plate center',
    ),
    "t34_roll_ball_through_gate": (
        "Ball centre passed the far Y face",
    ),
    "t27_place_largest_fitting_ball": (
        "Correct largest-fitting ball was touched and lifted",
        "Correct ball crossed the cup rim from above",
        "Correct ball entered and is physically supported by the cup",
        "Other three candidate balls remain outside the cup",
        "Cup remains upright and supported by the table",
        "Both hands are clear of the ball and cup",
        "Correct ball remains stable after release",
        "Cup linear/angular motion remains stable",
        "No multiple balls are inside the cup",
    ),
    "t41_sort_nested_cups": (
        "All three cups began as one stable nested stack",
        "Small cup was physically extracted from medium cup",
        "Medium cup was physically extracted from large cup",
        "All cups are upright, ordered large-to-small, and visibly separated",
        "Both hands released and stayed clear during final stability",
    ),
    "t28_place_tableware": (
        "Prompted tableware was physically lifted",
        "Target is in the prompted semantic slot and orientation",
        "Reference plate and non-target objects remain undisturbed",
        "Target is released on the table and stable for 0.5 seconds",
    ),
    "t39_set_timer": (
        "Exact prompted minutes and seconds programmed",
        "Every button press has right-fingertip contact and left-hand brace",
        "START/STOP pressed and released last",
        "Countdown display decremented at least once",
        "Both hands withdrawn and timer stable on table for 0.5 sec",
    ),
    "t07_flip_sponge": (
        "Right hand grasped and moved the spatula",
        "Blade contacted and reached beneath the sponge",
        "Sponge completed a valid flip without excessive skillet tilt",
        "Sponge is contained and supported by the skillet after release",
        "Neither hand directly touched the sponge",
    ),
    YCB11_TASK_NAME: YCB11_SUCCESS_CONDITION_LABELS,
    "eval_gen03_orient_bottle_upright": (
        "Bottle upright (tilt at most 25 degrees)",
        "Bottle bottom supported by the table",
        "Hand released from the bottle",
    ),
    "t19_orient_cup_upright": (
        "Cup opening faces up",
        "Cup at table height",
        "Hand clear of cup",
    ),
    "t04_cover_pitcher": (
        "Lid plug fully inside cup mouth",
        "Lid height fully seated",
        "Lid top is level",
        "Lid stable and hand clear 0.5 sec",
    ),
    "b02_nut_bolt_assembly": (
        "Nut centered on bolt",
        "Nut near target depth",
        "Nut axis aligned to bolt",
        "Thread engaged with twist progress",
        "Nut stable",
    ),
    "t09_insert_peg": (
        "Peg centered over socket",
        "Peg inserted into socket",
        "Peg axis aligned to socket",
        "Peg stable",
    ),
    "t44_stack_blocks": (
        "Four block layers formed",
        "Large block at bottom",
        "Block layers aligned in XY",
        "Layer heights are valid",
        "Hand clear of blocks",
    ),
    xy5.TASK_NAME: (
        "Pitcher lid seated and released first",
        "Box lid moved and seated afterwards",
        "Both lids remain seated, stable and released",
    ),
    "t03_cover_box": (
        "Box lid centered on box",
        "Box lid near seated height",
        "Box lid top is level",
        "Box lid yaw aligned",
        "Box lid stable",
    ),
    "t16_open_laptop": (
        "Laptop base is fixed at its randomized reset pose",
        "Right hand physically contacts the display front edge",
        "Display hinge reaches the round target angle",
        "Display remains stable for 15 control steps",
    ),
    "t37_scrub_bottle": (
        "Left hand physically stabilizes bottle",
        "Right hand physically holds scrub brush",
        "Lower, middle, upper bands covered in order",
        "Bristles maintain physical bottle contact",
        "Both objects released",
        "Bottle upright and stable for 0.5 seconds",
    ),
    "t48_unfold_fan": (
        "Left hand physically contacts the left outermost fan rib",
        "Right hand physically contacts the right outermost fan rib",
        "Physical joint-sum angle reaches 60, 90, or 120 degrees",
        "Left/right opening remains balanced",
        "Fan remains supported and stable for 15 control steps",
    ),
    "t17_open_latched_door": (
        "Right hand physically contacts the long Helindu near-side Handle002",
        "Handle002 stays at or above 25 degrees for five control steps",
        "Door hinge physically unlocks while the right hand holds Handle002",
        "Left hand physically braces and pushes the door panel after unlock",
        "Door reaches at least 45 degrees and settles for 15 steps",
        "No far-side Handle001 contact; protected handle penetration <= 5 mm",
    ),
    "p006_side_release_buckle_align_press": (
        "Right hand physically holds the blue female buckle",
        "Left hand physically holds the orange male buckle",
        "Buckle axes align within 15 mm and 12 degrees",
        "Insertion reaches at least 18 mm with buckle-body contact",
        "Pressed latch persists for five control steps",
        "Relative motion settles for twelve steps; no pull test",
    ),
    "t08_hammer_nail": (
        "Hammer has hit nail",
        "Nail driven deep enough",
        "Wood lifted by nail",
    ),
    "t36_scoop_marbles": (
        "Spoon picked up and loaded",
        "Loaded spoon over bowl",
        "Marble dropped into bowl",
        "Left-hand contact is allowed while the marble stays supported",
        "Right hand never directly contacts the marble",
    ),
    "t50_water_plant": (
        "Spray bottle held by hand",
        "Trigger pressed by hand",
        "Water hits plant or soil",
        "Hold valid spray for 30 steps",
    ),
    "t14_move_and_click_mouse": (
        "Mouse moved from its start position",
        "Mouse centered on the mouse pad",
        "No button pressed while moving",
        "Left mouse button clicked on the pad",
    ),
    xy3.TASK_NAME: ("Specified fruit released and stable in bucket", "Then aim at bucket logo and press shutter"),
    "t23_photograph_logo": ("Photo event triggered",),
    "c07_pitcher_pour_cup": (
        "At least 7 of 8 proxy particles in cup",
        "At most one spilled particle",
        "Pouring mug tilted then returned upright",
        "Pouring mug placed in return zone",
    ),
    "c08_tongs_ice_cubes": (
        "All four ice cubes in the glass",
        "At least four tongs closing cycles",
        "Glass remains upright",
    ),
    "c09_staple_paper_sheets": (
        "Stapler top opened before pressing",
        "Two sheets overlap and enter staple zone",
        "Stapler pressed past threshold",
    ),
    "t02_connect_charger_and_cable": (
        "Charger plug inserted into power strip",
        "Cable connector inserted into charger head",
        "Connections completed in order",
    ),
    "bench320_hl015_bimanual_cable_route_dual_plug": (
        "Cable passes the first cyan route gate",
        "Cable passes the second cyan route gate in order",
        "Charger plug inserted into power strip first",
        "Cable connector inserted into charger head second",
        "Both hands have participated in the manipulation",
        "Cover seated on the green target",
        "Both hands open and clear",
        "Final assembly remains stable for 0.5 s",
    ),
    "t46_store_charger": (
        "Left hand physically holds the lower drawer open",
        "Charger is fully inside the drawer cavity",
        "Right hand releases the charger after placement",
        "Left hand releases before the 0.5 s stability timer starts",
    ),
    "t40_slide_knife_blade": (
        "Right hand picks up the utility knife by at least 2 cm",
        "Left fingertips touch the slider after pickup",
        "Per-episode target is randomized: retract to bottom or extend to top",
        "Slider starts away from its target at a randomized position",
        "Knife XY position and full tabletop yaw are randomized",
        "Success is immediate at the requested endpoint; no final hold",
    ),
    "t22_pass_screwdriver": (
        "Right hand contacts the screwdriver first",
        "Left hand contacts the airborne screwdriver afterward",
        "Screwdriver lands inside the target zone",
    ),
    "c11_xylophone_sequence": (
        "Randomized 3-5 key sequence completed",
        "Wrong hit resets sequence",
        "Mallet returned to rest zone",
    ),
    "t11_line_up_blocks": (
        "Three block centers on baseline",
        "Block headings aligned",
        "Adjacent spacing in range",
        "Set square used without direct hand grasp",
    ),
    "t35_rotate_propeller": (
        "Propeller absolute angular travel reaches 1080 degrees (three turns)",
    ),
    "t13_load_cans": (
        "Left can stable in opposite negative-Y partition",
        "Right can stable in opposite positive-Y partition",
        "Both cans released within synchronization window",
    ),
    "p004_hold_spring_clamp_open_insert_card": (
        "Left hand picks up and holds the clamp",
        "Right hand inserts the card while left hand holds the clamp",
        "Left hand holds the clamp as it closes onto the card",
        "Right hand releases the card",
        "Left hand holds the clamped card stable for 0.5 s",
    ),
    "p005_key_insert_turn_unlock_open_shackle": (
        "Left hand lifts the padlock at least 35 mm while right hand inserts the key 12 mm",
        "Both objects remain held while the key turns clockwise 55 degrees",
        "Unlocked U-shaped shackle pops up automatically",
        "Shackle opens at least 16 mm and becomes stable",
    ),
    "t47_trigger_drill_on_block": (
        "Drill grasped by handle and lifted",
        "Drill aimed at block without contact",
        "Index finger physically contacts trigger",
        "Trigger pressed and held while aimed",
        "Trigger released while drill remains held",
    ),
    "t33_reposition_marker_grasp": (
        'Record a fresh thumb/index grasp point after each pickup',
        'Both contacting fingers reach 2.5 cm toward the writing tip from that grasp point',
        'Table drops are retryable; pick up again to record a new reference',
    ),
    **D_SUCCESS_CONDITION_LABELS,
    **D_LONG_SUCCESS_CONDITION_LABELS,
    **LOADED_TRAY_SUCCESS_CONDITION_LABELS,
    **E_SUCCESS_CONDITION_LABELS,
    **BC007_SUCCESS_CONDITION_LABELS,
    E02_TASK_NAME: E02_SUCCESS_CONDITION_LABELS,
    E03_TASK_NAME: E03_SUCCESS_CONDITION_LABELS,
    E04_TASK_NAME: E04_SUCCESS_CONDITION_LABELS,
    E05_TASK_NAME: E05_SUCCESS_CONDITION_LABELS,
    **{
        task_name: (
            G01_5_SUCCESS_CONDITION_LABELS
            if task_name == G01_5_TASK_NAME
            else G01_3_SUCCESS_CONDITION_LABELS
            if task_name in G01_CARTON_TASK_NAMES
            else G01_SUCCESS_CONDITION_LABELS
        )
        for task_name in G01_TASK_NAMES
    },
    G02_TASK_NAME: G02_SUCCESS_CONDITION_LABELS,
    FLEXIBLE_TOWEL_TASK_NAME: FLEXIBLE_TOWEL_SUCCESS_CONDITION_LABELS,
    **LW_SUCCESS_CONDITION_LABELS,
    **B320_SUCCESS_CONDITION_LABELS,
}


def _ensure_optional_task_metadata(task_name: str) -> None:
    """Load isolated optional task metadata only when that task is requested."""

    if task_name != _OPTIONAL_DUAL_KNOB_TASK_NAME or task_name in TASK_SPECS:
        return
    from . import dual_knob_panel as optional_panel

    if optional_panel.DUAL_KNOB_TASK_NAME != _OPTIONAL_DUAL_KNOB_TASK_NAME:
        raise RuntimeError("Optional dual-knob task identity changed unexpectedly")
    scene_xml = optional_panel.DUAL_KNOB_SCENE_XML
    _TASK_SCENES[task_name] = scene_xml
    TASK_SPECS[task_name] = IsaacTaskSpec(
        task_name=task_name,
        scene_xml=scene_xml,
    )
    _SUCCESS_CONDITION_LABELS[task_name] = (
        optional_panel.DUAL_KNOB_SUCCESS_CONDITION_LABELS
    )

_YCB_TABLE_TOP_Z = 0.89
# A pure +90 degree X rotation supports A06 only on the wide cup-mouth rim;
# the center of mass projects well beyond that one-ended contact line, so the
# cup must tip and can keep switching between neighboring convex contacts.
# This pose aligns the nearest three-point convex-hull face (two bottom-side
# points plus one mouth point) with the table.  The cup still begins visibly on
# its side and remains freely wakeable, but gravity has no reset torque to turn
# into a persistent spin.
_A06_SIDE_REST_QUAT_WXYZ = (
    0.741820442840,
    0.670510708086,
    0.007278510600,
    -0.008052590199,
)
# The aligned support plane is 0.0246224 m below the body origin.  Start only
# 0.5 mm above the table to avoid a reset impact while leaving PhysX, rather
# than an authored penetration, to establish contact.
_A06_SIDE_REST_Z_OFFSET = 0.025122404
_YCB_OBJECT_Z_OFFSETS = {
    "011_banana": 0.020,
    "003_cracker_box": 0.106,
    "004_sugar_box": 0.089,
    "008_pudding_box": 0.046,
    "025_mug": 0.043,
    "024_bowl": 0.001,
    "apple": 0.038,
    "orange": 0.038,
    "pear": 0.052,
    "a_cup": 0.031,
    "tennis_ball": 0.033,
    "golf_ball": 0.022,
    "mini_soccer_ball": 0.040,
    "spoon": 0.041,
    "a_marbles": 0.044,
    "rubiks_cube": 0.030,
    "019_pitcher_base": 0.118,
    "pitcher_lid": 0.024,
    "Clear_Box": 0.0745,
    "Box_lid": 0.010,
    "036_wood_block": 0.042689,
    "BuildingBlock017": 0.014925,
    "BuildingBlock018": 0.015,
    "BuildingBlock019": 0.015,
    "bucket": 0.140,
    "tray": 0.020,
}
_YCB_TASK_LAYOUTS = {
    xy3.TASK_NAME: {name: xy for name, xy in xy3.DEFAULT_XY.items() if name != "camera"},
    xy5.TASK_NAME: xy5.CENTERS,
    "t26_place_fruit_in_bucket": {
        # A01 is operated by the right arm while the inactive left hand is
        # raised. Keep every baseline center inside the recording workspace and
        # away from both reset hands; runtime reset adds wider per-object jitter.
        "bucket": (-0.35, 0.08),
        "011_banana": (-0.37, -0.265),
        "apple": (-0.285, -0.265),
        "orange": (-0.37, -0.14),
        "pear": (-0.285, -0.08),
    },
    # The current MuJoCo ROKAE backend implements A06 with the cup asset.
    "t19_orient_cup_upright": {"a_cup": (-0.30, -0.06)},
    "t04_cover_pitcher": {
        "019_pitcher_base": (-0.30, 0.10),
        "pitcher_lid": (-0.34, -0.20),
    },
    # HS014 composes its own exact bottle and cap; shared YCB objects must be
    # absent before USD conversion so first PhysX starts from an isolated scene.
    "bench320_hs014_bottle_stabilized_cap_opening": {},
    "t44_stack_blocks": {
        "036_wood_block": (-0.30, 0.16),
        "BuildingBlock017": (-0.38, -0.24),
        "BuildingBlock018": (-0.30, -0.25),
        "BuildingBlock019": (-0.22, -0.20),
    },
    "t03_cover_box": {
        # Keep both centers inside the recording workspace; runtime reset
        # spreads the lid/box pair further without relying on clamp.
        "Clear_Box": (-0.20, 0.15),
        "Box_lid": (-0.36, 0.15),
    },
    "t36_scoop_marbles": {
        "024_bowl": (-0.24, 0.18),
        "spoon": (-0.36, -0.12),
        "golf_ball": (-0.34, 0.02),
    },
    "c07_pitcher_pour_cup": {
        "019_pitcher_base": (-0.38, -0.26),
        "025_mug": (-0.20, -0.11),
    },
    "t34_roll_ball_through_gate": {
        # Source-scale YCB objects.  Runtime reset keeps the two boxes at this
        # centre pose and randomizes only the ball across the tabletop while
        # preserving the 130 mm clear opening and source dimensions.
        "004_sugar_box": (-0.3845, 0.055),
        "003_cracker_box": (-0.1301, 0.055),
        "tennis_ball": (-0.2750, -0.180),
    },
}
_YCB_SHARED_OBJECTS = frozenset(_YCB_OBJECT_Z_OFFSETS)
_A02_INACTIVE_OBJECTS = frozenset(
    {"003_cracker_box", "004_sugar_box", "008_pudding_box", "c_cups"}
)
_YCB06_SOURCE_BODY_BY_MATERIAL = {
    "plate_material": {
        "name": "plate",
        "density": "211.49",
        "pos": "-0.270000 0.000000 0.902346",
        "quat": "1 0 0 0",
        "dynamic": True,
    },
    "fork_material": {
        "name": "fork",
        "density": "2297.38",
        "pos": "0.000000 0.029000 0.914433294",
        "quat": "0.699789207 -0.101464606 -0.101464606 0.699789207",
        "dynamic": True,
    },
    "knife_material": {
        "name": "knife",
        "density": "1380.67",
        "pos": "0.000000 0.000000 0.916010919",
        "quat": "0.702561397 -0.080046760 -0.080046760 0.702561397",
        "dynamic": True,
    },
    "spoon_material": {
        "name": "spoon",
        "density": "976.845",
        "pos": "0.000000 -0.039000 0.915408545",
        "quat": "0.702205440 -0.083111491 -0.083111491 0.702205440",
        "dynamic": True,
    },
    "mug_material": {
        "name": "mug",
        "density": "619.54",
        "pos": "-0.140000 -0.250000 0.930765",
        "quat": "1 0 0 0",
        "dynamic": True,
    },
    "gelatin_box_material": {
        "name": "gelatin_box",
        "density": "557.205",
        "pos": "0.075000 0.000000 0.904000",
        "quat": "0.5 0.5 0.5 0.5",
        "dynamic": False,
    },
}
_B02_ISAAC_NUT_ROOT_Z = 0.894074
_B02_NUT_RING_INNER_RADIUS = 0.0053
_B02_NUT_RING_OUTER_APOTHEM = 0.007596
_B02_NUT_RING_OUTER_RADIUS = 0.008771
_B02_NUT_RING_HALF_HEIGHT = 0.004074
_C06_ISAAC_CAMERA_ROOT_Z = 0.955
_TASK_BODY_POSITION_OVERRIDES = {
    "t25_place_cup_on_plate": {
        "plate": (-0.22, 0.12, 0.932),
        "025_mug": (-0.34, -0.20, 0.963765),
    },
    "t24_place_ball_in_bowl": {
        # Keep both robots present and every center inside the recording
        # workspace; runtime reset adds wider per-object randomization.
        "024_bowl": (-0.22, 0.14, 0.92),
        "tennis_ball": (-0.38, -0.24, 0.954),
        "golf_ball": (-0.30, -0.18, 0.941),
        "mini_soccer_ball": (-0.20, -0.10, 0.958),
    },
    "b02_nut_bolt_assembly": {
        "bolt_root_body": (-0.24, 0.10, 0.92),
        "nut_root_body": (-0.34, -0.20, _B02_ISAAC_NUT_ROOT_Z),
    },
    "t09_insert_peg": {
        "industreal_round_peg_8mm": (-0.30, -0.42, 0.94),
        "industreal_tray_insert_round_peg_8mm": (-0.10, 0.00, 0.92),
    },
    "t08_hammer_nail": {
        "hammer_body": (-0.12, -0.33, 1.0),
        "wood": (-0.35, 0.20, 0.9165),
    },
    "t50_water_plant": {
        "link_2": (-0.30, -0.22, 0.955),
        "plant": (-0.30, 0.14, 1.06),
    },
    "eval_comb02_open_laptop_and_click_mouse": {
        "link_3": (-0.34, -0.30, 0.892),
        "mousepad": (-0.08, -0.30, 0.89),
    },
    "t14_move_and_click_mouse": {
        "link_3": (-0.38, -0.24, 0.892),
        "mousepad": (-0.14, -0.24, 0.89),
        "display_link_1": (0.10, 0.0, 1.16),
    },
    "t11_line_up_blocks": {
        "c12_set_square": (-0.42, -0.27, 0.9000),
        "c12_block_0": (-0.39, -0.10, 0.9115),
        "c12_block_1": (-0.28, -0.17, 0.9115),
        "c12_block_2": (-0.17, -0.24, 0.9115),
    },
    **D_TASK_BODY_POSITION_OVERRIDES,
}

_C_TOOL_TASK_BODIES = {
    "t40_slide_knife_blade": set(),
    # AS019 composes its reviewed screwdriver USD and marker after the generic
    # C-tool bodies have been removed.  Keeping this set empty preserves the
    # stable robot/table bootstrap without importing unrelated task props.
    "t22_pass_screwdriver": set(),
    "t13_load_cans": set(),
    "t16_open_laptop": set(),
    # HS005 composes its authoritative USD asset layer before the first PhysX
    # reset. Keep only the shared table/robot scene from this bootstrap MJCF.
    "t37_scrub_bottle": set(),
    "t48_unfold_fan": set(),
    "t17_open_latched_door": set(),
    "p003_bimanual_folding_ruler_shape_configuration": set(),
    "p006_side_release_buckle_align_press": set(),
    "c07_pitcher_pour_cup": {
        "c07_pitcher",
        "c07_cup",
        "c07_return_zone",
        "c07_particle_0",
        "c07_particle_1",
        "c07_particle_2",
        "c07_particle_3",
        "c07_particle_4",
        "c07_particle_5",
        "c07_particle_6",
        "c07_particle_7",
    },
    "c08_tongs_ice_cubes": {
        "c08_tongs",
        "c08_tongs_cradle",
        "c08_glass_cup",
        "c08_ice_0",
        "c08_ice_1",
        "c08_ice_2",
        "c08_ice_3",
    },
    "c09_staple_paper_sheets": {
        "c09_stapler",
        "c09_paper_0",
        "c09_paper_1",
    },
    "t02_connect_charger_and_cable": {
        "c10_power_strip",
        "c10_charger_plug",
        "c10_charging_cable",
    },
    "bench320_hl015_bimanual_cable_route_dual_plug": {
        "c10_power_strip",
        "c10_charger_plug",
        "c10_charging_cable",
        "hl015_cover",
    },
    # FU006 reuses only the dual ROKAE/Wuji table and lighting from this MJCF.
    # Its exact USD assets are composed below before the first PhysX reset.
    "t46_store_charger": set(),
    "c11_xylophone_sequence": {
        "c11_xylophone",
        "c11_xylophone_frame",
        "c11_key_red",
        "c11_key_yellow",
        "c11_key_blue",
        "c11_key_green",
        "c11_mallet",
        "c11_return_zone",
    },
    "t11_line_up_blocks": {
        "c12_set_square",
        "c12_block_0",
        "c12_block_1",
        "c12_block_2",
        "c12_baseline_pad",
    },
    # Pizza mounts its exact downloaded assets after the bootstrap import;
    # keep none of the generic C-tool bodies in the standalone preview.
    "user_make_pizza_long_horizon": set(),
}
_C_TOOL_ALL_BODIES = frozenset().union(*_C_TOOL_TASK_BODIES.values())


def _apply_c_tool_task_filter(root: ET.Element, task_name: str) -> int:
    if task_name == "eval_comb02_open_laptop_and_click_mouse":
        removed = 0
        removed_refs = set()
        for worldbody in root.findall("worldbody"):
            for element in list(worldbody):
                if element.tag == "body" and element.get("name") == "display_link_1":
                    removed_refs.update(n.get("name") for n in element.iter() if n.get("name"))
                    worldbody.remove(element)
                    removed += 1
        _remove_elements_referencing(root, removed_refs)
        _prune_unused_asset_definitions(root)
        return removed
    active_names = _C_TOOL_TASK_BODIES.get(task_name)
    if active_names is None:
        return 0
    removed_names = set(_C_TOOL_ALL_BODIES - active_names)
    removed_references = set(removed_names)
    removed = 0
    for worldbody in root.findall("worldbody"):
        retained: list[ET.Element] = []
        for element in worldbody:
            if element.tag != "body" or element.attrib.get("name") not in removed_names:
                retained.append(element)
                continue
            removed += 1
            removed_references.update(
                descendant.attrib["name"]
                for descendant in element.iter()
                if descendant.tag in {"joint", "freejoint"} and "name" in descendant.attrib
            )
        worldbody[:] = retained
    _remove_elements_referencing(root, removed_references)
    return removed


def _configure_ycb07_timer_source(root: ET.Element, task_name: str) -> int:
    """Promote the authoritative YCB timer include to one free articulation.

    ``lightwheel_ycb/timer/model.xml`` intentionally keeps the exporter
    wrapper (an unnamed body containing ``object``).  Isaac's MJCF importer
    treats that wrapper as a separate rigid body, which leaves the timer
    unrooted and makes its seven slide joints unavailable to the task state
    view.  Remove only that wrapper, retain the original timer/button meshes
    and joints, and add the single task freejoint expected by the runtime.
    """

    if task_name != _YCB07_TASK_NAME:
        return 0

    # Be idempotent. This matters when a prepared XML is inspected/re-written
    # after a failed converter run: do not add duplicate freejoints or raise
    # merely because the promotion already happened.
    for worldbody in root.findall("worldbody"):
        for timer in list(worldbody):
            if timer.tag != "body" or timer.attrib.get("name") != "ycb07_timer":
                continue
            timer.attrib.update(
                {
                    "pos": "-0.30 0 0.9107235",
                    "quat": "1 0 0 0",
                }
            )
            if not any(
                child.tag == "freejoint"
                and child.attrib.get("name") == "ycb07_timer_root"
                for child in timer
            ):
                timer.insert(0, ET.Element("freejoint", {"name": "ycb07_timer_root"}))
            for geom in list(timer):
                if geom.tag == "geom" and geom.attrib.get("name") == "reg_bbox":
                    timer.remove(geom)
            return 1

    for worldbody in root.findall("worldbody"):
        for outer_index, outer in enumerate(list(worldbody)):
            if outer.tag != "body":
                continue
            timer = next(
                (
                    child
                    for child in list(outer)
                    if child.tag == "body" and child.attrib.get("name") == "object"
                ),
                None,
            )
            if timer is None:
                continue
            outer.remove(timer)
            timer.attrib.update(
                {
                    "name": "ycb07_timer",
                    "pos": "-0.30 0 0.9107235",
                    "quat": "1 0 0 0",
                }
            )
            timer.insert(0, ET.Element("freejoint", {"name": "ycb07_timer_root"}))
            worldbody.remove(outer)
            worldbody.insert(outer_index, timer)
            for geom in list(timer):
                if geom.tag == "geom" and geom.attrib.get("name") == "reg_bbox":
                    timer.remove(geom)
            return 1

    raise RuntimeError(
        "YCB07 source does not contain the authoritative YCB timer body"
    )


def _apply_hl015_marker_filter(root: ET.Element, task_name: str) -> int:
    """Keep HL015's strip-mounted route guides out of other C-tool previews."""

    if task_name == "bench320_hl015_bimanual_cable_route_dual_plug":
        return 0
    removed = 0
    for parent in root.iter():
        retained: list[ET.Element] = []
        for child in parent:
            if child.tag == "geom" and str(child.attrib.get("name", "")).startswith("hl015_"):
                removed += 1
                continue
            retained.append(child)
        if len(retained) != len(list(parent)):
            parent[:] = retained
    return removed


def success_condition_labels(task_name: str) -> tuple[str, ...]:
    _ensure_optional_task_metadata(task_name)
    return _SUCCESS_CONDITION_LABELS.get(
        task_name,
        ("Task uses an event trigger",),
    )


def create_success_condition_overlay(task_name: str):
    """Create the static preview-side condition panel.

    The collection env owns the live check marks; preview imports only the MJCF
    scene, so it can show the condition list but cannot evaluate object state.
    """
    value = os.environ.get("HANDSON_ISAACSIM_SUCCESS_CONDITIONS", "1").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return None

    import omni.ui as ui

    panel_width = 520 if task_name == "t31_prepare_coffee" else 430
    label_width = panel_width - 50
    root = None
    overlay = None
    try:
        from omni.kit.viewport.utility import get_active_viewport_window

        viewport_window = get_active_viewport_window()
        if viewport_window is not None:
            root = viewport_window.get_frame("handson.preview_success_condition_overlay")
            root.visible = True
            overlay = root
    except Exception:
        root = None

    if root is None:
        window = ui.Window(
            "HandsOn",
            width=panel_width,
            height=320,
            visible=True,
        )
        for attr, pos in (("position_x", 1180), ("position_y", 30)):
            try:
                setattr(window, attr, pos)
            except Exception:
                pass
        root = window.frame
        overlay = window

    with root:
        with ui.VStack():
            ui.Spacer(height=30)
            with ui.HStack():
                ui.Spacer()
                with ui.ZStack(width=panel_width, height=320):
                    ui.Rectangle(
                        style={
                            "background_color": 0xB0202020,
                            "border_color": 0x70FFFFFF,
                            "border_width": 1,
                            "border_radius": 6,
                        }
                    )
                    with ui.VStack(spacing=6, style={"margin": 12}):
                        ui.Label("Success Conditions", height=24)
                        for label in success_condition_labels(task_name):
                            with ui.HStack(height=22):
                                ui.Label(label, width=label_width)
                                ui.Label("", width=24)
                ui.Spacer(width=20)
            ui.Spacer()
    return overlay

# Match the native Bimanual Assembly IsaacLab reset posture so collection
# starts inside the same reachable teleoperation workspace.
_TASK_ARM_HOME = {
    "right": (-1.5707963, 1.5707963, 1.5707963, 1.5707963, -1.5707963, 0.0, 0.0),
    "left": (1.5707963, 1.5707963, -1.5707963, 1.5707963, 1.5707963, 0.0, 0.0),
}
_RAISED_ARM_HOME = {
    "right": (0.441741, 0.721729, 0.239304, 1.982058, -0.346778, -0.925025, 0.925025),
    "left": (-0.441741, 0.721729, -0.239304, 1.982058, 0.346778, -0.925025, -0.925025),
}
_D06_LOWERED_ARM_HOME = {
    "right": (
        -0.227685,
        0.624040,
        0.095735,
        2.231799,
        -0.626066,
        -1.036307,
        0.428326,
    ),
    "left": (
        0.227685,
        0.624040,
        -0.095735,
        2.231799,
        0.626066,
        -1.036307,
        -0.428326,
    ),
}
_A01_CLEAR_ARM_HOME = {
    # Same collision-safe A01 homes as the native collection runtime.  The
    # right solution raises palm/link5 by 80 mm without changing the tabletop
    # TCP XY/orientation; the inactive left hand remains above the bucket.
    "right": (-1.545103, 1.5707963, 1.5707963, 1.833922, -1.5707963, 0.0, 0.288803),
    "left": _RAISED_ARM_HOME["left"],
}
_C_TOOL_SPREAD_ARM_HOME = {
    "right": (-1.5707963, 1.5707963, 2.0207963, 1.5707963, -1.5707963, 0.0, 0.0),
    "left": (1.5707963, 1.5707963, -2.0207963, 1.5707963, 1.5707963, 0.0, 0.0),
}
_C_TOOL_CLEAR_START_TASKS = frozenset(
    {
        "c07_pitcher_pour_cup",
        "c08_tongs_ice_cubes",
        "c09_staple_paper_sheets",
        "t02_connect_charger_and_cable",
        "bench320_hl015_bimanual_cable_route_dual_plug",
        "c11_xylophone_sequence",
        "t11_line_up_blocks",
        _YCB07_TASK_NAME,
        "t40_slide_knife_blade",
        "t13_load_cans",
    }
)
_D06_ALIGNED_START_TASKS = frozenset(
    {
        "c09_staple_paper_sheets",
        "t02_connect_charger_and_cable",
        "bench320_hl015_bimanual_cable_route_dual_plug",
        "c11_xylophone_sequence",
        "t42_sort_objects",
        "d12_prepare_cutting_station",
    }
)
_D_SPREAD_BIMANUAL_START_TASKS = frozenset(
    {
        "t12_load_and_start_microwave",
        "d11_presoak_pan",
        "d13_prepare_tea_tray",
        "d14_wash_fruit",
        "d15_pack_matching_lunches",
        "d16_sweep_blocks_to_bin",
        "d17_store_laptop_headphones",
        "d18_pack_eggs",
        "d19_memory_cover_sort",
        "d20_microwave_hot_dog_plating",
        *_d20_stages.STAGE_TASKS,
    }
)
_B320_SEASONING_START_TASKS = frozenset(
    {"t30_pour_seasonings"}
)
_D_SINGLE_HAND_START_TASKS = frozenset(
    {
        "d07_organize_office_desk",
        "d10_load_dishwasher",
    }
)


def _task_arm_home(task_name: str | None) -> dict[str, tuple[float, ...]]:
    if task_name == "eval_gen02_open_trash_bin":
        from .abc_collection_env import _BC021_TABLETOP_ARM_HOME
        return {"right": tuple(_BC021_TABLETOP_ARM_HOME["right"]),
                "left": _RAISED_ARM_HOME["left"]}
    if task_name == xy5.TASK_NAME:
        from .abc_collection_env import _YCB01_CLEAR_ARM_HOME
        return {"right": tuple(_YCB01_CLEAR_ARM_HOME["right"]), "left": _RAISED_ARM_HOME["left"]}
    if task_name in _D06_ALIGNED_START_TASKS:
        return _D06_LOWERED_ARM_HOME
    if task_name in (
        _C_TOOL_CLEAR_START_TASKS
        | _D_SPREAD_BIMANUAL_START_TASKS
        | _B320_SEASONING_START_TASKS
    ):
        return _C_TOOL_SPREAD_ARM_HOME
    if task_name in {"t26_place_fruit_in_bucket", xy3.TASK_NAME}:
        return _A01_CLEAR_ARM_HOME
    if task_name in {
        "t03_cover_box",
        xy5.TASK_NAME,
        "t50_water_plant",
    } | _D_SINGLE_HAND_START_TASKS:
        return {
            "right": _TASK_ARM_HOME["right"],
            "left": _RAISED_ARM_HOME["left"],
        }
    return _TASK_ARM_HOME


def _write_xml_if_changed(root: ET.Element, output_path: Path) -> Path:
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not output_path.exists() or output_path.read_bytes() != content:
        output_path.write_bytes(content)
    return output_path


def _task_external_asset_closure(source: Path) -> list[tuple[Path, Path]]:
    """Return one task asset and its path-relative OBJ material closure.

    B03's socket OBJ references a sibling MTL file.  Hashing or copying only
    the OBJ would let a material edit silently reuse an old USD conversion.
    Keep the closure labels relative to the OBJ so the resulting cache key is
    independent of both the checkout root and the cache root.
    """

    source = source.resolve()
    # The primary label must not inherit an alias basename produced from an
    # absolute source path.  Its suffix is semantic to the importer; the host
    # filename is not.
    rows: dict[Path, Path] = {Path(f"asset{source.suffix.lower()}"): source}
    if source.suffix.lower() != ".obj":
        return list(rows.items())

    material_paths: list[tuple[Path, Path]] = []
    with source.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped.startswith("mtllib "):
                continue
            raw_reference = stripped.split(maxsplit=1)[1].strip()
            relative = Path(raw_reference)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "B03 content-addressed OBJ assets require path-relative "
                    f"material references: {source}: {raw_reference!r}"
                )
            material = (source.parent / relative).resolve()
            if not material.is_file():
                # Existing task preparation intentionally synthesizes missing
                # MTLs before this boundary.  A genuinely unresolved material
                # here must not be hidden from the physical asset identity.
                raise FileNotFoundError(
                    f"Missing OBJ material dependency: {source}: {material}"
                )
            rows[relative] = material
            material_paths.append((relative, material))

    for material_relative, material in material_paths:
        with material.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                token = stripped.split(maxsplit=1)[0].lower()
                if not token.startswith("map_"):
                    continue
                # MuJoCo's shipped B03 MTLs use the ordinary ``map_* file``
                # form.  Taking the final field also tolerates common MTL map
                # options without baking an absolute source directory into
                # the generated MJCF.
                raw_reference = stripped.split()[-1]
                relative = material_relative.parent / raw_reference
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "B03 content-addressed MTL assets require path-relative "
                        f"texture references: {material}: {raw_reference!r}"
                    )
                texture = (source.parent / relative).resolve()
                if not texture.is_file():
                    raise FileNotFoundError(
                        f"Missing MTL texture dependency: {material}: {texture}"
                    )
                rows[relative] = texture
    return sorted(rows.items(), key=lambda row: row[0].as_posix())


def _cache_b03_external_assets(root: ET.Element, output_dir: Path) -> int:
    """Rewrite every B03 file reference to a content-addressed cache path."""

    rewritten = 0
    for element in root.iter():
        raw_reference = element.attrib.get("file")
        if not raw_reference or element.tag == "include":
            continue
        candidate = Path(raw_reference)
        source = (
            candidate.resolve()
            if candidate.is_absolute()
            else (output_dir / candidate).resolve()
        )
        try:
            stable_relative = source.relative_to(output_dir.resolve())
        except ValueError:
            stable_relative = None
        if (
            stable_relative is not None
            and len(stable_relative.parts) >= 3
            and stable_relative.parts[0] in {"assets", "task_assets"}
            and len(stable_relative.parts[1]) == 64
        ):
            # Robot generation already installs its meshes below
            # ``assets/<sha256>``.  Retain those byte-stable references.
            element.attrib["file"] = stable_relative.as_posix()
            continue

        closure = _task_external_asset_closure(source)
        digest = hashlib.sha256()
        for relative, dependency in closure:
            label = relative.as_posix().encode("utf-8")
            payload = dependency.read_bytes()
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        content_sha256 = digest.hexdigest()
        asset_dir = output_dir / "task_assets" / content_sha256
        for relative, dependency in closure:
            destination = asset_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                not destination.is_file()
                or destination.read_bytes() != dependency.read_bytes()
            ):
                shutil.copy2(dependency, destination)
        # Isaac's MJCF importer flattens texture/mesh source basenames.  Keep
        # them content-unique as well as path-stable so three different B03
        # PNGs (or peg/socket OBJs) cannot overwrite one another as
        # ``source_assets/asset.png``.
        stable_name = f"asset_{content_sha256}{source.suffix.lower()}"
        stable_source = asset_dir / stable_name
        if stable_source != asset_dir / source.name:
            payload = source.read_bytes()
            if not stable_source.is_file() or stable_source.read_bytes() != payload:
                shutil.copy2(source, stable_source)
        element.attrib["file"] = stable_source.relative_to(output_dir).as_posix()
        rewritten += 1
    return rewritten


def _task_mjcf_content_digest(task_mjcf: Path) -> str:
    """Hash the exact B03 MJCF and every external byte it can load."""

    digest = hashlib.sha256()
    for label, payload in (
        (b"schema", _TASK_USD_CONTENT_SCHEMA),
        (b"task_mjcf", task_mjcf.read_bytes()),
    ):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    root = ET.parse(task_mjcf).getroot()
    for index, element in enumerate(root.iter()):
        raw_reference = element.attrib.get("file")
        if not raw_reference:
            continue
        candidate = Path(raw_reference)
        source = (
            candidate.resolve()
            if candidate.is_absolute()
            else (task_mjcf.parent / candidate).resolve()
        )
        for relative, dependency in _task_external_asset_closure(source):
            label = (
                f"{index}:{element.tag}:{element.attrib.get('name', '')}:"
                f"{raw_reference}:{relative.as_posix()}"
            ).encode("utf-8")
            payload = dependency.read_bytes()
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _cached_include_name(source: Path) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"include_{source.stem}_{digest}.xml"


def _inline_cached_includes(root: ET.Element, output_dir: Path) -> None:
    """Expand includes because Isaac Sim 5.1 drops second-level MJCF includes."""
    for parent in list(root.iter()):
        children = list(parent)
        if not children:
            continue
        replacement_children: list[ET.Element] = []
        changed = False
        for child in children:
            if child.tag != "include":
                replacement_children.append(child)
                continue
            include_path = (output_dir / child.attrib["file"]).resolve()
            included_root = ET.parse(include_path).getroot()
            replacement_children.extend(list(included_root))
            changed = True
        if changed:
            parent[:] = replacement_children


def _coalesce_top_level_sections(root: ET.Element) -> None:
    """Merge repeated sections left by include expansion for Isaac's parser."""
    merge_children = {
        "asset",
        "worldbody",
        "default",
        "actuator",
        "sensor",
        "contact",
        "equality",
        "tendon",
        "keyframe",
    }
    merge_attributes = {"compiler", "option", "statistic", "size"}
    first_by_tag: dict[str, ET.Element] = {}
    result: list[ET.Element] = []
    for child in list(root):
        if not isinstance(child.tag, str):
            result.append(child)
            continue
        existing = first_by_tag.get(child.tag)
        if existing is None or child.tag not in merge_children | merge_attributes:
            first_by_tag.setdefault(child.tag, child)
            result.append(child)
            continue
        existing.attrib.update(child.attrib)
        if child.tag in merge_children:
            existing.extend(list(child))
    root[:] = result


def _alias_duplicate_mesh_sources(root: ET.Element, output_dir: Path) -> None:
    """Keep every importer scratch file in this attempt's conversion cache.

    Isaac writes ``*_tmp`` beside the mesh, even when its USD output is in a
    separate cache. External first-use meshes therefore need private copies
    too; isolating only duplicate references still races across processes.
    """
    seen: dict[Path, int] = {}
    for mesh in root.iter("mesh"):
        file_name = mesh.attrib.get("file")
        if not file_name:
            continue
        source = (output_dir / file_name).resolve()
        duplicate_index = seen.get(source, 0)
        seen[source] = duplicate_index + 1
        missing_mtl = _missing_obj_material_libraries(source) if source.suffix.lower() == ".obj" else []
        if duplicate_index == 0 and not missing_mtl and source.is_relative_to(output_dir.resolve()):
            continue

        digest_input = f"{source}:{mesh.attrib.get('name', '')}:{duplicate_index}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
        alias_dir = output_dir / "mesh_aliases" / digest
        alias_path = alias_dir / source.name
        if source.suffix.lower() == ".obj":
            shutil.copytree(source.parent, alias_dir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*_tmp", "*.tmp.usd"))
            _write_missing_obj_material_libraries(source, alias_dir, missing_mtl)
        else:
            alias_dir.mkdir(parents=True, exist_ok=True)
            if not alias_path.exists() or alias_path.stat().st_size != source.stat().st_size:
                shutil.copy2(source, alias_path)
        mesh.attrib["file"] = os.path.relpath(alias_path, output_dir)


def _missing_obj_material_libraries(source: Path) -> list[Path]:
    missing: list[Path] = []
    with source.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("mtllib "):
                continue
            library = Path(line.split(maxsplit=1)[1].strip())
            if not (source.parent / library).is_file():
                missing.append(library)
    return missing


def _write_missing_obj_material_libraries(source: Path, alias_dir: Path, libraries: list[Path]) -> None:
    if not libraries:
        return
    material_names: list[str] = []
    with source.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("usemtl "):
                name = line.split(maxsplit=1)[1].strip()
                if name and name not in material_names:
                    material_names.append(name)
    if not material_names:
        material_names.append("material_0")

    texture = alias_dir / "material_0.png"
    for library in libraries:
        library_path = alias_dir / library
        library_path.parent.mkdir(parents=True, exist_ok=True)
        blocks = []
        for name in material_names:
            lines = [
                f"newmtl {name}",
                "Ka 1.0 1.0 1.0",
                "Kd 0.8 0.8 0.8",
                "Ks 0.1 0.1 0.1",
                "Ns 25.0",
            ]
            if texture.is_file():
                lines.append(f"map_Kd {os.path.relpath(texture, library_path.parent)}")
            blocks.append("\n".join(lines))
        library_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _bind_b04_wood_block_visual_material(stage, root_path: str) -> int:
    """Override the B04 large wood block visual when OBJ/MTL import resolves black."""
    import numpy as np
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from pxr import Usd, UsdGeom, UsdShade

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return 0

    material_kwargs = {
        "prim_path": "/World/Looks/HandsOnB04LargeWoodBlock",
        "color": np.asarray(
            (1.0, 1.0, 1.0)
            if _B04_WOOD_BLOCK_TEXTURE.is_file()
            else _B04_WOOD_BLOCK_VISUAL_COLOR,
            dtype=np.float32,
        ),
    }
    if _B04_WOOD_BLOCK_TEXTURE.is_file():
        material_kwargs["texture_path"] = str(_B04_WOOD_BLOCK_TEXTURE.resolve())
    material = OmniPBR(**material_kwargs)
    material.set_project_uvw(False)
    material.set_reflection_roughness(0.72)
    material.set_metallic_constant(0.0)

    rebound = 0
    rebound_paths: set[str] = set()
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())
        prim_path_lower = prim_path.lower()
        if "036_wood_block" not in prim_path or not prim.IsA(UsdGeom.Gprim):
            continue
        if "collision" in prim_path_lower or "/collisions/" in prim_path_lower:
            continue
        if "visual" not in prim_path_lower and "/visuals/" not in prim_path_lower:
            continue

        target_prim = prim
        while target_prim.IsInstanceProxy():
            target_prim = target_prim.GetParent()
        target_path = str(target_prim.GetPath())
        if target_path in rebound_paths:
            continue
        UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            material.material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        rebound_paths.add(target_path)
        rebound += 1
    return rebound


def _is_50_water_plant_task(task_name: str | None) -> bool:
    return task_name in {"t50_water_plant", "water_plant"}


def _is_c04_plant_alpha_material_name(material_name: str) -> bool:
    return material_name in _C04_PLANT_ALPHA_MATERIALS


def _c04_model_material_name(prim_path: str) -> str | None:
    normalized = prim_path.lower()
    if "/visuals/" not in normalized:
        return None
    for model_name, material_name in _C04_PLANT_MODEL_MATERIALS.items():
        if model_name in normalized:
            return material_name
    return None


def _c04_alpha_texture_path(texture_path: Path) -> Path:
    if texture_path.name.endswith(_C04_PLANT_ALPHA_TEXTURE_SUFFIX):
        return texture_path
    return texture_path.with_name(f"{texture_path.stem}{_C04_PLANT_ALPHA_TEXTURE_SUFFIX}")


def _c04_material_texture_paths(task_mjcf: Path, task_name: str) -> dict[str, Path]:
    if not _is_50_water_plant_task(task_name):
        return {}

    root = ET.parse(task_mjcf).getroot()
    textures = {
        texture.attrib["name"]: texture.attrib["file"]
        for texture in root.findall(".//texture")
        if texture.attrib.get("name") in _C04_PLANT_TEXTURES
        and texture.attrib.get("file")
    }
    material_textures: dict[str, Path] = {}
    expected_materials = set(_C04_PLANT_MODEL_MATERIALS.values())
    for material in root.findall(".//material"):
        material_name = material.attrib.get("name", "")
        texture_name = material.attrib.get("texture", "")
        if (
            material_name not in expected_materials
            or texture_name not in textures
        ):
            continue
        texture_path = (task_mjcf.parent / textures[texture_name]).resolve()
        if _is_c04_plant_alpha_material_name(material_name):
            alpha_path = _c04_alpha_texture_path(texture_path)
            if alpha_path.is_file():
                texture_path = alpha_path
        if texture_path.is_file():
            material_textures[material_name] = texture_path
    return material_textures


def _set_shader_input(shader, input_name: str, value_type, value) -> bool:
    attr = shader.GetPrim().GetAttribute(f"inputs:{input_name}")
    if not attr.IsValid():
        attr = shader.GetPrim().CreateAttribute(
            f"inputs:{input_name}",
            value_type,
            custom=True,
        )
    previous = attr.Get()
    if previous == value:
        return False
    attr.Set(value)
    return True


def _configure_c04_plant_shader_opacity(shader, texture_path: Path) -> bool:
    from pxr import Sdf

    texture = str(texture_path.resolve())
    changed = False
    changed |= _set_shader_input(shader, "diffuse_texture", Sdf.ValueTypeNames.Asset, texture)
    changed |= _set_shader_input(shader, "opacity_texture", Sdf.ValueTypeNames.Asset, texture)
    changed |= _set_shader_input(shader, "enable_opacity", Sdf.ValueTypeNames.Bool, True)
    changed |= _set_shader_input(shader, "enable_opacity_texture", Sdf.ValueTypeNames.Bool, True)
    # OmniPBR opacity mode 0 is mono_alpha.  Mode 1 averages RGB and makes the
    # white pixels opaque even when the PNG alpha channel is zero.
    changed |= _set_shader_input(shader, "opacity_mode", Sdf.ValueTypeNames.Int, 0)
    changed |= _set_shader_input(shader, "opacity_threshold", Sdf.ValueTypeNames.Float, 0.5)
    changed |= _set_shader_input(shader, "opacity_constant", Sdf.ValueTypeNames.Float, 1.0)
    return changed


def _create_c04_plant_material(
    material_path: str,
    texture_path: Path,
    *,
    enable_alpha: bool,
):
    import numpy as np
    from isaacsim.core.api.materials.omni_pbr import OmniPBR

    material = OmniPBR(
        prim_path=material_path,
        texture_path=str(texture_path.resolve()),
        color=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
    )
    material.set_project_uvw(False)
    material.set_reflection_roughness(0.68)
    material.set_metallic_constant(0.0)
    if enable_alpha and material.shaders_list:
        _configure_c04_plant_shader_opacity(material.shaders_list[0], texture_path)
    return material.material


def _bind_c04_plant_materials(
    stage,
    root_path: str,
    task_name: str,
    material_textures: dict[str, Path],
) -> int:
    if not material_textures or not _is_50_water_plant_task(task_name):
        return 0

    from pxr import Usd, UsdGeom, UsdShade

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return 0

    # The MJCF converter packs all four plant visuals into one USD instance.
    # Instance proxies cannot receive per-mesh bindings, so de-instance only
    # this visual container before assigning the four source materials.
    for prim in list(Usd.PrimRange(root_prim)):
        prim_path = str(prim.GetPath()).lower()
        if (
            prim.IsInstance()
            and prim.GetName() == "visuals"
            and "/plant/" in prim_path
        ):
            prim.SetInstanceable(False)
            # The generic repair has placed one stronger-than-descendants
            # material on this common container.  Remove it so the four child
            # bindings below can resolve independently.
            UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()

    materials: dict[str, object] = {}
    for model_name, material_name in _C04_PLANT_MODEL_MATERIALS.items():
        texture_path = material_textures.get(material_name)
        if texture_path is None:
            continue
        materials[model_name] = _create_c04_plant_material(
            f"/World/Looks/HandsOnC04Plant_{model_name}",
            texture_path,
            enable_alpha=_is_c04_plant_alpha_material_name(material_name),
        )

    bound = 0
    bound_paths: set[str] = set()
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        prim_path = str(prim.GetPath())
        material_name = _c04_model_material_name(prim_path)
        if material_name is None:
            continue
        model_name = next(
            (
                candidate
                for candidate, candidate_material in _C04_PLANT_MODEL_MATERIALS.items()
                if candidate_material == material_name and candidate in prim_path.lower()
            ),
            None,
        )
        if model_name is None or model_name not in materials:
            continue

        target_prim = prim
        while target_prim.IsInstanceProxy():
            target_prim = target_prim.GetParent()
        target_path = str(target_prim.GetPath())
        if target_path in bound_paths:
            continue

        UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            materials[model_name],
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound_paths.add(target_path)
        bound += 1

    return bound


def _repair_c06_fixture_materials(stage, root_path: str, task_name: str) -> int:
    """Restore the C06 logo without exposing its hidden collision container."""
    if task_name not in {"t23_photograph_logo", xy3.TASK_NAME}:
        return 0

    import numpy as np
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    if task_name == xy3.TASK_NAME:
        # XY3's visual-only decal follows the bucket root. Do not expose the
        # bucket collision container when making its label visible.
        imported_logo = next((prim for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies())
                     if prim.GetName() == "photograph_logo_disk"
                     and str(prim.GetPath()).startswith(root_path + "/bucket")), None)
        if imported_logo is None:
            raise RuntimeError("XY3 bucket logo visual missing after MJCF import")
        bucket = imported_logo.GetParent()
        while bucket.IsValid() and not bucket.HasAPI(UsdPhysics.RigidBodyAPI):
            bucket = bucket.GetParent()
        if not bucket.IsValid():
            raise RuntimeError("XY3 logo is not attached to the physical bucket")
        cache = UsdGeom.XformCache()
        local_transform = cache.GetLocalToWorldTransform(imported_logo) * cache.GetLocalToWorldTransform(bucket).GetInverse()
        # Keep the imported primitive hidden, including when the importer puts
        # it inside an invisible collision instance. Author the UV decal as a
        # separate visual child of the same rigid body, not under that scope.
        if UsdGeom.Imageable(imported_logo).ComputeVisibility() != UsdGeom.Tokens.invisible:
            from isaaclab.sim import make_uninstanceable
            source_path = str(imported_logo.GetPath())
            make_uninstanceable(str(bucket.GetPath()), stage=stage)
            UsdGeom.Imageable(stage.GetPrimAtPath(source_path)).MakeInvisible()
        logo_xform = UsdGeom.Xform.Define(stage, f"{bucket.GetPath()}/xy3_bucket_logo")
        logo_xform.MakeMatrixXform().Set(local_transform)
        logo = logo_xform.GetPrim()
    else:
        collisions = stage.GetPrimAtPath(f"{root_path}/table/table/collisions")
        if not collisions.IsValid():
            return 0

        if collisions.IsInstance():
            collisions.SetInstanceable(False)

        logo = stage.GetPrimAtPath(f"{collisions.GetPath()}/photograph_logo_disk")
        table_collision = stage.GetPrimAtPath(
            f"{collisions.GetPath()}/table_collision"
        )
        if not logo.IsValid():
            return 0

        UsdGeom.Imageable(collisions).MakeVisible()
        if table_collision.IsValid():
            UsdGeom.Imageable(table_collision).MakeInvisible()
        UsdGeom.Imageable(logo).MakeVisible()

    UsdGeom.Imageable(logo).MakeVisible()

    logo_material = OmniPBR(
        prim_path="/World/Looks/HandsOnC06Logo",
        texture_path=str((_XML_DIR / "photograph" / "logo-CASIA.png").resolve()),
        color=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
    )
    logo_material.set_project_uvw(False)
    logo_material.set_reflection_roughness(0.72)
    logo_material.set_metallic_constant(0.0)
    logo_gprim = next(
        (
            child
            for child in logo.GetChildren()
            if child.IsA(UsdGeom.Gprim)
        ),
        logo,
    )
    # The imported MuJoCo cylinder has no usable UV set, so binding the image
    # directly renders a plain white disk.  Keep that cylinder for collision
    # and place a UV-authored disk at the same local transform for rendering.
    if task_name != xy3.TASK_NAME:
        UsdGeom.Imageable(logo_gprim).MakeInvisible()
    segment_count = 64
    radius = xy3.LOGO_RADIUS if task_name == xy3.TASK_NAME else 0.0798
    points = [Gf.Vec3f(0.0, 0.0, 0.0)]
    texcoords = [Gf.Vec2f(0.5, 0.5)]
    for index in range(segment_count):
        angle = 2.0 * np.pi * index / segment_count
        x = float(radius * np.cos(angle))
        y = float(radius * np.sin(angle))
        points.append(Gf.Vec3f(x, y, 0.0))
        texcoords.append(
            Gf.Vec2f(
                float(0.5 + 0.5 * np.cos(angle)),
                float(0.5 + 0.5 * np.sin(angle)),
            )
        )

    decal = UsdGeom.Mesh.Define(stage, f"{logo.GetPath()}/logo_decal")
    decal.CreatePointsAttr(points)
    decal.CreateFaceVertexCountsAttr([3] * segment_count)
    decal.CreateFaceVertexIndicesAttr(
        [
            vertex
            for index in range(segment_count)
            for vertex in (0, index + 1, (index + 1) % segment_count + 1)
        ]
    )
    decal.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    decal.CreateDoubleSidedAttr(True)
    st_primvar = UsdGeom.PrimvarsAPI(decal).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    st_primvar.Set(texcoords)
    UsdShade.MaterialBindingAPI.Apply(decal.GetPrim()).Bind(
        logo_material.material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    return 1


def _repair_d09_coffee_machine_material(
    stage,
    root_path: str,
    task_name: str,
) -> int:
    """Lift the licensed D09 albedo's crushed blacks in RTX renders.

    The OBJ has valid UVs and both visible meshes inherit the expected
    material.  The source 4K albedo itself, however, has more than half of its
    pixels below RGB 20.  OmniPBR's texture-space offset retains every
    authored scratch and metal highlight while turning zero-valued panels
    into readable charcoal.  This is visual-only: collision and physics
    material properties are untouched.
    """

    if task_name != "t31_prepare_coffee":
        return 0

    from pxr import Sdf, Usd, UsdShade

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return 0

    material_name = "material_d09_coffee_machine_material"
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() != material_name or not prim.IsA(UsdShade.Material):
            continue
        for candidate in Usd.PrimRange(prim):
            if not candidate.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(candidate)
            texture = shader.GetInput("diffuse_texture")
            if not texture or texture.Get() is None:
                continue
            shader.CreateInput(
                "albedo_add",
                Sdf.ValueTypeNames.Float,
            ).Set(0.10)
            shader.CreateInput(
                "albedo_brightness",
                Sdf.ValueTypeNames.Float,
            ).Set(1.0)
            shader.CreateInput(
                "reflection_roughness_constant",
                Sdf.ValueTypeNames.Float,
            ).Set(0.32)
            return 1
    return 0


def _hide_c04_spray_visual(stage, root_path: str, task_name: str) -> int:
    if not _is_50_water_plant_task(task_name):
        return 0

    from pxr import Usd, UsdGeom

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return 0
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() != "cone_vis" or not prim.IsA(UsdGeom.Xformable):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        return 1
    return 0


def _alias_texture_sources(root: ET.Element, output_dir: Path) -> None:
    """Give textures unique basenames before Isaac flattens its material cache."""
    alias_dir = output_dir / "texture_aliases"
    alias_dir.mkdir(parents=True, exist_ok=True)
    for texture in root.iter("texture"):
        file_name = texture.attrib.get("file")
        if not file_name:
            continue
        source = (output_dir / file_name).resolve()
        digest_input = f"{source}:{texture.attrib.get('name', '')}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
        alias_path = alias_dir / f"{source.stem}_{digest}{source.suffix}"
        if not alias_path.exists() or alias_path.stat().st_size != source.stat().st_size:
            shutil.copy2(source, alias_path)
        texture.attrib["file"] = os.path.relpath(alias_path, output_dir)


def _apply_c04_plant_texture_alpha(
    root: ET.Element,
    output_dir: Path,
    task_name: str,
) -> int:
    """Make C04 leaf texture white backgrounds transparent for Isaac import."""
    if task_name not in {"t50_water_plant", "water_plant"}:
        return 0

    try:
        from PIL import Image
    except Exception:
        return 0

    changed = 0
    for texture in root.iter("texture"):
        if texture.attrib.get("name") not in _C04_PLANT_ALPHA_TEXTURES:
            continue
        file_name = texture.attrib.get("file")
        if not file_name:
            continue
        source = (output_dir / file_name).resolve()
        if not source.is_file():
            continue

        image = Image.open(source).convert("RGBA")
        pixels = image.load()
        width, height = image.size
        transparent = 0
        for y in range(height):
            for x in range(width):
                red, green, blue, alpha = pixels[x, y]
                if red >= 245 and green >= 245 and blue >= 245:
                    pixels[x, y] = (red, green, blue, 0)
                    transparent += 1
                elif alpha != 255:
                    pixels[x, y] = (red, green, blue, 255)

        if transparent == 0:
            continue
        alpha_path = _c04_alpha_texture_path(source)
        image.save(alpha_path)
        texture.attrib["file"] = os.path.relpath(alpha_path, output_dir)
        changed += 1
    return changed


def _remove_c04_isaac_hidden_visuals(root: ET.Element, task_name: str) -> int:
    """Drop MuJoCo-only C04 target helpers that Isaac renders as solid geometry."""
    if not _is_50_water_plant_task(task_name):
        return 0

    removed = 0
    for body in root.iter("body"):
        retained: list[ET.Element] = []
        for element in body:
            is_hidden_target = (
                element.tag == "geom"
                and element.attrib.get("class") == "plant_visual"
                and element.attrib.get("type") == "cylinder"
                and element.attrib.get("group") == "5"
                and element.attrib.get("size") == "0.2 0.2"
            )
            if is_hidden_target:
                removed += 1
            else:
                retained.append(element)
        body[:] = retained
    return removed


def _remove_mocap_target_helper_geoms(root: ET.Element) -> int:
    """Drop visual helper boxes from mocap target bodies before Isaac import."""
    removed = 0
    for body in root.iter("body"):
        body_name = body.attrib.get("name", "")
        if (
            body.attrib.get("mocap") != "true"
            or body_name not in _MOCAP_TARGET_HELPER_GEOMS
        ):
            continue
        for element in list(body):
            if (
                element.tag == "geom"
                and element.attrib.get("name") == body_name
                and element.attrib.get("contype") == "0"
                and element.attrib.get("conaffinity") == "0"
            ):
                body.remove(element)
                removed += 1
    return removed


def _remove_c06_target_region_visual(
    root: ET.Element,
    task_name: str,
) -> int:
    """Remove C06's green success-region marker from the Isaac render scene."""
    if task_name not in {"t23_photograph_logo", xy3.TASK_NAME}:
        return 0

    removed = 0
    for parent in root.iter():
        for element in list(parent):
            if (
                element.tag == "geom"
                and element.attrib.get("name")
                == "photograph_target_region_visual"
                and element.attrib.get("contype") == "0"
                and element.attrib.get("conaffinity") == "0"
            ):
                parent.remove(element)
                removed += 1
    return removed


def _apply_c04_spray_inertials(root: ET.Element, task_name: str) -> int:
    """Preserve the source spray bottle's compiled rigid-body dynamics."""
    if not _is_50_water_plant_task(task_name):
        return 0

    applied = 0
    for body in root.iter("body"):
        attributes = _C04_SPRAY_BODY_INERTIALS.get(body.attrib.get("name", ""))
        if attributes is None:
            continue
        existing = body.find("inertial")
        if existing is not None:
            body.remove(existing)
        body.insert(0, ET.Element("inertial", dict(attributes)))
        applied += 1
    return applied


def _apply_c04_stable_spray_root_collision(
    root: ET.Element,
    task_name: str,
) -> int:
    """Keep one broad support box instead of overlapping scanned root hulls."""
    if not _is_50_water_plant_task(task_name):
        return 0

    link_1 = next(
        (
            body
            for body in root.iter("body")
            if body.attrib.get("name") == "link_1"
        ),
        None,
    )
    if link_1 is None:
        return 0
    removed = 0
    for geom in list(link_1.findall("geom")):
        if (
            geom.attrib.get("class") == "spray/collision"
            and geom.attrib.get("type", "mesh") == "mesh"
        ):
            link_1.remove(geom)
            removed += 1
    return removed


def _apply_c05_mousepad_inertial(root: ET.Element, task_name: str) -> int:
    """Keep Isaac from creating a negative-mass rigid body for the visual pad."""
    if task_name not in {"t14_move_and_click_mouse", "eval_comb02_open_laptop_and_click_mouse"}:
        return 0
    for body in root.iter("body"):
        if body.attrib.get("name") != "mousepad":
            continue
        existing = body.find("inertial")
        if existing is not None:
            body.remove(existing)
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "1e-6",
                    "diaginertia": "1e-12 1e-12 1e-12",
                },
            ),
        )
        return 1
    return 0


def _apply_d05_scissors_auxiliary_inertial(root: ET.Element, task_name: str) -> int:
    """Give D05's welded visual-only link valid native mass properties."""
    if task_name != "t43_squeeze_scissors":
        return 0
    for body in root.iter("body"):
        if body.attrib.get("name") != "d05_scissors_link":
            continue
        existing = body.find("inertial")
        if existing is not None:
            body.remove(existing)
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "1e-6",
                    "diaginertia": "1e-12 1e-12 1e-12",
                },
            ),
        )
        return 1
    return 0


def _fix_c05_display(root: ET.Element, task_name: str) -> int:
    """Keep the C05 monitor fixed at the rear of the tabletop."""
    if task_name != "t14_move_and_click_mouse":
        return 0
    for body in root.iter("body"):
        if body.attrib.get("name") != "display_link_1":
            continue
        removed = 0
        removed_joint_names: set[str] = set()
        for descendant in body.iter("body"):
            for child in list(descendant):
                if child.tag not in {"freejoint", "joint"}:
                    continue
                joint_name = child.attrib.get("name")
                if joint_name:
                    removed_joint_names.add(joint_name)
                descendant.remove(child)
                removed += 1
        # Do not leave actuators or sensors pointing at joints that were
        # removed when the display was welded into the C05 scene.
        for parent in root.iter():
            for child in list(parent):
                if child.attrib.get("joint") in removed_joint_names:
                    parent.remove(child)
        return removed
    return 0


def _remove_c06_camera_debug_visuals(
    root: ET.Element,
    task_name: str,
) -> int:
    """Remove camera helper visuals that are not used by the IsaacLab scene."""
    if task_name not in {"t23_photograph_logo", xy3.TASK_NAME}:
        return 0

    removed = 0
    for parent in root.iter():
        for element in list(parent):
            if (
                element.tag == "geom"
                and element.attrib.get("name")
                in _C06_ISAAC_ONLY_REMOVED_CAMERA_GEOMS
            ):
                parent.remove(element)
                removed += 1
    return removed


def _apply_c06_camera_auxiliary_inertials(
    root: ET.Element,
    task_name: str,
) -> int:
    """Give Isaac-valid mass properties to C06's welded visual-only links."""
    if task_name not in {"t23_photograph_logo", xy3.TASK_NAME}:
        return 0

    camera_root = next(
        (
            body
            for body in root.iter("body")
            if body.attrib.get("name") == "link_19"
            and any(
                joint.tag == "freejoint"
                and joint.attrib.get("name") == "camera_root"
                for joint in body
            )
        ),
        None,
    )
    if camera_root is None:
        return 0

    applied = 0
    for body in camera_root.findall("body"):
        if body.attrib.get("name") == "link_7" or body.find("inertial") is not None:
            continue
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "1e-6",
                    "diaginertia": "1e-12 1e-12 1e-12",
                },
            ),
        )
        applied += 1
    return applied


def _configure_c06_shutter_collision(
    root: ET.Element,
    task_name: str,
) -> int:
    """Enable the movable shutter collider that MuJoCo activates at runtime."""
    if task_name not in {"t23_photograph_logo", xy3.TASK_NAME}:
        return 0

    for geom in root.iter("geom"):
        if geom.attrib.get("name") != "camera_shutter_button_collision":
            continue
        geom.attrib.update(
            {
                "size": "0.015 0.0005",
                "contype": "1",
                "conaffinity": "1",
                "margin": "0.002",
                "gap": "0",
            }
        )
        return 1
    return 0


def _add_missing_fixed_body_inertials(root: ET.Element) -> None:
    inertials = {
        "rokae_center_components": ("8", "0.25 0.25 0.25"),
        "rokae_center_head_fixed_frame": ("1", "0.02 0.02 0.02"),
    }
    for body in root.iter("body"):
        values = inertials.get(body.attrib.get("name", ""))
        if values is None or body.find("inertial") is not None:
            continue
        mass, diaginertia = values
        body.insert(0, ET.Element("inertial", {"pos": "0 0 0", "mass": mass, "diaginertia": diaginertia}))


def _remove_elements_referencing(root: ET.Element, removed_names: set[str]) -> None:
    if not removed_names:
        return
    reference_sections = {"actuator", "sensor", "contact", "equality", "tendon"}
    for section in root:
        if section.tag not in reference_sections:
            continue
        section[:] = [
            element
            for element in section
            if not any(value in removed_names for value in element.attrib.values())
        ]


def _prune_unused_asset_definitions(root: ET.Element) -> int:
    """Remove file assets that no retained scene element can reference.

    The shared YCB source declares every object mesh and texture even after a
    task-specific body filter removes most objects. Isaac's MJCF importer still
    opens those unused files, so YCB01 would silently depend on non-task assets.
    """
    referenced_meshes = {
        element.attrib["mesh"]
        for element in root.iter()
        if element.tag == "geom" and element.attrib.get("mesh")
    }
    referenced_materials = {
        element.attrib["material"]
        for element in root.iter()
        if element.attrib.get("material")
    }
    referenced_textures = {
        element.attrib["texture"]
        for asset in root.findall("asset")
        for element in asset
        if element.tag == "material"
        and element.attrib.get("name") in referenced_materials
        and element.attrib.get("texture")
    }

    removed = 0
    for asset in root.findall("asset"):
        retained: list[ET.Element] = []
        for element in asset:
            name = element.attrib.get("name")
            unused = bool(
                name
                and (
                    (element.tag == "mesh" and name not in referenced_meshes)
                    or (
                        element.tag == "material"
                        and name not in referenced_materials
                    )
                    or (
                        element.tag == "texture"
                        and element.attrib.get("type") != "skybox"
                        and name not in referenced_textures
                    )
                )
            )
            if unused:
                removed += 1
            else:
                retained.append(element)
        asset[:] = retained
    return removed


def _apply_ycb_task_layout(root: ET.Element, task_name: str) -> None:
    """Match the MuJoCo reset layout and omit objects parked by that reset."""
    layout = _YCB_TASK_LAYOUTS.get(task_name)
    if layout is None:
        return

    active_names = set(layout)
    removed_names = set(_YCB_SHARED_OBJECTS - active_names)
    if task_name in {"z01_pipette", "z02_bottle_cap"}:
        removed_names.add("target")
    if task_name == "t34_roll_ball_through_gate":
        # The shared YCB arena includes a non-YCB bucket for older placement
        # tasks. YCB01 is deliberately built from exactly three YCB objects.
        removed_names.add("bucket")
    removed_references = set(removed_names)
    for worldbody in root.findall("worldbody"):
        retained: list[ET.Element] = []
        for element in worldbody:
            if element.tag != "body" or element.attrib.get("name") not in removed_names:
                retained.append(element)
                continue
            removed_references.update(
                descendant.attrib["name"]
                for descendant in element.iter()
                if descendant.tag in {"joint", "freejoint"} and "name" in descendant.attrib
            )
        worldbody[:] = retained

        for body in worldbody.findall("body"):
            name = body.attrib.get("name")
            if name not in layout:
                continue
            x, y = layout[name]
            is_a06_cup = task_name == "t19_orient_cup_upright" and name == "a_cup"
            z_offset = (
                _A06_SIDE_REST_Z_OFFSET
                if is_a06_cup
                else _YCB_OBJECT_Z_OFFSETS[name]
            )
            body.attrib["pos"] = (
                f"{x:.6f} {y:.6f} {_YCB_TABLE_TOP_Z + z_offset:.6f}"
            )
            if is_a06_cup:
                body.attrib["quat"] = " ".join(
                    f"{value:.10f}" for value in _A06_SIDE_REST_QUAT_WXYZ
                )

    _remove_elements_referencing(root, removed_references)
    if task_name == "t34_roll_ball_through_gate":
        _prune_unused_asset_definitions(root)


def _apply_a02_task_filter(root: ET.Element, task_name: str) -> int:
    """Keep only the mug and plate that participate in A02."""
    if task_name != "t25_place_cup_on_plate":
        return 0

    removed_references = set(_A02_INACTIVE_OBJECTS)
    removed_count = 0
    for worldbody in root.findall("worldbody"):
        retained: list[ET.Element] = []
        for element in worldbody:
            if (
                element.tag != "body"
                or element.attrib.get("name") not in _A02_INACTIVE_OBJECTS
            ):
                retained.append(element)
                continue
            removed_count += 1
            removed_references.update(
                descendant.attrib["name"]
                for descendant in element.iter()
                if descendant.tag in {"joint", "freejoint"}
                and "name" in descendant.attrib
            )
        worldbody[:] = retained

    _remove_elements_referencing(root, removed_references)
    return removed_count


def _apply_ycb06_original_asset_bodies(
    root: ET.Element,
    task_name: str,
) -> int:
    """Promote the six untouched YCB source bodies into the YCB06 scene.

    The upstream ``model.xml`` files intentionally all call their body
    ``object`` and reuse generic default-class names.  The task scene includes
    those files directly so their meshes, textures, densities, and collision
    decompositions remain authoritative.  This preparation step only assigns
    unique task identities/root joints and expands the source defaults into
    explicit geom attributes required by the combined MJCF.
    """
    if task_name != "t28_place_tableware":
        return 0

    task_names = {
        str(spec["name"])
        for spec in _YCB06_SOURCE_BODY_BY_MATERIAL.values()
    }
    promoted = 0
    for worldbody in root.findall("worldbody"):
        retained: list[ET.Element] = []
        for wrapper in list(worldbody):
            source_body = next(
                (
                    body
                    for body in wrapper.findall("body")
                    if body.attrib.get("name") == "object"
                ),
                None,
            )
            if source_body is None and wrapper.attrib.get("name") == "object":
                source_body = wrapper
            if source_body is None:
                retained.append(wrapper)
                continue

            material_name = next(
                (
                    geom.attrib.get("material", "")
                    for geom in source_body.findall("geom")
                    if geom.attrib.get("material")
                ),
                "",
            )
            spec = _YCB06_SOURCE_BODY_BY_MATERIAL.get(material_name)
            if spec is None:
                retained.append(wrapper)
                continue

            source_body.attrib.update(
                {
                    "name": str(spec["name"]),
                    "pos": str(spec["pos"]),
                    "quat": str(spec["quat"]),
                }
            )
            for geom in list(source_body.findall("geom")):
                if geom.attrib.get("name") == "reg_bbox":
                    source_body.remove(geom)
                    continue
                geom.attrib.pop("class", None)
                if geom.attrib.get("material"):
                    geom.attrib.update(
                        {
                            "type": "mesh",
                            "contype": "0",
                            "conaffinity": "0",
                            "group": "2",
                            "mass": "0",
                            "friction": "0.95 0.3 0.1",
                            "solref": "0.001 1",
                            "solimp": "0.998 0.998 0.001",
                        }
                    )
                else:
                    geom.attrib.update(
                        {
                            "type": "mesh",
                            "group": "3",
                            "density": str(spec["density"]),
                            "condim": "4",
                            "friction": "1.2 0.04 0.004",
                            "solref": "0.004 1",
                            "solimp": "0.998 0.998 0.001",
                            "rgba": "0 0 0 0",
                        }
                    )

            for joint in list(source_body):
                if joint.tag in {"joint", "freejoint"}:
                    source_body.remove(joint)
            if bool(spec["dynamic"]):
                source_body.insert(
                    0,
                    ET.Element(
                        "freejoint",
                        {"name": f"{spec['name']}_freejoint"},
                    ),
                )
            retained.append(source_body)
            promoted += 1
        worldbody[:] = retained

    generic_classes = {"visual", "collision", "spawn", "region"}
    for default in list(root.iter("default")):
        default[:] = [
            child
            for child in default
            if not (
                child.tag == "default"
                and child.attrib.get("class") in generic_classes
            )
        ]

    for sensor in root.findall("sensor"):
        sensor[:] = [
            element
            for element in sensor
            if not element.attrib.get("objname")
            or element.attrib.get("objname") in task_names
        ]

    if promoted != len(_YCB06_SOURCE_BODY_BY_MATERIAL):
        raise RuntimeError(
            "YCB06 source-body preparation resolved "
            f"{promoted}/{len(_YCB06_SOURCE_BODY_BY_MATERIAL)} YCB models"
        )
    return promoted


def _apply_b02_isaac_reset_repair(root: ET.Element, task_name: str) -> int:
    """Remove MuJoCo's moving helper and place the free nut on the table."""
    if task_name != "b02_nut_bolt_assembly":
        return 0

    changed = 0
    removed_names = {"b02_thread_support_body", "b02_thread_support"}
    for parent in root.iter():
        for element in list(parent):
            if (
                element.tag == "body"
                and element.attrib.get("name") == "b02_thread_support_body"
            ):
                parent.remove(element)
                changed += 1
    _remove_elements_referencing(root, removed_names)

    nut = next(
        (
            body
            for body in root.iter("body")
            if body.attrib.get("name") == "nut_root_body"
        ),
        None,
    )
    if nut is not None:
        existing_inertial = nut.find("inertial")
        if existing_inertial is not None:
            nut.remove(existing_inertial)
        nut.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "0.04",
                    "diaginertia": "1.3e-6 1.3e-6 2.1e-6",
                },
            ),
        )
        changed += 1
        position = nut.attrib.get("pos", "0 0 0").split()
        if len(position) == 3:
            position[2] = f"{_B02_ISAAC_NUT_ROOT_Z:.6f}"
            nut.attrib["pos"] = " ".join(position)
            nut.attrib["quat"] = "1 0 0 0"
            changed += 1
        radial_half = 0.5 * (
            _B02_NUT_RING_OUTER_APOTHEM - _B02_NUT_RING_INNER_RADIUS
        )
        radial_center = 0.5 * (
            _B02_NUT_RING_OUTER_APOTHEM + _B02_NUT_RING_INNER_RADIUS
        )
        tangential_half = 0.5 * _B02_NUT_RING_OUTER_RADIUS
        collisions = sorted(
            (
                geom
                for geom in nut.iter("geom")
                if geom.attrib.get("name", "").startswith("b02_nut_collision_")
            ),
            key=lambda geom: geom.attrib.get("name", ""),
        )
        for index, collision in enumerate(collisions):
            angle = index * math.pi / 3.0
            collision.attrib.pop("mesh", None)
            collision.attrib.update(
                {
                    "type": "box",
                    "size": (
                        f"{radial_half:.7f} {tangential_half:.7f} "
                        f"{_B02_NUT_RING_HALF_HEIGHT:.7f}"
                    ),
                    "pos": (
                        f"{radial_center * math.cos(angle):.7f} "
                        f"{radial_center * math.sin(angle):.7f} 0"
                    ),
                    "quat": (
                        f"{math.cos(0.5 * angle):.7f} 0 0 "
                        f"{math.sin(0.5 * angle):.7f}"
                    ),
                }
            )
            changed += 1
    return changed


def _apply_c06_stable_camera_reset_pose(root: ET.Element, task_name: str) -> int:
    """Start C06 upright on the camera body's flat face, without a drop."""
    if task_name not in {"t23_photograph_logo", xy3.TASK_NAME}:
        return 0
    camera = next(
        (
            body
            for body in root.iter("body")
            if body.attrib.get("name") == "link_19"
            and body.find("freejoint[@name='camera_root']") is not None
        ),
        None,
    )
    if camera is None:
        return 0
    position = camera.attrib.get("pos", "0 0 0").split()
    if len(position) != 3:
        return 0
    position[2] = f"{_C06_ISAAC_CAMERA_ROOT_Z:.6f}"
    camera.attrib["pos"] = " ".join(position)
    camera.attrib["quat"] = "1 0 0 0"
    return 1


def _apply_b04_analytic_small_block_collisions(
    root: ET.Element,
    task_name: str,
) -> int:
    """Use exact boxes for B04 cubes instead of noisy scanned mesh hulls."""
    if task_name != "t44_stack_blocks":
        return 0

    repaired = 0
    for body in root.iter("body"):
        name = body.attrib.get("name", "")
        half_extent = _B04_SMALL_BLOCK_HALF_EXTENTS.get(name)
        if half_extent is None:
            continue
        collision = body.find(f"geom[@name='{name}_collision_1']")
        if collision is None:
            continue
        collision.attrib.pop("mesh", None)
        collision.attrib.update(
            {
                "type": "box",
                "size": f"{half_extent:.6f} {half_extent:.6f} {half_extent:.6f}",
                "mass": "0.02",
                "condim": "6",
                "friction": "4.0 0.08 0.008",
            }
        )
        repaired += 1
    return repaired


def _remove_b04_wood_block_visual_collision(
    root: ET.Element,
    task_name: str,
) -> int:
    """Drop B04's transparent render helper; the stable box remains collision."""
    if task_name != "t44_stack_blocks":
        return 0

    for body in root.iter("body"):
        if body.attrib.get("name") != "036_wood_block":
            continue
        helper = body.find("geom[@name='036_wood_block_collision_1']")
        if helper is None:
            return 0
        # This geom is deliberately classed as visual and has no physical
        # role.  Isaac's MJCF importer places it below /visuals and renders it
        # opaque/black despite rgba=0.  The analytic stable collision geom is
        # kept untouched for all contact and mass behavior.
        body.remove(helper)
        return 1
    return 0


def _apply_ycb01_physical_truth(root: ET.Element, task_name: str) -> int:
    """Use source-size YCB geometry with deterministic physical properties."""
    if task_name != "t34_roll_ball_through_gate":
        return 0

    # Masses are the corresponding physical YCB object masses.  The inertias
    # use the unscaled source mesh bounds so Isaac never falls back to a
    # density-derived sentinel when importing the free bodies.
    body_specs = {
        "tennis_ball": (0.058, (0.0330205, 0.0330205, 0.0330205), True),
        "004_sugar_box": (0.514, (0.0445, 0.0190, 0.0875), False),
        "003_cracker_box": (0.411, (0.079906, 0.0299465, 0.1063595), False),
    }
    repaired = 0
    for body in root.iter("body"):
        name = body.attrib.get("name", "")
        spec = body_specs.get(name)
        if spec is None:
            continue
        mass, half_extents, is_sphere = spec
        hx, hy, hz = half_extents
        if is_sphere:
            diagonal = (0.4 * mass * hx * hx,) * 3
        else:
            diagonal = (
                mass * (hy * hy + hz * hz) / 3.0,
                mass * (hx * hx + hz * hz) / 3.0,
                mass * (hx * hx + hy * hy) / 3.0,
            )
        old_inertial = body.find("inertial")
        if old_inertial is not None:
            body.remove(old_inertial)
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "quat": "1 0 0 0",
                    "mass": f"{mass:.9f}",
                    "diaginertia": " ".join(f"{value:.12g}" for value in diagonal),
                },
            ),
        )
        if is_sphere:
            collision = body.find("geom[@name='tennis_ball_collision_1']")
            if collision is None:
                continue
            collision.attrib.pop("mesh", None)
            collision.attrib.update(
                {
                    "type": "sphere",
                    "size": "0.0330205",
                    "mass": "0.058",
                    "condim": "6",
                    "friction": "0.8 0.01 0.001",
                }
            )
        else:
            # The source convex hulls have slightly uneven support faces.  At
            # some yaw angles PhysX resolves that scan noise as an initial tip,
            # which can move a gate post before the operator acts.  A box made
            # from the measured, unscaled YCB bounds is the exact physical
            # envelope of these carton-shaped objects and gives every reset a
            # flat support face without changing their visual assets or size.
            collision = body.find(f"geom[@name='{name}_collision_1']")
            if collision is None:
                continue
            collision.attrib.pop("mesh", None)
            collision.attrib.update(
                {
                    "type": "box",
                    "size": f"{hx:.9g} {hy:.9g} {hz:.9g}",
                    "mass": f"{mass:.9g}",
                    "condim": "6",
                    "friction": "0.8 0.01 0.001",
                }
            )
        repaired += 1
    return repaired


def _apply_a01_analytic_fruit_collisions(root: ET.Element, task_name: str) -> int:
    """Replace noisy scanned fruit hulls with deterministic convex primitives."""
    if task_name not in {"t26_place_fruit_in_bucket", xy3.TASK_NAME}:
        return 0

    repaired = 0
    for body in root.iter("body"):
        name = body.attrib.get("name", "")
        spec = _A01_ANALYTIC_FRUIT_COLLISIONS.get(name)
        if spec is None:
            continue
        collisions = [
            geom
            for geom in body.findall("geom")
            if geom.attrib.get("name", "").startswith(f"{name}_collision_")
        ]
        if not collisions:
            continue
        primary, *obsolete = collisions
        primary.attrib.pop("mesh", None)
        primary.attrib.pop("pos", None)
        primary.attrib.update(
            {
                **spec,
                "condim": "6",
                "friction": "1.2 0.04 0.004",
            }
        )
        # Isaac classifies non-colliding MJCF geoms as render geometry.  Merely
        # disabling these obsolete hulls therefore puts them under ``visuals``
        # with an opaque black fallback material, which covers the real fruit
        # mesh (most visibly the apple).  The analytic primary fully replaces
        # them, so omit the stale hull geoms from the Isaac-only MJCF.
        for collision in obsolete:
            body.remove(collision)

        repaired += 1

    # Do not leave mass properties to the importer: it does not preserve
    # per-geom MJCF masses for these free bodies and authors invalid USD
    # sentinels instead.  This includes the bucket, whose geometry stays
    # unchanged.  Replacing inherited inertials makes the prepared MJCF itself
    # a complete, independently reproducible source.
    for body in root.iter("body"):
        source_inertial = _A01_ANALYTIC_FRUIT_INERTIALS.get(
            body.attrib.get("name", "")
        )
        if source_inertial is None:
            continue
        existing_inertial = body.find("inertial")
        if existing_inertial is not None:
            body.remove(existing_inertial)
        body.insert(
            0,
            ET.Element(
                "inertial",
                dict(source_inertial),
            ),
        )
    return repaired


def _append_hollow_vessel_collision(
    body: ET.Element,
    *,
    prefix: str,
    half_width: float,
    bottom_z: float,
    top_z: float,
    total_mass: float,
    friction: str = "1.1 0.03 0.002",
    wall_radial_insets: dict[int, float] | None = None,
    wall_top_z_overrides: dict[int, float] | None = None,
) -> None:
    """Create a PhysX-stable open vessel from a bottom and eight wall panels."""
    wall_thickness = 0.003
    side_half_length = half_width * 0.72
    common = {
        "group": "3",
        "condim": "6",
        "priority": "3",
        "friction": friction,
        "solref": "0.004 1",
        "solimp": "0.97 0.995 0.001",
        "rgba": "0 0 0 0",
    }
    bottom = ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": f"{prefix}_hollow_bottom",
            "type": "cylinder",
            "size": f"{half_width - 0.004:.6f} 0.003",
            "pos": f"0 0 {bottom_z + 0.003:.6f}",
            "mass": f"{total_mass * 0.36:.6f}",
        },
    )
    bottom.attrib.pop("priority", None)
    for index, angle in enumerate(range(0, 360, 45)):
        wall_top_z = float(
            (wall_top_z_overrides or {}).get(index, top_z)
        )
        if wall_top_z <= bottom_z:
            continue
        wall_half_height = (wall_top_z - bottom_z) * 0.5
        wall_center_z = (wall_top_z + bottom_z) * 0.5
        radians = math.radians(angle)
        radial_inset = float((wall_radial_insets or {}).get(index, 0.0))
        radius = half_width - wall_thickness - radial_inset
        x = radius * math.cos(radians)
        y = radius * math.sin(radians)
        ET.SubElement(
            body,
            "geom",
            {
                **common,
                "name": f"{prefix}_hollow_wall_{index}",
                "type": "box",
                "size": (
                    f"{wall_thickness:.6f} {side_half_length:.6f} "
                    f"{wall_half_height:.6f}"
                ),
                "pos": f"{x:.6f} {y:.6f} {wall_center_z:.6f}",
                "euler": f"0 0 {radians:.8f}",
                "mass": f"{total_mass * 0.08:.6f}",
            },
        )


def _apply_c07_hollow_vessel_collision_repair(
    root: ET.Element,
    task_name: str,
) -> None:
    """Open the 019 handle aperture without replacing either YCB vessel.

    The source decomposition already contains a hollow pitcher, a physical
    three-piece handle, an open spout, and a hollow target mug.  Only its rear
    body panel intrudes into the visible handle aperture.  Move that one panel
    six millimetres toward the vessel interior and disable the B01-only lid
    seat.  Keeping every other source collider is important: fingers must be
    able to grasp the real handle and particles must leave through the real
    spout under gravity rather than through a task-specific proxy channel.
    """
    if task_name != "c07_pitcher_pour_cup":
        return
    bodies = {
        body.attrib.get("name", ""): body
        for body in root.iter("body")
    }
    pitcher = bodies.get("019_pitcher_base")
    mug = bodies.get("025_mug")
    if pitcher is None or mug is None:
        return

    rear_wall = _geom_named(pitcher, "019_pitcher_base_collision_13")
    if rear_wall is not None:
        rear_wall.attrib["pos"] = "0 0.006 0"

    lid_seat = _geom_named(pitcher, "pitcher_base_lid_seat")
    if lid_seat is not None:
        lid_seat.attrib.update(
            {
                "contype": "0",
                "conaffinity": "0",
                "group": "4",
                "mass": "0",
            }
        )
    for index, angle in enumerate(range(0, 360, 45)):
        radians = math.radians(angle)
        ET.SubElement(
            mug,
            "geom",
            {
                "name": f"c07_cup_goal_rim_{index}",
                "type": "capsule",
                "fromto": (
                    f"{0.034 * math.cos(radians):.6f} "
                    f"{0.034 * math.sin(radians):.6f} 0.043 "
                    f"{0.034 * math.cos(radians + math.pi / 4):.6f} "
                    f"{0.034 * math.sin(radians + math.pi / 4):.6f} 0.043"
                ),
                "size": "0.0018",
                "rgba": "0.05 0.95 0.22 0",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "mass": "0",
            },
        )


def _apply_task_body_position_overrides(
    root: ET.Element,
    task_name: str,
    *,
    first_physx_safe: bool = False,
) -> None:
    positions = (
        D_TASK_FIRST_PHYSX_POSITION_OVERRIDES.get(task_name)
        if first_physx_safe
        else _TASK_BODY_POSITION_OVERRIDES.get(task_name)
    )
    quaternions = D_TASK_BODY_QUATERNION_OVERRIDES.get(task_name)
    if positions is None and quaternions is None:
        return
    for body in root.iter("body"):
        name = body.attrib.get("name", "")
        position = positions.get(name) if positions is not None else None
        if position is not None:
            body.attrib["pos"] = " ".join(f"{value:.6f}" for value in position)
        quaternion = quaternions.get(name) if quaternions is not None else None
        if quaternion is not None:
            body.attrib["quat"] = " ".join(
                f"{value:.9f}" for value in quaternion
            )
            body.attrib.pop("euler", None)


def _geom_named(body: ET.Element | None, name: str) -> ET.Element | None:
    if body is None:
        return None
    for geom in body.findall("geom"):
        if geom.attrib.get("name") == name:
            return geom
    return None


def _apply_04_cover_pitcher_collision_repair(
    root: ET.Element, task_name: str
) -> None:
    if task_name not in {"t04_cover_pitcher", xy5.TASK_NAME}:
        return

    bodies = {body.attrib.get("name", ""): body for body in root.iter("body")}

    base_seat = _geom_named(bodies.get("019_pitcher_base"), "pitcher_base_lid_seat")
    if base_seat is not None:
        # The MuJoCo source uses a solid disk as a support shim. In PhysX it
        # blocks the lid insert, so keep the real cup-wall meshes as collision.
        base_seat.attrib.update(
            {
                "contype": "0",
                "conaffinity": "0",
                "group": "4",
                "mass": "0",
            }
        )

    lid_body = bodies.get("pitcher_lid")
    oversized_insert = _geom_named(lid_body, "pitcher_lid_collision_1")
    if oversized_insert is not None:
        oversized_insert.attrib.update(
            {
                "contype": "0",
                "conaffinity": "0",
                "group": "4",
                "mass": "0",
            }
        )

    insert = _geom_named(lid_body, "pitcher_lid_bottom_seat")
    if insert is None and lid_body is not None:
        insert = ET.SubElement(lid_body, "geom", {"name": "pitcher_lid_bottom_seat"})
    if insert is not None:
        insert.attrib.update(
            {
                "type": "cylinder",
                "size": (
                    f"{_B01_ISAAC_LID_INSERT_RADIUS:.3f} "
                    f"{_B01_ISAAC_LID_INSERT_HALF_HEIGHT:.3f}"
                ),
                "pos": f"0 0 {_B01_ISAAC_LID_INSERT_CENTER_Z:.7f}",
                "rgba": "0 0 0 0",
                "group": "3",
                "mass": "0",
                "condim": "6",
                "priority": "4",
                "friction": "1.2 0.02 0.001",
                "solref": "0.0015 1",
                "solimp": "0.995 0.999 0.0001",
            }
        )


def _apply_ycb04_task_filter(root: ET.Element, task_name: str) -> int:
    """Keep only YCB04's three cups from the shared YCB bootstrap scene."""
    if task_name != "t41_sort_nested_cups":
        return 0

    removed_names = set(_YCB_SHARED_OBJECTS)
    removed_names.add("bucket")
    removed_references = set(removed_names)
    removed = 0
    for worldbody in root.findall("worldbody"):
        retained: list[ET.Element] = []
        for element in worldbody:
            if (
                element.tag != "body"
                or element.attrib.get("name") not in removed_names
            ):
                retained.append(element)
                continue
            removed += 1
            removed_references.update(
                descendant.attrib["name"]
                for descendant in element.iter()
                if descendant.tag in {"joint", "freejoint"}
                and "name" in descendant.attrib
            )
        worldbody[:] = retained

    _remove_elements_referencing(root, removed_references)
    _prune_unused_asset_definitions(root)
    return removed


def _replace_annular_rim_with_capsules(
    body: ET.Element,
    *,
    old_rim_name: str,
    new_rim_prefix: str,
    geom_class: str,
    segment_count: int,
    source_rim_mass: float,
    collision_spec: dict[str, object],
) -> None:
    """Replace one concave annular mesh with an importer-safe hollow ring."""

    old_rim = body.find(f"geom[@name='{old_rim_name}']")
    if old_rim is None:
        raise RuntimeError(f"Cup rim collider {old_rim_name!r} is missing")
    old_index = list(body).index(old_rim)
    body.remove(old_rim)

    segment_angle = 2.0 * math.pi / segment_count
    half_angle_cosine = math.cos(0.5 * segment_angle)
    inner_radius = float(collision_spec["inner_radius"])
    outer_radius = float(collision_spec["outer_radius"])
    centerline_radius = (
        outer_radius + inner_radius
    ) / (1.0 + half_angle_cosine)
    capsule_radius = outer_radius - centerline_radius
    if capsule_radius <= 0.0:
        raise RuntimeError(f"Cup rim dimensions for {old_rim_name!r} are invalid")
    segment_mass = source_rim_mass / segment_count
    center_z = float(collision_spec["center_z"])
    for segment_index in range(segment_count):
        angle_a = segment_index * segment_angle
        angle_b = (segment_index + 1) * segment_angle
        endpoint_a = (
            centerline_radius * math.cos(angle_a),
            centerline_radius * math.sin(angle_a),
            center_z,
        )
        endpoint_b = (
            centerline_radius * math.cos(angle_b),
            centerline_radius * math.sin(angle_b),
            center_z,
        )
        body.insert(
            old_index + segment_index,
            ET.Element(
                "geom",
                {
                    "name": f"{new_rim_prefix}{segment_index + 1:02d}",
                    "class": geom_class,
                    "type": "capsule",
                    "fromto": " ".join(
                        f"{value:.9f}"
                        for value in (*endpoint_a, *endpoint_b)
                    ),
                    "size": f"{capsule_radius:.9f}",
                    "mass": f"{segment_mass:.12f}",
                },
            ),
        )

    # Give the importer the exact source-compiled body inertia.  This keeps
    # the analytic collision substitution from changing how a cup falls or
    # responds to a grasp.
    inertial = body.find("inertial")
    if inertial is None:
        inertial = ET.Element("inertial")
        body.insert(0, inertial)
    inertial.attrib.update(
        {
            "mass": f"{float(collision_spec['mass']):.17g}",
            "pos": " ".join(
                f"{float(value):.17g}"
                for value in collision_spec["center_of_mass"]
            ),
            "quat": " ".join(
                f"{float(value):.17g}"
                for value in collision_spec["principal_axes"]
            ),
            "diaginertia": " ".join(
                f"{float(value):.17g}"
                for value in collision_spec["diagonal_inertia"]
            ),
        }
    )


def _apply_ycb02_hollow_c_rim_collision(
    root: ET.Element,
    task_name: str,
) -> int:
    """Keep the C-cup mouth open after Isaac's convex mesh import."""

    if task_name != "t27_place_largest_fitting_ball":
        return 0
    body = root.find(".//body[@name='c_cups']")
    if body is None:
        raise RuntimeError("YCB02 C cup body is missing")
    _replace_annular_rim_with_capsules(
        body,
        old_rim_name="c_cups_collision_27",
        new_rim_prefix="c_cups_collision_rim_",
        geom_class="ycb02_collision",
        segment_count=_YCB02_RIM_SEGMENT_COUNT,
        source_rim_mass=_YCB02_C_RIM_SOURCE_MASS,
        collision_spec=_YCB02_C_CUP_COLLISION_SPEC,
    )
    _prune_unused_asset_definitions(root)
    return 1


def _apply_ycb04_hollow_rim_collisions(root: ET.Element, task_name: str) -> int:
    """Replace PhysX-solid cup-mouth hulls with lightweight hollow rings.

    Collider27 is a concave annular mesh.  The MJCF importer requests a convex
    collision approximation, which fills its center and makes every nested
    inner cup overlap the outer cup at startup.  Twelve analytic capsule
    chords preserve the visible lip's outer radius while leaving a deliberate
    sub-millimetre collision clearance around the nested cup wall.
    """
    if task_name != "t41_sort_nested_cups":
        return 0

    repaired = 0
    for size_name, spec in _YCB04_CUP_COLLISION_SPECS.items():
        body = root.find(f".//body[@name='ycb04_{size_name}_cup']")
        if body is None:
            raise RuntimeError(f"YCB04 {size_name} cup body is missing")
        _replace_annular_rim_with_capsules(
            body,
            old_rim_name=f"ycb04_{size_name}_collision_27",
            new_rim_prefix=f"ycb04_{size_name}_collision_rim_",
            geom_class="ycb04_collision",
            segment_count=_YCB04_RIM_SEGMENT_COUNT,
            source_rim_mass=_YCB04_SOURCE_RIM_MASS,
            collision_spec=spec,
        )
        repaired += 1

    # Author a supported stack before the first PhysX reset, not in a later
    # settle/reset hook.  Only this prepared geometry has the hollow lips;
    # lowering the unmodified source cups would intersect their filled hulls.
    large_body = root.find(".//body[@name='ycb04_large_cup']")
    large_z = float(large_body.attrib["pos"].split()[2])
    for size_name, offset in _YCB04_NESTED_ROOT_Z_OFFSETS_M.items():
        body = root.find(f".//body[@name='ycb04_{size_name}_cup']")
        position = body.attrib["pos"].split()
        position[2] = f"{large_z + offset:.9f}"
        body.set("pos", " ".join(position))
    return repaired


def _prepare_included_mjcf(
    source: Path,
    output_dir: Path,
    robot_replacements: dict[str, Path],
    prepared: dict[Path, Path],
    *,
    output_name: str | None = None,
) -> Path:
    """Recursively rewrite one include tree to a shared, path-stable cache."""
    source = source.resolve()
    if source in prepared:
        return prepared[source]

    output_path = output_dir / (output_name or _cached_include_name(source))
    # Register before recursion so even an accidental include cycle terminates.
    prepared[source] = output_path

    root = ET.parse(source).getroot()
    root.insert(0, ET.Comment(f"HandsOn Isaac task migration v{_TASK_MJCF_CACHE_VERSION}"))
    compiler = root.find("compiler")
    mesh_dir = source.parent
    texture_dir = source.parent
    if compiler is not None:
        if compiler.attrib.get("meshdir"):
            mesh_dir = (source.parent / compiler.attrib.pop("meshdir")).resolve()
        if compiler.attrib.get("texturedir"):
            texture_dir = (source.parent / compiler.attrib.pop("texturedir")).resolve()

    for include in root.iter("include"):
        include_name = Path(include.attrib.get("file", "")).name
        replacement = robot_replacements.get(include_name)
        if replacement is None:
            included_source = (source.parent / include.attrib["file"]).resolve()
            replacement = _prepare_included_mjcf(
                included_source,
                output_dir,
                robot_replacements,
                prepared,
            )
        include.attrib["file"] = os.path.relpath(replacement, output_dir)

    for element in root.iter():
        if element.tag == "include":
            continue
        file_name = element.attrib.get("file")
        if not file_name:
            continue
        path = Path(file_name)
        if not path.is_absolute():
            if element.tag == "mesh":
                base_dir = mesh_dir
            elif element.tag == "texture":
                base_dir = texture_dir
            else:
                base_dir = source.parent
            path = (base_dir / path).resolve()
        element.attrib["file"] = os.path.relpath(path, output_dir)

    # Isaac keeps empty MuJoCo frame bodies as separate rigid links. Give those
    # wrappers, and invisible mocap targets, valid negligible mass properties.
    for body in root.iter("body"):
        if body.find("inertial") is not None:
            continue
        if body.find("geom") is None or body.attrib.get("mocap") == "true":
            body.insert(
                0,
                ET.Element(
                    "inertial",
                    {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-12 1e-12 1e-12"},
                ),
            )

    _inline_cached_includes(root, output_dir)
    _coalesce_top_level_sections(root)
    _add_missing_fixed_body_inertials(root)
    _alias_duplicate_mesh_sources(root, output_dir)
    _alias_texture_sources(root, output_dir)

    return _write_xml_if_changed(root, output_path)


def _prepare_bimanual_robot_mjcf(hand_type: str = "wuji1") -> Path:
    normalized = str(hand_type or "wuji1").strip().lower()
    if normalized not in {"wuji1", "wuji2"}:
        raise ValueError(
            "Isaac task robot supports only hand_type=wuji1 or wuji2; "
            f"got {hand_type!r}."
        )
    source = _XML_DIR / f"rokae_{normalized}_bimanual.xml"
    # MuJoCo resolves mesh paths from the outermost including file, so task and
    # robot derivatives must share one directory.
    output_dir = _CACHE_DIR / "mjcf"
    output_dir.mkdir(parents=True, exist_ok=True)

    replacements = {
        f"rokae_{normalized}_right.xml": _prepare_robot_mjcf(
            "right",
            preserve_base_pose=True,
            hand_type=normalized,
        ),
        f"rokae_{normalized}_left.xml": _prepare_robot_mjcf(
            "left",
            preserve_base_pose=True,
            hand_type=normalized,
        ),
    }
    return _prepare_included_mjcf(
        source,
        output_dir,
        replacements,
        {},
        output_name=f"rokae_{normalized}_bimanual_scene_isaac.xml",
    )


def _prepare_task_only_robot_placeholder(output_dir: Path) -> Path:
    """Replace a source robot include when collection spawns its own robots.

    The full task MJCF normally contains a bimanual robot.  Collection then
    spawns the two controlled articulations separately and deactivates the
    imported copy in USD.  Keeping those inactive articulations still makes
    PhysX allocate their GPU state during the first reset, which can exhaust
    the collision allocator before an episode starts.  An empty MJCF include
    preserves the source include structure without importing a second pair of
    robots; the collection scene remains responsible for spawning the real,
    collidable pair.
    """
    output_path = output_dir / "collection_task_only_robot_placeholder.xml"
    root = ET.Element("mujoco", {"model": "handson_collection_task_only"})
    root.insert(
        0,
        ET.Comment(
            f"HandsOn task-only collection include v{_TASK_MJCF_CACHE_VERSION}"
        ),
    )
    return _write_xml_if_changed(root, output_path)


def _validate_dlong_task_mjcf_contract(
    root: ET.Element,
    task_name: str,
    *,
    source_path: Path | None = None,
) -> None:
    """Fail closed if a task-only D-long asset loses its radian contract."""

    if task_name not in D_LONG_TASK_NAMES:
        return
    errors: list[str] = []
    compiler = root.find("compiler")
    angle = (
        str(compiler.attrib.get("angle", "degree")).strip().lower()
        if compiler is not None
        else "degree"
    )
    if angle != "radian":
        errors.append(f"compiler angle={angle!r}, expected 'radian'")

    expected_joint_ranges = {
        "t31_prepare_coffee": {
            "d09_cabinet_door_joint": (-1.45, 0.0),
            "d09_cabinet_right_door_joint": (0.0, 1.45),
        },
        "d10_load_dishwasher": {
            "d10_door_joint": (-1.45, 0.0),
            "d10_rack_joint": (-0.20, 0.0),
        },
        **{name: {"d20_door_joint": (-1.6, 0.0), "d20_tongs_joint": (0.0, 0.5)} for name in _d20_stages.STAGE_TASKS},
        "d20_microwave_hot_dog_plating": {
            "d20_door_joint": (-1.6, 0.0),
            "d20_tongs_joint": (0.0, 0.5),
        },
    }.get(task_name, {})
    joints = {
        joint.attrib.get("name", ""): joint
        for joint in root.iter("joint")
        if joint.attrib.get("name")
    }
    for joint_name, expected in expected_joint_ranges.items():
        joint = joints.get(joint_name)
        raw_range = joint.attrib.get("range") if joint is not None else None
        try:
            joint_range = tuple(float(value) for value in raw_range.split())
        except (AttributeError, TypeError, ValueError):
            joint_range = ()
        if len(joint_range) != 2 or any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9)
            for actual, wanted in zip(joint_range, expected)
        ):
            errors.append(
                f"{joint_name} range={raw_range!r}, expected "
                f"'{expected[0]:g} {expected[1]:g}'"
            )

    if errors:
        location = f" at {source_path}" if source_path is not None else ""
        raise ValueError(
            "D-long task MJCF contract failed"
            f"{location}: {'; '.join(errors)}. Rebuild the task asset from "
            "the current source instead of using a stale cache."
        )


def _validate_d11_task_mjcf_contract(
    root: ET.Element,
    *,
    source_path: Path | None = None,
) -> None:
    """Reject a stale D11 MJCF before it can become an Isaac USD cache.

    D11 has had two materially different scene layouts in the wild.  A
    generic cache from the old layout can still be syntactically valid XML,
    so checking only that conversion succeeded is not enough.  Keep this
    contract intentionally small and semantic: the authored body anchors,
    real visual meshes, and coverage markers must all be present.
    """

    errors: list[str] = []
    bodies: dict[str, ET.Element] = {}
    for body in root.iter("body"):
        name = body.attrib.get("name")
        if name:
            bodies.setdefault(name, body)

    expected_positions = {
        "d11_pan": (-0.185, -0.510, 0.915),
        "d11_sponge": (-0.270, 0.380, 0.902),
    }
    for body_name, expected in expected_positions.items():
        body = bodies.get(body_name)
        if body is None:
            errors.append(f"missing body {body_name!r}")
            continue
        raw_position = body.attrib.get("pos")
        try:
            position = tuple(float(value) for value in raw_position.split())
        except (AttributeError, TypeError, ValueError):
            position = ()
        if len(position) != 3 or any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=2e-5)
            for actual, wanted in zip(position, expected)
        ):
            errors.append(
                f"{body_name} pos={raw_position!r}, expected "
                f"{' '.join(f'{value:.6f}' for value in expected)}"
            )

    pan = bodies.get("d11_pan")
    if pan is not None:
        raw_quat = pan.attrib.get("quat")
        try:
            quat = tuple(float(value) for value in raw_quat.split())
        except (AttributeError, TypeError, ValueError):
            quat = ()
        if len(quat) != 4 or any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=2e-5)
            for actual, wanted in zip(quat, (0.0, 0.0, 0.0, 1.0))
        ):
            errors.append(
                f"d11_pan quat={raw_quat!r}, expected '0 0 0 1'"
            )

    expected_visuals = {
        "d11_pan_visual": "d11_skillet_visual_mesh",
        "d11_sponge_visual": "d11_sponge_visual_mesh",
    }
    for body_name, geom_name in (
        ("d11_pan", "d11_pan_visual"),
        ("d11_sponge", "d11_sponge_visual"),
    ):
        body = bodies.get(body_name)
        geom = None
        if body is not None:
            geom = next(
                (
                    candidate
                    for candidate in body.iter("geom")
                    if candidate.attrib.get("name") == geom_name
                ),
                None,
            )
        expected_mesh = expected_visuals[geom_name]
        if geom is None:
            errors.append(f"missing visual geom {geom_name!r}")
            continue
        if geom.attrib.get("class") != "dlong_marker":
            errors.append(
                f"{geom_name} class={geom.attrib.get('class')!r}, "
                "expected 'dlong_marker'"
            )
        if geom.attrib.get("type") != "mesh":
            errors.append(
                f"{geom_name} type={geom.attrib.get('type')!r}, "
                "expected 'mesh'"
            )
        if geom.attrib.get("mesh") != expected_mesh:
            errors.append(
                f"{geom_name} mesh={geom.attrib.get('mesh')!r}, "
                f"expected {expected_mesh!r}"
            )
        expected_material = (
            "d11_skillet_material"
            if geom_name == "d11_pan_visual"
            else "d11_sponge_material"
        )
        if geom.attrib.get("material") != expected_material:
            errors.append(
                f"{geom_name} material={geom.attrib.get('material')!r}, "
                f"expected {expected_material!r}"
            )

    pan_visual = (
        pan.find("geom[@name='d11_pan_visual']")
        if pan is not None
        else None
    )
    raw_euler = pan_visual.attrib.get("euler") if pan_visual is not None else None
    try:
        visual_euler = tuple(float(value) for value in raw_euler.split())
    except (AttributeError, TypeError, ValueError):
        visual_euler = ()
    if len(visual_euler) != 3 or any(
        not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=2e-9)
        for actual, wanted in zip(visual_euler, (0.0, 0.0, -math.pi / 2.0))
    ):
        errors.append(
            f"d11_pan_visual euler={raw_euler!r}, expected '0 0 -pi/2'"
        )

    for index in range(4):
        geom_name = f"d11_pan_coverage_{index}"
        geom = next(
            (
                candidate
                for candidate in (pan.iter("geom") if pan is not None else ())
                if candidate.attrib.get("name") == geom_name
            ),
            None,
        )
        if geom is None:
            errors.append(f"missing coverage geom {geom_name!r}")
        else:
            if geom.attrib.get("class") != "dlong_marker":
                errors.append(
                    f"{geom_name} class={geom.attrib.get('class')!r}, "
                    "expected 'dlong_marker'"
                )
            if geom.attrib.get("type") != "box":
                errors.append(
                    f"{geom_name} type={geom.attrib.get('type')!r}, "
                    "expected 'box'"
                )
            if geom.attrib.get("material") != "dlong_water":
                errors.append(
                    f"{geom_name} material={geom.attrib.get('material')!r}, "
                    "expected 'dlong_water'"
                )

    if errors:
        location = f" at {source_path}" if source_path is not None else ""
        details = "; ".join(errors)
        raise ValueError(
            "D11 task MJCF contract failed"
            f"{location}: {details}. Rebuild the task asset from the "
            "current source instead of using a stale cache."
        )


def prepare_task_mjcf(
    task_name: str,
    *,
    first_physx_safe: bool = False,
    hand_type: str = "wuji1",
) -> Path:
    """Create a cached full-scene MJCF with Isaac-friendly hand collisions."""
    _ensure_optional_task_metadata(task_name)
    try:
        spec = TASK_SPECS[task_name]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unsupported Isaac task {task_name!r}; expected one of: {supported}") from exc

    normalized_hand_type = str(hand_type or "wuji1").strip().lower()
    source = spec.source_path_for_hand(normalized_hand_type)
    if not source.is_file():
        raise FileNotFoundError(f"Missing task MJCF: {source}")

    output_dir = _CACHE_DIR / "mjcf"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_only_robot = bool(
        (first_physx_safe and task_name in COLLECTION_TASK_ONLY_ROBOT_TASKS)
        or task_name == "p003_bimanual_folding_ruler_shape_configuration"
    )
    if task_only_robot:
        bimanual_replacement = _prepare_task_only_robot_placeholder(output_dir)
    elif normalized_hand_type == "wuji1":
        # Preserve the original no-argument call contract used by cache tools
        # and downstream scene customizers.
        bimanual_replacement = _prepare_bimanual_robot_mjcf()
    else:
        bimanual_replacement = _prepare_bimanual_robot_mjcf(
            normalized_hand_type
        )
    robot_replacements = {
        f"rokae_{normalized_hand_type}_bimanual.xml": bimanual_replacement,
        f"rokae_{normalized_hand_type}_right.xml": _prepare_robot_mjcf(
            "right",
            preserve_base_pose=True,
            hand_type=normalized_hand_type,
        ),
        f"rokae_{normalized_hand_type}_left.xml": _prepare_robot_mjcf(
            "left",
            preserve_base_pose=True,
            hand_type=normalized_hand_type,
        ),
    }
    hand_suffix = "" if normalized_hand_type == "wuji1" else "_wuji2"
    # The public task key may deliberately use lower-kebab-case.  Isaac's
    # MJCF converter derives its USD root prim from the input filename, and a
    # hyphen is illegal in a USD identifier.  Keep the public key everywhere
    # else and normalize only this converter-owned filename boundary.
    converter_task_slug = task_name.replace("-", "_")
    output_path = _prepare_included_mjcf(
        source,
        output_dir,
        robot_replacements,
        {},
        output_name=(
            f"{converter_task_slug}{hand_suffix}_scene_first_physx_safe_isaac.xml"
            if first_physx_safe
            else f"{converter_task_slug}{hand_suffix}_scene_isaac.xml"
        ),
    )
    root = ET.parse(output_path).getroot()
    if task_name == "eval_gen04_pass_glue_stick":
        from .gen4_scene import prepare_gen4_mjcf
        prepare_gen4_mjcf(root)
    if task_name == xy3.TASK_NAME:
        xy3_runtime.attach_logo(root)
    _apply_ycb06_original_asset_bodies(root, task_name)
    if task_name == "eval_comb03_pass_block_and_close_box":
        from .xy1_composition import prepare_scene
        prepare_scene(root)
    _apply_ycb_task_layout(root, task_name)
    _apply_a02_task_filter(root, task_name)
    _apply_b02_isaac_reset_repair(root, task_name)
    _apply_a01_analytic_fruit_collisions(root, task_name)
    _apply_ycb01_physical_truth(root, task_name)
    _apply_b04_analytic_small_block_collisions(root, task_name)
    _remove_b04_wood_block_visual_collision(root, task_name)
    _apply_c07_hollow_vessel_collision_repair(root, task_name)
    _apply_task_body_position_overrides(
        root,
        task_name,
        first_physx_safe=first_physx_safe,
    )
    _apply_c_tool_task_filter(root, task_name)
    _configure_ycb07_timer_source(root, task_name)
    _apply_hl015_marker_filter(root, task_name)
    apply_d_long_task_filter(root, task_name)
    if task_name == "d08_2_realistic_toast":
        _prune_unused_asset_definitions(root)
    _apply_ycb02_hollow_c_rim_collision(root, task_name)
    _apply_ycb04_hollow_rim_collisions(root, task_name)
    _apply_ycb04_task_filter(root, task_name)
    _remove_mocap_target_helper_geoms(root)
    _remove_c06_target_region_visual(root, task_name)
    _apply_04_cover_pitcher_collision_repair(root, task_name)
    if task_name == xy5.TASK_NAME:
        xy5.apply_authored_layout(root)
        _prune_unused_asset_definitions(root)
    _apply_c04_plant_texture_alpha(root, output_dir, task_name)
    _remove_c04_isaac_hidden_visuals(root, task_name)
    _apply_c04_stable_spray_root_collision(root, task_name)
    _apply_c04_spray_inertials(root, task_name)
    _apply_c05_mousepad_inertial(root, task_name)
    _apply_d05_scissors_auxiliary_inertial(root, task_name)
    _fix_c05_display(root, task_name)
    _remove_c06_camera_debug_visuals(root, task_name)
    _apply_c06_camera_auxiliary_inertials(root, task_name)
    _configure_c06_shutter_collision(root, task_name)
    _apply_c06_stable_camera_reset_pose(root, task_name)
    if task_name == xy3.TASK_NAME:
        xy3_runtime.author_initial_layout(root)
        _prune_unused_asset_definitions(root)
    _validate_dlong_task_mjcf_contract(
        root,
        task_name,
        source_path=output_path,
    )
    if task_name == "d11_presoak_pan":
        _validate_d11_task_mjcf_contract(root, source_path=output_path)
    if task_name == "t09_insert_peg":
        _cache_b03_external_assets(root, output_dir)
    return _write_xml_if_changed(root, output_path)


def validate_task_assets(
    task_name: str,
    *,
    hand_type: str = "wuji1",
) -> IsaacTaskSpec:
    _ensure_optional_task_metadata(task_name)
    try:
        spec = TASK_SPECS[task_name]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unsupported Isaac task {task_name!r}; expected one of: {supported}") from exc
    source_path = spec.source_path_for_hand(hand_type)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing task MJCF: {source_path}")
    if task_name == E01_TASK_NAME:
        resolve_garmentlab_asset_path(required=True)
    return spec


def repair_fixed_root_joints(root_path: str = "/World/Task") -> int:
    """Replace Isaac 5.1 MJCF world joints whose anchors ignore body poses."""
    import omni.usd
    from omni.physx.scripts import utils as physx_utils
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    replacements: list[tuple[str, object]] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.FixedJoint) or not prim.GetName().startswith("rootJoint_"):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = joint.GetBody1Rel().GetTargets() or joint.GetBody0Rel().GetTargets()
        if not targets:
            continue
        target_prim = stage.GetPrimAtPath(targets[0])
        if target_prim.IsValid():
            replacements.append((str(prim.GetPath()), target_prim))

    for index, (joint_path, target_prim) in enumerate(replacements):
        stage.GetPrimAtPath(joint_path).SetActive(False)
        physx_utils.createJoint(
            stage=stage,
            joint_type="Fixed",
            from_prim=None,
            to_prim=target_prim,
        )
    return len(replacements)


def apply_robot_joint_defaults(root_path: str = "/World/Task", task_name: str | None = None) -> int:
    """Author the same deterministic arm/hand state used by MuJoCo reset."""
    import math
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    arm_home = _task_arm_home(task_name)
    joint_positions = {
        f"joint{index}_{side}": value
        for side in ("right", "left")
        for index, value in enumerate(arm_home[side], start=1)
    }
    joint_positions.update(
        {
            f"{side}_finger1_joint1": 0.10
            for side in ("right", "left")
        }
    )

    authored = 0
    for prim in Usd.PrimRange(root):
        position = joint_positions.get(prim.GetName())
        if position is None or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        position_degrees = math.degrees(position)
        if prim.HasAPI(PhysxSchema.JointStateAPI, "angular"):
            state_api = PhysxSchema.JointStateAPI.Get(prim, "angular")
        else:
            state_api = PhysxSchema.JointStateAPI.Apply(prim, "angular")
        state_api.GetPositionAttr().Set(position_degrees)
        state_api.GetVelocityAttr().Set(0.0)

        drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive_api:
            target = drive_api.GetTargetPositionAttr()
            if not target or not target.IsValid():
                target = drive_api.CreateTargetPositionAttr()
            target.Set(position_degrees)
        authored += 1
    return authored


def _hide_imported_task_sites(stage, root_path: str) -> int:
    """Hide imported MJCF site geometry while preserving its task transforms."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0

    hidden = 0
    for prim in Usd.PrimRange(root):
        if prim.GetName() != "sites":
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
            hidden += 1
    return hidden


def _task_usd_cache_is_incomplete(usd_dir: Path, task_mjcf: Path) -> bool:
    """Detect interrupted Isaac converter output that would load as an empty task."""
    if not usd_dir.exists():
        return False
    cache_entries = tuple(usd_dir.iterdir())
    if not cache_entries:
        return True

    stem = task_mjcf.stem
    if is_lab3():
        # Sim 6's Newton-aware converter emits an asset directory with an USDA
        # interface and payload layers instead of Lab 2's flat USD layout.
        asset_dir = usd_dir / stem
        root_usd = asset_dir / f"{stem}.usda"
        base_usd = asset_dir / "payloads" / "base.usda"
        if any(
            path.is_dir() and path.name.startswith(f"{stem}_")
            for path in cache_entries
        ):
            # The Lab 3 importer chooses these suffixes when it believes an
            # output name is occupied.  Certification cannot infer which tree
            # its converter returned, so rebuild the content-keyed directory
            # instead of binding identity to a stale base asset.
            return True
    else:
        root_usd = usd_dir / f"{stem}.usd"
        base_usd = usd_dir / "configuration" / f"{stem}_base.usd"
    asset_hash = usd_dir / ".asset_hash"
    b03_content_addressed = bool(
        stem == "t09_insert_peg_scene_isaac"
        and len(usd_dir.name) == 64
        and all(character in "0123456789abcdef" for character in usd_dir.name)
    )
    return bool(
        (not b03_content_addressed and not asset_hash.is_file())
        or not root_usd.is_file()
        or root_usd.stat().st_size < _TASK_USD_MIN_ROOT_BYTES
        or not base_usd.is_file()
        or base_usd.stat().st_size < _TASK_USD_MIN_BASE_BYTES
    )


def _task_cached_usd_path(usd_dir: Path, task_mjcf: Path) -> Path:
    """Return the only USD path allowed in a content-addressed task cache."""

    stem = task_mjcf.stem
    if is_lab3():
        return usd_dir / stem / f"{stem}.usda"
    return usd_dir / f"{stem}.usd"


def task_usd_cache_dir(
    task_name: str,
    task_mjcf: Path | None = None,
    *,
    first_physx_safe: bool = False,
    hand_type: str = "wuji1",
) -> Path:
    """Return the exact converter cache directory used by a task scene."""

    normalized_hand_type = str(hand_type or "wuji1").strip().lower()
    if normalized_hand_type not in {"wuji1", "wuji2"}:
        raise ValueError(
            "Isaac task USD caches support only hand_type=wuji1 or wuji2; "
            f"got {hand_type!r}."
        )
    hand_suffix = "" if normalized_hand_type == "wuji1" else "_wuji2"
    cache_task_name = f"{task_name}{hand_suffix}"
    if _is_50_water_plant_task(task_name):
        cache_task_name += "_plant_visual_copies_v1"
    if task_mjcf is not None and b"optimized_dlong_visuals" in task_mjcf.read_bytes():
        # Included visual references changed; converter config hashes omit them.
        cache_task_name += "_dlong_visual_copies_v1"
    if task_name == "t23_photograph_logo":
        # The converter config hash omits included XML/OBJ contents. The camera
        # now binds separate optimized visuals while retaining collision sources.
        cache_task_name += "_camera_visual_copies_v1"
    if task_name == "d08_2_realistic_toast":
        from .d08_realistic_variant import asset_digest
        cache_task_name += f"_blender_{asset_digest()}"
    if task_name in _d20_stages.TASK_NAMES:
        # Isaac's converter hash does not track replaced OBJ bytes. In
        # particular, a corrected bottle base must not reuse an older USD
        # merely because its mesh filename and hull count stayed the same.
        from .d20_selected_assets import asset_content_digest
        cache_task_name += f"_selected_{asset_content_digest()[:20]}"
    if task_name == "t27_place_largest_fitting_ball":
        cache_task_name += f"_{_YCB02_FOUR_BALL_ROWS_USD_CACHE_REVISION}"
    if task_name == "t41_sort_nested_cups":
        cache_task_name += f"_{_YCB04_ISOLATED_CUPS_USD_CACHE_REVISION}"
    if task_name == _YCB07_TASK_NAME:
        # The converter cache key does not include the post-include source
        # promotion. Isolate YCB07 from any USD generated while the timer was
        # still nested under the exporter wrapper.
        cache_task_name += f"_{_YCB07_TIMER_USD_CACHE_REVISION}"
    if task_name == "t31_prepare_coffee":
        # The converter hash omits MJCF contents.  Isolate the corrected left
        # hinge, physical handle, back wall, and mug placement from stale D09
        # USDs without invalidating every unrelated task cache.
        cache_task_name += f"_{_D09_CABINET_USD_CACHE_REVISION}"
    if task_name == "t10_lift_carton":
        # BC004 now has a dedicated D08-aligned source scene.  The shared
        # converter metadata does not hash MJCF contents, so isolate it from
        # any v214 USD generated from the former common B320 scene.
        cache_task_name += f"_{_BC004_D08_ALIGNED_USD_CACHE_REVISION}"
    if task_name == "p004_hold_spring_clamp_open_insert_card":
        # P0-04 now uses both textured YCB clamp halves, all source convex
        # colliders, and the source hinge instead of a procedural primitive
        # substitute. Keep it out of USD caches containing the old geometry.
        cache_task_name += f"_{_P004_REAL_YCB_CLAMP_USD_CACHE_REVISION}"
    if task_name == "p005_key_insert_turn_unlock_open_shackle":
        # The converter hash omits included MJCF contents. Isolate the free
        # 1.2x task-local padlock/key geometry, baked key collision meshes, and
        # locked-until-turn shackle
        # from older robot-facing/fixed-lock caches.
        cache_task_name += f"_{_P005_BOTTOM_KEYWAY_FREE_PADLOCK_USD_CACHE_REVISION}"
    if task_name == "loaded_tray_through_narrow_gate":
        # BC005 v6 starts the selected tray at +90 degrees yaw, fits the frame
        # opening to the tray's long dimension, and moves the reachable route
        # rearward.  Isaac's converter hash omits included MJCF contents, so
        # this topology/layout needs a dedicated cache namespace.
        cache_task_name += f"_{_BC005_FORWARD_LAYOUT_USD_CACHE_REVISION}"
    if task_name == "bench320_hs014_bottle_stabilized_cap_opening":
        # HS014 removes every shared YCB rigid body before conversion, then
        # composes its exact bottle/cap layer.  The converter metadata does
        # not hash that pre-import filter, so isolate it from an older generic
        # YCB cache for the same scene XML.
        cache_task_name = f"{cache_task_name}_preimport_ycb_filter_v1"
    if task_name in {"z01_pipette", "z02_bottle_cap"}:
        cache_task_name += "_no_generic_target"
    if first_physx_safe:
        cache_task_name += "_first_physx_safe"
    cache_dir = (
        _CACHE_DIR
        / "usd"
        / "tasks"
        / f"v{_TASK_MJCF_CACHE_VERSION}"
        / cache_task_name
    )
    if task_name != "t09_insert_peg":
        return cache_dir
    if task_mjcf is None:
        raise ValueError(
            "B03 USD cache lookup requires the prepared task MJCF so the "
            "loaded asset tree is content-addressed."
        )
    return cache_dir / _task_mjcf_content_digest(task_mjcf)


def _task_stage_reference_is_loaded(stage, prim_path: str) -> bool:
    root = stage.GetPrimAtPath(prim_path)
    return bool(root.IsValid() and tuple(root.GetChildren()))


def _resolve_studio_hdri() -> Path | None:
    """Find a bundled neutral studio HDRI without depending on Nucleus."""

    requested = os.environ.get("HANDSON_ISAACSIM_STUDIO_HDRI", "").strip()
    if requested.lower() in {"off", "none", "0"}:
        return None
    if requested:
        explicit = Path(requested).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        print(
            "[HandsOn/IsaacLab] Studio HDRI was not found; using the "
            f"bundled fallback search: {explicit}",
            flush=True,
        )

    relative_patterns = (
        # Neutral automotive light-card environment: clean gray primary ray
        # plus strong rectangular values for readable plastic reflections.
        "isaacsim/extscache/omni.kit.stage_templates-*/data/CarLight_512x256.hdr",
        "isaacsim/extscache/omni.kit.widget.material_preview-*"
        "/data/domeLight/photo_studio_01_4k.hdr",
    )
    for relative_pattern in relative_patterns:
        for entry in sys.path:
            if not entry:
                continue
            site_packages = Path(entry)
            if not site_packages.is_dir():
                continue
            for candidate in sorted(site_packages.glob(relative_pattern)):
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _spawn_task_lighting(*, studio_lighting: bool) -> Path | None:
    """Spawn either the stable default rig or the B03 cinematic studio rig."""

    if not studio_lighting:
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=180.0,
            color=(0.92, 0.94, 1.0),
        )
        dome_light_cfg.func("/World/HandsOnDomeLight", dome_light_cfg)
        # A large camera-side key produces the broad satin highlight visible
        # on the photographed moulded covers without clipping them to white.
        key_light_cfg = sim_utils.SphereLightCfg(
            intensity=900.0,
            color=(1.0, 0.96, 0.90),
            radius=0.34,
            treat_as_point=False,
        )
        key_light_cfg.func(
            "/World/HandsOnKeyLight",
            key_light_cfg,
            translation=(-1.40, -0.02, 2.22),
        )
        # A weak, oversized opposite-side source keeps the white shell's edge
        # separation and makes its reflection readable from oblique cameras.
        fill_light_cfg = sim_utils.SphereLightCfg(
            intensity=260.0,
            color=(0.76, 0.84, 1.0),
            radius=0.44,
            treat_as_point=False,
        )
        fill_light_cfg.func(
            "/World/HandsOnFillLight",
            fill_light_cfg,
            translation=(0.32, 0.24, 1.82),
        )
        catch_light_cfg = sim_utils.SphereLightCfg(
            intensity=900.0,
            color=(1.0, 0.98, 0.94),
            radius=0.055,
            treat_as_point=False,
        )
        catch_light_cfg.func(
            "/World/HandsOnGlossCatchLight",
            catch_light_cfg,
            translation=(0.10, -0.30, 1.10),
        )
        return None

    hdri = _resolve_studio_hdri()
    dome_light_cfg = sim_utils.DomeLightCfg(
        intensity=150.0,
        color=(0.94, 0.97, 1.0),
        texture_file=str(hdri) if hdri is not None else None,
        texture_format="latlong",
        visible_in_primary_ray=True,
    )
    dome_light_cfg.func("/World/HandsOnDomeLight", dome_light_cfg)

    # A broad warm source paints real specular streaks on the arms and metal
    # parts; the directional source supplies readable Newton-like form and
    # contact shadows, while a cool rim separates the hands from the scene.
    key_light_cfg = sim_utils.SphereLightCfg(
        intensity=850.0,
        color=(1.0, 0.92, 0.82),
        radius=0.36,
        treat_as_point=False,
    )
    key_light_cfg.func(
        "/World/HandsOnKeyLight",
        key_light_cfg,
        translation=(-0.48, -0.42, 2.12),
    )
    directional_cfg = sim_utils.DistantLightCfg(
        intensity=320.0,
        color=(1.0, 0.96, 0.90),
        angle=2.0,
    )
    directional_cfg.func(
        "/World/HandsOnStudioDirectionalLight",
        directional_cfg,
        orientation=(0.8880738, 0.3250576, 0.3250576, 0.0),
    )
    rim_light_cfg = sim_utils.SphereLightCfg(
        intensity=420.0,
        color=(0.70, 0.82, 1.0),
        radius=0.30,
        treat_as_point=False,
    )
    rim_light_cfg.func(
        "/World/HandsOnStudioRimLight",
        rim_light_cfg,
        translation=(0.55, 0.24, 1.72),
    )
    # Small real light source positioned at the mirror angle of the orange
    # joint cap.  It creates a genuine RTX catch-light instead of a painted
    # white spot, so the highlight moves correctly with view and articulation.
    catch_light_cfg = sim_utils.SphereLightCfg(
        intensity=1050.0,
        color=(1.0, 0.98, 0.94),
        radius=0.055,
        treat_as_point=False,
    )
    catch_light_cfg.func(
        "/World/HandsOnGlossCatchLight",
        catch_light_cfg,
        translation=(0.10, -0.30, 1.10),
    )
    print(
        "[HandsOn/IsaacLab] B03 studio lighting enabled: "
        f"HDRI={hdri if hdri is not None else 'uniform dome'}, "
        "warm area key + directional key + cool rim",
        flush=True,
    )
    return hdri


def reapply_task_studio_lighting(stage) -> int:
    """Restore the deterministic studio rig after optional visual randomization."""

    from pxr import Gf

    light_values = {
        "/World/HandsOnDomeLight": {
            "inputs:intensity": 150.0,
            "inputs:color": Gf.Vec3f(0.94, 0.97, 1.0),
        },
        "/World/HandsOnKeyLight": {
            "inputs:intensity": 850.0,
            "inputs:color": Gf.Vec3f(1.0, 0.92, 0.82),
            "inputs:radius": 0.36,
            "xformOp:translate": Gf.Vec3d(-0.48, -0.42, 2.12),
        },
        "/World/HandsOnStudioDirectionalLight": {
            "inputs:intensity": 320.0,
            "inputs:color": Gf.Vec3f(1.0, 0.96, 0.90),
            "inputs:angle": 2.0,
        },
        "/World/HandsOnStudioRimLight": {
            "inputs:intensity": 420.0,
            "inputs:color": Gf.Vec3f(0.70, 0.82, 1.0),
            "inputs:radius": 0.30,
            "xformOp:translate": Gf.Vec3d(0.55, 0.24, 1.72),
        },
        "/World/HandsOnGlossCatchLight": {
            "inputs:intensity": 1050.0,
            "inputs:color": Gf.Vec3f(1.0, 0.98, 0.94),
            "inputs:radius": 0.055,
            "xformOp:translate": Gf.Vec3d(0.10, -0.30, 1.10),
        },
    }
    updated = 0
    for prim_path, attributes in light_values.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        for attribute_name, value in attributes.items():
            attribute = prim.GetAttribute(attribute_name)
            if attribute and attribute.IsValid() and attribute.Set(value):
                updated += 1
    return updated


def _set_fu006_reference_transform(
    prim,
    *,
    translate: tuple[float, float, float],
    yaw_degrees: float,
    scale: float,
) -> None:
    """Override a referenced root's authored op order with one stable stack."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    )
    orient_op = xform.AddOrientOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    )
    scale_op = xform.AddScaleOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    )
    translate_op.Set(Gf.Vec3d(*translate))
    orient_op.Set(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_degrees).GetQuat())
    scale_op.Set(Gf.Vec3d(scale, scale, scale))


_FU006_DRAWER_SCALE = 0.35
_FU006_ORIGINAL_DRAWER_TRANSLATION_M = (-0.07, 0.0, 1.1931573060)
_FU006_DRAWER_BACK_SHIFT_M = 0.10
_FU006_ORIGINAL_LOWER_CABINET_BOTTOM_SOURCE_Z = -0.851878
_FU006_ORIGINAL_LOWER_CABINET_TOP_SOURCE_Z = -0.103595
_FU006_DRAWER_SUPPORT_Z_M = (
    _FU006_ORIGINAL_DRAWER_TRANSLATION_M[2]
    + _FU006_DRAWER_SCALE * _FU006_ORIGINAL_LOWER_CABINET_BOTTOM_SOURCE_Z
)
_FU006_DRAWER_TRANSLATION_M = (
    _FU006_ORIGINAL_DRAWER_TRANSLATION_M[0] + _FU006_DRAWER_BACK_SHIFT_M,
    _FU006_ORIGINAL_DRAWER_TRANSLATION_M[1],
    _FU006_DRAWER_SUPPORT_Z_M - _FU006_DRAWER_SCALE * 0.32,
)
_FU006_CHARGER_TRANSLATION_M = (-0.37, -0.22, 0.90750407)

# M4nT1Core's CC BY 4.0 cabinet is Y-up.  The retained vendor articulation
# already maps local X/Y/Z to world -Y/+Z/-X, so the downloaded geometry can
# inherit the validated link orientation.  Non-uniform scale makes the
# two-drawer cabinet a low organizer while preserving its authored width,
# depth, materials, and independently modeled drawer nodes.
_FU006_MODERN_CABINET_GEOMETRY_ROOT = "Bedside_Table_fbx/RootNode"
_FU006_MODERN_CABINET_STATIC_DRAWER = "BedsideTableDrawer1"
_FU006_MODERN_CABINET_MOVING_DRAWER = "BedsideTableDrawer2"
_FU006_MODERN_CABINET_SOURCE_Y_MIN_M = -0.029986
_FU006_MODERN_CABINET_SOURCE_Y_MAX_M = 0.429595
_FU006_MODERN_CABINET_WORLD_SCALE_XYZ = (0.625, 0.42, 0.80)
_FU006_MODERN_CABINET_ORIGIN_Z_M = (
    _FU006_DRAWER_SUPPORT_Z_M
    - _FU006_MODERN_CABINET_WORLD_SCALE_XYZ[1]
    * _FU006_MODERN_CABINET_SOURCE_Y_MIN_M
)
# Exact source-frame proxy boxes measured from the selected lower-drawer
# meshes after they are aligned to link_1.  The retained articulation applies
# its 0.35 root scale to these boxes, so the authored collision and the visible
# downloaded drawer share the same moving frame without a non-uniformly scaled
# dynamic triangle mesh.
_FU006_MODERN_DRAWER_COLLISION_BOXES_SOURCE = {
    "left_wall": (
        (-0.357103258371, -0.173102885485, 0.159862652421),
        (-0.321471542120, -0.017719343305, 1.027594923973),
    ),
    "right_wall": (
        (0.409920603037, -0.173102885485, 0.159862652421),
        (0.445552319288, -0.017719343305, 1.027594923973),
    ),
    "back_wall": (
        (-0.357103258371, -0.173102885485, 0.159862652421),
        (0.445552319288, -0.017719343305, 0.205499693751),
    ),
    "front": (
        (-0.402398675680, -0.191131621599, 1.027594804764),
        (0.490847766399, 0.054243292660, 1.073250770569),
    ),
    "handle": (
        (-0.089235253632, -0.082089945674, 1.073254466057),
        (0.177743837237, -0.067091614008, 1.108387708664),
    ),
}


def _compose_fu006_assets(stage, prim_path: str) -> tuple[str, str] | None:
    """Compose the exact FU006 USD bundles before PhysX creates tensor views."""
    from pxr import UsdGeom

    task_name = "t46_store_charger"
    asset_root = _PACKAGED_XML_DIR / "assets" / "bench320_fu006"
    drawer_asset = asset_root / "dv_drawer_45575" / "45575.usd"
    charger_asset = asset_root / "dexverse_charger" / "model_charger.usd"
    modern_cabinet_asset = (
        asset_root
        / "modern_wooden_bedside_table_m4nt1core"
        / "modern_wooden_bedside_table.usd"
    )
    required_assets = (drawer_asset, charger_asset, modern_cabinet_asset)
    missing = [str(path) for path in required_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{task_name} exact USD bundle is incomplete: {', '.join(missing)}"
        )

    drawer_path = f"{prim_path}/fu006_drawer"
    charger_path = f"{prim_path}/fu006_charger"
    for path, asset in ((drawer_path, drawer_asset), (charger_path, charger_asset)):
        existing = stage.GetPrimAtPath(path)
        if existing.IsValid():
            stage.RemovePrim(path)
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(str(asset))

    _set_fu006_reference_transform(
        stage.GetPrimAtPath(drawer_path),
        translate=_FU006_DRAWER_TRANSLATION_M,
        yaw_degrees=0.0,
        scale=_FU006_DRAWER_SCALE,
    )
    _set_fu006_reference_transform(
        stage.GetPrimAtPath(charger_path),
        translate=_FU006_CHARGER_TRANSLATION_M,
        yaw_degrees=-12.0,
        scale=1.0,
    )

    joint_path = f"{drawer_path}/joints/joint_1"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise RuntimeError(f"{task_name} drawer joint is missing: {joint_path}")
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

    drawer_scale = _FU006_DRAWER_SCALE
    # The upstream asset leaves render-purpose collision copies visible and
    # coplanar with its visual meshes.  Hide only their Hydra representation;
    # USD/PhysX collision APIs remain active for simulation and evaluation.
    deinstanced_collision_groups = 0
    deinstanced_collision_instances = 0
    hidden_collision_render_prims = 0
    for link_name in ("base", "link_0", "link_1", "link_2"):
        collision_group_path = f"{drawer_path}/{link_name}/collisions"
        collision_group = stage.GetPrimAtPath(collision_group_path)
        if not collision_group.IsValid():
            continue
        # The group itself is the instance root; collision_original_* and its
        # Mesh are otherwise read-only proxies.  De-instance only this physics
        # branch, leaving the separate visual instance untouched, then author
        # the render guards directly on every concrete collision descendant.
        if collision_group.IsInstance():
            collision_group.SetInstanceable(False)
            deinstanced_collision_groups += 1
            collision_group = stage.GetPrimAtPath(collision_group_path)
        render_paths = [collision_group.GetPath()]
        while render_paths:
            render_path = render_paths.pop()
            render_prim = stage.GetPrimAtPath(render_path)
            if render_prim.IsInstance():
                # collision_original_* is another instance boundary below the
                # group.  De-instance it before authoring visibility on the
                # concrete World/mesh prim consumed by Fabric/RTX.
                render_prim.SetInstanceable(False)
                deinstanced_collision_instances += 1
                render_prim = stage.GetPrimAtPath(render_path)
            render_paths.extend(child.GetPath() for child in render_prim.GetChildren())
            imageable = UsdGeom.Imageable(render_prim)
            if not imageable:
                continue
            imageable.MakeInvisible()
            imageable.CreatePurposeAttr(UsdGeom.Tokens.guide)
            hidden_collision_render_prims += 1
    print(
        "[HandsOn/IsaacLab] FU006 collision render geometry hidden: "
        f"groups_deinstanced={deinstanced_collision_groups}, "
        f"instances_deinstanced={deinstanced_collision_instances}, "
        f"prims_hidden={hidden_collision_render_prims}",
        flush=True,
    )

    # These two RGBA decal quads are authored directly on top of the wood
    # panels.  RTX stochastic alpha turns their coplanar overlap into moving
    # red/white speckles at oblique camera angles.  They are visual-only and
    # separate from every structural panel, handle, rigid body, and collider.
    # All vendor drawer visuals are replaced below, so no legacy decal needs
    # to be de-instanced for RTX.  The physical zero-thickness decal colliders
    # are still disabled separately.
    decal_roots = {}
    deinstanced_visual_groups = 0
    hidden_decal_prims = 0
    for link_name, decal_name in decal_roots.items():
        visual_group_path = f"{drawer_path}/{link_name}/visuals"
        visual_group = stage.GetPrimAtPath(visual_group_path)
        if not visual_group.IsValid():
            continue
        if visual_group.IsInstance():
            visual_group.SetInstanceable(False)
            deinstanced_visual_groups += 1
        decal_root = stage.GetPrimAtPath(f"{visual_group_path}/{decal_name}")
        if not decal_root.IsValid():
            continue
        # Fabric can retain referenced instance geometry after a visibility
        # override.  Deactivate the visual instance root so the coplanar plane
        # never reaches RTX while all structural wood visuals remain active.
        decal_root.SetActive(False)
        hidden_decal_prims += 1
    print(
        "[HandsOn/IsaacLab] FU006 coplanar RGBA decals hidden: "
        f"groups_deinstanced={deinstanced_visual_groups}, "
        f"prims_hidden={hidden_decal_prims}",
        flush=True,
    )

    # The asset duplicates the same zero-thickness decoration planes into its
    # collision branches.  They are not physical drawer surfaces; disable only
    # those two collider instances and retain every structural collider.
    decal_collision_roots = {
        "link_1": "collision_original_32",
        "link_2": "collision_original_45",
    }
    disabled_decal_colliders = 0
    for link_name, collision_name in decal_collision_roots.items():
        collision_decal = stage.GetPrimAtPath(
            f"{drawer_path}/{link_name}/collisions/{collision_name}"
        )
        if not collision_decal.IsValid():
            continue
        collision_decal.SetActive(False)
        disabled_decal_colliders += 1
    print(
        "[HandsOn/IsaacLab] FU006 coplanar decal colliders disabled: "
        f"count={disabled_decal_colliders}",
        flush=True,
    )

    # The vendor visual meshes contain pairs of faces with identical vertices
    # and opposite winding, often split between a wood material and a plain
    # white material.  RTX renders both material sides and they z-fight when the
    # camera moves.  Collapse only the visual topology to one face per plane,
    # preserving the better material, UVs, normals, and subset assignments.
    def _fu006_material_score(material) -> tuple[int, int, int]:
        if not material:
            return (-1, -1, -1)
        textured = 0
        nonwhite = 0
        opaque = 1
        shader_prims = [material.GetPrim()]
        while shader_prims:
            shader_prim = shader_prims.pop()
            shader_prims.extend(shader_prim.GetChildren())
            texture_attr = shader_prim.GetAttribute("inputs:diffuse_texture")
            if texture_attr.IsValid() and texture_attr.Get():
                textured = 1
            color_attr = shader_prim.GetAttribute("inputs:diffuse_color_constant")
            color = color_attr.Get() if color_attr.IsValid() else None
            if color is not None and any(abs(float(channel) - 1.0) > 0.05 for channel in color):
                nonwhite = 1
            opacity_attr = shader_prim.GetAttribute("inputs:enable_opacity")
            if opacity_attr.IsValid() and opacity_attr.Get():
                opaque = 0
        return (opaque, textured, nonwhite)

    sanitized_visual_groups = 0
    sanitized_visual_meshes = 0
    removed_visual_faces = 0
    remapped_visual_primvars = 0
    # The downloaded cabinet fully replaces the vendor render meshes.  Avoid
    # de-instancing and rewriting hidden legacy topology during every launch.
    for link_name in ():
        visual_group_path = f"{drawer_path}/{link_name}/visuals"
        visual_group = stage.GetPrimAtPath(visual_group_path)
        if not visual_group.IsValid():
            continue
        if visual_group.IsInstance():
            visual_group.SetInstanceable(False)
            sanitized_visual_groups += 1
            visual_group = stage.GetPrimAtPath(visual_group_path)
        visual_prims = [visual_group]
        while visual_prims:
            visual_prim = visual_prims.pop()
            visual_prims.extend(visual_prim.GetChildren())
            if not visual_prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(visual_prim)
            points = mesh.GetPointsAttr().Get() or []
            face_counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
            face_indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            if not points or not face_counts or sum(face_counts) != len(face_indices):
                continue

            face_offsets = []
            offset = 0
            face_groups = {}
            for face_index, face_count in enumerate(face_counts):
                face_offsets.append(offset)
                vertices = face_indices[offset : offset + face_count]
                offset += face_count
                key = tuple(
                    sorted(
                        (
                            round(float(points[vertex][0]), 7),
                            round(float(points[vertex][1]), 7),
                            round(float(points[vertex][2]), 7),
                        )
                        for vertex in vertices
                    )
                )
                face_groups.setdefault(key, []).append(face_index)
            duplicate_groups = [group for group in face_groups.values() if len(group) > 1]
            if not duplicate_groups:
                continue

            direct_material, _ = UsdShade.MaterialBindingAPI(
                visual_prim
            ).ComputeBoundMaterial()
            face_materials = {index: direct_material for index in range(len(face_counts))}
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
            for subset in subsets:
                subset_material, _ = UsdShade.MaterialBindingAPI(
                    subset.GetPrim()
                ).ComputeBoundMaterial()
                for face_index in subset.GetIndicesAttr().Get() or []:
                    face_materials[int(face_index)] = subset_material

            keep_faces = []
            for group in face_groups.values():
                keep_faces.append(
                    max(
                        group,
                        key=lambda index: (
                            _fu006_material_score(face_materials.get(index)),
                            -index,
                        ),
                    )
                )
            keep_faces.sort()
            old_to_new = {old: new for new, old in enumerate(keep_faces)}

            def _select_face_values(values, interpolation):
                if values is None:
                    return None
                if interpolation == UsdGeom.Tokens.uniform:
                    if len(values) < len(face_counts):
                        return None
                    return [values[index] for index in keep_faces]
                if interpolation == UsdGeom.Tokens.faceVarying:
                    if len(values) < len(face_indices):
                        return None
                    selected = []
                    for index in keep_faces:
                        start = face_offsets[index]
                        selected.extend(values[start : start + face_counts[index]])
                    return selected
                return None

            normals_attr = mesh.GetNormalsAttr()
            if normals_attr.IsValid():
                selected_normals = _select_face_values(
                    normals_attr.Get(), mesh.GetNormalsInterpolation()
                )
                if selected_normals is not None:
                    normals_attr.Set(selected_normals)

            for primvar in UsdGeom.PrimvarsAPI(visual_prim).GetPrimvars():
                interpolation = primvar.GetInterpolation()
                if primvar.IsIndexed():
                    selected_indices = _select_face_values(
                        primvar.GetIndices(), interpolation
                    )
                    if selected_indices is not None:
                        primvar.SetIndices(selected_indices)
                        remapped_visual_primvars += 1
                else:
                    selected_values = _select_face_values(primvar.Get(), interpolation)
                    if selected_values is not None:
                        primvar.Set(selected_values)
                        remapped_visual_primvars += 1

            for subset in subsets:
                subset.GetIndicesAttr().Set(
                    [
                        old_to_new[int(index)]
                        for index in subset.GetIndicesAttr().Get() or []
                        if int(index) in old_to_new
                    ]
                )

            new_counts = [face_counts[index] for index in keep_faces]
            new_indices = []
            for index in keep_faces:
                start = face_offsets[index]
                new_indices.extend(face_indices[start : start + face_counts[index]])
            mesh.GetFaceVertexCountsAttr().Set(new_counts)
            mesh.GetFaceVertexIndicesAttr().Set(new_indices)
            mesh.CreateDoubleSidedAttr(True)
            sanitized_visual_meshes += 1
            removed_visual_faces += len(face_counts) - len(keep_faces)

    print(
        "[HandsOn/IsaacLab] FU006 coplanar visual faces sanitized: "
        f"groups_deinstanced={sanitized_visual_groups}, "
        f"meshes={sanitized_visual_meshes}, faces_removed={removed_visual_faces}, "
        f"primvars_remapped={remapped_visual_primvars}",
        flush=True,
    )

    # Retain the vendor articulation and success frame, but replace every
    # vendor render mesh with the selected CC BY cabinet.  The upper downloaded
    # drawer stays closed; the lower downloaded drawer follows link_1.
    for obsolete_visual_path in (
        f"{drawer_path}/link_0/visuals",
        f"{drawer_path}/link_1/visuals",
        f"{drawer_path}/link_2/visuals",
    ):
        obsolete_visual = stage.GetPrimAtPath(obsolete_visual_path)
        if obsolete_visual.IsValid():
            UsdGeom.Imageable(obsolete_visual).MakeInvisible()

    # The old cabinet and lower drawer no longer provide fixture collision.
    # The visible downloaded drawer receives aligned moving proxies below.
    for obsolete_collision_path in (
        f"{drawer_path}/link_0/collisions",
        f"{drawer_path}/link_1/collisions",
        f"{drawer_path}/link_2/collisions",
    ):
        obsolete_collision = stage.GetPrimAtPath(obsolete_collision_path)
        if not obsolete_collision.IsValid():
            continue
        for collision_prim in Usd.PrimRange(obsolete_collision):
            if collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(
                    collision_prim
                ).CreateCollisionEnabledAttr().Set(False)

    static_wrapper_path = f"{drawer_path}/link_2/fu006_modern_cabinet_visual"
    existing_static_wrapper = stage.GetPrimAtPath(static_wrapper_path)
    if existing_static_wrapper.IsValid():
        stage.RemovePrim(static_wrapper_path)
    static_wrapper = UsdGeom.Xform.Define(stage, static_wrapper_path)
    static_xform = UsdGeom.Xformable(static_wrapper.GetPrim())
    static_xform.ClearXformOpOrder()
    local_vertical_offset = (
        _FU006_MODERN_CABINET_ORIGIN_Z_M - _FU006_DRAWER_TRANSLATION_M[2]
    ) / drawer_scale
    local_model_scale = tuple(
        value / drawer_scale
        for value in _FU006_MODERN_CABINET_WORLD_SCALE_XYZ
    )
    static_xform.AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    ).Set(Gf.Vec3d(0.0, local_vertical_offset, 0.0))
    static_xform.AddScaleOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    ).Set(Gf.Vec3d(*local_model_scale))

    static_asset_path = f"{static_wrapper_path}/asset"
    static_asset = stage.DefinePrim(static_asset_path, "Xform")
    # Reference the complete default prim so /World/Looks remains inside the
    # reference scope. Referencing RootNode directly drops the PBR materials.
    static_asset.GetReferences().AddReference(str(modern_cabinet_asset))
    static_moving_drawer = stage.GetPrimAtPath(
        f"{static_asset_path}/{_FU006_MODERN_CABINET_GEOMETRY_ROOT}/"
        f"{_FU006_MODERN_CABINET_MOVING_DRAWER}"
    )
    if not static_moving_drawer.IsValid():
        raise RuntimeError(
            f"{task_name} modern moving drawer is missing: "
            f"{static_asset_path}/{_FU006_MODERN_CABINET_MOVING_DRAWER}"
        )
    static_moving_drawer.SetActive(False)

    link_1_path = f"{drawer_path}/link_1"
    link_1 = stage.GetPrimAtPath(link_1_path)
    if not link_1.IsValid():
        raise RuntimeError(f"{task_name} moving link is missing: {link_1_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    static_wrapper_world = xform_cache.GetLocalToWorldTransform(
        static_wrapper.GetPrim()
    )
    link_1_world = xform_cache.GetLocalToWorldTransform(link_1)

    moving_wrapper_path = f"{link_1_path}/fu006_modern_drawer_visual"
    existing_moving_wrapper = stage.GetPrimAtPath(moving_wrapper_path)
    if existing_moving_wrapper.IsValid():
        stage.RemovePrim(moving_wrapper_path)
    moving_wrapper = UsdGeom.Xform.Define(stage, moving_wrapper_path)
    moving_relative = static_wrapper_world * link_1_world.GetInverse()
    moving_wrapper.AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    ).Set(moving_relative)
    moving_asset_path = f"{moving_wrapper_path}/asset"
    moving_asset = stage.DefinePrim(moving_asset_path, "Xform")
    moving_asset.GetReferences().AddReference(str(modern_cabinet_asset))
    moving_geometry_path = (
        f"{moving_asset_path}/{_FU006_MODERN_CABINET_GEOMETRY_ROOT}"
    )
    for static_part_name in (
        "BedsideTable",
        _FU006_MODERN_CABINET_STATIC_DRAWER,
    ):
        static_part = stage.GetPrimAtPath(
            f"{moving_geometry_path}/{static_part_name}"
        )
        if static_part.IsValid():
            static_part.SetActive(False)
    moving_mesh_count = sum(
        prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(moving_asset)
    )
    if moving_mesh_count != 4:
        raise RuntimeError(
            f"{task_name} modern moving drawer expected 4 meshes, "
            f"got {moving_mesh_count}"
        )

    moving_collision_root_path = f"{link_1_path}/fu006_modern_drawer_collision"
    existing_moving_collision_root = stage.GetPrimAtPath(
        moving_collision_root_path
    )
    if existing_moving_collision_root.IsValid():
        stage.RemovePrim(moving_collision_root_path)
    moving_collision_root = UsdGeom.Xform.Define(
        stage, moving_collision_root_path
    )
    moving_collision_imageable = UsdGeom.Imageable(
        moving_collision_root.GetPrim()
    )
    moving_collision_imageable.MakeInvisible()
    moving_collision_imageable.CreatePurposeAttr(UsdGeom.Tokens.guide)
    moving_collider_count = 0
    for collider_name, (source_min, source_max) in (
        _FU006_MODERN_DRAWER_COLLISION_BOXES_SOURCE.items()
    ):
        source_minimum = Gf.Vec3d(*source_min)
        source_maximum = Gf.Vec3d(*source_max)
        center = 0.5 * (source_minimum + source_maximum)
        size = source_maximum - source_minimum
        collider = UsdGeom.Cube.Define(
            stage, f"{moving_collision_root_path}/{collider_name}"
        )
        collider.CreateSizeAttr(1.0)
        collider.AddTranslateOp().Set(center)
        collider.AddScaleOp().Set(Gf.Vec3f(*size))
        collider.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        collision = UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(True)
        moving_collider_count += 1
    if moving_collider_count != len(_FU006_MODERN_DRAWER_COLLISION_BOXES_SOURCE):
        raise RuntimeError(
            f"{task_name} modern moving drawer collision proxy count mismatch"
        )

    # Compose an invisible collision copy from the same downloaded cabinet
    # geometry.  It remains a world-static triangle mesh outside the uniformly
    # scaled articulation; this avoids both hand-through-shell behavior and the
    # PhysX GPU failure caused by adding static shapes below an articulation.
    collision_wrapper_path = f"{prim_path}/fu006_modern_cabinet_collision"
    existing_collision_wrapper = stage.GetPrimAtPath(collision_wrapper_path)
    if existing_collision_wrapper.IsValid():
        stage.RemovePrim(collision_wrapper_path)
    collision_wrapper = UsdGeom.Xform.Define(stage, collision_wrapper_path)
    parent_prim = stage.GetPrimAtPath(prim_path)
    parent_world = xform_cache.GetLocalToWorldTransform(parent_prim)
    collision_relative = static_wrapper_world * parent_world.GetInverse()
    collision_wrapper.AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble,
        "fu006",
    ).Set(collision_relative)
    collision_imageable = UsdGeom.Imageable(collision_wrapper.GetPrim())
    collision_imageable.MakeInvisible()
    collision_imageable.CreatePurposeAttr(UsdGeom.Tokens.guide)

    collision_asset_path = f"{collision_wrapper_path}/asset"
    collision_asset = stage.DefinePrim(collision_asset_path, "Xform")
    collision_asset.GetReferences().AddReference(str(modern_cabinet_asset))
    collision_moving_drawer = stage.GetPrimAtPath(
        f"{collision_asset_path}/{_FU006_MODERN_CABINET_GEOMETRY_ROOT}/"
        f"{_FU006_MODERN_CABINET_MOVING_DRAWER}"
    )
    if not collision_moving_drawer.IsValid():
        raise RuntimeError(
            f"{task_name} modern collision drawer node is missing"
        )
    collision_moving_drawer.SetActive(False)

    static_collider_count = 0
    for collision_prim in Usd.PrimRange(collision_asset):
        if not collision_prim.IsA(UsdGeom.Mesh):
            continue
        collision = UsdPhysics.CollisionAPI.Apply(collision_prim)
        collision.CreateCollisionEnabledAttr().Set(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
        mesh_collision.CreateApproximationAttr().Set("none")
        static_collider_count += 1
    if static_collider_count != 5:
        raise RuntimeError(
            f"{task_name} modern cabinet expected 5 static mesh colliders, "
            f"got {static_collider_count}"
        )

    # The proxy side walls, back, and floor intentionally fit tightly inside
    # the downloaded cabinet shell.  Letting PhysX solve those fixture-internal
    # contacts wedges the prismatic link at its reset position.  Filter only
    # the moving-drawer/static-cabinet pair; both groups still collide with the
    # robot hands and charger.
    internal_group_root = f"{prim_path}/fu006_internal_collision_groups"
    moving_group = UsdPhysics.CollisionGroup.Define(
        stage, f"{internal_group_root}/moving_drawer"
    )
    moving_collection = moving_group.GetCollidersCollectionAPI()
    moving_collection.CreateIncludesRel().AddTarget(
        Sdf.Path(moving_collision_root_path)
    )
    # The floor is authored later by the environment before the first PhysX
    # reset.  USD collection relationships may safely target that future prim.
    moving_collection.CreateIncludesRel().AddTarget(
        Sdf.Path(f"{link_1_path}/fu006_floor_collision_repair")
    )
    static_group = UsdPhysics.CollisionGroup.Define(
        stage, f"{internal_group_root}/static_cabinet"
    )
    static_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(
        Sdf.Path(collision_wrapper_path)
    )
    moving_group.CreateFilteredGroupsRel().AddTarget(static_group.GetPath())
    static_group.CreateFilteredGroupsRel().AddTarget(moving_group.GetPath())

    original_lower_cabinet_height_m = drawer_scale * (
        _FU006_ORIGINAL_LOWER_CABINET_TOP_SOURCE_Z
        - _FU006_ORIGINAL_LOWER_CABINET_BOTTOM_SOURCE_Z
    )
    modern_cabinet_height_m = (
        _FU006_MODERN_CABINET_WORLD_SCALE_XYZ[1]
        * (
            _FU006_MODERN_CABINET_SOURCE_Y_MAX_M
            - _FU006_MODERN_CABINET_SOURCE_Y_MIN_M
        )
    )
    cabinet_height_limit_m = 0.75 * original_lower_cabinet_height_m
    if modern_cabinet_height_m > cabinet_height_limit_m + 1.0e-9:
        raise RuntimeError(
            "FU006 modern cabinet exceeds 3/4 of the original lower-cabinet "
            f"height: modern={modern_cabinet_height_m:.4f}m "
            f"limit={cabinet_height_limit_m:.4f}m"
        )
    modern_top_z_m = _FU006_DRAWER_SUPPORT_Z_M + modern_cabinet_height_m
    visible_mesh_count = sum(
        prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(static_asset)
    )
    print(
        "[HandsOn/IsaacLab] FU006 CC BY modern cabinet composed: "
        f"asset={modern_cabinet_asset.name}, "
        f"visible_static_meshes={visible_mesh_count}, "
        f"moving_drawer_meshes={moving_mesh_count}, "
        f"moving_drawer_colliders={moving_collider_count + 1}, "
        f"static_mesh_colliders={static_collider_count}, "
        "internal_collision_filter=enabled, "
        f"height={modern_cabinet_height_m:.4f}m, "
        f"height_limit={cabinet_height_limit_m:.4f}m, "
        f"top_z={modern_top_z_m:.4f}m, "
        f"back_shift={_FU006_DRAWER_BACK_SHIFT_M:.3f}m",
        flush=True,
    )

    initial_position = drawer_scale * 0.28
    upper_limit = drawer_scale * 0.7760000228881836
    prismatic_joint = UsdPhysics.PrismaticJoint(joint_prim)
    prismatic_joint.GetLowerLimitAttr().Set(0.0)
    prismatic_joint.GetUpperLimitAttr().Set(upper_limit)
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
    if drive:
        # Reset establishes the initial opening; runtime motion stays passive
        # so the operator can fully close and physically hold the drawer.
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(1.0)
        drive.GetTargetVelocityAttr().Set(0.0)
        drive.GetTargetPositionAttr().Set(initial_position)
        # Reset establishes the initial opening; runtime motion stays passive
        # so the operator can fully close and physically hold the drawer.
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(1.0)
        drive.GetTargetVelocityAttr().Set(0.0)
    position_attr = joint_prim.GetAttribute("state:linear:physics:position")
    if position_attr.IsValid():
        position_attr.Set(initial_position)
    return drawer_path, charger_path


def _compose_as019_preview_assets(
    stage,
    prim_path: str,
) -> tuple[str, str]:
    """Mount the reviewed AS019 screwdriver and safety zone for preview.

    AS019 used to borrow the generic bimanual-assembly MJCF just to get a
    robot/table pair.  That scene also imports the peg/tray assembly, and its
    first PhysX GPU step can fail before the preview has any task asset.  The
    C-tool bootstrap is already known to be stable; remove its props first and
    mount the exact AS019 USD bundle as a separate reference instead.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from .as019_asset_support import scene_contract as contract

    asset = (
        Path(__file__).resolve().parent.parent / "envs" / "usd"
        / "as019_task_assets" / "polyhaven_flathead_screwdriver"
        / "runtime" / "flathead_screwdriver.usd"
    )
    if not asset.is_file():
        raise FileNotFoundError(f"AS019 preview runtime USD is missing: {asset}")
    marker_orientation = contract.MARKER_ORIENTATION_WXYZ
    marker_position = contract.MARKER_POSITION_M
    marker_size = contract.MARKER_SIZE_M
    screwdriver_orientation = contract.SCREWDRIVER_ORIENTATION_WXYZ
    screwdriver_position = contract.SCREWDRIVER_POSITION_M
    screwdriver_scale = contract.SCREWDRIVER_SCALE_XYZ

    root_path = f"{prim_path}/AS019Preview"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)

    screwdriver_path = f"{root_path}/Screwdriver"
    screwdriver = UsdGeom.Xform.Define(stage, screwdriver_path)
    screwdriver.AddTranslateOp().Set(Gf.Vec3d(*screwdriver_position))
    screwdriver.AddOrientOp().Set(Gf.Quatf(*screwdriver_orientation))
    screwdriver.AddScaleOp().Set(Gf.Vec3f(*screwdriver_scale))
    screwdriver_asset = UsdGeom.Xform.Define(
        stage,
        f"{screwdriver_path}/Asset",
    )
    screwdriver_asset.GetPrim().GetReferences().AddReference(str(asset))
    # Resolve the reference before authoring preview-only collision overrides.
    # The caller performs another stage update before PhysX reset; this first
    # update is only for making the referenced prims addressable in the edit
    # layer.
    sim_utils.update_stage()
    # Referencing a USD's default prim into an existing Xform composes the
    # default prim's children directly under that target (the target itself is
    # the rigid-body owner), so the live path is ``.../Asset`` rather than
    # ``.../Asset/FlatheadScrewdriver``.
    asset_root_path = f"{screwdriver_path}/Asset"
    asset_root = stage.GetPrimAtPath(asset_root_path)
    if not asset_root.IsValid():
        descendants = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(screwdriver_asset.GetPrim())
        ]
        raise RuntimeError(
            "AS019 preview screwdriver reference did not load: "
            f"{asset_root_path}; descendants={descendants[:20]}"
        )
    # Isaac Sim's standalone preview uses the shared GPU PhysX importer, while
    # the production AS019 collector owns the exact five source-derived hulls
    # in its native scene.  The referenced hull meshes are valid for the
    # collector but trigger a 700 narrowphase fault in this generic preview
    # path.  Keep the reviewed visual reference and replace only preview-time
    # collision with one bounded primitive under the same rigid body.
    disabled_colliders = 0
    for prim in Usd.PrimRange(asset_root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        disabled_colliders += 1
    preview_collider = UsdGeom.Cube.Define(
        stage,
        f"{asset_root_path}/PreviewCollider",
    )
    preview_collider.CreateSizeAttr(1.0)
    preview_collider.AddTranslateOp().Set(Gf.Vec3d(0.005, 0.083, 0.0))
    preview_collider.AddScaleOp().Set(Gf.Vec3f(0.015, 0.13, 0.015))
    UsdPhysics.CollisionAPI.Apply(preview_collider.GetPrim())
    UsdPhysics.CollisionAPI(
        preview_collider.GetPrim()
    ).CreateCollisionEnabledAttr().Set(True)
    UsdGeom.Imageable(preview_collider.GetPrim()).MakeInvisible()
    print(
        "[HandsOn/IsaacLab] AS019 preview collision fallback: "
        f"disabled_source_colliders={disabled_colliders}, primitive=1",
        flush=True,
    )

    marker_path = f"{root_path}/SafeOrientationMarker"
    marker = UsdGeom.Cube.Define(stage, marker_path)
    marker.CreateSizeAttr(1.0)
    marker.AddTranslateOp().Set(Gf.Vec3d(*marker_position))
    marker.AddOrientOp().Set(Gf.Quatf(*marker_orientation))
    marker.AddScaleOp().Set(
        Gf.Vec3f(*(0.5 * value for value in marker_size))
    )
    marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.04, 0.95, 0.12)])
    marker.GetPrim().SetCustomDataByKey("as019:collisionEnabled", False)

    print(
        "[HandsOn/IsaacLab] AS019 preview assets mounted: "
        f"screwdriver={screwdriver_path}, marker={marker_path}",
        flush=True,
    )
    return screwdriver_path, marker_path


def _compose_bc005_selected_assets_for_visual_preview(
    stage,
    task_root_path: str,
) -> tuple[str, str]:
    """Show BC005's selected production assets in the generic UI preview.

    The formal collection environment performs a deeper collision/material
    audit in ``HandsOnLoadedTrayGateIsaacLabEnv``.  The generic visual preview
    does not instantiate that environment, so without this pre-PhysX compose
    step it showed the converter's transparent scaffolds (rendered as black by
    RTX) instead of the user's downloaded tray and wooden frame.
    """

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    asset_dir = (
        Path(__file__).resolve().parents[1]
        / "envs"
        / "usd"
        / "loaded_tray_gate_assets"
    )
    tray_asset = asset_dir / "bc005_selected_wooden_tray.usd"
    frame_asset = asset_dir / "bc005_selected_wooden_frame.usd"
    missing = [path for path in (tray_asset, frame_asset) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "B320-BC005 selected preview asset is missing: "
            + ", ".join(str(path) for path in missing)
        )

    root = stage.GetPrimAtPath(task_root_path)
    if not root.IsValid():
        raise RuntimeError(f"B320-BC005 preview root is missing: {task_root_path}")

    def named_rigid_body(name: str):
        matches = [
            prim
            for prim in Usd.PrimRange(root)
            if str(prim.GetName()) == name
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"B320-BC005 preview expected one rigid body {name!r}, "
                f"got {[str(prim.GetPath()) for prim in matches]}"
            )
        return matches[0]

    tray_body = named_rigid_body("loaded_tray")
    frame_body = named_rigid_body("loaded_tray_gate")
    body_paths = {
        "tray": str(tray_body.GetPath()),
        "frame": str(frame_body.GetPath()),
    }
    for body_path in body_paths.values():
        sim_utils.make_uninstanceable(body_path, stage=stage)
    root = stage.GetPrimAtPath(task_root_path)
    tray_body = named_rigid_body("loaded_tray")
    frame_body = named_rigid_body("loaded_tray_gate")

    def converter_scope(body, scope_name: str, *, required: bool = True):
        matches = [
            prim
            for prim in Usd.PrimRange(body)
            if prim != body and str(prim.GetName()) == scope_name
        ]
        if len(matches) > 1 or (required and not matches):
            raise RuntimeError(
                "B320-BC005 preview converter scope mismatch: "
                f"body={body.GetPath()}, scope={scope_name!r}, count={len(matches)}"
            )
        return matches[0] if matches else None

    tray_visuals = converter_scope(tray_body, "visuals")
    tray_collisions = converter_scope(tray_body, "collisions")
    frame_visuals = converter_scope(frame_body, "visuals", required=False)
    frame_collisions = converter_scope(frame_body, "collisions")
    if tray_visuals is None or tray_collisions is None or frame_collisions is None:
        raise RuntimeError("B320-BC005 preview converter scopes resolved to null")
    for scope in (tray_visuals, tray_collisions, frame_visuals):
        if scope is not None:
            scope.SetActive(False)

    tray_reference_path = f"{body_paths['tray']}/BC005ProductionTray"
    frame_reference_path = f"{body_paths['frame']}/BC005ProductionWoodenFrame"
    tray_reference = UsdGeom.Xform.Define(stage, tray_reference_path).GetPrim()
    frame_reference = UsdGeom.Xform.Define(stage, frame_reference_path).GetPrim()
    for reference, asset_path in (
        (tray_reference, tray_asset),
        (frame_reference, frame_asset),
    ):
        references = reference.GetReferences()
        references.ClearReferences()
        if not references.AddReference(str(asset_path.resolve())):
            raise RuntimeError(
                f"B320-BC005 preview could not reference {asset_path}"
            )

    tray_prims = list(Usd.PrimRange(tray_reference))
    frame_prims = list(Usd.PrimRange(frame_reference))
    tray_visual_meshes = [
        prim
        for prim in tray_prims
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    tray_colliders = [
        prim for prim in tray_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    frame_meshes = [prim for prim in frame_prims if prim.IsA(UsdGeom.Mesh)]
    frame_colliders = [
        prim for prim in frame_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if (
        len(tray_visual_meshes) != 1
        or len(tray_colliders) != 17
        or len(frame_meshes) != 1
        or frame_colliders
    ):
        raise RuntimeError(
            "B320-BC005 preview selected-asset topology mismatch: "
            f"tray_visuals={len(tray_visual_meshes)}, "
            f"tray_colliders={len(tray_colliders)}, "
            f"frame_visuals={len(frame_meshes)}, "
            f"frame_colliders={len(frame_colliders)}"
        )

    # Fail closed on the exact material issue that motivated this path.  The
    # selected frame must bind a resolvable RGB texture and fixed opaque
    # PreviewSurface; otherwise the preview may silently regress to black.
    frame_material, _ = UsdShade.MaterialBindingAPI(
        frame_meshes[0]
    ).ComputeBoundMaterial()
    if not frame_material:
        raise RuntimeError("B320-BC005 preview frame has no bound material")
    surface_sources, _ = frame_material.GetSurfaceOutput().GetConnectedSources()
    if len(surface_sources) != 1:
        raise RuntimeError("B320-BC005 preview frame surface is disconnected")
    surface = UsdShade.Shader(surface_sources[0].source)
    diffuse_sources, _ = surface.GetInput("diffuseColor").GetConnectedSources()
    if len(diffuse_sources) != 1:
        raise RuntimeError("B320-BC005 preview frame base color is disconnected")
    texture = UsdShade.Shader(diffuse_sources[0].source)
    texture_asset = texture.GetInput("file").Get()
    opacity = surface.GetInput("opacity")
    if (
        texture.GetIdAttr().Get() != "UsdUVTexture"
        or not texture_asset
        or not Path(texture_asset.resolvedPath).is_file()
        or float(opacity.Get()) != 1.0
        or opacity.HasConnectedSource()
    ):
        raise RuntimeError(
            "B320-BC005 preview frame material must resolve the selected "
            "base-color texture and remain explicitly opaque"
        )

    tray_reference.SetCustomDataByKey("previewSelectedAsset", True)
    frame_reference.SetCustomDataByKey("previewSelectedAsset", True)
    return tray_reference_path, frame_reference_path


def spawn_task_scene(
    task_name: str,
    prim_path: str = "/World/Task",
    *,
    first_physx_safe: bool = False,
    spawn_b03_center_rig: bool = True,
    studio_lighting: bool = False,
    hand_type: str = "wuji1",
    preview_appearance: bool = True,
) -> tuple[Path, tuple[int, int]]:
    """Import a complete task MJCF and repair nested mesh appearance."""
    task_mjcf = prepare_task_mjcf(
        task_name,
        first_physx_safe=first_physx_safe,
        hand_type=hand_type,
    )
    usd_dir = task_usd_cache_dir(
        task_name,
        task_mjcf,
        first_physx_safe=first_physx_safe,
        hand_type=hand_type,
    )
    print(
        f"[HandsOn/IsaacLab] Task asset paths: task={task_name} "
        f"mjcf={task_mjcf} usd_dir={usd_dir} "
        f"cache_version=v{_TASK_MJCF_CACHE_VERSION} "
        "xml_dir="
        f"{os.environ.get('HANDSON_ISAACLAB_XML_DIR', '<default>')}",
        flush=True,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        max_linear_velocity=10.0,
        max_angular_velocity=30.0,
        max_depenetration_velocity=3.0,
        enable_gyroscopic_forces=True,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=4,
        sleep_threshold=(
            _B04_SLEEP_THRESHOLD
            if task_name == "t44_stack_blocks"
            else 0.0
        ),
        stabilization_threshold=0.0005,
    )
    joint_drive_props = sim_utils.JointDrivePropertiesCfg(drive_type="force")
    spawn_cfg = make_mjcf_file_cfg(
        sim_utils,
        asset_path=str(task_mjcf),
        usd_dir=str(usd_dir),
        fix_base=mjcf_full_scene_fix_base(),
        import_sites=True,
        self_collision=False,
        make_instanceable=False,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive_props=joint_drive_props,
    )
    import omni.usd

    usd_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_hand_suffix = (
        ""
        if str(hand_type or "wuji1").strip().lower() == "wuji1"
        else ".wuji2"
    )
    lock_path = usd_dir.parent / (
        f".{task_name}{lock_hand_suffix}.conversion.lock"
    )
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if _task_usd_cache_is_incomplete(usd_dir, task_mjcf):
            shutil.rmtree(usd_dir)
            print(
                f"[HandsOn/IsaacLab] Removed incomplete task USD cache: {usd_dir}",
                flush=True,
            )

        cached_usd = _task_cached_usd_path(usd_dir, task_mjcf)
        if task_name == "t09_insert_peg" and cached_usd.is_file():
            # A valid content-addressed cache is immutable input.  Loading it
            # as USD bypasses MjcfConverter, whose generated config/commentary
            # contains host paths and timestamps and whose importer may create
            # a misleading ``_<N>`` sibling even though the base tree exists.
            active_spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=str(cached_usd),
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                joint_drive_props=joint_drive_props,
            )
        else:
            active_spawn_cfg = spawn_cfg

        active_spawn_cfg.func(prim_path, active_spawn_cfg)
        stage = omni.usd.get_context().get_stage()
        if not _task_stage_reference_is_loaded(stage, prim_path):
            stage.RemovePrim(prim_path)
            sim_utils.update_stage()
            shutil.rmtree(usd_dir, ignore_errors=True)
            print(
                f"[HandsOn/IsaacLab] Retrying empty task USD conversion: {task_name}",
                flush=True,
            )
            spawn_cfg.func(prim_path, spawn_cfg)
            stage = omni.usd.get_context().get_stage()
            if not _task_stage_reference_is_loaded(stage, prim_path):
                raise RuntimeError(
                    f"Isaac task USD reference is empty after rebuild: {task_name}"
                )

    hidden_task_site_groups = _hide_imported_task_sites(stage, prim_path)
    fu006_assets = (
        _compose_fu006_assets(stage, prim_path)
        if task_name == "t46_store_charger"
        else None
    )
    as019_preview_assets = (
        _compose_as019_preview_assets(stage, prim_path)
        if task_name == "t22_pass_screwdriver"
        else None
    )
    fixed_roots = 0 if is_lab3() else repair_fixed_root_joints(prim_path)
    initialized_joints = apply_robot_joint_defaults(prim_path, task_name)
    repaired = _repair_imported_visual_materials(stage, prim_path)
    if task_name == "d08_2_realistic_toast":
        from .d08_realistic_variant import compose_visuals
        compose_visuals(stage, prim_path)
    bc005_preview_assets = (
        _compose_bc005_selected_assets_for_visual_preview(stage, prim_path)
        if task_name == "loaded_tray_through_narrow_gate"
        and not first_physx_safe
        else ()
    )
    # Resolve this at runtime so the lightweight scene-module stand-ins used by
    # XML/layout tests do not need to emulate Isaac's PBR authoring helpers.
    # The production module always provides the function.
    from . import bimanual_assembly_scene as assembly_scene

    appearance_fn = getattr(
        assembly_scene,
        "apply_reference_robot_appearance",
        None,
    )
    reference_appearance = (
        0
        if appearance_fn is None or first_physx_safe or not preview_appearance
        else appearance_fn(stage, robot_root_paths=(prim_path,))
    )
    c06_fixture_material_repairs = _repair_c06_fixture_materials(
        stage,
        prim_path,
        task_name,
    )
    d09_coffee_material_repairs = _repair_d09_coffee_machine_material(
        stage,
        prim_path,
        task_name,
    )
    c04_plant_material_bindings = _bind_c04_plant_materials(
        stage,
        prim_path,
        task_name,
        _c04_material_texture_paths(task_mjcf, task_name),
    )
    c04_spray_visuals_hidden = _hide_c04_spray_visual(
        stage,
        prim_path,
        task_name,
    )
    if (
        hidden_task_site_groups
        or c04_plant_material_bindings
        or c04_spray_visuals_hidden
        or c06_fixture_material_repairs
        or d09_coffee_material_repairs
        or bc005_preview_assets
        or as019_preview_assets
    ):
        sim_utils.update_stage()
    extra_center_path = ""
    extra_center_repaired = (0, 0)
    if task_name == "t09_insert_peg" and spawn_b03_center_rig:
        extra_center_path = "/World/HandsOnRobots/CenterRig"
        if not stage.GetPrimAtPath(extra_center_path).IsValid():
            spawn_center_rig(extra_center_path)
        extra_center_repaired = _repair_imported_visual_materials(
            stage,
            extra_center_path,
        )
    b04_wood_material_bindings = 0
    if task_name == "t44_stack_blocks":
        b04_wood_material_bindings = _bind_b04_wood_block_visual_material(
            stage,
            prim_path,
        )
    _spawn_task_lighting(studio_lighting=studio_lighting)
    if not first_physx_safe and preview_appearance:
        from .default_appearance import install_preview
        install_preview(stage, task_name)
    print(
        f"[HandsOn/IsaacLab] Rebuilt {fixed_roots} fixed-root joints and "
        f"initialized {initialized_joints} robot joints "
        f"({hidden_task_site_groups} task site groups hidden, "
        f"{reference_appearance} robot appearance edits, "
        f"{b04_wood_material_bindings} B04 wood-block visual bindings, "
        f"{c04_plant_material_bindings} C04 plant material bindings, "
        f"{c04_spray_visuals_hidden} C04 spray visuals hidden, "
        f"{c06_fixture_material_repairs} C06 fixture material repairs, "
        f"{d09_coffee_material_repairs} D09 coffee material repairs, "
        f"extra_center_repaired={extra_center_repaired[0]})"
    )
    if fu006_assets is not None:
        print(
            "[HandsOn/IsaacLab] FU006 exact USD assets composed before first PhysX: "
            + ", ".join(fu006_assets),
            flush=True,
        )
    if bc005_preview_assets:
        print(
            "[HandsOn/BC005] visual-preview selected assets composed before "
            "first PhysX: " + ", ".join(bc005_preview_assets),
            flush=True,
        )
    return task_mjcf, repaired
