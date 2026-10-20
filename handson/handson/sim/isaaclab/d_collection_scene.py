"""Scene metadata for the native IsaacLab D01--D05 migrations.

This module deliberately contains no Isaac imports.  ``task_preview`` consumes
the metadata while preparing the cached MJCF/USD scene, and
``d_collection_env`` owns the live PhysX task logic.
"""

from __future__ import annotations

from ...tasks.eval_comb03_pass_block_and_close_box import contract as xy1


D_TASK_SCENES = {
    "eval_gen04_pass_glue_stick": "arena_arm_hand_gen4_glue_stick_handoff_rokae_wuji1.xml",
    xy1.TASK_NAME: xy1.SCENE_XML,
    "d01_bimanual_nut_bolt_assembly": (
        "arena_arm_hand_d01_bimanual_nut_bolt_assembly_rokae_wuji1.xml"
    ),
    "t12_load_and_start_microwave": (
        "arena_arm_hand_t12_load_and_start_microwave_rokae_wuji1.xml"
    ),
    "t21_pass_block": (
        "arena_arm_hand_d03_bimanual_handoff_rokae_wuji1.xml"
    ),
    "t43_squeeze_scissors": (
        "arena_arm_hand_t43_squeeze_scissors_rokae_wuji1.xml"
    ),
}


D_SUCCESS_CONDITION_LABELS = {
    "eval_gen04_pass_glue_stick": (
        "Left hand lifts the glue stick first",
        "Both hands contact the glue stick in the transfer zone",
        "Right hand receives after the left hand releases",
        "Glue stick released and supported inside the right tray",
    ),
    xy1.TASK_NAME: xy1.SUCCESS_LABELS,
    "d01_bimanual_nut_bolt_assembly": (
        "One hand holds the bolt and the other turns the nut",
        "Nut centered and aligned on the bolt",
        "Tightening rotation and axial travel completed",
        "Final assembly remains stable for 12 frames",
    ),
    "t12_load_and_start_microwave": (
        "Microwave door closed",
        "Plate and food inside microwave cavity",
        "Food remains on plate",
        "Green start button physically pressed while loaded",
    ),
    "t21_pass_block": (
        "Left hand lifts the target block",
        "Both hands contact the block in the transfer zone",
        "Right hand receives after the left hand releases",
        "Block released stably inside the right box",
    ),
    "t43_squeeze_scissors": (
        "Scissors lifted clear of the table",
        "Both handles actively held",
        "Three complete open-close cycles counted",
    ),
}


# The source D scenes were authored around both 0.92 m and 0.89 m tabletop
# variants.  Normalize them to the ROKAE/Wuji table before the first PhysX
# reset so no object begins floating or intersecting the table.
D_TASK_BODY_POSITION_OVERRIDES = {
    xy1.TASK_NAME: xy1.NOMINAL_POSITIONS,
    "d01_bimanual_nut_bolt_assembly": {
        "bolt_root_body": (-0.38, 0.06, 0.890),
        "nut_root_body": (-0.50, -0.20, 0.902),
    },
    "t12_load_and_start_microwave": {
        # Keep the default payload in the same door-clear lane used by the
        # seeded D02 reset sampler.
        "d02_plate": (-0.34, -0.54, 0.902),
        "003_cracker_box": (-0.34, -0.54, 0.9449465),
        # Cracker is the default active food.  Park pudding before the first
        # PhysX reset; reset() moves whichever variant is selected onto plate.
        "008_pudding_box": (-2.0, 0.0, 0.018804),
        "microwave_object": (-0.05, -0.04, 1.100),
    },
    "t21_pass_block": {
        "d03_target_root": (-0.18, 0.47, 0.905),
        "d03_box": (-0.24, -0.58, 0.9645),
        "d03_flat_screwdriver": (-0.37, -0.10, 0.896),
        "d03_phillips_screwdriver": (-0.32, 0.15, 0.896),
        "d03_spoon": (-0.14, -0.055, 0.894),
        "d03_plastic_bolt": (-0.06, -0.225, 0.895),
        "d03_a_marbles": (-0.035, 0.195, 0.927),
        "d03_b_marble_1": (-0.22, -0.19, 0.9075),
        "d03_b_marble_2": (-0.27, -0.15, 0.9075),
        "d03_hammer": (0.075, 0.105, 0.904),
        "d03_nails": (-0.43, 0.30, 0.8922),
        "d03_bowl": (-0.19, 0.34, 0.916),
    },
    "t43_squeeze_scissors": {
        "d05_scissors_root_body": (-0.28, 0.0, 0.899),
    },
}


# MJCF quaternions are authored in WXYZ order.  D02 computes the food support
# heights for packages resting on their broad side, so the prepared scene must
# carry the same orientation into the *first* PhysX reset.  Applying this only
# in the later environment reset leaves the upright boxes deeply intersecting
# the table/ground while SimulationContext.reset() is creating the scene.
D_TASK_BODY_QUATERNION_OVERRIDES = {
    "t12_load_and_start_microwave": {
        "003_cracker_box": (0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
        "008_pudding_box": (0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
    },
    # Keep collision-critical authored orientations identical in every cache:
    # the old D14 banana stood through the table, the D16 broom head entered
    # the bin, and the D17 ear cups fell below the tabletop.
    "d14_wash_fruit": {"d14_banana": (1.0, 0.0, 0.0, 0.0)},
    "d16_sweep_blocks_to_bin": {
        "d16_broom": (0.70710678, 0.0, 0.0, 0.70710678)
    },
    "d17_store_laptop_headphones": {
        "d17_headphones": (0.70710678, 0.0, 0.70710678, 0.0)
    },
}


# A complete collection scene adds two live robot articulations after importing
# the task MJCF.  Keep D02's movable payload away from the import-default robot
# pose for the first SimulationContext.reset().  The microwave is world-fixed,
# so it must already be authored at its final anchor before PhysX creates the
# articulation.  Moving that fixed joint after tensor views exist corrupts the
# Sim 5.1 GPU articulation/narrowphase state (CUDA error 700).
D_TASK_FIRST_PHYSX_POSITION_OVERRIDES = {
    xy1.TASK_NAME: xy1.NOMINAL_POSITIONS,
    "t12_load_and_start_microwave": {
        "d02_plate": (-2.0, 0.0, 0.012),
        "003_cracker_box": (-2.0, 0.0, 0.0549465),
        "008_pudding_box": (-2.0, 0.5, 0.018804),
        "microwave_object": (-0.05, -0.04, 1.100),
    },
    # Repeat the supported reset roots before SimulationContext creates the
    # first PhysX broadphase.  reset() uses the same positions below.
    "d13_prepare_tea_tray": {
        "d13_mug": (-0.40, 0.50, 0.986),
        "d13_tea_bag": (-0.16, 0.28, 0.922),
        "d13_cabinet_door": (0.18, 0.16, 0.13),
    },
    "d14_wash_fruit": {
        "d14_colander": (-0.40, -0.58, 0.912),
        "d14_banana": (-0.45, -0.32, 0.950),
        "d14_apple": (-0.28, -0.32, 0.955),
        "d14_orange": (-0.11, -0.32, 0.952),
        "d14_faucet_valve": (0.02, -0.07, 0.11),
        "d14_faucet_arm": (0.02, 0.0, 0.20),
    },
    "d16_sweep_blocks_to_bin": {"d16_broom": (-0.42, 0.62, 0.925)},
    "d17_store_laptop_headphones": {"d17_headphones": (-0.42, 0.52, 0.945)},
    "d19_memory_cover_sort": {"d19_basket": (-0.35, 0.52, 0.900)},
}


D_TASK_NAMES = frozenset(D_TASK_SCENES)
