"""XY1 identity, authored layout and recorded state (no simulator imports)."""
TASK_ID = "EVAL-Comb03"
TASK_NAME = "eval_comb03_pass_block_and_close_box"
PROMPT = (
    "Pick up the left block with the left hand, hand it to the right hand in "
    "the middle, and release it stably inside the clear box. Then use the "
    "right hand to pick up the lid, align and close the box, and release."
)
SCENE_XML = "arena_arm_hand_eval_comb03_pass_block_and_close_box_rokae_wuji1.xml"
BLOCK = "d03_target_root"
BOX = "Clear_Box"
LID = "Box_lid"
# Original D03 clutter, with only the lower spoon/bowl and small-prop slots
# moved to reserve the right-side lid approach. All values are metres.
NOMINAL_POSITIONS = {
    BLOCK: (-0.27, 0.22, 0.905),
    BOX: (-0.27, -0.50, 0.9655),
    LID: (-0.27, -0.08, 0.901),
    "d03_flat_screwdriver": (-0.45, 0.52, 0.896),
    "d03_phillips_screwdriver": (0.08, 0.50, 0.896),
    "d03_spoon": (-0.445, 0.29, 0.894),
    "d03_plastic_bolt": (-0.44, 0.16, 0.895),
    "d03_a_marbles": (0.08, 0.72, 0.927),
    "d03_b_marble_1": (0.08, 0.30, 0.9075),
    "d03_b_marble_2": (-0.20, 0.38, 0.9075),
    "d03_hammer": (-0.18, 0.62, 0.904),
    # The authored nail collider is a 4.4 mm tall box.  Leaving its lower
    # face exactly on the table top makes this tiny free body tip during the
    # first GPU contact solve for some reset seeds.  Keep a small support gap
    # so the first solver step establishes contact without an initial edge
    # penetration/torque; this is a reset-geometry correction, not damping.
    "d03_nails": (-0.08, 0.34, 0.8927),
    "d03_bowl": (0.105, 0.10, 0.916),
}
DISTRACTORS = tuple(n for n in NOMINAL_POSITIONS if n not in (BLOCK, BOX, LID))
XY_JITTER = {name: (0.004, 0.004) for name in DISTRACTORS}
XY_JITTER.update({BLOCK: (0.055, 0.045), BOX: (0.012, 0.012), LID: (0.012, 0.012)})
NOMINAL_YAW_DEG = {BOX: 90.0, LID: 90.0}
YAW_BOUNDS_DEG = {name: (-180.0, 180.0) for name in DISTRACTORS}
YAW_BOUNDS_DEG.update({BLOCK: (-25.0, 25.0), BOX: (-5.0, 5.0), LID: (-5.0, 5.0)})
A_STABLE_FRAMES = 8
# Exact planes through B05 Collider4/9 and Collider1/3 inner-wall vertices.
# x/y bounds at z=-0.067075, rising to z=0.071501. Do not rescale the assets.
INNER_WALL_Z0 = -0.067075
INNER_WALL_HALF0 = (0.081766, 0.140536)
INNER_WALL_SLOPE = ((0.124181 - 0.081766) / 0.138576, (0.182952 - 0.140536) / 0.138576)
BLOCK_HALF_EXTENT = 0.014925
CONTAINMENT_CONTACT_TOLERANCE = 0.002
# xy1_right_lid_seen is a right-only contact candidate with a rebased origin;
# lid_was_moved is the resulting pickup qualification. Both are needed to
# reproduce a contact attempt across recorded current/next observations.
HIDDEN_FIELDS = (
    "d03_stage", "d03_left_streak", "d03_both_streak", "d03_right_streak",
    "d03_left_confirmed", "d03_both_confirmed", "d03_right_confirmed",
    "d03_direct_right_seen", "xy1_a_stable_frames", "xy1_a_complete",
    "xy1_lid_ready", "xy1_right_lid_seen", "lid_was_moved", "success_counter",
    "lid_start_valid", "lid_start_x", "lid_start_y", "lid_start_z",
    "lid_start_qw", "lid_start_qx", "lid_start_qy", "lid_start_qz",
)
SUCCESS_LABELS = (
    "Lift with left hand; hand off to right in middle",
    "Right hand places block in box; release and settle",
    "After placement, right hand picks up lid and closes box",
    "Block stays inside; box and lid stable; hands released",
)
