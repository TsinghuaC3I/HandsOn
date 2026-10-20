"""CPU-only identity and inherited D03 requirements for GEN4."""

TASK_ID = "GEN4"
HIDDEN_STATE_FIELDS = (
    "stage", "left_streak", "both_streak", "right_streak",
    "left_confirmed", "both_confirmed", "right_confirmed", "direct_right_seen",
    "success_counter", "lost_steps",
)
TASK_NAME = "eval_gen04_pass_glue_stick"
PROMPT = (
    "Pick up the glue stick with the left hand, transfer it to the right hand "
    "in the middle, then place it stably inside the right tray and release it."
)
CONDITION_LABELS = (
    "Left hand lifts the glue stick first",
    "Both hands contact the glue stick in the transfer zone",
    "Right hand receives after the left hand releases",
    "Glue stick released and supported inside the right tray",
)
GLUE_BODY = "gen4_glue"
TRAY_BODY = "gen4_tray"
DISTRACTORS = (
    "d03_flat_screwdriver", "d03_phillips_screwdriver", "d03_spoon",
    "d03_plastic_bolt", "d03_a_marbles", "d03_b_marble_1",
    "d03_b_marble_2", "d03_hammer", "d03_nails", "d03_bowl",
)
TABLE_TOP_M = 0.890
GLUE_XY = (-0.10, 0.34)
GLUE_XY_HALF_RANGE = (0.03, 0.03)
GLUE_YAW_DEG = 15.0
# Keep the legacy tray's collision envelope clear of the right home hand;
# the old placement touched the first palm capsule during formal reset.
TRAY_XY = (-0.14, -0.48)
TRAY_YAW_DEG = 90.0
LIFT_CLEARANCE_M = 0.050
OWNERSHIP_FRAMES = 2
SUCCESS_FRAMES = 8
LINEAR_SPEED_MAX = 0.08
ANGULAR_SPEED_MAX = 1.8
HAND_CONTACT_FORCE_N = 0.15
TIME_LIMIT_S = 180.0
CONTROL_HZ = 30
PHYSICS_HZ = 120
PROPRIO_SHAPES = {
    "tcp_pose": (14,), "gripper_pose": (32,), "target_pose": (7,),
    "box_pose": (7,), "target_to_box": (3,), "handoff_metrics": (10,),
    "distractor_poses": (70,), "table_delta_height": (1,),
}
