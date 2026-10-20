"""Identity and sequential episode state for HS005 + KA021 (XY4)."""

from dataclasses import dataclass

TASK_NAME = "eval_comb05_scrub_bottle_then_fill_cup"
TASK_ID = "XY4"
TASK_LABEL = "XY4 Scrub Bottle Then Fill Cup"
TASK_PROMPT = (
    "Stabilize the bottle with the left hand and grasp the cleaning brush by "
    "its handle with the right hand. Scrub the bottle bands in order, either "
    "lower-middle-upper or upper-middle-lower, then release both objects and leave the bottle upright "
    "and stable. Next, hold the mug under the faucet with the left hand and "
    "open the faucet with the right hand. Keep the stream entering the upright "
    "mug for 0.5 seconds, then place the mug on the marked counter pad outside "
    "the sink and close the faucet. Leave the scrubbed bottle upright."
)
OPERATOR_STEPS = (
    "Stabilize the bottle with the left hand and hold the brush handle with the right hand. Scrub lower-middle-upper or upper-middle-lower.",
    "Put down the brush and release the bottle. Move both hands clear and wait until the bottle is upright and stable.",
    "Hold the mug under the outlet with the left hand and open the faucet with the right hand. Fill the mug for at least 0.5 seconds.",
    "Place the mug on the marked counter pad outside the sink, close the faucet, and keep the bottle upright.",
)
TIME_LIMIT_SECONDS = 90
CONTROL_HZ = 30
PHYSICS_HZ = 120
X_SUCCESS_LABEL = "HS005 bottom-to-top or top-to-bottom scrub, release and stability completed"
BOTTLE_RETAINED_LABEL = "Scrubbed bottle remains released, upright and in place"


@dataclass
class XY4Sequence:
    """An event latch, never an object/robot controller or policy input."""

    x_complete_step: int = -1
    last_update_step: int = -1

    @property
    def x_complete(self) -> bool:
        return self.x_complete_step >= 0

    def advance_x(self, *, step: int, success: bool) -> bool:
        if step <= self.last_update_step:
            return False
        self.last_update_step = int(step)
        if not self.x_complete and success:
            self.x_complete_step = int(step)
            return True
        return False

    def y_enabled(self, step: int) -> bool:
        # Do not count even the boundary control frame towards Y.
        return self.x_complete and step > self.x_complete_step

    def reset(self) -> None:
        self.x_complete_step = -1
        self.last_update_step = -1


# All decision-affecting scrub/fill history is explicit recording telemetry.
HIDDEN_SCALARS = (
    "_hs005_order_progress", "_hs005_order_violation", "_hs005_path_length",
    "_hs005_left_contact_seen", "_hs005_right_contact_seen",
    "_hs005_right_safe_grasp_seen", "_hs005_right_safe_grasp_now",
    "_hs005_right_bristle_touch_now", "_hs005_right_grasp_surface_error",
    "_hs005_right_grasp_tip_count", "_hs005_release_seen",
    "_hs005_stability_steps", "_hs005_stage_index",
    "_ka021_fill_hold_frames", "_ka021_water_filled", "_success_counter",
)
HIDDEN_FIELDS = (
    *(name.removeprefix("_") for name in HIDDEN_SCALARS),
    "coverage_lower", "coverage_middle", "coverage_upper",
    "previous_bristle_valid", "previous_bristle_x", "previous_bristle_y", "previous_bristle_z",
    "x_complete_step", "sequence_last_update_step", "fill_last_update_step", "xy4_fill_update_step",
    *(f"bottle_reset_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    *(f"brush_reset_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    *(f"hs005_reset_parameter_{i}" for i in range(9)),
    "park_x", "park_y", "park_z",
    "xy4_scrub_direction",
)
