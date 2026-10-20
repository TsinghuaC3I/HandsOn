"""XY2 composition history and seeded layout; atomic predicates live upstream."""

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping

import numpy as np

from .fu015_contract import (
    FU015ResetState, SUCCESS_HOLD_STEPS, TARGET_OPEN_CHOICES_DEG,
    sample_reset_state as sample_laptop_reset,
)

TASK_ID = "XY2"
TASK_NAME = "eval_comb02_open_laptop_and_click_mouse"
SOURCE_TASK_IDS = ("B320-FU015", "C05")
TASK_PROMPT = (
    "Open the initially closed laptop to the requested angle with either hand "
    "and let it settle; you may release the screen. Move the mouse onto its pad "
    "without pressing either button, then click the left button once. Keep the laptop at the target angle and the "
    "mouse on the pad while the scene settles."
)
TIME_LIMIT_SECONDS = 120.0
LAPTOP_XY = np.asarray([-0.06, 0.07])
LAPTOP_XY_HALF_RANGE = np.asarray([0.04, 0.025])
MOUSE_XY = np.asarray([-0.34, -0.41])
PAD_XY = np.asarray([-0.08, -0.30])
MOUSE_XY_HALF_RANGE = np.asarray([0.025, 0.025])
PAD_XY_HALF_RANGE = np.asarray([0.025, 0.025])
LAYOUT_ATTEMPTS = 128
FAILURE_CODES = {"": 0, "fu015_physical_failure": 1, "c05_early_click": 2}
RECORDING_LAYOUTS = {
    "xy2_history": (
        "phase", "laptop_steps", "mouse_steps", "buttons_armed", "a_completed_frame",
        "mouse_start_x", "mouse_start_y", "early_click_latched", "button_baseline_pending",
        "left_button_baseline", "right_button_baseline", "last_evaluated_frame", "failure_code",
        "left_click_seen",
    ),
    "xy2_laptop_control": (
        "drive_target_deg", "assist_active", "assist_substeps", "owner_substeps",
        "previous_right_tcp_z", "right_command_delta_z", "opening_intent",
        "display_contact_active", "hinge_lock_active", "hinge_lock_angle_deg", "physics_substeps",
    ),
    "xy2_left_laptop_control": (
        "previous_left_tcp_z", "opening_intent", "display_contact_active", "owner_substeps",
    ),
    "xy2_fu015_state": (
        "seed", "root_x", "root_y", "root_yaw_deg", "initial_hinge_deg", "target_angle_deg",
        "friction", "mass_scale", "drive_target_deg", "assist_active", "assist_substeps",
        "success_counter", "failure_latched", "hinge_error_deg", "base_displacement",
        "base_lift", "source_success_now",
    ),
}
RECORDING_ATOLS = {name: 1.0e-6 for name in RECORDING_LAYOUTS}
# Source USD bounds, extended in its depth direction for the 0..120° sweep.
# This conservative planar screen is a sampler, not the PhysX overlap proof.
LAPTOP_SWEEP_MIN_LOCAL_XY = np.asarray([-0.165, -0.144])
LAPTOP_SWEEP_MAX_LOCAL_XY = np.asarray([0.165, 0.201])


@dataclass(frozen=True)
class XY2Layout:
    laptop: FU015ResetState
    mouse_xy: tuple[float, float]
    pad_xy: tuple[float, float]
    mouse_yaw_deg: float

    def to_dict(self):
        return asdict(self)


def laptop_sweep_clearance(laptop: FU015ResetState, xy, radius: float) -> float:
    angle = math.radians(laptop.root_yaw_deg)
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    local = rotation.T @ (np.asarray(xy) - [laptop.root_x_m, laptop.root_y_m])
    nearest = np.clip(local, LAPTOP_SWEEP_MIN_LOCAL_XY, LAPTOP_SWEEP_MAX_LOCAL_XY)
    return float(np.linalg.norm(local - nearest) - radius)


def layout_clearances(layout: XY2Layout) -> dict[str, float]:
    return {
        "mouse_laptop_sweep": laptop_sweep_clearance(layout.laptop, layout.mouse_xy, 0.14),
        "pad_laptop_sweep": laptop_sweep_clearance(layout.laptop, layout.pad_xy, 0.15),
        "mouse_pad_offset": float(np.linalg.norm(np.subtract(layout.mouse_xy, layout.pad_xy))) - 0.22,
    }


def sample_layout(seed: int, *, randomize: bool, target_angle=None) -> XY2Layout:
    if target_angle is not None and float(target_angle) not in TARGET_OPEN_CHOICES_DEG:
        raise ValueError(f"XY2 target must be one of {TARGET_OPEN_CHOICES_DEG}")
    laptop = sample_laptop_reset(seed)
    if target_angle is not None:
        laptop = replace(laptop, target_open_deg=float(target_angle))
    # Like FU015, this is an episode-seeded local RNG; no global RNG or hashes.
    rng = np.random.RandomState(int(seed) % (2**32 - 1))
    for _ in range(LAYOUT_ATTEMPTS):
        xy = LAPTOP_XY + (rng.uniform(-LAPTOP_XY_HALF_RANGE, LAPTOP_XY_HALF_RANGE) if randomize else 0)
        yaw = -90.0 + (rng.uniform(-10.0, 10.0) if randomize else 0.0)
        mouse = MOUSE_XY + (rng.uniform(-MOUSE_XY_HALF_RANGE, MOUSE_XY_HALF_RANGE) if randomize else 0)
        pad = PAD_XY + (rng.uniform(-PAD_XY_HALF_RANGE, PAD_XY_HALF_RANGE) if randomize else 0)
        layout = XY2Layout(
            replace(laptop, root_x_m=float(xy[0]), root_y_m=float(xy[1]), root_yaw_deg=float(yaw)),
            tuple(float(v) for v in mouse), tuple(float(v) for v in pad),
            float(rng.uniform(-10.0, 10.0)) if randomize else 0.0,
        )
        clearances = layout_clearances(layout)
        if (clearances["mouse_laptop_sweep"] >= 0.015
                and clearances["pad_laptop_sweep"] >= 0.015
                and clearances["mouse_pad_offset"] >= 0):
            return layout
    raise ValueError(f"XY2 could not sample a clear joint layout for seed {seed}")


def laptop_retained(metrics: Mapping) -> bool:
    return bool(metrics["finite"] and not metrics["failure"]
                and metrics["hinge_target_ok"] and metrics["base_stable"]
                and metrics["velocity_stable"])


class XY2Progress:
    """Advance once per control transition, never in an observation getter."""

    def __init__(self, mouse_hold_steps: int):
        self.mouse_hold_steps = mouse_hold_steps
        self.reset()

    def reset(self):
        self.phase = 0  # 0: laptop, 1: mouse, 2: complete
        self.laptop_steps = 0
        self.mouse_steps = 0
        self.buttons_armed = False
        self.left_click_seen = False
        self.a_completed_frame = -1
        self.mouse_start_xy = np.zeros(2, dtype=np.float64)
        self.failure_reason = ""

    def observe_buttons(self, *, laptop: Mapping, mouse: Mapping, buttons_neutral: bool):
        """Remember a valid press edge, including presses between control frames."""
        if self.phase != 1:
            return
        if buttons_neutral:
            self.buttons_armed = True
        else:
            if self.buttons_armed and laptop_retained(laptop) and mouse["c05_left_clicked"]:
                self.left_click_seen = True
            self.buttons_armed = False

    def update(self, *, frame: int, laptop: Mapping,
               mouse_xy, mouse: Mapping, buttons_neutral: bool) -> bool:
        if self.failure_reason:
            return False
        if laptop["failure"]:
            self.failure_reason = "fu015_physical_failure"
            return False
        if self.phase == 0:
            self.laptop_steps = self.laptop_steps + 1 if laptop_retained(laptop) else 0
            if self.laptop_steps >= SUCCESS_HOLD_STEPS:
                self.phase = 1
                self.a_completed_frame = frame
                self.mouse_start_xy = np.asarray(mouse_xy, dtype=np.float64).copy()
                self.buttons_armed = bool(buttons_neutral)
            return False
        if mouse["c05_failed_early_click"]:
            self.failure_reason = "c05_early_click"
            return False
        self.observe_buttons(laptop=laptop, mouse=mouse, buttons_neutral=buttons_neutral)
        valid = bool(laptop_retained(laptop) and mouse["c05_mouse_on_pad"])
        if not valid:
            self.left_click_seen = False
        valid = valid and self.left_click_seen
        self.mouse_steps = self.mouse_steps + 1 if valid else 0
        if self.mouse_steps >= self.mouse_hold_steps:
            self.phase = 2
        return self.phase == 2 and valid
