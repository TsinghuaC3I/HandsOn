"""Small, simulator-independent sequencing and layout contract for XY3.

Atomic fruit/photograph metrics are evaluated by the shared ABC runtime. This
module owns only the ordered composition and its combined reset layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

TASK_NAME = "eval_comb04_place_fruit_and_photograph_logo"
TASK_ID = "XY3"
FRUIT_NAMES = ("apple", "011_banana", "orange", "pear")
DYNAMIC_NAMES = (*FRUIT_NAMES, "bucket", "camera")
PROMPT = (
    "Pick the specified fruit, place it in the bucket and release it. "
    "Then pick up the camera, aim at the logo on the bucket and press the shutter."
)
# Bucket local Y is up. Its local +X end faces the robot/camera lane. The
# source mesh end-wall plane is n.p=0.16685 (metres); follow its measured
# slope with a 0.5 mm visual offset. No collider or supplier asset is changed.
LOGO_NORMAL_LOCAL = np.asarray((np.sqrt(0.98), -np.sqrt(0.02), 0.0))
LOGO_LOCAL = np.asarray(((0.16685 - np.sqrt(0.02) * 0.005) / np.sqrt(0.98), -0.005, -0.005)) + 0.0005 * LOGO_NORMAL_LOCAL
LOGO_ROTATION_LOCAL = np.column_stack(((0.0, 0.0, -1.0), (np.sqrt(0.02), np.sqrt(0.98), 0.0), LOGO_NORMAL_LOCAL))
# Reuse the CASIA texture/disk at a size that fits below the original rim.
# C06's 159.6 mm disk would intersect the rim/bottom on this shorter wall.
LOGO_RADIUS = 0.070

DEFAULT_XY = {
    # Keep the camera's complete grip/shutter region in the shared calibrated
    # main view. Translate bucket and camera together to preserve their gap.
    "bucket": np.asarray((-0.055, 0.160)),
    "camera": np.asarray((-0.355, 0.150)),
    # Keep the fruit lane clear of the right home arm and the camera grip.
    # These positions preserve the old XY3 tabletop composition while giving
    # the legacy collision envelopes a full 15 mm margin from robot bodies.
    "011_banana": np.asarray((-0.370, -0.265)),
    "apple": np.asarray((-0.400, -0.105)),
    "orange": np.asarray((-0.230, -0.105)),
    "pear": np.asarray((-0.125, -0.210)),
}
XY_HALF_RANGE = {
    "bucket": np.asarray((0.010, 0.012)),
    "camera": np.asarray((0.005, 0.012)),
    "011_banana": np.asarray((0.015, 0.015)),
    "apple": np.asarray((0.015, 0.015)),
    "orange": np.asarray((0.015, 0.015)),
    "pear": np.asarray((0.015, 0.015)),
}
XY_BOUNDS = {
    name: np.stack((center - XY_HALF_RANGE[name], center + XY_HALF_RANGE[name]))
    for name, center in DEFAULT_XY.items()
}
SUPPORT_MIN_Z = {
    "bucket": -0.080, "011_banana": -0.019722, "apple": -0.038,
    "orange": -0.037, "pear": -0.052, "camera": -0.065,
}
HIDDEN_FIELDS = (
    "a_stable_steps", "a_completed", "photo_completed",
    "shutter_was_down", "right_contact_grace",
)


def recording_task_state_contract() -> tuple[dict, dict]:
    """Use the existing read-only current/next task-state replay protocol."""
    fields = {"xy3_hidden_state": {
        "shape": [len(HIDDEN_FIELDS)], "dtype": "float64",
        "source": "xy3_action_aligned_task_state", "layout": list(HIDDEN_FIELDS),
        "visibility": "hidden_task_logic", "replay_domain": "task_state",
        "physical_replay": True, "replay_restore_supported": False,
        "replay_atol": 0.0,
    }}
    envelope = {
        "task_name": TASK_NAME, "field_names": ["xy3_hidden_state"],
        "hidden_field_names": ["xy3_hidden_state"],
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }
    return fields, envelope


@dataclass
class SequenceState:
    a_stable_steps: int = 0
    a_completed: bool = False
    photo_completed: bool = False

    def observe_photo(self, *, a_valid: bool, photo_valid: bool,
                      press_edge: bool) -> None:
        """Latch a physical event without advancing the control-frame A timer.

        The existing recorded photo_completed field owns the latch, including
        events that begin and end between two recorded observations.
        """
        if self.a_completed and a_valid and photo_valid and press_edge:
            self.photo_completed = True

    def update(self, *, a_valid: bool, required_steps: int,
               photo_valid: bool, press_edge: bool) -> bool:
        # A press in the very frame that completes A is not a subsequent act.
        self.observe_photo(a_valid=a_valid, photo_valid=photo_valid,
                           press_edge=press_edge)
        self.a_stable_steps = self.a_stable_steps + 1 if a_valid else 0
        if self.a_stable_steps >= required_steps:
            self.a_completed = True
        return self.photo_completed

    def as_array(self, shutter_was_down: bool, contact_grace: int) -> np.ndarray:
        return np.asarray((self.a_stable_steps, self.a_completed,
                           self.photo_completed, shutter_was_down,
                           contact_grace), dtype=np.float64)


def sample_layout(rng, *, randomize: bool,
                  is_clear: Callable[[dict, dict], bool]) -> tuple[dict, dict]:
    """Sample jointly, never replace a failed random sample by a fixed layout."""
    for _ in range(256 if randomize else 1):
        xy = {name: (rng.uniform(*XY_BOUNDS[name]) if randomize else center.copy())
              for name, center in DEFAULT_XY.items()}
        yaw = {name: 0.0 for name in DYNAMIC_NAMES}
        if randomize:
            yaw.update({name: float(rng.uniform(-180.0, 180.0)) for name in FRUIT_NAMES})
            yaw["camera"] = float(rng.uniform(-10.0, 10.0))
        if is_clear(xy, yaw):
            return xy, yaw
    raise RuntimeError("XY3 could not sample a clear combined fruit/bucket/camera layout")
