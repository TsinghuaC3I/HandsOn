"""GEN5 identity and source-derived geometry for the shared B04 stack mechanics.

No replacement visible mesh is constructed here. ``geometry.json`` is produced
only after the approved source archive has been inspected and converted.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

TASK_ID = "GEN5"
TASK_NAME = "eval_gen07_stack_boxes"
DISPLAY_NAME = "GEN5 Four Package Box Stacking"
PROMPT = (
    "Arrange the four package boxes into four distinct height layers with the "
    "large package box on the bottom layer, then keep the hand from touching "
    "the boxes for 0.5 seconds."
)
BASE_NAME = "gen5_package_base"
SMALL_NAMES = ("gen5_package_small_1", "gen5_package_small_2", "gen5_package_small_3")
BODY_NAMES = (BASE_NAME, *SMALL_NAMES)
ASSET_DIR = Path(__file__).resolve().parents[1] / "envs/xmls/eval_gen07_stack_boxes"
SCENE_XML = "arena_arm_hand_eval_gen07_stack_boxes_rokae_wuji1.xml"
RESET_XY = {
    BASE_NAME: (-0.29, 0.10),
    # The raised right wrist and fisheye flange occupy the near-left lane at
    # reset.  Put the first small carton in the lower lane so its source hull
    # cannot start inside that swept hand volume.
    SMALL_NAMES[0]: (-0.38, -0.45),
    SMALL_NAMES[1]: (-0.28, -0.22),
    SMALL_NAMES[2]: (-0.18, -0.16),
}
RESET_XY_BOUNDS = {
    name: np.asarray([np.asarray(xy) - 0.015, np.asarray(xy) + 0.015], dtype=np.float64)
    for name, xy in RESET_XY.items()
}
RESET_YAW_DEGREES = 20.0
HIDDEN_STATE_FIELDS = (
    "success_counter", "velocity_sample_count", "velocity_stable_latched",
    *(f"sample_{i}_{kind}_speed" for i in range(5) for kind in ("linear", "angular")),
)
HIDDEN_STATE_REPLAY_ATOL = 1.0e-6
SUCCESS_CONDITIONS = (
    ("stack_four_layers", "Four package boxes form four height layers"),
    ("stack_large_block_on_bottom", "The large package box is on the bottom"),
    ("stack_support_graph_valid", "Each upper box is supported by the box below"),
    ("stack_upright", "The package boxes are not excessively tilted"),
    ("stack_bottom_on_table", "The bottom package box rests on the table"),
    ("stack_hand_released", "The right hand is not touching any package box"),
    ("stack_velocity_stable", "The stack is stable"),
)


def recording_task_state() -> tuple[dict, dict]:
    """Action-aligned filter evidence, recomputed and compared during replay."""
    field = {
        "shape": [len(HIDDEN_STATE_FIELDS)], "dtype": "float64",
        "source": "gen5_action_aligned_task_state", "layout": list(HIDDEN_STATE_FIELDS),
        "visibility": "hidden_task_logic", "replay_domain": "task_state",
        "physical_replay": True, "replay_restore_supported": False,
        "replay_atol": HIDDEN_STATE_REPLAY_ATOL,
    }
    envelope = {
        "task_name": TASK_NAME, "field_names": ["gen5_hidden_state"],
        "hidden_field_names": ["gen5_hidden_state"],
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }
    return field, envelope


@lru_cache(maxsize=1)
def geometry() -> dict:
    """Read measured, centered source geometry; never substitute B04 assets."""
    with (ASSET_DIR / "geometry.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    if set(result["objects"]) != set(BODY_NAMES):
        raise ValueError("GEN5 geometry must describe exactly the four package bodies")
    for name, body in result["objects"].items():
        dimensions = np.asarray(body["dimensions_m"], dtype=np.float64)
        if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
            raise ValueError(f"GEN5 source dimensions are invalid for {name}")
    return result


@lru_cache(maxsize=1)
def half_extents() -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(body["dimensions_m"], dtype=np.float64) / 2.0
        for name, body in geometry()["objects"].items()
    }
    for extents in result.values():
        extents.setflags(write=False)
    return result


def capture_hidden_state(env) -> np.ndarray:
    """Expose the inherited success debounce without changing policy input."""
    state = np.zeros(len(HIDDEN_STATE_FIELDS), dtype=np.float64)
    samples = env._b04_velocity_samples
    state[:3] = (env._success_counter, len(samples), env._b04_velocity_stable_latched)
    if samples:
        state[3:3 + 2 * len(samples)] = np.asarray(samples).ravel()
    return state


def restore_hidden_state(env, state) -> None:
    """Restore the action-0/filter boundary, never a future trajectory frame."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (len(HIDDEN_STATE_FIELDS),) or not np.all(np.isfinite(state)):
        raise ValueError("GEN5 hidden state must contain 13 finite values")
    counter, count, latched = state[:3]
    if counter < 0 or counter != int(counter) or count != int(count) or not 0 <= count <= 5 or latched not in (0, 1):
        raise ValueError("GEN5 success/filter state is invalid")
    env._success_counter = int(counter)
    env._b04_velocity_samples = [tuple(row) for row in state[3:3 + 2 * int(count)].reshape(-1, 2)]
    env._b04_velocity_stable_latched = bool(latched)
    env._last_b04_stack_metrics = {}
