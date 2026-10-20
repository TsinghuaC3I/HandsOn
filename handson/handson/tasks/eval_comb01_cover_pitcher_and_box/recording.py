"""XY5's existing read-only task-state replay contract."""
from .layout import TASK_NAME

HIDDEN_FIELDS = (
    "phase", "pitcher_lid_moved", "pitcher_release_steps", "box_lid_moved",
    "final_stable_steps", "pitcher_completed_step", "box_completed_step",
    *(f"pitcher_start_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    *(f"box_phase_start_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
)
REPLAY_ATOL = 1.0e-6


def hidden_field_schema():
    return {
        "shape": [len(HIDDEN_FIELDS)], "dtype": "float64",
        "source": "xy5_action_aligned_task_state", "layout": list(HIDDEN_FIELDS),
        "visibility": "hidden_task_logic", "replay_domain": "task_state",
        "physical_replay": True, "replay_restore_supported": False,
        "replay_atol": REPLAY_ATOL,
    }


def task_state_envelope():
    return {
        "task_name": TASK_NAME, "field_names": ["xy5_hidden_state"],
        "hidden_field_names": ["xy5_hidden_state"],
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }


TASK_STATE_CONTRACT = {
    "field_names": ("xy5_hidden_state",),
    "layouts": {"xy5_hidden_state": HIDDEN_FIELDS},
    "replay_atols": {"xy5_hidden_state": REPLAY_ATOL},
    "hidden_field_names": ("xy5_hidden_state",),
    "source": "xy5_action_aligned_task_state",
}
