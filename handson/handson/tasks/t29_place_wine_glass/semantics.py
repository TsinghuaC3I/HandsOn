"""Immutable physical contract constants for YCB09.

The runtime deliberately keeps all stateful latches in the Isaac environment;
this module is the single source of truth for geometry and acceptance values.
"""

from __future__ import annotations

import math

import numpy as np


TASK_ID = "YCB09"
TASK_NAME = "t29_place_wine_glass"
GLASS_NAME = "wine_glass"
PLATE_NAME = "plate"

# Original, unscaled Lightwheel YCB collision bounds (metres).
GLASS_LOCAL_MIN_Z_M = -0.0685675
GLASS_LOCAL_MAX_Z_M = 0.0685675
PLATE_LOCAL_MIN_Z_M = -0.0123455
PLATE_LOCAL_MAX_Z_M = 0.0123455
# Exact maximum XY radii measured over every vertex in the original,
# unscaled collision decompositions.  Use the swept radii for conservative
# table and arbitrary-yaw clearance checks rather than axis-aligned half sizes.
GLASS_FOOTPRINT_RADIUS_M = 0.0453315268907854
PLATE_FOOTPRINT_RADIUS_M = 0.13109281035396259
GLASS_BASE_LOCAL_POINT_M = np.asarray(
    [0.0, 0.0, GLASS_LOCAL_MIN_Z_M],
    dtype=np.float64,
)

# Central, right-hand-reachable reset lanes. Both bodies receive independent,
# seeded position samples. The full plate footprint remains on the table.
RESET_XY_BOUNDS_M = {
    GLASS_NAME: np.asarray([[-0.39, -0.30], [-0.19, -0.10]], dtype=np.float64),
    # Both domains lie inside the shared recording workspace, so no sample
    # is clipped to a fixed boundary. The physical pair/hand sampler rejects
    # overlap while retaining independent positions over these wider lanes.
    PLATE_NAME: np.asarray([[-0.39, -0.08], [-0.17, 0.10]], dtype=np.float64),
}
RESET_MIN_CENTER_DISTANCE_M = 0.200
TABLE_X_BOUNDS_M = (-0.55, 0.25)
TABLE_Y_BOUNDS_M = (-0.80, 0.80)

HIDDEN_STATE_SIZE = 17

LIFT_HEIGHT_M = 0.025
LIFT_HOLD_SECONDS = 0.15
TARGET_CENTER_TOLERANCE_M = 0.025
MAX_TILT_DEG = 10.0
MIN_UPRIGHT_ALIGNMENT = float(math.cos(math.radians(MAX_TILT_DEG)))
HAND_CLEAR_SECONDS = 0.50
STABILITY_HOLD_SECONDS = 0.75
MAX_LINEAR_SPEED_M_S = 0.020
MAX_ANGULAR_SPEED_RAD_S = 0.20
OFF_TABLE_DEBOUNCE_SECONDS = 0.20
CONTACT_FORCE_THRESHOLD_N = 1.0e-4


def required_steps(seconds: float, control_dt: float) -> int:
    """Convert a physical hold duration to a fail-safe control-frame count."""

    return max(1, int(math.ceil(float(seconds) / max(float(control_dt), 1.0e-6))))


__all__ = [name for name in globals() if name.isupper()] + ["required_steps"]
