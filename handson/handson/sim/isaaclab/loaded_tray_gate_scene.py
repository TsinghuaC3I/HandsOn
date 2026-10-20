"""Isaac-independent scene metadata for B320-BC005.

The runtime imports this module before Isaac is available in interface-only
tests, so keep it limited to immutable Python data.
"""

from __future__ import annotations

import math


TASK_NAME = "loaded_tray_through_narrow_gate"
TASK_ID = "B320-BC005"
SCENE_XML = "arena_arm_hand_loaded_tray_through_narrow_gate_rokae_wuji1.xml"

TASK_SCENES = {TASK_NAME: SCENE_XML}

SUCCESS_CONDITION_LABELS = {
    TASK_NAME: (
        "Left and right hands own their corresponding long side rims",
        "Loaded tray crosses the narrow frame with continuous dual-hand support",
        "Tray and both payloads are fully beyond the frame",
        "Loaded tray is level and stable over the green target",
    )
}

TABLE_TOP_Z = 0.890
# The downloaded wooden tray is baked to task axes with one uniform scale and
# centred on its exact visual AABB.  It starts at +90 degrees yaw so its short
# face leads along world +X.  All bounds below are copied from the generated
# selected-asset manifest.
# Root Z leaves 1 mm between the real tray bottom and the table.
TRAY_HALF_EXTENTS = (0.106555481, 0.188675260, 0.025647638)
TRAY_BASE_HALF_EXTENTS = (0.099371570, 0.181324831, 0.001919625)
TRAY_USABLE_HALF_EXTENTS = (0.099371570, 0.181324830)
TRAY_DECK_TOP_LOCAL_Z = -0.021808388
TRAY_MASS_KG = 0.30
TRAY_START_YAW_DEG = 90.0
TRAY_FORWARD_HALF_EXTENT = TRAY_HALF_EXTENTS[1]
TRAY_LATERAL_HALF_EXTENT = TRAY_HALF_EXTENTS[0]
TRAY_GATE_CLEARANCE_M = 0.010
TRAY_RESET_GATE_GAP_M = 0.020

# The selected source frame has a 377.35 mm visual opening.  Collection feedback
# showed that opening was too forgiving and, more importantly, that the old
# converter boxes covered only a 20 mm slice through the 108 mm-deep visible
# wood.  The runtime now scales the visible frame in Y and authors three
# source-fitted, full-depth collision boxes from the values below.  At 340 mm
# the passage still leaves 63.4 mm nominal clearance on either side of the
# narrow-face-first tray while making alignment a real part of the task.
GATE_SOURCE_CLEAR_WIDTH = 0.377350520
GATE_CLEAR_WIDTH = 0.340
GATE_VISUAL_Y_SCALE = GATE_CLEAR_WIDTH / GATE_SOURCE_CLEAR_WIDTH

# Canonical frame root.  Moving it 25 mm toward the robot compensates for the
# newly physical post depth, keeping the authored tray centre essentially at
# its already-audited reachable X coordinate.
GATE_PLANE_X = -0.234023062

# Source-fit collision inventory in frame-local coordinates.  Positive task Y
# is the operator's left, hence that member uses the asset's source-named right
# post.  X half extents retain the full visible-member depth instead of the old
# 10 mm task slice.  The Y coordinates/extents follow the same scale as the
# visible reference so no invisible collider protrudes into the opening.
GATE_COLLIDER_BOXES = {
    "left_post": {
        "center": (
            0.001141285,
            0.248042187 * GATE_VISUAL_Y_SCALE,
            0.240000000,
        ),
        "half_extents": (
            0.034835653,
            0.059366927 * GATE_VISUAL_Y_SCALE,
            0.240000000,
        ),
    },
    "right_post": {
        "center": (
            0.000784183,
            -0.249698951 * GATE_VISUAL_Y_SCALE,
            0.240000000,
        ),
        "half_extents": (
            0.032969545,
            0.061023691 * GATE_VISUAL_Y_SCALE,
            0.240000000,
        ),
    },
    "lintel": {
        "center": (
            0.0,
            -0.001094293 * GATE_VISUAL_Y_SCALE,
            0.536812932,
        ),
        "half_extents": (
            0.054189438,
            0.340238271 * GATE_VISUAL_Y_SCALE,
            0.056812933,
        ),
    },
}

# The semantic crossing slab spans both post AABBs, including their small
# source-derived X offsets.  This is deliberately distinct from the lintel's
# deeper decorative overhang, which is still physically collidable overhead.
GATE_POST_HALF_X = max(
    abs(float(spec["center"][0])) + float(spec["half_extents"][0])
    for name, spec in GATE_COLLIDER_BOXES.items()
    if name.endswith("post")
)
GATE_POST_HEIGHT = 0.480
GATE_BASE_Z = 0.890
GATE_LINTEL_BOTTOM_Z = 1.370
TRAY_START_POSITION = (
    GATE_PLANE_X
    - GATE_POST_HALF_X
    - TRAY_RESET_GATE_GAP_M
    - TRAY_FORWARD_HALF_EXTENT,
    0.0,
    0.916647638,
)

PAYLOAD_NAMES = ("loaded_tray_can", "loaded_tray_cup")
PAYLOAD_HORIZONTAL_RADII = {
    "loaded_tray_can": 0.032697,
    "loaded_tray_cup": 0.050380,
}
# Axis-aligned reset footprints after applying each payload's authored
# orientation.  The cup's +90-degree yaw points its handle along +Y so the
# narrower body axis, rather than the handle, faces the close X rim.
PAYLOAD_HORIZONTAL_HALF_EXTENTS = {
    "loaded_tray_can": (0.032697, 0.032697),
    "loaded_tray_cup": (0.036520, 0.050380),
}
PAYLOAD_HALF_HEIGHTS = {
    "loaded_tray_can": 0.049697,
    "loaded_tray_cup": 0.042680,
}
PAYLOAD_START_LOCAL_XY = {
    "loaded_tray_can": (-0.025, -0.070),
    "loaded_tray_cup": (0.025, 0.070),
}
# Rz(+90 degrees) maps tray-local (x, y) to world (-y, x).
PAYLOAD_START_POSITIONS = {
    "loaded_tray_can": (
        TRAY_START_POSITION[0] - PAYLOAD_START_LOCAL_XY["loaded_tray_can"][1],
        PAYLOAD_START_LOCAL_XY["loaded_tray_can"][0],
        0.944536250,
    ),
    "loaded_tray_cup": (
        TRAY_START_POSITION[0] - PAYLOAD_START_LOCAL_XY["loaded_tray_cup"][1],
        PAYLOAD_START_LOCAL_XY["loaded_tray_cup"][0],
        0.937519250,
    ),
}

# The downloaded end handles remain visible, but they are not viable transport
# grips after the 90-degree rotation: one arm would have to follow the leading
# end through the frame.  The task therefore uses short, source-derived
# segments on the two opposite long side rims.  Both palms remain side by side
# throughout transport, and real contact in the selected rim region is still
# required for role ownership.
HANDLE_ARCH_LOCAL_X_RANGE = (-0.060978176, 0.060978176)
SIDE_RIM_GRIP_LOCAL_Y_RANGE = (-0.050, 0.050)
SIDE_RIM_GRIP_LOCAL_Z_RANGE = (-0.011419748, 0.002808142)
SIDE_RIM_GRIP_LOCAL_Z = SIDE_RIM_GRIP_LOCAL_Z_RANGE[1]
LEFT_GRIP_LOCAL_POSITION = (
    0.102963523,
    0.0,
    SIDE_RIM_GRIP_LOCAL_Z,
)
RIGHT_GRIP_LOCAL_POSITION = (
    -0.102963529,
    0.0,
    SIDE_RIM_GRIP_LOCAL_Z,
)
# The visible/collision frame is narrowed as one aligned assembly.
GATE_OUTER_WIDTH = 0.680476546 * GATE_VISUAL_Y_SCALE
GATE_PHYSICAL_POST_OUTER_WIDTH = 0.618131756 * GATE_VISUAL_Y_SCALE
GATE_LINTEL_HEIGHT = 0.113625866
TARGET_HALF_EXTENTS = (0.205, 0.120)
TARGET_CENTER_XY = (
    GATE_PLANE_X + GATE_POST_HALF_X + 0.010 + TARGET_HALF_EXTENTS[0],
    0.0,
)
TARGET_CENTER_TOLERANCE = 0.080

# Most-rear terminal tray centre that both clears the far frame face and lies
# inside the green target disk.  Keeping this derived avoids silently moving
# the hands to the target centre when the tray itself still spans the frame.
TRAY_FULLY_CROSSED_MIN_CENTER_X = (
    GATE_PLANE_X
    + GATE_POST_HALF_X
    + TRAY_GATE_CLEARANCE_M
    + TRAY_FORWARD_HALF_EXTENT
)
TRAY_PLACED_CENTER_X = max(
    TRAY_FULLY_CROSSED_MIN_CENTER_X,
    TARGET_CENTER_XY[0] - TARGET_CENTER_TOLERANCE,
)
if TRAY_PLACED_CENTER_X > TARGET_CENTER_XY[0] + TARGET_CENTER_TOLERANCE:
    raise ValueError("BC005 target disk does not intersect the fully-crossed range")
# Scripted validation aims 10 mm beyond the exact acceptance boundary so
# normal pose/contact noise cannot leave the tray tangent to the frame plane.
# This is only a route target; it does not relax the physical success test.
TRAY_SCRIPTED_PLACED_CENTER_X = min(
    TRAY_PLACED_CENTER_X + 0.010,
    TARGET_CENTER_XY[0] + TARGET_CENTER_TOLERANCE,
)


def tray_local_position_to_world(
    tray_position: tuple[float, float, float],
    local_position: tuple[float, float, float],
    *,
    yaw_deg: float = TRAY_START_YAW_DEG,
) -> tuple[float, float, float]:
    """Transform one tray-local point with the authored planar tray yaw."""
    yaw = math.radians(float(yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local_x, local_y, local_z = local_position
    return (
        float(tray_position[0]) + cosine * local_x - sine * local_y,
        float(tray_position[1]) + sine * local_x + cosine * local_y,
        float(tray_position[2]) + local_z,
    )


__all__ = [
    "GATE_COLLIDER_BOXES",
    "GATE_CLEAR_WIDTH",
    "GATE_BASE_Z",
    "GATE_LINTEL_BOTTOM_Z",
    "GATE_PLANE_X",
    "GATE_POST_HALF_X",
    "GATE_POST_HEIGHT",
    "GATE_OUTER_WIDTH",
    "GATE_PHYSICAL_POST_OUTER_WIDTH",
    "GATE_SOURCE_CLEAR_WIDTH",
    "GATE_VISUAL_Y_SCALE",
    "GATE_LINTEL_HEIGHT",
    "HANDLE_ARCH_LOCAL_X_RANGE",
    "LEFT_GRIP_LOCAL_POSITION",
    "PAYLOAD_HORIZONTAL_HALF_EXTENTS",
    "PAYLOAD_HORIZONTAL_RADII",
    "PAYLOAD_HALF_HEIGHTS",
    "PAYLOAD_NAMES",
    "PAYLOAD_START_LOCAL_XY",
    "PAYLOAD_START_POSITIONS",
    "RIGHT_GRIP_LOCAL_POSITION",
    "SCENE_XML",
    "SIDE_RIM_GRIP_LOCAL_Y_RANGE",
    "SIDE_RIM_GRIP_LOCAL_Z",
    "SIDE_RIM_GRIP_LOCAL_Z_RANGE",
    "SUCCESS_CONDITION_LABELS",
    "TABLE_TOP_Z",
    "TARGET_CENTER_TOLERANCE",
    "TARGET_CENTER_XY",
    "TARGET_HALF_EXTENTS",
    "TASK_ID",
    "TASK_NAME",
    "TASK_SCENES",
    "TRAY_BASE_HALF_EXTENTS",
    "TRAY_DECK_TOP_LOCAL_Z",
    "TRAY_FORWARD_HALF_EXTENT",
    "TRAY_GATE_CLEARANCE_M",
    "TRAY_HALF_EXTENTS",
    "TRAY_LATERAL_HALF_EXTENT",
    "TRAY_FULLY_CROSSED_MIN_CENTER_X",
    "TRAY_MASS_KG",
    "TRAY_PLACED_CENTER_X",
    "TRAY_RESET_GATE_GAP_M",
    "TRAY_SCRIPTED_PLACED_CENTER_X",
    "TRAY_START_POSITION",
    "TRAY_START_YAW_DEG",
    "TRAY_USABLE_HALF_EXTENTS",
    "tray_local_position_to_world",
]
