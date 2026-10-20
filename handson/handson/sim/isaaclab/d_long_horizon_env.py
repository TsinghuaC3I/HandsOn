"""Native IsaacLab runtime for the D06--D19 long-horizon tasks."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _D06_LOWERED_ARM_HOME,
    _D06_LOWERED_LEFT_START_TCP,
    _D06_LOWERED_RIGHT_START_TCP,
    _D08_BUTTON_DAMPING,
    _D08_BUTTON_MAX_FORCE,
    _D08_BUTTON_STIFFNESS,
    _D08_BUTTON_UP_TARGET,
    _HS005_CLEAR_ARM_HOME,
    _HS005_CLEAR_LEFT_START_TCP,
    _HS005_CLEAR_RIGHT_START_TCP,
    _RAISED_ARM_HOME,
    _RAISED_LEFT_START_TCP,
    _RAISED_RIGHT_START_TCP,
    _invert_pose_matrix,
    _pose_vector_to_matrix,
    _quat_wxyz_to_matrix,
    _randomized_physx_pose,
    _TASK_ROOT,
    _torch_view,
)
from . import d20_microwave_hot_dog as _d20
from . import d20_stages as _d20_stages
from . import d20_tool_grip as _d20_grip
from .d_long_horizon_scene import D_LONG_TASK_BODIES, D_LONG_TASK_NAMES
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE


_D06 = "t42_sort_objects"
_D07 = "d07_organize_office_desk"
from . import d08_realistic_variant as _d08_variant

_D08 = "d08_toast_bread"
_D09 = "t31_prepare_coffee"
_D10 = "d10_load_dishwasher"
_D11 = "d11_presoak_pan"
_D12 = "d12_prepare_cutting_station"
_D13 = "d13_prepare_tea_tray"
_D14 = "d14_wash_fruit"
_D15 = "d15_pack_matching_lunches"
_D16 = "d16_sweep_blocks_to_bin"
_D17 = "d17_store_laptop_headphones"
_D18 = "d18_pack_eggs"
_D19 = "d19_memory_cover_sort"
_D20 = _d20.TASK_NAME

# Match the established ABC reset convention: genuinely bimanual tasks keep
# both hands at working height, while sequential one-hand tasks start with the
# right hand at the ordinary tabletop home and park only the inactive left
# hand.  D07 and D08 use the ordinary two-hand home.  D06, D11, and D12 share
# lowered bimanual home: it remains above their fixtures while cutting 25 cm
# from the fully raised pose's 55 cm TCP offset.  D10 keeps
# the full raised home because its tall dishwasher occupies the lower hand
# lanes.  Their reset samplers use the live palm/finger envelope, rather than
# treating every arm-link origin as a tabletop hand obstacle, so random
# layouts remain feasible and collision-free.
_D_LONG_SINGLE_HAND_START_TASKS = frozenset()
_D_LONG_LOWERED_START_TASKS = frozenset({_D06, _D11, _D12})
_D_LONG_DUAL_RAISED_START_TASKS = frozenset(
    {_D10, _D13, _D14, _D15, _D16, _D17, _D18, _D19}
)

_D06_TARGETS = (
    "d06_cracker_box",
    "d06_sugar_box",
    "d06_tomato_soup_can",
    "d06_tuna_fish_can",
    "d06_bowl",
    "d06_mug",
)
_D06_BASKETS = ("d06_basket_box", "d06_basket_can", "d06_basket_dish")
_D06_BASKET_BORDER_GEOMS = {
    basket: (
        f"{basket}_border",
        f"{basket}_border_left",
        f"{basket}_border_right",
    )
    for basket in _D06_BASKETS
}
_D06_CATEGORY = {
    "d06_cracker_box": "d06_basket_box",
    "d06_sugar_box": "d06_basket_box",
    "d06_tomato_soup_can": "d06_basket_can",
    "d06_tuna_fish_can": "d06_basket_can",
    "d06_bowl": "d06_basket_dish",
    "d06_mug": "d06_basket_dish",
}
_D06_FOOTPRINT_HALF_EXTENTS = {
    "d06_cracker_box": (0.060, 0.080),
    "d06_sugar_box": (0.048, 0.090),
    "d06_tomato_soup_can": (0.034, 0.034),
    "d06_tuna_fish_can": (0.044, 0.044),
    "d06_bowl": (0.074, 0.074),
    "d06_mug": (0.045, 0.078),
}
_D06_RANDOMIZATION_FOOTPRINT_RADII = {
    name: float(np.hypot(*half_extents))
    for name, half_extents in _D06_FOOTPRINT_HALF_EXTENTS.items()
}
_D06_RANDOMIZATION_FOOTPRINT_RADII.update(
    {
        "d06_distractor_0": float(np.hypot(0.032, 0.026)),
        "d06_distractor_1": 0.031,
        "d06_distractor_2": 0.025,
        "d06_distractor_3": 0.040,
        "d06_distractor_4": float(np.hypot(0.028, 0.034)),
        "d06_distractor_5": 0.028,
        "d06_distractor_6": float(np.hypot(0.018, 0.063)),
        "d06_distractor_7": 0.032,
        "d06_distractor_8": float(np.hypot(0.044, 0.018)),
        "d06_distractor_9": 0.040,
        "d06_distractor_10": float(np.hypot(0.017, 0.055)),
        "d06_distractor_11": 0.024,
    }
)
_D06_DISTRACTORS = tuple(f"d06_distractor_{index}" for index in range(12))
_D06_DISTRACTOR_DEFAULT_POSITIONS = {
    "d06_distractor_0": (-0.46, -0.48, 0.920),
    "d06_distractor_1": (-0.22, -0.55, 0.922),
    "d06_distractor_2": (-0.47, 0.02, 0.925),
    "d06_distractor_3": (-0.22, 0.24, 0.918),
    "d06_distractor_4": (-0.45, 0.49, 0.920),
    "d06_distractor_5": (-0.20, 0.67, 0.918),
    "d06_distractor_6": (-0.18, -0.23, 0.908),
    "d06_distractor_7": (-0.48, 0.22, 0.913),
    "d06_distractor_8": (-0.29, -0.27, 0.915),
    "d06_distractor_9": (-0.17, 0.43, 0.916),
    "d06_distractor_10": (-0.45, -0.65, 0.907),
    "d06_distractor_11": (-0.20, 0.04, 0.914),
}
_D07_OBJECTS = (
    "d07_keyboard",
    "d07_mouse",
    "d07_figurine",
    "d07_alarm_clock",
    "d07_misc_0",
    "d07_misc_1",
    "d07_misc_2",
)
_D08_BREAD = ("d08_bread_0", "d08_bread_1")
_D08_LEVER_PRESS_TRAVEL = 0.035
_D08_LEVER_LATCH_TRAVEL = 0.050
_D08_LEVER_RETURN_TRAVEL = 0.006
# Compatibility alias retained for existing recordings and analysis code.
_D08_BUTTON_PRESS_TRAVEL = _D08_LEVER_PRESS_TRAVEL
_D08_TOAST_CYCLE_SECONDS = 2.0
_D08_POPPED_MIN_LOCAL_Z = 0.125
_D08_RELEASE_HOLD_SECONDS = 0.5
_D08_BREAD_RACK_CENTER_LOCAL_Z = 0.137
_D08_LEVER_IDLE_STIFFNESS = 110.0
_D08_LEVER_IDLE_DAMPING = 8.0
_D08_LEVER_IDLE_MAX_FORCE = 5.0
_D08_RACK_COLLISION_NAMES = (
    "d08_toaster_rack_slot_0",
    "d08_toaster_rack_slot_1",
)
_D08_HIDDEN_COLLISION_VISUALS = {
    "d08_toaster_collision_base",
    "d08_toaster_collision_side_n",
    "d08_toaster_collision_side_p",
    "d08_toaster_collision_divider",
    "d08_toaster_collision_front",
    "d08_toaster_collision_back",
    "d08_slot_0_guide_n",
    "d08_slot_0_guide_p",
    "d08_slot_0_guide_front",
    "d08_slot_0_guide_back",
    "d08_slot_1_guide_n",
    "d08_slot_1_guide_p",
    "d08_slot_1_guide_front",
    "d08_slot_1_guide_back",
    "d08_toaster_rack_slot_0",
    "d08_toaster_rack_slot_1",
    "d08_toaster_low_stop_slot_0",
    "d08_toaster_low_stop_slot_1",
    "d08_toaster_lever_collision",
    "d08_toaster_moving_rack_collision",
    "d08_bread_tray_floor",
    "d08_bread_0_collision",
    "d08_bread_1_collision",
}
_D10_DISHES = ("d10_mug", "d10_bowl", "d10_plate")
_D13_PAYLOADS = ("d13_kettle", "d13_mug", "d13_tea_bag")
_D14_FRUITS = ("d14_banana", "d14_apple", "d14_orange")
_D15_FOODS = tuple(
    f"d15_food_{kind}_{index}" for kind in range(3) for index in range(4)
)
_D15_BOXES = ("d15_box_left", "d15_box_right")
_D15_LIDS = ("d15_lid_left", "d15_lid_right")
_D16_BLOCKS = tuple(f"d16_block_{index}" for index in range(5))
_D16_TOOLS = ("d16_broom", "d16_dustpan")
_D18_EGGS = tuple(f"d18_egg_{index}" for index in range(4))
_D19_BLOCKS = ("d19_red_block", "d19_green_block", "d19_blue_block")
_D19_COVERS = tuple(f"d19_cover_{index}" for index in range(3))
_D19_COLOR_SEQUENCE = _D19_BLOCKS

_DLONG_RESET_MAX_ATTEMPTS = 160
_DLONG_RESET_CANDIDATES_PER_OBJECT = 96
_D06_RESET_CANDIDATES_PER_OBJECT = 256
_DLONG_RESET_MIN_CLEARANCE_SLACK = 0.005
_DLONG_OBJECT_HAND_CLEARANCE = 0.085
_DLONG_OBJECT_HAND_SURFACE_CLEARANCE = 0.025
_DLONG_DEFAULT_PAIR_DISTANCE = 0.08
_DLONG_HAND_CLEARANCE_MAX_Z = 1.25

# Match the spherical arm-target limiter in bimanual_assembly_env.  Long-
# horizon task bounds use this shared model to keep tabletop interaction
# points inside at least one arm's workspace with a positive IK margin.
_DLONG_ARM_BASE_XY = np.asarray(
    [[-0.75, -0.065], [-0.75, 0.065]], dtype=np.float64
)
_DLONG_ARM_BASE_Z = 1.34
_DLONG_ARM_WORKSPACE_RADIUS = 0.78
_DLONG_TABLETOP_PALM_Z = 0.92
_DLONG_MIN_REACH_MARGIN = 0.02
_D11_MIN_REACH_MARGIN = 0.08


def _dlong_arm_reach_margins(
    xy: np.ndarray, *, palm_z: float
) -> np.ndarray:
    vertical_distance = abs(_DLONG_ARM_BASE_Z - float(palm_z))
    horizontal_radius_sq = (
        _DLONG_ARM_WORKSPACE_RADIUS**2 - vertical_distance**2
    )
    if horizontal_radius_sq <= 0.0:
        return np.full(len(_DLONG_ARM_BASE_XY), -math.inf, dtype=np.float64)
    horizontal_radius = math.sqrt(horizontal_radius_sq)
    horizontal_distances = np.linalg.norm(
        _DLONG_ARM_BASE_XY
        - np.asarray(xy, dtype=np.float64).reshape(1, 2),
        axis=1,
    )
    return horizontal_radius - horizontal_distances


def _dlong_arm_reach_margin(xy: np.ndarray, *, palm_z: float) -> float:
    return float(_dlong_arm_reach_margins(xy, palm_z=palm_z).max())


# Center height and half-height of each D06 body's conservative collision
# envelope.  Like the proven B04/B05 reset samplers, D06 must measure a real
# 3-D point-to-body distance: projecting both working hands into the table
# plane makes the dense 18-object layout mathematically infeasible and used to
# collapse every object onto the same midpoint fallback.
_D06_RANDOMIZATION_VERTICAL_ENVELOPES = {
    "d06_cracker_box": (0.997, 0.107),
    "d06_sugar_box": (0.979, 0.089),
    "d06_tomato_soup_can": (0.940, 0.050),
    "d06_tuna_fish_can": (0.907, 0.017),
    "d06_bowl": (0.917, 0.027),
    "d06_mug": (0.936, 0.045),
    "d06_distractor_0": (0.920, 0.030),
    "d06_distractor_1": (0.922, 0.031),
    "d06_distractor_2": (0.925, 0.035),
    "d06_distractor_3": (0.918, 0.028),
    "d06_distractor_4": (0.920, 0.030),
    "d06_distractor_5": (0.918, 0.028),
    "d06_distractor_6": (0.908, 0.018),
    "d06_distractor_7": (0.913, 0.023),
    "d06_distractor_8": (0.915, 0.025),
    "d06_distractor_9": (0.916, 0.026),
    "d06_distractor_10": (0.907, 0.017),
    "d06_distractor_11": (0.914, 0.024),
}

_D06_BASKET_X_BOUNDS = (0.04, 0.06)
_D06_BASKET_Y_CENTERS = {
    "d06_basket_box": -0.28,
    "d06_basket_can": 0.0,
    "d06_basket_dish": 0.28,
}
_D06_BASKET_Y_JITTER = 0.030
_D06_BASKET_CENTERLINE_HALF_XY = (0.132, 0.107)
_D06_RELEASE_HOLD_SECONDS = 0.50
_D06_TABLE_LEFT_X = -0.55
_D06_TABLE_HALF_Y = 0.80
# The robot approaches from negative X, so the near edge remains the easiest
# part of the table to reach.  Give every loose body's full footprint a useful
# setback there and an even larger setback at the lateral edges, where the arm
# workspace used to make sampled objects unreachable.
_D06_TABLE_APPROACH_EDGE_CLEARANCE = 0.080
_D06_TABLE_SIDE_EDGE_CLEARANCE = 0.140
_D06_TARGET_Y_BOUNDS = (-0.30, 0.30)
_D06_TABLETOP_PALM_Z = _DLONG_TABLETOP_PALM_Z
_D06_MIN_REACH_MARGIN = _DLONG_MIN_REACH_MARGIN
_D06_HANDOFF_MUG = "d06_mug"
# Negative Y is image-right in the front camera.  Keep the handoff mug in a
# centered-depth lane instead of pushing it to the lateral table edge: both
# arms retain at least 10 cm of spherical reach margin at every lane corner.
_D06_HANDOFF_MUG_X_MAX = -0.34
_D06_HANDOFF_MUG_Y_BOUNDS = (-0.30, -0.24)
_D06_HANDOFF_EDGE_CLEARANCE = 0.010
_D06_HANDOFF_MIN_EACH_ARM_REACH_MARGIN = 0.10
# The most-left possible edge of a yawed basket is about -0.115 m.  Reserve a
# further 7 mm so even the largest randomized target remains separate.
_D06_RESERVED_BASKET_LEFT_X = -0.122
_D06_XY_BOUNDS = {
    name: (
        np.asarray(
            [
                _D06_TABLE_LEFT_X
                + radius
                + _D06_TABLE_APPROACH_EDGE_CLEARANCE,
                -_D06_TABLE_HALF_Y
                + radius
                + _D06_TABLE_SIDE_EDGE_CLEARANCE,
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                _D06_RESERVED_BASKET_LEFT_X - radius,
                _D06_TABLE_HALF_Y
                - radius
                - _D06_TABLE_SIDE_EDGE_CLEARANCE,
            ],
            dtype=np.float64,
        ),
    )
    for name, radius in _D06_RANDOMIZATION_FOOTPRINT_RADII.items()
}
# Distractors retain the wider inner-table lane, while all six objects that
# must be grasped stay in the proven bimanual center band.  Each target still
# varies continuously in X, Y, and yaw; only unreachable edge samples are
# removed.
for _d06_target_name in _D06_TARGETS:
    _d06_lower, _d06_upper = _D06_XY_BOUNDS[_d06_target_name]
    _D06_XY_BOUNDS[_d06_target_name] = (
        np.asarray(
            [
                _d06_lower[0],
                max(float(_d06_lower[1]), _D06_TARGET_Y_BOUNDS[0]),
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                _d06_upper[0],
                min(float(_d06_upper[1]), _D06_TARGET_Y_BOUNDS[1]),
            ],
            dtype=np.float64,
        ),
    )

# The prompt identifies the far-right target, and the physical handoff check
# is intentionally attached to the mug.  Give the mug a dedicated bimanual
# lane, then move every other target's rightmost possible footprint edge to
# its left.  Distractors must retain the full clutter lane: forcing all 18
# loose bodies to one side of the mug leaves no feasible layout around the
# live tabletop hand envelopes.  Comparing footprint edges (not only centers)
# still keeps large target boxes and bowls from visually extending beyond the
# mug after randomization.
_d06_mug_lower = _D06_XY_BOUNDS[_D06_HANDOFF_MUG][0]
_D06_XY_BOUNDS[_D06_HANDOFF_MUG] = (
    np.asarray(
        [_d06_mug_lower[0], _D06_HANDOFF_MUG_Y_BOUNDS[0]],
        dtype=np.float64,
    ),
    np.asarray(
        [_D06_HANDOFF_MUG_X_MAX, _D06_HANDOFF_MUG_Y_BOUNDS[1]],
        dtype=np.float64,
    ),
)
_d06_mug_radius = _D06_RANDOMIZATION_FOOTPRINT_RADII[_D06_HANDOFF_MUG]
for _d06_other_name in _D06_TARGETS:
    if _d06_other_name == _D06_HANDOFF_MUG:
        continue
    _d06_lower, _d06_upper = _D06_XY_BOUNDS[_d06_other_name]
    _d06_other_radius = _D06_RANDOMIZATION_FOOTPRINT_RADII[
        _d06_other_name
    ]
    _d06_min_center_y = (
        _D06_HANDOFF_MUG_Y_BOUNDS[1]
        + max(0.0, _d06_other_radius - _d06_mug_radius)
        + _D06_HANDOFF_EDGE_CLEARANCE
    )
    _D06_XY_BOUNDS[_d06_other_name] = (
        np.asarray(
            [_d06_lower[0], max(float(_d06_lower[1]), _d06_min_center_y)],
            dtype=np.float64,
        ),
        _d06_upper,
    )


def _d06_layout_reach_margins(
    xy_by_name: Mapping[str, np.ndarray],
) -> dict[str, float]:
    return {
        name: _dlong_arm_reach_margin(
            xy_by_name[name], palm_z=_D06_TABLETOP_PALM_Z
        )
        for name in _D06_TARGETS
        if name in xy_by_name
    }

_D07_BASE_XY = {
    # Representative centers of the bounds below.  D07 used to place the
    # keyboard goal at the far negative-Y edge and the drawer root at positive
    # X.  Both looked like invisible collision planes in teleoperation because
    # the 0.78 m arm-target sphere stopped the palm before it reached them.
    "d07_keyboard_frame": (-0.375, -0.200),
    "d07_mouse_pad": (-0.405, 0.100),
    "d07_stand": (-0.260, 0.160),
    "d07_drawer": (-0.068, 0.500),
    "d07_keyboard": (-0.312, -0.470),
    "d07_mouse": (-0.300, -0.090),
    "d07_figurine": (-0.150, 0.015),
    "d07_alarm_clock": (-0.425, 0.325),
    "d07_misc_0": (-0.465, 0.560),
    "d07_misc_1": (-0.160, -0.190),
    "d07_misc_2": (-0.475, -0.065),
}
_D07_XY_BOUNDS = {
    # These compact lanes keep every tabletop target inside the intersection
    # of the table and at least one arm's real spherical workspace.  The
    # drawer remains fixed because moving an imported fixed-root articulation
    # during reset is interpreted as tray travel by PhysX; its authored handle
    # is nevertheless moved into the same reachable band.
    "d07_keyboard_frame": (
        np.asarray([-0.39, -0.22], dtype=np.float64),
        np.asarray([-0.36, -0.18], dtype=np.float64),
    ),
    "d07_mouse_pad": (
        np.asarray([-0.42, 0.08], dtype=np.float64),
        np.asarray([-0.39, 0.12], dtype=np.float64),
    ),
    "d07_stand": (
        np.asarray([-0.28, 0.14], dtype=np.float64),
        np.asarray([-0.24, 0.18], dtype=np.float64),
    ),
    "d07_drawer": (
        np.asarray(_D07_BASE_XY["d07_drawer"], dtype=np.float64),
        np.asarray(_D07_BASE_XY["d07_drawer"], dtype=np.float64),
    ),
    "d07_keyboard": (
        np.asarray([-0.325, -0.49], dtype=np.float64),
        np.asarray([-0.30, -0.45], dtype=np.float64),
    ),
    "d07_mouse": (
        np.asarray([-0.32, -0.11], dtype=np.float64),
        np.asarray([-0.28, -0.07], dtype=np.float64),
    ),
    "d07_figurine": (
        np.asarray([-0.17, -0.01], dtype=np.float64),
        np.asarray([-0.13, 0.04], dtype=np.float64),
    ),
    "d07_alarm_clock": (
        np.asarray([-0.45, 0.30], dtype=np.float64),
        np.asarray([-0.40, 0.35], dtype=np.float64),
    ),
    "d07_misc_0": (
        np.asarray([-0.48, 0.54], dtype=np.float64),
        np.asarray([-0.45, 0.58], dtype=np.float64),
    ),
    "d07_misc_1": (
        np.asarray([-0.18, -0.21], dtype=np.float64),
        np.asarray([-0.14, -0.17], dtype=np.float64),
    ),
    "d07_misc_2": (
        np.asarray([-0.49, -0.09], dtype=np.float64),
        np.asarray([-0.46, -0.04], dtype=np.float64),
    ),
}
# Compatibility aliases keep the D07-specific tests and diagnostics readable.
_D07_ARM_BASE_XY = _DLONG_ARM_BASE_XY
_D07_ARM_BASE_Z = _DLONG_ARM_BASE_Z
_D07_ARM_WORKSPACE_RADIUS = _DLONG_ARM_WORKSPACE_RADIUS
_D07_TABLETOP_PALM_Z = _DLONG_TABLETOP_PALM_Z
_D07_MIN_REACH_MARGIN = _DLONG_MIN_REACH_MARGIN
_D07_DRAWER_HANDLE_LOCAL_XY = np.asarray([-0.183, 0.0], dtype=np.float64)
_D07_DRAWER_HANDLE_Z = 1.035


def _d07_arm_reach_margin(xy: np.ndarray, *, palm_z: float) -> float:
    return _dlong_arm_reach_margin(xy, palm_z=palm_z)


def _d07_layout_reach_margins(
    xy_by_name: Mapping[str, np.ndarray],
) -> dict[str, float]:
    margins = {
        name: _d07_arm_reach_margin(xy, palm_z=_D07_TABLETOP_PALM_Z)
        for name, xy in xy_by_name.items()
        if name != "d07_drawer"
    }
    drawer_xy = xy_by_name.get("d07_drawer")
    if drawer_xy is not None:
        margins["d07_drawer_handle"] = _d07_arm_reach_margin(
            np.asarray(drawer_xy, dtype=np.float64)
            + _D07_DRAWER_HANDLE_LOCAL_XY,
            palm_z=_D07_DRAWER_HANDLE_Z,
        )
    return margins


# The drawer is a fixed-root articulation.  Moving its root while its imported
# world joint remains active is interpreted as 0.18 m of tray travel by PhysX.
_D07_RANDOMIZATION_FOOTPRINT_RADII = {
    # Goal-only keyboard/mouse fixtures have no collision geometry.
    "d07_keyboard_frame": 0.0,
    "d07_mouse_pad": 0.0,
    "d07_stand": 0.060,
    "d07_drawer": float(np.hypot(0.20, 0.17)),
    "d07_keyboard": float(np.hypot(0.205, 0.070)),
    "d07_mouse": float(np.hypot(0.054, 0.028)),
    "d07_figurine": float(np.hypot(0.034, 0.052)),
    "d07_alarm_clock": float(np.hypot(0.055, 0.025)),
    "d07_misc_0": float(np.hypot(0.065, 0.009)),
    "d07_misc_1": float(np.hypot(0.040, 0.024)),
    "d07_misc_2": float(np.hypot(0.052, 0.022)),
}
_D07_RANDOMIZATION_VERTICAL_ENVELOPES = {
    "d07_keyboard_frame": (0.898, 0.0),
    "d07_mouse_pad": (0.896, 0.0),
    "d07_stand": (0.902, 0.010),
    "d07_drawer": (1.010, 0.120),
    "d07_keyboard": (0.904, 0.014),
    "d07_mouse": (0.905, 0.015),
    "d07_figurine": (0.959, 0.069),
    "d07_alarm_clock": (0.932, 0.042),
    "d07_misc_0": (0.899, 0.009),
    "d07_misc_1": (0.908, 0.018),
    "d07_misc_2": (0.911, 0.021),
}
# Avoid starting even the restored compact alarm hull exactly coplanar with
# the table.  A 1 mm gravity-settle gap makes the first contact manifold
# unambiguous and remains well inside the strict recording-origin 5 mm
# reset-drift budget.
_D07_RESET_Z = {
    "d07_alarm_clock": 0.943,
}
# The drawer is fixed at 180 degrees around world Z, so its 0.22 x 0.17 m
# analytic footprint stays axis-aligned.  A circumscribed 0.262 m cylinder is
# useful for conservative object/object separation, but it falsely covers the
# empty corner beside the left reset hand and can make every D07 sample
# impossible.  Use the real rectangular envelope only for hand clearance.
_D07_HAND_FOOTPRINT_HALF_EXTENTS = {
    "d07_drawer": np.asarray([0.220, 0.170], dtype=np.float64),
}

_D08_XY_BOUNDS = {
    # The toaster is a reduced-coordinate articulation with a physical lever;
    # its imported world root is intentionally fixed so reset cannot fight the
    # lever joint.  Randomize the independently resettable basket and plate
    # around it instead of issuing a root teleport that PhysX will snap back.
    "d08_toaster": (np.asarray([-0.38, 0.0]), np.asarray([-0.38, 0.0])),
    "d08_bread_basket": (
        np.asarray([-0.44, -0.52]),
        np.asarray([-0.34, -0.36]),
    ),
    "d08_plate": (np.asarray([-0.40, 0.32]), np.asarray([-0.32, 0.50])),
}
_D08_BREAD_LOCAL_XY = {
    "d08_bread_0": np.asarray([0.0, -0.055], dtype=np.float64),
    "d08_bread_1": np.asarray([0.0, 0.055], dtype=np.float64),
}
# Keep the two slices side-by-side in the basket's local frame.  The local
# jitter is deliberately smaller along Y than the 110 mm slot gap, so a
# randomized reset cannot make the slices overlap or leave the tray.  The
# basket and toaster yaw ranges below make the task vary in both translation
# and orientation while preserving the authored interaction geometry.
_D08_BREAD_LOCAL_X_JITTER = 0.025
_D08_BREAD_LOCAL_Y_JITTER = 0.006
_D08_TOASTER_YAW_BOUNDS_DEG = (0.0, 0.0)
_D08_BASKET_YAW_BOUNDS_DEG = (-12.0, 12.0)
_D08_PLATE_YAW_BOUNDS_DEG = (-25.0, 25.0)
_D08_BREAD_YAW_JITTER_DEG = (-15.0, 15.0)

_D09_CABINET_XY = np.asarray([-0.350, 0.580], dtype=np.float64)
_D09_XY_BOUNDS = {
    # Both roots are imported reduced-coordinate articulations.  Moving them
    # through a reset tensor fights their world anchor and destabilizes the
    # child door/start-button DOFs, so keep them authored and randomize the
    # mug/tray payloads instead.
    "d09_cabinet": (_D09_CABINET_XY.copy(), _D09_CABINET_XY.copy()),
    # The licensed machine is rotated +90 degrees: its actual world footprint
    # is X=+/-140 mm and Y=[-240, 0] mm relative to the root.  Put its complete
    # body in the central lane in front (+X) of both ordinary D08/A01 hands.
    # The former x=-0.135, y=-0.070 root made the body span
    # X=[-0.275, 0.005], Y=[-0.310, -0.070], directly through the open right
    # finger2/finger3 envelope.  At this band the actual asymmetric footprint
    # is about X=[-0.245, 0.045], Y=[-0.125, 0.125], leaving roughly 17 cm of
    # planar clearance from both measured D08/A01 physical palm origins while
    # keeping the outlet and real button inside the shared arm workspace.
    "d09_coffee_machine": (
        np.asarray([-0.100, 0.120]),
        np.asarray([-0.100, 0.120]),
    ),
    # The tray is a resettable kinematic fixture in the near/front lane.  Keep
    # its conservative authored envelope while exposing bounded XY variation.
    "d09_tray": (np.asarray([-0.38, -0.60]), np.asarray([-0.35, -0.56])),
}
_D09_FOOTPRINT_HALF_EXTENTS = {
    # These fixed-yaw fixtures are strongly rectangular, so their disjoint
    # staging bands are checked with AABBs instead of over-conservative circles.
    "d09_cabinet": np.asarray([0.192, 0.184], dtype=np.float64),
    "d09_coffee_machine": np.asarray([0.140, 0.240], dtype=np.float64),
    "d09_tray": np.asarray([0.148, 0.110], dtype=np.float64),
}
_D09_MUG_LOCAL_XY_BOUNDS = (
    # The cabinet is rotated +110 degrees.  These world-axis offsets put the
    # mug around cabinet-local (+.050, +.015), visibly inside the operator's
    # left bay behind the articulated door.  Keep the conservative authored
    # cavity envelope; it still gives independent seeded cup variation while
    # avoiding a reset-time mug/cabinet contact impulse.
    np.asarray([-0.034, 0.039], dtype=np.float64),
    np.asarray([-0.028, 0.045], dtype=np.float64),
)
_D09_MUG_YAW_BOUNDS_DEG = (-4.0, 4.0)
_D09_CABINET_YAW_DEG = 110.0
# Operator-left moving-door hinge (+.222280, .073245) plus the centre of the
# enlarged grip (-.180, .041), expressed in cabinet-local XY.
_D09_CABINET_HANDLE_LOCAL_XY = np.asarray(
    [0.042280, 0.114245], dtype=np.float64
)
_D09_CABINET_HANDLE_Z = 1.0225
# The licensed machine is rotated +90 degrees in the MJCF.  Mesh sections put
# the circular group head at local XY=(-.0533,-.0872), with its lower face at
# Z=.1771.  The matching base top is Z=.055, so an upright 97 mm mug fits below
# it without entering either the right column or the brewer collision.
_D09_COFFEE_OUTLET_LOCAL_XYZ = np.asarray(
    [-0.0533, -0.0872, 0.1771], dtype=np.float64
)
_D09_MUG_BREW_LOCAL_XYZ = np.asarray(
    [-0.0533, -0.0872, 0.0550], dtype=np.float64
)
# Physical button body authored in d_long_horizon_objects.xml.  Reach checks
# must use the real joint origin, not the former visual estimate on the
# opposite X face of the machine.
_D09_START_BUTTON_LOCAL_XY = np.asarray(
    [-0.117488, -0.079022], dtype=np.float64
)
_D09_START_BUTTON_Z = 1.168290
_D09_BREW_SECONDS = 2.0
_D09_BUTTON_TRIGGER_TRAVEL = 0.0015
_D09_MUG_OUTLET_XY_TOLERANCE = np.asarray(
    [0.040, 0.040], dtype=np.float64
)
_D09_MUG_BASE_Z_TOLERANCE = 0.030
_D09_WATER_STREAM_VISUALS = {"d09_water_stream"}
_D09_HIDDEN_COLLISION_VISUALS = {
    "d09_cabinet_body_collision",
    "d09_cabinet_support_floor",
    "d09_cabinet_support_top",
    "d09_cabinet_side_left",
    "d09_cabinet_side_right",
    "d09_cabinet_support_back",
    # Both door panels reuse this source mesh, so Isaac imports both
    # collision prims under the same mesh name.
    "d09_cabinet_moving_door_collision",
    "d09_cabinet_handle_grip_collision",
    "d09_cabinet_handle_upper_mount_collision",
    "d09_cabinet_handle_lower_mount_collision",
    "d09_cabinet_right_handle_grip_collision",
    "d09_cabinet_right_handle_upper_mount_collision",
    "d09_cabinet_right_handle_lower_mount_collision",
    "d09_coffee_machine_collision",
    "d09_coffee_machine_base_collision",
    "d09_coffee_machine_column_collision",
    "d09_coffee_machine_head_collision",
    "d09_coffee_machine_group_collision",
    "d09_coffee_button_collision",
    "d09_mug_body_collision",
    "d09_mug_handle_upper_collision",
    "d09_mug_handle_outer_collision",
    "d09_mug_handle_lower_collision",
    "d09_tray_collision",
}
# Keep a post-construction limit guard in addition to the source-level radian
# compiler contract.  This prevents an old or externally generated USD from
# collapsing the passive hinge back to its historical ~0.025 rad stop.
_D09_LIVE_DOOR_LOWER_LIMIT = -1.40
_D09_LIVE_RIGHT_DOOR_UPPER_LIMIT = 1.40

# Apply the same guard to D10 and retain a small margin at the authored stop.
_D10_LIVE_DOOR_LOWER_LIMIT = -1.40

_D10_DISHWASHER_XY = np.asarray([-0.42, 0.0], dtype=np.float64)
_D10_XY_BOUNDS = {
    # As with the D07 drawer, this articulated fixture must retain its authored
    # root so the door and rack anchors remain coherent.
    "d10_dishwasher": (
        _D10_DISHWASHER_XY.copy(),
        _D10_DISHWASHER_XY.copy(),
    ),
    # Stage dishes beside the fixed dishwasher instead of behind its far
    # (+X) face.  The side lanes clear the real 0.298 m half-width fixture and
    # put every grasp inside at least one arm's workspace.
    "d10_mug": (np.asarray([-0.46, -0.43]), np.asarray([-0.42, -0.39])),
    "d10_bowl": (np.asarray([-0.46, 0.39]), np.asarray([-0.42, 0.41])),
    "d10_plate": (np.asarray([-0.46, 0.57]), np.asarray([-0.42, 0.60])),
}
_D10_RANDOMIZATION_FOOTPRINT_RADII = {
    "d10_dishwasher": float(np.hypot(0.237, 0.298)),
    "d10_mug": float(np.hypot(0.045, 0.075)),
    "d10_bowl": 0.081,
    "d10_plate": 0.078,
}
_D10_MOVABLE_XY_BOUNDS = {
    name: bounds
    for name, bounds in _D10_XY_BOUNDS.items()
    if name != "d10_dishwasher"
}
_D10_DISH_FOOTPRINT_RADII = {
    name: _D10_RANDOMIZATION_FOOTPRINT_RADII[name]
    for name in _D10_DISHES
}

_D11_SINK_XY = np.asarray([-0.26, 0.0], dtype=np.float64)
_D11_XY_BOUNDS = {
    "d11_sink": (_D11_SINK_XY.copy(), _D11_SINK_XY.copy()),
    # The D11 sink occupies the ordinary ABC dual-hand home, so the scene uses
    # the collision-clear lowered bimanual home.  Stage the pan in the clear
    # front-right lane with its handle pointing inward, and keep the sponge in
    # the opposite left-hand lane.  Pulling the pan 70--110 mm inward keeps
    # every sampled handle pose at least 80 mm inside the arm target sphere;
    # the former y=-0.60 edge left only about 24 mm and was fragile under DLS.
    "d11_pan": (np.asarray([-0.19, -0.53]), np.asarray([-0.18, -0.49])),
    "d11_sponge": (np.asarray([-0.30, 0.34]), np.asarray([-0.26, 0.38])),
}
_D11_MOVABLE_XY_BOUNDS = {
    name: bounds for name, bounds in _D11_XY_BOUNDS.items() if name != "d11_sink"
}
_D11_RANDOMIZATION_FOOTPRINT_RADII = {
    # The handle points toward negative X, away from the reset right hand, so
    # the circular bowl is the nearest physical surface for hand-clearance
    # sampling.  Treating the full handle length as a circumscribed circle
    # falsely made every otherwise-clear sample overlap that hand.
    "d11_pan": 0.145,
    "d11_sponge": float(np.hypot(0.040, 0.060)),
}

_D12_DRAWER_XY = np.asarray([-0.24, 0.30], dtype=np.float64)
_D12_XY_BOUNDS = {
    "d12_drawer": (_D12_DRAWER_XY.copy(), _D12_DRAWER_XY.copy()),
    # The safety zone is 145 mm toward +X from the goal root.  Moving the goal
    # inward keeps the final outward-pointing knife placement reachable.
    "d12_board_goal": (np.asarray([-0.38, -0.06]), np.asarray([-0.34, 0.02])),
    # Give the board a visibly different reset pose across episodes instead
    # of the former 20 x 40 mm jitter that looked fixed in the operator view.
    # The complete board and raised handle remain inside the tabletop and its
    # near (-X) handle stays in the shared arm workspace.
    "d12_cutting_board": (
        np.asarray([-0.06, -0.28]),
        np.asarray([0.04, -0.12]),
    ),
    "d12_plate": (np.asarray([-0.46, -0.59]), np.asarray([-0.42, -0.55])),
}
_D12_RANDOMIZATION_FOOTPRINT_RADII = {
    # The drawer uses a rectangular footprint with a 220 mm half-width; the
    # diagonal over-conservative radius made the camera-friendly placement
    # reject the adjacent board even though their X faces remain separated.
    "d12_drawer": 0.22,
    "d12_board_goal": 0.0,
    # Include the complete raised handle in reset clearance sampling.
    "d12_cutting_board": float(np.hypot(0.227, 0.14)),
    "d12_plate": 0.13,
}
_D12_KNIFE_LOCAL_XY_BOUNDS = (
    # Keep the complete knife on the moving tray shelf while retaining
    # visible, seeded variation.
    np.asarray([-0.015, -0.035], dtype=np.float64),
    np.asarray([0.015, 0.035], dtype=np.float64),
)
_D12_BOARD_YAW_BOUNDS_DEG = (-12.0, 12.0)
_D12_BOARD_NEAR_HANDLE_LOCAL_XY = np.asarray([-0.215, 0.0], dtype=np.float64)
_D12_MEAT_PLATE_LOCAL_XY_BOUNDS = (
    np.asarray([-0.035, -0.035], dtype=np.float64),
    np.asarray([0.035, 0.035], dtype=np.float64),
)
_D12_DRAWER_HANDLE_LOCAL_XY = np.asarray([-0.195, 0.0], dtype=np.float64)
_D12_DRAWER_HANDLE_Z = 1.040
_D12_KNIFE_GRASP_Z = 1.020
# Reset the dynamic support bodies to their authored post-settle heights.  The
# generic reset path preserves the pose seen after scene construction; on GPU
# PhysX that pose is already one gravity frame below the MJCF and produces a
# visible first-frame drop.  Keep a 1 mm clearance over the tabletop/plate so
# the first contact is solved once instead of starting coplanar.
_D12_BOARD_RESET_Z = 0.901
_D12_MEAT_RESET_Z = 0.937
_D12_BOARD_NEAR_HANDLE_Z = 0.976
_D12_BOARD_STATIC_FRICTION = 0.32
_D12_BOARD_DYNAMIC_FRICTION = 0.20
_D12_BOARD_GRIP_STATIC_FRICTION = 0.85
_D12_BOARD_GRIP_DYNAMIC_FRICTION = 0.65
_D12_KNIFE_TRAY_STATIC_FRICTION = 0.80
_D12_KNIFE_TRAY_DYNAMIC_FRICTION = 0.60
_D12_BOARD_CONTACT_OFFSET = 0.001
_D12_BOARD_REST_OFFSET = 0.0
_D12_BOARD_MASS = 0.38
_D12_BOARD_DIAGONAL_INERTIA = (0.0025, 0.0064, 0.0085)
_D12_BOARD_FRICTION_COMBINE_MODE = "multiply"
_D12_DRAWER_STIFFNESS = 1.5
_D12_DRAWER_DAMPING = 0.30
_D12_DRAWER_MAX_FORCE = 2.0
_D12_DRAWER_FRICTION = 0.04
_D12_KNIFE_TRANSPORT_RELEASE_DISTANCE = 0.105
_D12_KNIFE_TRANSPORT_RELEASE_COMMAND = 0.72
_D12_KNIFE_TRAY_LOCAL_Z = 0.039


def _dlong_required_reach_margins(
    task_name: str,
    xy_by_name: Mapping[str, np.ndarray],
    *,
    yaw_by_name: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return arm-workspace margins for every mandatory manipulation point.

    A root being visible on the table does not prove that the operator can
    reach its handle, slot, button, or final target.  These points mirror the
    actual workflow affordances and the same spherical limiter used by live
    teleoperation.  Positive values are reachable; callers require the shared
    20 mm safety margin so randomization cannot land exactly on the clamp.
    """

    xy = {
        name: np.asarray(value, dtype=np.float64).reshape(2)
        for name, value in xy_by_name.items()
    }
    yaws = yaw_by_name or {}
    points: dict[str, tuple[np.ndarray, float]] = {}

    if task_name in _d08_variant.TOAST_TASKS:
        toaster = xy["d08_toaster"]
        toaster_yaw = math.radians(float(yaws.get("d08_toaster", 0.0)))
        toaster_rotation = np.asarray(
            [
                [math.cos(toaster_yaw), -math.sin(toaster_yaw)],
                [math.sin(toaster_yaw), math.cos(toaster_yaw)],
            ],
            dtype=np.float64,
        )
        points = {
            "d08_bread_0_grasp": (xy["d08_bread_0"], 0.98),
            "d08_bread_1_grasp": (xy["d08_bread_1"], 0.98),
            "d08_slot_0": (
                toaster + toaster_rotation @ np.asarray([0.0, -0.032]),
                1.045,
            ),
            "d08_slot_1": (
                toaster + toaster_rotation @ np.asarray([0.0, 0.032]),
                1.045,
            ),
            "d08_button": (
                toaster + toaster_rotation @ np.asarray([-0.108, 0.043]),
                0.995,
            ),
            "d08_plate": (xy["d08_plate"], 0.98),
        }
    elif task_name == _D09:
        cabinet = xy["d09_cabinet"]
        machine = xy["d09_coffee_machine"]
        cabinet_yaw = math.radians(_D09_CABINET_YAW_DEG)
        cabinet_rotation = np.asarray(
            [
                [math.cos(cabinet_yaw), -math.sin(cabinet_yaw)],
                [math.sin(cabinet_yaw), math.cos(cabinet_yaw)],
            ],
            dtype=np.float64,
        )
        points = {
            "d09_mug_grasp": (xy["d09_mug"], 1.00),
            "d09_cabinet_handle": (
                cabinet
                + cabinet_rotation @ _D09_CABINET_HANDLE_LOCAL_XY,
                _D09_CABINET_HANDLE_Z,
            ),
            "d09_coffee_outlet": (
                machine + _D09_COFFEE_OUTLET_LOCAL_XYZ[:2],
                0.90 + float(_D09_COFFEE_OUTLET_LOCAL_XYZ[2]),
            ),
            "d09_start_button": (
                machine + _D09_START_BUTTON_LOCAL_XY,
                _D09_START_BUTTON_Z,
            ),
            "d09_tray": (xy["d09_tray"], 0.98),
        }
    elif task_name == _D10:
        dishwasher = xy.get("d10_dishwasher", _D10_DISHWASHER_XY)
        points = {
            "d10_mug_grasp": (xy["d10_mug"], 1.00),
            "d10_bowl_grasp": (xy["d10_bowl"], 0.98),
            "d10_plate_grasp": (xy["d10_plate"], 0.98),
            "d10_door_handle": (
                dishwasher + np.asarray([0.255, 0.0]),
                1.235,
            ),
            "d10_rack_mug_slot": (
                dishwasher + np.asarray([0.220, -0.120]),
                1.05,
            ),
            "d10_rack_bowl_slot": (
                dishwasher + np.asarray([0.220, 0.0]),
                1.05,
            ),
            "d10_rack_plate_slot": (
                dishwasher + np.asarray([0.220, 0.120]),
                1.05,
            ),
        }
    elif task_name == _D11:
        pan_yaw = math.radians(float(yaws.get("d11_pan", 180.0)))
        pan_handle_local = np.asarray([0.260, 0.0], dtype=np.float64)
        rotation = np.asarray(
            [
                [math.cos(pan_yaw), -math.sin(pan_yaw)],
                [math.sin(pan_yaw), math.cos(pan_yaw)],
            ],
            dtype=np.float64,
        )
        sink = xy.get("d11_sink", _D11_SINK_XY)
        points = {
            "d11_pan_handle": (xy["d11_pan"] + rotation @ pan_handle_local, 0.98),
            "d11_sponge_grasp": (xy["d11_sponge"], 0.96),
            "d11_sink_basin": (sink + np.asarray([0.085, 0.0]), 0.98),
            "d11_faucet_valve": (sink + np.asarray([-0.200, -0.095]), 1.079),
            "d11_faucet_spout": (sink + np.asarray([0.120, 0.0]), 1.11),
        }
    elif task_name == _D12:
        drawer = xy["d12_drawer"]
        board = xy["d12_cutting_board"]
        goal = xy["d12_board_goal"]
        board_yaw = math.radians(
            float(yaws.get("d12_cutting_board", 0.0))
        )
        board_rotation = np.asarray(
            [
                [math.cos(board_yaw), -math.sin(board_yaw)],
                [math.sin(board_yaw), math.cos(board_yaw)],
            ],
            dtype=np.float64,
        )
        points = {
            "d12_knife_grasp": (xy["d12_knife"], _D12_KNIFE_GRASP_Z),
            "d12_drawer_handle": (
                drawer + _D12_DRAWER_HANDLE_LOCAL_XY,
                _D12_DRAWER_HANDLE_Z,
            ),
            "d12_board_near_handle": (
                board
                + board_rotation @ _D12_BOARD_NEAR_HANDLE_LOCAL_XY,
                _D12_BOARD_NEAR_HANDLE_Z,
            ),
            "d12_meat_grasp": (xy["d12_meat"], 0.98),
            "d12_board_goal": (goal, 0.98),
            "d12_knife_safety_zone": (goal + np.asarray([0.145, 0.0]), 0.98),
        }
    elif task_name in _d20_stages.TASK_NAMES:
        plate = xy.get(_d20.PLATE, _d20.RESET_XY[_d20.PLATE])
        tongs = xy.get(_d20.TONGS, _d20.RESET_XY[_d20.TONGS])
        microwave = _d20.MICROWAVE_XY
        tongs_yaw = np.deg2rad(_d20.RESET_BASE_YAW_DEG[_d20.TONGS])
        tongs_handle = tongs - 0.10 * np.asarray([np.cos(tongs_yaw), np.sin(tongs_yaw)])
        points = {
            "d20_hot_dog_grasp": (xy.get(_d20.HOT_DOG, _d20.RESET_XY[_d20.HOT_DOG]), 0.97),
            "d20_tongs_handle": (tongs_handle, 0.97),
            # Button, cavity front, and door handle are fixed microwave
            # affordances (body frame rotated -90 deg about Z).
            "d20_power_knob": (microwave + np.asarray([_d20.POWER_LOCAL[1], -_d20.POWER_LOCAL[0]]), 1.021602 + _d20.POWER_LOCAL[2]),
            "d20_timer_knob": (microwave + np.asarray([_d20.TIMER_LOCAL[1], -_d20.TIMER_LOCAL[0]]), 1.021602 + _d20.TIMER_LOCAL[2]),
            # The hot dog is carried at the jaws, 23 cm ahead of the handle
            # grip (-10 cm handle vs +13 cm jaw centre in the tool frame).
            # Load the food crosswise near the front of the cavity: its centre
            # is 6 cm forward of the oven root, with the full collider inside.
            # This checks the actual handle grip rather than the empty palm.
            "d20_tongs_loading_grip": (microwave + np.asarray([-0.29, 0.04]), 0.975),
            "d20_door_handle": (microwave + np.asarray([-0.155, -0.115]), 1.05),
            "d20_plate": (plate, 0.97),
            "d20_fork_goal": (plate + _d20.FORK_OFFSET, 0.97),
            "d20_mustard_goal": (plate + _d20.MUSTARD_OFFSET, 1.00),
            "d20_mug_goal": (plate + _d20.MUG_OFFSET, 1.00),
            "d20_fork_grasp": (xy.get(_d20.FORK, _d20.RESET_XY[_d20.FORK]), 0.97),
            "d20_mustard_grasp": (xy.get(_d20.MUSTARD, _d20.RESET_XY[_d20.MUSTARD]), 1.00),
            "d20_mug_grasp": (xy.get(_d20.MUG, _d20.RESET_XY[_d20.MUG]), 1.00),
        }

    if task_name in _d20_stages.TASK_NAMES:
        # A loaded tool must stay in the right hand; "either arm can reach"
        # previously accepted a layout that forced a fragile mid-task handoff.
        assigned_arms = {
            "d20_hot_dog_grasp": 0, "d20_tongs_handle": 0,
            "d20_tongs_loading_grip": 0, "d20_plate": 0,
            "d20_power_knob": 1, "d20_timer_knob": 1, "d20_door_handle": 1,
        }
        return {
            label: float(_dlong_arm_reach_margins(point_xy, palm_z=palm_z)[assigned_arms[label]])
            if label in assigned_arms else _dlong_arm_reach_margin(point_xy, palm_z=palm_z)
            for label, (point_xy, palm_z) in points.items()
        }

    return {
        label: _dlong_arm_reach_margin(point_xy, palm_z=palm_z)
        for label, (point_xy, palm_z) in points.items()
    }


def _require_dlong_required_reach(
    task_name: str,
    xy_by_name: Mapping[str, np.ndarray],
    *,
    yaw_by_name: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fail closed when a mandatory workflow affordance is unreachable."""

    margins = _dlong_required_reach_margins(
        task_name,
        xy_by_name,
        yaw_by_name=yaw_by_name,
    )
    minimum_margin = (
        _D11_MIN_REACH_MARGIN
        if task_name == _D11
        else _DLONG_MIN_REACH_MARGIN
    )
    unreachable = {
        label: margin
        for label, margin in margins.items()
        if margin < minimum_margin
    }
    if unreachable:
        details = ", ".join(
            f"{label}={margin:.4f}m"
            for label, margin in sorted(unreachable.items())
        )
        raise RuntimeError(
            f"{task_name} randomized layout exceeds the arm workspace: "
            + details
        )
    return margins

_EXTRA_POSES = {
    _D07: {"d07_drawer_tray"},
    _D08: {"d08_toaster_button"},
    _D09: {
        "d09_cabinet_door",
        "d09_cabinet_right_door",
        "d09_coffee_button",
    },
    _D10: {"d10_rack"},
    _D11: {"d11_faucet_arm", "d11_faucet_valve"},
    _D12: {"d12_drawer_tray"},
    _D13: {"d13_cabinet_door"},
    _D14: {"d14_faucet_arm", "d14_faucet_valve"},
    _D17: {"d17_laptop_screen"},
    _D18: {"d18_carton_lid"},
    _D20: set(_d20.EXTRA_POSES),
}

_TABLE_TOP_Z = 0.89
_STABLE_LINEAR_SPEED = 0.045
_STABLE_ANGULAR_SPEED = 0.50
_HAND_NEAR_DISTANCE = 0.095
_D06_HAND_CONTACT_FORCE = 0.15
_D06_HANDOFF_LIFT_HEIGHT = 0.03
_D11_HAND_CONTACT_FORCE = 0.15
_DLONG_RESET_LINEAR_DAMPING = 0.40
_DLONG_RESET_ANGULAR_DAMPING = 2.00
_DLONG_RESET_SLEEP_THRESHOLD = 0.020
_DLONG_RESET_STABILITY_BODIES = {
    _D08: (*_D08_BREAD, "d08_toaster_button"),
    _D09: ("d09_mug",),
    _D10: (*_D10_DISHES, "d10_dishwasher_door", "d10_rack"),
    _D11: ("d11_pan", "d11_sponge"),
    _D12: ("d12_knife", "d12_cutting_board", "d12_meat"),
    _D13: ("d13_tray", *_D13_PAYLOADS),
    _D14: ("d14_colander", *_D14_FRUITS),
    _D15: (*_D15_BOXES, *_D15_LIDS, *_D15_FOODS),
    _D16: (*_D16_TOOLS, *_D16_BLOCKS),
    _D17: ("d17_headphones", "d17_laptop"),
    _D18: _D18_EGGS,
    _D19: (*_D19_BLOCKS, *_D19_COVERS),
    # The tongs are a floating-base articulation (root + hinged arm).  The
    # named stability tuning strips "redundant" articulation roots from free
    # bodies, which would demote the tongs hinge to a maximal-coordinate
    # joint and remove its DOF view, so the tongs stay out of this list.
    _D20: tuple(name for name in _d20.FREE_BODIES if name != _d20.TONGS),
}
_DLONG_RESET_STABILITY_PHYSICS = {
    # Thin dishes retain a small gravity/contact rocking velocity at the
    # generic values.  These values are still fully manipulable and let the
    # bodies sleep once their real tabletop manifolds settle.
    _D10: (0.40, 6.00, 0.050),
    # The thin D20 dishes can retain a small rotational rocking velocity after
    # randomized placement.  Higher reset-only angular damping lets the
    # recording-origin gate settle within its 3 s budget without affecting live
    # manipulation dynamics.
    _D20: (1.20, 12.00, 0.050),
}


class HandsOnDLongHorizonIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Ordered, event-aware collection runtime for D06--D19."""

    def __init__(self, task_name: str, **kwargs: Any):
        if task_name not in D_LONG_TASK_NAMES:
            supported = ", ".join(sorted(D_LONG_TASK_NAMES))
            raise ValueError(
                f"Unsupported long-horizon D task {task_name!r}; expected one of: {supported}"
            )

        # Isaac setup invokes virtual reset and observation hooks from super.
        self._long_metrics: dict[str, Any] = {}
        self._event_flags: dict[str, bool] = {}
        self._timers: dict[str, float] = {}
        self._initial_fixture_poses: dict[str, np.ndarray] = {}
        self._d06_handoff_source: str | None = None
        self._d06_both_confirmed = False
        self._d06_handoff_complete = False
        self._d06_contact_report_interface: Any | None = None
        self._d06_contact_report_subscription: Any | None = None
        self._d06_target_actor_names: dict[int, str] = {}
        self._d06_target_contact_names: dict[int, str] = {}
        self._d06_hand_actor_sides: dict[int, str] = {}
        self._d06_hand_contact_sides: dict[int, str] = {}
        self._d06_active_hand_contacts: dict[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[str, str, float],
        ] = {}
        self._d06_contact_callback_error: Exception | None = None
        self._d06_contact_reporting_available = False
        self._d06_contact_view_errors: set[str] = set()
        self._d11_pan_contact_views: dict[str, Any] = {}
        self._d11_contact_view_errors: set[str] = set()
        self._d08_2_visual_key = None
        self._d08_slots_seen = False
        self._d08_cycle_started = False
        self._d08_release_started = False
        self._d08_cycle_complete = False
        self._d08_lever_target = _D08_BUTTON_UP_TARGET
        self._d08_pop_failure_reason: str | None = None
        self._d08_removed_after_pop: set[str] = set()
        self._d08_latched_bread_poses: dict[str, np.ndarray] = {}
        self._d08_rack_collision_paths: tuple[str, ...] = ()
        self._d09_brew_started = False
        self._d09_brew_complete = False
        self._d09_water_stream_visible: bool | None = None
        self._d09_water_stream_paths: tuple[str, ...] = ()
        self._d11_coverage_seconds = np.zeros(4, dtype=np.float64)
        self._d13_peak_tilt_degrees = 0.0
        self._d14_coverage_seconds = np.zeros(3, dtype=np.float64)
        self._d16_dustpan_seen: set[str] = set()
        self._d18_peak_impact_speed = np.zeros(4, dtype=np.float64)
        self._d19_cover_order: list[str] = []
        self._d19_revealed: list[str] = []
        self._d19_basketed: list[str] = []
        self._d19_previous_covered: dict[str, bool] = {}
        self._d19_spatial_order: tuple[str, ...] = ()
        self._d19_sequence_errors = 0
        self._d20_stage = _d20_stages.StageProgress(_d20_stages.stage_number(task_name))
        self._d20_heating_time_scale = 1.0
        self._d20 = _d20.D20StateMachine(contact_release=self._d20_stage.stage == 4)
        self._d20_release_contact_views: dict[str, Any] = {}
        self._d20_release_frame_contact: dict[str, tuple[float, bool]] = {}
        self._d20_reset_power_angle = 0.0
        self._d20_hot_dog_contact_views: dict[str, Any] = {}
        self._d20_contact_view_errors: set[str] = set()
        self._d20_tongs_holding_side = ""
        self._d20_tongs_jaw_target = _d20.TONGS_OPEN_RAD
        self._d20_tongs_latched = False
        self._d20_tongs_latch_offset: np.ndarray | None = None
        self._d20_tongs_latch_last_target: np.ndarray | None = None
        self._d20_tool_grip = _d20_grip.ToolGrip()
        self._d20_tool_contact_views = {}
        self._d20_payload_contact_view = None
        self._d20_knob_contact_views: dict[str, Any] = {}
        self._d20_knob_contact_view_errors: set[str] = set()
        # Isaac's MJCF importer cannot nest a free joint under the drawer
        # articulation.  Keep the knife as an independent rigid body, but use
        # a short-lived transport latch until a closed hand reaches it; this
        # makes the authored knife genuinely travel with the tray and then
        # leaves it dynamic for the actual pickup.
        self._d12_knife_transport_active = False
        self._d12_knife_transport_offset: np.ndarray | None = None
        self._d12_knife_transport_last_target: np.ndarray | None = None
        self._d12_knife_transport_release_reason = ""
        if task_name in {*_d20_stages.TASK_NAMES, _d08_variant.TASK_NAME}:
            # One 30 Hz control frame is exactly four 120 Hz physics steps.
            # The generic 20 ms default rounds to two steps and makes a
            # minute on the rotary timer disagree with simulation time.
            kwargs.setdefault("control_dt", 1.0 / max(1, int(kwargs.get("hz", 30))))
        super().__init__(task_name=task_name, **kwargs)
        if (
            task_name == _D09
            and self._launch_isaac
            and self._physics_backend == "physx"
        ):
            self._configure_d09_live_door_limit()
        if (
            task_name == _D10
            and self._launch_isaac
            and self._physics_backend == "physx"
        ):
            self._configure_d10_live_door_limit()

    def _configure_task_specific_pre_physx(self) -> bool:
        configured = super()._configure_task_specific_pre_physx()
        if self.task_name == _d08_variant.TASK_NAME and not self._direct_hand_control:
            # The external full-arm USD is spawned after the task's initial
            # joint authoring. Install its normal home before the first
            # PhysX reset, instead of initializing straight, zero-angle arms
            # and correcting them only during the later episode reset.
            from .task_preview import apply_robot_joint_defaults

            apply_robot_joint_defaults("/World/HandsOnRobots", self.task_name)
            configured = True
        return configured

    def _ready_state_settle_physics_steps(self) -> int:
        if self.task_name == _d08_variant.TASK_NAME:
            return 120
        return super()._ready_state_settle_physics_steps()

    def _validate_task_initial_state_before_ready(self):
        report = super()._validate_task_initial_state_before_ready()
        if self.task_name != _d08_variant.TASK_NAME or not self._launch_isaac:
            return report
        passed, robots = _d08_variant.initial_control_state_report(self)
        errors = [] if passed else ["actual robot home failed: " + str({side: [key for key, ok in row.get("checks", {}).items() if not ok] for side, row in robots.items() if not row.get("passed", False)})]
        bodies = {}
        board_z = float(self._pose("d08_bread_basket")[2])
        for name in _D08_BREAD:
            pose = self._physics_body_pose(name)
            velocity = self._physics_body_velocity(name)
            finite = pose is not None and velocity is not None and np.all(np.isfinite(pose)) and np.all(np.isfinite(velocity))
            if not finite:
                errors.append(f"{name}: missing or non-finite physical state")
                continue
            speed = float(np.linalg.norm(velocity[:3]))
            spin = float(np.linalg.norm(velocity[3:]))
            support_error = abs(float(pose[2]) - (board_z + .0262))
            bodies[name] = {"pose": np.asarray(pose).tolist(), "velocity": np.asarray(velocity).tolist(), "support_error_m": support_error}
            if support_error > .004 or speed > .01 or spin > .10:
                errors.append(f"{name}: support error={support_error:.6f}m, speed={speed:.6f}m/s, spin={spin:.6f}rad/s")
        report = {"passed": not errors, "robots": robots, "bodies": bodies, "errors": errors}
        self._d08_2_initial_state_report = report
        if errors:
            raise RuntimeError("D08-2 initial physical state: " + "; ".join(errors))
        return report

    def _configure_d09_live_door_limit(self) -> float:
        """Unlock the D09 cabinet hinge after its safe import reset.

        Current task assets enforce radians before conversion.  Keep this
        live reduced-coordinate guard for externally generated or stale USDs,
        just as D02 does; do not write the door pose or drive target, so the
        operator keeps control of the passive hinge.
        """

        door_specs = (
            (
                "d09_cabinet_door_joint",
                _D09_LIVE_DOOR_LOWER_LIMIT,
                0.0,
            ),
            (
                "d09_cabinet_right_door_joint",
                0.0,
                _D09_LIVE_RIGHT_DOOR_UPPER_LIMIT,
            ),
        )
        if self._torch is None:
            raise RuntimeError(
                "D09 live PhysX scene cannot configure cabinet door limits"
            )
        for joint_name, lower, upper in door_specs:
            dof_entry = self._task_dof_views.get(joint_name)
            if dof_entry is None:
                raise RuntimeError(
                    f"D09 live PhysX scene is missing {joint_name}"
                )
            view, dof_index, _authored_position = dof_entry
            limits = view.get_dof_limits().clone()
            limits[:, int(dof_index), 0] = lower
            limits[:, int(dof_index), 1] = upper
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device="cpu",
            )
            view.set_dof_limits(limits.cpu(), indices)
        print(
            "[HandsOn/IsaacLab] D09 cabinet doors unlocked after safe "
            "construction reset "
            f"(left_lower={_D09_LIVE_DOOR_LOWER_LIMIT:.3f} rad, "
            f"right_upper={_D09_LIVE_RIGHT_DOOR_UPPER_LIMIT:.3f} rad)",
            flush=True,
        )
        return _D09_LIVE_DOOR_LOWER_LIMIT

    def _set_d09_water_stream_visible(self, visible: bool) -> bool:
        """Show D09's visual water only during an uninterrupted brew."""

        visible = bool(visible)
        if (
            self.task_name != _D09
            or getattr(self, "_stage", None) is None
            or self._d09_water_stream_visible is visible
        ):
            return False
        from pxr import UsdGeom

        changed = 0
        for path in self._d09_water_stream_paths:
            prim = self._stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
            changed += 1
        if changed <= 0:
            return False
        self._d09_water_stream_visible = visible
        return True

    def _configure_d10_live_door_limit(self) -> float:
        """Unlock the D10 dishwasher door after its safe import reset."""

        dof_entry = self._task_dof_views.get("d10_door_joint")
        if dof_entry is None or self._torch is None:
            raise RuntimeError(
                "D10 live PhysX scene is missing the dishwasher door DOF"
            )
        view, dof_index, _authored_position = dof_entry
        limits = view.get_dof_limits().clone()
        limits[:, int(dof_index), 0] = _D10_LIVE_DOOR_LOWER_LIMIT
        limits[:, int(dof_index), 1] = 0.0
        indices = self._torch.arange(
            int(view.count),
            dtype=self._torch.long,
            device="cpu",
        )
        view.set_dof_limits(limits.cpu(), indices)
        print(
            "[HandsOn/IsaacLab] D10 dishwasher door unlocked after safe "
            f"construction reset (lower={_D10_LIVE_DOOR_LOWER_LIMIT:.3f} rad)",
            flush=True,
        )
        return _D10_LIVE_DOOR_LOWER_LIMIT

    # ------------------------------------------------------------------
    # Spaces, poses, reset, and replay
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        box = lambda shape: spaces.Box(  # noqa: E731
            -np.inf, np.inf, shape=shape, dtype=np.float64
        )
        common: dict[str, spaces.Space] = {
            "tcp_pose": box((14,)),
            "gripper_pose": box((32,)),
            "table_delta_height": box((1,)),
        }
        if self.task_name == _D06:
            task_spaces = {
                "target_poses": box((6, 7)),
                "basket_poses": box((3, 7)),
                "distractor_poses": box((12, 7)),
                "classification_progress": box((12,)),
            }
        elif self.task_name == _D07:
            task_spaces = {
                "object_poses": box((7, 7)),
                "goal_poses": box((4, 7)),
                "drawer_state": box((4,)),
                "workflow_progress": box((12,)),
            }
        elif self.task_name in _d08_variant.TOAST_TASKS:
            task_spaces = {
                "bread_poses": box((2, 7)),
                "toaster_pose": box((7,)),
                "plate_pose": box((7,)),
                "toaster_state": box((6,)),
                "workflow_progress": box((10,)),
            }
        elif self.task_name == _D09:
            task_spaces = {
                "mug_pose": box((7,)),
                "cabinet_pose": box((7,)),
                "coffee_machine_pose": box((7,)),
                "tray_pose": box((7,)),
                "coffee_state": box((8,)),
                "workflow_progress": box((10,)),
            }
        elif self.task_name == _D10:
            task_spaces = {
                "dish_poses": box((3, 7)),
                "dishwasher_pose": box((7,)),
                "rack_pose": box((7,)),
                "dishwasher_state": box((8,)),
                "workflow_progress": box((10,)),
            }
        elif self.task_name == _D11:
            task_spaces = {
                "pan_pose": box((7,)),
                "sponge_pose": box((7,)),
                "sink_pose": box((7,)),
                "faucet_pose": box((7,)),
                "faucet_state": box((8,)),
                "coverage_progress": box((8,)),
            }
        elif self.task_name == _D12:
            task_spaces = {
                "knife_pose": box((7,)),
                "cutting_board_pose": box((7,)),
                "meat_pose": box((7,)),
                "plate_pose": box((7,)),
                "drawer_pose": box((7,)),
                "board_goal_pose": box((7,)),
                "drawer_state": box((4,)),
                "workflow_progress": box((10,)),
            }
        elif self.task_name == _D13:
            task_spaces = {
                "object_poses": box((4, 7)),
                "fixture_poses": box((2, 7)),
                "cabinet_state": box((4,)),
                "transport_state": box((8,)),
                "workflow_progress": box((12,)),
            }
        elif self.task_name == _D14:
            task_spaces = {
                "fruit_poses": box((3, 7)),
                "colander_pose": box((7,)),
                "fixture_poses": box((3, 7)),
                "faucet_state": box((6,)),
                "wash_progress": box((6,)),
                "workflow_progress": box((12,)),
            }
        elif self.task_name == _D15:
            task_spaces = {
                "food_poses": box((12, 7)),
                "box_poses": box((2, 7)),
                "lid_poses": box((2, 7)),
                "goal_poses": box((2, 7)),
                "packing_state": box((12,)),
                "workflow_progress": box((12,)),
            }
        elif self.task_name == _D16:
            task_spaces = {
                "block_poses": box((5, 7)),
                "tool_poses": box((2, 7)),
                "fixture_poses": box((3, 7)),
                "sweeping_state": box((10,)),
                "workflow_progress": box((12,)),
            }
        elif self.task_name == _D17:
            task_spaces = {
                "headphones_pose": box((7,)),
                "laptop_pose": box((7,)),
                "fixture_poses": box((3, 7)),
                "laptop_state": box((8,)),
                "workflow_progress": box((10,)),
            }
        elif self.task_name == _D18:
            task_spaces = {
                "egg_poses": box((4, 7)),
                "carton_pose": box((7,)),
                "basket_pose": box((7,)),
                "carton_state": box((6,)),
                "impact_state": box((4,)),
                "workflow_progress": box((10,)),
            }
        elif self.task_name == _D19:
            task_spaces = {
                "block_poses": box((3, 7)),
                "cover_poses": box((3, 7)),
                "basket_pose": box((7,)),
                "memory_state": box((6,)),
                "sequence_progress": box((8,)),
            }
        elif self.task_name in _d20_stages.TASK_NAMES:
            task_spaces = {
                "hot_dog_pose": box((7,)),
                "tongs_pose": box((7,)),
                "microwave_pose": box((7,)),
                "tableware_poses": box((4, 7)),
                "tray_pose": box((7,)),
                "microwave_state": box((10,)),
                "microwave_controls": box((4,)),
                "d20_hidden_state": box((_d20.HIDDEN_STATE_SIZE,)),
                "d20_stage_state": box((_d20_stages.StageProgress.SIZE,)),
                "d20_assist_state": box((37,)),
                "d20_tool_grip_state": box((_d20_grip.STATE_SIZE,)),
                "tongs_state": box((6,)),
                "workflow_progress": box((12,)),
            }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict({**common, **task_spaces})
        }
        if self.image_obs:
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": spaces.Box(
                        0, 255, shape=self._image_shape, dtype=np.uint8
                    ),
                    "wrist_left": spaces.Box(
                        0,
                        255,
                        shape=(wrist_height, wrist_width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist_right": spaces.Box(
                        0,
                        255,
                        shape=(wrist_height, wrist_width, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        return spaces.Dict(observation_spaces)

    def _task_extra_pose_names(self) -> set[str]:
        return set(D_LONG_TASK_BODIES[self.task_name]) | set(
            _EXTRA_POSES.get(_d20_stages.family_name(_d08_variant.family_name(self.task_name)), ())
        )

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = super().get_physical_replay_reset_contract()
        if self.task_name in _d20_stages.TASK_NAMES:
            contract['physx']['max_velocity_iteration_count'] = 8
            contract['physx']['d20_mustard_solver_iterations'] = [64, 8]
            contract['physx']['d20_mug_solver_iterations'] = [64, 8]
        return contract

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        if self.task_name in _d20_stages.TASK_NAMES:
            # One seeded discrete detent per episode; zero time is always off.
            power_index = int(self._random.choice(range(len(_d20.POWER_ANGLES)))) if self.randomize else 0
            self._d20_reset_power_angle = float(_d20.POWER_ANGLES[power_index])
            # Preserve the ordinary home orientation with only 8 cm of
            # clearance over the selected tray and its tableware.
            self._right_tcp_pose = _HS005_CLEAR_RIGHT_START_TCP.copy()
            self._left_tcp_pose = _HS005_CLEAR_LEFT_START_TCP.copy()
        elif self.task_name in _D_LONG_LOWERED_START_TASKS:
            self._right_tcp_pose = _D06_LOWERED_RIGHT_START_TCP.copy()
            self._left_tcp_pose = _D06_LOWERED_LEFT_START_TCP.copy()
        elif self.task_name in _D_LONG_DUAL_RAISED_START_TASKS:
            self._right_tcp_pose = _RAISED_RIGHT_START_TCP.copy()
            self._left_tcp_pose = _RAISED_LEFT_START_TCP.copy()
        elif self.task_name in _D_LONG_SINGLE_HAND_START_TASKS:
            self._left_tcp_pose = _RAISED_LEFT_START_TCP.copy()
        # These scenes contain fixed and articulated fixtures that cannot
        # follow the generic movable-table offset as a single rigid group.
        self.delta_h[:] = 0.0
        self._long_metrics = {}
        self._event_flags = {}
        self._timers = {}
        self._initial_fixture_poses = {}
        self._d06_handoff_source = None
        self._d06_both_confirmed = False
        self._d06_handoff_complete = False
        self._d06_active_hand_contacts.clear()
        self._d06_contact_callback_error = None
        self._d08_2_visual_key = None
        self._d08_slots_seen = False
        self._d08_cycle_started = False
        self._d08_release_started = False
        self._d08_cycle_complete = False
        self._d08_lever_target = _D08_BUTTON_UP_TARGET
        self._d08_pop_failure_reason = None
        self._d08_removed_after_pop.clear()
        self._d08_latched_bread_poses.clear()
        self._d09_brew_started = False
        self._d09_brew_complete = False
        self._d09_water_stream_visible = None
        self._set_d09_water_stream_visible(False)
        self._d11_coverage_seconds.fill(0.0)
        self._d13_peak_tilt_degrees = 0.0
        self._d14_coverage_seconds.fill(0.0)
        self._d16_dustpan_seen.clear()
        self._d18_peak_impact_speed.fill(0.0)
        self._d19_cover_order.clear()
        self._d19_revealed.clear()
        self._d19_basketed.clear()
        self._d19_previous_covered.clear()
        self._d19_spatial_order = ()
        self._d19_sequence_errors = 0
        self._d20.reset()
        self._d20_release_frame_contact = {}
        self._d20_stage.reset()
        self._d20_tool_grip.reset()
        self._d20_tongs_holding_side = ""
        self._d20_tongs_jaw_target = _d20.TONGS_OPEN_RAD
        self._d20_tongs_latched = False
        self._d20_tongs_latch_offset = None
        self._d20_tongs_latch_last_target = None
        if self.task_name == _D12:
            self._d12_knife_transport_active = bool(
                self._launch_isaac and self._physics_backend == "physx"
            )
            # Keep the deterministic (``randomize=False``) reset on the same
            # SE(3) path as randomized resets.  A 3-vector here would make
            # the first transport update attempt ``4x4 @ 3`` and silently
            # disable/poison the native preview path.
            self._d12_knife_transport_offset = np.eye(4, dtype=np.float64)
            self._d12_knife_transport_offset[2, 3] = _D12_KNIFE_TRAY_LOCAL_Z
            self._d12_knife_transport_last_target = None
            self._d12_knife_transport_release_reason = "reset"

    def _set_d12_knife_transport_offset(
        self,
        knife_xy: np.ndarray,
        drawer_xy: np.ndarray,
        knife_yaw_degrees: float,
    ) -> None:
        """Cache the sampled knife pose relative to the closed tray body."""

        yaw = math.radians(float(knife_yaw_degrees))
        c, s = math.cos(yaw), math.sin(yaw)
        offset = np.eye(4, dtype=np.float64)
        offset[:3, :3] = np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        offset[:2, 3] = (
            np.asarray(knife_xy, dtype=np.float64).reshape(2)
            - np.asarray(drawer_xy, dtype=np.float64).reshape(2)
        )
        offset[2, 3] = _D12_KNIFE_TRAY_LOCAL_Z
        self._d12_knife_transport_offset = offset
        self._d12_knife_transport_last_target = None

    def _d12_maybe_release_knife_transport(self) -> None:
        """Release the transport latch only when a hand is closing on knife."""

        if (
            self.task_name != _D12
            or not self._d12_knife_transport_active
            or self._sim is None
        ):
            return
        knife_pose = self._physics_body_pose("d12_knife")
        if knife_pose is None or not np.all(np.isfinite(knife_pose[:3])):
            return
        for side in ("right", "left"):
            if self._raw_hand_command_max(side) < _D12_KNIFE_TRANSPORT_RELEASE_COMMAND:
                continue
            distance = self._d_hand_distance_to_point(side, knife_pose[:3])
            if distance <= _D12_KNIFE_TRANSPORT_RELEASE_DISTANCE:
                self._d12_knife_transport_active = False
                self._d12_knife_transport_release_reason = (
                    f"{side}_hand_contact"
                )
                self._d12_knife_transport_last_target = None
                return

    def _update_d12_knife_transport(self) -> None:
        """Carry the free knife with the tray until the operator reaches it."""

        if (
            self.task_name != _D12
            or not self._d12_knife_transport_active
            or self._sim is None
            or self._d12_knife_transport_offset is None
        ):
            return
        tray_pose = self._physics_body_pose("d12_drawer_tray")
        if tray_pose is None or not np.all(np.isfinite(tray_pose)):
            return
        target = (
            _pose_vector_to_matrix(tray_pose)
            @ np.asarray(self._d12_knife_transport_offset, dtype=np.float64)
        )
        applied, _linear_error, _angular_error = self._set_body_rigid_grasp_target(
            "d12_knife",
            target,
            previous_target=self._d12_knife_transport_last_target,
        )
        if applied:
            self._d12_knife_transport_last_target = target.copy()

    def _before_control_step(self) -> None:
        self._d20_release_frame_contact = {}
        super()._before_control_step()
        self._d12_maybe_release_knife_transport()
        self._d20_update_tongs_control()
        self._d20_update_knob_drives()

    def _before_task_physics_substep(self) -> None:
        super()._before_task_physics_substep()
        if self.task_name in _d20_stages.TASK_NAMES:
            self._d20_apply_tool_grip()
            self._d20_update_tongs_latch()

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        self._update_d12_knife_transport()
        if self.task_name in _d20_stages.STAGE_TASKS:
            self._d20_observe_stage_tool_use()
        if self._d20_stage.stage == 4:
            self._d20_sample_release_contact()

    def _apply_raised_hand_home(self, robot_cfg: Any, side: str) -> bool:
        if self.task_name in _d20_stages.TASK_NAMES:
            arm_home = _HS005_CLEAR_ARM_HOME.get(side)
        elif self.task_name in _D_LONG_LOWERED_START_TASKS:
            arm_home = _D06_LOWERED_ARM_HOME.get(side)
        elif self.task_name in _D_LONG_DUAL_RAISED_START_TASKS:
            arm_home = _RAISED_ARM_HOME.get(side)
        elif self.task_name in _D_LONG_SINGLE_HAND_START_TASKS and side == "left":
            arm_home = _RAISED_ARM_HOME.get(side)
        else:
            return super()._apply_raised_hand_home(robot_cfg, side)
        if arm_home is None:
            return False
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None)
        if not isinstance(joint_pos, dict):
            return False
        for index, value in enumerate(arm_home, start=1):
            # Match the base robot configuration's Sim 5/6-compatible regex.
            joint_pos[f"joint{index}_{side}(_1)?"] = float(value)
        return True

    def _configure_task_reset_stability(self, sim_utils: Any) -> int:
        """Tune D-long free bodies before the first PhysX reset.

        The random-layout pass also reapplies these opinions, but the
        recording-origin gate runs immediately after the initial reset.  Put
        the D20 damping in the shared pre-reset hook so stage startup cannot
        exit before the later layout hook is reached.
        """
        tuned = super()._configure_task_reset_stability(sim_utils)
        if self._stage is None or self.task_name not in _d20_stages.TASK_NAMES:
            return tuned
        names = _DLONG_RESET_STABILITY_BODIES.get(
            _d20_stages.family_name(self.task_name), ()
        )
        linear, angular, sleep = _DLONG_RESET_STABILITY_PHYSICS.get(
            _D20,
            (_DLONG_RESET_LINEAR_DAMPING, _DLONG_RESET_ANGULAR_DAMPING,
             _DLONG_RESET_SLEEP_THRESHOLD),
        )
        return tuned + self._configure_named_reset_stability(
            sim_utils, names, linear_damping=linear,
            angular_damping=angular, sleep_threshold=sleep,
            warning_label="D20", strip_redundant_articulation_root=True,
        )

    def _apply_task_pose_randomization(self) -> None:
        if (
            not self.randomize
            or (
                not self._task_reset_body_states
                and not self._task_reset_articulation_states
            )
        ):
            return

        placed: list[str] = []
        if self.task_name == _D06:
            object_xy = self._dlong_sample_separated_layout(
                _D06_XY_BOUNDS,
                pair_distance=0.075,
                hand_clearance=_DLONG_OBJECT_HAND_SURFACE_CLEARANCE,
                footprint_radii=_D06_RANDOMIZATION_FOOTPRINT_RADII,
                vertical_envelopes=_D06_RANDOMIZATION_VERTICAL_ENVELOPES,
                candidates_per_object=_D06_RESET_CANDIDATES_PER_OBJECT,
            )
            reach_margins = _d06_layout_reach_margins(object_xy)
            unreachable = {
                name: margin
                for name, margin in reach_margins.items()
                if margin < _D06_MIN_REACH_MARGIN
            }
            if unreachable:
                details = ", ".join(
                    f"{name}={margin:.4f}m"
                    for name, margin in sorted(unreachable.items())
                )
                raise RuntimeError(
                    "D06 randomized layout exceeds the arm workspace: "
                    + details
                )
            mug_arm_margins = _dlong_arm_reach_margins(
                object_xy[_D06_HANDOFF_MUG],
                palm_z=_D06_TABLETOP_PALM_Z,
            )
            if (
                float(mug_arm_margins.min())
                < _D06_HANDOFF_MIN_EACH_ARM_REACH_MARGIN
            ):
                raise RuntimeError(
                    "D06 handoff mug is outside the shared bimanual "
                    "workspace: "
                    + ", ".join(
                        f"arm_{index}={margin:.4f}m"
                        for index, margin in enumerate(mug_arm_margins)
                    )
                )
            basket_shift = self._random.uniform(
                -_D06_BASKET_Y_JITTER,
                _D06_BASKET_Y_JITTER,
            )
            for name in _D06_BASKETS:
                xy = np.asarray(
                    [
                        self._random.uniform(*_D06_BASKET_X_BOUNDS),
                        _D06_BASKET_Y_CENTERS[name] + basket_shift,
                    ],
                    dtype=np.float64,
                )
                yaw = self._random.uniform(-8.0, 8.0)
                if self._write_reset_pose(name, xy=xy, yaw_degrees=yaw):
                    placed.append(name)
            for name, xy in object_xy.items():
                yaw = self._random.uniform(-60.0, 60.0)
                if self._write_reset_pose(name, xy=xy, yaw_degrees=yaw):
                    placed.append(name)

        elif self.task_name == _D07:
            xy = self._dlong_sample_separated_layout(
                _D07_XY_BOUNDS,
                pair_distance=0.075,
                hand_clearance=_DLONG_OBJECT_HAND_SURFACE_CLEARANCE,
                footprint_radii=_D07_RANDOMIZATION_FOOTPRINT_RADII,
                vertical_envelopes=_D07_RANDOMIZATION_VERTICAL_ENVELOPES,
                hand_footprint_half_extents=(
                    _D07_HAND_FOOTPRINT_HALF_EXTENTS
                ),
            )
            reach_margins = _d07_layout_reach_margins(xy)
            unreachable = {
                name: margin
                for name, margin in reach_margins.items()
                if margin < _D07_MIN_REACH_MARGIN
            }
            if unreachable:
                details = ", ".join(
                    f"{name}={margin:.4f}m"
                    for name, margin in sorted(unreachable.items())
                )
                raise RuntimeError(
                    "D07 randomized layout exceeds the arm workspace: "
                    + details
                )
            for name in (
                "d07_keyboard_frame",
                "d07_mouse_pad",
                "d07_stand",
                "d07_drawer",
            ):
                if self._write_reset_pose(name, xy=xy[name]):
                    placed.append(name)
            yaw_bounds = {
                "d07_keyboard": (-18.0, 18.0),
                "d07_mouse": (-25.0, 25.0),
                "d07_figurine": (-35.0, 35.0),
                "d07_alarm_clock": (-20.0, 20.0),
                "d07_misc_0": (-180.0, 180.0),
                "d07_misc_1": (-180.0, 180.0),
                "d07_misc_2": (-180.0, 180.0),
            }
            for name in _D07_OBJECTS:
                yaw = self._random.uniform(*yaw_bounds.get(name, (-20.0, 20.0)))
                if self._write_reset_pose(
                    name,
                    xy=xy[name],
                    z=_D07_RESET_Z.get(name),
                    yaw_degrees=yaw,
                ):
                    placed.append(name)

        elif self.task_name in _d08_variant.TOAST_TASKS:
            xy = None
            # D08's three stations occupy disjoint lateral bands, so sample
            # each one independently and accept the first fully clear layout.
            # The generic max-clearance sampler biased all three roots toward
            # the same edge of their bands, making otherwise random resets
            # look nearly fixed in the operator view.
            for _ in range(_DLONG_RESET_MAX_ATTEMPTS):
                candidate_xy = {
                    name: self._dlong_sample_xy(bounds)
                    for name, bounds in _D08_XY_BOUNDS.items()
                }
                if (
                    self._dlong_layout_margin(
                        candidate_xy,
                        pair_distance=0.16,
                    )
                    >= _DLONG_RESET_MIN_CLEARANCE_SLACK
                ):
                    xy = candidate_xy
                    break
            if xy is None:
                raise RuntimeError(
                    "D08 randomized toaster/basket/plate layout has no "
                    "collision-clear sample inside the configured bands"
                )
            toaster_yaw = 0.0
            basket_yaw = self._random.uniform(*_D08_BASKET_YAW_BOUNDS_DEG)
            plate_yaw = self._random.uniform(*_D08_PLATE_YAW_BOUNDS_DEG)
            basket_angle = math.radians(float(basket_yaw))
            basket_rotation = np.asarray(
                [
                    [math.cos(basket_angle), -math.sin(basket_angle)],
                    [math.sin(basket_angle), math.cos(basket_angle)],
                ],
                dtype=np.float64,
            )
            reach_xy = dict(xy)
            bread_yaw: dict[str, float] = {}
            for name in _D08_BREAD:
                local_offset = _D08_BREAD_LOCAL_XY[name] + np.asarray(
                    [
                        self._random.uniform(
                            -_D08_BREAD_LOCAL_X_JITTER,
                            _D08_BREAD_LOCAL_X_JITTER,
                        ),
                        self._random.uniform(
                            -_D08_BREAD_LOCAL_Y_JITTER,
                            _D08_BREAD_LOCAL_Y_JITTER,
                        ),
                    ],
                    dtype=np.float64,
                )
                reach_xy[name] = (
                    xy["d08_bread_basket"]
                    + basket_rotation @ local_offset
                )
                bread_yaw[name] = basket_yaw + self._random.uniform(
                    *_D08_BREAD_YAW_JITTER_DEG
                )
            _require_dlong_required_reach(
                self.task_name,
                reach_xy,
                yaw_by_name={"d08_toaster": toaster_yaw},
            )
            station_yaws = {
                "d08_toaster": toaster_yaw,
                "d08_bread_basket": basket_yaw,
                "d08_plate": plate_yaw,
            }
            for name in (
                "d08_toaster",
                "d08_bread_basket",
                "d08_plate",
            ):
                if self._write_reset_pose(
                    name,
                    xy=xy[name],
                    yaw_degrees=station_yaws[name],
                ):
                    placed.append(name)
            for name in _D08_BREAD:
                if self._write_reset_pose(
                    name,
                    xy=reach_xy[name],
                    yaw_degrees=bread_yaw[name],
                ):
                    placed.append(name)
            self._set_task_dof_state("d08_start_button_joint", 0.0, 0.0)
            self._d08_set_lever_target(_D08_BUTTON_UP_TARGET)
            self._d08_set_rack_enabled(True)

        elif self.task_name == _D09:
            # These station bands are intentionally disjoint.  The
            # generic random greedy sampler treated every link-origin in the
            # right arm as a 2-D obstacle and rejected the tray in all 160
            # attempts, even though its 3-D placement was clear.  Sample
            # directly inside the authored safe bands and validate the actual
            # workflow points below instead of failing startup before PhysX
            # can construct a usable episode.
            xy = {
                # Both roots are fixed imported articulations: writing them
                # through a reset tensor is overwritten by the repaired world
                # joint on the first PhysX step and can destabilize the doors
                # or physical start button.  Keep them authored and randomize
                # the resettable tray plus the mug payload below.
                "d09_cabinet": _D09_CABINET_XY.copy(),
                "d09_coffee_machine": np.asarray(
                    [-0.100, 0.120], dtype=np.float64
                ),
                "d09_tray": self._dlong_sample_xy(
                    _D09_XY_BOUNDS["d09_tray"]
                ),
            }
            mug_xy = None
            # The mug must move with the cabinet so it stays in the cavity,
            # but the cabinet envelope makes the far edge of the old mug box
            # marginal for the left reset hand.  Rejection-sample the mug
            # offset using the same required-affordance reach gate used by
            # the live reset.  This preserves visible cup variation without
            # allowing an unreachable seed through.
            for _ in range(_DLONG_RESET_MAX_ATTEMPTS):
                candidate_mug_xy = xy["d09_cabinet"] + self._dlong_sample_xy(
                    _D09_MUG_LOCAL_XY_BOUNDS
                )
                candidate_reach = {
                    **xy,
                    "d09_mug": candidate_mug_xy,
                }
                candidate_margins = _dlong_required_reach_margins(
                    self.task_name,
                    candidate_reach,
                )
                if all(
                    margin >= _DLONG_MIN_REACH_MARGIN
                    for margin in candidate_margins.values()
                ):
                    mug_xy = candidate_mug_xy
                    break
            if mug_xy is None:
                raise RuntimeError(
                    "D09 randomized cabinet/mug layout has no reachable "
                    "mug sample inside the configured cavity"
                )
            mug_yaw = self._random.uniform(*_D09_MUG_YAW_BOUNDS_DEG)
            reach_xy = {**xy, "d09_mug": mug_xy}
            _require_dlong_required_reach(self.task_name, reach_xy)
            if self._write_reset_pose(
                "d09_mug", xy=mug_xy, yaw_degrees=mug_yaw
            ):
                placed.append("d09_mug")
            for name in (
                "d09_cabinet",
                "d09_coffee_machine",
                "d09_tray",
            ):
                if self._write_reset_pose(name, xy=xy[name]):
                    placed.append(name)
            self._set_task_dof_state("d09_cabinet_door_joint", 0.0, 0.0)
            self._set_task_dof_state(
                "d09_cabinet_right_door_joint", 0.0, 0.0
            )
            self._set_task_dof_state("d09_start_button_joint", 0.0, 0.0)

        elif self.task_name == _D10:
            xy = self._dlong_sample_separated_layout(
                _D10_MOVABLE_XY_BOUNDS,
                pair_distance=0.08,
                footprint_radii=_D10_DISH_FOOTPRINT_RADII,
            )
            reach_xy = {**xy, "d10_dishwasher": _D10_DISHWASHER_XY}
            _require_dlong_required_reach(self.task_name, reach_xy)
            if self._write_reset_pose(
                "d10_dishwasher", xy=_D10_DISHWASHER_XY
            ):
                placed.append("d10_dishwasher")
            for name in _D10_DISHES:
                yaw = self._random.uniform(-35.0, 35.0)
                if self._write_reset_pose(name, xy=xy[name], yaw_degrees=yaw):
                    placed.append(name)
            self._set_task_dof_state("d10_door_joint", 0.0, 0.0)
            self._set_task_dof_state("d10_rack_joint", 0.0, 0.0)

        elif self.task_name == _D11:
            xy = self._dlong_sample_separated_layout(
                _D11_MOVABLE_XY_BOUNDS,
                pair_distance=0.08,
                footprint_radii=_D11_RANDOMIZATION_FOOTPRINT_RADII,
            )
            pan_yaw = self._random.uniform(168.0, 192.0)
            sponge_yaw = self._random.uniform(-180.0, 180.0)
            reach_xy = {**xy, "d11_sink": _D11_SINK_XY}
            _require_dlong_required_reach(
                self.task_name,
                reach_xy,
                yaw_by_name={"d11_pan": pan_yaw},
            )
            for name in ("d11_sink", "d11_faucet"):
                if self._write_reset_pose(name, xy=_D11_SINK_XY):
                    placed.append(name)
            # The authored pan already faces 180 degrees so its handle is
            # reachable and clear in the non-randomized scene.  Apply only
            # the sampled delta to preserve that absolute world yaw.
            if self._write_reset_pose(
                "d11_pan",
                xy=xy["d11_pan"],
                yaw_degrees=pan_yaw - 180.0,
            ):
                placed.append("d11_pan")
            if self._write_reset_pose(
                "d11_sponge", xy=xy["d11_sponge"], yaw_degrees=sponge_yaw
            ):
                placed.append("d11_sponge")
            self._set_task_dof_state("d11_faucet_valve_joint", 0.0, 0.0)
            self._set_task_dof_state("d11_faucet_swivel_joint", 0.0, 0.0)

        elif self.task_name == _D12:
            xy = self._dlong_sample_separated_layout(
                _D12_XY_BOUNDS,
                pair_distance=0.10,
                footprint_radii=_D12_RANDOMIZATION_FOOTPRINT_RADII,
            )
            knife_xy = xy["d12_drawer"] + self._dlong_sample_xy(
                _D12_KNIFE_LOCAL_XY_BOUNDS
            )
            knife_yaw = self._random.uniform(-12.0, 12.0)
            board_yaw = self._random.uniform(*_D12_BOARD_YAW_BOUNDS_DEG)
            meat_xy = xy["d12_plate"] + self._dlong_sample_xy(
                _D12_MEAT_PLATE_LOCAL_XY_BOUNDS
            )
            meat_yaw = self._random.uniform(-25.0, 25.0)
            reach_xy = {
                **xy,
                "d12_knife": knife_xy,
                "d12_meat": meat_xy,
            }
            _require_dlong_required_reach(
                self.task_name,
                reach_xy,
                yaw_by_name={"d12_cutting_board": board_yaw},
            )
            for name in (
                "d12_drawer",
                "d12_board_goal",
                "d12_plate",
            ):
                if self._write_reset_pose(name, xy=xy[name]):
                    placed.append(name)
            if self._write_reset_pose(
                "d12_cutting_board",
                xy=xy["d12_cutting_board"],
                z=_D12_BOARD_RESET_Z,
                yaw_degrees=board_yaw,
            ):
                placed.append("d12_cutting_board")
            if self._write_reset_pose(
                "d12_knife", xy=knife_xy, yaw_degrees=knife_yaw
            ):
                placed.append("d12_knife")
            self._set_d12_knife_transport_offset(
                knife_xy,
                xy["d12_drawer"],
                knife_yaw,
            )
            if self._write_reset_pose(
                "d12_meat",
                xy=meat_xy,
                z=_D12_MEAT_RESET_Z,
                yaw_degrees=meat_yaw,
            ):
                placed.append("d12_meat")
            self._set_task_dof_state("d12_drawer_joint", 0.0, 0.0)

        elif self.task_name == _D13:
            base_xy = {
                "d13_tray": (-0.28, -0.08),
                "d13_kettle": (-0.38, 0.20),
                "d13_mug": (-0.40, 0.50),
                "d13_tea_bag": (-0.16, 0.28),
            }
            for name, base in base_xy.items():
                jitter = self._random.uniform(-0.012, 0.012, size=2)
                if self._write_reset_pose(
                    name,
                    xy=np.asarray(base) + jitter,
                    yaw_degrees=self._random.uniform(-8.0, 8.0),
                ):
                    placed.append(name)
            self._set_task_dof_state("d13_cabinet_door_joint", 0.0, 0.0)

        elif self.task_name == _D14:
            base_xy = {
                "d14_colander": (-0.40, -0.58),
                "d14_banana": (-0.45, -0.32),
                # Keep a 170 mm centre spacing between the elongated banana,
                # apple, and orange.  Independent seeded ±12 mm jitter then
                # retains a clearance margin instead of making the banana's
                # capsule tangent to the neighbouring sphere.
                "d14_apple": (-0.28, -0.32),
                "d14_orange": (-0.11, -0.32),
            }
            for name, base in base_xy.items():
                jitter = self._random.uniform(-0.012, 0.012, size=2)
                if self._write_reset_pose(
                    name,
                    xy=np.asarray(base) + jitter,
                    yaw_degrees=self._random.uniform(-10.0, 10.0),
                ):
                    placed.append(name)
            self._set_task_dof_state("d14_faucet_valve_joint", 0.0, 0.0)
            self._set_task_dof_state("d14_faucet_swivel_joint", 0.0, 0.0)

        elif self.task_name == _D15:
            for name in (*_D15_BOXES, *_D15_LIDS, *_D15_FOODS):
                pose = self._pose(name)
                jitter = self._random.uniform(-0.008, 0.008, size=2)
                if self._write_reset_pose(
                    name,
                    xy=pose[:2] + jitter,
                    yaw_degrees=self._random.uniform(-7.0, 7.0),
                ):
                    placed.append(name)

        elif self.task_name == _D16:
            for name in (*_D16_TOOLS, *_D16_BLOCKS):
                pose = self._pose(name)
                jitter = self._random.uniform(-0.012, 0.012, size=2)
                if self._write_reset_pose(
                    name,
                    xy=pose[:2] + jitter,
                    yaw_degrees=self._random.uniform(-8.0, 8.0),
                ):
                    placed.append(name)

        elif self.task_name == _D17:
            for name in ("d17_headphones", "d17_laptop"):
                pose = self._pose(name)
                jitter = self._random.uniform(-0.010, 0.010, size=2)
                if self._write_reset_pose(
                    name,
                    xy=pose[:2] + jitter,
                    yaw_degrees=self._random.uniform(-5.0, 5.0),
                ):
                    placed.append(name)
            self._set_task_dof_state("d17_laptop_hinge_joint", 0.0, 0.0)

        elif self.task_name == _D18:
            for name in _D18_EGGS:
                pose = self._pose(name)
                jitter = self._random.uniform(-0.008, 0.008, size=2)
                if self._write_reset_pose(name, xy=pose[:2] + jitter):
                    placed.append(name)
            self._set_task_dof_state("d18_carton_lid_joint", 0.0, 0.0)

        elif self.task_name in _d20_stages.TASK_NAMES:
            placed.extend(self._d20_randomize_layout())

        else:
            lanes = np.asarray([-0.32, 0.0, 0.32], dtype=np.float64)
            permutation = self._random.permutation(len(lanes))
            for block, lane_index in zip(_D19_BLOCKS, permutation):
                xy = np.asarray(
                    [-0.42 + self._random.uniform(-0.01, 0.01), lanes[lane_index]],
                    dtype=np.float64,
                )
                if self._write_reset_pose(
                    block,
                    xy=xy,
                    yaw_degrees=self._random.uniform(-15.0, 15.0),
                ):
                    placed.append(block)
            for cover, lane in zip(_D19_COVERS, lanes):
                if self._write_reset_pose(
                    cover,
                    xy=np.asarray([-0.16, lane], dtype=np.float64),
                ):
                    placed.append(cover)

        if placed:
            print(
                "[HandsOn/IsaacLab] Randomized D-long task poses: "
                f"task={self.task_name}, objects={','.join(placed)}",
                flush=True,
            )

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        tuned_bodies, tuned_colliders = super()._configure_task_collision_safety(
            sim_utils
        )
        if self.task_name in _d20_stages.TASK_NAMES and self._stage is not None:
            self._d20_configure_stage(sim_utils)
        if self.task_name == _d08_variant.TASK_NAME and self._stage is not None:
            self._d08_2_visual_paths = _d08_variant.compose_visuals(self._stage, _TASK_ROOT)
        if self.task_name in _d08_variant.TOAST_TASKS and self._stage is not None:
            # Isaac's MJCF importer can render rgba=0 analytic geoms as opaque
            # boxes.  Make the task subtree editable and hide only the named
            # collision visuals; their CollisionAPI remains active.
            sim_utils.make_uninstanceable(_TASK_ROOT, stage=self._stage)
            sim_utils.update_stage()
            for name in (
                "d08_toaster",
                "d08_toaster_button",
                "d08_bread_basket",
                *_D08_BREAD,
            ):
                self._prim_cache.pop(name, None)
            hidden = self._set_task_visual_names_visible(
                _D08_HIDDEN_COLLISION_VISUALS,
                False,
            )
            if hidden < len(_D08_HIDDEN_COLLISION_VISUALS):
                self._add_task_state_warning(
                    "D08 could not hide every analytic collision visual "
                    f"({hidden}/{len(_D08_HIDDEN_COLLISION_VISUALS)})"
                )

            from pxr import UsdPhysics

            lever_joint = self._stage.GetPrimAtPath(
                f"{_TASK_ROOT}/joints/d08_start_button_joint"
            )
            if lever_joint is None or not lever_joint.IsValid():
                self._add_task_state_warning("Could not configure D08 toaster lever")
            else:
                drive = UsdPhysics.DriveAPI.Get(lever_joint, "linear")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(lever_joint, "linear")
                drive.GetTargetPositionAttr().Set(_D08_BUTTON_UP_TARGET)
                drive.GetTargetVelocityAttr().Set(0.0)
                drive.GetStiffnessAttr().Set(_D08_BUTTON_STIFFNESS)
                drive.GetDampingAttr().Set(_D08_BUTTON_DAMPING)
                drive.GetMaxForceAttr().Set(_D08_BUTTON_MAX_FORCE)
            if not self._d08_set_rack_enabled(True):
                self._add_task_state_warning("Could not configure D08 toast rack")
        if self.task_name == _D09 and self._stage is not None:
            # Isaac can render rgba=0 collision geometry as opaque.  Because
            # those meshes coincide with the licensed visuals, the overlap
            # flickers in RTX.  Make only this task subtree editable, hide all
            # collision render prims (CollisionAPI stays active), and cache
            # the water paths for constant-time brew feedback.
            sim_utils.make_uninstanceable(_TASK_ROOT, stage=self._stage)
            sim_utils.update_stage()
            for name in (
                "d09_cabinet",
                "d09_cabinet_door",
                "d09_cabinet_right_door",
                "d09_coffee_machine",
                "d09_coffee_button",
                "d09_mug",
                "d09_tray",
            ):
                self._prim_cache.pop(name, None)
            hidden = self._set_task_visual_names_visible(
                _D09_HIDDEN_COLLISION_VISUALS,
                False,
            )
            if hidden < len(_D09_HIDDEN_COLLISION_VISUALS):
                self._add_task_state_warning(
                    "D09 could not hide every collision render prim "
                    f"({hidden}/{len(_D09_HIDDEN_COLLISION_VISUALS)})"
                )
            machine_prim = self._find_rigid_body_prim("d09_coffee_machine")
            if machine_prim is None:
                self._add_task_state_warning(
                    "Could not prepare the D09 coffee water visual"
                )
            else:
                machine_root = machine_prim
                machine_parent = machine_prim.GetParent()
                if (
                    machine_parent is not None
                    and machine_parent.IsValid()
                    and machine_parent.GetName() == machine_prim.GetName()
                ):
                    machine_root = machine_parent
                machine_root_path = str(machine_root.GetPath())
                from pxr import Usd

                machine_root = self._stage.GetPrimAtPath(machine_root_path)
                self._d09_water_stream_paths = tuple(
                    str(prim.GetPath())
                    for prim in Usd.PrimRange(machine_root)
                    if prim.GetName() in _D09_WATER_STREAM_VISUALS
                )
                self._d09_water_stream_visible = None
                if not self._set_d09_water_stream_visible(False):
                    self._add_task_state_warning(
                        "Could not hide the D09 coffee water visual at reset"
                    )
        stability_names = _DLONG_RESET_STABILITY_BODIES.get(_d20_stages.family_name(_d08_variant.family_name(self.task_name)), ())
        if self._stage is not None and stability_names:
            linear_damping, angular_damping, sleep_threshold = (
                _DLONG_RESET_STABILITY_PHYSICS.get(
                    _D20 if self.task_name in _d20_stages.TASK_NAMES else self.task_name,
                    (
                        _DLONG_RESET_LINEAR_DAMPING,
                        _DLONG_RESET_ANGULAR_DAMPING,
                        _DLONG_RESET_SLEEP_THRESHOLD,
                    ),
                )
            )
            tuned_bodies += self._configure_named_reset_stability(
                sim_utils,
                stability_names,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
                sleep_threshold=sleep_threshold,
                warning_label=self.task_name.split("_", 1)[0].upper(),
                strip_redundant_articulation_root=True,
            )
            # ``make_uninstanceable`` can recompose an imported MJCF subtree.
            # Resolve every D12 body again before authoring contact/material
            # opinions; cached Prim handles from before the update may point
            # at the discarded instance proxy.
            if self.task_name == _D12:
                for name in stability_names:
                    self._prim_cache.pop(name, None)
        if self.task_name == _D11 and self._stage is not None:
            # The faucet is mounted 3 mm above the sink bottom after import.
            # PhysX's 20 mm default shells made those two fixed fixtures look
            # interpenetrating and injected a large, non-integrated velocity
            # into the faucet articulation.  Tight shells preserve the real
            # collision surfaces and their visible mounting clearance.
            from pxr import PhysxSchema, Usd, UsdPhysics

            for fixture_name in ("d11_sink", "d11_faucet"):
                fixture_prim = self._find_rigid_body_prim(fixture_name)
                if fixture_prim is None:
                    self._add_task_state_warning(
                        f"Could not tune D11 fixture contacts for {fixture_name}"
                    )
                    continue
                for prim in Usd.PrimRange(fixture_prim):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                    if not collision_api:
                        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision_api.CreateContactOffsetAttr().Set(0.001)
                    collision_api.CreateRestOffsetAttr().Set(0.0)
                    tuned_colliders += 1
        if self.task_name == _D12 and self._stage is not None:
            # Thin tabletop/drawer parts otherwise inherit a contact shell
            # almost as large as the part itself.  Use the same narrow shell
            # as the audited D07 drawer so a board or flat-bottomed food item
            # cannot be kept in a solver-induced rocking state at reset.
            from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade

            contact_paths: dict[str, str] = {}
            for object_name in (
                "d12_knife",
                "d12_cutting_board",
                "d12_plate",
                "d12_meat",
            ):
                object_prim = self._find_rigid_body_prim(object_name)
                if object_prim is None:
                    self._add_task_state_warning(
                        f"Could not tune D12 contact shells for {object_name}"
                    )
                    continue
                object_path = str(object_prim.GetPath())
                contact_paths[object_name] = object_path
                # The plate and some imported collision descendants can remain
                # instance proxies even after the reset-stability pass.  Make
                # every D12 contact root editable before traversing it.
                sim_utils.make_uninstanceable(
                    object_path, stage=self._stage
                )
            if contact_paths:
                sim_utils.update_stage()
            for object_name, object_path in contact_paths.items():
                self._prim_cache.pop(object_name, None)
                object_prim = self._stage.GetPrimAtPath(object_path)
                if object_prim is None or not object_prim.IsValid():
                    self._add_task_state_warning(
                        f"D12 contact root disappeared for {object_name}"
                    )
                    continue
                for prim in Usd.PrimRange(object_prim):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                    if not collision_api:
                        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision_api.CreateContactOffsetAttr().Set(
                        _D12_BOARD_CONTACT_OFFSET
                    )
                    collision_api.CreateRestOffsetAttr().Set(
                        _D12_BOARD_REST_OFFSET
                    )
                    tuned_colliders += 1

            # The knife shelf and the articulated tray are fixed/child
            # geometry of the drawer rather than independent free bodies, so
            # they are not covered by the rigid-body list above.  Tune their
            # shells as well; otherwise the imported default shell can reach
            # through the 1 mm knife support gap and kick the knife at reset.
            knife_shelf_paths: list[str] = []
            drawer_prim = self._find_prim("d12_drawer")
            if drawer_prim is not None and drawer_prim.IsValid():
                # Isaac's MJCF importer can wrap the authored body in an
                # outer Xform and put the slide body beside the fixed-body
                # child.  Break the outer subtree first so both the shelf and
                # tray collision proxies become editable.
                drawer_root = drawer_prim
                parent_prim = drawer_prim.GetParent()
                if (
                    parent_prim is not None
                    and parent_prim.IsValid()
                    and parent_prim.GetName() == drawer_prim.GetName()
                ):
                    drawer_root = parent_prim
                drawer_path = str(drawer_root.GetPath())
                sim_utils.make_uninstanceable(
                    drawer_path, stage=self._stage
                )
                sim_utils.update_stage()
                self._prim_cache.pop("d12_drawer", None)
                drawer_root = self._stage.GetPrimAtPath(drawer_path)
                if drawer_root is None or not drawer_root.IsValid():
                    self._add_task_state_warning(
                        "D12 drawer root disappeared after uninstancing"
                    )
                else:
                    for prim in Usd.PrimRange(drawer_root):
                        if not prim.HasAPI(UsdPhysics.CollisionAPI):
                            continue
                        prim_path = str(prim.GetPath())
                        if "d12_knife_shelf" in prim_path.lower():
                            knife_shelf_paths.append(prim_path)
                        collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                        if not collision_api:
                            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                        contact_attr = collision_api.GetContactOffsetAttr()
                        if not contact_attr or not contact_attr.IsValid():
                            contact_attr = collision_api.CreateContactOffsetAttr()
                        contact_attr.Set(_D12_BOARD_CONTACT_OFFSET)
                        rest_attr = collision_api.GetRestOffsetAttr()
                        if not rest_attr or not rest_attr.IsValid():
                            rest_attr = collision_api.CreateRestOffsetAttr()
                        rest_attr.Set(_D12_BOARD_REST_OFFSET)
                        tuned_colliders += 1

            # Give the free knife and its moving shelf the same explicit
            # high-friction PhysX material.  The pair then rides with a
            # hand-pulled drawer through static contact, but the knife remains
            # a dynamic rigid body that can be lifted away normally.
            knife_material_path = (
                "/World/PhysicsMaterials/HandsOnD12KnifeTrayGrip"
            )
            knife_material_cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=_D12_KNIFE_TRAY_STATIC_FRICTION,
                dynamic_friction=_D12_KNIFE_TRAY_DYNAMIC_FRICTION,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="min",
            )
            knife_material_cfg.func(
                knife_material_path, knife_material_cfg
            )
            knife_path = contact_paths.get("d12_knife")
            knife_material_targets = [
                path
                for path in (knife_path, *knife_shelf_paths)
                if path
            ]
            for target_path in knife_material_targets:
                sim_utils.bind_physics_material(
                    target_path,
                    knife_material_path,
                    stage=self._stage,
                    stronger_than_descendants=True,
                )
            knife_material_bindings = 0
            verification_roots = []
            if knife_path:
                knife_root = self._stage.GetPrimAtPath(knife_path)
                if knife_root is not None and knife_root.IsValid():
                    verification_roots.append(knife_root)
            verification_roots.extend(
                prim
                for path in knife_shelf_paths
                for prim in (self._stage.GetPrimAtPath(path),)
                if prim is not None and prim.IsValid()
            )
            seen_verification_paths: set[str] = set()
            for root in verification_roots:
                for prim in (root, *Usd.PrimRange(root)):
                    prim_path = str(prim.GetPath())
                    if prim_path in seen_verification_paths:
                        continue
                    seen_verification_paths.add(prim_path)
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    targets = (
                        UsdShade.MaterialBindingAPI(prim)
                        .GetDirectBindingRel("physics")
                        .GetTargets()
                    )
                    if any(
                        str(target) == knife_material_path
                        for target in targets
                    ):
                        knife_material_bindings += 1
            if knife_material_bindings < 2:
                self._add_task_state_warning(
                    "D12 knife/tray grip material did not bind to both "
                    f"contact partners (bindings={knife_material_bindings})"
                )

            # MJCF frictionloss/stiffness are not consistently carried over
            # by the Isaac importer.  Re-author the D12 slide's closed reset
            # drive and disable self-collision between the connected drawer
            # links, while keeping the modest max force low enough for a hand
            # to pull the real handle open.
            joint = self._stage.GetPrimAtPath(
                f"{_TASK_ROOT}/joints/d12_drawer_joint"
            )
            if joint is None or not joint.IsValid():
                self._add_task_state_warning(
                    "Could not tune D12 drawer joint drive"
                )
            else:
                joint_schema = UsdPhysics.Joint(joint)
                collision_enabled = joint_schema.GetCollisionEnabledAttr()
                if not collision_enabled or not collision_enabled.IsValid():
                    collision_enabled = joint_schema.CreateCollisionEnabledAttr()
                collision_enabled.Set(False)
                drive = UsdPhysics.DriveAPI.Get(joint, "linear")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(joint, "linear")
                drive.GetTargetPositionAttr().Set(0.0)
                drive.GetTargetVelocityAttr().Set(0.0)
                drive.GetStiffnessAttr().Set(_D12_DRAWER_STIFFNESS)
                drive.GetDampingAttr().Set(_D12_DRAWER_DAMPING)
                drive.GetMaxForceAttr().Set(_D12_DRAWER_MAX_FORCE)
                joint.GetAttribute("physxJoint:jointFriction").Set(
                    _D12_DRAWER_FRICTION
                )
                joint.GetAttribute("physxLimit:X:stiffness").Set(
                    _D12_DRAWER_STIFFNESS
                )
                joint.GetAttribute("physxLimit:X:damping").Set(
                    _D12_DRAWER_DAMPING
                )
                if joint.HasAPI(PhysxSchema.JointStateAPI, "linear"):
                    state = PhysxSchema.JointStateAPI.Get(joint, "linear")
                else:
                    state = PhysxSchema.JointStateAPI.Apply(joint, "linear")
                state.GetPositionAttr().Set(0.0)
                state.GetVelocityAttr().Set(0.0)

            # The imported MJCF material is commonly replaced by the global
            # IsaacLab tabletop material (static=1.0, dynamic=.8).  Bind the
            # low-friction slide material only to the board slab, and a
            # separate high-grip material to the raised handles.  Applying
            # one low-friction material to the whole body made the new handle
            # just as slippery as the tabletop contact.
            # The preceding stability pass may have recomposed an MJCF
            # instance.  Drop the stale lookup before resolving the live root.
            self._prim_cache.pop("d12_cutting_board", None)
            board_prim = self._find_rigid_body_prim("d12_cutting_board")
            if board_prim is None or not board_prim.IsValid():
                self._add_task_state_warning(
                    "Could not tune D12 cutting-board friction: rigid body is unavailable"
                )
            else:
                board_path = str(board_prim.GetPath())
                try:
                    # The stability pass normally already breaks the
                    # imported instance.  Repeat the canonical operation here
                    # because an instance child can remain even when the root
                    # Prim itself is not marked as an instance; the helper is
                    # idempotent for an already-editable subtree.
                    sim_utils.make_uninstanceable(
                        board_path, stage=self._stage
                    )
                    sim_utils.update_stage()
                    self._prim_cache.pop("d12_cutting_board", None)
                    board_prim = self._stage.GetPrimAtPath(board_path)
                    if board_prim is None or not board_prim.IsValid():
                        raise RuntimeError(
                            "cutting-board prim disappeared after uninstancing"
                        )
                    slide_material_path = (
                        "/World/PhysicsMaterials/HandsOnD12BoardSlide"
                    )
                    grip_material_path = (
                        "/World/PhysicsMaterials/HandsOnD12BoardGrip"
                    )
                    slide_material_cfg = sim_utils.RigidBodyMaterialCfg(
                        static_friction=_D12_BOARD_STATIC_FRICTION,
                        dynamic_friction=_D12_BOARD_DYNAMIC_FRICTION,
                        restitution=0.0,
                        friction_combine_mode=_D12_BOARD_FRICTION_COMBINE_MODE,
                        restitution_combine_mode="min",
                    )
                    grip_material_cfg = sim_utils.RigidBodyMaterialCfg(
                        static_friction=_D12_BOARD_GRIP_STATIC_FRICTION,
                        dynamic_friction=_D12_BOARD_GRIP_DYNAMIC_FRICTION,
                        restitution=0.0,
                        friction_combine_mode="multiply",
                        restitution_combine_mode="min",
                    )
                    slide_material_cfg.func(
                        slide_material_path, slide_material_cfg
                    )
                    grip_material_cfg.func(
                        grip_material_path, grip_material_cfg
                    )

                    bound_slide_colliders = 0
                    bound_grip_colliders = 0
                    # ``bind_physics_material`` may succeed on the root
                    # collider and stop descending.  Include the root
                    # explicitly and de-duplicate it from PrimRange so the
                    # verification cannot report a false zero on importer
                    # variants that put CollisionAPI on the rigid root.
                    board_prims = (board_prim, *Usd.PrimRange(board_prim))
                    seen_board_paths: set[str] = set()
                    for prim in board_prims:
                        prim_path = str(prim.GetPath())
                        if prim_path in seen_board_paths:
                            continue
                        seen_board_paths.add(prim_path)
                        if not prim.HasAPI(UsdPhysics.CollisionAPI):
                            continue
                        is_handle = "d12_cutting_board_handle_" in (
                            prim_path.lower()
                        )
                        expected_material_path = (
                            grip_material_path
                            if is_handle
                            else slide_material_path
                        )
                        sim_utils.bind_physics_material(
                            prim_path,
                            expected_material_path,
                            stage=self._stage,
                            stronger_than_descendants=True,
                        )
                        collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                        if not collision_api:
                            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(
                                prim
                            )
                        contact_attr = collision_api.GetContactOffsetAttr()
                        if not contact_attr or not contact_attr.IsValid():
                            contact_attr = collision_api.CreateContactOffsetAttr()
                        contact_attr.Set(_D12_BOARD_CONTACT_OFFSET)
                        rest_attr = collision_api.GetRestOffsetAttr()
                        if not rest_attr or not rest_attr.IsValid():
                            rest_attr = collision_api.CreateRestOffsetAttr()
                        rest_attr.Set(_D12_BOARD_REST_OFFSET)
                        targets = (
                            UsdShade.MaterialBindingAPI(prim)
                            .GetDirectBindingRel("physics")
                            .GetTargets()
                        )
                        if any(
                            str(target) == expected_material_path
                            for target in targets
                        ):
                            if is_handle:
                                bound_grip_colliders += 1
                            else:
                                bound_slide_colliders += 1

                    mass_api = UsdPhysics.MassAPI(board_prim)
                    if not mass_api:
                        mass_api = UsdPhysics.MassAPI.Apply(board_prim)
                    mass_attr = mass_api.GetMassAttr()
                    if not mass_attr or not mass_attr.IsValid():
                        mass_attr = mass_api.CreateMassAttr()
                    mass_attr.Set(_D12_BOARD_MASS)
                    center_attr = mass_api.GetCenterOfMassAttr()
                    if not center_attr or not center_attr.IsValid():
                        center_attr = mass_api.CreateCenterOfMassAttr()
                    center_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    inertia_attr = mass_api.GetDiagonalInertiaAttr()
                    if not inertia_attr or not inertia_attr.IsValid():
                        inertia_attr = mass_api.CreateDiagonalInertiaAttr()
                    inertia_attr.Set(Gf.Vec3f(*_D12_BOARD_DIAGONAL_INERTIA))
                    if (
                        bound_slide_colliders == 0
                        or bound_grip_colliders == 0
                    ):
                        self._add_task_state_warning(
                            "D12 cutting-board slide/grip materials did not "
                            "bind to every required collider class "
                            f"(slide={bound_slide_colliders}, "
                            f"grip={bound_grip_colliders})"
                        )
                except Exception as exc:
                    self._add_task_state_warning(
                        f"Could not author D12 cutting-board friction material: {exc}"
                    )
        if self._stage is None:
            return tuned_bodies, tuned_colliders
        if self.task_name == _D06:
            contact_targets = tuple(
                (target_name, f"D06 {target_name}")
                for target_name in _D06_TARGETS
            )
        elif self.task_name == _D11:
            contact_targets = (("d11_pan", "D11 pan"),)
        elif self.task_name in _d20_stages.TASK_NAMES:
            # These actors back the hand/tool and two-jaw payload contact views.
            # Enable reporting before PhysX creates its contact tensor entries.
            contact_targets = tuple(
                (name, f"D20 {name}")
                for name in (_d20.RELEASE_CONTACT_BODIES if self._d20_stage.stage == 4
                             else (_d20.HOT_DOG, _d20.TONGS, _d20.TONGS_MOVING_ARM))
            )
        else:
            return tuned_bodies, tuned_colliders

        from pxr import PhysxSchema

        configured_targets = 0
        for target_name, target_label in contact_targets:
            target_prim = self._find_rigid_body_prim(target_name)
            if target_prim is None:
                self._add_task_state_warning(
                    f"{target_label} contact-report rigid body is unavailable"
                )
                continue
            try:
                if target_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    report_api = PhysxSchema.PhysxContactReportAPI.Get(
                        self._stage, target_prim.GetPath()
                    )
                else:
                    report_api = PhysxSchema.PhysxContactReportAPI.Apply(
                        target_prim
                    )
                report_api.CreateThresholdAttr().Set(0.0)
                configured_targets += 1
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not enable {target_label} contact reporting: {exc}"
                )
        return tuned_bodies + configured_targets, tuned_colliders

    def _init_task_specific_runtime(self) -> None:
        """Create fail-closed hand/contact readers for physical gates."""
        self._d06_contact_report_subscription = None
        self._d06_contact_report_interface = None
        self._d06_target_actor_names = {}
        self._d06_target_contact_names = {}
        self._d06_hand_actor_sides = {}
        self._d06_hand_contact_sides = {}
        self._d06_active_hand_contacts = {}
        self._d06_contact_callback_error = None
        self._d06_contact_reporting_available = False
        self._d06_contact_view_errors = set()
        self._d11_pan_contact_views = {}
        self._d11_contact_view_errors = set()
        if (
            self.task_name not in {_D06, _D11, *_d20_stages.TASK_NAMES}
            or not self._robots
            or self._stage is None
        ):
            return

        hand_paths = {
            side: self._d06_hand_body_paths(side)
            for side in ("left", "right")
        }

        if self.task_name in _d20_stages.TASK_NAMES:
            self._d20_init_runtime(hand_paths)
            return

        if self.task_name == _D06:
            target_prims = {
                target_name: self._find_rigid_body_prim(target_name)
                for target_name in _D06_TARGETS
            }
            missing_targets = [
                name for name, prim in target_prims.items() if prim is None
            ]
            if missing_targets:
                self._d06_contact_view_errors.update(("left", "right"))
                self._add_task_state_warning(
                    "D06 contact target rigid bodies are unavailable: "
                    + ",".join(missing_targets)
                )
                return

            missing_sides = sorted(
                side for side, paths in hand_paths.items() if not paths
            )
            if missing_sides:
                self._d06_contact_view_errors.update(missing_sides)
                self._add_task_state_warning(
                    "D06 hand contact reporting is incomplete and will fail "
                    f"closed in Isaac: {','.join(missing_sides)}"
                )
                return
            try:
                import omni.physx
                from pxr import PhysicsSchemaTools, Sdf

                interface = getattr(
                    self, "_physx_contact_report_interface", None
                ) or omni.physx.get_physx_simulation_interface()
                subscribe = getattr(
                    interface, "subscribe_contact_report_events", None
                )
                if interface is None or not callable(subscribe):
                    raise RuntimeError(
                        "PhysX contact-report event interface is unavailable"
                    )
                self._d06_target_actor_names = {
                    self._d06_contact_id(
                        PhysicsSchemaTools.sdfPathToInt(prim.GetPath())
                    ): name
                    for name, prim in target_prims.items()
                }
                self._d06_target_contact_names = {
                    self._d06_contact_id(
                        PhysicsSchemaTools.sdfPathToInt(path)
                    ): name
                    for name, prim in target_prims.items()
                    for path in self._d06_contact_paths_for_prim(prim)
                }
                self._d06_hand_actor_sides = {
                    self._d06_contact_id(
                        PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path))
                    ): side
                    for side, paths in hand_paths.items()
                    for path in paths
                }
                self._d06_hand_contact_sides = {
                    self._d06_contact_id(
                        PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path))
                    ): side
                    for side, paths in hand_paths.items()
                    for body_path in paths
                    for path in self._d06_contact_paths_for_prim(
                        self._stage.GetPrimAtPath(body_path)
                    )
                }
                self._d06_contact_report_interface = interface
                self._d06_contact_report_subscription = subscribe(
                    self._on_d06_contact_report_event
                )
                if self._d06_contact_report_subscription is None:
                    raise RuntimeError(
                        "PhysX contact-report subscription was not created"
                    )
                self._d06_contact_reporting_available = True
            except Exception as exc:
                self._d06_contact_view_errors.update(("left", "right"))
                self._add_task_state_warning(
                    f"Could not bind D06 hand contact events: {exc}"
                )
            return

        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            self._add_task_state_warning(
                "D11 PhysX hand/object contact reporting is unavailable"
            )
            return
        target_prim = self._find_rigid_body_prim("d11_pan")
        if target_prim is None:
            self._add_task_state_warning(
                "D11 contact target rigid body is unavailable: d11_pan"
            )
            return
        max_contacts = max(
            64,
            max((len(paths) for paths in hand_paths.values()), default=0) * 8,
        )
        for side, paths in hand_paths.items():
            if not paths:
                self._d11_contact_view_errors.add(side)
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(target_prim.GetPath()),
                    filter_patterns=paths,
                    max_contact_data_count=max_contacts,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._d11_pan_contact_views[side] = view
                else:
                    self._d11_contact_view_errors.add(side)
            except Exception as exc:
                self._d11_contact_view_errors.add(side)
                self._add_task_state_warning(
                    f"Could not bind D11 {side}-hand contact: {exc}"
                )
        missing_sides = sorted(
            {"left", "right"} - set(self._d11_pan_contact_views)
        )
        if missing_sides:
            self._add_task_state_warning(
                "D11 hand contact reporting is incomplete and will fail "
                f"closed in Isaac: {','.join(missing_sides)}"
            )

    def _d06_hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        finger_prefix = f"{side}_finger"
        palm_name = f"{side}_palm_link"
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (
                prim.GetName() == palm_name
                or prim.GetName().startswith(finger_prefix)
            )
        ]

    def _d06_contact_paths_for_prim(self, root: Any) -> list[Any]:
        if self._stage is None or root is None or not root.IsValid():
            return []
        from pxr import Usd, UsdPhysics

        paths = [root.GetPath()]
        paths.extend(
            prim.GetPath()
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        return paths

    @staticmethod
    def _d06_contact_id(value: Any) -> int:
        """Normalize PhysX's signed path ids to stable unsigned values."""
        return int(value) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _d06_contact_vector_norm(value: Any) -> float:
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            vector = np.asarray(
                [value.x, value.y, value.z], dtype=np.float64
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("PhysX returned a non-finite contact impulse")
        return float(np.linalg.norm(vector))

    def _on_d06_contact_report_event(
        self,
        contact_headers: Any,
        contact_points: Any,
    ) -> None:
        """Track only D06 target/hand pairs without GPU contact tensors.

        Isaac Sim 5.1 can corrupt the dense D06 GPU narrowphase when several
        rigid-contact tensor views are created.  The public contact-report
        callback consumes the same physical manifolds without allocating
        those views, while retaining per-target and per-hand identity.
        """

        try:
            for header in contact_headers:
                actor0 = self._d06_contact_id(header.actor0)
                actor1 = self._d06_contact_id(header.actor1)
                collider0 = self._d06_contact_id(header.collider0)
                collider1 = self._d06_contact_id(header.collider1)
                target0 = self._d06_target_actor_names.get(
                    actor0
                ) or self._d06_target_contact_names.get(collider0)
                target1 = self._d06_target_actor_names.get(
                    actor1
                ) or self._d06_target_contact_names.get(collider1)
                side0 = self._d06_hand_actor_sides.get(
                    actor0
                ) or self._d06_hand_contact_sides.get(collider0)
                side1 = self._d06_hand_actor_sides.get(
                    actor1
                ) or self._d06_hand_contact_sides.get(collider1)
                if target0 is not None and side1 is not None:
                    target_name = target0
                    side = side1
                elif target1 is not None and side0 is not None:
                    target_name = target1
                    side = side0
                else:
                    continue
                endpoints = tuple(
                    sorted(((actor0, collider0), (actor1, collider1)))
                )
                event_type = int(header.type)
                if event_type == 1:  # ContactEventType.CONTACT_LOST
                    self._d06_active_hand_contacts.pop(endpoints, None)
                    continue
                if event_type not in (0, 2):
                    continue

                offset = int(header.contact_data_offset)
                count = int(header.num_contact_data)
                if offset < 0 or count < 0 or offset + count > len(contact_points):
                    raise ValueError(
                        "D06 PhysX contact header references points outside "
                        "the callback buffer"
                    )
                impulse = math.fsum(
                    self._d06_contact_vector_norm(contact_points[index].impulse)
                    for index in range(offset, offset + count)
                )
                dt = max(float(getattr(self, "_sim_dt", 0.0)), 1e-9)
                force = max(
                    impulse / dt,
                    float(np.nextafter(_D06_HAND_CONTACT_FORCE, math.inf)),
                )
                self._d06_active_hand_contacts[endpoints] = (
                    target_name,
                    side,
                    force,
                )
            self._d06_contact_callback_error = None
        except Exception as exc:
            self._d06_contact_callback_error = exc

    def _d06_contact_readings(
        self, side: str
    ) -> dict[str, tuple[float, bool]]:
        unavailable = {name: (0.0, False) for name in _D06_TARGETS}
        if (
            not self._d06_contact_reporting_available
            or self._d06_contact_callback_error is not None
            or side not in {"left", "right"}
        ):
            return unavailable
        force_by_target = {name: 0.0 for name in _D06_TARGETS}
        for target_name, contact_side, force in (
            self._d06_active_hand_contacts.values()
        ):
            if contact_side == side:
                force_by_target[target_name] += float(force)
        return {
            target_name: (float(force_by_target[target_name]), True)
            for target_name in _D06_TARGETS
        }

    def _d06_contact_reading(
        self,
        side: str,
        target_name: str = "d06_mug",
    ) -> tuple[float, bool]:
        return self._d06_contact_readings(side).get(
            target_name, (0.0, False)
        )

    def _d06_contact_event_readings(
        self, side: str
    ) -> dict[str, tuple[bool, bool]]:
        """Return active PhysX contact events without a force threshold."""

        unavailable = {name: (False, False) for name in _D06_TARGETS}
        if (
            not self._d06_contact_reporting_available
            or self._d06_contact_callback_error is not None
            or side not in {"left", "right"}
        ):
            return unavailable
        active_by_target = {name: False for name in _D06_TARGETS}
        for target_name, contact_side, _force in (
            self._d06_active_hand_contacts.values()
        ):
            if contact_side == side:
                active_by_target[target_name] = True
        return {
            target_name: (bool(active_by_target[target_name]), True)
            for target_name in _D06_TARGETS
        }

    def _d06_contact_event_reading(
        self,
        side: str,
        target_name: str = "d06_mug",
    ) -> tuple[bool, bool]:
        return self._d06_contact_event_readings(side).get(
            target_name, (False, False)
        )

    def _d11_pan_contact_reading(self, side: str) -> tuple[float, bool]:
        view = self._d11_pan_contact_views.get(side)
        if view is None:
            return 0.0, False
        return float(self._c04_contact_force(view)), True

    def close(self) -> None:
        # Releasing the subscription before the simulation/stage prevents a
        # stale callback from retaining this environment across a rebuild.
        self._d06_contact_report_subscription = None
        self._d06_contact_report_interface = None
        self._d06_contact_reporting_available = False
        super().close()

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        super().restore_recorded_initial_state(state_parts)
        key_names = {
            "target_poses": _D06_TARGETS,
            "basket_poses": _D06_BASKETS,
            "distractor_poses": _D06_DISTRACTORS,
            "object_poses": _D07_OBJECTS,
            "goal_poses": (
                "d07_mouse_pad",
                "d07_keyboard_frame",
                "d07_stand",
                "d07_drawer",
            ),
            "bread_poses": _D08_BREAD,
            "dish_poses": _D10_DISHES,
        }
        scalar_names = {
            "toaster_pose": "d08_toaster",
            "mug_pose": "d09_mug",
            "cabinet_pose": "d09_cabinet",
            "coffee_machine_pose": "d09_coffee_machine",
            "tray_pose": "d09_tray",
            "dishwasher_pose": "d10_dishwasher",
            "rack_pose": "d10_rack",
            "pan_pose": "d11_pan",
            "sponge_pose": "d11_sponge",
            "sink_pose": "d11_sink",
            "faucet_pose": "d11_faucet",
            "knife_pose": "d12_knife",
            "cutting_board_pose": "d12_cutting_board",
            "meat_pose": "d12_meat",
            "drawer_pose": "d12_drawer",
            "board_goal_pose": "d12_board_goal",
        }
        if self.task_name == _D13:
            key_names = {
                "object_poses": ("d13_tray", *_D13_PAYLOADS),
                "fixture_poses": ("d13_cabinet", "d13_service_zone"),
            }
            scalar_names = {}
        elif self.task_name == _D14:
            key_names = {
                "fruit_poses": _D14_FRUITS,
                "fixture_poses": ("d14_sink", "d14_faucet", "d14_drying_area"),
            }
            scalar_names = {"colander_pose": "d14_colander"}
        elif self.task_name == _D15:
            key_names = {
                "food_poses": _D15_FOODS,
                "box_poses": _D15_BOXES,
                "lid_poses": _D15_LIDS,
                "goal_poses": ("d15_goal_left", "d15_goal_right"),
            }
            scalar_names = {}
        elif self.task_name == _D16:
            key_names = {
                "block_poses": _D16_BLOCKS,
                "tool_poses": _D16_TOOLS,
                "fixture_poses": ("d16_bin", "d16_broom_goal", "d16_dustpan_goal"),
            }
            scalar_names = {}
        elif self.task_name == _D17:
            key_names = {
                "fixture_poses": ("d17_headphone_stand", "d17_flat_stand", "d17_vertical_stand"),
            }
            scalar_names = {
                "headphones_pose": "d17_headphones",
                "laptop_pose": "d17_laptop",
            }
        elif self.task_name == _D18:
            key_names = {"egg_poses": _D18_EGGS}
            scalar_names = {
                "carton_pose": "d18_carton",
                "basket_pose": "d18_basket",
            }
        elif self.task_name == _D19:
            key_names = {
                "block_poses": _D19_BLOCKS,
                "cover_poses": _D19_COVERS,
            }
            scalar_names = {"basket_pose": "d19_basket"}
        elif self.task_name in _d20_stages.TASK_NAMES:
            key_names = {"tableware_poses": _d20.TABLEWARE}
            scalar_names = {
                "hot_dog_pose": _d20.HOT_DOG,
                "tongs_pose": _d20.TONGS,
                "tray_pose": _d20.SERVING_TRAY,
            }
        if self.task_name in _d08_variant.TOAST_TASKS:
            scalar_names["plate_pose"] = "d08_plate"
        elif self.task_name == _D12:
            scalar_names["plate_pose"] = "d12_plate"
        restored = 0
        for key, names in key_names.items():
            values = np.asarray(state_parts.get(key, np.zeros((0, 7))), dtype=np.float64)
            for name, pose in zip(names, values):
                view = self._task_body_views.get(name)
                restored += int(
                    view is not None and self._set_task_body_world_pose(view, pose)
                )
        for key, name in scalar_names.items():
            pose = np.asarray(state_parts.get(key, np.zeros(0)), dtype=np.float64).ravel()
            view = self._task_body_views.get(name)
            if pose.size >= 7 and view is not None:
                restored += int(self._set_task_body_world_pose(view, pose[:7]))
        joint_state_keys = {
            _D07: ("drawer_state", ("d07_drawer_joint",)),
            _D08: ("toaster_state", ("d08_start_button_joint",)),
            _D09: (
                "coffee_state",
                ("d09_cabinet_door_joint", "d09_start_button_joint"),
            ),
            _D10: ("dishwasher_state", ("d10_door_joint", "d10_rack_joint")),
            _D11: (
                "faucet_state",
                ("d11_faucet_valve_joint", "d11_faucet_swivel_joint"),
            ),
            _D12: ("drawer_state", ("d12_drawer_joint",)),
            _D13: ("cabinet_state", ("d13_cabinet_door_joint",)),
            _D14: (
                "faucet_state",
                ("d14_faucet_valve_joint", "d14_faucet_swivel_joint"),
            ),
            _D17: ("laptop_state", ("d17_laptop_hinge_joint",)),
            _D18: ("carton_state", ("d18_carton_lid_joint",)),
            _D20: ("microwave_state", (_d20.DOOR_JOINT, _d20.TIMER_JOINT)),
        }
        joint_key, joint_names = joint_state_keys.get(
            _d20_stages.family_name(_d08_variant.family_name(self.task_name)), ("", ())
        )
        joint_values = np.asarray(
            state_parts.get(joint_key, np.zeros(0)), dtype=np.float64
        ).ravel()
        for index, joint_name in enumerate(joint_names):
            if index < joint_values.size:
                self._set_task_dof_state(joint_name, float(joint_values[index]), 0.0)
        if self.task_name == _D09 and joint_values.size >= 8:
            # Preserve the original seven D09 coffee-state fields for
            # existing recordings and use its former padding slot for the
            # independently articulated right door.
            self._set_task_dof_state(
                "d09_cabinet_right_door_joint",
                float(joint_values[7]),
                0.0,
            )
        if self.task_name in _d08_variant.TOAST_TASKS:
            self._d08_release_started = False
            self._d08_lever_target = _D08_BUTTON_UP_TARGET
            self._d08_set_lever_target(_D08_BUTTON_UP_TARGET)
            self._d08_set_rack_enabled(True)
        if self.task_name in _d20_stages.TASK_NAMES:
            controls = np.asarray(state_parts.get("microwave_controls", []), dtype=np.float64).ravel()
            if controls.size >= 2:
                self._set_task_dof_state(_d20.POWER_JOINT, float(controls[0]), 0.0)
                self._set_task_dof_state(_d20.TIMER_JOINT, float(controls[1]), 0.0)
            hidden = state_parts.get("d20_hidden_state")
            if hidden is not None:
                self._d20.restore_hidden_observation(hidden)
            tongs_state = np.asarray(
                state_parts.get("tongs_state", np.zeros(0)), dtype=np.float64
            ).ravel()
            if tongs_state.size >= 1 and np.isfinite(tongs_state[0]):
                self._set_task_dof_state(_d20.TONGS_JOINT, float(tongs_state[0]), 0.0)
            self._d20_tongs_latched = False
            self._d20_tongs_latch_offset = None
            self._d20_tongs_latch_last_target = None
            if "d20_stage_state" in state_parts:
                self._d20_stage.restore(state_parts["d20_stage_state"])
            self._d20_tool_grip.restore(state_parts.get("d20_tool_grip_state", []))
            assist = np.asarray(state_parts.get("d20_assist_state", []), dtype=np.float64).ravel()
            if assist.size == 37:
                self._d20_tongs_latched = bool(assist[0])
                self._d20_tongs_holding_side = {0:"",1:"right",2:"left"}.get(int(assist[1]), "")
                self._d20_tongs_jaw_target = float(assist[2])
                self._d20_tongs_latch_offset = assist[5:21].reshape(4,4).copy() if assist[3] else None
                self._d20_tongs_latch_last_target = assist[21:37].reshape(4,4).copy() if assist[4] else None
        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()

    # ------------------------------------------------------------------
    # Geometry and event helpers
    # ------------------------------------------------------------------

    def _fallback_pose(self, name: str) -> np.ndarray:
        position = _D06_DISTRACTOR_DEFAULT_POSITIONS.get(name)
        if position is None:
            return super()._fallback_pose(name)
        return np.asarray(
            [*position, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def _pose(self, name: str) -> np.ndarray:
        return self._poses.get(name, self._fallback_pose(name))

    def _relative_pose(self, name: str, reference: str) -> np.ndarray:
        return _invert_pose_matrix(
            _pose_vector_to_matrix(self._pose(reference))
        ) @ _pose_vector_to_matrix(self._pose(name))

    def _inside(
        self,
        name: str,
        reference: str,
        half_xy: tuple[float, float],
        z_bounds: tuple[float, float],
    ) -> bool:
        local = self._relative_pose(name, reference)[:3, 3]
        return bool(
            abs(float(local[0])) <= half_xy[0]
            and abs(float(local[1])) <= half_xy[1]
            and z_bounds[0] <= float(local[2]) <= z_bounds[1]
        )

    def _upright(self, name: str, threshold: float = 0.78) -> bool:
        rotation = _quat_wxyz_to_matrix(self._pose(name)[3:7])
        return bool(float(np.dot(rotation[:, 2], np.asarray([0.0, 0.0, 1.0]))) >= threshold)

    def _stable(self, name: str) -> bool:
        velocity = self._physics_body_velocity(name)
        if velocity is None:
            return False
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
            return False
        return bool(
            np.linalg.norm(velocity[:3]) <= _STABLE_LINEAR_SPEED
            and np.linalg.norm(velocity[3:]) <= _STABLE_ANGULAR_SPEED
        )

    def _not_dropped(self, names: tuple[str, ...]) -> bool:
        return all(float(self._pose(name)[2]) >= _TABLE_TOP_Z - 0.12 for name in names)

    def _joint(self, name: str) -> tuple[float, float]:
        reading = self._physics_dof_position(name)
        return (0.0, 0.0) if reading is None else reading

    def _joint_delta(self, name: str) -> float:
        value, initial = self._joint(name)
        return float(value - initial)

    def _set_task_dof_state(
        self, name: str, position: float, velocity: float
    ) -> bool:
        entry = self._task_dof_views.get(name)
        if entry is None:
            entry = next(
                (
                    value
                    for dof_name, value in self._task_dof_views.items()
                    if name in dof_name
                ),
                None,
            )
        if entry is None or self._torch is None:
            return False
        view, index, _initial = entry
        positions = view.get_dof_positions().clone()
        velocities = view.get_dof_velocities().clone()
        positions[:, int(index)] = float(position)
        velocities[:, int(index)] = float(velocity)
        indices = self._torch.arange(
            int(view.count),
            dtype=self._torch.long,
            device=self._isaac_device,
        )
        view.set_dof_positions(positions, indices)
        view.set_dof_velocities(velocities, indices)
        return True

    def _reset_state_maps(self):
        rigid = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity in self._task_reset_body_states
        }
        articulations = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity, _q, _qd
            in self._task_reset_articulation_states
        }
        return rigid, articulations

    def _write_reset_pose(
        self,
        name: str,
        *,
        xy: np.ndarray | tuple[float, float] | None = None,
        z: float | None = None,
        yaw_degrees: float = 0.0,
    ) -> bool:
        prim = self._find_rigid_body_prim(name)
        if prim is None:
            self._add_task_state_warning(f"D-long task body is unavailable: {name}")
            return False
        path = str(prim.GetPath())
        rigid, articulations = self._reset_state_maps()
        state = self._reset_transform_state_for_path(path, rigid, articulations)
        if state is None:
            self._add_task_state_warning(
                f"D-long task reset tensor is unavailable: {name}"
            )
            return False
        state_kind, view, indices, root_pose, root_velocity = state
        pose_np = root_pose.detach().cpu().numpy()
        xy_offset = np.zeros(2, dtype=np.float64)
        if xy is not None:
            xy_offset = np.asarray(xy, dtype=np.float64).reshape(2) - pose_np[0, :2]
        z_offset = 0.0 if z is None else float(z) - float(pose_np[0, 2])
        randomized = _randomized_physx_pose(
            pose_np,
            xy_offset=xy_offset,
            z_offset=z_offset,
            yaw_degrees=float(yaw_degrees),
        )
        pose_tensor = self._torch.as_tensor(
            randomized,
            dtype=root_pose.dtype,
            device=root_pose.device,
        )
        if state_kind == "articulation":
            view.set_root_transforms(pose_tensor, indices)
            zero_velocity = self._torch.zeros_like(root_velocity)
            view.set_root_velocities(zero_velocity, indices)
            if self.task_name in _d20_stages.TASK_NAMES:
                # GPU readback may still contain the pre-write base transform.
                # Register the exact seeded payload just as the rigid writer
                # does, so the origin audit compares physics with this reset.
                self._record_authored_task_reset_write(
                    path, transform_xyzw=pose_tensor, velocity=zero_velocity,
                )
        else:
            self._write_rigid_reset_state(
                path,
                view,
                indices,
                pose_tensor,
                self._torch.zeros_like(root_velocity),
            )
        return True

    def _dlong_sample_xy(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        return self._random.uniform(bounds[0], bounds[1]).astype(np.float64)

    def _dlong_reset_hand_points(self, side: str) -> tuple[np.ndarray, ...]:
        """Return live palm/finger origins for tabletop clearance sampling.

        The shared ABC helper intentionally returns every arm-link origin.
        That is useful for compact A/B/C layouts, but D06's wide clutter lane
        runs below the upper-arm projection.  Counting those high links as
        tabletop hand obstacles makes the dense sampler declare a feasible
        scene impossible and use its deterministic fallback.  The actual
        collision risk here is the complete palm/finger envelope, which this
        task family enables and samples directly.
        """

        robot = self._robots.get(side)
        if robot is not None:
            points: list[np.ndarray] = []
            try:
                body_names = list(robot.body_names)
                body_poses = robot.data.body_pose_w[0]
                for index, name in enumerate(body_names):
                    normalized = str(name).lower()
                    if "palm" not in normalized and "finger" not in normalized:
                        continue
                    point = body_poses[int(index), :3]
                    if hasattr(point, "detach"):
                        point = point.detach().cpu().numpy()
                    point_np = np.asarray(point, dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(point_np)):
                        points.append(point_np)
            except Exception:
                points = []
            if points:
                return tuple(points)

        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        return (np.asarray(fallback[:3], dtype=np.float64).copy(),)

    def _dlong_layout_margin(
        self,
        xy_by_name: dict[str, np.ndarray],
        *,
        pair_distance: float = _DLONG_DEFAULT_PAIR_DISTANCE,
        hand_clearance: float = _DLONG_OBJECT_HAND_CLEARANCE,
        footprint_radii: Mapping[str, float] | None = None,
        vertical_envelopes: Mapping[str, tuple[float, float]] | None = None,
        hand_footprint_half_extents: Mapping[str, np.ndarray] | None = None,
    ) -> float:
        all_hand_points = (
            *self._dlong_reset_hand_points("right"),
            *self._dlong_reset_hand_points("left"),
        )
        hand_points = tuple(
            point
            for point in all_hand_points
            if float(point[2]) <= _DLONG_HAND_CLEARANCE_MAX_Z
        )
        margin = math.inf
        for name, xy in xy_by_name.items():
            footprint_radius = (
                float(footprint_radii.get(name, 0.0))
                if footprint_radii is not None
                else 0.0
            )
            vertical_envelope = (
                vertical_envelopes.get(name)
                if vertical_envelopes is not None
                else None
            )
            if vertical_envelope is not None:
                half_extents = (
                    hand_footprint_half_extents.get(name)
                    if hand_footprint_half_extents is not None
                    else None
                )
                clearance = min(
                    (
                        (
                            self._dlong_point_box_margin(
                                point,
                                xy,
                                half_extents_xy=half_extents,
                                center_z=float(vertical_envelope[0]),
                                half_height=float(vertical_envelope[1]),
                            )
                            if half_extents is not None
                            else self._dlong_point_cylinder_margin(
                                point,
                                xy,
                                radius=footprint_radius,
                                center_z=float(vertical_envelope[0]),
                                half_height=float(vertical_envelope[1]),
                            )
                        )
                        for point in hand_points
                    ),
                    default=math.inf,
                )
                margin = min(margin, clearance - hand_clearance)
            else:
                clearance = min(
                    (
                        float(np.linalg.norm(xy - point[:2]))
                        for point in hand_points
                    ),
                    default=math.inf,
                )
                margin = min(
                    margin,
                    clearance - hand_clearance - footprint_radius,
                )
        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                distance = float(np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b]))
                required_distance = pair_distance
                if footprint_radii is not None:
                    required_distance = max(
                        required_distance,
                        float(footprint_radii.get(name_a, 0.0))
                        + float(footprint_radii.get(name_b, 0.0)),
                    )
                margin = min(margin, distance - required_distance)
        return float(margin)

    @staticmethod
    def _dlong_point_cylinder_margin(
        point_xyz: np.ndarray,
        body_xy: np.ndarray,
        *,
        radius: float,
        center_z: float,
        half_height: float,
    ) -> float:
        """Signed 3-D distance to a conservative finite body cylinder."""
        point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        xy = np.asarray(body_xy, dtype=np.float64).reshape(2)
        radial_offset = float(np.linalg.norm(point[:2] - xy)) - radius
        vertical_offset = abs(float(point[2]) - center_z) - half_height
        outside = np.maximum(
            np.asarray([radial_offset, vertical_offset], dtype=np.float64),
            0.0,
        )
        if np.any(outside > 0.0):
            return float(np.linalg.norm(outside))
        return max(radial_offset, vertical_offset)

    @staticmethod
    def _dlong_point_box_margin(
        point_xyz: np.ndarray,
        body_xy: np.ndarray,
        *,
        half_extents_xy: np.ndarray,
        center_z: float,
        half_height: float,
    ) -> float:
        """Signed 3-D distance to an axis-aligned finite body box."""
        point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        center = np.asarray(
            [body_xy[0], body_xy[1], center_z],
            dtype=np.float64,
        )
        half_extents = np.asarray(
            [half_extents_xy[0], half_extents_xy[1], half_height],
            dtype=np.float64,
        )
        offsets = np.abs(point - center) - half_extents
        outside = np.maximum(offsets, 0.0)
        if np.any(outside > 0.0):
            return float(np.linalg.norm(outside))
        return float(np.max(offsets))

    @staticmethod
    def _dlong_incremental_layout_margin(
        xy: np.ndarray,
        placed_xy: np.ndarray,
        hand_xy: np.ndarray,
        *,
        pair_distance: float,
        hand_clearance: float,
    ) -> float:
        """Return only the constraints introduced by one sampled point."""
        margin = math.inf
        if hand_xy.size:
            clearance = float(np.linalg.norm(hand_xy - xy, axis=1).min())
            margin = min(margin, clearance - hand_clearance)
        if placed_xy.size:
            separation = float(np.linalg.norm(placed_xy - xy, axis=1).min())
            margin = min(margin, separation - pair_distance)
        return float(margin)

    def _dlong_sample_separated_layout(
        self,
        bounds_by_name: Mapping[str, tuple[np.ndarray, np.ndarray]],
        *,
        pair_distance: float = _DLONG_DEFAULT_PAIR_DISTANCE,
        hand_clearance: float = _DLONG_OBJECT_HAND_CLEARANCE,
        footprint_radii: Mapping[str, float] | None = None,
        vertical_envelopes: Mapping[str, tuple[float, float]] | None = None,
        hand_footprint_half_extents: Mapping[str, np.ndarray] | None = None,
        candidates_per_object: int = _DLONG_RESET_CANDIDATES_PER_OBJECT,
    ) -> dict[str, np.ndarray]:
        all_hand_points = (
            *self._dlong_reset_hand_points("right"),
            *self._dlong_reset_hand_points("left"),
        )
        hand_points = tuple(
            point
            for point in all_hand_points
            if float(point[2]) <= _DLONG_HAND_CLEARANCE_MAX_Z
        )
        hand_xy = (
            np.asarray(hand_points, dtype=np.float64)[:, :2]
            if hand_points
            else np.empty((0, 2), dtype=np.float64)
        )
        best: dict[str, np.ndarray] | None = None
        best_margin = -math.inf
        rejected_counts: dict[str, int] = {}
        rejected_best_margins: dict[str, float] = {}
        for _ in range(_DLONG_RESET_MAX_ATTEMPTS):
            sample_order = list(bounds_by_name)
            self._random.shuffle(sample_order)
            if footprint_radii is not None:
                # Place the physically largest objects first.  Random greedy
                # ordering frequently cornered D06 after 15-17 placements and
                # then collapsed every body onto the midpoint fallback.
                # Python's stable sort preserves the seeded shuffle for equal
                # radii, while sampled positions remain fully randomized.
                sample_order.sort(
                    key=lambda name: float(footprint_radii.get(name, 0.0)),
                    reverse=True,
                )
            candidate: dict[str, np.ndarray] = {}
            failed = False

            for name in sample_order:
                bounds = bounds_by_name[name]
                accepted_xy: np.ndarray | None = None
                accepted_margin = -math.inf
                placed_xy = (
                    np.asarray(tuple(candidate.values()), dtype=np.float64)
                    if candidate
                    else np.empty((0, 2), dtype=np.float64)
                )
                placed_names = tuple(candidate)
                footprint_radius = (
                    float(footprint_radii.get(name, 0.0))
                    if footprint_radii is not None
                    else 0.0
                )
                valid_candidates: list[tuple[float, np.ndarray]] = []

                for _candidate_index in range(max(1, int(candidates_per_object))):
                    xy = self._dlong_sample_xy(bounds)
                    margin = math.inf
                    if hand_xy.size:
                        vertical_envelope = (
                            vertical_envelopes.get(name)
                            if vertical_envelopes is not None
                            else None
                        )
                        if vertical_envelope is not None:
                            half_extents = (
                                hand_footprint_half_extents.get(name)
                                if hand_footprint_half_extents is not None
                                else None
                            )
                            hand_distance = min(
                                (
                                    self._dlong_point_box_margin(
                                        point,
                                        xy,
                                        half_extents_xy=half_extents,
                                        center_z=float(vertical_envelope[0]),
                                        half_height=float(vertical_envelope[1]),
                                    )
                                    if half_extents is not None
                                    else self._dlong_point_cylinder_margin(
                                        point,
                                        xy,
                                        radius=footprint_radius,
                                        center_z=float(vertical_envelope[0]),
                                        half_height=float(vertical_envelope[1]),
                                    )
                                )
                                for point in hand_points
                            )
                            margin = min(
                                margin,
                                hand_distance - hand_clearance,
                            )
                        else:
                            hand_distance = float(
                                np.linalg.norm(hand_xy - xy, axis=1).min()
                            )
                            margin = min(
                                margin,
                                hand_distance
                                - hand_clearance
                                - footprint_radius,
                            )
                    if placed_xy.size:
                        separations = np.linalg.norm(placed_xy - xy, axis=1)
                        required_distances = np.full(
                            separations.shape,
                            pair_distance,
                            dtype=np.float64,
                        )
                        if footprint_radii is not None:
                            required_distances = np.maximum(
                                required_distances,
                                np.asarray(
                                    [
                                        footprint_radius
                                        + float(
                                            footprint_radii.get(
                                                placed_name,
                                                0.0,
                                            )
                                        )
                                        for placed_name in placed_names
                                    ],
                                    dtype=np.float64,
                                ),
                            )
                        margin = min(
                            margin,
                            float((separations - required_distances).min()),
                        )
                    if margin > accepted_margin:
                        accepted_xy = xy
                        accepted_margin = margin
                    if margin >= _DLONG_RESET_MIN_CLEARANCE_SLACK:
                        valid_candidates.append((margin, xy))

                if valid_candidates:
                    # Choose among the best few max-clearance candidates.  A
                    # first-valid point can trap the remaining dense clutter;
                    # choosing from a seeded top set is robust without making
                    # layouts deterministic.
                    valid_candidates.sort(key=lambda item: item[0], reverse=True)
                    top_candidates = valid_candidates[:16]
                    selected = int(self._random.randint(len(top_candidates)))
                    accepted_margin, accepted_xy = top_candidates[selected]

                if (
                    accepted_xy is None
                    or accepted_margin < _DLONG_RESET_MIN_CLEARANCE_SLACK
                ):
                    rejected_counts[name] = rejected_counts.get(name, 0) + 1
                    rejected_best_margins[name] = max(
                        rejected_best_margins.get(name, -math.inf),
                        accepted_margin,
                    )
                    failed = True
                    break

                candidate[name] = accepted_xy

            if failed or set(candidate) != set(bounds_by_name):
                continue

            margin = self._dlong_layout_margin(
                candidate,
                pair_distance=pair_distance,
                hand_clearance=hand_clearance,
                footprint_radii=footprint_radii,
                vertical_envelopes=vertical_envelopes,
                hand_footprint_half_extents=hand_footprint_half_extents,
            )
            if margin > best_margin:
                best = {name: xy.copy() for name, xy in candidate.items()}
                best_margin = margin
            if margin >= _DLONG_RESET_MIN_CLEARANCE_SLACK:
                return candidate
        if best is not None and best_margin >= _DLONG_RESET_MIN_CLEARANCE_SLACK:
            return best
        midpoint = {
            name: 0.5 * (bounds[0] + bounds[1])
            for name, bounds in bounds_by_name.items()
        }
        midpoint_margin = self._dlong_layout_margin(
            midpoint,
            pair_distance=pair_distance,
            hand_clearance=hand_clearance,
            footprint_radii=footprint_radii,
            vertical_envelopes=vertical_envelopes,
            hand_footprint_half_extents=hand_footprint_half_extents,
        )
        if midpoint_margin >= _DLONG_RESET_MIN_CLEARANCE_SLACK:
            return midpoint
        rejected_summary = ",".join(
            f"{name}:{count}@{rejected_best_margins[name]:.6f}"
            for name, count in sorted(
                rejected_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        hand_bounds = "none"
        if hand_points:
            hand_xyz = np.asarray(hand_points, dtype=np.float64)
            hand_bounds = (
                f"min={np.round(hand_xyz.min(axis=0), 4).tolist()},"
                f"max={np.round(hand_xyz.max(axis=0), 4).tolist()}"
            )
        raise RuntimeError(
            f"{self.task_name} 无法生成无穿模初始布局: "
            f"best_margin={best_margin:.6f}, "
            f"midpoint_margin={midpoint_margin:.6f}, "
            f"rejected={rejected_summary or 'none'}, "
            f"hand_bounds={hand_bounds}"
        )

    def _hand_near(self, side: str, name: str, distance: float = _HAND_NEAR_DISTANCE) -> bool:
        points = self._a06_hand_points(side)
        if not points:
            return False
        position = self._pose(name)[:3]
        return min(float(np.linalg.norm(point - position)) for point in points) <= distance

    def _hands_released_from(
        self,
        names: tuple[str, ...],
        distance: float = _HAND_NEAR_DISTANCE,
    ) -> tuple[bool, bool, float]:
        """Return fail-closed geometric hand clearance for terminal states."""
        points = tuple(self._a06_hand_points("left")) + tuple(
            self._a06_hand_points("right")
        )
        if not points or not names:
            return False, False, 0.0
        min_distance = min(
            float(np.linalg.norm(point - self._pose(name)[:3]))
            for point in points
            for name in names
        )
        return bool(min_distance >= distance), True, float(min_distance)

    def _d06_hand_supports_mug(self, side: str) -> bool:
        active, available = self._d06_contact_event_reading(side)
        return bool(available and active)

    def _progress(self, values: tuple[Any, ...], size: int) -> np.ndarray:
        result = np.zeros(size, dtype=np.float64)
        flat = np.asarray(values, dtype=np.float64).ravel()
        result[: min(size, flat.size)] = flat[:size]
        return result

    def _fixture_displacement(self, name: str) -> float:
        current = self._pose(name)
        initial = self._initial_fixture_poses.setdefault(name, current.copy())
        return float(np.linalg.norm(current[:3] - initial[:3]))

    # ------------------------------------------------------------------
    # Task state machines and metrics
    # ------------------------------------------------------------------

    def _d06_metrics(self, *, update: bool) -> dict[str, Any]:
        placement: dict[str, str | None] = {}
        category_counts = {basket: 0 for basket in _D06_BASKETS}
        wrong_baskets: set[str] = set()
        correct_count = 0
        wrong_count = 0
        for target in _D06_TARGETS:
            containing = [
                basket
                for basket in _D06_BASKETS
                if self._inside(
                    target,
                    basket,
                    _D06_BASKET_CENTERLINE_HALF_XY,
                    (-0.01, 0.24),
                )
            ]
            placement[target] = containing[0] if containing else None
            correct = _D06_CATEGORY[target] in containing
            correct_count += int(correct)
            category_counts[_D06_CATEGORY[target]] += int(correct)
            wrong_count += int(
                any(basket != _D06_CATEGORY[target] for basket in containing)
            )
            wrong_baskets.update(
                basket for basket in containing if basket != _D06_CATEGORY[target]
            )

        distractor_placement: dict[str, str | None] = {}
        for distractor in _D06_DISTRACTORS:
            containing = [
                basket
                for basket in _D06_BASKETS
                if self._inside(
                    distractor,
                    basket,
                    (0.12, 0.095),
                    (-0.01, 0.24),
                )
            ]
            distractor_placement[distractor] = (
                containing[0] if containing else None
            )
            wrong_count += int(bool(containing))
            wrong_baskets.update(containing)

        mug_position = self._pose("d06_mug")[:3]
        mug_lifted = (
            float(mug_position[2]) > _TABLE_TOP_Z + _D06_HANDOFF_LIFT_HEIGHT
        )
        mug_in_basket = placement["d06_mug"] is not None
        side_contact_readings = {
            side: self._d06_contact_readings(side)
            for side in ("left", "right")
        }
        contact_readings = {
            target: {
                side: side_contact_readings[side][target]
                for side in ("left", "right")
            }
            for target in _D06_TARGETS
        }
        side_contact_event_readings = {
            side: self._d06_contact_event_readings(side)
            for side in ("left", "right")
        }
        contact_event_readings = {
            target: {
                side: side_contact_event_readings[side][target]
                for side in ("left", "right")
            }
            for target in _D06_TARGETS
        }
        left_force, left_contact_available = contact_readings["d06_mug"][
            "left"
        ]
        right_force, right_contact_available = contact_readings["d06_mug"][
            "right"
        ]
        left_contact_active, left_contact_event_available = (
            contact_event_readings["d06_mug"]["left"]
        )
        right_contact_active, right_contact_event_available = (
            contact_event_readings["d06_mug"]["right"]
        )
        left_hand_near = self._hand_near("left", "d06_mug")
        right_hand_near = self._hand_near("right", "d06_mug")
        left_support = bool(
            left_hand_near
            or (left_contact_event_available and left_contact_active)
        )
        right_support = bool(
            right_hand_near
            or (right_contact_event_available and right_contact_active)
        )
        simultaneous_contact = bool(left_support and right_support)
        if update and not self._d06_handoff_complete:
            if (
                self._d06_handoff_source is None
                and left_support != right_support
            ):
                self._d06_handoff_source = (
                    "left" if left_support else "right"
                )
            # A single frame with both hands on the mug is the complete D06
            # handoff.  Geometric hand points keep this robust when PhysX
            # contact callbacks miss a short overlap at low collection FPS.
            if simultaneous_contact:
                if self._d06_handoff_source is None:
                    self._d06_handoff_source = "both"
                self._d06_both_confirmed = True
                self._d06_handoff_complete = True
        baskets_upright = all(
            self._upright(name) and self._stable(name) for name in _D06_BASKETS
        )
        target_stable = {
            name: self._stable(name)
            for name in _D06_TARGETS
        }
        _hands_clear, hand_pose_available, hand_min_distance = (
            self._hands_released_from(_D06_TARGETS)
        )
        release_evidence_available = all(
            available
            for target_readings in contact_readings.values()
            for _force, available in target_readings.values()
        )
        contacted_targets = tuple(
            target
            for target, target_readings in contact_readings.items()
            if any(
                available and force >= _D06_HAND_CONTACT_FORCE
                for force, available in target_readings.values()
            )
        )
        release_ready = bool(
            correct_count == len(_D06_TARGETS)
            and wrong_count == 0
            and release_evidence_available
            and not contacted_targets
        )
        if update:
            self._timers["d06_release"] = (
                min(
                    _D06_RELEASE_HOLD_SECONDS,
                    self._timers.get("d06_release", 0.0)
                    + max(float(self._control_dt), 0.0),
                )
                if release_ready
                else 0.0
            )
        release_seconds = float(self._timers.get("d06_release", 0.0))
        targets_released = bool(
            release_ready
            and release_seconds + 1e-9 >= _D06_RELEASE_HOLD_SECONDS
        )
        success = (
            correct_count == len(_D06_TARGETS)
            and wrong_count == 0
            and self._d06_handoff_complete
            and baskets_upright
            and targets_released
        )
        return {
            "d06_correct_count": correct_count,
            "d06_wrong_count": wrong_count,
            "d06_placements": placement,
            "d06_distractor_placements": distractor_placement,
            "d06_category_counts": category_counts,
            "d06_wrong_baskets": tuple(sorted(wrong_baskets)),
            "d06_handoff_source": self._d06_handoff_source or "",
            "d06_left_support": left_support,
            "d06_right_support": right_support,
            "d06_left_hand_near": left_hand_near,
            "d06_right_hand_near": right_hand_near,
            "d06_left_contact_available": bool(left_contact_available),
            "d06_right_contact_available": bool(right_contact_available),
            "d06_left_contact_active": bool(left_contact_active),
            "d06_right_contact_active": bool(right_contact_active),
            "d06_left_contact_event_available": bool(
                left_contact_event_available
            ),
            "d06_right_contact_event_available": bool(
                right_contact_event_available
            ),
            "d06_left_contact_force": float(left_force),
            "d06_right_contact_force": float(right_force),
            "d06_mug_in_basket": mug_in_basket,
            "d06_simultaneous_hand_contact": simultaneous_contact,
            "d06_both_hands_confirmed": self._d06_both_confirmed,
            "d06_handoff_complete": self._d06_handoff_complete,
            "d06_baskets_upright": baskets_upright,
            "d06_target_stable": target_stable,
            "d06_targets_stable": all(target_stable.values()),
            "d06_hands_released": targets_released,
            "d06_release_evidence_available": release_evidence_available,
            "d06_contacted_targets": contacted_targets,
            "d06_release_seconds": release_seconds,
            "d06_release_hold_seconds": _D06_RELEASE_HOLD_SECONDS,
            "d06_hand_min_distance": hand_min_distance,
            "d06_hand_pose_available": hand_pose_available,
            "success_now": success,
        }

    def _d07_metrics(self, *, update: bool) -> dict[str, Any]:
        mouse_ok = self._inside("d07_mouse", "d07_mouse_pad", (0.095, 0.075), (-0.03, 0.08))
        keyboard_local = self._relative_pose("d07_keyboard", "d07_keyboard_frame")
        keyboard_ok = bool(
            self._inside(
                "d07_keyboard",
                "d07_keyboard_frame",
                (0.22, 0.085),
                (-0.03, 0.08),
            )
            and float(keyboard_local[0, 0]) >= math.cos(math.radians(15.0))
        )
        figurine_ok = bool(
            self._inside(
                "d07_figurine",
                "d07_stand",
                (0.055, 0.055),
                (-0.01, 0.16),
            )
            and self._upright("d07_figurine")
        )
        alarm_inside = bool(
            self._inside(
                "d07_alarm_clock",
                "d07_drawer_tray",
                (0.145, 0.12),
                (-0.02, 0.12),
            )
            and self._upright("d07_alarm_clock")
        )
        drawer_open = abs(self._joint_delta("d07_drawer_joint")) >= 0.12
        drawer_closed = abs(self._joint_delta("d07_drawer_joint")) <= 0.025
        misc_inside = tuple(
            self._inside(name, "d07_drawer_tray", (0.16, 0.15), (-0.02, 0.12))
            for name in ("d07_misc_0", "d07_misc_1", "d07_misc_2")
        )
        if update:
            self._event_flags["d07_drawer_opened"] = (
                self._event_flags.get("d07_drawer_opened", False) or drawer_open
            )
        no_falls = self._not_dropped(_D07_OBJECTS)
        goals_done = sum((mouse_ok, keyboard_ok, figurine_ok, alarm_inside))
        object_stable = {
            name: self._stable(name)
            for name in _D07_OBJECTS
        }
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(_D07_OBJECTS)
        )
        success = (
            goals_done == 4
            and all(misc_inside)
            and self._event_flags.get("d07_drawer_opened", False)
            and drawer_closed
            and no_falls
            and all(object_stable.values())
            and hand_pose_available
            and hands_released
        )
        return {
            "d07_goal_count": goals_done,
            "d07_mouse_on_pad": mouse_ok,
            "d07_keyboard_in_frame": keyboard_ok,
            "d07_figurine_on_stand": figurine_ok,
            "d07_alarm_in_drawer": alarm_inside,
            "d07_misc_inside": sum(misc_inside),
            "d07_drawer_opened": self._event_flags.get("d07_drawer_opened", False),
            "d07_drawer_closed": drawer_closed,
            "d07_no_falls": no_falls,
            "d07_object_stable": object_stable,
            "d07_objects_stable": all(object_stable.values()),
            "d07_hand_pose_available": hand_pose_available,
            "d07_hands_released": hands_released,
            "d07_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d08_set_lever_target(self, travel: float) -> bool:
        """Latch or release the physical toaster lever through its USD drive."""
        target = float(
            np.clip(
                travel,
                _D08_BUTTON_UP_TARGET,
                _D08_LEVER_LATCH_TRAVEL,
            )
        )
        self._d08_lever_target = target
        if self._stage is None:
            return not bool(getattr(self, "_launch_isaac", False))
        from pxr import UsdPhysics

        joint = self._stage.GetPrimAtPath(
            f"{_TASK_ROOT}/joints/d08_start_button_joint"
        )
        if joint is None or not joint.IsValid():
            return False
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        if not drive:
            return False
        drive.GetTargetPositionAttr().Set(target)
        drive.GetTargetVelocityAttr().Set(0.0)
        latched = target >= _D08_LEVER_PRESS_TRAVEL
        drive.GetStiffnessAttr().Set(
            _D08_BUTTON_STIFFNESS if latched else _D08_LEVER_IDLE_STIFFNESS
        )
        drive.GetDampingAttr().Set(
            _D08_BUTTON_DAMPING if latched else _D08_LEVER_IDLE_DAMPING
        )
        drive.GetMaxForceAttr().Set(
            _D08_BUTTON_MAX_FORCE if latched else _D08_LEVER_IDLE_MAX_FORCE
        )
        return True

    def _d08_set_rack_enabled(self, enabled: bool) -> bool:
        """Toggle the fixed high support behind the lever-linked rack visual."""
        if self._stage is None:
            return False
        from pxr import Usd, UsdPhysics

        if not self._d08_rack_collision_paths:
            rack_names = set(_D08_RACK_COLLISION_NAMES)
            matched_names: set[str] = set()
            collider_paths: list[str] = []
            for prim in self._stage.Traverse():
                geom_name = prim.GetName()
                if geom_name not in rack_names:
                    continue
                for candidate in Usd.PrimRange(prim):
                    if candidate.HasAPI(UsdPhysics.CollisionAPI):
                        collider_paths.append(str(candidate.GetPath()))
                        matched_names.add(geom_name)
            if matched_names == rack_names:
                self._d08_rack_collision_paths = tuple(
                    dict.fromkeys(collider_paths)
                )
        if not self._d08_rack_collision_paths:
            return False
        for path in self._d08_rack_collision_paths:
            prim = self._stage.GetPrimAtPath(path)
            if (
                prim is None
                or not prim.IsValid()
                or not prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                return False
            collision = UsdPhysics.CollisionAPI(prim)
            attr = collision.GetCollisionEnabledAttr()
            if not attr or not attr.IsValid():
                attr = collision.CreateCollisionEnabledAttr()
            attr.Set(bool(enabled))
        self._event_flags["d08_rack_enabled"] = bool(enabled)
        return True

    def _d08_latch_bread_to_rack(self, lever_travel: float) -> bool:
        """Keep both slices rigidly aligned with the lever-driven rack."""
        if not self._d08_latched_bread_poses:
            return False
        toaster_z = float(self._pose("d08_toaster")[2])
        raised_z = (_d08_variant.BREAD_RAISED_CENTER_Z if self.task_name == _d08_variant.TASK_NAME else _D08_BREAD_RACK_CENTER_LOCAL_Z)
        rack_travel = float(np.clip(lever_travel, 0.0, _D08_LEVER_LATCH_TRAVEL))
        bread_z = toaster_z + raised_z - rack_travel
        updated = True
        for name, latched_pose in self._d08_latched_bread_poses.items():
            view = self._task_body_views.get(name)
            if view is None:
                updated = False
                continue
            pose = latched_pose.copy()
            pose[2] = bread_z
            updated = self._set_task_body_world_pose(view, pose) and updated
        return updated

    def _d08_metrics(self, *, update: bool) -> dict[str, Any]:
        # The compact toaster uses 32 mm slot spacing.  Assign each slice to
        # its nearest slot with a non-overlapping band so two slices cannot be
        # credited to the same physical opening.
        variant = self.task_name == _d08_variant.TASK_NAME
        cycle_seconds = _d08_variant.CYCLE_SECONDS if variant else _D08_TOAST_CYCLE_SECONDS
        slot_centers = _d08_variant.SLOT_CENTERS if variant else (-0.032, 0.032)
        assignments: list[int | None] = []
        bread_local_z: list[float] = []
        for bread in _D08_BREAD:
            relative = self._relative_pose(bread, "d08_toaster")
            local = relative[:3, 3]
            bread_local_z.append(float(local[2]))
            nearest_slot = min(
                range(len(slot_centers)),
                key=lambda index: abs(float(local[1]) - slot_centers[index]),
            )
            slot_offset = abs(
                float(local[1]) - slot_centers[nearest_slot]
            )
            # Local Y is the 12 mm thickness axis.  It must point across the
            # slot, but either local X or local Z may point upward: the new
            # nearly-square toast has no required top edge and may be inserted
            # after an in-plane quarter turn.
            thin_axis_aligned = abs(float(relative[1, 1])) >= math.cos(
                math.radians(60.0)
            )
            inserted = bool(
                abs(float(local[0])) <= 0.075
                and slot_offset <= 0.024
                and 0.050 <= float(local[2]) <= 0.185
                and thin_axis_aligned
            )
            if variant:
                inserted = bool(
                    abs(float(local[0])) <= .018
                    and slot_offset <= .0025
                    and .091 <= float(local[2]) <= .153
                    and abs(float(relative[1, 1])) >= .97
                    and abs(float(relative[2, 2])) >= .94
                )
            assignments.append(nearest_slot if inserted else None)
        both_inserted = None not in assignments and len(set(assignments)) == 2
        bread_popped = tuple(
            assignment is not None and local_z >= _D08_POPPED_MIN_LOCAL_Z
            for assignment, local_z in zip(assignments, bread_local_z)
        )
        physical_pop_complete = both_inserted and all(bread_popped)
        lever_travel = self._joint_delta("d08_start_button_joint")
        lever_pressed = lever_travel >= _D08_LEVER_PRESS_TRAVEL
        lever_returned = lever_travel <= _D08_LEVER_RETURN_TRAVEL
        lever_edge = lever_pressed and not self._event_flags.get(
            "d08_button_down", False
        )
        if update and not self._d08_cycle_started and not self._d08_cycle_complete:
            self._d08_set_lever_target(_D08_BUTTON_UP_TARGET)
            self._d08_set_rack_enabled(True)
            # Bread weight and pickup forces must never back-drive the rack.
            # Keep it mechanically pinned at the top until both slices are
            # currently inserted; only then may the operator press the lever.
            if not both_inserted:
                self._set_task_dof_state(
                    "d08_start_button_joint",
                    0.0,
                    0.0,
                )
        if update and both_inserted:
            self._d08_slots_seen = True
            self._d08_pop_failure_reason = None
        if (
            update
            and lever_edge
            and not self._d08_slots_seen
            and not self._d08_cycle_started
            and not self._d08_cycle_complete
        ):
            self._d08_pop_failure_reason = "lever_pressed_before_two_slices_inserted"
        if (
            update
            and self._d08_slots_seen
            and (not variant or both_inserted)
            and lever_edge
            and not self._d08_cycle_started
            and not self._d08_cycle_complete
        ):
            self._d08_cycle_started = True
            self._d08_release_started = False
            self._event_flags["d08_button_triggered"] = True
            self._event_flags["d08_cycle_interrupted"] = False
            self._event_flags["d08_physical_pop_complete"] = False
            self._timers["d08_cycle"] = 0.0
            self._d08_latched_bread_poses = {
                name: self._pose(name).copy() for name in _D08_BREAD
            }
            self._d08_set_lever_target(_D08_LEVER_LATCH_TRAVEL)
            self._d08_set_rack_enabled(False)
        if update:
            self._event_flags["d08_button_down"] = lever_pressed
        if (
            update
            and self._d08_cycle_started
            and not self._d08_cycle_complete
        ):
            # Once a valid press starts the cycle, keep the lever latched for
            # the full toast duration.  The moving rack changes the bread pose
            # while descending, so the initial insertion predicate must not
            # cancel an already-started cycle on a transient frame.
            if not self._event_flags.get("d08_cycle_interrupted", False):
                if not self._d08_release_started:
                    # Lock the lever and both slices as one mechanism.  The
                    # bread pose is written to the exact rack height, avoiding
                    # contact-solver oscillation while the toaster is latched.
                    self._d08_set_lever_target(_D08_LEVER_LATCH_TRAVEL)
                    self._set_task_dof_state(
                        "d08_start_button_joint",
                        _D08_LEVER_LATCH_TRAVEL,
                        0.0,
                    )
                    self._d08_latch_bread_to_rack(_D08_LEVER_LATCH_TRAVEL)
                    self._timers["d08_cycle"] = min(
                        cycle_seconds,
                        self._timers.get("d08_cycle", 0.0) + self._control_dt,
                    )
                    if self._timers["d08_cycle"] >= cycle_seconds:
                        if self._d08_set_lever_target(_D08_BUTTON_UP_TARGET):
                            self._d08_release_started = True
                            self._d08_pop_failure_reason = None
                        else:
                            self._d08_pop_failure_reason = (
                                "lever_release_target_unavailable"
                            )
                else:
                    self._d08_latch_bread_to_rack(lever_travel)
                if (
                    self._d08_release_started
                    and lever_returned
                ):
                    # The two slices have been explicitly latched to the rack
                    # since a valid two-slot trigger.  Snap the coupled system
                    # to its upper stop before completing the pop instead of
                    # rechecking the stale pose captured at frame start.
                    self._set_task_dof_state(
                        "d08_start_button_joint",
                        0.0,
                        0.0,
                    )
                    self._d08_latch_bread_to_rack(0.0)
                    self._d08_cycle_complete = True
                    self._event_flags["d08_physical_pop_complete"] = True
                    self._event_flags["d08_pop_applied"] = True
                    self._d08_set_rack_enabled(True)
                    self._d08_latched_bread_poses.clear()
                    self._d08_pop_failure_reason = None
        in_toaster_after_pop = {
            name: self._inside(
                name,
                "d08_toaster",
                (0.075, 0.085),
                (0.025, 0.18),
            )
            for name in _D08_BREAD
        }
        if update and self._d08_cycle_complete:
            # Hold the popped rack at its upper stop while the operator removes
            # the slices.  Pulling a slice can no longer drag the rack and the
            # other slice back into the toaster.
            self._d08_set_lever_target(_D08_BUTTON_UP_TARGET)
            self._set_task_dof_state(
                "d08_start_button_joint",
                0.0,
                0.0,
            )
            self._d08_removed_after_pop.update(
                name
                for name, still_in_toaster in in_toaster_after_pop.items()
                if not still_in_toaster
            )
        removed_after_pop = all(
            name in self._d08_removed_after_pop for name in _D08_BREAD
        )
        on_plate = tuple(
            bool(
                self._inside(
                    name,
                    "d08_plate",
                    (0.09, 0.09),
                    (-0.02, 0.10),
                )
            )
            for name in _D08_BREAD
        )
        bread_stable = {
            name: self._stable(name)
            for name in _D08_BREAD
        }
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(_D08_BREAD)
        )
        toaster_displacement = self._fixture_displacement("d08_toaster")
        toaster_stationary = toaster_displacement <= 0.04
        placement_ready = bool(
            self._d08_cycle_complete
            and removed_after_pop
            and all(on_plate)
            and all(bread_stable.values())
            and toaster_stationary
        )
        release_ready = bool(
            placement_ready
            and hand_pose_available
            and hands_released
        )
        if update:
            self._timers["d08_release"] = (
                min(
                    _D08_RELEASE_HOLD_SECONDS,
                    self._timers.get("d08_release", 0.0)
                    + max(float(self._control_dt), 0.0),
                )
                if release_ready
                else 0.0
            )
        release_seconds = float(self._timers.get("d08_release", 0.0))
        release_complete = bool(
            release_ready
            and release_seconds + 1e-9 >= _D08_RELEASE_HOLD_SECONDS
        )
        success = (
            self._d08_slots_seen
            and self._event_flags.get("d08_button_triggered", False)
            and self._d08_cycle_complete
            and placement_ready
            and release_complete
        )
        return {
            "d08_bread_inserted": sum(item is not None for item in assignments),
            "d08_two_slots_seen": self._d08_slots_seen,
            "d08_button_pressed": lever_pressed,
            # Preserve the old metric as a read-only compatibility alias for
            # recordings produced before the physical button was introduced.
            "d08_lever_pressed": lever_pressed,
            "d08_button_triggered": self._event_flags.get(
                "d08_button_triggered", False
            ),
            "d08_lever_returned": lever_returned,
            "d08_lever_target": self._d08_lever_target,
            "d08_cycle_started": self._d08_cycle_started,
            "d08_cycle_seconds": self._timers.get("d08_cycle", 0.0),
            "d08_release_started": self._d08_release_started,
            "d08_cycle_complete": self._d08_cycle_complete,
            "d08_cycle_interrupted": self._event_flags.get(
                "d08_cycle_interrupted", False
            ),
            "d08_pop_applied": self._event_flags.get(
                "d08_pop_applied", False
            ),
            "d08_bread_popped": bread_popped,
            "d08_physical_pop_complete": self._event_flags.get(
                "d08_physical_pop_complete", False
            ),
            "d08_pop_failure_reason": self._d08_pop_failure_reason or "",
            "d08_removed_after_pop_count": len(self._d08_removed_after_pop),
            "d08_removed_after_pop": removed_after_pop,
            "d08_plate_count": sum(on_plate),
            "d08_bread_stable": bread_stable,
            "d08_placement_ready": placement_ready,
            "d08_toaster_displacement": toaster_displacement,
            "d08_toaster_stationary": toaster_stationary,
            "d08_hand_pose_available": hand_pose_available,
            "d08_hands_released": hands_released,
            "d08_hand_min_distance": hand_min_distance,
            "d08_release_seconds": release_seconds,
            "d08_release_hold_seconds": _D08_RELEASE_HOLD_SECONDS,
            "d08_release_complete": release_complete,
            "success_now": success,
        }

    def _d09_metrics(self, *, update: bool) -> dict[str, Any]:
        left_door_delta = abs(self._joint_delta("d09_cabinet_door_joint"))
        right_door_delta = abs(
            self._joint_delta("d09_cabinet_right_door_joint")
        )
        door_open = max(left_door_delta, right_door_delta) >= 0.65
        door_closed = max(left_door_delta, right_door_delta) <= 0.10
        mug_out = not self._inside("d09_mug", "d09_cabinet", (0.18, 0.16), (-0.05, 0.30))
        outlet_local = (
            self._relative_pose("d09_mug", "d09_coffee_machine")[:3, 3]
            - _D09_MUG_BREW_LOCAL_XYZ
        )
        under_outlet = bool(
            abs(float(outlet_local[0]))
            <= float(_D09_MUG_OUTLET_XY_TOLERANCE[0])
            and abs(float(outlet_local[1]))
            <= float(_D09_MUG_OUTLET_XY_TOLERANCE[1])
            and abs(float(outlet_local[2])) <= _D09_MUG_BASE_Z_TOLERANCE
            and self._upright("d09_mug")
        )
        button_pressed = (
            abs(self._joint_delta("d09_start_button_joint"))
            >= _D09_BUTTON_TRIGGER_TRAVEL
        )
        if update:
            self._event_flags["d09_door_opened"] = (
                self._event_flags.get("d09_door_opened", False) or door_open
            )
            if self._event_flags["d09_door_opened"] and mug_out:
                self._event_flags["d09_mug_removed"] = True
            if self._event_flags.get("d09_mug_removed", False) and door_closed:
                self._event_flags["d09_door_reclosed"] = True
            if self._event_flags.get("d09_mug_removed", False) and under_outlet:
                self._event_flags["d09_positioned"] = True
            # Brewing is a direct physical interaction: an aligned mug plus
            # a depressed button starts the machine immediately.  Cabinet
            # history remains part of final task success, but must never make
            # a correctly pressed coffee-machine button appear broken.
            if under_outlet and button_pressed and not self._d09_brew_complete:
                self._d09_brew_started = True
            if self._d09_brew_started:
                if not under_outlet and not self._d09_brew_complete:
                    self._event_flags["d09_brew_interrupted"] = True
                if under_outlet and not self._event_flags.get(
                    "d09_brew_interrupted", False
                ):
                    self._timers["d09_brew"] = (
                        self._timers.get("d09_brew", 0.0) + self._control_dt
                    )
                    if self._timers["d09_brew"] >= _D09_BREW_SECONDS:
                        self._d09_brew_complete = True
            self._event_flags["d09_button_down"] = button_pressed
        brew_interrupted = self._event_flags.get(
            "d09_brew_interrupted", False
        )
        water_emitting = bool(
            self._d09_brew_started
            and not self._d09_brew_complete
            and not brew_interrupted
            and under_outlet
        )
        on_tray = self._inside("d09_mug", "d09_tray", (0.12, 0.10), (-0.02, 0.13))
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(("d09_mug",))
        )
        success = (
            self._event_flags.get("d09_door_reclosed", False)
            and self._d09_brew_complete
            and door_closed
            and on_tray
            and self._stable("d09_mug")
            and hand_pose_available
            and hands_released
        )
        return {
            "d09_door_opened": self._event_flags.get("d09_door_opened", False),
            "d09_mug_removed": self._event_flags.get("d09_mug_removed", False),
            "d09_door_closed": door_closed,
            "d09_under_outlet": under_outlet,
            "d09_button_pressed": button_pressed,
            "d09_brew_started": self._d09_brew_started,
            "d09_brew_seconds": self._timers.get("d09_brew", 0.0),
            "d09_brew_complete": self._d09_brew_complete,
            "d09_brew_interrupted": brew_interrupted,
            "d09_water_emitting": water_emitting,
            "d09_on_tray": on_tray,
            "d09_hand_pose_available": hand_pose_available,
            "d09_hands_released": hands_released,
            "d09_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d10_metrics(self, *, update: bool) -> dict[str, Any]:
        door_delta = abs(self._joint_delta("d10_door_joint"))
        rack_delta = abs(self._joint_delta("d10_rack_joint"))
        door_open = door_delta >= 0.70
        door_closed = door_delta <= 0.10
        rack_out = rack_delta >= 0.16
        rack_in = rack_delta <= 0.025
        slot_y = {"d10_mug": -0.12, "d10_bowl": 0.0, "d10_plate": 0.12}
        slot_ok: dict[str, bool] = {}
        for name, center_y in slot_y.items():
            local = self._relative_pose(name, "d10_rack")[:3, 3]
            slot_ok[name] = bool(
                abs(float(local[0])) <= 0.17
                and abs(float(local[1]) - center_y) <= 0.055
                and -0.02 <= float(local[2]) <= 0.18
                and self._stable(name)
            )
        if update:
            self._event_flags["d10_door_opened"] = (
                self._event_flags.get("d10_door_opened", False) or door_open
            )
            if self._event_flags["d10_door_opened"] and rack_out:
                self._event_flags["d10_rack_pulled"] = True
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(_D10_DISHES)
        )
        success = (
            all(slot_ok.values())
            and self._event_flags.get("d10_rack_pulled", False)
            and rack_in
            and door_closed
            and hand_pose_available
            and hands_released
        )
        return {
            "d10_slot_count": sum(slot_ok.values()),
            "d10_slot_ok": slot_ok,
            "d10_door_opened": self._event_flags.get("d10_door_opened", False),
            "d10_rack_pulled": self._event_flags.get("d10_rack_pulled", False),
            "d10_rack_in": rack_in,
            "d10_door_closed": door_closed,
            "d10_hand_pose_available": hand_pose_available,
            "d10_hands_released": hands_released,
            "d10_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d11_metrics(self, *, update: bool) -> dict[str, Any]:
        pan_inside = self._inside("d11_pan", "d11_sink", (0.25, 0.22), (-0.04, 0.20))
        sponge_inside = self._inside("d11_sponge", "d11_sink", (0.25, 0.22), (-0.04, 0.20))
        valve_open = abs(self._joint_delta("d11_faucet_valve_joint")) >= 0.35
        left_pan_near = self._hand_near("left", "d11_pan", 0.14)
        right_pan_near = self._hand_near("right", "d11_pan", 0.14)
        left_pan_force, left_pan_contact_available = (
            self._d11_pan_contact_reading("left")
        )
        right_pan_force, right_pan_contact_available = (
            self._d11_pan_contact_reading("right")
        )
        left_pan = bool(
            left_pan_near
            and left_pan_contact_available
            and left_pan_force >= _D11_HAND_CONTACT_FORCE
        )
        right_pan = bool(
            right_pan_near
            and right_pan_contact_available
            and right_pan_force >= _D11_HAND_CONTACT_FORCE
        )
        left_valve = self._hand_near("left", "d11_faucet_valve", 0.12)
        right_valve = self._hand_near("right", "d11_faucet_valve", 0.12)
        pan_held = left_pan or right_pan
        bimanual_open = (left_pan and right_valve) or (
            right_pan and left_valve
        )
        if update and valve_open and bimanual_open:
            self._event_flags["d11_bimanual_open"] = True
        if "d11_faucet_spout" in self._poses:
            # Compatibility with state fixtures recorded before the invalid
            # zero-mass spout body was folded into its parent arm.
            outlet = self._poses["d11_faucet_spout"][:2]
        else:
            faucet_arm = _pose_vector_to_matrix(self._pose("d11_faucet_arm"))
            outlet = (
                faucet_arm
                @ np.asarray([0.12, 0.0, -0.10, 1.0], dtype=np.float64)
            )[:2]
        pan_xy = self._pose("d11_pan")[:2]
        delta = outlet - pan_xy
        over_pan = bool(abs(float(delta[0])) <= 0.16 and abs(float(delta[1])) <= 0.14)
        quadrant = int(delta[0] >= 0.0) * 2 + int(delta[1] >= 0.0)
        pan_stable = pan_held and self._stable("d11_pan")
        if (
            update
            and valve_open
            and pan_inside
            and pan_stable
            and over_pan
        ):
            self._d11_coverage_seconds[quadrant] += self._control_dt
        coverage_ok = bool(np.all(self._d11_coverage_seconds >= 0.25))
        sponge_stable = self._stable("d11_sponge")
        success = (
            pan_inside
            and sponge_inside
            and sponge_stable
            and valve_open
            and pan_stable
            and self._event_flags.get("d11_bimanual_open", False)
            and coverage_ok
        )
        return {
            "d11_pan_inside": pan_inside,
            "d11_sponge_inside": sponge_inside,
            "d11_sponge_stable": sponge_stable,
            "d11_valve_open": valve_open,
            "d11_bimanual_open": self._event_flags.get(
                "d11_bimanual_open", False
            ),
            "d11_pan_held": pan_held,
            "d11_pan_contact_available": bool(
                left_pan_contact_available or right_pan_contact_available
            ),
            "d11_left_pan_contact_force": left_pan_force,
            "d11_right_pan_contact_force": right_pan_force,
            "d11_pan_stable": pan_stable,
            "d11_outlet_over_pan": over_pan,
            "d11_coverage_seconds": self._d11_coverage_seconds.copy(),
            "d11_coverage_ok": coverage_ok,
            "success_now": success,
        }

    def _d12_metrics(self, *, update: bool) -> dict[str, Any]:
        drawer_delta = abs(self._joint_delta("d12_drawer_joint"))
        drawer_open = drawer_delta >= 0.12
        drawer_closed = drawer_delta <= 0.025
        if update:
            self._event_flags["d12_drawer_opened"] = (
                self._event_flags.get("d12_drawer_opened", False) or drawer_open
            )
        board_local = self._relative_pose("d12_cutting_board", "d12_board_goal")
        board_aligned = bool(
            np.linalg.norm(board_local[:2, 3]) <= 0.055
            and abs(float(board_local[2, 3])) <= 0.055
            and float(board_local[0, 0]) >= math.cos(math.radians(12.0))
            and float(board_local[1, 1]) >= math.cos(math.radians(12.0))
            and float(board_local[2, 2]) >= math.cos(math.radians(12.0))
        )
        meat_centered = self._inside(
            "d12_meat", "d12_cutting_board", (0.075, 0.065), (-0.02, 0.10)
        )
        knife_local = self._relative_pose("d12_knife", "d12_board_goal")
        knife_position = knife_local[:3, 3]
        knife_corners = np.asarray(
            [
                knife_local
                @ np.asarray([x, y, 0.0, 1.0], dtype=np.float64)
                for x in (-0.012, 0.012)
                for y in (-0.110, 0.110)
            ]
        )
        knife_safe = bool(
            np.all(np.abs(knife_corners[:, 0] - 0.145) <= 0.055)
            and np.all(np.abs(knife_corners[:, 1]) <= 0.225)
            and -0.02 <= float(knife_position[2]) <= 0.08
        )
        # Knife meshes use local +Y from handle toward blade tip.
        blade_outward = bool(
            float(np.dot(knife_local[:3, 1], np.asarray([0.0, -1.0, 0.0])))
            >= math.cos(math.radians(20.0))
        )
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(
                ("d12_cutting_board", "d12_meat", "d12_knife")
            )
        )
        success = (
            self._event_flags.get("d12_drawer_opened", False)
            and drawer_closed
            and board_aligned
            and meat_centered
            and knife_safe
            and blade_outward
            and self._stable("d12_cutting_board")
            and self._stable("d12_meat")
            and self._stable("d12_knife")
            and hand_pose_available
            and hands_released
        )
        return {
            "d12_drawer_opened": self._event_flags.get("d12_drawer_opened", False),
            "d12_drawer_closed": drawer_closed,
            "d12_board_aligned": board_aligned,
            "d12_meat_centered": meat_centered,
            "d12_knife_safe": knife_safe,
            "d12_blade_outward": blade_outward,
            "d12_hand_pose_available": hand_pose_available,
            "d12_hands_released": hands_released,
            "d12_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d13_metrics(self, *, update: bool) -> dict[str, Any]:
        local = {
            name: self._relative_pose(name, "d13_tray")[:3, 3]
            for name in _D13_PAYLOADS
        }
        kettle_ok = bool(
            abs(float(local["d13_kettle"][0])) <= 0.10
            and abs(float(local["d13_kettle"][1]) - 0.075) <= 0.075
            and -0.02 <= float(local["d13_kettle"][2]) <= 0.16
            and self._upright("d13_kettle")
        )
        mug_ok = bool(
            abs(float(local["d13_mug"][0])) <= 0.10
            and abs(float(local["d13_mug"][1]) + 0.075) <= 0.065
            and -0.02 <= float(local["d13_mug"][2]) <= 0.16
            and self._upright("d13_mug")
        )
        tea_ok = bool(
            abs(float(local["d13_tea_bag"][0]) - 0.13) <= 0.055
            and abs(float(local["d13_tea_bag"][1])) <= 0.040
            and -0.02 <= float(local["d13_tea_bag"][2]) <= 0.10
        )
        door_delta = abs(self._joint_delta("d13_cabinet_door_joint"))
        door_open = door_delta >= 0.65
        door_closed = door_delta <= 0.10
        tray_in_service = self._inside(
            "d13_tray", "d13_service_zone", (0.25, 0.17), (-0.04, 0.14)
        )
        tray_rotation = _quat_wxyz_to_matrix(self._pose("d13_tray")[3:7])
        tilt_degrees = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(tray_rotation[:, 2], np.asarray([0.0, 0.0, 1.0])),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        tray_lifted = float(self._pose("d13_tray")[2]) >= _TABLE_TOP_Z + 0.07
        transport_started = bool(
            self._event_flags.get("d13_transport_started", False)
            or tray_lifted
            or self._fixture_displacement("d13_tray") >= 0.10
        )
        both_hands_support = bool(
            tray_lifted
            and self._hand_near("left", "d13_tray", 0.34)
            and self._hand_near("right", "d13_tray", 0.34)
        )
        if update:
            self._event_flags["d13_cabinet_opened"] = bool(
                self._event_flags.get("d13_cabinet_opened", False) or door_open
            )
            self._event_flags["d13_transport_started"] = transport_started
            self._event_flags["d13_bimanual_support"] = bool(
                self._event_flags.get("d13_bimanual_support", False)
                or both_hands_support
            )
            if transport_started:
                self._d13_peak_tilt_degrees = max(
                    self._d13_peak_tilt_degrees, tilt_degrees
                )
                dropped = any(
                    float(self._pose(name)[2]) < _TABLE_TOP_Z - 0.05
                    or float(np.linalg.norm(local[name][:2])) > 0.36
                    for name in _D13_PAYLOADS
                )
                self._event_flags["d13_payload_dropped"] = bool(
                    self._event_flags.get("d13_payload_dropped", False) or dropped
                )
        payloads_ok = kettle_ok and mug_ok and tea_ok
        no_drop = not self._event_flags.get("d13_payload_dropped", False)
        stable = all(self._stable(name) for name in ("d13_tray", *_D13_PAYLOADS))
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(("d13_tray", *_D13_PAYLOADS), distance=0.11)
        )
        success = bool(
            payloads_ok
            and self._event_flags.get("d13_cabinet_opened", False)
            and door_closed
            and self._event_flags.get("d13_bimanual_support", False)
            and transport_started
            and tray_in_service
            and self._d13_peak_tilt_degrees <= 15.0
            and no_drop
            and stable
            and hand_pose_available
            and hands_released
        )
        return {
            "d13_kettle_left": kettle_ok,
            "d13_mug_right": mug_ok,
            "d13_tea_bag_on_tray": tea_ok,
            "d13_payload_count": sum((kettle_ok, mug_ok, tea_ok)),
            "d13_cabinet_opened": self._event_flags.get("d13_cabinet_opened", False),
            "d13_cabinet_closed": door_closed,
            "d13_transport_started": transport_started,
            "d13_bimanual_support": self._event_flags.get("d13_bimanual_support", False),
            "d13_tray_in_service": tray_in_service,
            "d13_tilt_degrees": tilt_degrees,
            "d13_peak_tilt_degrees": self._d13_peak_tilt_degrees,
            "d13_payload_dropped": not no_drop,
            "d13_stable": stable,
            "d13_hands_released": hands_released,
            "d13_hand_pose_available": hand_pose_available,
            "d13_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d14_metrics(self, *, update: bool) -> dict[str, Any]:
        fruits_inside = tuple(
            self._inside(name, "d14_colander", (0.135, 0.135), (-0.03, 0.18))
            for name in _D14_FRUITS
        )
        colander_in_sink = self._inside(
            "d14_colander", "d14_sink", (0.22, 0.23), (-0.06, 0.22)
        )
        colander_in_drying = self._inside(
            "d14_colander", "d14_drying_area", (0.20, 0.16), (-0.04, 0.18)
        )
        valve_delta = abs(self._joint_delta("d14_faucet_valve_joint"))
        valve_open = valve_delta >= 0.35
        valve_closed = valve_delta <= 0.10
        arm_matrix = _pose_vector_to_matrix(self._pose("d14_faucet_arm"))
        outlet = (
            arm_matrix @ np.asarray([0.20, 0.0, -0.26, 1.0], dtype=np.float64)
        )[:3]
        covered_now = tuple(
            bool(
                valve_open
                and fruits_inside[index]
                and float(np.linalg.norm(self._pose(name)[:2] - outlet[:2])) <= 0.10
            )
            for index, name in enumerate(_D14_FRUITS)
        )
        if update:
            self._event_flags["d14_faucet_opened"] = bool(
                self._event_flags.get("d14_faucet_opened", False) or valve_open
            )
            for index, covered in enumerate(covered_now):
                if covered:
                    self._d14_coverage_seconds[index] = min(
                        0.25,
                        self._d14_coverage_seconds[index] + self._control_dt,
                    )
            if all(fruits_inside):
                self._event_flags["d14_all_loaded"] = True
            if self._event_flags.get("d14_all_loaded", False):
                dropped = any(
                    float(self._pose(name)[2]) < _TABLE_TOP_Z - 0.05
                    for name in _D14_FRUITS
                )
                self._event_flags["d14_fruit_dropped"] = bool(
                    self._event_flags.get("d14_fruit_dropped", False) or dropped
                )
        washed = self._d14_coverage_seconds >= 0.25
        all_washed = bool(np.all(washed))
        fruits_stable = all(self._stable(name) for name in _D14_FRUITS)
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(("d14_colander", *_D14_FRUITS), distance=0.11)
        )
        success = bool(
            all(fruits_inside)
            and all_washed
            and self._event_flags.get("d14_faucet_opened", False)
            and valve_closed
            and colander_in_drying
            and self._stable("d14_colander")
            and fruits_stable
            and not self._event_flags.get("d14_fruit_dropped", False)
            and hand_pose_available
            and hands_released
        )
        return {
            "d14_fruit_inside": fruits_inside,
            "d14_fruit_count": sum(fruits_inside),
            "d14_colander_in_sink": colander_in_sink,
            "d14_colander_in_drying": colander_in_drying,
            "d14_valve_open": valve_open,
            "d14_valve_closed": valve_closed,
            "d14_faucet_opened": self._event_flags.get("d14_faucet_opened", False),
            "d14_covered_now": covered_now,
            "d14_coverage_seconds": self._d14_coverage_seconds.copy(),
            "d14_washed_count": int(np.sum(washed)),
            "d14_all_washed": all_washed,
            "d14_fruit_dropped": self._event_flags.get("d14_fruit_dropped", False),
            "d14_hands_released": hands_released,
            "d14_hand_pose_available": hand_pose_available,
            "d14_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d15_metrics(self, *, update: bool) -> dict[str, Any]:
        counts = np.zeros((2, 3), dtype=np.int64)
        assignments: dict[str, int | None] = {}
        for kind in range(3):
            for index in range(4):
                name = f"d15_food_{kind}_{index}"
                boxes = [
                    box_index
                    for box_index, box_name in enumerate(_D15_BOXES)
                    if self._inside(name, box_name, (0.13, 0.10), (-0.03, 0.15))
                ]
                assignments[name] = boxes[0] if len(boxes) == 1 else None
                if len(boxes) == 1:
                    counts[boxes[0], kind] += 1
        lids_closed = []
        for lid_name, box_name in zip(_D15_LIDS, _D15_BOXES):
            relative = self._relative_pose(lid_name, box_name)
            local = relative[:3, 3]
            lids_closed.append(
                bool(
                    np.linalg.norm(local[:2]) <= 0.035
                    and 0.045 <= float(local[2]) <= 0.095
                    and float(relative[2, 2]) >= math.cos(math.radians(12.0))
                )
            )
        goals = tuple(
            self._inside(box, goal, (0.16, 0.13), (-0.04, 0.15))
            for box, goal in zip(_D15_BOXES, ("d15_goal_left", "d15_goal_right"))
        )
        matching = bool(np.array_equal(counts[0], counts[1]))
        exact = bool(np.all(counts == 2))
        all_packed = sum(item is not None for item in assignments.values()) == 12
        stable = all(
            self._stable(name)
            for name in (*_D15_BOXES, *_D15_LIDS, *_D15_FOODS)
        )
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from((*_D15_BOXES, *_D15_LIDS, *_D15_FOODS))
        )
        success = bool(
            matching
            and exact
            and all_packed
            and all(lids_closed)
            and all(goals)
            and stable
            and hand_pose_available
            and hands_released
        )
        return {
            "d15_counts": counts.copy(),
            "d15_assignments": assignments,
            "d15_matching": matching,
            "d15_exact_counts": exact,
            "d15_packed_count": sum(item is not None for item in assignments.values()),
            "d15_lids_closed": tuple(lids_closed),
            "d15_goals": goals,
            "d15_stable": stable,
            "d15_hands_released": hands_released,
            "d15_hand_pose_available": hand_pose_available,
            "d15_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d16_metrics(self, *, update: bool) -> dict[str, Any]:
        in_dustpan = tuple(
            self._inside(name, "d16_dustpan", (0.13, 0.11), (-0.04, 0.14))
            for name in _D16_BLOCKS
        )
        in_bin = tuple(
            self._inside(name, "d16_bin", (0.15, 0.13), (-0.04, 0.20))
            for name in _D16_BLOCKS
        )
        broom_lifted = float(self._pose("d16_broom")[2]) >= _TABLE_TOP_Z + 0.06
        left_broom = self._hand_near("left", "d16_broom", 0.24)
        right_broom = self._hand_near("right", "d16_broom", 0.24)
        if update:
            if broom_lifted and left_broom and not right_broom:
                self._event_flags["d16_left_source"] = True
            if (
                self._event_flags.get("d16_left_source", False)
                and broom_lifted
                and left_broom
                and right_broom
            ):
                self._event_flags["d16_both_hands"] = True
            if (
                self._event_flags.get("d16_both_hands", False)
                and broom_lifted
                and right_broom
                and not left_broom
            ):
                self._event_flags["d16_handoff_complete"] = True
            self._d16_dustpan_seen.update(
                name for name, inside in zip(_D16_BLOCKS, in_dustpan) if inside
            )
            direct_pick = any(
                name not in self._d16_dustpan_seen
                and float(self._pose(name)[2]) >= _TABLE_TOP_Z + 0.055
                and (
                    self._hand_near("left", name, 0.075)
                    or self._hand_near("right", name, 0.075)
                )
                for name in _D16_BLOCKS
            )
            self._event_flags["d16_direct_pick"] = bool(
                self._event_flags.get("d16_direct_pick", False) or direct_pick
            )
        broom_returned = self._inside(
            "d16_broom", "d16_broom_goal", (0.22, 0.08), (-0.08, 0.14)
        )
        dustpan_returned = self._inside(
            "d16_dustpan", "d16_dustpan_goal", (0.15, 0.13), (-0.06, 0.15)
        )
        all_bin = all(in_bin)
        stable = all(self._stable(name) for name in (*_D16_BLOCKS, *_D16_TOOLS))
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from((*_D16_BLOCKS, *_D16_TOOLS), distance=0.11)
        )
        success = bool(
            self._event_flags.get("d16_handoff_complete", False)
            and len(self._d16_dustpan_seen) == len(_D16_BLOCKS)
            and all_bin
            and not self._event_flags.get("d16_direct_pick", False)
            and broom_returned
            and dustpan_returned
            and stable
            and hand_pose_available
            and hands_released
        )
        return {
            "d16_handoff_complete": self._event_flags.get("d16_handoff_complete", False),
            "d16_dustpan_count": sum(in_dustpan),
            "d16_dustpan_seen_count": len(self._d16_dustpan_seen),
            "d16_bin_count": sum(in_bin),
            "d16_direct_pick": self._event_flags.get("d16_direct_pick", False),
            "d16_broom_returned": broom_returned,
            "d16_dustpan_returned": dustpan_returned,
            "d16_stable": stable,
            "d16_hands_released": hands_released,
            "d16_hand_pose_available": hand_pose_available,
            "d16_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d17_metrics(self, *, update: bool) -> dict[str, Any]:
        headphone_local = self._relative_pose(
            "d17_headphones", "d17_headphone_stand"
        )[:3, 3]
        headphones_hung = bool(
            abs(float(headphone_local[0])) <= 0.10
            and abs(float(headphone_local[1])) <= 0.13
            and 0.14 <= float(headphone_local[2]) <= 0.36
            and self._stable("d17_headphones")
        )
        hinge_delta = abs(self._joint_delta("d17_laptop_hinge_joint"))
        close_error_degrees = abs(math.degrees(math.pi / 2.0 - hinge_delta))
        laptop_closed = close_error_degrees <= 5.0
        laptop_local_matrix = self._relative_pose(
            "d17_laptop", "d17_vertical_stand"
        )
        laptop_local = laptop_local_matrix[:3, 3]
        laptop_rotation = _quat_wxyz_to_matrix(self._pose("d17_laptop")[3:7])
        laptop_vertical = bool(
            abs(float(np.dot(laptop_rotation[:, 2], np.asarray([0.0, 0.0, 1.0]))))
            <= math.sin(math.radians(12.0))
        )
        insertion_depth = float(np.clip((float(laptop_local[2]) - 0.05) / 0.10, 0.0, 1.0))
        inserted = bool(
            abs(float(laptop_local[0])) <= 0.055
            and abs(float(laptop_local[1])) <= 0.13
            and 0.08 <= float(laptop_local[2]) <= 0.25
            and laptop_vertical
            and insertion_depth >= 0.80
        )
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(("d17_headphones", "d17_laptop"), distance=0.11)
        )
        success = bool(
            headphones_hung
            and laptop_closed
            and inserted
            and self._stable("d17_laptop")
            and hand_pose_available
            and hands_released
        )
        return {
            "d17_headphones_hung": headphones_hung,
            "d17_hinge_delta": hinge_delta,
            "d17_close_error_degrees": close_error_degrees,
            "d17_laptop_closed": laptop_closed,
            "d17_laptop_vertical": laptop_vertical,
            "d17_insertion_depth": insertion_depth,
            "d17_inserted": inserted,
            "d17_hands_released": hands_released,
            "d17_hand_pose_available": hand_pose_available,
            "d17_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d18_metrics(self, *, update: bool) -> dict[str, Any]:
        slot_centers = np.asarray(
            [(-0.085, -0.055), (-0.085, 0.055), (0.085, -0.055), (0.085, 0.055)],
            dtype=np.float64,
        )
        assignments: list[int | None] = []
        for egg_index, name in enumerate(_D18_EGGS):
            local = self._relative_pose(name, "d18_carton")[:3, 3]
            distances = np.linalg.norm(slot_centers - local[:2], axis=1)
            slot = int(np.argmin(distances))
            assignments.append(
                slot
                if float(distances[slot]) <= 0.045
                and 0.015 <= float(local[2]) <= 0.14
                else None
            )
            if update and abs(float(local[0])) <= 0.20 and abs(float(local[1])) <= 0.15:
                velocity = self._physics_body_velocity(name)
                if velocity is not None:
                    speed = float(np.linalg.norm(np.asarray(velocity).reshape(-1)[:3]))
                    self._d18_peak_impact_speed[egg_index] = max(
                        self._d18_peak_impact_speed[egg_index], speed
                    )
        occupied = len({slot for slot in assignments if slot is not None})
        lid_delta = abs(self._joint_delta("d18_carton_lid_joint"))
        lid_open = lid_delta >= 1.0
        lid_closed = lid_delta <= 0.12
        if update:
            self._event_flags["d18_lid_opened"] = bool(
                self._event_flags.get("d18_lid_opened", False) or lid_open
            )
        impact_threshold = 0.65
        intact = bool(np.all(self._d18_peak_impact_speed <= impact_threshold))
        carton_in_goal = self._inside(
            "d18_carton", "d18_carton_goal", (0.19, 0.14), (-0.05, 0.16)
        )
        eggs_stable = all(self._stable(name) for name in _D18_EGGS)
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from(("d18_carton", *_D18_EGGS), distance=0.11)
        )
        success = bool(
            occupied == 4
            and None not in assignments
            and intact
            and self._event_flags.get("d18_lid_opened", False)
            and lid_closed
            and carton_in_goal
            and eggs_stable
            and hand_pose_available
            and hands_released
        )
        return {
            "d18_assignments": tuple(assignments),
            "d18_occupied_slots": occupied,
            "d18_lid_opened": self._event_flags.get("d18_lid_opened", False),
            "d18_lid_closed": lid_closed,
            "d18_peak_impact_speed": self._d18_peak_impact_speed.copy(),
            "d18_impact_threshold": impact_threshold,
            "d18_intact": intact,
            "d18_carton_in_goal": carton_in_goal,
            "d18_hands_released": hands_released,
            "d18_hand_pose_available": hand_pose_available,
            "d18_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    def _d19_metrics(self, *, update: bool) -> dict[str, Any]:
        if not self._d19_spatial_order:
            self._d19_spatial_order = tuple(
                sorted(_D19_BLOCKS, key=lambda name: float(self._pose(name)[1]))
            )
        spatial_order = self._d19_spatial_order
        covered = {
            block: any(
                float(np.linalg.norm(self._pose(cover)[:2] - self._pose(block)[:2]))
                <= 0.060
                and -0.02
                <= float(self._pose(cover)[2] - self._pose(block)[2])
                <= 0.12
                for cover in _D19_COVERS
            )
            for block in _D19_BLOCKS
        }
        in_basket = {
            block: self._inside(block, "d19_basket", (0.15, 0.12), (-0.04, 0.20))
            for block in _D19_BLOCKS
        }
        if update:
            previous = self._d19_previous_covered or {
                block: False for block in _D19_BLOCKS
            }
            if not self._event_flags.get("d19_all_covered", False):
                newly_covered = [
                    block
                    for block in spatial_order
                    if covered[block] and not previous.get(block, False)
                ]
                for block in newly_covered:
                    expected_index = len(self._d19_cover_order)
                    if expected_index >= len(spatial_order) or block != spatial_order[expected_index]:
                        self._d19_sequence_errors += 1
                    if block not in self._d19_cover_order:
                        self._d19_cover_order.append(block)
                if tuple(self._d19_cover_order) == spatial_order and all(covered.values()):
                    self._event_flags["d19_all_covered"] = True
            else:
                newly_revealed = [
                    block
                    for block in _D19_COLOR_SEQUENCE
                    if previous.get(block, False)
                    and not covered[block]
                    and block not in self._d19_revealed
                    and block not in self._d19_basketed
                ]
                for block in newly_revealed:
                    expected_index = len(self._d19_revealed)
                    ready = len(self._d19_basketed) == expected_index
                    if (
                        expected_index >= len(_D19_COLOR_SEQUENCE)
                        or block != _D19_COLOR_SEQUENCE[expected_index]
                        or not ready
                    ):
                        self._d19_sequence_errors += 1
                    self._d19_revealed.append(block)
                for block in _D19_COLOR_SEQUENCE:
                    if in_basket[block] and block not in self._d19_basketed:
                        expected_index = len(self._d19_basketed)
                        if (
                            expected_index >= len(_D19_COLOR_SEQUENCE)
                            or block != _D19_COLOR_SEQUENCE[expected_index]
                            or block not in self._d19_revealed
                        ):
                            self._d19_sequence_errors += 1
                        self._d19_basketed.append(block)
            self._d19_previous_covered = dict(covered)
        all_in_basket = all(in_basket.values())
        stable = all(self._stable(name) for name in _D19_BLOCKS)
        hands_released, hand_pose_available, hand_min_distance = (
            self._hands_released_from((*_D19_BLOCKS, *_D19_COVERS), distance=0.11)
        )
        success = bool(
            self._event_flags.get("d19_all_covered", False)
            and tuple(self._d19_cover_order) == spatial_order
            and tuple(self._d19_revealed) == _D19_COLOR_SEQUENCE
            and tuple(self._d19_basketed) == _D19_COLOR_SEQUENCE
            and all_in_basket
            and self._d19_sequence_errors == 0
            and stable
            and hand_pose_available
            and hands_released
        )
        return {
            "d19_spatial_order": spatial_order,
            "d19_covered": covered,
            "d19_cover_count": sum(covered.values()),
            "d19_cover_order": tuple(self._d19_cover_order),
            "d19_all_covered": self._event_flags.get("d19_all_covered", False),
            "d19_revealed": tuple(self._d19_revealed),
            "d19_basketed": tuple(self._d19_basketed),
            "d19_basket_count": sum(in_basket.values()),
            "d19_sequence_errors": self._d19_sequence_errors,
            "d19_stable": stable,
            "d19_hands_released": hands_released,
            "d19_hand_pose_available": hand_pose_available,
            "d19_hand_min_distance": hand_min_distance,
            "success_now": success,
        }

    # ------------------------------------------------------------------
    # D20: microwave hot-dog heating, tongs handling, and plating
    # ------------------------------------------------------------------

    def _d20_configure_stage(self, sim_utils: Any) -> None:
        """Hide D20 collision visuals and author the passive/spring drives."""
        from pxr import PhysxSchema, Usd, UsdPhysics
        from .d20_selected_assets import configure_visuals

        sim_utils.make_uninstanceable(_TASK_ROOT, stage=self._stage)
        sim_utils.update_stage()
        for name in (*_d20.TASK_BODIES, *_d20.EXTRA_POSES):
            self._prim_cache.pop(name, None)
        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        configure_visuals(self._stage, task_root)
        from pxr import Gf
        from .d20_selected_assets import manifest
        # The 5.1 MJCF importer leaves explicit MJCF inertials unauthored in
        # these mesh-only bodies. Author source-derived mass models on the
        # actual USD rigid links before PhysX cooks them; the manifest records
        # the bottle's explicit container/fill assumptions.
        physical_parts = {
            _d20.HOT_DOG: ('hot_dog', .12),
            _d20.TONGS: ('tongs_fixed', .11),
            'd20_tongs_moving_arm': ('tongs_moving', .055),
            _d20.PLATE: ('plate', .28),
            _d20.FORK: ('fork', .035),
            _d20.MUSTARD: ('mustard', .25),
            _d20.MUG: ('mug', .24),
            _d20.MICROWAVE_DOOR: ('microwave_door', .65),
            _d20.POWER_KNOB: ('power_knob', .015),
            _d20.TIMER_KNOB: ('timer_knob', .015),
        }
        for name, (part, mass) in physical_parts.items():
            body = self._find_rigid_body_prim(name)
            if body is None:
                raise RuntimeError(f'D20 mass frame missing: {name}')
            asset = manifest()['assets'][part]
            bounds = np.asarray(asset['bounds'])
            size = bounds[1] - bounds[0]
            diagonal = mass / 12 * np.asarray([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
            center = bounds.mean(0)
            principal_axes = Gf.Quatf(1.0)
            if part in ('mustard', 'mug'):
                properties = asset['mass_properties']
                center = np.asarray(properties['center_of_mass_m'])
                tensor = mass * np.asarray(properties['inertia_per_kg_m2'])
                diagonal, axes = np.linalg.eigh(tensor)
                if np.linalg.det(axes) < 0:
                    axes[:, 0] *= -1
                principal_axes = Gf.Quatf(Gf.Matrix3d(*axes.T.reshape(-1)).ExtractRotation().GetQuat())
            mass_api = UsdPhysics.MassAPI.Apply(body)
            mass_api.CreateMassAttr(float(mass))
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
            mass_api.CreatePrincipalAxesAttr(principal_axes)
        # Thin ceramic/metal meshes need contact solving at their actual
        # scale. These settings keep all bodies dynamic and wakeable.
        for prim in Usd.PrimRange(task_root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                body_api.CreateSolverPositionIterationCountAttr(64 if prim.GetName() in (_d20.MUSTARD, _d20.MUG) else 16)
                body_api.CreateSolverVelocityIterationCountAttr(8 if prim.GetName() in (_d20.MUSTARD, _d20.MUG) else 4)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                articulation_api.CreateSolverPositionIterationCountAttr(64 if prim.GetName() in (_d20.MUSTARD, _d20.MUG) else 16)
                articulation_api.CreateSolverVelocityIterationCountAttr(8 if prim.GetName() in (_d20.MUSTARD, _d20.MUG) else 4)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision_api.CreateContactOffsetAttr(0.001)
                collision_api.CreateRestOffsetAttr(0.0)
        wanted = {_d20.DOOR_JOINT, _d20.POWER_JOINT, _d20.TIMER_JOINT, _d20.TONGS_JOINT}
        joints = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(task_root)
            if prim.GetName() in wanted
        }
        door = joints.get(_d20.DOOR_JOINT)
        if door is not None and door.IsA(UsdPhysics.RevoluteJoint):
            # Passive hinge: the operator owns the door, light damping only.
            drive = UsdPhysics.DriveAPI.Get(door, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(door, "angular")
            drive.GetTargetPositionAttr().Set(0.0)
            drive.GetTargetVelocityAttr().Set(0.0)
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(1.2)
            drive.GetMaxForceAttr().Set(5.0)
        else:
            self._add_task_state_warning("Could not configure the D20 microwave door")
        self._d20_knob_drives = {}
        for name, stiffness, damping in (
            (_d20.POWER_JOINT, 0.015, _d20.POWER_KNOB_DRIVE_DAMPING),
            (_d20.TIMER_JOINT, 0.0, _d20.TIMER_KNOB_DRIVE_DAMPING),
        ):
            joint = joints.get(name)
            if joint is None or not joint.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"D20 rotary joint missing: {name}")
            # Physical end stops, in USD degrees, including cached MJCF imports.
            limit = _d20.POWER_ANGLES[-1] if name == _d20.POWER_JOINT else _d20.TIMER_ANGLES[-1]
            revolute = UsdPhysics.RevoluteJoint(joint)
            revolute.CreateLowerLimitAttr(0.0)
            revolute.CreateUpperLimitAttr(math.degrees(float(limit)))
            initial_degrees = math.degrees(self._d20_reset_power_angle) if name == _d20.POWER_JOINT else 0.0
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
            drive.GetTargetPositionAttr().Set(initial_degrees)
            drive.GetTargetVelocityAttr().Set(0.0)
            drive.GetStiffnessAttr().Set(stiffness)
            drive.GetDampingAttr().Set(damping)
            drive.GetMaxForceAttr().Set(0.015)
            friction_attr = joint.GetAttribute("physxJoint:jointFriction")
            if friction_attr:
                friction_attr.Set(_d20.KNOB_JOINT_FRICTION)
            self._d20_knob_drives[name] = drive
            state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
            state.GetPositionAttr().Set(initial_degrees)
            state.GetVelocityAttr().Set(0.0)
        tongs = joints.get(_d20.TONGS_JOINT)
        if tongs is not None and tongs.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Get(tongs, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(tongs, "angular")
            rest_degrees = math.degrees(_d20.TONGS_OPEN_RAD)
            drive.GetTargetPositionAttr().Set(rest_degrees)
            drive.GetTargetVelocityAttr().Set(0.0)
            drive.GetStiffnessAttr().Set(_d20.TONGS_SPRING_STIFFNESS)
            drive.GetDampingAttr().Set(_d20.TONGS_SPRING_DAMPING)
            drive.GetMaxForceAttr().Set(_d20.TONGS_SPRING_MAX_FORCE)
            friction_attr = tongs.GetAttribute("physxJoint:jointFriction")
            if friction_attr:
                friction_attr.Set(0.012)
            if tongs.HasAPI(PhysxSchema.JointStateAPI, "angular"):
                state = PhysxSchema.JointStateAPI.Get(tongs, "angular")
            else:
                state = PhysxSchema.JointStateAPI.Apply(tongs, "angular")
            state.GetPositionAttr().Set(rest_degrees)
            state.GetVelocityAttr().Set(0.0)
        else:
            self._add_task_state_warning("Could not configure the D20 tongs spring")

    def _d20_init_runtime(self, hand_paths: Mapping[str, list[str]]) -> None:
        """Bind exact hand/hot-dog contact views (tongs open in the DOF hook)."""
        from pxr import PhysxSchema

        self._d20_hot_dog_contact_views = {}
        self._d20_release_contact_views = {}
        self._d20_release_frame_contact = {}
        self._d20_contact_view_errors = set()
        self._d20_knob_contact_views = {}
        self._d20_knob_contact_view_errors = set()
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            self._add_task_state_warning(
                "D20 PhysX hand/hot-dog contact reporting is unavailable"
            )
            return
        target_prim = self._find_rigid_body_prim(_d20.HOT_DOG)
        if target_prim is None:
            self._add_task_state_warning(
                f"D20 contact target rigid body is unavailable: {_d20.HOT_DOG}"
            )
            return
        max_contacts = max(
            64,
            max((len(paths) for paths in hand_paths.values()), default=0) * 8,
        )
        for side, paths in hand_paths.items():
            if not paths:
                self._d20_contact_view_errors.add(side)
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(target_prim.GetPath()),
                    filter_patterns=paths,
                    max_contact_data_count=max_contacts,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._d20_hot_dog_contact_views[side] = view
                else:
                    self._d20_contact_view_errors.add(side)
            except Exception as exc:
                self._d20_contact_view_errors.add(side)
                self._add_task_state_warning(
                    f"Could not bind D20 {side}-hand contact: {exc}"
                )
        missing_sides = sorted(
            {"left", "right"} - set(self._d20_hot_dog_contact_views)
        )
        if missing_sides:
            self._add_task_state_warning(
                "D20 hand contact reporting is incomplete and will fail "
                f"closed in Isaac: {','.join(missing_sides)}"
            )

        # A control is considered held when any hand link is in actual PhysX
        # contact with its rigid body.  This is deliberately a contact sensor
        # rather than a proximity test so a nearby open hand cannot pause the
        # clock or make a replay diverge from collection.
        for knob_name in (_d20.POWER_KNOB, _d20.TIMER_KNOB):
            knob_prim = self._find_rigid_body_prim(knob_name)
            if knob_prim is None or not hand_paths:
                self._d20_knob_contact_view_errors.add(knob_name)
                continue
            try:
                if knob_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    report_api = PhysxSchema.PhysxContactReportAPI.Get(
                        self._stage, knob_prim.GetPath()
                    )
                else:
                    report_api = PhysxSchema.PhysxContactReportAPI.Apply(knob_prim)
                report_api.CreateThresholdAttr().Set(0.0)
                filters = [path for paths in hand_paths.values() for path in paths]
                view = physics_sim_view.create_rigid_contact_view(
                    str(knob_prim.GetPath()),
                    filter_patterns=filters,
                    max_contact_data_count=max(64, len(filters) * 8),
                )
                if view is not None and int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                    self._d20_knob_contact_views[knob_name] = view
                else:
                    self._d20_knob_contact_view_errors.add(knob_name)
            except Exception as exc:
                self._d20_knob_contact_view_errors.add(knob_name)
                self._add_task_state_warning(
                    f"Could not bind D20 {knob_name} hand contact: {exc}"
                )

        tool_paths = [str(self._find_rigid_body_prim(n).GetPath()) for n in (_d20.TONGS, _d20.TONGS_MOVING_ARM)]
        self._d20_tool_contact_views = {
            side: physics_sim_view.create_rigid_contact_view(
                tool_paths, filter_patterns=[list(paths), list(paths)], max_contact_data_count=128,
            ) for side, paths in hand_paths.items() if paths
        }
        self._d20_payload_contact_view = physics_sim_view.create_rigid_contact_view(
            str(target_prim.GetPath()), filter_patterns=tool_paths, max_contact_data_count=64,
        )
        if self._d20_stage.stage == 4:
            self._d20_init_release_contacts(physics_sim_view, hand_paths)

    def _d20_init_release_contacts(self, physics_sim_view, hand_paths) -> None:
        """Reuse the existing hand-link enumeration for the final release check."""
        self._d20_release_contact_views = {}
        targets = [self._find_rigid_body_prim(name) for name in _d20.RELEASE_CONTACT_BODIES]
        if any(prim is None for prim in targets):
            self._add_task_state_warning("D20-3 release contact target is unavailable")
            return
        target_paths = [str(prim.GetPath()) for prim in targets]
        for side in ("left", "right"):
            paths = list(hand_paths.get(side, ()))
            if not paths:
                self._add_task_state_warning(f"D20-3 {side}-hand release detection is unavailable")
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    target_paths, filter_patterns=[paths for _ in target_paths],
                    max_contact_data_count=max(128, len(target_paths) * len(paths) * 8))
                if (view is None or int(view.sensor_count) != len(target_paths)
                        or int(view.filter_count) != len(paths)):
                    raise RuntimeError("incomplete hand/object contact coverage")
                self._d20_release_contact_views[side] = view
            except Exception as exc:
                self._add_task_state_warning(f"D20-3 {side}-hand release detection is unavailable: {exc}")

    def _d20_hand_object_contact(self, side: str) -> tuple[float, bool]:
        view = self._d20_release_contact_views.get(side)
        if view is None:
            return 0.0, False
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            forces = np.asarray(matrix, dtype=np.float64)
            if (forces.ndim != 3 or forces.shape != (len(_d20.RELEASE_CONTACT_BODIES), int(view.filter_count), 3)
                    or forces.shape[1] == 0 or not np.isfinite(forces).all()):
                return 0.0, False
            # Max per body/link avoids opposing finger forces cancelling out.
            return float(np.linalg.norm(forces, axis=-1).max()), True
        except Exception:
            return 0.0, False

    def _d20_sample_release_contact(self) -> None:
        # Preserve a brief touch from any of the four physics substeps until
        # the control-frame success update consumes it. Clear before the next action.
        for side in ("left", "right"):
            force, available = self._d20_hand_object_contact(side)
            previous_force, previous_available = self._d20_release_frame_contact.get(side, (0.0, True))
            self._d20_release_frame_contact[side] = (
                max(force, previous_force), available and previous_available)

    def _d20_knob_hand_hold(self, knob_name: str) -> bool:
        """Return whether any hand link is pressing a D20 knob."""
        view = self._d20_knob_contact_views.get(knob_name)
        if view is None:
            return False
        try:
            forces = view.get_contact_force_matrix(float(self._sim_dt)).detach().cpu().numpy()
            if forces.ndim != 3:
                return False
            link_forces = np.linalg.norm(forces, axis=-1)
            return bool(np.any(np.max(link_forces, axis=0) >= _d20.KNOB_CONTACT_FORCE))
        except Exception:
            return False

    # Compatibility alias for older callers/tests; the detection semantics
    # are intentionally hand-contact (not a two-finger requirement).
    _d20_knob_two_finger_hold = _d20_knob_hand_hold

    def _d20_hand_hot_dog_contact(self, side: str) -> tuple[float, bool]:
        view = self._d20_hot_dog_contact_views.get(side)
        if view is None:
            return 0.0, False
        return float(self._c04_contact_force(view)), True

    def _d20_tongs_dof_entry(self) -> tuple[str, tuple[Any, int, float]] | None:
        """Resolve the tongs hinge DOF view with the shared substring fallback."""
        entry = self._task_dof_views.get(_d20.TONGS_JOINT)
        if entry is not None:
            return _d20.TONGS_JOINT, entry
        for dof_name, value in self._task_dof_views.items():
            if _d20.TONGS_JOINT in dof_name:
                return dof_name, value
        return None

    def _capture_task_scene_state(self, *args: Any, **kwargs: Any):
        # The shared runtime creates the task DOF views here, after
        # ``_init_task_specific_runtime``.  Open the tongs only once those
        # views exist so the cached reset state and the live DOF agree.
        result = super()._capture_task_scene_state(*args, **kwargs)
        if self.task_name in _d20_stages.TASK_NAMES:
            print(
                "[HandsOn/IsaacLab] D20 task DOF views: "
                + ",".join(sorted(self._task_dof_views)),
                flush=True,
            )
            self._d20_initialize_tongs_open_state()
            for name, value in ((_d20.POWER_JOINT, self._d20_reset_power_angle), (_d20.TIMER_JOINT, 0.0)):
                for dof_key, (view, index, _initial) in list(self._task_dof_views.items()):
                    if name not in dof_key:
                        continue
                    self._set_task_dof_state(dof_key, value, 0.0)
                    for _path, cached_view, _indices, _root_pose, _root_velocity, positions, velocities in self._task_reset_articulation_states:
                        if cached_view is view:
                            positions[:, index] = value
                            velocities[:, index] = 0.0
                    self._task_dof_views[dof_key] = (view, index, value)
        return result

    def _restore_task_scene_state(self) -> None:
        if self.task_name in _d20_stages.TASK_NAMES:
            # The shared restore journals these exact tensors for the origin
            # drift gate. Refresh the sampled detent before that transaction;
            # writing the new angle afterwards leaves the old cached detent
            # in the journal and falsely reports up to pi radians of drift.
            targets = {_d20.DOOR_JOINT: 0.0,
                       _d20.POWER_JOINT: self._d20_reset_power_angle,
                       _d20.TIMER_JOINT: 0.0,
                       _d20.TONGS_JOINT: _d20.TONGS_OPEN_RAD}
            for key, (view, index, _initial) in self._task_dof_views.items():
                target = next((v for name, v in targets.items() if name in key), None)
                if target is None:
                    continue
                for _path, cached_view, _ids, _pose, _velocity, positions, velocities in self._task_reset_articulation_states:
                    if cached_view is view:
                        positions[:, index] = target
                        velocities[:, index] = 0.0
        super()._restore_task_scene_state()
        if self.task_name in _d20_stages.TASK_NAMES:
            self._set_task_dof_state(_d20.POWER_JOINT, self._d20_reset_power_angle, 0.0)
            # A previous episode may leave a countdown velocity or a closed
            # jaw target. Reset the actuators as well as their physical state
            # before the first physics frame, including randomize=False.
            for name, target in (
                (_d20.DOOR_JOINT, 0.0), (_d20.POWER_JOINT, self._d20_reset_power_angle),
                (_d20.TIMER_JOINT, 0.0), (_d20.TONGS_JOINT, _d20.TONGS_OPEN_RAD),
            ):
                self._d20_set_drive_target(name, position=target, velocity=0.0)

    def _d20_initialize_tongs_open_state(self) -> bool:
        """Force live and cached reset DOF state to the open spring angle."""
        if self.task_name not in _d20_stages.TASK_NAMES or self._torch is None:
            return False
        resolved = self._d20_tongs_dof_entry()
        if resolved is None:
            self._add_task_state_warning(
                "Could not initialize D20 tongs open: DOF view is unavailable"
            )
            return False
        dof_key, dof_state = resolved
        view, dof_index, _initial = dof_state
        try:
            positions = view.get_dof_positions().clone()
            velocities = view.get_dof_velocities().clone()
            positions[:, dof_index] = _d20.TONGS_OPEN_RAD
            velocities[:, dof_index] = 0.0
            indices = self._torch.arange(
                int(view.count), dtype=self._torch.long, device=self._isaac_device
            )
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(velocities, indices)
            for (
                _path, cached_view, _indices, _root_pose, _root_velocity,
                cached_positions, cached_velocities,
            ) in self._task_reset_articulation_states:
                if cached_view is not view:
                    continue
                cached_positions[:, dof_index] = _d20.TONGS_OPEN_RAD
                cached_velocities[:, dof_index] = 0.0
            self._task_dof_views[dof_key] = (view, dof_index, _d20.TONGS_OPEN_RAD)
            print(
                "[HandsOn/IsaacLab] D20 tongs spring initialized open at "
                f"{_d20.TONGS_OPEN_RAD:.3f} rad (dof={dof_key})",
                flush=True,
            )
            return True
        except Exception as exc:
            self._add_task_state_warning(f"Could not initialize D20 tongs open: {exc}")
            return False

    def _d20_set_drive_target(self, name: str, *, position=None, velocity=None) -> bool:
        """GPU tensor drive targets; USD runtime writes are illegal with Direct GPU API."""
        entry = self._task_dof_views.get(name)
        if entry is None or self._torch is None:
            return False
        view, index, _initial = entry
        indices = self._torch.arange(int(view.count), dtype=self._torch.long, device=self._isaac_device)
        if position is not None:
            targets = view.get_dof_position_targets().clone()
            targets[:, index] = float(position)
            view.set_dof_position_targets(targets, indices)
        if velocity is not None:
            targets = view.get_dof_velocity_targets().clone()
            targets[:, index] = float(velocity)
            view.set_dof_velocity_targets(targets, indices)
        return True

    def _d20_set_tongs_joint_target(
        self, target_angle: float, *, max_rate: float = _d20.TONGS_JOINT_MAX_RATE
    ) -> bool:
        """The glove commands a physical spring drive, not a joint teleport."""
        current = float(self._joint(_d20.TONGS_JOINT)[0])
        max_step = max_rate * float(self._control_dt)
        target = current + float(np.clip(target_angle-current, -max_step, max_step))
        return self._d20_set_drive_target(_d20.TONGS_JOINT, position=float(np.clip(target, 0.0, _d20.TONGS_MAX_RAD)))

    def _d20_grip_command(self, side: str) -> float:
        raw = getattr(self, f"_c_assist_raw_{side}_hand", self._right_hand if side == "right" else self._left_hand)
        return _d20_grip.hand_grip_command(raw)

    def _d20_tool_contact_link_count(self, side: str) -> int:
        view = self._d20_tool_contact_views.get(side)
        if view is None:
            return 0
        forces = view.get_contact_force_matrix(float(self._sim_dt)).detach().cpu().numpy()
        return int(np.count_nonzero(np.max(np.linalg.norm(forces, axis=-1), axis=0) >= _d20_grip.CONTACT_FORCE))

    def _d20_update_tongs_control(self) -> None:
        """Acquire on closed-hand contact; squeeze and support with finite forces.

        Both jaw contacts are required before food grip acquisition. Opening
        the hand releases it; proximity alone cannot attach either object.
        """
        if self.task_name not in _d20_stages.TASK_NAMES or self._sim is None:
            return
        tongs_pose = self._physics_body_pose(_d20.TONGS)
        if tongs_pose is None:
            tongs_pose = self._poses.get(_d20.TONGS)
        if tongs_pose is None or not np.all(np.isfinite(tongs_pose)):
            return
        grip = self._d20_tool_grip
        holding = grip.side
        if holding and self._d20_grip_command(holding) <= _d20_grip.RELEASE_COMMAND:
            grip.reset(); holding = ""
        if not holding:
            for index, side in enumerate(("right", "left")):
                # A real closed hand must touch at least two link surfaces.
                contact_links = self._d20_tool_contact_link_count(side)
                candidate = (
                    self._d20_grip_command(side) >= _d20_grip.ACQUIRE_COMMAND
                    and contact_links >= 2
                    and _d20.hand_grips_tongs(self._a06_hand_points(side), tongs_pose)
                )
                grip.contacts[index] = grip.contacts[index] + 1 if candidate else 0
                if grip.contacts[index] >= _d20_grip.ACQUIRE_FRAMES:
                    palm = self._robot_body_pose_matrix(self._robots[side], self._ee_body_indices[side])
                    grip.offset = np.linalg.inv(palm) @ _pose_vector_to_matrix(tongs_pose)
                    grip.previous = None
                    grip.side = holding = side
                    break
        self._d20_tongs_holding_side = holding
        if holding:
            target = _d20.tongs_hand_target_angle(self._d20_grip_command(holding))
            self._d20_tongs_jaw_target = target
            self._d20_set_tongs_joint_target(target)
        else:
            self._d20_tongs_jaw_target = _d20.TONGS_OPEN_RAD
            self._d20_set_tongs_joint_target(_d20.TONGS_OPEN_RAD)
        jaw_angle = float(self._joint(_d20.TONGS_JOINT)[0])
        hot_dog_pose = self._physics_body_pose(_d20.HOT_DOG)
        if hot_dog_pose is None:
            hot_dog_pose = self._poses.get(_d20.HOT_DOG)
        if hot_dog_pose is None or not np.all(np.isfinite(hot_dog_pose)):
            return
        closure = _d20.tongs_hand_close_fraction(self._d20_grip_command(holding)) if holding else 0.0
        if self._d20_tongs_latched:
            if not holding or closure < .35:
                self._d20_tongs_latched = False
                self._d20_tongs_latch_offset = None
                self._d20_tongs_latch_last_target = None
        elif (
            holding
            and closure >= .65
            and self._d20_payload_between_contacting_jaws()
            and _d20.hot_dog_between_tongs_pads(hot_dog_pose, tongs_pose)
        ):
            offset = _invert_pose_matrix(
                _pose_vector_to_matrix(np.asarray(tongs_pose, dtype=np.float64))
            ) @ _pose_vector_to_matrix(np.asarray(hot_dog_pose, dtype=np.float64))
            self._d20_tongs_latch_offset = offset
            self._d20_tongs_latch_last_target = None
            self._d20_tongs_latched = True

    def _d20_update_tongs_latch(self) -> None:
        if (
            self.task_name not in _d20_stages.TASK_NAMES
            or not self._d20_tongs_latched
            or self._sim is None
            or self._d20_tongs_latch_offset is None
        ):
            return
        tongs_pose = self._physics_body_pose(_d20.TONGS)
        if tongs_pose is None or not np.all(np.isfinite(tongs_pose)):
            return
        target = (
            _pose_vector_to_matrix(np.asarray(tongs_pose, dtype=np.float64))
            @ np.asarray(self._d20_tongs_latch_offset, dtype=np.float64)
        )
        self._d20_tool_grip.payload_wrench = self._d20_apply_grip_wrench(
            _d20.HOT_DOG, target, self._d20_tongs_latch_last_target, mass=.12,
        )
        self._d20_tongs_latch_last_target = target.copy()

    def _d20_payload_between_contacting_jaws(self) -> bool:
        view = self._d20_payload_contact_view
        if view is None:
            return False
        force = view.get_contact_force_matrix(float(self._sim_dt)).detach().cpu().numpy()
        contacts = force.reshape(-1, 3)
        return bool(contacts.shape[0] == 2 and np.isfinite(contacts).all()
                    and np.all(np.linalg.norm(contacts, axis=1) >= .04))

    def _d20_payload_clamped(self) -> bool:
        """Actual two-jaw contact, independent of the optional force assist."""
        if self._sim is None or not self._d20_payload_between_contacting_jaws():
            return False
        food = self._physics_body_pose(_d20.HOT_DOG)
        tongs = self._physics_body_pose(_d20.TONGS)
        return bool(food is not None and tongs is not None
                    and np.isfinite(food).all() and np.isfinite(tongs).all()
                    and _d20.hot_dog_between_tongs_pads(food, tongs))

    def _d20_observe_stage_tool_use(self) -> None:
        progress = self._d20_stage
        if progress.stage not in (1, 2, 3):
            return
        if (progress.stage in (1, 2) and progress.grasp_seen) or (progress.stage == 3 and progress.retrieved_with_tongs):
            return
        # A valid latch is already evidence that both jaws closed on the
        # payload.  The force tensor can briefly dip below its reporting
        # threshold during the handoff, so do not lose the grasp event merely
        # because that one substep has no qualifying contact sample.
        if not self._d20_payload_clamped() and not self._d20_tongs_latched:
            return
        tongs = self._physics_body_pose(_d20.TONGS)
        # Physical hand/tool contact can carry a clamped payload without the
        # assist's command threshold/ownership latch ever being acquired.
        held = bool(self._d20_tongs_latched or self._d20_tool_grip.side) or any(
            self._d20_tool_contact_link_count(side) >= 2
            and _d20.hand_grips_tongs(self._a06_hand_points(side), tongs)
            for side in ("right", "left")
        )
        if held:
            reading = self._d20_reading()
            # The ordinary pose cache refreshes at control cadence. Evaluate
            # the substep crossing with this step's physical payload pose.
            reading.poses[_d20.HOT_DOG] = self._physics_body_pose(_d20.HOT_DOG)
            reading.poses[_d20.TONGS] = tongs
            metrics = self._d20.update(reading, update=False)
            progress.record_tongs_grasp(inside=metrics["d20_hot_dog_inside"],
                                       cooked=metrics["d20_cook_ok"], stopped=metrics["d20_stop_ok"])

    def _d20_apply_tool_grip(self) -> None:
        grip = self._d20_tool_grip
        grip.tool_wrench.fill(0); grip.payload_wrench.fill(0)
        if not grip.side or grip.offset is None:
            return
        current = self._physics_body_pose(_d20.TONGS)
        palm = self._robot_body_pose_matrix(self._robots[grip.side], self._ee_body_indices[grip.side])
        target = palm @ grip.offset
        if current is None or np.linalg.norm(target[:3, 3]-current[:3]) > _d20_grip.BREAK_DISTANCE:
            grip.reset()
            self._d20_tongs_latched = False
            return
        grip.tool_wrench[:] = self._d20_apply_grip_wrench(_d20.TONGS, target, grip.previous, mass=.165)
        grip.previous = target.copy()

    def _d20_apply_grip_wrench(self, name, target, previous, *, mass):
        pose = self._physics_body_pose(name)
        velocity = self._physics_body_velocity(name)
        if pose is None or velocity is None:
            return np.zeros(6)
        wrench = _d20_grip.grip_wrench(_pose_vector_to_matrix(pose), velocity, target, previous,
            dt=self._sim_dt, mass=mass, payload=name == _d20.HOT_DOG)
        articulation = self._task_articulation_views.get(name)
        if articulation is not None:
            view, indices = articulation
            shape = (int(view.count), int(view.max_links), 3)
            forces = self._torch.zeros(shape, device=self._isaac_device, dtype=self._torch.float32)
            torques = self._torch.zeros_like(forces)
            forces[0, 0] = self._torch.as_tensor(wrench[:3], device=self._isaac_device, dtype=forces.dtype)
            torques[0, 0] = self._torch.as_tensor(wrench[3:], device=self._isaac_device, dtype=forces.dtype)
        else:
            view = self._task_body_views[name]
            indices = self._torch.arange(int(view.count), device=self._isaac_device, dtype=self._torch.long)
            forces = self._torch.zeros((int(view.count), 3), device=self._isaac_device, dtype=self._torch.float32)
            torques = self._torch.zeros_like(forces)
            forces[0] = self._torch.as_tensor(wrench[:3], device=self._isaac_device, dtype=forces.dtype)
            torques[0] = self._torch.as_tensor(wrench[3:], device=self._isaac_device, dtype=forces.dtype)
        view.apply_forces_and_torques_at_position(forces, torques, None, indices, True)
        return wrench

    def _d20_randomize_layout(self) -> list[str]:
        """Seeded XY/yaw jitter for every free D20 body with reach checks."""
        from .d20_selected_assets import manifest, sample_layout
        placed: list[str] = []
        points = (*self._dlong_reset_hand_points("right"), *self._dlong_reset_hand_points("left"))
        chosen_xy, chosen_yaw = sample_layout(
            self._random, hand_points=points,
            reach=lambda xy: min(_dlong_required_reach_margins(_D20, xy).values()) >= _DLONG_MIN_REACH_MARGIN,
        )
        _require_dlong_required_reach(_D20, chosen_xy)
        for name in _d20.RESET_XY:
            # The shared helper applies yaw relative to its cached pose.
            # Cache the authored D20 base, not a body already tipped by the
            # first construction frame, and apply only the sampled delta.
            data = manifest()['reset'][name]
            prim = self._find_rigid_body_prim(name)
            path = str(prim.GetPath())
            rigid, articulations = self._reset_state_maps()
            entry = self._reset_transform_state_for_path(path, rigid, articulations)
            if entry is None:
                raise RuntimeError(f'D20 reset body unavailable: {name}')
            _kind, _view, _indices, base_pose, _velocity = entry
            # Low-level PhysX transform tensors use xyzw on Sim 5.1.
            base_pose[:, :3] = self._torch.as_tensor([*data['xy'], data['z']], device=base_pose.device, dtype=base_pose.dtype)
            base_pose[:, 3:] = self._torch.as_tensor(data['quat_xyzw'], device=base_pose.device, dtype=base_pose.dtype)
            if self._write_reset_pose(
                name, xy=chosen_xy[name], z=data['z'],
                yaw_degrees=chosen_yaw[name] - data['yaw_degrees'],
            ):
                placed.append(name)
        self._set_task_dof_state(_d20.DOOR_JOINT, 0.0, 0.0)
        self._set_task_dof_state(_d20.POWER_JOINT, self._d20_reset_power_angle, 0.0)
        self._set_task_dof_state(_d20.TIMER_JOINT, 0.0, 0.0)
        self._set_task_dof_state(_d20.TONGS_JOINT, _d20.TONGS_OPEN_RAD, 0.0)
        self._d20_tongs_latched = False
        self._d20_tongs_latch_offset = None
        self._d20_tongs_latch_last_target = None
        return placed

    def _d20_reading(self) -> "_d20.D20Reading":
        names = (*_d20.TASK_BODIES, *_d20.EXTRA_POSES)
        poses = {name: np.asarray(self._pose(name), dtype=np.float64) for name in names}
        velocities = {
            name: self._physics_body_velocity(name) for name in _d20.FREE_BODIES
        }
        hand_points = {
            side: tuple(self._a06_hand_points(side)) for side in ("left", "right")
        }
        if self._sim is None:
            contact: dict[str, tuple[float, bool]] = {}
        else:
            contact = {
                side: self._d20_hand_hot_dog_contact(side)
                for side in ("left", "right")
            }
        return _d20.D20Reading(
            dt=float(self._control_dt),
            elapsed_seconds=float(self.env_step) * float(self._control_dt),
            table_top_z=float(self._current_table_top_z()),
            door_angle=float(self._joint_delta(_d20.DOOR_JOINT)),
            timer_angle=float(self._joint(_d20.TIMER_JOINT)[0]),
            power_angle=float(self._joint(_d20.POWER_JOINT)[0]),
            tongs_angle=float(self._joint(_d20.TONGS_JOINT)[0]),
            poses=poses,
            velocities=velocities,
            hand_points=hand_points,
            hand_hot_dog_contact=contact,
            hand_object_contact={
                side: self._d20_release_frame_contact.get(side, self._d20_hand_object_contact(side))
                for side in ("left", "right")
            } if self._d20_stage.stage == 4 else {},
            timer_two_finger_hold=bool(
                self._d20_knob_hand_hold(_d20.TIMER_KNOB)
                or self._d20_knob_hand_hold(_d20.POWER_KNOB)
            ),
            tongs_latched=bool(self._d20_tongs_latched),
            tongs_holding_side=str(self._d20_tongs_holding_side),
        )

    def _d20_metrics(self, *, update: bool) -> dict[str, Any]:
        reading = self._d20_reading()
        if self._d20_heating_time_scale != 1.0:
            from dataclasses import replace
            reading = replace(reading, dt=reading.dt * self._d20_heating_time_scale)
        metrics = self._d20.update(reading, update=update)
        if self._d20_stage.stage:
            metrics["d20_tongs_payload_contact"] = self._d20_payload_clamped()
        metrics = self._d20_stage.update(reading, metrics, update=update)
        metrics["d20_tongs_jaw_target"] = float(self._d20_tongs_jaw_target)
        return metrics

    def _d20_update_visuals(self) -> None:
        # This mechanical microwave has no invented external status light.
        # Cook status is reported in the collection HUD/observations.
        pass

    def _d20_update_knob_drives(self) -> None:
        if self.task_name not in _d20_stages.TASK_NAMES:
            return
        drives = getattr(self, "_d20_knob_drives", {})
        power = float(self._joint(_d20.POWER_JOINT)[0])
        timer = float(self._joint(_d20.TIMER_JOINT)[0])
        if _d20.POWER_JOINT in drives:
            detent = _d20.POWER_ANGLES[_d20.power_index(power)]
            self._d20_set_drive_target(_d20.POWER_JOINT, position=detent)
        if _d20.TIMER_JOINT in drives:
            closed = abs(float(self._joint_delta(_d20.DOOR_JOINT))) <= _d20.DOOR_CLOSED_ABS
            held = bool(
                self._d20_knob_hand_hold(_d20.TIMER_KNOB)
                or self._d20_knob_hand_hold(_d20.POWER_KNOB)
            )
            velocity = (
                _d20.timer_return_velocity(timer) * self._d20_heating_time_scale
                if closed and not self._d20.state.burnt and not held
                else 0.0
            )
            self._d20_set_drive_target(_d20.TIMER_JOINT, velocity=velocity)

    def _d20_observation_state(self, metrics: Mapping[str, Any]) -> dict[str, np.ndarray]:
        side_code = {"": 0.0, "right": 1.0, "left": 2.0}.get(
            str(metrics.get("d20_tongs_holding_side", "")), 0.0
        )
        return {
            "hot_dog_pose": self._pose(_d20.HOT_DOG).copy(),
            "tongs_pose": self._pose(_d20.TONGS).copy(),
            "microwave_pose": self._pose(_d20.MICROWAVE).copy(),
            "tableware_poses": self._poses_array(_d20.TABLEWARE),
            "tray_pose": self._pose(_d20.SERVING_TRAY).copy(),
            "microwave_state": self._progress(
                (
                    metrics["d20_door_angle"],
                    metrics["d20_timer_angle"],
                    metrics["d20_running"],
                    metrics["d20_cook_progress"],
                    metrics["d20_light_on"],
                    metrics["d20_burnt"],
                    metrics["d20_hot_dog_inside"],
                    metrics["d20_door_closed"],
                    metrics["d20_power_angle"],
                    metrics["d20_timer_minutes"],
                ),
                10,
            ),
            "microwave_controls": self._progress((metrics["d20_power_angle"], metrics["d20_timer_angle"], metrics["d20_power_index"], metrics["d20_timer_minutes"]), 4),
            "d20_hidden_state": self._d20.hidden_observation(),
            "d20_stage_state": self._d20_stage.observation(),
            "d20_tool_grip_state": self._d20_tool_grip.observation(),
            "d20_assist_state": np.concatenate((np.asarray([self._d20_tongs_latched, side_code, self._d20_tongs_jaw_target, self._d20_tongs_latch_offset is not None, self._d20_tongs_latch_last_target is not None]), np.asarray(self._d20_tongs_latch_offset if self._d20_tongs_latch_offset is not None else np.zeros((4,4))).ravel(), np.asarray(self._d20_tongs_latch_last_target if self._d20_tongs_latch_last_target is not None else np.zeros((4,4))).ravel())).astype(np.float32),
            "tongs_state": self._progress(
                (
                    metrics["d20_tongs_angle"],
                    metrics.get("d20_tongs_jaw_target", _d20.TONGS_OPEN_RAD),
                    side_code,
                    metrics["d20_tongs_latched"],
                    metrics["d20_heated_touch_count"],
                    metrics["d20_hand_in_cavity_violation"],
                ),
                6,
            ),
            "workflow_progress": self._progress(
                (
                    metrics["d20_loaded_seen"],
                    metrics["d20_start_seen"],
                    metrics["d20_light_seen"],
                    metrics["d20_stop_ok"],
                    metrics["d20_removed_after_cook"],
                    metrics["d20_plated_seen"],
                    metrics["d20_fork_ok"],
                    metrics["d20_mustard_ok"],
                    metrics["d20_mug_ok"],
                    metrics["d20_microwave_off_closed"],
                    min(1.0, metrics["d20_release_seconds"] / _d20.RELEASE_HOLD_SECONDS),
                    metrics["d20_score"] / 100.0,
                ),
                12,
            ),
        }

    def _d20_condition_rows(self, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self._d20_stage.stage in (1, 2, 3):
            labels = _d20_stages.CONDITIONS[self._d20_stage.stage - 1]
            if self._d20_stage.stage == 3:
                labels = (*labels[:-1], _d20_stages.retrieval_release_status(metrics))
            return [self._condition_row(f"d20_stage_{i}", label, bool(value))
                    for i, (label, value) in enumerate(zip(labels, metrics["d20_stage_rows"]))]
        labels = list(_d20_stages.CONDITIONS[3] if self._d20_stage.stage == 4
                      else _d20.SUCCESS_CONDITION_LABELS)
        if self._d20_stage.stage == 4:
            for index, name in ((3, _d20.FORK), (4, _d20.MUSTARD), (5, _d20.MUG)):
                labels[index] = _d20.tableware_position_text(name, metrics, labels[index])
        return [
            self._condition_row(
                "d20_load",
                labels[0],
                metrics["d20_loaded_seen"] and metrics["d20_start_seen"],
            ),
            self._condition_row("d20_cook", labels[1], metrics["d20_stop_ok"] and metrics["d20_cook_ok"]),
            self._condition_row(
                "d20_plate",
                labels[2],
                metrics["d20_removed_after_cook"] and metrics["d20_plated_now"],
            ),
            self._condition_row("d20_fork", labels[3], metrics["d20_fork_ok"]),
            self._condition_row("d20_mustard", labels[4], metrics["d20_mustard_ok"]),
            self._condition_row("d20_mug", labels[5], metrics["d20_mug_ok"]),
            self._condition_row("d20_off_closed", labels[6], metrics["d20_microwave_off_closed"]),
            self._condition_row("d20_order", labels[7], metrics["d20_ordered_ok"]),
            self._condition_row(
                "d20_safety",
                labels[8],
                metrics["d20_no_heated_touch"]
                and not metrics["d20_hand_in_cavity_violation"]
                and not metrics["d20_dropped"]
                and not metrics["d20_burnt"],
            ),
            self._condition_row(
                "d20_terminal",
                labels[9] + f" (Held: {float(metrics['d20_release_seconds']):.1f}/0.5 s)",
                not metrics["d20_dropped"]
                and not metrics["d20_burnt"]
                and metrics["d20_release_complete"],
            ),
        ]

    def _metrics(self, *, update: bool) -> dict[str, Any]:
        dispatch = {
            _D06: self._d06_metrics,
            _D07: self._d07_metrics,
            _D08: self._d08_metrics,
            _D09: self._d09_metrics,
            _D10: self._d10_metrics,
            _D11: self._d11_metrics,
            _D12: self._d12_metrics,
            _D13: self._d13_metrics,
            _D14: self._d14_metrics,
            _D15: self._d15_metrics,
            _D16: self._d16_metrics,
            _D17: self._d17_metrics,
            _D18: self._d18_metrics,
            _D19: self._d19_metrics,
            _D20: self._d20_metrics,
        }
        return dispatch[_d20_stages.family_name(_d08_variant.family_name(self.task_name))](update=update)

    def _success_steps_required(self) -> int:
        # D06 and D08 perform their physical release debounce internally.  Do
        # not append the generic hold after the requested 0.5 second interval.
        if self.task_name in {_D06, _D08, _d08_variant.TASK_NAME, *_d20_stages.TASK_NAMES}:
            return 1
        return super()._success_steps_required()

    def _compute_failure(self) -> bool:
        if self.task_name in _d20_stages.TASK_NAMES:
            metrics = self._long_metrics or self._metrics(update=False)
            # Arm terminal failures only after the origin settle window so
            # one transient first-frame contact cannot end the episode.
            return bool(metrics.get("d20_failed", False)) and int(self.env_step) >= 15
        return super()._compute_failure()

    def _compute_success_now(self) -> bool:
        self._long_metrics = self._metrics(update=True)
        if self.task_name == _D06:
            border_colors: dict[str, tuple[float, float, float]] = {}
            for basket, border_geoms in _D06_BASKET_BORDER_GEOMS.items():
                color = (
                    (0.88, 0.08, 0.08)
                    if basket in self._long_metrics["d06_wrong_baskets"]
                    else (0.10, 0.72, 0.24)
                )
                border_colors.update(
                    {geom_name: color for geom_name in border_geoms}
                )
            self._set_task_visual_colors(
                border_colors
            )
        elif self.task_name == _D07:
            self._set_task_visual_colors(
                {
                    "d07_mouse_goal": (
                        (0.10, 0.78, 0.24)
                        if self._long_metrics["d07_mouse_on_pad"]
                        else (0.78, 0.08, 0.08)
                    ),
                    "d07_keyboard_goal": (
                        (0.10, 0.78, 0.24)
                        if self._long_metrics["d07_keyboard_in_frame"]
                        else (0.78, 0.08, 0.08)
                    ),
                    "d07_figurine_goal": (
                        (0.10, 0.78, 0.24)
                        if self._long_metrics["d07_figurine_on_stand"]
                        else (0.78, 0.08, 0.08)
                    ),
                    "d07_alarm_goal": (
                        (0.10, 0.78, 0.24)
                        if self._long_metrics["d07_alarm_in_drawer"]
                        else (0.78, 0.08, 0.08)
                    ),
                },
                opacity=0.58,
            )
        elif self.task_name == _D10:
            colors = {
                f"{name}_slot": (
                    (0.10, 0.78, 0.24)
                    if self._long_metrics["d10_slot_ok"][name]
                    else (0.78, 0.08, 0.08)
                )
                for name in _D10_DISHES
            }
            self._set_task_visual_colors(colors)
        elif self.task_name == _d08_variant.TASK_NAME:
            progress = 1.0 if self._d08_cycle_complete else min(1.0, self._timers.get("d08_cycle", 0.0) / _d08_variant.CYCLE_SECONDS)
            heating = self._d08_cycle_started and not self._d08_release_started and not self._d08_cycle_complete
            key = (int(progress * 4 + 1e-6), heating)
            if self._stage is not None and key != self._d08_2_visual_key:
                _d08_variant.set_appearance(self._stage, self._d08_2_visual_paths, progress, heating)
                self._d08_2_visual_key = key
        elif self.task_name in _d08_variant.TOAST_TASKS:
            colors = {
                "d08_toast_timer_light": (
                    (0.10, 0.82, 0.24)
                    if self._d08_cycle_complete
                    else (0.96, 0.64, 0.06)
                    if self._d08_cycle_started
                    else (0.32, 0.34, 0.35)
                )
            }
            if self._d08_cycle_complete:
                colors.update(
                    {
                        "d08_bread_0_visual": (0.58, 0.30, 0.10),
                        "d08_bread_0_visual_crown": (0.58, 0.30, 0.10),
                        "d08_bread_0_face": (0.76, 0.45, 0.18),
                        "d08_bread_0_face_crown": (0.76, 0.45, 0.18),
                        "d08_bread_1_visual": (0.58, 0.30, 0.10),
                        "d08_bread_1_visual_crown": (0.58, 0.30, 0.10),
                        "d08_bread_1_face": (0.76, 0.45, 0.18),
                        "d08_bread_1_face_crown": (0.76, 0.45, 0.18),
                    }
                )
            self._set_task_visual_colors(colors)
        elif self.task_name == _D09:
            self._set_d09_water_stream_visible(
                self._long_metrics["d09_water_emitting"]
            )
        elif self.task_name == _D11:
            colors = {
                f"d11_pan_coverage_{index}": (
                    (0.08, 0.68, 0.96)
                    if seconds >= 0.25
                    else (0.28, 0.40, 0.48)
                )
                for index, seconds in enumerate(self._d11_coverage_seconds)
            }
            self._set_task_visual_colors(colors, opacity=0.58)
            self._set_task_visual_colors(
                {"d11_water_stream": (0.08, 0.55, 0.98)},
                opacity=0.62 if self._long_metrics["d11_valve_open"] else 0.03,
            )
        elif self.task_name == _D12:
            knife_ok = bool(
                self._long_metrics["d12_knife_safe"]
                and self._long_metrics["d12_blade_outward"]
            )
            self._set_task_visual_colors(
                {
                    "d12_board_goal_outline": (
                        (0.10, 0.78, 0.24)
                        if self._long_metrics["d12_board_aligned"]
                        else (0.78, 0.08, 0.08)
                    ),
                    "d12_board_center": (
                        (0.10, 0.78, 0.24)
                        if self._long_metrics["d12_meat_centered"]
                        else (0.78, 0.08, 0.08)
                    ),
                    "d12_knife_safe_zone": (
                        (0.10, 0.78, 0.24) if knife_ok else (0.78, 0.08, 0.08)
                    )
                },
                opacity=0.58,
            )
        elif self.task_name == _D13:
            self._set_task_visual_colors(
                {
                    "d13_tray_left_zone": (0.10, 0.78, 0.24) if self._long_metrics["d13_kettle_left"] else (0.78, 0.08, 0.08),
                    "d13_tray_right_zone": (0.10, 0.78, 0.24) if self._long_metrics["d13_mug_right"] else (0.78, 0.08, 0.08),
                    "d13_tray_tea_zone": (0.10, 0.78, 0.24) if self._long_metrics["d13_tea_bag_on_tray"] else (0.78, 0.08, 0.08),
                    "d13_service_outline": (0.10, 0.78, 0.24) if self._long_metrics["d13_tray_in_service"] else (0.78, 0.08, 0.08),
                },
                opacity=0.58,
            )
        elif self.task_name == _D14:
            self._set_task_visual_colors(
                {"d14_water_stream": (0.08, 0.55, 0.98)},
                opacity=0.62 if self._long_metrics["d14_valve_open"] else 0.03,
            )
            self._set_task_visual_colors(
                {"d14_drying_outline": (0.10, 0.78, 0.24) if self._long_metrics["d14_colander_in_drying"] else (0.78, 0.08, 0.08)},
                opacity=0.58,
            )
            washed = np.asarray(self._long_metrics["d14_coverage_seconds"]) >= 0.25
            self._set_task_visual_colors(
                {
                    "d14_banana_visual": (0.72, 0.82, 0.24) if washed[0] else (0.94, 0.70, 0.10),
                    "d14_apple_visual": (0.88, 0.26, 0.20) if washed[1] else (0.78, 0.09, 0.08),
                    "d14_orange_visual": (0.98, 0.52, 0.12) if washed[2] else (0.90, 0.35, 0.08),
                }
            )
        elif self.task_name == _D15:
            self._set_task_visual_colors(
                {
                    "d15_goal_left_outline": (0.10, 0.78, 0.24) if self._long_metrics["d15_goals"][0] else (0.78, 0.08, 0.08),
                    "d15_goal_right_outline": (0.10, 0.78, 0.24) if self._long_metrics["d15_goals"][1] else (0.78, 0.08, 0.08),
                },
                opacity=0.58,
            )
        elif self.task_name == _D16:
            self._set_task_visual_colors(
                {
                    "d16_broom_goal_outline": (0.10, 0.78, 0.24) if self._long_metrics["d16_broom_returned"] else (0.78, 0.08, 0.08),
                    "d16_dustpan_goal_outline": (0.10, 0.78, 0.24) if self._long_metrics["d16_dustpan_returned"] else (0.78, 0.08, 0.08),
                },
                opacity=0.58,
            )
        elif self.task_name == _D17:
            self._set_task_visual_colors(
                {"d17_insertion_marker": (0.10, 0.78, 0.24) if self._long_metrics["d17_inserted"] else (0.78, 0.08, 0.08)},
                opacity=0.58,
            )
        elif self.task_name == _D18:
            colors = {
                f"d18_slot_{index}": (
                    (0.10, 0.78, 0.24)
                    if index in self._long_metrics["d18_assignments"]
                    else (0.78, 0.08, 0.08)
                )
                for index in range(4)
            }
            colors["d18_carton_goal_outline"] = (
                (0.10, 0.78, 0.24)
                if self._long_metrics["d18_carton_in_goal"]
                else (0.78, 0.08, 0.08)
            )
            self._set_task_visual_colors(colors, opacity=0.58)
            self._set_task_visual_colors(
                {
                    f"d18_egg_{index}_visual": (
                        (0.82, 0.08, 0.08)
                        if float(speed) > self._long_metrics["d18_impact_threshold"]
                        else (0.93, 0.94, 0.92)
                    )
                    for index, speed in enumerate(
                        self._long_metrics["d18_peak_impact_speed"]
                    )
                }
            )
        elif self.task_name in _d20_stages.TASK_NAMES:
            self._d20_update_visuals()
        return bool(self._long_metrics["success_now"])

    # ------------------------------------------------------------------
    # Recorder state and live UI
    # ------------------------------------------------------------------

    def _robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(
                np.float64
            ),
            np.concatenate([self._right_hand[:16], self._left_hand[:16]]).astype(
                np.float64
            ),
        )

    def _poses_array(self, names: tuple[str, ...]) -> np.ndarray:
        return np.stack([self._pose(name).copy() for name in names], axis=0)

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._long_metrics or self._metrics(update=False)
        tcp, gripper = self._robot_state()
        state: dict[str, np.ndarray] = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        if self.task_name == _D06:
            state.update(
                {
                    "target_poses": self._poses_array(_D06_TARGETS),
                    "basket_poses": self._poses_array(_D06_BASKETS),
                    "distractor_poses": self._poses_array(_D06_DISTRACTORS),
                    "classification_progress": self._progress(
                        (
                            metrics["d06_correct_count"] / 6.0,
                            metrics["d06_wrong_count"],
                            metrics["d06_handoff_complete"],
                            metrics["d06_baskets_upright"],
                            min(
                                1.0,
                                metrics["d06_release_seconds"]
                                / _D06_RELEASE_HOLD_SECONDS,
                            ),
                            metrics["d06_hands_released"],
                            metrics["d06_release_evidence_available"],
                        ),
                        12,
                    ),
                }
            )
        elif self.task_name == _D07:
            state.update(
                {
                    "object_poses": self._poses_array(_D07_OBJECTS),
                    "goal_poses": self._poses_array(
                        (
                            "d07_mouse_pad",
                            "d07_keyboard_frame",
                            "d07_stand",
                            "d07_drawer",
                        )
                    ),
                    "drawer_state": self._progress(
                        (
                            self._joint("d07_drawer_joint")[0],
                            self._joint_delta("d07_drawer_joint"),
                            metrics["d07_drawer_opened"],
                            metrics["d07_drawer_closed"],
                        ),
                        4,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d07_goal_count"] / 4.0,
                            metrics["d07_misc_inside"] / 3.0,
                            metrics["d07_mouse_on_pad"],
                            metrics["d07_keyboard_in_frame"],
                            metrics["d07_figurine_on_stand"],
                            metrics["d07_alarm_in_drawer"],
                            metrics["d07_no_falls"],
                            metrics["d07_objects_stable"],
                            metrics["d07_hands_released"],
                            metrics["d07_hand_pose_available"],
                        ),
                        12,
                    ),
                }
            )
        elif self.task_name in _d08_variant.TOAST_TASKS:
            state.update(
                {
                    "bread_poses": self._poses_array(_D08_BREAD),
                    "toaster_pose": self._pose("d08_toaster").copy(),
                    "plate_pose": self._pose("d08_plate").copy(),
                    "toaster_state": self._progress(
                        (
                            self._joint("d08_start_button_joint")[0],
                            self._joint_delta("d08_start_button_joint"),
                            metrics["d08_button_pressed"],
                            metrics["d08_cycle_seconds"],
                            metrics["d08_cycle_complete"],
                            metrics["d08_plate_count"],
                        ),
                        6,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d08_bread_inserted"] / 2.0,
                            metrics["d08_two_slots_seen"],
                            metrics["d08_cycle_started"],
                            metrics["d08_cycle_complete"],
                            metrics["d08_plate_count"] / 2.0,
                            metrics["d08_release_started"],
                            metrics["d08_lever_returned"],
                            metrics["d08_physical_pop_complete"],
                        ),
                        10,
                    ),
                }
            )
        elif self.task_name == _D09:
            state.update(
                {
                    "mug_pose": self._pose("d09_mug").copy(),
                    "cabinet_pose": self._pose("d09_cabinet").copy(),
                    "coffee_machine_pose": self._pose("d09_coffee_machine").copy(),
                    "tray_pose": self._pose("d09_tray").copy(),
                    "coffee_state": self._progress(
                        (
                            self._joint("d09_cabinet_door_joint")[0],
                            self._joint("d09_start_button_joint")[0],
                            metrics["d09_under_outlet"],
                            metrics["d09_button_pressed"],
                            metrics["d09_brew_seconds"],
                            metrics["d09_brew_complete"],
                            metrics["d09_on_tray"],
                            self._joint("d09_cabinet_right_door_joint")[0],
                        ),
                        8,
                    ),
                    "workflow_progress": self._progress(
                        tuple(
                            metrics[key]
                            for key in (
                                "d09_door_opened",
                                "d09_mug_removed",
                                "d09_door_closed",
                                "d09_under_outlet",
                                "d09_button_pressed",
                                "d09_brew_complete",
                                "d09_on_tray",
                            )
                        ),
                        10,
                    ),
                }
            )
        elif self.task_name == _D10:
            state.update(
                {
                    "dish_poses": self._poses_array(_D10_DISHES),
                    "dishwasher_pose": self._pose("d10_dishwasher").copy(),
                    "rack_pose": self._pose("d10_rack").copy(),
                    "dishwasher_state": self._progress(
                        (
                            self._joint("d10_door_joint")[0],
                            self._joint("d10_rack_joint")[0],
                            metrics["d10_door_opened"],
                            metrics["d10_rack_pulled"],
                            metrics["d10_slot_count"],
                            metrics["d10_rack_in"],
                            metrics["d10_door_closed"],
                        ),
                        8,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d10_door_opened"],
                            metrics["d10_rack_pulled"],
                            metrics["d10_slot_count"] / 3.0,
                            metrics["d10_rack_in"],
                            metrics["d10_door_closed"],
                        ),
                        10,
                    ),
                }
            )
        elif self.task_name == _D11:
            state.update(
                {
                    "pan_pose": self._pose("d11_pan").copy(),
                    "sponge_pose": self._pose("d11_sponge").copy(),
                    "sink_pose": self._pose("d11_sink").copy(),
                    "faucet_pose": self._pose("d11_faucet").copy(),
                    "faucet_state": self._progress(
                        (
                            self._joint("d11_faucet_valve_joint")[0],
                            self._joint("d11_faucet_swivel_joint")[0],
                            metrics["d11_valve_open"],
                            metrics["d11_bimanual_open"],
                            metrics["d11_pan_held"],
                            metrics["d11_pan_stable"],
                            metrics["d11_outlet_over_pan"],
                            metrics["d11_coverage_ok"],
                        ),
                        8,
                    ),
                    "coverage_progress": self._progress(
                        (
                            *metrics["d11_coverage_seconds"],
                            metrics["d11_pan_inside"],
                            metrics["d11_sponge_inside"],
                            metrics["d11_coverage_ok"],
                        ),
                        8,
                    ),
                }
            )
        elif self.task_name == _D12:
            state.update(
                {
                    "knife_pose": self._pose("d12_knife").copy(),
                    "cutting_board_pose": self._pose("d12_cutting_board").copy(),
                    "meat_pose": self._pose("d12_meat").copy(),
                    "plate_pose": self._pose("d12_plate").copy(),
                    "drawer_pose": self._pose("d12_drawer").copy(),
                    "board_goal_pose": self._pose("d12_board_goal").copy(),
                    "drawer_state": self._progress(
                        (
                            self._joint("d12_drawer_joint")[0],
                            self._joint_delta("d12_drawer_joint"),
                            metrics["d12_drawer_opened"],
                            metrics["d12_drawer_closed"],
                        ),
                        4,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d12_drawer_opened"],
                            metrics["d12_drawer_closed"],
                            metrics["d12_board_aligned"],
                            metrics["d12_meat_centered"],
                            metrics["d12_knife_safe"],
                            metrics["d12_blade_outward"],
                        ),
                        10,
                    ),
                }
            )
        elif self.task_name == _D13:
            state.update(
                {
                    "object_poses": self._poses_array(("d13_tray", *_D13_PAYLOADS)),
                    "fixture_poses": self._poses_array(("d13_cabinet", "d13_service_zone")),
                    "cabinet_state": self._progress(
                        (
                            self._joint("d13_cabinet_door_joint")[0],
                            self._joint_delta("d13_cabinet_door_joint"),
                            metrics["d13_cabinet_opened"],
                            metrics["d13_cabinet_closed"],
                        ), 4,
                    ),
                    "transport_state": self._progress(
                        (
                            metrics["d13_tilt_degrees"],
                            metrics["d13_peak_tilt_degrees"],
                            metrics["d13_transport_started"],
                            metrics["d13_bimanual_support"],
                            metrics["d13_tray_in_service"],
                            metrics["d13_payload_dropped"],
                            metrics["d13_stable"],
                        ), 8,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d13_kettle_left"], metrics["d13_cabinet_opened"],
                            metrics["d13_mug_right"], metrics["d13_tea_bag_on_tray"],
                            metrics["d13_cabinet_closed"], metrics["d13_bimanual_support"],
                            metrics["d13_tray_in_service"], not metrics["d13_payload_dropped"],
                        ), 12,
                    ),
                }
            )
        elif self.task_name == _D14:
            state.update(
                {
                    "fruit_poses": self._poses_array(_D14_FRUITS),
                    "colander_pose": self._pose("d14_colander").copy(),
                    "fixture_poses": self._poses_array(("d14_sink", "d14_faucet", "d14_drying_area")),
                    "faucet_state": self._progress(
                        (
                            self._joint("d14_faucet_valve_joint")[0],
                            self._joint("d14_faucet_swivel_joint")[0],
                            metrics["d14_valve_open"], metrics["d14_valve_closed"],
                            metrics["d14_washed_count"], metrics["d14_colander_in_drying"],
                        ), 6,
                    ),
                    "wash_progress": self._progress(
                        (*metrics["d14_coverage_seconds"], *metrics["d14_covered_now"]), 6
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d14_colander_in_sink"], metrics["d14_fruit_count"] / 3.0,
                            metrics["d14_faucet_opened"], metrics["d14_washed_count"] / 3.0,
                            metrics["d14_valve_closed"], metrics["d14_colander_in_drying"],
                            not metrics["d14_fruit_dropped"],
                        ), 12,
                    ),
                }
            )
        elif self.task_name == _D15:
            counts = np.asarray(metrics["d15_counts"], dtype=np.float64).reshape(-1)
            state.update(
                {
                    "food_poses": self._poses_array(_D15_FOODS),
                    "box_poses": self._poses_array(_D15_BOXES),
                    "lid_poses": self._poses_array(_D15_LIDS),
                    "goal_poses": self._poses_array(("d15_goal_left", "d15_goal_right")),
                    "packing_state": self._progress(
                        (*counts, metrics["d15_matching"], metrics["d15_exact_counts"],
                         *metrics["d15_lids_closed"], *metrics["d15_goals"]), 12,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d15_packed_count"] / 12.0,
                            metrics["d15_matching"], metrics["d15_exact_counts"],
                            *metrics["d15_lids_closed"], *metrics["d15_goals"],
                            metrics["d15_stable"], metrics["d15_hands_released"],
                        ), 12,
                    ),
                }
            )
        elif self.task_name == _D16:
            state.update(
                {
                    "block_poses": self._poses_array(_D16_BLOCKS),
                    "tool_poses": self._poses_array(_D16_TOOLS),
                    "fixture_poses": self._poses_array(("d16_bin", "d16_broom_goal", "d16_dustpan_goal")),
                    "sweeping_state": self._progress(
                        (
                            metrics["d16_handoff_complete"], metrics["d16_dustpan_count"] / 5.0,
                            metrics["d16_dustpan_seen_count"] / 5.0, metrics["d16_bin_count"] / 5.0,
                            metrics["d16_direct_pick"], metrics["d16_broom_returned"],
                            metrics["d16_dustpan_returned"], metrics["d16_stable"],
                        ), 10,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d16_handoff_complete"], metrics["d16_dustpan_seen_count"] / 5.0,
                            metrics["d16_bin_count"] / 5.0, not metrics["d16_direct_pick"],
                            metrics["d16_broom_returned"], metrics["d16_dustpan_returned"],
                        ), 12,
                    ),
                }
            )
        elif self.task_name == _D17:
            state.update(
                {
                    "headphones_pose": self._pose("d17_headphones").copy(),
                    "laptop_pose": self._pose("d17_laptop").copy(),
                    "fixture_poses": self._poses_array(("d17_headphone_stand", "d17_flat_stand", "d17_vertical_stand")),
                    "laptop_state": self._progress(
                        (
                            metrics["d17_hinge_delta"], metrics["d17_close_error_degrees"],
                            metrics["d17_laptop_closed"], metrics["d17_laptop_vertical"],
                            metrics["d17_insertion_depth"], metrics["d17_inserted"],
                            metrics["d17_headphones_hung"],
                        ), 8,
                    ),
                    "workflow_progress": self._progress(
                        (
                            metrics["d17_headphones_hung"], metrics["d17_laptop_closed"],
                            metrics["d17_laptop_vertical"], metrics["d17_insertion_depth"],
                            metrics["d17_inserted"], metrics["d17_hands_released"],
                        ), 10,
                    ),
                }
            )
        elif self.task_name == _D18:
            state.update(
                {
                    "egg_poses": self._poses_array(_D18_EGGS),
                    "carton_pose": self._pose("d18_carton").copy(),
                    "basket_pose": self._pose("d18_basket").copy(),
                    "carton_state": self._progress(
                        (
                            self._joint("d18_carton_lid_joint")[0],
                            self._joint_delta("d18_carton_lid_joint"),
                            metrics["d18_lid_opened"], metrics["d18_lid_closed"],
                            metrics["d18_occupied_slots"] / 4.0, metrics["d18_carton_in_goal"],
                        ), 6,
                    ),
                    "impact_state": np.asarray(metrics["d18_peak_impact_speed"], dtype=np.float64).copy(),
                    "workflow_progress": self._progress(
                        (
                            metrics["d18_lid_opened"], metrics["d18_occupied_slots"] / 4.0,
                            metrics["d18_intact"], metrics["d18_lid_closed"],
                            metrics["d18_carton_in_goal"], metrics["d18_hands_released"],
                        ), 10,
                    ),
                }
            )
        elif self.task_name in _d20_stages.TASK_NAMES:
            state.update(self._d20_observation_state(metrics))
        else:
            spatial_order = tuple(metrics["d19_spatial_order"])
            state.update(
                {
                    "block_poses": self._poses_array(_D19_BLOCKS),
                    "cover_poses": self._poses_array(_D19_COVERS),
                    "basket_pose": self._pose("d19_basket").copy(),
                    "memory_state": self._progress(
                        (
                            *(spatial_order.index(name) for name in _D19_COLOR_SEQUENCE),
                            metrics["d19_all_covered"], metrics["d19_cover_count"] / 3.0,
                            metrics["d19_sequence_errors"],
                        ), 6,
                    ),
                    "sequence_progress": self._progress(
                        (
                            len(metrics["d19_cover_order"]) / 3.0,
                            len(metrics["d19_revealed"]) / 3.0,
                            len(metrics["d19_basketed"]) / 3.0,
                            metrics["d19_basket_count"] / 3.0,
                            metrics["d19_sequence_errors"], metrics["d19_hands_released"],
                        ), 8,
                    ),
                }
            )
        observation: dict[str, Any] = {"state": state}
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _task_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "task_id": "D08-2" if self.task_name == _d08_variant.TASK_NAME else self.task_name[:3].upper(),
            "task_name": self.task_name,
            "success_counter": int(self._success_counter),
            "success_steps_required": int(self._success_steps_required()),
        }
        info.update(self._long_metrics)
        return info

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._long_metrics or self._metrics(update=False)
        if self.task_name == _D06:
            rows = [
                self._condition_row(
                    "d06_sorted",
                    "Six targets are in the correct category baskets",
                    metrics["d06_correct_count"] == 6,
                ),
                self._condition_row(
                    "d06_no_errors",
                    "No incorrect object is in any basket",
                    metrics["d06_wrong_count"] == 0,
                ),
                self._condition_row(
                    "d06_handoff",
                    "Far-right target completed a physical two-hand handoff",
                    metrics["d06_handoff_complete"],
                ),
                self._condition_row(
                    "d06_baskets",
                    "All three baskets remain upright",
                    metrics["d06_baskets_upright"],
                ),
                self._condition_row(
                    "d06_terminal",
                    "Neither hand touches any target for 0.5 seconds",
                    metrics["d06_hands_released"],
                ),
            ]
        elif self.task_name == _D07:
            rows = [
                self._condition_row(
                    "d07_goals",
                    "Mouse, keyboard, and figurine are in their goal regions",
                    metrics["d07_mouse_on_pad"]
                    and metrics["d07_keyboard_in_frame"]
                    and metrics["d07_figurine_on_stand"],
                ),
                self._condition_row(
                    "d07_misc",
                    "Alarm clock, pen, eraser, and stapler are inside the drawer",
                    metrics["d07_alarm_in_drawer"]
                    and metrics["d07_misc_inside"] == 3,
                ),
                self._condition_row(
                    "d07_drawer",
                    "Drawer was opened and is finally closed",
                    metrics["d07_drawer_opened"]
                    and metrics["d07_drawer_closed"],
                ),
                self._condition_row(
                    "d07_falls",
                    "No required object fell from the desk",
                    metrics["d07_no_falls"],
                ),
                self._condition_row(
                    "d07_terminal",
                    "All objects are stable and both hands are released",
                    metrics["d07_objects_stable"]
                    and metrics["d07_hands_released"],
                ),
            ]
        elif self.task_name in _d08_variant.TOAST_TASKS:
            rows = [
                self._condition_row(
                    "d08_slots",
                    "Both slices entered separate toaster slots",
                    metrics["d08_two_slots_seen"],
                ),
                self._condition_row(
                    "d08_lever",
                    "The toaster's original lever was pushed down only after both slices were inserted",
                    metrics["d08_button_triggered"],
                ),
                self._condition_row(
                    "d08_cycle",
                    "Toast cycle completed, the lever returned up, and both slices popped",
                    metrics["d08_cycle_complete"],
                ),
                self._condition_row(
                    "d08_plate",
                    "After popping, both slices left the toaster and were placed on the plate",
                    metrics["d08_removed_after_pop"]
                    and metrics["d08_plate_count"] == 2,
                ),
                self._condition_row(
                    "d08_stationary",
                    "Toaster did not move significantly",
                    metrics["d08_toaster_stationary"],
                ),
                self._condition_row(
                    "d08_terminal",
                    "Both slices stay stable on the plate with both hands away for 0.5 s",
                    metrics["d08_release_complete"],
                ),
            ]
        elif self.task_name == _D09:
            brew_seconds = min(
                _D09_BREW_SECONDS,
                float(metrics["d09_brew_seconds"]),
            )
            if (
                metrics["d09_brew_interrupted"]
                and not metrics["d09_brew_complete"]
            ):
                brew_label = "6. Brew interrupted: reset and keep the mug aligned"
            else:
                brew_label = (
                    "6. Keep mug aligned during brew: "
                    f"{brew_seconds:.1f}/{_D09_BREW_SECONDS:.1f} s"
                )
            rows = [
                self._condition_row(
                    "d09_door_opened",
                    "1. Open either cabinet door >= 0.65 rad",
                    metrics["d09_door_opened"],
                ),
                self._condition_row(
                    "d09_mug_removed",
                    "2. Remove mug after opening a cabinet door",
                    metrics["d09_mug_removed"],
                ),
                self._condition_row(
                    "d09_doors_closed",
                    "3. Close both cabinet doors <= 0.10 rad",
                    metrics["d09_door_closed"],
                ),
                self._condition_row(
                    "d09_position",
                    "4. Upright under outlet: XY <= 40 mm, Z +/- 30 mm",
                    self._event_flags.get("d09_positioned", False),
                ),
                self._condition_row(
                    "d09_button",
                    "5. Press bright-green start button >= 1.5 mm",
                    metrics["d09_brew_started"],
                ),
                self._condition_row(
                    "d09_brew",
                    brew_label,
                    metrics["d09_brew_complete"],
                ),
                self._condition_row(
                    "d09_tray",
                    "7. Place filled mug upright on serving tray",
                    metrics["d09_on_tray"] and self._upright("d09_mug"),
                ),
                self._condition_row(
                    "d09_terminal",
                    "8. Mug stable on tray; move both hands clear",
                    metrics["d09_on_tray"]
                    and self._stable("d09_mug")
                    and metrics["d09_hand_pose_available"]
                    and metrics["d09_hands_released"],
                ),
            ]
        elif self.task_name == _D10:
            rows = [
                self._condition_row(
                    "d10_slots",
                    "Mug, bowl, and plate are stable in assigned slots",
                    metrics["d10_slot_count"] == 3,
                ),
                self._condition_row(
                    "d10_rack",
                    "Upper rack was pulled out and fully pushed back",
                    metrics["d10_rack_pulled"] and metrics["d10_rack_in"],
                ),
                self._condition_row(
                    "d10_door",
                    "Dishwasher door is fully closed",
                    metrics["d10_door_closed"],
                ),
            ]
        elif self.task_name == _D11:
            rows = [
                self._condition_row(
                    "d11_items",
                    "Pan and sponge are inside the sink",
                    metrics["d11_pan_inside"] and metrics["d11_sponge_inside"],
                ),
                self._condition_row(
                    "d11_water",
                    "Opposite hands opened the faucet and stabilized the pan",
                    metrics["d11_bimanual_open"]
                    and metrics["d11_valve_open"]
                    and metrics["d11_pan_stable"],
                ),
                self._condition_row(
                    "d11_coverage",
                    "Water outlet covered all required pan regions",
                    metrics["d11_coverage_ok"],
                ),
                self._condition_row(
                    "d11_pan",
                    "Pan remains inside the sink",
                    metrics["d11_pan_inside"],
                ),
            ]
        elif self.task_name == _D12:
            rows = [
                self._condition_row(
                    "d12_drawer",
                    "Knife was retrieved and drawer is closed",
                    metrics["d12_drawer_opened"]
                    and metrics["d12_drawer_closed"],
                ),
                self._condition_row(
                    "d12_board",
                    "Cutting board is aligned and meat is centered",
                    metrics["d12_board_aligned"]
                    and metrics["d12_meat_centered"],
                ),
                self._condition_row(
                    "d12_safe",
                    "Knife is fully inside the safety zone",
                    metrics["d12_knife_safe"],
                ),
                self._condition_row(
                    "d12_blade",
                    "Knife blade points toward the prescribed outer side",
                    metrics["d12_blade_outward"],
                ),
            ]
        elif self.task_name == _D13:
            rows = [
                self._condition_row("d13_payloads", "Kettle, mug, and tea bag occupy their marked tray regions", metrics["d13_payload_count"] == 3),
                self._condition_row("d13_cabinet", "Cabinet was opened and is finally closed", metrics["d13_cabinet_opened"] and metrics["d13_cabinet_closed"]),
                self._condition_row("d13_transport", "Both hands supported the tray and it reached the service region", metrics["d13_bimanual_support"] and metrics["d13_tray_in_service"]),
                self._condition_row("d13_tilt", "Peak transport tilt is at most 15 degrees", metrics["d13_peak_tilt_degrees"] <= 15.0),
                self._condition_row("d13_payload_safe", "No payload fell and the tray is stable", not metrics["d13_payload_dropped"] and metrics["d13_stable"]),
            ]
        elif self.task_name == _D14:
            rows = [
                self._condition_row("d14_loaded", "All three fruits are inside the colander", metrics["d14_fruit_count"] == 3),
                self._condition_row("d14_washed", "The outlet covered all three fruits", metrics["d14_all_washed"]),
                self._condition_row("d14_water_off", "The faucet was used and is finally off", metrics["d14_faucet_opened"] and metrics["d14_valve_closed"]),
                self._condition_row("d14_drying", "The loaded colander is stable in the draining region", metrics["d14_colander_in_drying"]),
                self._condition_row("d14_no_drop", "No fruit fell during transport", not metrics["d14_fruit_dropped"]),
            ]
        elif self.task_name == _D15:
            rows = [
                self._condition_row("d15_match", "Both boxes contain identical class counts", metrics["d15_matching"]),
                self._condition_row("d15_counts", "Each box contains two items of each class", metrics["d15_exact_counts"]),
                self._condition_row("d15_lids", "Both lids are fully closed", all(metrics["d15_lids_closed"])),
                self._condition_row("d15_goals", "Boxes occupy their assigned left and right regions", all(metrics["d15_goals"])),
            ]
        elif self.task_name == _D16:
            rows = [
                self._condition_row("d16_handoff", "The broom completed the left-to-right physical handoff", metrics["d16_handoff_complete"]),
                self._condition_row("d16_swept", "All five blocks passed through the dustpan", metrics["d16_dustpan_seen_count"] == 5),
                self._condition_row("d16_bin", "All five blocks are inside the storage bin", metrics["d16_bin_count"] == 5),
                self._condition_row("d16_no_grasp", "No block was directly picked up", not metrics["d16_direct_pick"]),
                self._condition_row("d16_tools", "Broom and dustpan are back in their marked regions", metrics["d16_broom_returned"] and metrics["d16_dustpan_returned"]),
            ]
        elif self.task_name == _D17:
            rows = [
                self._condition_row("d17_headphones", "Headphones are stably hung on the stand", metrics["d17_headphones_hung"]),
                self._condition_row("d17_closed", "Laptop close error is at most five degrees", metrics["d17_laptop_closed"]),
                self._condition_row("d17_vertical", "Laptop is in a vertical pose", metrics["d17_laptop_vertical"]),
                self._condition_row("d17_inserted", "Laptop reaches the required insertion depth", metrics["d17_inserted"]),
            ]
        elif self.task_name == _D18:
            rows = [
                self._condition_row("d18_slots", "Four eggs occupy four independent slots", metrics["d18_occupied_slots"] == 4),
                self._condition_row("d18_impact", "Every conservative impact-speed proxy is below threshold", metrics["d18_intact"]),
                self._condition_row("d18_lid", "Carton was opened and is finally closed", metrics["d18_lid_opened"] and metrics["d18_lid_closed"]),
                self._condition_row("d18_goal", "Carton remains inside its target region", metrics["d18_carton_in_goal"]),
            ]
        elif self.task_name in _d20_stages.TASK_NAMES:
            rows = self._d20_condition_rows(metrics)
        else:
            rows = [
                self._condition_row("d19_cover", "All blocks were covered from left to right", metrics["d19_all_covered"]),
                self._condition_row("d19_reveal", "Covers were removed in red, green, blue order", tuple(metrics["d19_revealed"]) == _D19_COLOR_SEQUENCE),
                self._condition_row("d19_basket", "Blocks entered the basket in red, green, blue order", tuple(metrics["d19_basketed"]) == _D19_COLOR_SEQUENCE),
                self._condition_row("d19_errors", "No incorrect reveal or placement was recorded", metrics["d19_sequence_errors"] == 0),
            ]
        return self._append_success_hold_row(rows)

    def _success_condition_display_label(self, row: dict[str, Any]) -> str:
        # D09 and D20 have ordered operator checklists with threshold/progress
        # text.  The shared fallback intentionally turns unknown keys into
        # machine-like labels (for example, ``d09 door opened``), so use the
        # authored D09 text while preserving the established labels elsewhere.
        if str(row.get("key") or "").startswith(("d09_", "d20_")) and row.get("label"):
            return str(row["label"])
        return super()._success_condition_display_label(row)

    def _task_status_text(self) -> str:
        metrics = self._long_metrics
        if self.task_name == _D06:
            counts = metrics.get("d06_category_counts", {})
            return (
                f"Boxes {int(counts.get('d06_basket_box', 0))}/2"
                f"  |  Cans {int(counts.get('d06_basket_can', 0))}/2"
                f"  |  Dishes {int(counts.get('d06_basket_dish', 0))}/2"
                f"  |  Errors {int(metrics.get('d06_wrong_count', 0))}"
            )
        if self.task_name == _D07:
            return (
                f"Goal regions {int(metrics.get('d07_goal_count', 0))}/4"
                f"  |  Drawer items {int(metrics.get('d07_misc_inside', 0))}/3"
            )
        if self.task_name in _d08_variant.TOAST_TASKS:
            pop_reason = str(metrics.get("d08_pop_failure_reason", ""))
            pop_status = (
                f"  | Pop blocked: {pop_reason.replace('_', ' ')}"
                if pop_reason
                else ""
            )
            return (
                f"Toaster {int(metrics.get('d08_bread_inserted', 0))}/2"
                f"  |  Plate {int(metrics.get('d08_plate_count', 0))}/2"
                f"{pop_status}"
            )
        if self.task_name == _D09:
            if (
                metrics.get("d09_brew_interrupted", False)
                and not metrics.get("d09_brew_complete", False)
            ):
                return "Brew interrupted - reset task and keep mug aligned"
            return (
                f"Brew {min(_D09_BREW_SECONDS, float(metrics.get('d09_brew_seconds', 0.0))):.1f}/{_D09_BREW_SECONDS:.1f} s"
                f"  |  Tray {'done' if metrics.get('d09_on_tray', False) else 'pending'}"
            )
        if self.task_name == _D10:
            return (
                f"Rack slots {int(metrics.get('d10_slot_count', 0))}/3"
                f"  |  Rack {'in' if metrics.get('d10_rack_in', False) else 'out'}"
                f"  |  Door {'closed' if metrics.get('d10_door_closed', False) else 'open'}"
            )
        if self.task_name == _D11:
            covered = int(np.sum(self._d11_coverage_seconds >= 0.25))
            return f"Water coverage {covered}/4"
        if self.task_name == _D12:
            return (
            f"Board {'done' if metrics.get('d12_board_aligned', False) else 'pending'}"
            f"  |  Knife {'safe' if metrics.get('d12_knife_safe', False) else 'unsafe'}"
            )
        if self.task_name == _D13:
            return f"Tray payloads {int(metrics.get('d13_payload_count', 0))}/3  |  Peak tilt {float(metrics.get('d13_peak_tilt_degrees', 0.0)):.1f}°"
        if self.task_name == _D14:
            return f"Fruit {int(metrics.get('d14_fruit_count', 0))}/3  |  Washed {int(metrics.get('d14_washed_count', 0))}/3"
        if self.task_name == _D15:
            counts = np.asarray(metrics.get("d15_counts", np.zeros((2, 3))), dtype=int)
            return f"Left {counts[0].tolist()}  |  Right {counts[1].tolist()}"
        if self.task_name == _D16:
            return f"Dustpan seen {int(metrics.get('d16_dustpan_seen_count', 0))}/5  |  Bin {int(metrics.get('d16_bin_count', 0))}/5"
        if self.task_name == _D17:
            return f"Laptop close {float(metrics.get('d17_close_error_degrees', 90.0)):.1f}°  |  Insert {int(float(metrics.get('d17_insertion_depth', 0.0)) * 100)}%"
        if self.task_name == _D18:
            return f"Egg slots {int(metrics.get('d18_occupied_slots', 0))}/4  |  {'intact' if metrics.get('d18_intact', True) else 'cracked'}"
        if self.task_name in _d20_stages.TASK_NAMES:
            if getattr(self, "_d20_prelude_status", ""):
                return self._d20_prelude_status
            if self._d20_stage.stage:
                return _d20_stages.operator_text(self._d20_stage.stage, metrics)
            return _d20.operator_status_text(metrics)
        return f"Covered {int(metrics.get('d19_cover_count', 0))}/3  |  Basket {int(metrics.get('d19_basket_count', 0))}/3  |  Errors {int(metrics.get('d19_sequence_errors', 0))}"


__all__ = ["HandsOnDLongHorizonIsaacLabEnv"]
