"""Locked geometry and timing contract for YCB02.

All dimensions are metres and come from the unscaled Lightwheel YCB meshes.
This module deliberately contains no simulator-side object motion or assists;
it is the shared, testable source for reset layout and physical evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


TASK_NAME = "t27_place_largest_fitting_ball"

BALL_NAMES = (
    "d_marble",
    "golf_ball",
    "tennis_ball",
    "baseball",
)
CUP_NAMES = ("c_cups", "g_cups", "j_cups")

BALL_IDS = {name: index for index, name in enumerate(BALL_NAMES)}
CUP_IDS = {name: index for index, name in enumerate(CUP_NAMES)}

# Half of the measured collision-mesh diameter.  These values also define the
# authored table-supported root height; no scale is applied in MJCF or USD.
BALL_RADIUS_M = {
    "d_marble": 0.0108510,
    "golf_ball": 0.0216590,
    "tennis_ball": 0.0330205,
    "baseball": 0.0385000,
}
BALL_SUPPORT_LOCAL_MIN_Z_M = {
    name: -radius for name, radius in BALL_RADIUS_M.items()
}

# Exact lower/upper mesh bounds in the source upright orientation.  The cup
# meshes are essentially centred in Z, so the upper bound is the rim plane.
CUP_SUPPORT_LOCAL_MIN_Z_M = {
    "c_cups": -0.0320000,
    "g_cups": -0.0370385,
    "j_cups": -0.0391450,
}
CUP_RIM_LOCAL_Z_M = {
    "c_cups": 0.0320000,
    "g_cups": 0.0370385,
    "j_cups": 0.0391450,
}

# Conservative usable centre-line radii measured across the original compound
# cup interiors. A ball centre must be inside this radius minus its own
# measured radius while crossing the rim; physical cup contact is required too.
CUP_INNER_CENTER_RADIUS_M = {
    "c_cups": 0.02490,
    "g_cups": 0.03369,
    "j_cups": 0.04060,
}


@dataclass(frozen=True)
class YCB02Variant:
    cup: str
    correct_ball: str

    @property
    def balls(self) -> tuple[str, ...]:
        """Every episode exposes the same four physical candidates."""

        return BALL_NAMES

    @property
    def smaller_fitting_balls(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.balls
            if name != self.correct_ball and ball_fits_cup(self.cup, name)
        )

    @property
    def oversized_balls(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.balls
            if not ball_fits_cup(self.cup, name)
        )

    @property
    def fit_mask(self) -> tuple[bool, ...]:
        return tuple(ball_fits_cup(self.cup, name) for name in self.balls)


VARIANTS = (
    YCB02Variant(
        cup="c_cups",
        correct_ball="golf_ball",
    ),
    YCB02Variant(
        cup="g_cups",
        correct_ball="tennis_ball",
    ),
    YCB02Variant(
        cup="j_cups",
        correct_ball="baseball",
    ),
)

# Reset-layout geometry.  ``RESET_XY_BOUNDS_M`` is only a cheap rejection
# envelope; the sampler also applies the reach and camera envelope below.  The
# old 28 cm x 60 cm rectangle was safe but kept every body in the middle of the
# table, so policies never saw the left/right arm lanes.  This envelope covers
# the physical table's usable centre region while leaving a 60 mm margin at the
# far y edges.  The reach-aware predicate trims it to the union of the two arm
# workspaces, and the camera ellipse trims the corners that would leave the
# calibrated head-camera image.  Values are centre bounds, not permission to
# place a mesh through a table edge; the sampler subtracts each footprint and
# ``RESET_EDGE_CLEARANCE_M`` before drawing a candidate.
RESET_XY_BOUNDS_M = ((-0.52, -0.74), (-0.04, 0.74))
# Public aliases make the reset contract easy to consume from offline workers
# without importing Isaac-specific camera modules.
RESET_OBJECT_XY_BOUNDS_M = RESET_XY_BOUNDS_M

# Nominal ROKAE arm-root positions in the table/world frame.  The bimanual
# controller clamps each arm target to a 0.78 m sphere around its root.  Keep a
# 25 mm radial reserve in the configured radius plus a further 10 mm acceptance
# margin for IK interpolation and table-height randomization.  The sampler
# checks the full 3-D target (object centre + grasp offset), rather than using a
# misleading axis-aligned XY rectangle.
RESET_ARM_BASES_M = (
    (-0.75, -0.065, 1.34),  # right arm
    (-0.75, 0.065, 1.34),   # left arm
)
RESET_ARM_WORKSPACE_RADIUS_M = 0.755
RESET_ARM_REACH_CLEARANCE_M = 0.010
RESET_OBJECT_GRASP_HEIGHT_M = 0.080

# Broad inner approximation of the calibrated real-v1 head-camera tabletop
# footprint.  It is intentionally an ellipse rather than a box so the
# corners are screened smoothly, but it is wider than the old recording
# rectangle: the reach predicate, rather than an arbitrary camera crop, is
# now the limiting factor for the left/right arm lanes.  The sampler shrinks
# both radii by each object's footprint plus a small image-margin reserve, so
# an arbitrary-yaw cup remains in frame rather than merely having its centre
# in frame.  This is a camera-safe support region, not a second reach model.
RESET_CAMERA_ELLIPSE_CENTER_M = (-0.20, 0.0)
RESET_CAMERA_ELLIPSE_RADII_M = (0.55, 1.00)
RESET_CAMERA_CLEARANCE_M = 0.015
RESET_EDGE_CLEARANCE_M = 0.008
RESET_PAIR_CLEARANCE_M = 0.010
RESET_HAND_CLEARANCE_M = 0.045
RESET_MIN_MARGIN_M = 0.005

# The operator always sees two clean horizontal rows.  Only the identity-to-
# slot assignment changes between seeded resets.  The complete two-row group
# may translate inside the bounds below when the live Isaac arm/body cloud
# occupies one of the nominal slots.  X is table depth and Y is the left/right
# direction in the calibrated operator view.
CUP_ROW_X_M = -0.17
CUP_ROW_Y_M = (-0.36, 0.0, 0.36)
BALL_ROW_X_M = -0.30
BALL_ROW_Y_M = (-0.30, -0.10, 0.10, 0.30)
RESET_ROW_TRANSLATION_X_BOUNDS_M = (-0.14, 0.05)
RESET_ROW_TRANSLATION_Y_BOUNDS_M = (-0.20, 0.20)
RESET_ROW_RANDOM_ATTEMPTS = 128
RESET_ROW_FALLBACK_GRID_SIZE = (9, 11)
# If the complete fixed-slot group cannot clear the live Isaac articulation,
# keep cups and balls in separate depth rows but resample each row's lateral
# slots.  This is the production fallback for a robot home whose link origins
# occupy too much of the nominal fixed-row translation envelope.
RESET_ADAPTIVE_CUP_ROW_X_BOUNDS_M = (-0.24, -0.08)
RESET_ADAPTIVE_BALL_ROW_X_BOUNDS_M = (-0.47, -0.29)
RESET_ADAPTIVE_ROW_Y_BOUNDS_M = (-0.60, 0.60)
RESET_ADAPTIVE_ROW_ATTEMPTS = 2048
# Final fallback for a live home whose hand/link cloud blocks every shared row
# depth. Objects remain inside the cup/ball depth bands, but each one may use
# its own X coordinate. This adds free space without weakening clearance.
RESET_STAGGERED_ROW_ATTEMPTS = 8192
ROW_ALIGNMENT_TOLERANCE_M = 1.0e-9

# Temporary poses used while Isaac constructs YCB02's first PhysX scene.  They
# use the same safe two-row geometry as normal resets with canonical identity
# order, and are replaced by the seeded row permutations after construction.
FIRST_PHYSX_BOOTSTRAP_XY_M = {
    **{
        name: (CUP_ROW_X_M, y)
        for name, y in zip(CUP_NAMES, CUP_ROW_Y_M)
    },
    **{
        name: (BALL_ROW_X_M, y)
        for name, y in zip(BALL_NAMES, BALL_ROW_Y_M)
    },
}

# Circumscribed XY radii of the original collision meshes, rounded upward by
# roughly 1--2 mm.  Cups are sampled at arbitrary yaw, so an axis-aligned
# half-extent would not be safe; the radial envelope is conservative for all
# yaw values.  Balls use the exact analytic sphere radii below.
CUP_FOOTPRINT_RADIUS_M = {
    "c_cups": 0.0345,
    "g_cups": 0.0475,
    "j_cups": 0.0520,
}

MIN_LIFT_M = 0.020
RIM_INTERIOR_DEPTH_M = 0.005
HAND_CLEAR_SECONDS = 0.25
SUCCESS_STABLE_SECONDS = 0.50
WRONG_STABLE_SECONDS = 0.30
OVERSIZED_WEDGED_SECONDS = 1.00
OFF_TABLE_SECONDS = 0.20
CUP_TOPPLED_SECONDS = 0.50

BALL_STABLE_LINEAR_SPEED_MPS = 0.05
BALL_STABLE_ANGULAR_SPEED_RADPS = 1.0
CUP_SUCCESS_TILT_DEG = 12.0
CUP_TOPPLED_TILT_DEG = 60.0
CUP_TABLE_HEIGHT_TOLERANCE_M = 0.012
CONTACT_FORCE_EPSILON_N = 1.0e-6

# Full physical table support footprint from the authored YCB02 arena.
TABLE_X_BOUNDS_M = (-0.55, 0.25)
TABLE_Y_BOUNDS_M = (-0.80, 0.80)
OFF_TABLE_Z_MARGIN_M = 0.05


def variant_by_cup(cup_name: str) -> YCB02Variant:
    return next(variant for variant in VARIANTS if variant.cup == cup_name)


def permuted_fit_mask(
    variant: YCB02Variant,
    candidate_order: tuple[str, ...],
) -> tuple[bool, ...]:
    return tuple(ball_fits_cup(variant.cup, name) for name in candidate_order)


def ball_fits_cup(cup_name: str, ball_name: str) -> bool:
    """Return the locked geometric capacity classification."""

    return BALL_RADIUS_M[ball_name] <= CUP_INNER_CENTER_RADIUS_M[cup_name]


def largest_fitting_ball(cup_name: str) -> str:
    """Return the unique largest candidate that fits ``cup_name``."""

    fitting = [name for name in BALL_NAMES if ball_fits_cup(cup_name, name)]
    if not fitting:
        raise ValueError(f"YCB02 cup {cup_name!r} has no fitting candidate")
    return max(fitting, key=BALL_RADIUS_M.__getitem__)


def ball_inside_radial_limit(cup_name: str, ball_name: str) -> float:
    """Rim-plane fit margin used to classify candidates and crossings."""

    return CUP_INNER_CENTER_RADIUS_M[cup_name] - BALL_RADIUS_M[ball_name]


def settled_ball_radial_limit(
    cup_name: str,
    ball_name: str,
    ball_local_z: float,
) -> float:
    """Physical centre limit after accounting for the sphere below the rim.

    Once the ball's top falls below the rim, its rim-plane cross-section
    shrinks to zero.  Subtracting the full ball radius at every depth rejects
    valid side-supported settled poses even though the original cup collider
    prevents escape or interpenetration.
    """

    ball_radius = BALL_RADIUS_M[ball_name]
    depth_below_rim = max(
        0.0,
        CUP_RIM_LOCAL_Z_M[cup_name] - float(ball_local_z),
    )
    rim_cross_section_radius = math.sqrt(
        max(0.0, ball_radius * ball_radius - depth_below_rim * depth_below_rim)
    )
    return CUP_INNER_CENTER_RADIUS_M[cup_name] - rim_cross_section_radius
