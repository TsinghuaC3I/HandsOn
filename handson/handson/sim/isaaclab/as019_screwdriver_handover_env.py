"""Isaac Sim 5.1 PhysX runtime for B320-AS019 screwdriver handover.

The task reuses the production dual-ROKAE/Wuji lifecycle and replaces only the
two assembly objects with the normalized Poly Haven flathead screwdriver and a
collision-free tool-zone marker.  Success is an ordered physical event chain:
right-hand contact, left-hand contact while airborne, then supported landing in
the target zone.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .bimanual_assembly_env import (
    BimanualAssemblyIsaacLabEnv,
    _DEFAULT_RENDER_SIZE,
    _invert_pose_matrix,
    _matrix_to_quat_wxyz,
    _pose_vector_to_matrix,
    _quat_mul_wxyz,
    _quat_wxyz_to_matrix,
    _torch_view,
)
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE


TASK_NAME = "t22_pass_screwdriver"
TASK_ID = "B320-AS019"
_CONTROL_TRACE_ENV = "HANDSON_AS019_CONTROL_TRACE"
_CONTROL_TRACE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
AS019_SUCCESS_CONDITION_LABELS = (
    "Right hand contacts the screwdriver first",
    "Left hand contacts the airborne screwdriver afterward",
    "Screwdriver lands inside the target zone",
)
_AS019_SUCCESS_CONDITION_KEYS = (
    "as019_right_hand_contact",
    "as019_left_airborne_contact",
    "as019_zone_landed",
)
AS019_HIDDEN_STATE_FIELDS = (
    "stage",
    "receiver_hold_counter",
    "receiver_hold_complete_step",
    "left_only_hold_counter",
    "stability_counter",
    "progress_reset_candidate_counter",
    "progress_reset_count",
    "progress_reset_reason_code",
    "success",
    "failure_latched",
    "right_assist_anchor_latched",
    "left_assist_anchor_latched",
    "right_active_wrench_owner",
    "left_active_wrench_owner",
    "right_wrench_gate_contact",
    "right_wrench_invariant",
    "automatic_phase",
    "screwdriver_mass_kg",
    "screwdriver_friction",
    "env_step",
    "table_safety_guard_armed",
    "table_safety_failure_candidate_code",
    "table_safety_failure_counter",
    "table_safety_guard_last_step",
    "inward_orientation_failure_counter",
    "inward_orientation_guard_last_step",
    "right_handle_lift_event_step",
    "dual_handle_touch_event_step",
    "left_only_hold_event_step",
    "zone_placed_event_step",
    "complete_event_step",
    "slider_orientation_latched",
    "slider_orientation_latched_step",
    "slider_right_active_owner",
    "slider_left_active_owner",
    "slider_right_contact_grace_owner",
    "slider_left_contact_grace_owner",
    "slider_right_last_live_step",
    "slider_left_last_live_step",
    "slider_right_wrist_lock_valid",
    "slider_right_wrist_lock_qw",
    "slider_right_wrist_lock_qx",
    "slider_right_wrist_lock_qy",
    "slider_right_wrist_lock_qz",
    "slider_left_wrist_lock_valid",
    "slider_left_wrist_lock_qw",
    "slider_left_wrist_lock_qx",
    "slider_left_wrist_lock_qy",
    "slider_left_wrist_lock_qz",
)
_AS019_HIDDEN_STATE_REPLAY_ATOL = 1.0e-6
# Live MANUS teleoperation must keep publishing the operator's left-hand
# joints after the receiver touches the screwdriver.  The original task-side
# pinch latch was useful for scripted/automatic trajectories, but it silently
# replaced live finger commands with a measured one-frame target during
# handover.  Keep the state fields and assist-anchor bookkeeping intact while
# bypassing only that target-freeze path in the live task.
_AS019_RECEIVER_PINCH_FREEZE_ENABLED = False
# Haorui can occasionally return a valid but different IK branch when the
# wrist orientation crosses a near-singular pose.  AS019 is a physical
# handover: a branch switch is unsafe even when the solver's output is finite
# and inside the joint limits, because the resulting servo impulse can launch
# the screwdriver.  Keep this task-local cap independent of generic runtime
# knobs.  Match the shared conservative arm default (0.03 rad/frame) so a
# replay manifest cannot silently make this physical handover less safe while
# still rejecting the 3.95 rad step seen in AS019 event 00000001.
_AS019_ARM_MAX_JOINT_STEP_RAD = 0.03
# A valid Cartesian command can move an arm only a bounded amount in one
# 30-Hz frame.  A multi-radian Haorui result is therefore treated as a branch
# switch, not as a target to chase gradually: even a 0.03-rad slew repeated
# over a few frames can drag a held screwdriver out of the workspace.
_AS019_ARM_BRANCH_REJECT_RAD = 0.50
_CONTACT_FORCE_N = 0.015
# PhysX contact offsets may report a speculative contact before the rendered
# surfaces touch.  Giver ownership is admitted only from detailed points at or
# below this separation, so a visually separated hand cannot activate the
# compliant grasp wrench.
_CONTACT_MAX_SEPARATION_M = 0.0
# Deeply interpenetrating shapes are not a valid grasp either.  This matches
# the task-wide pairwise penetration ceiling and prevents a tunneled finger
# pair from being promoted into a force owner.
_CONTACT_MAX_PENETRATION_M = 0.005
# A table contact manifold may remain speculative by the combined 1 mm tool
# and 2 mm table contact offsets.  Placement accepts that bounded manifold only
# when the independently computed collision-envelope gap agrees.  Deeper than
# 5 mm is unsafe for the tool and hands.  Arm/table contact is retained as
# diagnostic telemetry only and does not terminate the task.
_TABLE_SUPPORT_MAX_SEPARATION_M = 0.003
_TABLE_MAX_PENETRATION_M = 0.005
# PhysX detailed-contact buffers are float32.  Compare their float64-converted
# values against the exact float32 encodings of the contract boundaries so a
# nominal +3 mm/-5 mm row remains inclusive without admitting the adjacent
# out-of-contract float32 value.
_TABLE_SUPPORT_MAX_SEPARATION_PHYSX_M = float(
    np.float32(_TABLE_SUPPORT_MAX_SEPARATION_M)
)
_TABLE_MIN_SEPARATION_PHYSX_M = float(
    np.float32(-_TABLE_MAX_PENETRATION_M)
)
# The analytic gap is derived independently in float64 from pose transforms.
# A one-nanometre allowance covers transform roundoff only; it is not applied
# to PhysX detailed-contact rows and does not relax the millimetre contract.
_TABLE_ANALYTIC_GAP_ROUNDOFF_M = 1.0e-9
# Reset/hard-rebuild contact views can retain one stale detailed row when the
# first operator action wakes the scene.  AS019 already waits for a stable
# recording origin, but terminal safety state lives inside the task and must
# be explicitly armed after action capture begins.  Three grace frames reject
# that stale boundary; three further, distinct 30 Hz frames are required for a
# genuine table-safety failure.  The physical task-owned arm home below keeps
# the complete hand envelope clear throughout the grace rather than relying on
# the debounce to hide an invalid reset.
_TABLE_SAFETY_ARM_AFTER_CONTROL_STEPS = 3
_TABLE_SAFETY_FAILURE_DEBOUNCE_STEPS = 3
_TABLE_SAFETY_FAILURE_CODES = {
    "": 0,
    "table_safety_query_invalid": 1,
    "screwdriver_table_penetration_exceeded": 2,
    "hand_table_penetration_exceeded": 3,
}
# Detailed PhysX points may lie slightly inside a convex contact surface.  Keep
# the semantic handle test tight to the authored visual envelope while allowing
# the tool's 1 mm contact offset and ordinary solver tolerance.  This upper
# bound is deliberately smaller than the 5 mm penetration rejection limit: a
# point outside the rendered handle must never become a compliant owner merely
# because an oversized proxy reports a contact there.
_CONTACT_GEOMETRY_TOLERANCE_M = 0.002
# Keep the giver and receiver roles disjoint while placing both on the real
# wooden handle.  The source-derived TailMain/HandleNeck split is at local
# x=-0.0508144237799039; the narrower windows leave ordinary contact-offset
# tolerance at either end without allowing one detailed point to satisfy both
# ordered handover roles.
_RIGHT_TAIL_CONTACT_NORMALIZED_X_BOUNDS_M = (-0.120, -0.08333333333333333)
_LEFT_MIDDLE_CONTACT_NORMALIZED_X_BOUNDS_M = (-0.070, -0.023333333333333334)
_RIGHT_PINCH_MAX_RADIAL_COSINE = -0.20
_RIGHT_PINCH_MIN_RADIAL_SPAN_M = 0.012
_RIGHT_PINCH_MIN_RADIAL_RADIUS_M = 0.005
_RELEASE_FORCE_N = 0.006
_OUTWARD_MAX_ERROR_RAD = float(np.deg2rad(30.0))
_OUTWARD_COS = float(np.cos(_OUTWARD_MAX_ERROR_RAD))
# The sharp tip remains a hard stage-aware failure, but a single solver/contact
# impulse may cross the 30-degree cone for one rendered frame.  Require a
# sustained violation over four distinct control frames.  Physical contact of
# the sharp tip with the receiving palm remains independently latched on the
# very substep where it occurs.
_SHARP_TIP_INWARD_DEBOUNCE_STEPS = 4
# A real right-tail grasp may drift slightly while it clears the table.  Admit
# the ordered grasp/slider path anywhere inside the operator-approved 15-degree
# horizontal-left lock window.  This remains strictly inside the authoritative
# 30-degree failure boundary: once the event begins, every subsequent frame is
# still fail-closed against ``_OUTWARD_COS``.
_OUTWARD_ENTRY_MAX_ERROR_RAD = float(np.deg2rad(15.0))
_OUTWARD_ENTRY_COS = float(np.cos(_OUTWARD_ENTRY_MAX_ERROR_RAD))
_RECEIVER_HOLD_STEPS = 12
_STABILITY_STEPS = 12
_RELEASE_DEBOUNCE_STEPS = 3
_RIGHT_LIFT_MIN_TABLE_GAP_M = 0.010
_PROGRESS_RESET_DEBOUNCE_STEPS = 3
_PROGRESS_RESET_REASON_CODES = {
    "": 0,
    "dropped_before_left_only_hold": 1,
    "dropped_outside_target_zone": 2,
}
# Begin only after the authoritative receiver hold.  Release the opposing
# thumb before the four support fingers so the hand unloads its pinch instead
# of sweeping both sides of the handle at once.
_RIGHT_RELEASE_THUMB_STEPS = 8
_RIGHT_RELEASE_FINGER_STEPS = 12
_RIGHT_RELEASE_OPEN_STEPS = (
    _RIGHT_RELEASE_THUMB_STEPS + _RIGHT_RELEASE_FINGER_STEPS
)
_RIGHT_POST_OPEN_UNLOAD_STEPS = 12


def _as019_robot_table_contact_role(rigid_owner_path: str) -> str:
    """Classify one exact robot collision owner as hand or arm.

    Wuji palm/finger rigid owners are the complete hand collision set.  Every
    other collision-owning rigid body under LeftRobot/RightRobot belongs to the
    mounted arm (including the wrist); keeping the partition path-exact avoids
    broad globs that could accidentally include the visual target marker.
    """

    name = str(rigid_owner_path).rstrip("/").rsplit("/", 1)[-1].lower()
    if "_palm_link" in name or "_finger" in name:
        return "hand"
    return "arm"


def _as019_taskdev_module(module_name: str):
    """Load the packaged source scene without depending on the checkout root."""
    return importlib.import_module(".as019_asset_support." + module_name, __package__)


_AS019_SCENE_CONTRACT = _as019_taskdev_module("scene_contract")
_SCREWDRIVER_SCALE_XYZ = tuple(
    float(value)
    for value in _AS019_SCENE_CONTRACT.SCREWDRIVER_SCALE_XYZ
)
if len(_SCREWDRIVER_SCALE_XYZ) != 3 or not np.allclose(
    _SCREWDRIVER_SCALE_XYZ,
    (_SCREWDRIVER_SCALE_XYZ[0],) * 3,
    atol=0.0,
    rtol=0.0,
):
    raise RuntimeError("AS019 screwdriver task scale must remain uniform")
_SCREWDRIVER_UNIFORM_SCALE = _SCREWDRIVER_SCALE_XYZ[0]
_AS019_RIGHT_HOME = np.asarray(
    _AS019_SCENE_CONTRACT.AS019_TABLETOP_TCP_HOME_WXYZ["right"],
    dtype=np.float64,
).copy()
_AS019_LEFT_HOME = np.asarray(
    _AS019_SCENE_CONTRACT.AS019_TABLETOP_TCP_HOME_WXYZ["left"],
    dtype=np.float64,
).copy()
if (
    _AS019_RIGHT_HOME.shape != (7,)
    or _AS019_LEFT_HOME.shape != (7,)
    or not np.all(np.isfinite(_AS019_RIGHT_HOME))
    or not np.all(np.isfinite(_AS019_LEFT_HOME))
):
    raise RuntimeError("AS019 tabletop TCP homes must be finite 7D poses")
_RIGHT_TAIL_CONTACT_LOCAL_X_MIN_M = (
    _RIGHT_TAIL_CONTACT_NORMALIZED_X_BOUNDS_M[0]
    * _SCREWDRIVER_UNIFORM_SCALE
)
_RIGHT_TAIL_CONTACT_LOCAL_X_MAX_M = (
    _RIGHT_TAIL_CONTACT_NORMALIZED_X_BOUNDS_M[1]
    * _SCREWDRIVER_UNIFORM_SCALE
)
_LEFT_MIDDLE_CONTACT_LOCAL_X_MIN_M = (
    _LEFT_MIDDLE_CONTACT_NORMALIZED_X_BOUNDS_M[0]
    * _SCREWDRIVER_UNIFORM_SCALE
)
_LEFT_MIDDLE_CONTACT_LOCAL_X_MAX_M = (
    _LEFT_MIDDLE_CONTACT_NORMALIZED_X_BOUNDS_M[1]
    * _SCREWDRIVER_UNIFORM_SCALE
)

# Clear the enlarged handle's radial envelope plus a full open-finger margin
# full open-finger margin before the giver starts its longer home retreat.
_RIGHT_POST_OPEN_UNLOAD_DISTANCE_M = 0.075
# The Poly Haven flathead is normalized by the task asset builder into the
# legacy task frame: blunt tail at local -X, sharp tip at local +X, and the
# radial center at local Y=+0.025.  Keep these values in exact parity with that
# compatibility transform instead of querying a render mesh at runtime.
_TOOL_LOCAL_MIN = np.asarray(
    [-0.125, 0.0100594805, -0.0149405186], dtype=np.float64
) * _SCREWDRIVER_UNIFORM_SCALE
_TOOL_LOCAL_MAX = np.asarray(
    [0.13500000163912773, 0.0399405195, 0.0149405186], dtype=np.float64
) * _SCREWDRIVER_UNIFORM_SCALE
_TOOL_LOCAL_TIP = np.asarray(
    [0.13500000163912773, 0.025, 0.0], dtype=np.float64
) * _SCREWDRIVER_UNIFORM_SCALE
_TOOL_LOCAL_BOUNDS_CENTER = 0.5 * (_TOOL_LOCAL_MIN + _TOOL_LOCAL_MAX)
_TOOL_HALF_EXTENTS = 0.5 * (_TOOL_LOCAL_MAX - _TOOL_LOCAL_MIN)

# Exact wooden-handle visual envelope after the same normalization.  Contact
# ownership is restricted to this box as well as the longitudinal role window;
# shaft, ferrule, blade, and oversized-proxy contacts therefore cannot attach a
# hand or restore an external wrench.
_HANDLE_LOCAL_MIN = np.asarray(
    [-0.125, 0.0100594805, -0.0149405186], dtype=np.float64
) * _SCREWDRIVER_UNIFORM_SCALE
_HANDLE_LOCAL_MAX = np.asarray(
    [-0.016476666758274833, 0.0399405195, 0.0149405186], dtype=np.float64
) * _SCREWDRIVER_UNIFORM_SCALE
_HANDLE_LOCAL_CENTER = 0.5 * (_HANDLE_LOCAL_MIN + _HANDLE_LOCAL_MAX)

# The sharp flat blade begins at this measured source-mesh transition.  The
# full radial visual-envelope tolerance below keeps edge contacts conservative
# without classifying contacts on the preceding round stem as blade contact.
_SHARP_TIP_LOCAL_X_MIN_M = (
    0.11085820227730864 * _SCREWDRIVER_UNIFORM_SCALE
)

# Source-derived AABBs for the five cooked collision roles.  These deliberately
# preserve the task-local +Y=25 mm radial offset.  Support math therefore stays
# in parity with TailMain, HandleNeck, Ferrule, Stem, and FlatBlade for any
# randomized roll instead of reusing the old two-cylinder proxy.
# Each tuple is (local center, local half extents).
_UNSCALED_TOOL_BOX_SUPPORTS = (
    (
        np.asarray([-0.08790721188995195, 0.025, 0.0], dtype=np.float64),
        np.asarray(
            [0.03709278811004805, 0.014940519507416085, 0.014940518580705114],
            dtype=np.float64,
        ),
    ),
    (
        np.asarray(
            [-0.03364554526908937, 0.025, 0.0], dtype=np.float64
        ),
        np.asarray(
            [
                0.017168878510814535,
                0.014110953795441937,
                0.014110954722152909,
            ],
            dtype=np.float64,
        ),
    ),
    (
        np.asarray(
            [-0.005490573084820288, 0.025, 0.0], dtype=np.float64
        ),
        np.asarray(
            [
                0.011539721927155397,
                0.008594035810851752,
                0.00859403581085175,
            ],
            dtype=np.float64,
        ),
    ),
    (
        np.asarray([0.04835526164323776, 0.025, 0.0], dtype=np.float64),
        np.asarray(
            [
                0.06250294063407088,
                0.0018862355001680598,
                0.001886235500168058,
            ],
            dtype=np.float64,
        ),
    ),
    (
        np.asarray([0.1229291019582182, 0.025, 0.0], dtype=np.float64),
        np.asarray(
            [
                0.012070899680909561,
                0.0035102704104412115,
                0.0020471742665886697,
            ],
            dtype=np.float64,
        ),
    ),
)
_TOOL_BOX_SUPPORTS = tuple(
    (
        center * _SCREWDRIVER_UNIFORM_SCALE,
        half_extents * _SCREWDRIVER_UNIFORM_SCALE,
    )
    for center, half_extents in _UNSCALED_TOOL_BOX_SUPPORTS
)
_TOOL_STABLE_FACET_STEP_RAD = float(np.deg2rad(90.0))
# Kept as a public diagnostic value; the receiver waypoints below carry their
# measured radial compensation explicitly.
_LEFT_RECEIVER_IK_COMPENSATION_LOCAL = np.zeros(3, dtype=np.float64)
# Approach the blunt handle radially from its open side.  These poses are the
# measured collision-clear primary receiver branch from the live multi-roll sweep: the
# pregrasp stays outside the handle and the final pose establishes a real
# thumb/opposing-finger pinch on the middle window.  A ten-degree tool-local
# wrist-roll correction selects the reachable DLS branch, and the 8 mm lower
# radial path is the shallowest tested offset that reached simultaneous support
# without a safety-axis violation; deeper offsets produced contact impulses.
_LEFT_RECEIVER_PREGRASP_LOCAL = np.asarray(
    [-0.160, 0.247, -0.018], dtype=np.float64
)
_LEFT_RECEIVER_GRASP_LOCAL = np.asarray(
    [-0.10138225, 0.13681291, 0.07072747], dtype=np.float64
)
_LEFT_RECEIVER_PREGRASP_LOCAL_QUAT = np.asarray(
    [0.7071067812, 0.7071067812, 0.0, 0.0], dtype=np.float64
)
_LEFT_RECEIVER_LOCAL_QUAT = np.asarray(
    [0.5376058138, 0.8115544557, 0.0779338205, 0.2151410868],
    dtype=np.float64,
)
# The lower-Y edge of the required reset envelope selects a different, already
# collision-verified wrist branch.  A single DLS orientation cannot cover both
# edges: the primary branch is robust through the central/upper workspace,
# while this measured branch is required at the lower-Y edge.  The position
# switch is based only on the sampled reset pose and does not alter contact or
# success semantics.  The primary wrist branch accumulates an unrecoverable lateral DLS residual
# across the lowest 4 mm of the accepted reset range.  Select the already
# collision-checked low-Y wrist branch throughout that measured slice instead
# of only at the proposal boundary, where it was effectively never exercised.
_LEFT_RECEIVER_LOW_Y_THRESHOLD_M = -0.042
_LEFT_RECEIVER_LOW_Y_PREGRASP_LOCAL = np.asarray(
    [-0.160, 0.247, -0.010], dtype=np.float64
)
_LEFT_RECEIVER_LOW_Y_GRASP_LOCAL = np.asarray(
    [-0.10138225, 0.13681291, 0.07872747], dtype=np.float64
)
_LEFT_RECEIVER_LOW_Y_LOCAL_QUAT = np.asarray(
    [0.46482843, 0.85532168, 0.09638804, 0.20753003],
    dtype=np.float64,
)
# A disturbed grasp can leave the screwdriver just outside the 15-degree
# stage-entry gate.  Once the giver has held real tail contact for long enough
# to settle, let the open receiver fingers make a short, low-force approach
# from the handle side.  This is an action-only, operator-reproducible
# pre-alignment; the task remains ballistic and the sharp-tip/palm exclusion
# continues to be evaluated by the authoritative contact query.
_LEFT_STAGE0_PREALIGN_LOCAL = np.asarray(
    [0.030, 0.140, 0.030], dtype=np.float64
)
_LEFT_STAGE0_PREALIGN_WORLD_LIFT_M = 0.120
_LEFT_STAGE0_PREALIGN_DELAY_STEPS = 40
_LEFT_STAGE0_PREALIGN_APPROACH_STEPS = 30
_LEFT_STAGE0_PREALIGN_DESCENT_STEPS = 40
_LEFT_STAGE0_PREALIGN_PUSH_STEPS = 50
_LEFT_STAGE0_PREALIGN_PUSH_LOCAL_Y_M = -0.060
_RIGHT_HANDOVER_TRANSLATION_WORLD = np.asarray(
    [0.0, 0.005, -0.005], dtype=np.float64
)
_RIGHT_HANDOVER_PRETRANSFER_HOLD_STEPS = 1
_RIGHT_HANDOVER_STEPS = 600
_RIGHT_HANDOVER_SETTLE_STEPS = 8
# Start the long, collision-clear receiver pregrasp while the giver finishes
# its wrist transition.  Waiting for the giver to settle made the receiver
# chase a moving tool and arrive only after giver contact had unloaded.  The
# four-step event gate reproduces the physically successful approach timing;
# the receiver is still far above the handle throughout the wrist transition.
_LEFT_RECEIVER_APPROACH_DELAY_STEPS = 4
_LEFT_RECEIVER_APPROACH_STEPS = 60
_LEFT_RECEIVER_DESCENT_STEPS = 40
_LEFT_RECEIVER_CONTACT_PRELOAD_DISTANCE_M = 0.003
_LEFT_RECEIVER_CONTACT_PRELOAD_STEPS = 8
# A latched thumb touch is no longer a viable receiver grasp once the thumb
# has cleared while an opposing finger is carrying load.  Confirm that
# one-sided state for two control frames before using the existing safe-open
# reseat path; otherwise the stale thumb latch keeps closing the loaded finger
# and can turn the sharp axis inward.
_LEFT_RECEIVER_THUMB_LOSS_DEBOUNCE_STEPS = 2
# On some reset poses the ROKAE DLS controller settles a few centimetres
# outside the same collision-checked grasp endpoint.  Let the nominal descent
# settle first, then close that measured palm error while the receiver is
# still contact-free.  This is robot-command feedback only: the screwdriver
# remains ballistic and the first live thumb contact still gates hand closure.
_LEFT_RESEAT_TARGET_LOCAL = np.asarray(
    [-0.106, 0.135, 0.066], dtype=np.float64
)
_LEFT_RESEAT_DELAY_STEPS = 12
_LEFT_RESEAT_COMMAND_FEEDBACK_GAIN = 1.0
_LEFT_RESEAT_MAX_CORRECTION_M = 0.015
_LEFT_RESEAT_SINGLE_CONTACT_OPEN_STEPS = 24
_LEFT_RESEAT_RETRY_RETURN_STEPS = 12
# A dual-contact latch can settle several millimetres away from the earlier
# first-thumb pose.  Above this measured separation, keep the live receiver
# palm instead of snapping back to the stale touch pose during giver release.
_STAGE2_DUAL_CONTACT_LIVE_HOLD_MIN_M = 0.004
# If the receiver loses its opposed pinch, unload the failed pinch, move on
# the stable live wrist branch, pre-shape it to 40%, then close while
# approaching from the opposite side.  Stage 3 reopens only finger 5; a stage
# 1 retry also reopens its already-separated thumb.  Four control frames
# unload the measured one-finger failure before it can torque the giver-held
# tool across the safety boundary; the subsequent phases remain slow.
_STAGE3_REGRASP_LOCAL_OFFSET = np.asarray(
    [-0.005, 0.0, -0.058], dtype=np.float64
)
_STAGE1_REGRASP_LOCAL_OFFSET = np.asarray(
    [-0.005, 0.0, 0.0], dtype=np.float64
)
_STAGE3_REGRASP_NEAR_LOCAL_Y_M = 0.040
_STAGE3_REGRASP_OPEN_STEPS = 4
_STAGE3_REGRASP_MOVE_STEPS = 24
_STAGE3_REGRASP_PRESHAPE_STEPS = 24
_STAGE3_REGRASP_APPROACH_STEPS = 24
_STAGE3_REGRASP_CLOSE_STEPS = 24
_STAGE3_REGRASP_PRESHAPE_FRACTION = 0.40
_STAGE3_REGRASP_CONTACT_CONFIRM_STEPS = 8
_MARKER_HALF_EXTENTS = np.asarray([0.260, 0.070, 0.00075], dtype=np.float64)
_RESET_TOOL_ZONE_VISIBLE_GAP_M = 0.010
_RESET_MAX_SAMPLE_ATTEMPTS = 256
# The task contract explicitly rejects unreachable randomized samples.  The
# shared ROKAE/Wuji Cartesian controller cannot reach the receiver pose once
# the screwdriver root is farther forward than this live-swept boundary; it
# settles above the handle even after bounded robot-command feedback.  Keep the
# full authored +/-35 mm proposal distribution, but resample those unreachable
# proposals instead of presenting impossible collection episodes.
_RESET_CONTROLLER_REACHABLE_TOOL_X_MAX_M = -0.138
_RESET_CONTROLLER_REACHABLE_TOOL_Y_MIN_M = -0.042
_RESET_CONTROLLER_REACHABLE_TOOL_Y_MAX_M = 0.0
# At the lower-Y edge, the same receiver wrist branch reaches the handle only
# when the tool is farther back in the shared workspace.  Live sweeps showed
# the closer corner settling above/outside the handle even after bounded DLS
# feedback.  Keep proposing the full authored rectangle and reject only this
# measured controller-unreachable corner.
_RESET_CONTROLLER_REACHABLE_LOW_Y_TOOL_X_MAX_M = -0.145
# The reset controller was physically swept only through +/-8 degrees.  Keep
# that proven reset envelope independent from the wider 15-degree post-grasp
# lock window, so this assist repair does not silently expand reset sampling.
_RESET_CONTROLLER_REACHABLE_ABS_YAW_MAX_RAD = float(np.deg2rad(8.0))
_HIGH_HANDOVER_ASSIST_TOOL_Y_THRESHOLD_M = -0.040
# The visual-only marker center is authored 4 mm above the physical table so
# the bright target remains visible above the rendered tabletop.
# Placement uses the table, not the translucent marker top, as its support
# surface.  Reset randomization changes marker x/y only, so this offset is
# invariant across poses and seeds.
_MARKER_CENTER_ABOVE_TABLE_M = 0.004
_ZONE_SUPPORT_CLEARANCE_M = 0.001
_ZONE_SUPPORT_READY_GAP_M = 0.003
_ZONE_HOVER_CLEARANCE_M = 0.005
_RIGHT_RETREAT_NORMAL_STEPS = 24
_RIGHT_RETREAT_HOME_STEPS = 48
_RIGHT_RETREAT_DISTANCE_M = float(_TOOL_HALF_EXTENTS[1] + 0.120)
_RIGHT_CLEAR_READY_STEPS = 5
_ZONE_X_ALIGN_STEPS = 80
_ZONE_Y_ALIGN_STEPS = 208
_ZONE_DESCENT_STEPS = 48
_PLACEMENT_REACQUIRE_STEPS = 40
_PLACEMENT_REACQUIRE_CONTACT_STEPS = 8
# PhysX can report a one- or two-control-frame force gap while the closed
# receiver remains geometrically seated on the cylindrical handle.  Pause the
# placement path through that short gap; sustained loss still enters the
# existing robot-only radial reacquisition path.
_PLACEMENT_CONTACT_LOSS_DEBOUNCE_STEPS = 3
_ZONE_STABLE_ROLL_LATCH_DELAY_STEPS = 12
_ZONE_STABLE_ROLL_EARLY_LATCH_STEPS = 2
_ZONE_STABLE_ROLL_EARLY_MIN_CONTACT_FORCE_N = 4.0
_ZONE_STABLE_ROLL_EARLY_MAX_OUTWARD_ALIGNMENT = 0.9975
_ZONE_STABLE_ROLL_STEPS = 30
_ZONE_STABLE_ROLL_MIN_CONTACT_FORCE_N = 1.0
_ZONE_PLACEMENT_ROLL_RAD = 0.0
# The open receiver withdraws toward the blunt end after table support.  Live
# A recovered side pinch loses its opposing finger if dragged all the way to
# the geometric center.  Aim 15 mm toward the entry side instead; the unchanged
# zone predicate permits +/-28 mm in Y, so the endpoint retains 13 mm of margin
# while shortening the contact-loaded lateral motion.
_ZONE_PLACEMENT_CENTER_OFFSET_LOCAL = np.asarray(
    [0.0, -0.015, 0.0], dtype=np.float64
)
# The ROKAE DLS controller can retain a few centimetres of Cartesian error on
# the low-table branch even after its command stops moving.  Close that
# measured endpoint error through the arm command while the receiver still
# physically owns the handle.  This changes only the scripted demonstrator;
# the task's zone and safety acceptance remain unchanged.
_ZONE_FINAL_COMMAND_CORRECTION_MAX_M = 0.060
_ZONE_FINAL_VERTICAL_COMMAND_CORRECTION_GAIN = 2.0
_ZONE_RELEASE_READY_STEPS = 5
_ZONE_RELEASE_MAX_LINEAR_SPEED_MPS = 0.020
_ZONE_RELEASE_MAX_ANGULAR_SPEED_RADPS = 0.20
_LEFT_RELEASE_OPEN_DWELL_STEPS = 2
_LEFT_RELEASE_THUMB_CONTACT_UNLOAD_STEPS = 24
_LEFT_RELEASE_THUMB_FINISH_OPEN_STEPS = 12
_LEFT_RELEASE_THUMB_OPEN_STEPS = (
    _LEFT_RELEASE_THUMB_CONTACT_UNLOAD_STEPS
    + _LEFT_RELEASE_THUMB_FINISH_OPEN_STEPS
)
_LEFT_RELEASE_FINGER_OPEN_STEPS = 12
_LEFT_RELEASE_FINGER_PEEL_STEPS = 12
_LEFT_RELEASE_SIMULTANEOUS_OPEN = True
# The recovered seed-108 grasp is already physically clear on the first
# opening frame.  Complete the thumb pre-peel four frames sooner so the
# unchanged 12-frame post-release stability gate fits inside the 900-step
# episode without making the main whole-hand opening any faster.
_LEFT_RELEASE_PREPEEL_STEPS = 20
_LEFT_RELEASE_PREPEEL_START_INDEX = 0
_LEFT_RELEASE_PREPEEL_END_INDEX = 4
_LEFT_RELEASE_OPEN_STEPS = 24
_LEFT_RELEASE_OPEN_UNLOAD_DISTANCE_M = 0.045
_LEFT_RELEASE_PRECLEAR_RETREAT_DISTANCE_M = 0.070
_LEFT_RELEASE_PRECLEAR_RETREAT_STEPS = 12
_LEFT_RELEASE_PARTIAL_THUMB_TARGET = 0.0
_LEFT_RELEASE_RETRACT_READY_STEPS = 1
_LEFT_RELEASE_RETRACT_STEPS = 48
_LEFT_RELEASE_CONTACT_RETRACT_STEPS = 24
_LEFT_RELEASE_CLEAR_DISTANCE_M = 0.060
_LEFT_RELEASE_CLEAR_ESCAPE_DISTANCE_M = 0.0
_RELEASE_TELEMETRY_MAX_FRAMES = 1024
_HAND_OPEN_THRESHOLD = 0.15
_HAND_CLOSE_THRESHOLD = 0.35
# Keep compliant ownership synchronized with the unchanged closed-hand event
# gate.  Sub-threshold preshape contact is intentionally ballistic so a hand
# can make a gentle tabletop alignment push before committing to the grasp.
_RIGHT_EARLY_ASSIST_ATTACH_THRESHOLD = _HAND_CLOSE_THRESHOLD
_ASSIST_LINEAR_KP = 850.0
_ASSIST_LINEAR_KD = 16.0
_ASSIST_ANGULAR_KP = 5.0
# The two contact-owned anchors can disagree briefly at the instant the
# receiver closes.  Dampen that real overlap impulse enough to keep the tool
# inside the 30-degree safety cone.  The default 2.5 Nm cap and the
# measured low-Y tier below both remain bounded; placement retains its separate
# stronger damping tier.
_ASSIST_ANGULAR_KD = 0.75
_ASSIST_OUTWARD_KP = 12.0
_STAGE0_OUTWARD_KP = 4.0
_HANDOVER_HOLD_OUTWARD_KP = 8.0
_HIGH_HANDOVER_HOLD_OUTWARD_KP = 10.0
_RECEIVER_RETRY_HANDOVER_HOLD_OUTWARD_KP = 24.0
# Once the ordered right-tail grasp has promoted the task to stage 1 and the
# tool is visibly clear of the table, AS019 behaves like a screwdriver carried
# by a three-axis sliding guide: translation remains free, but the complete
# task-authored horizontal-left orientation is restored by a bounded physical
# torque.  The owner wrist target is frozen at the measured contact pose as
# well, so a rotation-only operator command cannot wind the grasp against the
# guide.  This is deliberately not a rigid-body state write; losing live
# opposed contact removes translation immediately and retains orientation only
# for the bounded query-flicker grace below.
_SLIDER_LATCH_MIN_TABLE_GAP_M = 0.010
# Preserve only the horizontal-left orientation guide across a short PhysX
# contact-query flicker after a real lifted grasp has latched it.  Translation
# remains ballistic during the gap, and opening the hand cancels the grace on
# the next physics substep.  Four 30 Hz control frames cover one noisy contact
# manifold refresh without turning the assist into a hidden grasp.
_SLIDER_CONTACT_GRACE_STEPS = 4
_SLIDER_ORIENTATION_ANGULAR_KP = 20.0
_SLIDER_ORIENTATION_ANGULAR_KD = 1.5
_SLIDER_ORIENTATION_MAX_TORQUE_NM = 3.5
_ASSIST_MAX_FORCE_N = 45.0
_ASSIST_MAX_TORQUE_NM = 2.5
# The low-Y receiver branch and the high-reset-yaw alignment branch need the
# measured 3.5 Nm cap while the giver is the sole owner and through the short
# dual-contact debounce.  Applying that cap to the central/low-yaw branch made
# its already stable dual grasp over-correct; placement and release retain
# their separately bounded tiers.
_HIGH_HANDOVER_ASSIST_MAX_TORQUE_NM = 3.5
_PLACEMENT_ASSIST_LINEAR_KP = 1800.0
_PLACEMENT_ASSIST_LINEAR_KD = 24.0
_PLACEMENT_ASSIST_ANGULAR_KP = 2.0
_PLACEMENT_ASSIST_ANGULAR_KD = 0.80
_PLACEMENT_ASSIST_OUTWARD_KP = 12.0
_PLACEMENT_ASSIST_MAX_FORCE_N = 18.0
_PLACEMENT_ASSIST_MAX_TORQUE_NM = 0.80
_RELEASE_SUPPORT_LINEAR_KP = 120.0
_RELEASE_SUPPORT_LINEAR_KD = 28.0
_RELEASE_SUPPORT_ANGULAR_KP = 1.0
_RELEASE_SUPPORT_ANGULAR_KD = 1.20
_RELEASE_SUPPORT_OUTWARD_KP = 4.0
_RELEASE_SUPPORT_MAX_FORCE_N = 8.0
_RELEASE_SUPPORT_MAX_TORQUE_NM = 0.50
_AUTOMATIC_GIVER_GRASP_TARGET = np.asarray(
    [
        1.41641399,
        0.045268495,
        1.41739004,
        0.553414384,
        0.037472055,
        0.245187309,
        0.151820915,
        0.264527498,
        1.204081680,
        -0.312368777,
        1.034289708,
        -0.365826094,
        1.0,
        0.0,
        1.0,
        1.0,
        0.769586480,
        0.040202724,
        0.938132077,
        0.372722983,
    ],
    dtype=np.float64,
)
# Preserve a scalar closure level for generic ownership thresholds and legacy
# diagnostics.  The giver itself must use the calibrated non-uniform target:
# scalar ``fill`` commands spread the thin replacement handle away from the
# opposing thumb/index contact pair.
_AUTOMATIC_GRASP_TARGET = float(
    np.max(np.abs(_AUTOMATIC_GIVER_GRASP_TARGET))
)
_AUTOMATIC_HANDOVER_REGRASP_OPEN_TARGET = 0.26
_AUTOMATIC_GIVER_REGRASP_OPEN_TARGET = (
    _AUTOMATIC_GIVER_GRASP_TARGET
    * (_AUTOMATIC_HANDOVER_REGRASP_OPEN_TARGET / _AUTOMATIC_GRASP_TARGET)
)
_AUTOMATIC_HANDOVER_REGRASP_OPEN_STEPS = 4
_AUTOMATIC_HANDOVER_REGRASP_CLOSE_STEPS = 8
_RIGHT_APPROACH_HIGH_WORLD_Z_M = 1.09
_RIGHT_APPROACH_LIFT_STEPS = 48
_RIGHT_APPROACH_ALIGN_STEPS = 48
_RIGHT_APPROACH_DESCENT_STEPS = 40
_RIGHT_APPROACH_LIFT_END_STEP = _RIGHT_APPROACH_LIFT_STEPS
_RIGHT_APPROACH_ALIGN_END_STEP = (
    _RIGHT_APPROACH_LIFT_END_STEP + _RIGHT_APPROACH_ALIGN_STEPS
)
_RIGHT_GRASP_CLOSE_START_STEP = (
    _RIGHT_APPROACH_ALIGN_END_STEP + _RIGHT_APPROACH_DESCENT_STEPS
)
_RIGHT_GRASP_CLOSE_STEPS = 35
_RIGHT_GRASP_CLOSE_END_STEP = (
    _RIGHT_GRASP_CLOSE_START_STEP + _RIGHT_GRASP_CLOSE_STEPS
)
_RIGHT_PICK_TARGET_LOCAL = np.asarray(
    [-0.177808511, 0.117600038, 0.103877986], dtype=np.float64
)
_RIGHT_HANDLE_CENTER_Y_COMPENSATION_M = float(
    _HANDLE_LOCAL_CENTER[1] - 0.025
)
_RIGHT_HANDLE_RADIUS_COMPENSATION_M = float(
    _TOOL_HALF_EXTENTS[2] - 0.014940518580705114
)
_RIGHT_PICK_TARGET_LOCAL[1] += _RIGHT_HANDLE_CENTER_Y_COMPENSATION_M
_RIGHT_PICK_TARGET_LOCAL[2] += _RIGHT_HANDLE_RADIUS_COMPENSATION_M
_RIGHT_ACQUIRE_APPROACH_STEPS = 64
_RIGHT_ACQUIRE_TARGET_LOCAL = np.asarray(
    # The screwdriver visual/collision asset is authored at the current 1.75
    # scale.  The previous -0.1477 m waypoint left the thumb's distal contact
    # on the positive (handle-neck) side of TailMain.  A 25 mm retreat plus
    # an 8 mm radial clearance keeps the palm in the calibrated wrist
    # workspace while putting the thumb and opposing finger in the legal tail
    # interval.  This is a robot target only (the tool is never moved here).
    [-0.172738511, 0.125600038, 0.052227986], dtype=np.float64
)
_RIGHT_ACQUIRE_TARGET_LOCAL[1] += _RIGHT_HANDLE_CENTER_Y_COMPENSATION_M
_RIGHT_ACQUIRE_TARGET_LOCAL[2] += _RIGHT_HANDLE_RADIUS_COMPENSATION_M
_RIGHT_ACQUIRE_END_STEP = (
    _RIGHT_GRASP_CLOSE_END_STEP + _RIGHT_ACQUIRE_APPROACH_STEPS
)
_RIGHT_TERMINAL_RESEAT_START_STEP = _RIGHT_ACQUIRE_END_STEP + 1
_RIGHT_ARRIVAL_POSITION_TOLERANCE_M = 0.010
_RIGHT_ARRIVAL_ROTATION_TOLERANCE_RAD = float(np.deg2rad(10.0))
_RIGHT_ARRIVAL_STABLE_STEPS = 4
_RIGHT_PICK_ARRIVAL_TIMEOUT_STEPS = 30
_RIGHT_ACQUIRE_ARRIVAL_TIMEOUT_STEPS = 30
_RIGHT_CONTACT_PRELOAD_DISTANCE_M = 0.0
_RIGHT_CONTACT_PRELOAD_STEPS = 12
# The ordered event now requires a measured 10 mm table clearance, so the
# unattended validator must perform the same observable lift as an operator
# before it can leave stage 0.
_RIGHT_POST_GRASP_LIFT_DISTANCE_M = 0.015
_RIGHT_POST_GRASP_LIFT_STEPS = 18
_RIGHT_ORIENTATION_ALIGNMENT_STEPS = 64
_RIGHT_WRIST_ROLL_OFFSET_RAD = -1.574236454637
# The physical wrist can retain several degrees of reset-dependent axis error
# even after a Cartesian command settles.  Feed the live safety-critical tool
# axis back into the giver command, without constraining longitudinal roll.
_RIGHT_AXIS_COMMAND_FEEDBACK_GAIN = 1.0
_RIGHT_AXIS_COMMAND_FEEDBACK_MAX_RAD = float(np.deg2rad(8.0))
_LEFT_AXIS_COMMAND_FEEDBACK_GAIN = 0.5
_LEFT_AXIS_COMMAND_FEEDBACK_STAGE4_GAIN = 0.10
_LEFT_AXIS_COMMAND_FEEDBACK_MAX_RAD = float(np.deg2rad(12.0))
_LEFT_OWNER_AXIS_ALIGNMENT_GAIN = 0.03
# A reset-specific pinch carries a larger hand-to-tool residual than the
# normalized template.  Keep the overlap correction gentle while two hands
# constrain the handle, then use the ordinary placement correction after
# giver-clear; the old 15% stage-3 response could not recover before the
# 30-degree fail-closed boundary.
_LEFT_OWNER_AXIS_ALIGNMENT_DISTINCT_GAIN = 0.03
_LEFT_OWNER_AXIS_ALIGNMENT_STAGE3_GAIN = 0.50
_LEFT_OWNER_AXIS_ALIGNMENT_DISTINCT_STAGE3_GAIN = 0.50
# Once the live tool is within five degrees of the outward axis, continuing
# to chase sub-degree solver noise moves the contact-owned palm by millimetres
# around the long handle.  Keep that safety margin as a command deadband; the
# 30-degree runtime failure gate remains authoritative.
_LEFT_OWNER_AXIS_ALIGNMENT_ACTIVATION_COS = float(
    math.cos(math.radians(5.0))
)
# A latched semantic owner can momentarily lose physical handle contact at the
# extremes of the authored reset envelope.  This small tool-frame correction
# re-seats the closed hand; it is a robot command, never an object-state write.
_RIGHT_RESEAT_TARGET_LOCAL = np.asarray(
    _RIGHT_ACQUIRE_TARGET_LOCAL, dtype=np.float64
)
_RIGHT_RESEAT_COMMAND_FEEDBACK_GAIN = 1.0
_RIGHT_RESEAT_MAX_CORRECTION_M = 0.001
_HAND_ABDUCTION_INDICES = np.asarray([1, 5, 9, 13, 17], dtype=np.int64)
_HAND_GRASP_INDICES = np.asarray(
    [index for index in range(20) if index not in _HAND_ABDUCTION_INDICES],
    dtype=np.int64,
)
_RIGHT_FINGER_FLEXION_INDICES = np.asarray(
    [index for index in range(4, 20) if index not in _HAND_ABDUCTION_INDICES],
    dtype=np.int64,
)
# Use a thumb-opposed, lightly preshaped four-finger grasp.  The 0.025 rad
# finger values leave the side aperture open during approach while avoiding a
# singular fully-straight hand; the thumb is curled before contact so the
# handle is caged rather than pushed away by a late thumb motion.
_AUTOMATIC_RECEIVER_GRASP_TARGET = np.asarray(
    [
        1.245, 0.8195, 0.15, 0.0225,
        1.5, -0.35, 1.066667, 0.2,
        1.5, -0.35, 1.066667, 0.2,
        1.5, 0.35, 1.066667, 0.2,
        0.216667, -0.35, -0.016667, 0.2,
    ],
    dtype=np.float64,
)
# A few reachable reset poses leave the little finger just outside the
# cylinder after the nominal close.  Preserve the nominal target first, then
# make this small, bounded secondary close only if no physical two-sided
# receiver grasp was admitted.  Successful nominal grasps never see it.
_AUTOMATIC_RECEIVER_FALLBACK_GRASP_TARGET = np.asarray(
    [
        1.254, 0.822, 0.15, 0.0244,
        1.5, -0.35, 1.066667, 0.2,
        1.5, -0.35, 1.066667, 0.2,
        1.5, 0.35, 1.066667, 0.2,
        0.216667, -0.35, -0.016667, 0.2,
    ],
    dtype=np.float64,
)
_AUTOMATIC_RECEIVER_FALLBACK_DELAY_STEPS = 8
_AUTOMATIC_RECEIVER_FALLBACK_CLOSE_STEPS = 12
_AUTOMATIC_RECEIVER_APPROACH_TARGET = np.asarray(
    [
        1.2, 0.8, 0.15, 0.0,
        1.5, -0.35, 1.066667, 0.2,
        1.5, -0.35, 1.066667, 0.2,
        1.5, 0.35, 1.066667, 0.2,
        1.5, 0.35, 1.066667, 0.2,
    ],
    dtype=np.float64,
)
# The live pinch is detected from contact forces, then held at that measured
# substep target.  A near-identical pinch may be normalized to this verified
# configuration to remove solver-order variation of at most 0.0083 rad.  A
# genuinely different physical pinch is retained through placement instead
# of being driven toward this template and twisting the narrow handle.  These
# remain robot joint targets and never manufacture contact or move the tool.
_AUTOMATIC_RECEIVER_PINCH_HOLD_TARGET = np.asarray(
    [
        1.2391843796, 0.8121295571, 0.1461130977, 0.0191546343,
        1.4999701977, -0.3500491083, 1.0666955709, 0.2000044137,
        1.4999707937, -0.3500840366, 1.0666904449, 0.2000036538,
        1.4999639988, 0.3499988914, 1.0666978359, 0.2000047117,
        0.2886103392, -0.3112403452, 0.0430791378, 0.2010030597,
    ],
    dtype=np.float64,
)
_AUTOMATIC_RECEIVER_PINCH_NORMALIZATION_MAX_RAD = 0.02
_AUTOMATIC_RECEIVER_PINCH_PROGRESS_SCALE = 1.20
# The lightest randomized screwdrivers react to the usual 20% measured-pinch
# overshoot with a single-finger angular impulse.  Preserve their first real
# pinch exactly; if it later clears after giver release, the robot-only stage-3
# regrasp below can reacquire it without violating the unchanged safety gate.
_AUTOMATIC_RECEIVER_LOW_INERTIA_MASS_KG = 0.150
_AUTOMATIC_RECEIVER_LOW_INERTIA_PINCH_PROGRESS_SCALE = 1.0
_AUTOMATIC_RECEIVER_STAGE3_CORRECTION_MAX_RAD = 0.02
_AUTOMATIC_RECEIVER_STAGE3_TARGET = np.asarray(
    [
        1.35, 0.85, 0.15, 0.03,
        1.49997, -0.35005, 1.06670, 0.20000,
        1.49997, -0.35008, 1.06669, 0.20000,
        1.49996, 0.35000, 1.06670, 0.20000,
        0.28861, -0.31124, 0.04308, 0.20100,
    ],
    dtype=np.float64,
)
_AUTOMATIC_RECEIVER_CLOSE_STEPS = 40
_AUTOMATIC_RECEIVER_STAGE3_CLOSE_STEPS = 60
_AUTOMATIC_RECEIVER_PREPLACEMENT_GRIP_STEPS = 60
_AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_GAIN = 0.5
_AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_MAX_M = 0.012
# Keep the full low-table Cartesian correction while contact is light, then
# taper it as finger load rises.  A fixed lower gain lets marginal pinches
# fall behind the path, while a fixed half-gain can keep increasing an already
# loaded grasp until the screwdriver crosses the unchanged safety boundary.
_AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_TAPER_START_N = 4.0
_AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_TAPER_END_N = 8.0
# A stage-3 recovery is detected while the fingers are still closing.  Freeze
# mostly at that measured contact frame, with only a small fraction of the
# intended close target retained as preload so the light tool does not slip.
_STAGE3_REGRASP_HOLD_TARGET_BLEND = 0.125
_AUTOMATIC_RECEIVER_RELEASE_TARGET = np.zeros(20, dtype=np.float64)


def _cap_norm(value: np.ndarray, limit: float) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= limit or magnitude <= 1.0e-12:
        return vector
    return vector * (float(limit) / magnitude)


def _cap_torque_preserving_axis_alignment(
    value: np.ndarray,
    tool_axis: np.ndarray,
    limit: float,
) -> np.ndarray:
    """Cap torque while reserving authority for safety-critical axis tilt."""

    vector = np.asarray(value, dtype=np.float64).reshape(3)
    axis = np.asarray(tool_axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    longitudinal = axis * float(np.dot(vector, axis))
    transverse = _cap_norm(vector - longitudinal, limit)
    remaining = math.sqrt(
        max(0.0, float(limit) ** 2 - float(np.dot(transverse, transverse)))
    )
    return transverse + _cap_norm(longitudinal, remaining)


def _grasp_command_max(hand: np.ndarray) -> float:
    """Return grasp closure without treating lateral spread as closure."""

    values = np.asarray(hand, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(values[_HAND_GRASP_INDICES])))


def _planar_world_half_extents(
    orientation_wxyz: np.ndarray,
    local_half_extents: np.ndarray,
) -> np.ndarray:
    """Return the XY half extents of a rotated local bounding box."""

    rotation = np.abs(
        _quat_wxyz_to_matrix(
            np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
        )[:2, :]
    )
    local = np.asarray(local_half_extents, dtype=np.float64).reshape(3)
    return rotation @ local


def _world_footprint_center_and_half_extents(
    pose: np.ndarray,
    local_center: np.ndarray,
    local_half_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the true XY center/AABB half extents of a rotated local box."""

    value = np.asarray(pose, dtype=np.float64).reshape(7)
    rotation = _quat_wxyz_to_matrix(value[3:7])
    center = value[:3] + rotation @ np.asarray(
        local_center, dtype=np.float64
    ).reshape(3)
    half_extents = np.abs(rotation[:2, :]) @ np.asarray(
        local_half_extents, dtype=np.float64
    ).reshape(3)
    return center[:2], half_extents


def _rotation_error_vector(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Axis-angle error rotating ``current`` onto ``target``."""

    relative = np.asarray(target, dtype=np.float64) @ np.asarray(
        current, dtype=np.float64
    ).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(math.acos(cosine))
    if angle <= 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    sine = float(math.sin(angle))
    if abs(sine) <= 1.0e-8:
        values, vectors = np.linalg.eigh(relative)
        axis = vectors[:, int(np.argmin(np.abs(values - 1.0)))]
    else:
        axis = np.asarray(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * sine)
    return axis * angle


def _axis_alignment_error_vector(
    target_axis: np.ndarray,
    current_axis: np.ndarray,
) -> np.ndarray:
    """Minimum rotation that aligns one direction without constraining roll."""

    target = np.asarray(target_axis, dtype=np.float64).reshape(3)
    current = np.asarray(current_axis, dtype=np.float64).reshape(3)
    target /= max(float(np.linalg.norm(target)), 1.0e-12)
    current /= max(float(np.linalg.norm(current)), 1.0e-12)
    cross = np.cross(current, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(current, target), -1.0, 1.0))
    if sine <= 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    return cross * (math.atan2(sine, cosine) / sine)


def _minimal_axis_aligned_rotation(
    target_axis: np.ndarray,
    current_rotation: np.ndarray,
) -> np.ndarray:
    """Align the tool axis by the shortest rotation and preserve its roll."""

    current = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    error = _axis_alignment_error_vector(target_axis, current[:, 0])
    angle = float(np.linalg.norm(error))
    if angle <= 1.0e-12:
        return current.copy()
    axis = error / angle
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    delta = (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    return delta @ current


def _bounded_axis_feedback_rotation(
    target_axis: np.ndarray,
    current_axis: np.ndarray,
    command_rotation: np.ndarray,
    *,
    gain: float,
    max_angle_rad: float,
) -> np.ndarray:
    """Apply a bounded live-axis correction to a wrist command."""

    command = np.asarray(command_rotation, dtype=np.float64).reshape(3, 3)
    error = _axis_alignment_error_vector(target_axis, current_axis)
    error_norm = float(np.linalg.norm(error))
    angle = min(error_norm * float(gain), float(max_angle_rad))
    if angle <= 1.0e-12:
        return command.copy()
    axis = error / max(error_norm, 1.0e-12)
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    delta = (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    return delta @ command


def _tool_frame_pose(
    tool_pose: np.ndarray,
    local_position: np.ndarray,
    local_orientation_wxyz: np.ndarray,
) -> np.ndarray:
    """Map one task-local palm waypoint through the current tool transform."""

    tool = np.asarray(tool_pose, dtype=np.float64).reshape(7)
    local = np.asarray(local_position, dtype=np.float64).reshape(3)
    local_orientation = np.asarray(
        local_orientation_wxyz, dtype=np.float64
    ).reshape(4)
    world_position = (
        _pose_vector_to_matrix(tool)
        @ np.asarray([*local, 1.0], dtype=np.float64)
    )[:3]
    world_orientation = _quat_mul_wxyz(tool[3:7], local_orientation)
    world_orientation /= max(float(np.linalg.norm(world_orientation)), 1.0e-12)
    return np.concatenate([world_position, world_orientation])


def _matrix_pose_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert one rigid transform to the task's position/wxyz convention."""

    value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return np.concatenate(
        [value[:3, 3], _matrix_to_quat_wxyz(value[:3, :3])]
    ).astype(np.float64)


def _owner_palm_pose(
    tool_pose: np.ndarray,
    hand_to_tool: np.ndarray,
) -> np.ndarray:
    """Solve the palm target that carries a contact-owned tool target."""

    tool = _pose_vector_to_matrix(np.asarray(tool_pose, dtype=np.float64).reshape(7))
    anchor = np.asarray(hand_to_tool, dtype=np.float64).reshape(4, 4)
    return _matrix_pose_vector(tool @ _invert_pose_matrix(anchor))


def _tool_vertical_support_depth(tool_quat_wxyz: np.ndarray) -> float:
    """Return root-to-lowest-surface depth for the compound flathead proxy."""

    down_projection = _quat_wxyz_to_matrix(
        np.asarray(tool_quat_wxyz, dtype=np.float64).reshape(4)
    )[2, :]
    lowest = float("inf")
    for center, half_extents in _TOOL_BOX_SUPPORTS:
        projected_half_extent = float(
            np.dot(np.abs(down_projection), half_extents)
        )
        lowest = min(
            lowest,
            float(np.dot(down_projection, center)) - projected_half_extent,
        )
    return -lowest


def _tool_table_gap_m(tool_pose: np.ndarray, zone_pose: np.ndarray) -> float:
    """Return the conservative physical gap above the AS019 tabletop."""

    tool = np.asarray(tool_pose, dtype=np.float64).reshape(7)
    zone = np.asarray(zone_pose, dtype=np.float64).reshape(7)
    table_top = float(zone[2]) - _MARKER_CENTER_ABOVE_TABLE_M
    return float(
        tool[2] - _tool_vertical_support_depth(tool[3:7]) - table_top
    )


def _stable_tool_quaternion(
    live_tool_pose: np.ndarray,
    roll_rad: float = 0.0,
) -> np.ndarray:
    """Return the nearest stable-facet root roll while preserving outward yaw."""

    live = np.asarray(live_tool_pose, dtype=np.float64).reshape(7)
    live_axis = _quat_wxyz_to_matrix(live[3:7])[:, 0].copy()
    live_axis[2] = 0.0
    live_axis /= max(float(np.linalg.norm(live_axis)), 1.0e-12)
    stable_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    stable_y = np.cross(stable_z, live_axis)
    stable_y /= max(float(np.linalg.norm(stable_y)), 1.0e-12)
    stable_rotation = np.column_stack(
        [live_axis, stable_y, np.cross(live_axis, stable_y)]
    )
    stable_quat = _matrix_to_quat_wxyz(stable_rotation)
    live_rotation = _quat_wxyz_to_matrix(live[3:7])
    relative = stable_rotation.T @ live_rotation
    live_roll = math.atan2(float(relative[2, 1]), float(relative[1, 1]))
    nearest_facet_roll = (
        round(live_roll / _TOOL_STABLE_FACET_STEP_RAD)
        * _TOOL_STABLE_FACET_STEP_RAD
    )
    target_roll = nearest_facet_roll + float(roll_rad)
    roll_quat = np.asarray(
        [
            math.cos(0.5 * target_roll),
            math.sin(0.5 * target_roll),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return _quat_mul_wxyz(stable_quat, roll_quat)


def _zone_tool_targets(
    live_tool_pose: np.ndarray,
    zone_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive outward-preserving hover and supported tool poses from zone bounds."""

    live = np.asarray(live_tool_pose, dtype=np.float64).reshape(7)
    zone = np.asarray(zone_pose, dtype=np.float64).reshape(7)
    # Preserve the contact-owned handover roll while moving into the zone.
    # The receiver finds the authored stable facet only after the tool is
    # table-supported; free-space roll correction selected an unstable DLS
    # branch and squeezed the cylindrical handle out of the side grasp.
    roll_quat = np.asarray(
        [
            math.cos(0.5 * _ZONE_PLACEMENT_ROLL_RAD),
            math.sin(0.5 * _ZONE_PLACEMENT_ROLL_RAD),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    placed_quat = _quat_mul_wxyz(live[3:7], roll_quat)
    vertical_support_depth = _tool_vertical_support_depth(placed_quat)
    placed = live.copy()
    zone_rotation = _quat_wxyz_to_matrix(zone[3:7])
    placement_offset_world = (
        zone_rotation @ _ZONE_PLACEMENT_CENTER_OFFSET_LOCAL
    )
    placed[:2] = zone[:2] + placement_offset_world[:2]
    placed[3:7] = placed_quat
    placed[2] = (
        float(zone[2])
        - _MARKER_CENTER_ABOVE_TABLE_M
        + vertical_support_depth
        + _ZONE_SUPPORT_CLEARANCE_M
    )
    hover = placed.copy()
    hover[2] += _ZONE_HOVER_CLEARANCE_M
    return hover, placed


def _tool_blunt_release_direction(tool_pose: np.ndarray) -> np.ndarray:
    """Return the level local -X direction, away from the sharp +X tip."""

    tool = np.asarray(tool_pose, dtype=np.float64).reshape(7)
    direction = -_quat_wxyz_to_matrix(tool[3:7])[:, 0]
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def _tool_normal_retreat_pose(
    live_tool_pose: np.ndarray,
    live_palm_pose: np.ndarray,
    zone_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the tool-normal retreat in the right-home/away-zone half-plane."""

    tool = np.asarray(live_tool_pose, dtype=np.float64).reshape(7)
    palm = np.asarray(live_palm_pose, dtype=np.float64).reshape(7)
    zone = np.asarray(zone_pose, dtype=np.float64).reshape(7)
    rotation = _quat_wxyz_to_matrix(tool[3:7])
    axis = rotation[:, 0]
    # The two possible radial sides cannot be selected from the touching palm:
    # v6 did so and chose the side that crossed the tool as the right arm
    # returned home.  Instead project the zone-to-right-home direction onto the
    # live tool-normal plane, then verify it remains in the right-home half-plane.
    zone_to_home = _AS019_RIGHT_HOME[:3] - zone[:3]
    normal = zone_to_home - axis * float(np.dot(zone_to_home, axis))
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1.0e-9:
        tool_to_home = _AS019_RIGHT_HOME[:3] - tool[:3]
        normal = tool_to_home - axis * float(np.dot(tool_to_home, axis))
        magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1.0e-9:
        normal = rotation[:, 1].copy()
    else:
        normal /= magnitude
    home_half_plane = _AS019_RIGHT_HOME[:3] - tool[:3]
    home_half_plane -= axis * float(np.dot(home_half_plane, axis))
    if float(np.dot(normal, home_half_plane)) < 0.0:
        normal = -normal
    clear = palm.copy()
    clear[:3] += normal * _RIGHT_RETREAT_DISTANCE_M
    return clear, normal


class AS019ScrewdriverHandoverEnv(BimanualAssemblyIsaacLabEnv):
    """Ordered right-to-left screwdriver contact and zone-landing task."""

    runtime_family = "isaac_sim_5_1"
    physics_engine = "PhysX"
    task_name = TASK_NAME
    task_id = TASK_ID
    # Explicitly expose the native controller defaults used by this standalone
    # environment; these modes were previously read with getattr(..., False).

    def __init__(
        self,
        seed: int | None = None,
        control_dt: float | None = None,
        physics_dt: float = 1.0 / 120.0,
        time_limit: float = 120.0,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        hand_type: str = "wuji1",
        arm_type: str = "rokae",
        launch_isaac: bool | None = None,
        headless: bool | None = None,
        render_resolution: tuple[int, int] = _DEFAULT_RENDER_SIZE,
        hz: int = 30,
        isaac_device: str | None = None,
        physics_backend: str = "physx",
        **kwargs: Any,
    ):
        if str(physics_backend).strip().lower() != "physx":
            raise ValueError("B320-AS019 is Isaac Sim 5.1 PhysX-only; Newton is forbidden.")
        if str(hand_type).lower() != "wuji1" or str(arm_type).lower() != "rokae":
            raise ValueError("B320-AS019 requires dual ROKAE arms with Wuji hands.")
        self._randomize_dynamics = bool(randomize_dynamics)
        self._as019_contact_views: dict[str, Any] = {}
        self._as019_contact_filter_paths: dict[str, tuple[str, ...]] = {}
        self._as019_owner_contact_records: dict[str, list[dict[str, Any]]] = {}
        self._as019_right_per_finger_views: dict[str, Any] = {}
        self._as019_per_finger_views: dict[str, Any] = {}
        self._as019_table_safety_views: dict[str, Any] = {}
        self._as019_table_safety_filter_paths: dict[str, tuple[str, ...]] = {}
        self._as019_table_safety_paths: dict[str, Any] = {}
        self._as019_receiver_pinch_frozen = False
        self._as019_receiver_pinch_target: Any | None = None
        self._as019_receiver_pinch_freeze: dict[str, Any] = {}
        self._as019_contact_available = False
        self._as019_contact_query_valid = False
        self._as019_contact_query_error = "contact_views_not_initialized"
        self._as019_table_safety_available = False
        self._as019_table_safety_query_valid = False
        self._as019_table_safety_query_error = (
            "table_safety_views_not_initialized"
        )
        self._as019_assist_anchor: dict[str, dict[str, Any]] = {}
        self._as019_stage = 0
        self._as019_stage_name = "await_right_contact"
        self._as019_failure_reason = ""
        self._as019_table_safety_guard_armed = False
        self._as019_table_safety_failure_candidate = ""
        self._as019_table_safety_failure_counter = 0
        self._as019_table_safety_guard_last_step = -1
        self._as019_inward_orientation_failure_counter = 0
        self._as019_inward_orientation_guard_last_step = -1
        self._as019_success = False
        self._as019_stability_counter = 0
        self._as019_progress_reset_candidate_counter = 0
        self._as019_progress_reset_count = 0
        self._as019_last_progress_reset_reason = ""
        self._as019_receiver_hold_counter = 0
        self._as019_receiver_hold_complete_step = -1
        self._as019_stage2_left_contact_active = False
        self._as019_right_release_counter = 0
        self._as019_substep_latches: dict[str, bool] = {}
        self._as019_event_steps = np.full(5, -1, dtype=np.int64)
        self._as019_last_metrics: dict[str, Any] = {}
        self._as019_mass_kg = 0.18
        self._as019_friction = 0.68
        self._as019_automatic_phase = 0
        self._as019_automatic_acquire_gate_required = False
        self._as019_automatic_acquire_gate_open = False
        self._as019_automatic_arrival_gate: dict[str, Any] = {}
        self._as019_low_y_receiver_branch = False
        self._as019_high_handover_assist_tier = False
        self._as019_receiver_retry_assist_tier = False
        self._as019_placement_assist_enabled = False
        self._as019_placement_motion_started = False
        self._as019_release_support_enabled = False
        self._as019_wrench_active = False
        self._as019_active_wrench_owners: tuple[str, ...] = ()
        self._as019_active_wrench_gate_snapshot: dict[str, Any] = {}
        self._as019_slider_orientation_latched = False
        self._as019_slider_orientation_latched_step = -1
        self._as019_slider_orientation_active_owners: tuple[str, ...] = ()
        self._as019_slider_orientation_grace_owners: tuple[str, ...] = ()
        self._as019_slider_owner_last_live_step: dict[str, int] = {}
        self._as019_slider_hand_orientation_locks: dict[str, np.ndarray] = {}
        # Monotonic within one reset.  Wrench telemetry records the exact
        # cached-contact generation consumed by each pre-physics-substep
        # decision so post-step validators never have to combine two samples.
        self._as019_metrics_generation = 0
        self._as019_release_frame_trace: list[dict[str, Any]] = []
        self._as019_release_trace_last_step = -1
        self._as019_release_trace_dropped_frames = 0
        self._as019_release_event_steps = np.full(4, -1, dtype=np.int64)
        self._as019_last_policy_action = np.zeros(54, dtype=np.float32)
        self._as019_last_policy_action_step = -1
        self._as019_control_trace_enabled = (
            str(os.environ.get(_CONTROL_TRACE_ENV, "")).strip().lower()
            in _CONTROL_TRACE_TRUE_VALUES
        )
        self._as019_arm_control_trace: dict[str, dict[str, Any]] = {
            "right": {},
            "left": {},
        }
        self._as019_arm_branch_guard_count = 0
        self._as019_reset_attempts = 1
        self._as019_reset_serial = 0
        scene_contract = _as019_taskdev_module("scene_contract")
        self._as019_outward_reference_rotation = _quat_wxyz_to_matrix(
            np.asarray(
                scene_contract.SCREWDRIVER_ORIENTATION_WXYZ,
                dtype=np.float64,
            )
        )
        # The shared bimanual runtime only owns overlay cleanup.  AS019 is a
        # custom native-USD environment, so it owns the live panel lifecycle
        # locally instead of inheriting the ABC task runtime just for its UI.
        self._success_condition_window: Any | None = None
        self._success_condition_frame: Any | None = None
        self._success_condition_labels: dict[str, Any] = {}
        self._success_condition_text_labels: dict[str, Any] = {}
        self._task_status_label: Any | None = None
        self._success_condition_overlay_failed = False
        self.task_name = TASK_NAME
        if control_dt is None:
            control_dt = 1.0 / max(1, int(hz))
        super().__init__(
            task_name=TASK_NAME,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_mode=render_mode,
            image_obs=image_obs,
            randomize=randomize,
            hand_type=hand_type,
            arm_type=arm_type,
            launch_isaac=launch_isaac,
            headless=headless,
            render_resolution=render_resolution,
            hz=hz,
            isaac_device=isaac_device,
            physics_backend="physx",
            **kwargs,
        )

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        values = super()._in_process_scene_rebuild_kwargs(seed)
        values["randomize_dynamics"] = bool(self._randomize_dynamics)
        return values

    def _apply_reset_visual_randomization(self) -> None:
        """Use the same tabletop material path as ordinary ABC/B320 tasks."""

        if not self.randomize:
            self._visual_randomization_seed = None
            return
        self._visual_random_state_for_reset()
        if not self._launch_isaac:
            return
        from .default_appearance import refresh_environment
        refresh_environment(self)

    def _make_native_scene_cfg(self, *, num_envs: int, env_spacing: float):
        scene_runtime = _as019_taskdev_module("scene_runtime")

        if self._simulation_app is None:
            raise RuntimeError("AS019 asset conversion requires the scheduler-owned Isaac app.")
        runtime_usd, _metadata = scene_runtime.ensure_converted_screwdriver(
            self._simulation_app
        )
        cfg = scene_runtime.make_scene_cfg(
            runtime_usd,
            include_cameras=(
                self.image_obs and not self._deferred_camera_render
            ),
            include_wrist_cameras=(
                self.image_obs and not self._deferred_camera_render
            ),
        )
        cfg.num_envs = int(num_envs)
        cfg.env_spacing = float(env_spacing)
        return cfg

    def _native_task_contact_suffixes(self) -> tuple[str, ...]:
        return ("/Screwdriver",)

    def _build_observation_space(self) -> spaces.Dict:
        state = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
                "screwdriver_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "tool_zone_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "screwdriver_velocity": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
                "handover_metrics": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float64),
                "handover_stage": spaces.Box(0, 5, shape=(1,), dtype=np.float64),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )
        result: dict[str, Any] = {
            "state": state,
            "as019_hidden_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(len(AS019_HIDDEN_STATE_FIELDS),),
                dtype=np.float64,
            ),
        }
        if self.image_obs:
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            result["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(
                        0, 255, shape=(wrist_height, wrist_width, 3), dtype=np.uint8
                    ),
                    "wrist_right": spaces.Box(
                        0, 255, shape=(wrist_height, wrist_width, 3), dtype=np.uint8
                    ),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0, 255, shape=self._image_shape, dtype=np.uint8
                    ),
                }
            )
        return spaces.Dict(result)

    def _setup_isaaclab(self) -> None:
        # Base construction seeds its assembly peg/socket placeholders before
        # invoking this override.  Replace them before the first PhysX reset so
        # the exact screwdriver never visits an assembly-only pose.
        scene_contract = _as019_taskdev_module("scene_contract")

        self._peg_ori_pose = np.asarray(
            (
                *scene_contract.SCREWDRIVER_POSITION_M,
                *scene_contract.SCREWDRIVER_ORIENTATION_WXYZ,
            ),
            dtype=np.float64,
        )
        self._socket_ori_pose = np.asarray(
            (
                *scene_contract.MARKER_POSITION_M,
                *scene_contract.MARKER_ORIENTATION_WXYZ,
            ),
            dtype=np.float64,
        )
        super()._setup_isaaclab()
        self._setup_as019_contact_views()
        self._handson_pre_physics_substep_callback = self._as019_pre_substep
        self._handson_post_physics_substep_callback = self._as019_post_substep
        self._init_success_condition_overlay()
        self._update_success_condition_overlay()

    def reset(self, seed: int | None = None, **kwargs: Any):
        result = super().reset(seed=seed, **kwargs)
        self._update_success_condition_overlay()
        return result

    @staticmethod
    def _as019_control_vector(value: Any) -> np.ndarray:
        """Copy one tensor/array control row into CPU float64 storage."""

        viewed = _torch_view(value)
        if hasattr(viewed, "detach"):
            viewed = viewed.detach()
        if hasattr(viewed, "cpu"):
            viewed = viewed.cpu()
        if hasattr(viewed, "numpy"):
            viewed = viewed.numpy()
        result = np.asarray(viewed, dtype=np.float64)
        if result.ndim == 2 and result.shape[0] == 1:
            result = result[0]
        return result.reshape(-1).copy()

    def _compute_arm_joint_target(
        self,
        side: str,
        robot: Any,
        pose: np.ndarray,
        commanded: bool,
    ):
        """Cache the request, admitted pose, IK status, and raw arm target."""

        # AS019's held-tool safety contract requires the Haorui branch-aware
        # solver.  The shared runtime historically disabled Haorui after an
        # initialization error and silently fell through to DLS, which can
        # select the same discontinuous branch that launched the screwdriver.
        # Fail closed for this task instead of substituting an unverified IK
        # backend; the caller will hold the last applied arm target.
        if (
            commanded
            and self._arm_ik_backend == "haorui"
            and not bool(getattr(self, "_haorui_ik_enabled", False))
        ):
            message = (
                "AS019 Haorui IK unavailable; holding the last arm target "
                "instead of falling back to DLS"
            )
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None, "haorui_unavailable"

        if not self._as019_control_trace_enabled:
            return super()._compute_arm_joint_target(
                side,
                robot,
                pose,
                commanded,
            )

        requested_pose = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        current_matrix = self._robot_body_pose_matrix(
            robot,
            self._ee_body_indices[side],
        )
        current_pose = np.concatenate(
            [
                current_matrix[:3, 3],
                _matrix_to_quat_wxyz(current_matrix[:3, :3]),
            ]
        )
        arm_target, status = super()._compute_arm_joint_target(
            side,
            robot,
            pose,
            commanded,
        )
        filter_state = self._arm_pose_filters.get(side)
        admitted_pose = None
        if isinstance(filter_state, dict):
            try:
                admitted_pose = np.concatenate(
                    [
                        np.asarray(filter_state["pos"], dtype=np.float64).reshape(3),
                        np.asarray(filter_state["quat"], dtype=np.float64).reshape(4),
                    ]
                )
            except Exception:
                admitted_pose = None
        trace: dict[str, Any] = {
            "env_step": int(self.env_step),
            "policy_action_step": int(self._as019_last_policy_action_step),
            "commanded": bool(commanded),
            "status": str(status),
            "requested_pose_wxyz": requested_pose,
            "current_pose_before_ik_wxyz": current_pose,
            "admitted_pose_wxyz": admitted_pose,
            "pre_joint_limit_target_7d_rad": None,
            "previous_applied_target_7d_rad": None,
            "post_joint_limit_target_7d_rad": None,
        }
        if arm_target is not None:
            try:
                trace["pre_joint_limit_target_7d_rad"] = (
                    self._as019_control_vector(arm_target)
                )
            except Exception as exc:
                trace["pre_joint_limit_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        self._as019_arm_control_trace[side] = trace
        return arm_target, status

    def _slider_live_owner_sides(
        self, metrics: dict[str, Any]
    ) -> tuple[str, ...]:
        """Return contact owners eligible for the translation-only guide."""

        if (
            self._as019_failure_reason
            or self._as019_stage < 1
            or self._as019_stage >= 5
        ):
            return ()
        owners: list[str] = []
        if (
            self._as019_stage <= 2
            and "right" in self._as019_assist_anchor
            and bool(metrics.get("right_tail_physical", False))
            and _grasp_command_max(self._right_hand) > _HAND_OPEN_THRESHOLD
        ):
            owners.append("right")
        if (
            "left" in self._as019_assist_anchor
            and bool(metrics.get("contact_query_valid", False))
            and bool(metrics.get("left_middle_physical", False))
            and _grasp_command_max(self._left_hand) >= _HAND_CLOSE_THRESHOLD
        ):
            owners.append("left")
        return tuple(owners)

    def _update_slider_orientation_guide(
        self, metrics: dict[str, Any]
    ) -> None:
        """Latch the guide and measured wrist attitudes from physical owners."""

        live_owners = self._slider_live_owner_sides(metrics)
        for side in live_owners:
            self._as019_slider_owner_last_live_step[side] = int(self.env_step)
        lifted_clear_of_table = bool(
            _tool_table_gap_m(
                self._peg_ori_pose,
                self._socket_ori_pose,
            )
            >= _SLIDER_LATCH_MIN_TABLE_GAP_M
        )
        if (
            not self._as019_slider_orientation_latched
            and live_owners
            and lifted_clear_of_table
        ):
            self._as019_slider_orientation_latched = True
            self._as019_slider_orientation_latched_step = int(self.env_step)
        if not self._as019_slider_orientation_latched:
            self._as019_slider_hand_orientation_locks.clear()
            return
        retained_owners = set(live_owners)
        retained_owners.update(
            self._slider_orientation_grace_owner_sides(metrics)
        )
        for side in ("right", "left"):
            if side not in retained_owners:
                # A new grasp is allowed to establish a new measured wrist
                # attitude.  Keep the old lock only for the bounded contact
                # grace; an expired gap must not freeze a detached hand.
                self._as019_slider_hand_orientation_locks.pop(side, None)
                continue
            if (
                side not in live_owners
                or side in self._as019_slider_hand_orientation_locks
            ):
                continue
            quaternion = _matrix_to_quat_wxyz(
                self._palm_matrix(side)[:3, :3]
            )
            quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
            self._as019_slider_hand_orientation_locks[side] = quaternion

    def _slider_orientation_grace_owner_sides(
        self, metrics: dict[str, Any]
    ) -> tuple[str, ...]:
        """Return recently physical owners eligible for orientation-only help."""

        if (
            not self._as019_slider_orientation_latched
            or self._as019_failure_reason
            or self._as019_stage < 1
            or self._as019_stage >= 5
        ):
            return ()
        live_owners = set(self._slider_live_owner_sides(metrics))
        grace_owners: list[str] = []
        for side in ("right", "left"):
            if side in live_owners or side not in self._as019_assist_anchor:
                continue
            if side == "right":
                if self._as019_stage > 2:
                    continue
                hand = self._right_hand
                close_threshold = _HAND_OPEN_THRESHOLD
            else:
                hand = self._left_hand
                close_threshold = _HAND_CLOSE_THRESHOLD
            if _grasp_command_max(hand) <= close_threshold:
                continue
            last_live_step = self._as019_slider_owner_last_live_step.get(side)
            if last_live_step is None:
                continue
            elapsed = int(self.env_step) - int(last_live_step)
            if 0 <= elapsed <= _SLIDER_CONTACT_GRACE_STEPS:
                grace_owners.append(side)
        return tuple(grace_owners)

    def _slider_wrist_lock_active(self, side: str) -> bool:
        """Return whether a live owner must ignore wrist-rotation commands."""

        return bool(
            self._as019_slider_orientation_latched
            and side in self._as019_slider_hand_orientation_locks
            and (
                side in self._slider_live_owner_sides(
                    self._as019_last_metrics
                )
                or side in self._slider_orientation_grace_owner_sides(
                    self._as019_last_metrics
                )
            )
        )

    def _limit_arm_pose_target(
        self,
        side: str,
        robot: Any,
        pose: np.ndarray,
    ) -> np.ndarray:
        """Apply AS019's physical wrist lock through the shared limiter."""

        requested = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        if self._slider_wrist_lock_active(side):
            requested[3:7] = self._as019_slider_hand_orientation_locks[
                side
            ].copy()
        return np.asarray(
            super()._limit_arm_pose_target(side, robot, requested),
            dtype=np.float64,
        ).reshape(7)

    def _physx_contact_arm_target_limits_enabled(self) -> bool:
        """Keep both arm target limiters active for every AS019 PhysX frame.

        The shared runtime enables this path only for D03 by default.  AS019
        uses the same PhysX implicit servo but has a held screwdriver whose
        safety depends on continuity even if Haorui temporarily falls back to
        DLS or a manifest disables generic limit environment variables.
        """

        return self._physics_backend == "physx"

    def _haorui_ik_first_success(self, side: str) -> bool:
        """Score every Haorui seed so a held screwdriver stays on one branch.

        The first seed is normally allowed to terminate multi-start IK for
        speed.  Near the screwdriver handover pose, however, Haorui can report
        a numerically valid elbow/wrist branch that is several radians from
        the previously applied target even though the Cartesian command moved
        only a few millimetres.  Evaluating all qualified seeds lets the
        solver's reference-q score select the continuous branch; the local
        0.03-rad limiter remains as the final PhysX safety backstop.
        """

        # Before the right hand has physically lifted the screwdriver, the
        # command path is ordinary free-space motion.  Keep the fast
        # first-success search there so teleoperation remains responsive; the
        # branch-reject guard below still protects a discontinuous result.
        # Once the ordered state machine or contact ownership says that a
        # hand is carrying the tool, evaluate every seed to prefer the
        # reference-q-continuous branch.  This avoids paying the multi-start
        # cost for every idle/reset frame while retaining continuity during
        # the actual handover.
        del side
        return not bool(
            getattr(self, "_as019_stage", 0) >= 1
            or getattr(self, "_as019_assist_anchor", None)
            or getattr(self, "_as019_active_wrench_owners", ())
        )

    def _limit_arm_joint_target(self, side: str, robot: Any, arm_target: Any):
        """Apply an AS019 branch-jump guard after the shared joint limiter.

        The shared limiter is configurable for other tasks and can be disabled
        by a replay manifest.  AS019 must never pass a whole-arm IK branch
        switch to an implicit PhysX servo, so reject a distant branch outright
        and enforce one small per-joint step here regardless of those global
        knobs.  Non-finite solver output and multi-radian branch switches are
        held at the previously applied target; nearby finite output is clipped
        to the same physical-palm continuity reference.
        """

        arm_joint_ids = self._arm_joint_ids[side]
        previous = self._joint_targets.get(side)
        if previous is None:
            previous = _torch_view(robot.data.joint_pos)
        previous_q = _torch_view(previous)[:, arm_joint_ids]
        # Preserve the solver result before the shared limiter.  The generic
        # PhysX limiter normally clips a 3.95-rad branch switch to 0.03 rad;
        # inspecting only that clipped value would make the AS019 guard walk
        # toward the wrong branch over many frames.
        raw_result = _torch_view(arm_target)
        try:
            raw_branch_jump = float(
                self._torch.max(
                    self._torch.abs(raw_result - previous_q)
                ).item()
            )
        except Exception:
            raw_branch_jump = float("inf")
        result = super()._limit_arm_joint_target(side, robot, arm_target)
        result = _torch_view(result)
        branch_delta = result - previous_q
        try:
            finite = bool(self._torch.isfinite(result).all().item())
        except Exception:
            finite = False
        if not finite:
            result = previous_q.clone()
            branch_jump = float("inf")
        else:
            try:
                branch_jump = float(
                    self._torch.max(self._torch.abs(branch_delta)).item()
                )
            except Exception:
                branch_jump = 0.0
            branch_rejected = bool(
                raw_branch_jump > _AS019_ARM_BRANCH_REJECT_RAD
            )
            if branch_rejected:
                # Keep the last physically applied branch until the IK solver
                # returns a nearby solution.  Clipping a distant branch to the
                # normal slew limit would still walk the implicit PhysX servo
                # into the wrong elbow/wrist posture and eject the held tool.
                result = previous_q.clone()
            else:
                result = previous_q + self._torch.clamp(
                    branch_delta,
                    -_AS019_ARM_MAX_JOINT_STEP_RAD,
                    _AS019_ARM_MAX_JOINT_STEP_RAD,
                )
        if (
            not finite
            or raw_branch_jump > _AS019_ARM_BRANCH_REJECT_RAD
            or branch_jump > _AS019_ARM_MAX_JOINT_STEP_RAD
        ):
            self._as019_arm_branch_guard_count += 1
            reason = (
                "non-finite"
                if not finite
                else (
                    f"branch={raw_branch_jump:.3f}rad"
                    if raw_branch_jump > _AS019_ARM_BRANCH_REJECT_RAD
                    else f"step={branch_jump:.3f}rad"
                )
            )
            message = (
                f"{side} AS019 arm IK branch jump guarded ({reason}); "
                f"step limited to {_AS019_ARM_MAX_JOINT_STEP_RAD:.3f}rad"
            )
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
        if not self._as019_control_trace_enabled:
            return result
        trace = self._as019_arm_control_trace.setdefault(side, {})
        try:
            trace["previous_applied_target_7d_rad"] = (
                self._as019_control_vector(previous_q)
            )
            trace["post_joint_limit_target_7d_rad"] = (
                self._as019_control_vector(result)
            )
        except Exception as exc:
            trace["post_joint_limit_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def step_with_current_joint_targets(self, *, start_time: float | None = None):
        # Both live actions and exact-drive replay pass through this low-level
        # entry point.  Refresh after the base method has evaluated success and
        # failure so the terminal frame never displays one-step-old status.
        result = super().step_with_current_joint_targets(start_time=start_time)
        self._update_success_condition_overlay()
        return result

    def close(self) -> None:
        self._release_scene_overlay_for_rebuild()
        super().close()

    def _setup_as019_contact_views(self) -> None:
        self._as019_contact_views = {}
        self._as019_contact_filter_paths = {}
        self._as019_owner_contact_records = {}
        self._as019_right_per_finger_views = {}
        self._as019_per_finger_views = {}
        self._as019_table_safety_views = {}
        self._as019_table_safety_filter_paths = {}
        self._as019_table_safety_paths = {}
        self._as019_contact_available = False
        self._as019_table_safety_available = False
        if self._stage is None or self._scene is None:
            return
        physics_view = self._scene["right_robot"]._physics_sim_view
        tool_path = "/World/envs/env_0/Screwdriver"
        table_path = "/World/envs/env_0/Table"
        marker_path = "/World/envs/env_0/SafeOrientationMarker"
        tool_prim = self._stage.GetPrimAtPath(tool_path)
        table_prim = self._stage.GetPrimAtPath(table_path)
        if (
            physics_view is None
            or not tool_prim.IsValid()
            or not table_prim.IsValid()
        ):
            self._isaac_warnings.append("AS019 exact contact views are unavailable")
            return

        from pxr import Usd, UsdPhysics

        def hand_paths(side: str, *, palm_only: bool = False) -> list[str]:
            root = self._stage.GetPrimAtPath(
                f"/World/envs/env_0/{side.capitalize()}Robot"
            )
            if not root.IsValid():
                return []
            palm_name = f"{side}_palm_link"
            finger_prefix = f"{side}_finger"
            return [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and (
                    prim.GetName() == palm_name
                    if palm_only
                    else prim.GetName().startswith(finger_prefix)
                )
            ]

        definitions = {
            "right_fingers": hand_paths("right"),
            "left_fingers": hand_paths("left"),
            "left_palm": hand_paths("left", palm_only=True),
        }
        for name, filters in definitions.items():
            if not filters:
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    tool_path,
                    filter_patterns=filters,
                    max_contact_data_count=max(128, len(filters) * 16),
                )
                if int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                    self._as019_contact_views[name] = view
                    self._as019_contact_filter_paths[name] = tuple(filters)
            except Exception as exc:
                self._isaac_warnings.append(f"AS019 {name} contact view failed: {exc}")

        def collision_enabled(prim: Any) -> bool:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                return False
            value = UsdPhysics.CollisionAPI(
                prim
            ).GetCollisionEnabledAttr().Get()
            return value is not False

        def rigid_owner_path(collision_prim: Any) -> str | None:
            owner = collision_prim
            while owner.IsValid() and not owner.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                owner = owner.GetParent()
            if not owner.IsValid():
                return None
            enabled = UsdPhysics.RigidBodyAPI(
                owner
            ).GetRigidBodyEnabledAttr().Get()
            return str(owner.GetPath()) if enabled is not False else None

        all_robot_collision_owners: set[str] = set()
        arm_collision_owners: set[str] = set()
        hand_collision_owners: set[str] = set()
        for side in ("right", "left"):
            robot_root = self._stage.GetPrimAtPath(
                f"/World/envs/env_0/{side.capitalize()}Robot"
            )
            if not robot_root.IsValid():
                continue
            for prim in Usd.PrimRange(
                robot_root, Usd.TraverseInstanceProxies()
            ):
                if not collision_enabled(prim):
                    continue
                owner_path = rigid_owner_path(prim)
                if owner_path is None:
                    continue
                all_robot_collision_owners.add(owner_path)
                role = _as019_robot_table_contact_role(owner_path)
                if role == "hand":
                    hand_collision_owners.add(owner_path)
                else:
                    arm_collision_owners.add(owner_path)

        arm_paths = tuple(sorted(arm_collision_owners))
        hand_paths_exact = tuple(sorted(hand_collision_owners))
        partition_complete = bool(
            arm_paths
            and hand_paths_exact
            and arm_collision_owners.isdisjoint(hand_collision_owners)
            and (
                arm_collision_owners | hand_collision_owners
            ) == all_robot_collision_owners
        )
        safety_definitions = {
            "tool_table": (tool_path, (table_path,)),
            "table_arm": (table_path, arm_paths),
            "table_hand": (table_path, hand_paths_exact),
        }
        for name, (sensor_pattern, filters) in safety_definitions.items():
            if not filters:
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    sensor_pattern,
                    filter_patterns=list(filters),
                    max_contact_data_count=max(256, len(filters) * 32),
                )
                if (
                    int(view.sensor_count) == 1
                    and int(view.filter_count) == len(filters)
                ):
                    self._as019_table_safety_views[name] = view
                    self._as019_table_safety_filter_paths[name] = filters
                else:
                    self._isaac_warnings.append(
                        f"AS019 {name} contact view cardinality mismatch"
                    )
            except Exception as exc:
                self._isaac_warnings.append(
                    f"AS019 {name} contact view failed: {exc}"
                )
        safety_filters = {
            path
            for filters in self._as019_table_safety_filter_paths.values()
            for path in filters
        }
        marker_excluded = marker_path not in safety_filters
        self._as019_table_safety_paths = {
            "tool": tool_path,
            "table": table_path,
            "arm_collision_owners": arm_paths,
            "hand_collision_owners": hand_paths_exact,
            "all_robot_collision_owners": tuple(
                sorted(all_robot_collision_owners)
            ),
            "marker": marker_path,
            "marker_excluded": marker_excluded,
            "robot_partition_complete": partition_complete,
        }
        self._as019_contact_available = bool(
            len(self._as019_contact_views) == 3
            and set(self._as019_contact_filter_paths)
            == {"right_fingers", "left_fingers", "left_palm"}
        )
        if not self._as019_contact_available:
            self._isaac_warnings.append(
                "AS019 contact ownership is incomplete; success will fail closed"
            )
        self._as019_table_safety_available = bool(
            {"tool_table", "table_hand"}.issubset(
                self._as019_table_safety_views
            )
            and partition_complete
            and marker_excluded
        )
        if not self._as019_table_safety_available:
            self._isaac_warnings.append(
                "AS019 table safety views are incomplete; task will fail closed"
            )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        scene_contract = _as019_taskdev_module("scene_contract")
        marker_orientation = scene_contract.MARKER_ORIENTATION_WXYZ
        marker_position = scene_contract.MARKER_POSITION_M
        marker_randomization_xy = scene_contract.MARKER_RANDOMIZATION_XY_M
        screwdriver_friction_range = scene_contract.SCREWDRIVER_FRICTION_RANGE
        screwdriver_mass_kg = scene_contract.SCREWDRIVER_MASS_KG
        screwdriver_mass_multiplier_range = (
            scene_contract.SCREWDRIVER_MASS_MULTIPLIER_RANGE
        )
        screwdriver_orientation = scene_contract.SCREWDRIVER_ORIENTATION_WXYZ
        screwdriver_position = scene_contract.SCREWDRIVER_POSITION_M
        screwdriver_randomization_xy = (
            scene_contract.SCREWDRIVER_RANDOMIZATION_XY_M
        )
        screwdriver_randomization_yaw = (
            scene_contract.SCREWDRIVER_RANDOMIZATION_YAW_RAD
        )

        self.delta_h = np.zeros(1, dtype=np.float64)
        self._right_tcp_pose = _AS019_RIGHT_HOME.copy()
        self._left_tcp_pose = _AS019_LEFT_HOME.copy()
        self._right_hand.fill(0.0)
        self._left_hand.fill(0.0)
        tool = np.asarray(
            (*screwdriver_position, *screwdriver_orientation), dtype=np.float64
        )
        reset_yaw_quat = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        zone = np.asarray(
            (*marker_position, *marker_orientation), dtype=np.float64
        )
        self._as019_reset_attempts = 1
        if self.randomize:
            canonical_tool = tool.copy()
            canonical_zone = zone.copy()
            accepted = False
            for attempt in range(1, _RESET_MAX_SAMPLE_ATTEMPTS + 1):
                candidate_tool = canonical_tool.copy()
                candidate_zone = canonical_zone.copy()
                candidate_tool[:2] += self._random.uniform(
                    -screwdriver_randomization_xy,
                    screwdriver_randomization_xy,
                    size=2,
                )
                yaw = float(
                    self._random.uniform(
                        -screwdriver_randomization_yaw,
                        screwdriver_randomization_yaw,
                    )
                )
                yaw_quat = np.asarray(
                    [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
                    dtype=np.float64,
                )
                candidate_tool[3:7] = _quat_mul_wxyz(
                    yaw_quat, candidate_tool[3:7]
                )
                candidate_zone[:2] += self._random.uniform(
                    -marker_randomization_xy,
                    marker_randomization_xy,
                    size=2,
                )
                tool_world_center, tool_world_half_extents = (
                    _world_footprint_center_and_half_extents(
                        candidate_tool,
                        _TOOL_LOCAL_BOUNDS_CENTER,
                        _TOOL_HALF_EXTENTS,
                    )
                )
                marker_world_half_extents = _planar_world_half_extents(
                    candidate_zone[3:7], _MARKER_HALF_EXTENTS
                )
                supported = bool(
                    -0.55 + tool_world_half_extents[0]
                    <= tool_world_center[0]
                    <= 0.25 - tool_world_half_extents[0]
                    and -0.80 + tool_world_half_extents[1]
                    <= tool_world_center[1]
                    <= 0.80 - tool_world_half_extents[1]
                )
                marker_supported = bool(
                    -0.55 + marker_world_half_extents[0]
                    <= candidate_zone[0]
                    <= 0.25 - marker_world_half_extents[0]
                    and -0.80 + marker_world_half_extents[1]
                    <= candidate_zone[1]
                    <= 0.80 - marker_world_half_extents[1]
                )
                visible_zone_gap = float(
                    tool_world_center[0]
                    - tool_world_half_extents[0]
                    - (
                        candidate_zone[0]
                        + marker_world_half_extents[0]
                    )
                )
                palm_clearance = min(
                    float(
                        np.linalg.norm(candidate_tool[:2] - _AS019_RIGHT_HOME[:2])
                    ),
                    float(
                        np.linalg.norm(candidate_tool[:2] - _AS019_LEFT_HOME[:2])
                    ),
                )
                reachable = bool(
                    -0.48
                    <= candidate_tool[0]
                    <= _RESET_CONTROLLER_REACHABLE_TOOL_X_MAX_M
                    and candidate_tool[1]
                    >= _RESET_CONTROLLER_REACHABLE_TOOL_Y_MIN_M
                    and candidate_tool[1]
                    <= _RESET_CONTROLLER_REACHABLE_TOOL_Y_MAX_M
                    and (
                        candidate_tool[1]
                        > _HIGH_HANDOVER_ASSIST_TOOL_Y_THRESHOLD_M
                        or candidate_tool[0]
                        <= _RESET_CONTROLLER_REACHABLE_LOW_Y_TOOL_X_MAX_M
                    )
                    and abs(yaw)
                    <= _RESET_CONTROLLER_REACHABLE_ABS_YAW_MAX_RAD
                )
                if (
                    supported
                    and marker_supported
                    and visible_zone_gap >= _RESET_TOOL_ZONE_VISIBLE_GAP_M
                    and palm_clearance >= 0.070
                    and reachable
                ):
                    tool = candidate_tool
                    zone = candidate_zone
                    reset_yaw_quat = yaw_quat
                    accepted = True
                    self._as019_reset_attempts = attempt
                    break
            if not accepted:
                tool = canonical_tool
                zone = canonical_zone
                self._as019_reset_attempts = _RESET_MAX_SAMPLE_ATTEMPTS
            self._as019_mass_kg = float(
                screwdriver_mass_kg
                * self._random.uniform(*screwdriver_mass_multiplier_range)
            )
            self._as019_friction = float(
                self._random.uniform(*screwdriver_friction_range)
            )
        else:
            self._as019_mass_kg = float(screwdriver_mass_kg)
            self._as019_friction = 0.68
        self._peg_ori_pose = tool
        self._socket_ori_pose = zone
        self._as019_low_y_receiver_branch = bool(
            tool[1] <= _LEFT_RECEIVER_LOW_Y_THRESHOLD_M
        )
        reset_yaw_rad = 2.0 * math.atan2(
            float(reset_yaw_quat[3]), float(reset_yaw_quat[0])
        )
        self._as019_high_handover_assist_tier = bool(
            tool[1] <= _HIGH_HANDOVER_ASSIST_TOOL_Y_THRESHOLD_M
            or abs(reset_yaw_rad)
            > _RESET_CONTROLLER_REACHABLE_ABS_YAW_MAX_RAD
        )
        self._as019_receiver_retry_assist_tier = False
        self._as019_reset_yaw_quat = reset_yaw_quat
        # Randomized yaw is a reset perturbation, not a new definition of
        # "outward".  Contact-owned assistance may first secure a pose sampled
        # just outside the 30-degree success cone, then physically steer it to
        # the task's authoritative rotated local +X orientation before any
        # handover event is allowed to latch.
        self._as019_outward_reference_rotation = _quat_wxyz_to_matrix(
            np.asarray(screwdriver_orientation, dtype=np.float64)
        )
        self.env_step = 0
        self._success_counter = 0
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        self._arm_pose_filters = {"right": None, "left": None}
        self._as019_stage = 0
        self._as019_stage_name = "await_right_contact"
        self._as019_failure_reason = ""
        self._as019_table_safety_guard_armed = False
        self._as019_table_safety_failure_candidate = ""
        self._as019_table_safety_failure_counter = 0
        self._as019_table_safety_guard_last_step = -1
        self._as019_inward_orientation_failure_counter = 0
        self._as019_inward_orientation_guard_last_step = -1
        self._as019_success = False
        self._as019_stability_counter = 0
        self._as019_progress_reset_candidate_counter = 0
        self._as019_progress_reset_count = 0
        self._as019_last_progress_reset_reason = ""
        self._as019_receiver_hold_counter = 0
        self._as019_receiver_hold_complete_step = -1
        self._as019_stage2_left_contact_active = False
        self._as019_right_release_counter = 0
        self._as019_substep_latches = {
            "right_tail": False,
            "dual_contact": False,
            "right_clear": False,
            "zone_placed": False,
            "both_clear_stable": False,
            "tip_left_palm": False,
            "left_first": False,
        }
        self._as019_event_steps.fill(-1)
        self._as019_last_metrics.clear()
        self._as019_assist_anchor.clear()
        self._as019_receiver_pinch_frozen = False
        self._as019_receiver_pinch_target = None
        self._as019_receiver_pinch_freeze.clear()
        self._as019_automatic_phase = 0
        self._as019_automatic_acquire_gate_required = False
        self._as019_automatic_acquire_gate_open = False
        self._as019_automatic_arrival_gate = {}
        self._as019_placement_assist_enabled = False
        self._as019_placement_motion_started = False
        self._as019_release_support_enabled = False
        self._as019_wrench_active = False
        self._as019_active_wrench_owners = ()
        self._as019_active_wrench_gate_snapshot.clear()
        self._as019_slider_orientation_latched = False
        self._as019_slider_orientation_latched_step = -1
        self._as019_slider_orientation_active_owners = ()
        self._as019_slider_orientation_grace_owners = ()
        self._as019_slider_owner_last_live_step.clear()
        self._as019_slider_hand_orientation_locks.clear()
        self._as019_release_frame_trace.clear()
        self._as019_release_trace_last_step = -1
        self._as019_release_trace_dropped_frames = 0
        self._as019_release_event_steps.fill(-1)
        self._as019_last_policy_action.fill(0.0)
        self._as019_last_policy_action_step = -1
        self._as019_arm_control_trace = {"right": {}, "left": {}}
        self._as019_arm_branch_guard_count = 0
        self._as019_contact_query_valid = False
        self._as019_contact_query_error = "contact_query_not_run_after_reset"
        self._as019_table_safety_query_valid = False
        self._as019_table_safety_query_error = (
            "table_safety_query_not_run_after_reset"
        )
        self._as019_metrics_generation = 0
        if clear_manual_request:
            self._manual_reset_requested = False
            self._manual_reset_viewport_camera_state = None
        # Wrapper-local plans use this serial to detect a raw-env reset that
        # bypassed ``AS019AutomaticPolicyWrapper.reset`` (the acceptance runner
        # intentionally constructs a fresh policy after resetting the raw env).
        self._as019_reset_serial += 1

    def _reset_scene_state(self) -> None:
        super()._reset_scene_state()
        if self._scene is None or self._torch is None:
            return
        try:
            tool = self._scene["peg"]
            masses = tool.root_physx_view.get_masses().clone()
            masses[:] = float(self._as019_mass_kg)
            indices = self._torch.arange(
                int(tool.root_physx_view.count),
                dtype=self._torch.long,
                device=masses.device,
            )
            tool.root_physx_view.set_masses(masses, indices)
            # Reset the composer itself so neither a previous grasp wrench nor
            # an active zero-wrench writer carries into the next episode.
            # IsaacLab 2.3.2's empty-tensor compatibility path still marks the
            # composer active; ``reset()`` is the API that clears that flag.
            tool.permanent_wrench_composer.reset()
            if self._stage is not None:
                from pxr import Usd, UsdPhysics

                root = self._stage.GetPrimAtPath("/World/envs/env_0/Screwdriver")
                for prim in Usd.PrimRange(root):
                    if not prim.HasAPI(UsdPhysics.MaterialAPI):
                        continue
                    material = UsdPhysics.MaterialAPI.Get(
                        self._stage, prim.GetPath()
                    )
                    material.CreateStaticFrictionAttr().Set(
                        float(self._as019_friction)
                    )
                    material.CreateDynamicFrictionAttr().Set(
                        float(self._as019_friction)
                    )
                # The base reset advances one mandatory PhysX frame after
                # this method.  Defer sleep until that frame has flushed all
                # scene buffers; write_data_to_sim wakes a body if sleep is
                # requested any earlier.
                self._as019_reset_sleep_pending = True
        except Exception as exc:
            self._isaac_warnings.append(f"AS019 reset dynamics update failed: {exc}")

    def _put_reset_screwdriver_to_sleep(self) -> None:
        """Restore the authored starts-asleep state at the final reset boundary."""

        if self._stage is None:
            raise RuntimeError("AS019 screwdriver reset has no USD stage")
        import omni.physx
        import omni.usd
        from pxr import PhysicsSchemaTools, Sdf, Usd, UsdPhysics

        simulation = omni.physx.get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        root = self._stage.GetPrimAtPath("/World/envs/env_0/Screwdriver")
        slept_body_count = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            simulation.put_to_sleep(
                stage_id,
                PhysicsSchemaTools.sdfPathToInt(
                    Sdf.Path(str(prim.GetPath()))
                ),
            )
            slept_body_count += 1
        if slept_body_count == 0:
            raise RuntimeError(
                "AS019 screwdriver reset found no dynamic rigid body"
            )

    def _restore_authored_reset_tool_boundary(self) -> None:
        """Finalize reset pose/velocity after PhysX consumes pending writes."""

        if self._scene is None or self._sim is None or self._torch is None:
            raise RuntimeError("AS019 screwdriver reset boundary is unavailable")
        tool = self._scene["peg"]
        root_state = tool.data.root_state_w.clone()
        pose = self._torch.tensor(
            np.asarray(self._peg_ori_pose, dtype=np.float64).reshape(1, 7),
            dtype=root_state.dtype,
            device=root_state.device,
        ).repeat(root_state.shape[0], 1)
        if hasattr(self._scene, "env_origins"):
            pose[:, :3] += self._scene.env_origins
        velocity = self._torch.zeros_like(root_state[:, 7:])
        tool.write_root_pose_to_sim(pose)
        tool.write_root_velocity_to_sim(velocity)
        tool.permanent_wrench_composer.reset()
        # Flush the reset-only state writes without advancing time.  Sleeping
        # after this sync leaves no pending pose write that can wake the body
        # on the first recorded action frame.
        self._sim.forward()
        self._put_reset_screwdriver_to_sleep()
        self._scene.update(0.0)
        self._refresh_object_poses()

    def _step_isaaclab(self) -> None:
        if not getattr(self, "_as019_reset_sleep_pending", False):
            super()._step_isaaclab()
            return
        if self._scene is None or self._sim is None:
            return
        # This branch is used only by reset's single mandatory physics frame.
        # Flush robot, object, and wrench-composer buffers first, then put the
        # exact task rigid body to sleep immediately before stepping.  The
        # ordinary trajectory path remains the unmodified base implementation.
        self._resume_timeline_if_paused(source="Timeline pause")
        self._scene.write_data_to_sim()
        self._put_reset_screwdriver_to_sleep()
        self._sim.step(render=False)
        self._scene.update(self._sim_dt)
        # PhysX applies a teleported dynamic root before honoring a sleep
        # request, which otherwise produces exactly one gravity substep on
        # every reused-scene reset.  Re-establish the already-authored reset
        # boundary after that mandatory frame, then sleep without another
        # physics advance.  This remains outside trajectory action capture.
        self._restore_authored_reset_tool_boundary()
        self._as019_reset_sleep_pending = False

    def _contact_force(self, name: str) -> float:
        view = self._as019_contact_views.get(name)
        if view is None:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = f"{name}:contact_view_unavailable"
            return 0.0
        try:
            values = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            array = np.asarray(values, dtype=np.float64)
            return float(np.linalg.norm(array, axis=-1).sum()) if array.size else 0.0
        except Exception as exc:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = (
                f"{name}:{type(exc).__name__}: {exc}"
            )
            return 0.0

    @staticmethod
    def _contact_buffer_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _empty_opposed_contact_evidence() -> dict[str, Any]:
        return {
            "pinch": False,
            "touch": False,
            "total_force": 0.0,
            "thumb_force": 0.0,
            "opposing_force": 0.0,
            "opposed_finger_pair": (),
            "contact_local_x": None,
            "maximum_separation": float("inf"),
            "minimum_separation": float("inf"),
            "radial_cosine": float("nan"),
            "radial_span_m": 0.0,
            "penetration_violation": False,
        }

    def _validated_detailed_contact_records(
        self,
        view: Any,
        query_name: str,
        *,
        filter_paths: tuple[str, ...] = (),
    ) -> tuple[list[dict[str, Any]], str]:
        """Return path-attributed rows only when all six buffers are valid.

        Counts/starts index the shared detailed buffers per sensor/filter pair.
        Treat malformed or saturated buffers as unavailable evidence; silently
        accepting their prefix could manufacture contact support or conceal an
        unsafe robot/table row.
        """

        records: list[dict[str, Any]] = []
        try:
            data = view.get_contact_data(self._sim_dt)
            if len(data) != 6:
                raise RuntimeError(
                    f"expected 6 contact buffers, received {len(data)}"
                )
            forces, points, normals, separations, counts, starts = (
                self._contact_buffer_numpy(value) for value in data
            )
            forces = np.asarray(forces, dtype=np.float64).reshape(-1)
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
            separations = np.asarray(separations, dtype=np.float64).reshape(-1)
            raw_counts = np.asarray(counts).reshape(-1)
            raw_starts = np.asarray(starts).reshape(-1)
            if not (
                forces.size
                == points.shape[0]
                == normals.shape[0]
                == separations.size
            ):
                raise RuntimeError("contact detail buffer length mismatch")
            if not (
                np.all(np.isfinite(forces))
                and np.all(np.isfinite(points))
                and np.all(np.isfinite(normals))
                and np.all(np.isfinite(separations))
            ):
                raise RuntimeError("non-finite contact detail buffer")
            if (
                not np.all(np.isfinite(raw_counts))
                or not np.all(np.isfinite(raw_starts))
                or not np.all(raw_counts == np.floor(raw_counts))
                or not np.all(raw_starts == np.floor(raw_starts))
            ):
                raise RuntimeError("contact count/start buffers are invalid")
            counts = raw_counts.astype(np.int64, copy=False)
            starts = raw_starts.astype(np.int64, copy=False)
            if counts.shape != starts.shape:
                raise RuntimeError("contact count/start buffer mismatch")
            sensor_count = int(getattr(view, "sensor_count", 0))
            filter_count = int(getattr(view, "filter_count", 0))
            if sensor_count <= 0 or filter_count <= 0:
                raise RuntimeError("contact view cardinality is invalid")
            expected_pairs = sensor_count * filter_count
            if counts.size != expected_pairs:
                raise RuntimeError("contact count/start pair shape mismatch")
            if filter_paths and len(filter_paths) != filter_count:
                raise RuntimeError("contact filter path cardinality mismatch")
            if np.any(counts < 0) or np.any(starts < 0):
                raise RuntimeError("negative contact count/start index")
            max_contact_count = getattr(view, "max_contact_data_count", None)
            if (
                max_contact_count is not None
                and int(np.sum(counts)) >= int(max_contact_count)
            ):
                raise RuntimeError("contact detail buffer saturated")
            used_indices: set[int] = set()
            for pair_index, (count, start) in enumerate(zip(counts, starts)):
                end = int(start) + int(count)
                if end > forces.size:
                    raise RuntimeError("contact buffer index out of range")
                for contact_index in range(int(start), int(start) + int(count)):
                    if contact_index in used_indices:
                        raise RuntimeError("overlapping contact buffer slices")
                    used_indices.add(contact_index)
                    force = abs(float(forces[contact_index]))
                    separation = float(separations[contact_index])
                    point = points[contact_index]
                    normal = normals[contact_index]
                    filter_index = pair_index % filter_count
                    records.append(
                        {
                            "force": force,
                            "point": point.copy(),
                            "normal": normal.copy(),
                            "separation": separation,
                            "sensor_index": pair_index // filter_count,
                            "filter_index": filter_index,
                            "filter_path": (
                                filter_paths[filter_index]
                                if filter_paths
                                else ""
                            ),
                        }
                    )
        except Exception as exc:
            return [], f"{query_name}:{type(exc).__name__}: {exc}"
        return records, ""

    def _detailed_contact_rows(
        self,
        view: Any,
        query_name: str,
    ) -> tuple[list[tuple[float, np.ndarray, float]], bool]:
        """Return owner-contact rows through the shared fail-closed parser."""

        records, error = self._validated_detailed_contact_records(
            view, query_name
        )
        if error:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = error
            return [], False
        rows = [
            (
                float(record["force"]),
                np.asarray(record["point"], dtype=np.float64).copy(),
                float(record["separation"]),
            )
            for record in records
        ]
        penetration_violation = any(
            separation < -_CONTACT_MAX_PENETRATION_M
            for _force, _point, separation in rows
        )
        return rows, penetration_violation

    @staticmethod
    def _finger_name_from_filter_path(side: str, filter_path: str) -> str | None:
        """Map an exact Wuji rigid-owner path to its semantic finger id."""

        name = str(filter_path).rstrip("/").rsplit("/", 1)[-1].lower()
        prefix = f"{side}_finger"
        if not name.startswith(prefix):
            return None
        digits = []
        for character in name[len(prefix) :]:
            if not character.isdigit():
                break
            digits.append(character)
        if not digits:
            return None
        index = int("".join(digits))
        return f"finger{index}" if 1 <= index <= 5 else None

    def _sample_owner_contact_records(self) -> dict[str, list[dict[str, Any]]]:
        """Read each grouped hand/tool view once for the current substep.

        Each group view already has one filter per exact rigid owner, so the
        detailed buffer contains the same finger attribution as the legacy ten
        one-finger views.  Batching avoids dozens of redundant GPU-to-CPU
        synchronizations without weakening contact ownership semantics.
        """

        records_by_name: dict[str, list[dict[str, Any]]] = {}
        errors: list[str] = []
        for name in ("right_fingers", "left_fingers", "left_palm"):
            view = self._as019_contact_views.get(name)
            filter_paths = self._as019_contact_filter_paths.get(name, ())
            if view is None or not filter_paths:
                errors.append(f"{name}:contact_view_unavailable")
                records_by_name[name] = []
                continue
            records, error = self._validated_detailed_contact_records(
                view,
                name,
                filter_paths=filter_paths,
            )
            records_by_name[name] = records
            if error:
                errors.append(error)
        if errors:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = "; ".join(errors)
        self._as019_owner_contact_records = records_by_name
        return records_by_name

    @staticmethod
    def _contact_force_from_records(records: list[dict[str, Any]]) -> float:
        return float(
            sum(abs(float(record.get("force", 0.0))) for record in records)
        )

    def _table_safety_metrics(self) -> dict[str, Any]:
        """Sample the three exact table-safety views without owner aliases."""

        analytic_gap = float(
            _tool_table_gap_m(self._peg_ori_pose, self._socket_ori_pose)
        )
        query_valid = bool(self._as019_table_safety_available)
        query_error = (
            ""
            if query_valid
            else "table_safety_views_incomplete"
        )
        records_by_name: dict[str, list[dict[str, Any]]] = {
            "tool_table": [],
            "table_arm": [],
            "table_hand": [],
        }
        required_names = {"tool_table", "table_hand"}
        for name in records_by_name:
            view = self._as019_table_safety_views.get(name)
            filter_paths = self._as019_table_safety_filter_paths.get(name, ())
            if view is None or not filter_paths:
                if name in required_names:
                    query_valid = False
                if name in required_names and not query_error:
                    query_error = f"{name}:contact_view_unavailable"
                continue
            records, error = self._validated_detailed_contact_records(
                view,
                name,
                filter_paths=filter_paths,
            )
            if error and name in required_names:
                query_valid = False
                if not query_error:
                    query_error = error
                continue
            records_by_name[name] = records
        if not math.isfinite(analytic_gap):
            query_valid = False
            if not query_error:
                query_error = "tool_table:non_finite_analytic_gap"

        def separations(name: str) -> list[float]:
            return [
                float(record["separation"])
                for record in records_by_name[name]
            ]

        def contact_paths(
            name: str,
            predicate: Callable[[float], bool],
        ) -> tuple[str, ...]:
            return tuple(
                sorted(
                    {
                        str(record["filter_path"])
                        for record in records_by_name[name]
                        if predicate(float(record["separation"]))
                    }
                )
            )

        tool_separations = separations("tool_table")
        arm_separations = separations("table_arm")
        hand_separations = separations("table_hand")
        tool_penetration_violation = any(
            value < _TABLE_MIN_SEPARATION_PHYSX_M
            for value in tool_separations
        )
        arm_contact = any(value <= 0.0 for value in arm_separations)
        hand_penetration_violation = any(
            value < _TABLE_MIN_SEPARATION_PHYSX_M
            for value in hand_separations
        )
        bounded_tool_manifold = any(
            _TABLE_MIN_SEPARATION_PHYSX_M
            <= value
            <= _TABLE_SUPPORT_MAX_SEPARATION_PHYSX_M
            for value in tool_separations
        )
        analytic_gap_min = (
            -_TABLE_MAX_PENETRATION_M
            - _TABLE_ANALYTIC_GAP_ROUNDOFF_M
        )
        analytic_gap_max = (
            _TABLE_SUPPORT_MAX_SEPARATION_M
            + _TABLE_ANALYTIC_GAP_ROUNDOFF_M
        )
        analytic_gap_supported = bool(
            math.isfinite(analytic_gap)
            and analytic_gap_min <= analytic_gap <= analytic_gap_max
        )
        tool_supported = bool(
            query_valid
            and bounded_tool_manifold
            and analytic_gap_supported
            and not tool_penetration_violation
        )
        self._as019_table_safety_query_valid = bool(query_valid)
        self._as019_table_safety_query_error = str(query_error)

        def minimum(values: list[float]) -> float:
            return min(values) if values else float("inf")

        def maximum(values: list[float]) -> float:
            return max(values) if values else float("inf")

        return {
            "table_safety_views_available": bool(
                self._as019_table_safety_available
            ),
            "table_safety_query_valid": bool(query_valid),
            "table_safety_query_error": str(query_error),
            "tool_table_supported": tool_supported,
            "tool_table_analytic_gap_m": analytic_gap,
            "tool_table_contact_row_count": len(tool_separations),
            "tool_table_physical_contact": any(
                value <= 0.0 for value in tool_separations
            ),
            "tool_table_contact_paths": contact_paths(
                "tool_table",
                lambda value: (
                    value <= _TABLE_SUPPORT_MAX_SEPARATION_PHYSX_M
                ),
            ),
            "tool_table_min_separation_m": minimum(tool_separations),
            "tool_table_max_separation_m": maximum(tool_separations),
            "tool_table_penetration_violation": (
                tool_penetration_violation
            ),
            "arm_table_contact_row_count": len(arm_separations),
            "arm_table_physical_contact": arm_contact,
            "arm_table_contact_paths": contact_paths(
                "table_arm", lambda value: value <= 0.0
            ),
            "arm_table_min_separation_m": minimum(arm_separations),
            "hand_table_contact_row_count": len(hand_separations),
            "hand_table_physical_contact": any(
                value <= 0.0 for value in hand_separations
            ),
            "hand_table_contact_paths": contact_paths(
                "table_hand", lambda value: value <= 0.0
            ),
            "hand_table_min_separation_m": minimum(hand_separations),
            "hand_table_penetration_violation": (
                hand_penetration_violation
            ),
        }

    @staticmethod
    def _table_safety_failure_from_metrics(
        metrics: dict[str, Any],
    ) -> str:
        if not bool(metrics.get("table_safety_query_valid", False)):
            return "table_safety_query_invalid"
        if bool(metrics.get("tool_table_penetration_violation", False)):
            return "screwdriver_table_penetration_exceeded"
        if bool(metrics.get("hand_table_penetration_violation", False)):
            return "hand_table_penetration_exceeded"
        return ""

    def _latch_table_safety_failure(self, metrics: dict[str, Any]) -> None:
        """Arm and debounce table failures on distinct control frames.

        ``_metrics`` is sampled after every 120 Hz substep and reused at the
        30 Hz boundary.  Updating the terminal timer at most once per
        ``env_step`` prevents four callbacks over one action from masquerading
        as four independent unsafe observations.
        """

        if self._as019_failure_reason:
            return
        current_step = int(self.env_step)
        if self._as019_table_safety_guard_last_step == current_step:
            return
        self._as019_table_safety_guard_last_step = current_step
        if current_step < _TABLE_SAFETY_ARM_AFTER_CONTROL_STEPS:
            self._as019_table_safety_guard_armed = False
            self._as019_table_safety_failure_candidate = ""
            self._as019_table_safety_failure_counter = 0
            return
        self._as019_table_safety_guard_armed = True
        reason = self._table_safety_failure_from_metrics(metrics)
        if not reason:
            self._as019_table_safety_failure_candidate = ""
            self._as019_table_safety_failure_counter = 0
            return
        if reason == self._as019_table_safety_failure_candidate:
            self._as019_table_safety_failure_counter += 1
        else:
            self._as019_table_safety_failure_candidate = reason
            self._as019_table_safety_failure_counter = 1
        if (
            self._as019_table_safety_failure_counter
            >= _TABLE_SAFETY_FAILURE_DEBOUNCE_STEPS
        ):
            self._as019_failure_reason = reason

    @staticmethod
    def _inside_local_envelope(
        local_point: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        tolerance_m: float = _CONTACT_GEOMETRY_TOLERANCE_M,
    ) -> bool:
        point = np.asarray(local_point, dtype=np.float64).reshape(3)
        return bool(
            np.all(point >= np.asarray(low, dtype=np.float64) - tolerance_m)
            and np.all(point <= np.asarray(high, dtype=np.float64) + tolerance_m)
        )

    def _opposed_handle_contact_evidence(
        self,
        side: str,
        *,
        local_x_min_m: float,
        local_x_max_m: float,
        require_thumb_opposition: bool = True,
        envelope_low: np.ndarray = _HANDLE_LOCAL_MIN,
        envelope_high: np.ndarray = _HANDLE_LOCAL_MAX,
    ) -> dict[str, Any]:
        """Return validated physical contact evidence on a tool envelope.

        The optional ``require_thumb_opposition`` flag remains available for
        the compliant assist, but the AS019 success path consumes ``touch``
        only: any non-speculative contact row inside the full screwdriver
        envelope is sufficient, regardless of contact location or pinch
        geometry.
        """

        empty_evidence = self._empty_opposed_contact_evidence()
        tool_matrix = np.asarray(self._tool_matrix(), dtype=np.float64)
        if tool_matrix.shape != (4, 4) or not np.all(
            np.isfinite(tool_matrix)
        ):
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = f"{side}_tool_transform_invalid"
            return empty_evidence
        rotation = tool_matrix[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4)
            or not math.isclose(
                float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-4
            )
        ):
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = f"{side}_tool_transform_invalid"
            return empty_evidence

        accepted_forces = {
            f"finger{finger_index}": 0.0 for finger_index in range(1, 6)
        }
        accepted_points: dict[str, list[tuple[float, np.ndarray]]] = {
            f"finger{finger_index}": [] for finger_index in range(1, 6)
        }
        weighted_x = 0.0
        total_force = 0.0
        minimum_separation = float("inf")
        maximum_separation = -float("inf")
        penetration_violation = False
        tool_inverse = _invert_pose_matrix(tool_matrix)
        grouped_records = getattr(
            self, "_as019_owner_contact_records", {}
        ).get(f"{side}_fingers")
        finger_rows: list[tuple[str, float, np.ndarray, float]] = []
        if grouped_records is not None:
            for record in grouped_records:
                name = self._finger_name_from_filter_path(
                    side, str(record.get("filter_path", ""))
                )
                if name is None:
                    self._as019_contact_query_valid = False
                    self._as019_contact_query_error = (
                        f"{side}_fingers:unattributed_filter_path"
                    )
                    continue
                finger_rows.append(
                    (
                        name,
                        float(record["force"]),
                        np.asarray(record["point"], dtype=np.float64),
                        float(record["separation"]),
                    )
                )
        else:
            views = (
                self._as019_right_per_finger_views
                if side == "right"
                else self._as019_per_finger_views
            )
            for finger_index in range(1, 6):
                name = f"finger{finger_index}"
                view = views.get(name)
                if view is None:
                    self._as019_contact_query_valid = False
                    self._as019_contact_query_error = (
                        f"{side}_{name}:contact_view_unavailable"
                    )
                    continue
                rows, view_penetration_violation = self._detailed_contact_rows(
                    view, f"{side}_{name}"
                )
                penetration_violation |= view_penetration_violation
                finger_rows.extend(
                    (name, force, point, separation)
                    for force, point, separation in rows
                )
        penetration_violation |= any(
            separation < -_CONTACT_MAX_PENETRATION_M
            for _name, _force, _point, separation in finger_rows
        )
        for name, force, point, separation in finger_rows:
            if separation < -_CONTACT_MAX_PENETRATION_M:
                continue
            local_point = tool_inverse @ np.asarray(
                [*point, 1.0], dtype=np.float64
            )
            if not np.all(np.isfinite(local_point)):
                self._as019_contact_query_valid = False
                self._as019_contact_query_error = (
                    f"{side}_{name}:non_finite_local_contact"
                )
                continue
            local_x = float(local_point[0])
            if (
                force < _CONTACT_FORCE_N
                or separation > _CONTACT_MAX_SEPARATION_M
                or local_x < local_x_min_m
                or local_x > local_x_max_m
                or not self._inside_local_envelope(
                    local_point[:3], envelope_low, envelope_high
                )
            ):
                continue
            accepted_forces[name] += force
            accepted_points[name].append(
                (force, local_point[:3].copy())
            )
            weighted_x += force * local_x
            total_force += force
            minimum_separation = min(minimum_separation, separation)
            maximum_separation = max(maximum_separation, separation)

        thumb_force = float(accepted_forces["finger1"])
        opposing_force = max(
            (
                float(accepted_forces[f"finger{finger_index}"])
                for finger_index in range(2, 6)
            ),
            default=0.0,
        )
        best_radial_cosine = 1.0
        best_radial_span = 0.0
        geometric_opposition = False
        opposed_finger_pair: tuple[str, str] = ()
        best_opposed_radial_cosine = 1.0
        radial_center = _HANDLE_LOCAL_CENTER[1:3]
        finger_pairs = (
            tuple((1, finger_index) for finger_index in range(2, 6))
            if require_thumb_opposition
            else tuple(
                (first_index, second_index)
                for first_index in range(1, 6)
                for second_index in range(first_index + 1, 6)
            )
        )
        for first_index, second_index in finger_pairs:
            first_name = f"finger{first_index}"
            second_name = f"finger{second_index}"
            for _first_force, first_point in accepted_points[first_name]:
                first_radial = first_point[1:3] - radial_center
                first_radius = float(np.linalg.norm(first_radial))
                if first_radius < _RIGHT_PINCH_MIN_RADIAL_RADIUS_M:
                    continue
                for _second_force, second_point in accepted_points[second_name]:
                    second_radial = second_point[1:3] - radial_center
                    second_radius = float(np.linalg.norm(second_radial))
                    if second_radius < _RIGHT_PINCH_MIN_RADIAL_RADIUS_M:
                        continue
                    radial_cosine = float(
                        np.dot(first_radial, second_radial)
                        / (first_radius * second_radius)
                    )
                    radial_span = float(
                        np.linalg.norm(first_radial - second_radial)
                    )
                    opposed = bool(
                        radial_cosine <= _RIGHT_PINCH_MAX_RADIAL_COSINE
                        and radial_span >= _RIGHT_PINCH_MIN_RADIAL_SPAN_M
                    )
                    if opposed and (
                        not opposed_finger_pair
                        or radial_cosine < best_opposed_radial_cosine
                    ):
                        opposed_finger_pair = (first_name, second_name)
                        best_opposed_radial_cosine = radial_cosine
                    geometric_opposition |= opposed
                    best_radial_cosine = min(
                        best_radial_cosine, radial_cosine
                    )
                    best_radial_span = max(best_radial_span, radial_span)
        if penetration_violation:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = (
                f"{side}_hand_tool_penetration_exceeded"
            )
        if require_thumb_opposition:
            required_force_evidence = bool(
                thumb_force >= _CONTACT_FORCE_N
                and opposing_force >= _CONTACT_FORCE_N
            )
        else:
            required_force_evidence = bool(opposed_finger_pair)
        pinch = bool(
            self._as019_contact_query_valid
            and not penetration_violation
            and required_force_evidence
            and geometric_opposition
        )
        return {
            "pinch": pinch,
            "touch": bool(
                self._as019_contact_query_valid
                and not penetration_violation
                and total_force >= _CONTACT_FORCE_N
            ),
            "total_force": float(total_force),
            "thumb_force": thumb_force,
            "opposing_force": opposing_force,
            "opposed_finger_pair": opposed_finger_pair,
            "contact_local_x": (
                weighted_x / total_force
                if total_force > 0.0
                else None
            ),
            "maximum_separation": (
                maximum_separation
                if total_force > 0.0
                else float("inf")
            ),
            "minimum_separation": (
                minimum_separation
                if total_force > 0.0
                else float("inf")
            ),
            "radial_cosine": (
                best_radial_cosine
                if best_radial_cosine < 1.0
                else float("nan")
            ),
            "radial_span_m": best_radial_span,
            "penetration_violation": penetration_violation,
        }

    def _right_tail_contact_evidence(self) -> dict[str, Any]:
        """Return any validated right-finger contact on the screwdriver."""

        return self._opposed_handle_contact_evidence(
            "right",
            local_x_min_m=float(_TOOL_LOCAL_MIN[0]),
            local_x_max_m=float(_TOOL_LOCAL_MAX[0]),
            require_thumb_opposition=False,
            envelope_low=_TOOL_LOCAL_MIN,
            envelope_high=_TOOL_LOCAL_MAX,
        )

    def _left_middle_contact_evidence(self) -> dict[str, Any]:
        """Return any validated left-finger contact on the screwdriver."""

        return self._opposed_handle_contact_evidence(
            "left",
            local_x_min_m=float(_TOOL_LOCAL_MIN[0]),
            local_x_max_m=float(_TOOL_LOCAL_MAX[0]),
            require_thumb_opposition=False,
            envelope_low=_TOOL_LOCAL_MIN,
            envelope_high=_TOOL_LOCAL_MAX,
        )

    def _left_tip_palm_contact_evidence(self) -> dict[str, Any]:
        """Return detailed receiving-palm contact on the conservative blade."""

        evidence = {
            "contact": False,
            "force": 0.0,
            "contact_local_x": None,
            "minimum_separation": float("inf"),
            "penetration_violation": False,
        }
        view = self._as019_contact_views.get("left_palm")
        if view is None:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = "left_palm:contact_view_unavailable"
            return evidence
        tool_matrix = np.asarray(self._tool_matrix(), dtype=np.float64)
        if tool_matrix.shape != (4, 4) or not np.all(np.isfinite(tool_matrix)):
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = "left_palm_tool_transform_invalid"
            return evidence
        rotation = tool_matrix[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4)
            or not math.isclose(
                float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-4
            )
        ):
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = "left_palm_tool_transform_invalid"
            return evidence
        tool_inverse = _invert_pose_matrix(tool_matrix)
        grouped_records = getattr(
            self, "_as019_owner_contact_records", {}
        ).get("left_palm")
        if grouped_records is None:
            rows, penetration_violation = self._detailed_contact_rows(
                view, "left_palm"
            )
        else:
            rows = [
                (
                    float(record["force"]),
                    np.asarray(record["point"], dtype=np.float64),
                    float(record["separation"]),
                )
                for record in grouped_records
            ]
            penetration_violation = any(
                separation < -_CONTACT_MAX_PENETRATION_M
                for _force, _point, separation in rows
            )
        weighted_x = 0.0
        total_force = 0.0
        minimum_separation = float("inf")
        for force, point, separation in rows:
            local_point = tool_inverse @ np.asarray(
                [*point, 1.0], dtype=np.float64
            )
            if not np.all(np.isfinite(local_point)):
                self._as019_contact_query_valid = False
                self._as019_contact_query_error = (
                    "left_palm:non_finite_local_contact"
                )
                continue
            if (
                force < _CONTACT_FORCE_N
                or separation > _CONTACT_MAX_SEPARATION_M
                or float(local_point[0]) < _SHARP_TIP_LOCAL_X_MIN_M
                or not self._inside_local_envelope(
                    local_point[:3], _TOOL_LOCAL_MIN, _TOOL_LOCAL_MAX
                )
            ):
                continue
            total_force += force
            weighted_x += force * float(local_point[0])
            minimum_separation = min(minimum_separation, separation)
        if penetration_violation:
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = (
                "left_palm_tool_penetration_exceeded"
            )
        evidence.update(
            {
                # A deep blade contact remains a forbidden event even though it
                # also invalidates all ownership and success evidence.
                "contact": bool(total_force >= _CONTACT_FORCE_N),
                "force": total_force,
                "contact_local_x": (
                    weighted_x / total_force if total_force > 0.0 else None
                ),
                "minimum_separation": minimum_separation,
                "penetration_violation": penetration_violation,
            }
        )
        return evidence

    def _receiver_has_physical_pinch(
        self, metrics: dict[str, Any] | None = None
    ) -> bool:
        values = self._as019_last_metrics if metrics is None else metrics
        return bool(
            values.get("contact_query_valid", False)
            and values.get("left_middle_physical", False)
        )

    def _receiver_pinch_progress_scale(self) -> float:
        if self._as019_mass_kg <= _AUTOMATIC_RECEIVER_LOW_INERTIA_MASS_KG:
            return _AUTOMATIC_RECEIVER_LOW_INERTIA_PINCH_PROGRESS_SCALE
        return _AUTOMATIC_RECEIVER_PINCH_PROGRESS_SCALE

    def _update_receiver_pinch_freeze(self, substep: int) -> None:
        """Freeze the receiver hand at the first live two-sided pinch.

        This is a compliant robot target, not an object-pose write.  It avoids
        driving a fully closed hand through the narrow cylindrical handle and
        preserves the reset-relative measured grasp until the giver has cleared.
        """

        if not _AS019_RECEIVER_PINCH_FREEZE_ENABLED:
            return
        if self._as019_stage != 1 or self._as019_receiver_pinch_frozen:
            return
        metrics = self._as019_last_metrics
        if not self._receiver_has_physical_pinch(metrics):
            return
        thumb_force = float(metrics.get("left_thumb_middle_force", 0.0))
        opposing_force = float(
            metrics.get("left_opposing_middle_force", 0.0)
        )
        robot = self._scene["left_robot"]
        joint_ids = list(self._hand_joint_ids["left"])
        measured = robot.data.joint_pos[:, joint_ids].clone()
        validated = measured.new_tensor(
            _AUTOMATIC_RECEIVER_PINCH_HOLD_TARGET
        ).reshape(1, -1)
        max_correction = float(
            (measured - validated).abs().max().detach().cpu().item()
        )
        normalization_applied = bool(
            max_correction
            <= _AUTOMATIC_RECEIVER_PINCH_NORMALIZATION_MAX_RAD
        )
        approach = measured.new_tensor(
            _AUTOMATIC_RECEIVER_APPROACH_TARGET
        ).reshape(1, -1)
        measured_progress_scale = self._receiver_pinch_progress_scale()
        measured_hold = approach + (
            measured_progress_scale * (measured - approach)
        )
        target = validated if normalization_applied else measured_hold
        self._as019_receiver_pinch_target = target
        self._left_hand[:] = (
            target[0].detach().cpu().numpy().astype(np.float64)
        )
        self._as019_receiver_pinch_frozen = True
        self._as019_receiver_pinch_freeze = {
            "env_step": int(self.env_step),
            "substep": int(substep),
            "thumb_force_n": thumb_force,
            "opposing_force_n": opposing_force,
            "measured_joint_positions": (
                measured[0].detach().cpu().numpy().tolist()
            ),
            "normalization_applied": normalization_applied,
            "max_normalization_correction_rad": max_correction,
            "measured_progress_scale": (
                1.0 if normalization_applied else measured_progress_scale
            ),
            "joint_positions": self._left_hand.tolist(),
        }

    def _freeze_receiver_pinch_at_owner_event(
        self, metrics: dict[str, Any] | None = None
    ) -> None:
        """Latch the measured receiver joints when contact wins the step race."""

        if not _AS019_RECEIVER_PINCH_FREEZE_ENABLED:
            return
        values = self._as019_last_metrics if metrics is None else metrics
        if (
            self._as019_receiver_pinch_frozen
            or self._scene is None
            or not self._receiver_has_physical_pinch(values)
        ):
            return
        try:
            robot = self._scene["left_robot"]
        except (KeyError, TypeError):
            return
        joint_ids = list(self._hand_joint_ids["left"])
        measured = robot.data.joint_pos[:, joint_ids].clone()
        validated = measured.new_tensor(
            _AUTOMATIC_RECEIVER_PINCH_HOLD_TARGET
        ).reshape(1, -1)
        max_correction = float(
            (measured - validated).abs().max().detach().cpu().item()
        )
        normalization_applied = bool(
            max_correction
            <= _AUTOMATIC_RECEIVER_PINCH_NORMALIZATION_MAX_RAD
        )
        approach = measured.new_tensor(
            _AUTOMATIC_RECEIVER_APPROACH_TARGET
        ).reshape(1, -1)
        thumb_force = float(values.get("left_thumb_middle_force", 0.0))
        opposing_force = float(
            values.get("left_opposing_middle_force", 0.0)
        )
        measured_progress_scale = self._receiver_pinch_progress_scale()
        measured_hold = approach + (
            measured_progress_scale * (measured - approach)
        )
        # Match the pre-substep pinch path exactly.  Solver ordering must not
        # decide whether the same two-sided physical grasp is held at the
        # verified template or strengthened along its measured closing path.
        target = validated if normalization_applied else measured_hold
        self._as019_receiver_pinch_target = target
        self._left_hand[:] = (
            target[0].detach().cpu().numpy().astype(np.float64)
        )
        self._as019_receiver_pinch_frozen = True
        self._as019_receiver_pinch_freeze = {
            "env_step": int(self.env_step),
            "substep": -1,
            "thumb_force_n": thumb_force,
            "opposing_force_n": opposing_force,
            "measured_joint_positions": (
                measured[0].detach().cpu().numpy().tolist()
            ),
            "normalization_applied": normalization_applied,
            "max_normalization_correction_rad": max_correction,
            "measured_progress_scale": (
                1.0 if normalization_applied else measured_progress_scale
            ),
            "owner_event_latched": True,
            "joint_positions": self._left_hand.tolist(),
        }

    def _tool_velocity(self) -> np.ndarray:
        if self._scene is None:
            return np.zeros(6, dtype=np.float64)
        try:
            value = _torch_view(self._scene["peg"].data.root_state_w)[0, 7:13]
            return value.detach().cpu().numpy().astype(np.float64, copy=True)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    def _palm_matrix(self, side: str) -> np.ndarray:
        fallback = self._right_tcp_pose if side == "right" else self._left_tcp_pose
        return self._body_pose_matrix(
            f"{side}_robot", f"{side}_palm_link", fallback
        )

    def _tool_matrix(self) -> np.ndarray:
        return _pose_vector_to_matrix(self._peg_ori_pose)

    def _palm_local_x(self, side: str) -> float:
        tool_inverse = _invert_pose_matrix(self._tool_matrix())
        palm = self._palm_matrix(side)
        local = tool_inverse @ np.asarray([*palm[:3, 3], 1.0], dtype=np.float64)
        return float(local[0])

    def _outward_alignment(self) -> float:
        current_axis = _quat_wxyz_to_matrix(self._peg_ori_pose[3:7])[:, 0]
        reference_axis = self._as019_outward_reference_rotation[:, 0]
        return float(np.dot(current_axis, reference_axis))

    def _tip_to_left_palm_distance(self) -> float:
        tip = (self._tool_matrix() @ np.asarray([*_TOOL_LOCAL_TIP, 1.0]))[:3]
        return float(np.linalg.norm(tip - self._palm_matrix("left")[:3, 3]))

    def _metrics(self) -> dict[str, Any]:
        self._as019_metrics_generation += 1
        metrics_generation = int(self._as019_metrics_generation)
        self._as019_contact_query_valid = bool(self._as019_contact_available)
        self._as019_contact_query_error = (
            "" if self._as019_contact_query_valid else "contact_views_incomplete"
        )
        owner_records = self._sample_owner_contact_records()
        right_force = self._contact_force_from_records(
            owner_records["right_fingers"]
        )
        left_force = self._contact_force_from_records(
            owner_records["left_fingers"]
        )
        left_palm_force = self._contact_force_from_records(
            owner_records["left_palm"]
        )
        right_tail_evidence = self._right_tail_contact_evidence()
        left_middle_evidence = self._left_middle_contact_evidence()
        left_tip_palm_evidence = self._left_tip_palm_contact_evidence()
        owner_contact_query_valid = bool(self._as019_contact_query_valid)
        owner_contact_query_error = str(self._as019_contact_query_error)
        table_safety = self._table_safety_metrics()
        velocity = self._tool_velocity()
        right_local_x = self._palm_local_x("right")
        left_local_x = self._palm_local_x("left")
        contact_query_valid = bool(
            owner_contact_query_valid
            and table_safety["table_safety_query_valid"]
        )
        query_errors = []
        if not owner_contact_query_valid:
            query_errors.append(owner_contact_query_error)
        if not table_safety["table_safety_query_valid"]:
            query_errors.append(
                str(table_safety["table_safety_query_error"])
            )
        self._as019_contact_query_valid = contact_query_valid
        self._as019_contact_query_error = "; ".join(
            error for error in query_errors if error
        )
        outward = self._outward_alignment()
        marker_rotation = _quat_wxyz_to_matrix(self._socket_ori_pose[3:7])
        tool_rotation = _quat_wxyz_to_matrix(self._peg_ori_pose[3:7])
        relative_rotation = marker_rotation.T @ tool_rotation
        delta = marker_rotation.T @ (
            self._peg_ori_pose[:3] - self._socket_ori_pose[:3]
        )
        tool_center_delta = (
            delta + relative_rotation @ _TOOL_LOCAL_BOUNDS_CENTER
        )
        tool_half_extents_in_zone = (
            np.abs(relative_rotation[:2, :]) @ _TOOL_HALF_EXTENTS
        )
        zone_inside = bool(
            abs(float(tool_center_delta[0]))
            <= _MARKER_HALF_EXTENTS[0]
            - tool_half_extents_in_zone[0]
            + 0.012
            and abs(float(tool_center_delta[1]))
            <= _MARKER_HALF_EXTENTS[1]
            - tool_half_extents_in_zone[1]
            + 0.008
            and float(self._peg_ori_pose[2]) <= 0.965
        )
        tip_palm_distance = self._tip_to_left_palm_distance()
        right_tail_physical = bool(
            contact_query_valid
            and right_tail_evidence["touch"]
        )
        left_middle_physical = bool(
            contact_query_valid
            and left_middle_evidence["touch"]
        )
        right_handle_touch = bool(
            contact_query_valid and right_tail_evidence["touch"]
        )
        left_handle_touch = bool(
            contact_query_valid and left_middle_evidence["touch"]
        )
        left_any_touch = bool(
            contact_query_valid
            and (
                left_handle_touch
                or left_palm_force >= _CONTACT_FORCE_N
            )
        )
        return {
            "metrics_generation": metrics_generation,
            "right_force": right_force,
            "right_thumb_tail_force": right_tail_evidence["thumb_force"],
            "right_opposing_tail_force": right_tail_evidence["opposing_force"],
            "right_pinch_physical": right_tail_evidence["pinch"],
            "right_handle_touch": right_handle_touch,
            "right_handle_touch_force": right_tail_evidence["total_force"],
            "left_force": left_force,
            "left_palm_force": left_palm_force,
            "left_thumb_middle_force": left_middle_evidence["thumb_force"],
            "left_opposing_middle_force": left_middle_evidence[
                "opposing_force"
            ],
            "right_tail_physical": right_tail_physical,
            "left_middle_physical": left_middle_physical,
            "left_handle_touch": left_handle_touch,
            "left_handle_touch_force": left_middle_evidence["total_force"],
            "left_any_touch": left_any_touch,
            "right_tail": bool(
                right_tail_physical and outward >= _OUTWARD_COS
            ),
            "left_middle": bool(
                left_middle_physical and outward >= _OUTWARD_COS
            ),
            "right_clear": bool(
                contact_query_valid and right_force <= _RELEASE_FORCE_N
            ),
            "left_clear": bool(
                contact_query_valid and left_force <= _RELEASE_FORCE_N
            ),
            "tip_left_palm": bool(
                left_tip_palm_evidence["contact"]
            ),
            "contact_owner_query_valid": owner_contact_query_valid,
            "contact_owner_query_error": owner_contact_query_error,
            "contact_query_valid": contact_query_valid,
            "contact_query_error": self._as019_contact_query_error,
            "outward_alignment": outward,
            "right_local_x": right_local_x,
            "right_contact_local_x": (
                float(right_tail_evidence["contact_local_x"])
                if right_tail_evidence["contact_local_x"] is not None
                else float("nan")
            ),
            "right_contact_max_separation_m": right_tail_evidence[
                "maximum_separation"
            ],
            "right_contact_min_separation_m": right_tail_evidence[
                "minimum_separation"
            ],
            "right_contact_radial_cosine": right_tail_evidence[
                "radial_cosine"
            ],
            "right_contact_radial_span_m": right_tail_evidence[
                "radial_span_m"
            ],
            "right_contact_penetration_violation": right_tail_evidence[
                "penetration_violation"
            ],
            "left_local_x": left_local_x,
            "left_contact_local_x": (
                float(left_middle_evidence["contact_local_x"])
                if left_middle_evidence["contact_local_x"] is not None
                else float("nan")
            ),
            "left_contact_max_separation_m": left_middle_evidence[
                "maximum_separation"
            ],
            "left_contact_min_separation_m": left_middle_evidence[
                "minimum_separation"
            ],
            "left_contact_radial_cosine": left_middle_evidence[
                "radial_cosine"
            ],
            "left_contact_radial_span_m": left_middle_evidence[
                "radial_span_m"
            ],
            "left_contact_penetration_violation": left_middle_evidence[
                "penetration_violation"
            ],
            "tip_left_palm_force": left_tip_palm_evidence["force"],
            "tip_left_palm_local_x": (
                float(left_tip_palm_evidence["contact_local_x"])
                if left_tip_palm_evidence["contact_local_x"] is not None
                else float("nan")
            ),
            "tip_left_palm_min_separation_m": left_tip_palm_evidence[
                "minimum_separation"
            ],
            "tip_left_palm_penetration_violation": left_tip_palm_evidence[
                "penetration_violation"
            ],
            "tip_left_palm_distance": tip_palm_distance,
            "zone_inside": zone_inside,
            "linear_speed": float(np.linalg.norm(velocity[:3])),
            "angular_speed": float(np.linalg.norm(velocity[3:])),
            **table_safety,
        }

    def _try_attach(self, side: str, metrics: dict[str, Any]) -> None:
        if (
            side == "right"
            and self._as019_stage == 0
            and self._as019_automatic_acquire_gate_required
            and not self._as019_automatic_acquire_gate_open
        ):
            return
        if side in self._as019_assist_anchor:
            if (
                side == "right"
                and self._as019_stage <= 2
                and bool(metrics.get("right_tail_physical", False))
                and not bool(
                    self._as019_last_metrics.get(
                        "right_tail_physical", False
                    )
                )
            ):
                # A grasp may briefly clear while the closed palm performs a
                # bounded reseat.  Re-capture at the real opposed re-contact
                # frame before restoring the wrench, so a stale anchor cannot
                # pull the tool across a visible gap.
                self._as019_assist_anchor[side] = {
                    "hand_to_tool": _invert_pose_matrix(
                        self._palm_matrix(side)
                    )
                    @ self._tool_matrix(),
                    "attached_step": int(self.env_step),
                }
            elif (
                side == "left"
                and self._as019_stage >= 1
                and self._receiver_has_physical_pinch(metrics)
                and not self._receiver_has_physical_pinch(
                    self._as019_last_metrics
                )
                and _grasp_command_max(self._left_hand)
                >= _HAND_CLOSE_THRESHOLD
            ):
                # Restore receiver support only from the current detailed,
                # opposed middle-handle contact.  Rebind on that exact frame
                # so a stale semantic anchor cannot pull the tool across a
                # genuine contact gap.
                self._as019_assist_anchor[side] = {
                    "hand_to_tool": _invert_pose_matrix(
                        self._palm_matrix(side)
                    )
                    @ self._tool_matrix(),
                    "attached_step": int(self.env_step),
                }
            return
        if side == "right" and self._as019_stage > 1:
            # The giver may acquire only before receiver transfer.  A late
            # brush during release/placement must never create a new hidden
            # spring owner, even if the right-hand close command is stale.
            return
        hand = self._right_hand if side == "right" else self._left_hand
        contact_key = (
            "right_tail_physical" if side == "right" else "left_middle_physical"
        )
        attach_threshold = (
            _RIGHT_EARLY_ASSIST_ATTACH_THRESHOLD
            if side == "right" and self._as019_stage == 0
            else _HAND_CLOSE_THRESHOLD
        )
        if not metrics[contact_key] or _grasp_command_max(hand) < attach_threshold:
            return
        if side == "left" and not self._receiver_has_physical_pinch(metrics):
            return
        if side == "left":
            # Contact queries are sampled after the final physics substep.  A
            # two-sided pinch can therefore admit ownership one frame before
            # the pre-substep freeze observes it.  Latch the measured robot
            # joints at that exact event so stage 2 never continues closing
            # through an already valid physical grasp.
            if _AS019_RECEIVER_PINCH_FREEZE_ENABLED:
                self._freeze_receiver_pinch_at_owner_event(metrics)
        self._as019_assist_anchor[side] = {
            "hand_to_tool": _invert_pose_matrix(self._palm_matrix(side)) @ self._tool_matrix(),
            "attached_step": int(self.env_step),
        }

    def _update_assist_ownership(self, metrics: dict[str, Any]) -> None:
        self._try_attach("right", metrics)
        if self._as019_stage >= 1:
            self._try_attach("left", metrics)
        if "right" in self._as019_assist_anchor:
            if _grasp_command_max(self._right_hand) <= _HAND_OPEN_THRESHOLD:
                self._as019_assist_anchor.pop("right", None)
        if "left" in self._as019_assist_anchor:
            if (
                self._as019_stage >= 3
                and _grasp_command_max(self._left_hand) <= _HAND_OPEN_THRESHOLD
                and bool(metrics.get("left_clear", False))
            ):
                self._as019_assist_anchor.pop("left", None)

    def _as019_pre_substep(self, _index: int, _count: int) -> None:
        if self._scene is None or self._torch is None:
            return
        if self._as019_failure_reason:
            if self._as019_wrench_active:
                try:
                    self._scene["peg"].permanent_wrench_composer.reset()
                except Exception:
                    pass
            self._as019_active_wrench_owners = ()
            self._as019_slider_orientation_active_owners = ()
            self._as019_slider_orientation_grace_owners = ()
            self._as019_wrench_active = False
            return
        if _AS019_RECEIVER_PINCH_FREEZE_ENABLED:
            self._update_receiver_pinch_freeze(_index)
        if (
            _AS019_RECEIVER_PINCH_FREEZE_ENABLED
            and self._as019_receiver_pinch_frozen
            and self._as019_stage < 4
            and self._as019_receiver_pinch_target is not None
        ):
            self._scene["left_robot"].set_joint_position_target(
                self._as019_receiver_pinch_target,
                joint_ids=list(self._hand_joint_ids["left"]),
            )
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        prehold_receiver_damping_only = False
        owners = [
            side for side in ("right", "left") if side in self._as019_assist_anchor
        ]
        live_metrics = self._as019_last_metrics
        if "right" in owners and (
            self._as019_stage >= 3
            or _grasp_command_max(self._right_hand)
            <= _HAND_OPEN_THRESHOLD
            or not bool(
                self._as019_last_metrics.get("right_tail_physical", False)
            )
        ):
            # The anchor records ordered grasp lineage, but it is never force
            # ownership by itself.  Every giver wrench requires the latest
            # detailed opposed tail contact; loss of that contact removes the
            # translation spring immediately.  A previously latched slider may
            # retain only its bounded orientation torque for the short grace
            # declared below.
            owners = [side for side in owners if side != "right"]
        if "left" in owners and (
            self._as019_stage == 0
            or self._as019_stage >= 5
            or not bool(live_metrics.get("contact_query_valid", False))
            or not bool(live_metrics.get("left_middle_physical", False))
            or _grasp_command_max(self._left_hand) < _HAND_CLOSE_THRESHOLD
        ):
            # A receiver anchor is ordered-event memory only.  Every receiver
            # wrench, in every stage, requires the latest detailed opposed
            # middle-handle contact and a currently closed grasp command.
            owners = [side for side in owners if side != "left"]
        if self._as019_stage == 1:
            # The receiver must establish its 12-frame support through live
            # physical contact, but averaging two freshly captured anchors
            # before that proof winds the cylindrical handle between small
            # DLS residuals.  Keep the live giver as the sole wrench source
            # until the receiver hold is authoritative.  If giver contact is
            # absent, neither semantic anchor may exert a wrench in stage 1.
            owners = ["right"] if "right" in owners else []
        if self._as019_stage == 2:
            # Semantic anchors preserve the ordered handover lineage across a
            # short contact flicker, but a separated hand must never continue
            # contributing an object wrench.  Gate each overlap owner by the
            # authoritative live PhysX force while leaving the anchor itself
            # latched for the release audit and robot-only recovery policy.
            metrics = live_metrics
            if not bool(metrics.get("right_tail_physical", False)):
                owners = [side for side in owners if side != "right"]
            if not bool(metrics.get("left_middle_physical", False)):
                owners = [side for side in owners if side != "left"]
            if (
                self._as019_receiver_hold_complete_step < 0
                and "right" in owners
            ):
                owners = ["right"]
            elif (
                self._as019_receiver_hold_complete_step < 0
                and "left" in owners
            ):
                # The receiver has physical contact but has not yet completed
                # the unchanged 12-frame support proof.  If giver contact
                # flickers here, damp the measured motion without applying the
                # receiver's early anchor spring; that pre-hold anchor can
                # carry a large reset-dependent rotation residual.  Once the
                # hold completes, the existing zero-jump rebind makes the
                # receiver the ordinary sole wrench owner.
                owners = ["left"]
                prehold_receiver_damping_only = True
        if (
            self._as019_stage == 2
            and self._as019_receiver_hold_complete_step >= 0
        ):
            # The required receiver hold has transferred the supported load.
            # Prefer the receiver as the sole wrench owner while its contact is
            # physical.  If that contact flickers while the opening giver still
            # has real contact, fall back to the giver instead of leaving its
            # residual fingers to push a completely ballistic tool.  Never
            # average the two anchors after transfer, and switch back to the
            # receiver immediately when its contact returns.
            if "left" in owners:
                if not self._as019_stage2_left_contact_active:
                    left_anchor = self._as019_assist_anchor.get("left")
                    if left_anchor is not None:
                        # The tool may move ballistically during a genuine
                        # contact gap.  Rebind on the first authoritative
                        # receiver re-contact so restoring support is
                        # zero-jump instead of pulling toward the stale
                        # pre-gap transform.
                        left_anchor["hand_to_tool"] = _invert_pose_matrix(
                            self._palm_matrix("left")
                        ) @ self._tool_matrix()
                        left_anchor["attached_step"] = int(self.env_step)
                self._as019_stage2_left_contact_active = True
                owners = ["left"]
            elif "right" in owners:
                self._as019_stage2_left_contact_active = False
                owners = ["right"]
            else:
                self._as019_stage2_left_contact_active = False
                owners = []
        if self._as019_stage == 3 and owners == ["left"]:
            metrics = live_metrics
            physical_left_contact = bool(
                metrics.get("contact_query_valid", False)
                and metrics.get("left_middle_physical", False)
            )
            if (
                not bool(metrics.get("right_clear", False))
                or not physical_left_contact
                or (
                    self._as019_placement_motion_started
                    and not self._as019_placement_assist_enabled
                )
            ):
                # Stage 3 is entered only after the giver-clear debounce.  The
                # receiver may keep contact-owned support while the giver
                # retreats, but a giver re-contact or any loss of receiver
                # contact removes the wrench on the next authoritative sample.
                # This remains true after placement starts: the robot policy
                # pauses and seeks contact, while the tool itself is ballistic.
                owners = []
        if (
            self._as019_stage >= 3
            and owners == ["left"]
            and _grasp_command_max(self._left_hand) <= _HAND_OPEN_THRESHOLD
        ):
            # The latched anchor remains as semantic ownership until the
            # physical contact view is clear, but an open-hand command must
            # immediately remove every task wrench.  This separates force
            # ownership from the ordered contact-clear -> owner-clear audit.
            owners = []
        stage4_left_owned = bool(
            self._as019_stage == 4 and owners == ["left"]
        )
        release_support_owned = bool(
            stage4_left_owned and self._as019_release_support_enabled
        )
        if stage4_left_owned:
            metrics = live_metrics
            physical_left_contact = bool(
                metrics.get("contact_query_valid", False)
                and metrics.get("left_middle_physical", False)
            )
            if (
                (
                    not self._as019_slider_orientation_latched
                    and not (
                        self._as019_placement_assist_enabled
                        or self._as019_release_support_enabled
                    )
                )
                or not physical_left_contact
            ):
                # The slider path remains available to a manual receiver in
                # stage 4 even though automatic-policy placement flags are not
                # set.  Every path is still legal only while receiver contact
                # is physical; once it clears the tool is strictly ballistic.
                owners = []
                release_support_owned = False
        slider_orientation_grace_owners = (
            self._slider_orientation_grace_owner_sides(live_metrics)
        )
        slider_orientation_owners = tuple(owners) or tuple(
            slider_orientation_grace_owners
        )
        slider_orientation_owned = bool(
            self._as019_slider_orientation_latched
            and self._as019_stage >= 1
            and slider_orientation_owners
        )
        self._as019_slider_orientation_active_owners = (
            slider_orientation_owners if slider_orientation_owned else ()
        )
        self._as019_slider_orientation_grace_owners = (
            tuple(
                side
                for side in slider_orientation_owners
                if side not in owners
            )
            if slider_orientation_owned
            else ()
        )
        if owners or slider_orientation_owned:
            current = self._tool_matrix()
            velocity = self._tool_velocity()
            targets = [
                self._palm_matrix(side)
                @ self._as019_assist_anchor[side]["hand_to_tool"]
                for side in owners
            ]
            # The overlap phase is a physical two-hand constraint.  Average
            # both contact-owned targets instead of silently letting the right
            # anchor win while the left fingers twist the tool against it.
            position_error = (
                np.mean(
                    [
                        target[:3, 3] - current[:3, 3]
                        for target in targets
                    ],
                    axis=0,
                )
                if targets
                else np.zeros(3, dtype=np.float64)
            )
            right_only_handover = bool(
                self._as019_stage <= 1 and owners == ["right"]
            )
            handover_owned = bool(self._as019_stage <= 2)
            placement_owned = bool(self._as019_stage >= 3 and owners == ["left"])
            if slider_orientation_owned:
                # The virtual slider has three translational degrees of
                # freedom and no rotational degree of freedom.  Correct the
                # complete rotation (including longitudinal roll) toward the
                # one canonical horizontal-left task frame.
                rotation_error = _rotation_error_vector(
                    self._as019_outward_reference_rotation,
                    current[:3, :3],
                )
            else:
                rotation_error = np.mean(
                    [
                        _axis_alignment_error_vector(
                            target[:3, 0], current[:3, 0]
                        )
                        if handover_owned or placement_owned
                        else _rotation_error_vector(
                            target[:3, :3], current[:3, :3]
                        )
                        for target in targets
                    ],
                    axis=0,
                )
            if prehold_receiver_damping_only:
                linear_kp = 0.0
                linear_kd = _ASSIST_LINEAR_KD
                angular_kp = 0.0
                angular_kd = _ASSIST_ANGULAR_KD
                outward_kp = _RECEIVER_RETRY_HANDOVER_HOLD_OUTWARD_KP
                max_force = _ASSIST_MAX_FORCE_N
                max_torque = _HIGH_HANDOVER_ASSIST_MAX_TORQUE_NM
            elif release_support_owned:
                # Once the tool is table-supported, use this tier only as
                # contact-gated velocity damping.  An anchor-position spring
                # fights a firmly closed hand and sustains the very angular
                # jitter that the unchanged quiet-release gate is measuring.
                linear_kp = 0.0
                linear_kd = _RELEASE_SUPPORT_LINEAR_KD
                angular_kp = 0.0
                angular_kd = _RELEASE_SUPPORT_ANGULAR_KD
                outward_kp = 0.0
                max_force = _RELEASE_SUPPORT_MAX_FORCE_N
                max_torque = _RELEASE_SUPPORT_MAX_TORQUE_NM
            elif placement_owned:
                linear_kp = _PLACEMENT_ASSIST_LINEAR_KP
                linear_kd = _PLACEMENT_ASSIST_LINEAR_KD
                angular_kp = _PLACEMENT_ASSIST_ANGULAR_KP
                angular_kd = _PLACEMENT_ASSIST_ANGULAR_KD
                outward_kp = _PLACEMENT_ASSIST_OUTWARD_KP
                max_force = _PLACEMENT_ASSIST_MAX_FORCE_N
                max_torque = _PLACEMENT_ASSIST_MAX_TORQUE_NM
            else:
                linear_kp = _ASSIST_LINEAR_KP
                linear_kd = _ASSIST_LINEAR_KD
                angular_kp = _ASSIST_ANGULAR_KP
                angular_kd = _ASSIST_ANGULAR_KD
                outward_kp = _ASSIST_OUTWARD_KP
                max_force = _ASSIST_MAX_FORCE_N
                max_torque = _ASSIST_MAX_TORQUE_NM
                if (
                    self._as019_stage <= 2
                    and bool(self._as019_high_handover_assist_tier)
                ):
                    max_torque = _HIGH_HANDOVER_ASSIST_MAX_TORQUE_NM
            if self._as019_stage == 0 and owners == ["right"]:
                # The re-seat policy now follows each authoritative
                # re-contact anchor, so a modest restorative term can correct
                # reset yaw without commanding the wrist to roll the thin
                # cylinder against the tabletop.  The selected branch torque
                # cap still bounds the physical load.
                outward_kp = _STAGE0_OUTWARD_KP
            elif self._as019_stage == 1 and owners == ["right"]:
                # Once the grasp event is accepted, a bounded restorative term
                # prevents wrist/solver drift across the 30-degree
                # safety boundary.  The selected branch torque cap still
                # limits the physical load on the handle contact.
                outward_kp = _HANDOVER_HOLD_OUTWARD_KP
                if bool(self._as019_high_handover_assist_tier):
                    outward_kp = _HIGH_HANDOVER_HOLD_OUTWARD_KP
                if bool(self._as019_receiver_retry_assist_tier):
                    # A one-sided receiver brush can leave the giver-held tool
                    # with residual angular velocity after contact clears.
                    # Use the existing 3.5 Nm bounded tier with stronger
                    # safety-axis authority only during that retry window.
                    outward_kp = _RECEIVER_RETRY_HANDOVER_HOLD_OUTWARD_KP
            if slider_orientation_owned:
                angular_kp = _SLIDER_ORIENTATION_ANGULAR_KP
                angular_kd = _SLIDER_ORIENTATION_ANGULAR_KD
                outward_kp = 0.0
                max_torque = _SLIDER_ORIENTATION_MAX_TORQUE_NM
                if not owners:
                    # Contact grace corrects rotation only.  Never let a stale
                    # hand-to-tool transform carry or translate the object.
                    linear_kp = 0.0
                    linear_kd = 0.0
            force = _cap_norm(
                linear_kp * position_error - linear_kd * velocity[:3],
                max_force,
            )
            if slider_orientation_owned:
                outward_error = np.zeros(3, dtype=np.float64)
            elif placement_owned or handover_owned:
                # A single physical grasp does not constrain longitudinal roll
                # reliably, and even two real hands retain some cylindrical
                # roll freedom during transfer.  Correct only the task's
                # safety-critical +X direction throughout handover so an
                # arbitrary roll target cannot fight the fingers.  Placement
                # intentionally keeps the same freedom.
                outward_error = _axis_alignment_error_vector(
                    self._as019_outward_reference_rotation[:, 0],
                    current[:3, 0],
                )
            else:
                outward_error = _rotation_error_vector(
                    self._as019_outward_reference_rotation,
                    current[:3, :3],
                )
            uncapped_torque = (
                angular_kp * rotation_error
                + outward_kp * outward_error
                - angular_kd * velocity[3:]
            )
            if slider_orientation_owned:
                # The full three-axis rotation error is authoritative, so no
                # longitudinal component may be discarded by the old
                # safety-axis-prioritizing cap.
                torque = _cap_norm(uncapped_torque, max_torque)
            elif handover_owned or placement_owned:
                # Rolling a cylindrical handle is harmless to the +X safety
                # direction.  Keep the unchanged total torque cap, but do not
                # let longitudinal roll damping consume the authority needed
                # to prevent the sharp tip from crossing the safety boundary.
                # Placement preserves the same longitudinal freedom and uses
                # the same safety-prioritized cap at its lower 0.8 Nm limit.
                torque = _cap_torque_preserving_axis_alignment(
                    uncapped_torque,
                    current[:3, 0],
                    max_torque,
                )
            else:
                torque = _cap_norm(uncapped_torque, max_torque)
        wrench_active = bool(owners or slider_orientation_owned)
        self._as019_active_wrench_owners = tuple(owners)
        self._as019_active_wrench_gate_snapshot = {
            "control_step_before": int(self.env_step),
            "physics_substep_index": int(_index),
            "physics_substep_count": int(_count),
            "metrics_generation": int(
                live_metrics.get("metrics_generation", -1)
            ),
            "stage": int(self._as019_stage),
            "slider_orientation_latched": bool(
                self._as019_slider_orientation_latched
            ),
            "slider_orientation_active": bool(slider_orientation_owned),
            "slider_orientation_active_owners": tuple(
                self._as019_slider_orientation_active_owners
            ),
            "slider_orientation_grace_owners": tuple(
                self._as019_slider_orientation_grace_owners
            ),
            "slider_owner_last_live_step": dict(
                self._as019_slider_owner_last_live_step
            ),
            # Preserve the decision-time owner lineage even when the backend
            # writer below fails and live ownership is cleared for safety.
            "active_wrench_owners": tuple(owners),
            "contact_query_valid": bool(
                live_metrics.get("contact_query_valid", False)
            ),
            "right_tail_physical": bool(
                live_metrics.get("right_tail_physical", False)
            ),
            "right_close_command": bool(
                _grasp_command_max(self._right_hand)
                > _HAND_OPEN_THRESHOLD
            ),
            "right_wrench_invariant": bool(
                "right" not in owners
                or (
                    self._as019_stage <= 2
                    and bool(
                        live_metrics.get("right_tail_physical", False)
                    )
                    and _grasp_command_max(self._right_hand)
                    > _HAND_OPEN_THRESHOLD
                )
            ),
            "left_middle_physical": bool(
                live_metrics.get("contact_query_valid", False)
                and live_metrics.get("left_middle_physical", False)
            ),
            "left_close_command": bool(
                _grasp_command_max(self._left_hand)
                >= _HAND_CLOSE_THRESHOLD
            ),
            "left_wrench_invariant": bool(
                "left" not in owners
                or (
                    bool(live_metrics.get("contact_query_valid", False))
                    and bool(
                        live_metrics.get("left_middle_physical", False)
                    )
                    and _grasp_command_max(self._left_hand)
                    >= _HAND_CLOSE_THRESHOLD
                )
            ),
            "apply_succeeded": False,
        }
        if not wrench_active and not self._as019_wrench_active:
            return
        try:
            if wrench_active:
                device = _torch_view(
                    self._scene["peg"].data.root_state_w
                ).device
                self._scene["peg"].set_external_force_and_torque(
                    self._torch.as_tensor(
                        force, dtype=self._torch.float32, device=device
                    ).reshape(1, 1, 3),
                    self._torch.as_tensor(
                        torque, dtype=self._torch.float32, device=device
                    ).reshape(1, 1, 3),
                    is_global=True,
                )
            else:
                # Deactivate the permanent writer once no hand owns the tool.
                # This is intentionally stronger than writing a zero wrench:
                # an active zero writer still wakes a resting rigid body.
                self._scene["peg"].permanent_wrench_composer.reset()
            self._as019_wrench_active = wrench_active
            self._as019_active_wrench_gate_snapshot[
                "apply_succeeded"
            ] = True
        except Exception as exc:
            try:
                self._scene["peg"].permanent_wrench_composer.reset()
            except Exception:
                pass
            self._as019_active_wrench_owners = ()
            self._as019_slider_orientation_active_owners = ()
            self._as019_slider_orientation_grace_owners = ()
            self._as019_wrench_active = False
            self._as019_active_wrench_gate_snapshot[
                "apply_succeeded"
            ] = False
            message = f"AS019 compliant grasp force failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)

    def _as019_post_substep(self, _index: int, _count: int) -> None:
        self._refresh_object_poses()
        metrics = self._metrics()
        if int(_index) + 1 >= int(_count):
            self._latch_table_safety_failure(metrics)
        if not self._as019_failure_reason:
            self._update_assist_ownership(metrics)
            self._update_slider_orientation_guide(metrics)
        latches = self._as019_substep_latches
        latches["tip_left_palm"] |= bool(metrics["tip_left_palm"])
        if self._as019_stage == 0:
            latches["right_tail"] |= bool(
                metrics.get(
                    "right_handle_touch",
                    metrics.get("right_tail_physical", False),
                )
                and (
                    not self._as019_automatic_acquire_gate_required
                    or self._as019_automatic_acquire_gate_open
                )
            )
        elif self._as019_stage == 1:
            latches["dual_contact"] |= bool(
                metrics.get(
                    "left_any_touch",
                    metrics.get(
                        "left_handle_touch",
                        metrics.get("left_middle_physical", False),
                    ),
                )
                and float(metrics.get("tool_table_analytic_gap_m", -math.inf))
                >= _RIGHT_LIFT_MIN_TABLE_GAP_M
            )
        elif self._as019_stage == 2:
            latches["zone_placed"] |= bool(
                metrics["table_safety_query_valid"]
                and metrics["tool_table_supported"]
                and metrics["zone_inside"]
            )
        self._as019_last_metrics = metrics

    def _clear_handover_progress_after_drop(self, reason: str) -> None:
        """Clear latched OK rows after a recoverable out-of-zone drop."""

        if reason not in _PROGRESS_RESET_REASON_CODES or not reason:
            raise ValueError(f"Unknown AS019 progress reset reason: {reason}")
        self._as019_stage = 0
        self._as019_stage_name = "await_right_contact"
        self._as019_success = False
        self._as019_stability_counter = 0
        self._as019_receiver_hold_counter = 0
        self._as019_receiver_hold_complete_step = -1
        self._as019_stage2_left_contact_active = False
        self._as019_right_release_counter = 0
        self._as019_progress_reset_candidate_counter = 0
        self._as019_progress_reset_count += 1
        self._as019_last_progress_reset_reason = str(reason)
        self._as019_event_steps.fill(-1)
        self._as019_release_event_steps.fill(-1)
        self._as019_assist_anchor.clear()
        self._as019_active_wrench_owners = ()
        self._as019_wrench_active = False
        self._as019_slider_orientation_latched = False
        self._as019_slider_orientation_latched_step = -1
        self._as019_slider_orientation_active_owners = ()
        self._as019_slider_orientation_grace_owners = ()
        self._as019_slider_owner_last_live_step.clear()
        self._as019_slider_hand_orientation_locks.clear()
        self._as019_substep_latches = {
            key: False for key in self._as019_substep_latches
        }

    def _update_handover_progress_rollback(
        self, metrics: dict[str, Any]
    ) -> bool:
        """Debounce the two operator-requested progress rollback cases."""

        contact_valid = bool(metrics.get("contact_query_valid", False))
        no_contact_owner = bool(
            contact_valid
            and metrics.get("right_clear", False)
            and metrics.get("left_clear", False)
            and not self._as019_assist_anchor
        )
        analytic_gap = float(
            metrics.get("tool_table_analytic_gap_m", math.inf)
        )
        landed_or_fallen = bool(
            metrics.get("tool_table_supported", False)
            or analytic_gap <= _TABLE_SUPPORT_MAX_SEPARATION_M
            or float(self._peg_ori_pose[2]) < 0.90
        )
        reason = ""
        if self._as019_stage == 2 and no_contact_owner and landed_or_fallen:
            reason = "dropped_before_left_only_hold"
        elif (
            self._as019_stage in (3, 4)
            and no_contact_owner
            and landed_or_fallen
            and not bool(metrics.get("zone_inside", False))
        ):
            reason = "dropped_outside_target_zone"

        if not reason:
            self._as019_progress_reset_candidate_counter = 0
            return False
        self._as019_progress_reset_candidate_counter += 1
        if (
            self._as019_progress_reset_candidate_counter
            < _PROGRESS_RESET_DEBOUNCE_STEPS
        ):
            return False
        self._clear_handover_progress_after_drop(reason)
        return True

    def _compute_success(self) -> bool:
        # The fourth post-physics callback already sampled the final 120 Hz
        # substep.  Reuse that exact snapshot for the 30 Hz transition logic;
        # querying PhysX again here duplicated every contact-buffer transfer
        # without advancing simulation time.
        metrics = self._as019_last_metrics
        if not metrics:
            metrics = self._metrics()
            self._as019_last_metrics = metrics
        self._latch_table_safety_failure(metrics)
        if not self._as019_failure_reason:
            self._update_assist_ownership(metrics)
            self._update_slider_orientation_guide(metrics)
        self._update_release_event_steps(metrics)
        latches = self._as019_substep_latches
        # AS019 now has only three ordered success events.  Contact location,
        # simultaneous touch, left-only holding, tip orientation, and a final
        # stability window are deliberately not success gates.
        if not self._as019_failure_reason:
            if (
                self._as019_stage == 0
                and latches.get("right_tail")
                and (
                    not self._as019_automatic_acquire_gate_required
                    or self._as019_automatic_acquire_gate_open
                )
            ):
                self._as019_stage = 1
                self._as019_stage_name = "await_left_airborne_contact"
                self._as019_event_steps[0] = self.env_step
            elif (
                self._as019_stage == 1
                and latches.get("dual_contact")
            ):
                self._as019_stage = 2
                self._as019_stage_name = "await_zone_landing"
                self._as019_event_steps[1] = self.env_step
            elif (
                self._as019_stage == 2
                and latches.get("zone_placed")
            ):
                self._as019_stage = 5
                self._as019_stage_name = "complete"
                self._as019_event_steps[2] = self.env_step
                # Preserve the legacy hidden-state lineage slots for replay
                # consumers while making the third event terminal immediately.
                self._as019_event_steps[3] = self.env_step
                self._as019_event_steps[4] = self.env_step
                self._as019_success = True

        if not self._as019_failure_reason:
            # A stage-0 right grasp is promoted to stage 1 in this method,
            # after the final post-substep sample.  Refresh once more so the
            # guide and its wrist lock are visible before the next action.
            self._update_slider_orientation_guide(metrics)

        self._record_release_gate_frame(metrics)
        self._as019_substep_latches = {key: False for key in latches}
        self._as019_last_metrics = metrics
        return bool(self._as019_success)

    def _update_release_event_steps(self, metrics: dict[str, Any]) -> None:
        """Latch the physical release lineage one control frame at a time."""

        if self._as019_stage < 4:
            return
        steps = self._as019_release_event_steps
        left_open_commanded = bool(
            float(np.max(np.abs(self._left_hand))) <= _HAND_OPEN_THRESHOLD
        )
        if steps[0] < 0 and left_open_commanded:
            steps[0] = self.env_step
        if (
            steps[0] >= 0
            and steps[0] < self.env_step
            and steps[1] < 0
            and bool(metrics.get("contact_query_valid", False))
            and bool(metrics.get("left_clear", False))
        ):
            steps[1] = self.env_step
        if (
            steps[1] >= 0
            and steps[1] < self.env_step
            and steps[2] < 0
            and "left" not in self._as019_assist_anchor
        ):
            steps[2] = self.env_step

    def _compute_failure(self) -> bool:
        if self._as019_failure_reason:
            return True
        if self._as019_last_metrics:
            self._latch_table_safety_failure(self._as019_last_metrics)
        if self._as019_failure_reason:
            return True
        pose = np.asarray(self._peg_ori_pose, dtype=np.float64)
        if not np.all(np.isfinite(pose)):
            self._as019_failure_reason = "non_finite_tool_state"
        elif pose[2] < 0.84:
            self._as019_failure_reason = "tool_dropped_below_table"
        elif abs(float(pose[0])) > 0.65 or abs(float(pose[1])) > 0.85:
            self._as019_failure_reason = "tool_left_workspace"
        return bool(self._as019_failure_reason)

    def _update_inward_orientation_failure_guard(self) -> None:
        """Latch only a sustained post-grasp violation of the safe tip cone."""

        current_step = int(self.env_step)
        if self._as019_inward_orientation_guard_last_step == current_step:
            return
        self._as019_inward_orientation_guard_last_step = current_step
        if self._as019_stage <= 0 or self._as019_stage >= 5:
            self._as019_inward_orientation_failure_counter = 0
            return
        if self._outward_alignment() >= _OUTWARD_COS:
            self._as019_inward_orientation_failure_counter = 0
            return
        self._as019_inward_orientation_failure_counter += 1
        if (
            self._as019_inward_orientation_failure_counter
            >= _SHARP_TIP_INWARD_DEBOUNCE_STEPS
        ):
            self._as019_failure_reason = "sharp_tip_turned_inward"

    def _left_hand_joint_position_telemetry(self) -> tuple[list[float], str]:
        """Read the physical Wuji joints without changing controller state."""

        joint_ids = list(self._hand_joint_ids.get("left", ()))
        if self._scene is None or self._torch is None or not joint_ids:
            return [], "joint_state_unavailable"
        try:
            robot = self._control_robot("left")
            values = _torch_view(robot.data.joint_pos)[0, joint_ids]
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            return np.asarray(values, dtype=np.float64).reshape(-1).tolist(), ""
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"

    def _record_release_gate_frame(self, metrics: dict[str, Any]) -> None:
        """Retain every release-stage frame for the validation runner."""

        if self._as019_stage < 4 or self._as019_release_trace_last_step == self.env_step:
            return
        self._update_release_event_steps(metrics)
        frame = {
            "step": int(self.env_step),
            "event_steps": self._as019_event_steps.tolist(),
            **self._release_gate_telemetry(metrics),
        }
        frame["release_policy_action_trace_aligned"] = bool(
            frame["release_policy_action_step"] == frame["step"]
            and frame["release_policy_action_dim"] == 54
        )
        self._as019_release_frame_trace.append(frame)
        self._as019_release_trace_last_step = int(self.env_step)
        overflow = len(self._as019_release_frame_trace) - _RELEASE_TELEMETRY_MAX_FRAMES
        if overflow > 0:
            del self._as019_release_frame_trace[:overflow]
            self._as019_release_trace_dropped_frames += int(overflow)

    def _release_gate_telemetry(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Expose the ordered release gates without changing their policy."""

        left_hand_command = float(np.max(np.abs(self._left_hand)))
        left_hand_action = np.asarray(self._left_hand, dtype=np.float64).reshape(-1)
        left_hand_joints, left_hand_joint_error = (
            self._left_hand_joint_position_telemetry()
        )
        policy_action = np.asarray(
            self._as019_last_policy_action, dtype=np.float32
        ).reshape(-1)
        release_event_steps = self._as019_release_event_steps.tolist()
        left_owner = "left" in self._as019_assist_anchor
        release_action_started = bool(self._as019_automatic_phase >= 11)
        left_open_commanded = bool(left_hand_command <= _HAND_OPEN_THRESHOLD)
        left_contact_clear = bool(metrics.get("left_clear", False))
        zone_inside = bool(metrics.get("zone_inside", False))
        outward_alignment = float(metrics.get("outward_alignment", -float("inf")))
        right_clear = bool(metrics.get("right_clear", False))
        linear_speed = float(metrics.get("linear_speed", float("inf")))
        angular_speed = float(metrics.get("angular_speed", float("inf")))
        all_owners_clear = not self._as019_assist_anchor
        contact_query_valid = bool(metrics.get("contact_query_valid", False))
        table_safety_query_valid = bool(
            metrics.get("table_safety_query_valid", False)
        )
        tool_table_supported = bool(
            metrics.get("tool_table_supported", False)
        )

        release_ready_gates = {
            "contact_query_valid": contact_query_valid,
            "table_safety_query_valid": table_safety_query_valid,
            "tool_table_supported": tool_table_supported,
            "zone_inside": zone_inside,
            "linear_speed": linear_speed <= _ZONE_RELEASE_MAX_LINEAR_SPEED_MPS,
            "angular_speed": angular_speed <= _ZONE_RELEASE_MAX_ANGULAR_SPEED_RADPS,
            "right_clear": right_clear,
            "left_owner_latched": left_owner,
            "left_middle_physical": bool(
                contact_query_valid
                and metrics.get("left_middle_physical", False)
            ),
            "left_hand_closed": left_hand_command >= _HAND_CLOSE_THRESHOLD,
        }
        success_gates = {
            "contact_query_valid": contact_query_valid,
            "table_safety_query_valid": table_safety_query_valid,
            "tool_table_supported": tool_table_supported,
            "zone_inside": zone_inside,
            "outward": outward_alignment >= _OUTWARD_COS,
            "right_clear": right_clear,
            "left_contact_clear": left_contact_clear,
            "linear_speed": linear_speed <= _ZONE_RELEASE_MAX_LINEAR_SPEED_MPS,
            "angular_speed": angular_speed <= _ZONE_RELEASE_MAX_ANGULAR_SPEED_RADPS,
            "all_contact_owners_clear": all_owners_clear,
        }

        first_unmet_gate = ""
        if self._as019_stage < 4:
            first_unmet_gate = "zone_placed_event"
        elif not release_action_started:
            for gate in (
                "contact_query_valid",
                "table_safety_query_valid",
                "tool_table_supported",
                "zone_inside",
                "linear_speed",
                "angular_speed",
                "right_clear",
                "left_owner_latched",
                "left_middle_physical",
                "left_hand_closed",
            ):
                if not release_ready_gates[gate]:
                    first_unmet_gate = f"release_ready_{gate}"
                    break
            else:
                first_unmet_gate = "release_action_started"
        elif release_event_steps[0] < 0:
            first_unmet_gate = "left_open_command"
        elif release_event_steps[1] < 0:
            first_unmet_gate = "left_contact_clear"
        elif release_event_steps[2] < 0:
            first_unmet_gate = "left_owner_clear"
        else:
            for gate in (
                "contact_query_valid",
                "table_safety_query_valid",
                "tool_table_supported",
                "zone_inside",
                "outward",
                "right_clear",
                "linear_speed",
                "angular_speed",
                "all_contact_owners_clear",
            ):
                if not success_gates[gate]:
                    first_unmet_gate = gate
                    break
            else:
                if self._as019_stability_counter < _STABILITY_STEPS:
                    first_unmet_gate = f"stable_{_STABILITY_STEPS}_steps"

        return {
            "release_gate_first_unmet": first_unmet_gate,
            "release_ready_all": bool(all(release_ready_gates.values())),
            **{
                f"release_ready_{key}": value
                for key, value in release_ready_gates.items()
            },
            "release_action_started": release_action_started,
            "release_action_phase": int(self._as019_automatic_phase),
            "release_policy_action_step": int(
                self._as019_last_policy_action_step
            ),
            "release_policy_action_dim": int(policy_action.size),
            "release_policy_action_54d": policy_action.tolist(),
            "release_left_hand_action_dim": int(left_hand_action.size),
            "release_left_hand_action_20d": left_hand_action.tolist(),
            "release_left_hand_command_max_abs": left_hand_command,
            "release_left_hand_joint_count": len(left_hand_joints),
            "release_left_hand_joint_positions": left_hand_joints,
            "release_left_hand_joint_state_error": left_hand_joint_error,
            "release_left_open_commanded": left_open_commanded,
            "release_left_middle_contact": bool(metrics.get("left_middle", False)),
            "release_left_force_n": float(metrics.get("left_force", 0.0)),
            "release_left_contact_clear": left_contact_clear,
            "release_left_owner_latched": left_owner,
            "release_support_enabled": bool(self._as019_release_support_enabled),
            "release_contact_query_valid": contact_query_valid,
            "release_contact_query_error": str(
                metrics.get("contact_query_error", "")
            ),
            "release_event_steps": release_event_steps,
            "release_lineage_complete": bool(
                all(step >= 0 for step in release_event_steps[:3])
            ),
            **{f"release_success_{key}": value for key, value in success_gates.items()},
            "release_stability_steps": int(self._as019_stability_counter),
            "release_stability_required_steps": int(_STABILITY_STEPS),
        }

    def _handover_metric_vector(self) -> np.ndarray:
        metrics = self._as019_last_metrics or self._metrics()
        return np.asarray(
            [
                metrics.get("right_force", 0.0),
                metrics.get("left_force", 0.0),
                metrics.get("left_palm_force", 0.0),
                metrics.get("right_local_x", 0.0),
                metrics.get("left_local_x", 0.0),
                metrics.get("outward_alignment", 0.0),
                metrics.get("tip_left_palm_distance", 0.0),
                float(bool(metrics.get("zone_inside", False))),
                metrics.get("linear_speed", 0.0),
                metrics.get("angular_speed", 0.0),
                float("right" in self._as019_assist_anchor),
                float("left" in self._as019_assist_anchor),
                float(self._as019_right_release_counter),
                float(self._as019_stability_counter),
                float(
                    self._as019_contact_available
                    and self._as019_table_safety_available
                ),
                float(bool(self._as019_failure_reason)),
            ],
            dtype=np.float64,
        )

    def _hidden_state_vector(self) -> np.ndarray:
        slider_lock_values: list[float] = []
        for side in ("right", "left"):
            quaternion = self._as019_slider_hand_orientation_locks.get(side)
            slider_lock_values.append(float(quaternion is not None))
            slider_lock_values.extend(
                np.zeros(4, dtype=np.float64)
                if quaternion is None
                else np.asarray(quaternion, dtype=np.float64).reshape(4)
            )
        return np.concatenate(
            [
                np.asarray(
                    [
                        self._as019_stage,
                        self._as019_receiver_hold_counter,
                        self._as019_receiver_hold_complete_step,
                        self._as019_right_release_counter,
                        self._as019_stability_counter,
                        self._as019_progress_reset_candidate_counter,
                        self._as019_progress_reset_count,
                        _PROGRESS_RESET_REASON_CODES.get(
                            self._as019_last_progress_reset_reason,
                            -1,
                        ),
                        int(self._as019_success),
                        int(bool(self._as019_failure_reason)),
                        int("right" in self._as019_assist_anchor),
                        int("left" in self._as019_assist_anchor),
                        int("right" in self._as019_active_wrench_owners),
                        int("left" in self._as019_active_wrench_owners),
                        int(
                            bool(
                                self._as019_active_wrench_gate_snapshot.get(
                                    "right_tail_physical", False
                                )
                            )
                        ),
                        int(
                            "right" not in self._as019_active_wrench_owners
                            or bool(
                                self._as019_active_wrench_gate_snapshot.get(
                                    "right_wrench_invariant", False
                                )
                            )
                        ),
                        self._as019_automatic_phase,
                        self._as019_mass_kg,
                        self._as019_friction,
                        self.env_step,
                        int(self._as019_table_safety_guard_armed),
                        _TABLE_SAFETY_FAILURE_CODES.get(
                            self._as019_table_safety_failure_candidate,
                            -1,
                        ),
                        self._as019_table_safety_failure_counter,
                        self._as019_table_safety_guard_last_step,
                        self._as019_inward_orientation_failure_counter,
                        self._as019_inward_orientation_guard_last_step,
                    ],
                    dtype=np.float64,
                ),
                self._as019_event_steps.astype(np.float64),
                np.asarray(
                    [
                        int(self._as019_slider_orientation_latched),
                        self._as019_slider_orientation_latched_step,
                        int(
                            "right"
                            in self._as019_slider_orientation_active_owners
                        ),
                        int(
                            "left"
                            in self._as019_slider_orientation_active_owners
                        ),
                        int(
                            "right"
                            in self._as019_slider_orientation_grace_owners
                        ),
                        int(
                            "left"
                            in self._as019_slider_orientation_grace_owners
                        ),
                        self._as019_slider_owner_last_live_step.get(
                            "right", -1
                        ),
                        self._as019_slider_owner_last_live_step.get(
                            "left", -1
                        ),
                        *slider_lock_values,
                    ],
                    dtype=np.float64,
                ),
            ]
        )

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        state = observation["state"]
        state.pop("peg_ori_pose", None)
        state.pop("socket_ori_pose", None)
        state.update(
            {
                "screwdriver_pose": self._peg_ori_pose.astype(np.float64, copy=True),
                "tool_zone_pose": self._socket_ori_pose.astype(np.float64, copy=True),
                "screwdriver_velocity": self._tool_velocity(),
                "handover_metrics": self._handover_metric_vector(),
                "handover_stage": np.asarray([self._as019_stage], dtype=np.float64),
            }
        )
        observation["as019_hidden_state"] = self._hidden_state_vector()
        return observation

    def get_recording_schema(self) -> dict[str, Any] | None:
        """Declare the slider/assist state as read-only replay evidence."""

        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        raw_fields = result.get("fields")
        if not isinstance(raw_fields, dict):
            raise RuntimeError("AS019 recording schema has no PhysX fields")
        fields = {str(name): dict(spec) for name, spec in raw_fields.items()}
        hidden_space = self.observation_space.spaces.get(
            "as019_hidden_state"
        )
        if not isinstance(hidden_space, spaces.Box):
            raise RuntimeError(
                "AS019 recording observation space has no hidden state"
            )
        fields["as019_hidden_state"] = {
            "shape": list(hidden_space.shape),
            "dtype": hidden_space.dtype.name,
            "source": "as019_action_aligned_task_state",
            "layout": list(AS019_HIDDEN_STATE_FIELDS),
            "visibility": "hidden_task_logic",
            "replay_domain": "task_state",
            "physical_replay": True,
            "replay_restore_supported": False,
            "replay_atol": _AS019_HIDDEN_STATE_REPLAY_ATOL,
        }
        result["fields"] = fields
        result["task_state"] = {
            "task_name": TASK_NAME,
            "field_names": ["as019_hidden_state"],
            "hidden_field_names": ["as019_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return result

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        # Reuse the metrics already sampled by the task step.  Querying PhysX
        # again from a UI refresh would add a second contact read and CPU sync
        # to every control frame.
        metrics = dict(self._as019_last_metrics)
        event_steps = np.asarray(self._as019_event_steps, dtype=np.int64).reshape(-1)

        def event_seen(index: int) -> bool:
            return bool(index < event_steps.size and int(event_steps[index]) >= 0)

        zone_landed = bool(
            event_seen(2)
            and bool(metrics.get("table_safety_query_valid", False))
            and bool(metrics.get("tool_table_supported", False))
            and bool(metrics.get("zone_inside", False))
        )
        values = (
            event_seen(0),
            event_seen(1),
            zone_landed or bool(self._as019_success),
        )
        return [
            {"key": key, "label": label, "ok": bool(ok)}
            for key, label, ok in zip(
                _AS019_SUCCESS_CONDITION_KEYS,
                AS019_SUCCESS_CONDITION_LABELS,
                values,
            )
        ]

    def _task_status_text(self) -> str:
        if self._as019_failure_reason:
            reason = self._as019_failure_reason.replace("_", " ")
            return f"FAILED | {reason}"
        if self._as019_success:
            return "COMPLETE | screwdriver landed in target zone"

        stage = max(0, min(5, int(self._as019_stage)))
        detail = {
            0: "right hand touches the screwdriver first",
            1: "left hand touches the screwdriver while it is airborne",
            2: "drop the screwdriver into the target zone",
            5: "screwdriver landed in target zone",
        }.get(stage, str(self._as019_stage_name).replace("_", " "))
        return f"Stage {stage}/5 | {detail}"

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        return str(row.get("label") or row.get("key") or "Condition")

    def _success_condition_overlay_warning(self, message: str) -> None:
        if self._success_condition_overlay_failed:
            return
        self._success_condition_overlay_failed = True
        if message not in self._isaac_warnings:
            self._isaac_warnings.append(message)
        print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _init_success_condition_overlay(self) -> None:
        if (
            self._headless
            or not self._launch_isaac
            or self._success_condition_window is not None
            or self._success_condition_frame is not None
        ):
            return
        try:
            import omni.ui as ui

            viewport_window = None
            try:
                from omni.kit.viewport.utility import get_active_viewport_window

                viewport_window = get_active_viewport_window()
            except Exception:
                viewport_window = None

            if viewport_window is not None:
                self._success_condition_frame = viewport_window.get_frame(
                    "handson.success_condition_overlay"
                )
                self._success_condition_frame.visible = True
                self._build_success_condition_overlay(
                    ui,
                    self._success_condition_frame,
                )
            else:
                panel_width = 520
                panel_height = 340
                self._success_condition_window = ui.Window(
                    "HandsOn",
                    width=panel_width,
                    height=panel_height,
                    visible=True,
                )
                for attr, value in (
                    (
                        "position_x",
                        max(20, int(self._render_width) - panel_width - 20),
                    ),
                    ("position_y", 30),
                ):
                    try:
                        setattr(self._success_condition_window, attr, value)
                    except Exception:
                        pass
                self._build_success_condition_overlay(
                    ui,
                    self._success_condition_window.frame,
                )
        except Exception as exc:
            self._success_condition_overlay_warning(
                f"Could not create AS019 success-condition overlay: {exc}"
            )

    def _build_success_condition_overlay(self, ui: Any, root: Any) -> None:
        if root is None:
            return
        self._success_condition_labels.clear()
        self._success_condition_text_labels.clear()
        self._task_status_label = None
        try:
            panel_width = 520
            panel_height = 340
            label_width = panel_width - 70
            with root:
                with ui.VStack():
                    ui.Spacer(height=30)
                    with ui.HStack():
                        ui.Spacer()
                        with ui.ZStack(width=panel_width, height=panel_height):
                            ui.Rectangle(
                                style={
                                    "background_color": 0xB0202020,
                                    "border_color": 0x70FFFFFF,
                                    "border_width": 1,
                                    "border_radius": 6,
                                }
                            )
                            with ui.VStack(spacing=6, style={"margin": 12}):
                                ui.Label("Success Conditions", height=24)
                                self._task_status_label = ui.Label(
                                    self._task_status_text(),
                                    height=28,
                                )
                                for row in self._success_condition_rows():
                                    with ui.HStack(height=22):
                                        condition_label = ui.Label(
                                            self._success_condition_display_label(row),
                                            width=label_width,
                                        )
                                        mark = ui.Label("", width=42)
                                        self._success_condition_text_labels[
                                            row["key"]
                                        ] = condition_label
                                        self._success_condition_labels[
                                            row["key"]
                                        ] = mark
                        ui.Spacer(width=20)
                    ui.Spacer()
        except Exception as exc:
            self._success_condition_overlay_warning(
                f"Could not build AS019 success-condition overlay: {exc}"
            )

    def _update_success_condition_overlay(self) -> None:
        if not self._success_condition_labels:
            return
        try:
            if self._success_condition_frame is not None:
                self._success_condition_frame.visible = True
            if self._success_condition_window is not None:
                self._success_condition_window.visible = True
            rows = self._success_condition_rows()
            if self._task_status_label is not None:
                self._task_status_label.text = self._task_status_text()
            for row in rows:
                condition_label = self._success_condition_text_labels.get(
                    row["key"]
                )
                if condition_label is not None:
                    condition_label.text = self._success_condition_display_label(row)
                mark = self._success_condition_labels.get(row["key"])
                if mark is not None:
                    mark.text = "OK" if row["ok"] else ""
        except Exception as exc:
            self._success_condition_overlay_warning(
                f"Could not update AS019 success-condition overlay: {exc}"
            )

    def _task_info(self) -> dict[str, Any]:
        metrics = self._as019_last_metrics or self._metrics()
        release_telemetry = self._release_gate_telemetry(metrics)
        return {
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "runtime_family": self.runtime_family,
            "physics_engine": self.physics_engine,
            "handover_stage": int(self._as019_stage),
            "handover_stage_name": self._as019_stage_name,
            "handover_event_steps": self._as019_event_steps.tolist(),
            "right_hand_contact_first": bool(
                self._as019_event_steps[0] >= 0
            ),
            "left_hand_airborne_contact": bool(
                self._as019_event_steps[1] >= 0
            ),
            "screwdriver_zone_landed": bool(
                self._as019_event_steps[2] >= 0
            ),
            "receiver_hold_counter": int(self._as019_receiver_hold_counter),
            "receiver_hold_required_steps": 0,
            "left_only_hold_counter": int(
                self._as019_right_release_counter
            ),
            "left_only_hold_required_steps": int(
                0
            ),
            "receiver_hold_complete_step": int(
                self._as019_receiver_hold_complete_step
            ),
            "failure_reason": self._as019_failure_reason,
            "progress_reset_candidate_counter": int(
                self._as019_progress_reset_candidate_counter
            ),
            "progress_reset_required_steps": int(
                0
            ),
            "progress_reset_count": int(self._as019_progress_reset_count),
            "last_progress_reset_reason": (
                self._as019_last_progress_reset_reason
            ),
            "table_safety_guard_armed": bool(
                self._as019_table_safety_guard_armed
            ),
            "table_safety_guard_arm_after_control_steps": int(
                _TABLE_SAFETY_ARM_AFTER_CONTROL_STEPS
            ),
            "table_safety_failure_candidate": (
                self._as019_table_safety_failure_candidate
            ),
            "table_safety_failure_counter": int(
                self._as019_table_safety_failure_counter
            ),
            "table_safety_failure_required_steps": int(
                _TABLE_SAFETY_FAILURE_DEBOUNCE_STEPS
            ),
            "inward_orientation_failure_counter": int(
                self._as019_inward_orientation_failure_counter
            ),
            "inward_orientation_failure_required_steps": int(
                _SHARP_TIP_INWARD_DEBOUNCE_STEPS
            ),
            "contact_owner_evidence_available": bool(self._as019_contact_available),
            "table_safety_evidence_available": bool(
                self._as019_table_safety_available
            ),
            "table_safety_exact_paths": dict(
                self._as019_table_safety_paths
            ),
            "compliant_grasp_assist": (
                "contact_gated_translation_and_orientation_assist_"
                "(not a success condition; no state write)"
            ),
            "compliant_grasp_owners": tuple(sorted(self._as019_assist_anchor)),
            "compliant_grasp_active_wrench_owners": tuple(
                sorted(self._as019_active_wrench_owners)
            ),
            "orientation_slider_mode": (
                "free_xyz_fixed_horizontal_left_rotation"
            ),
            "orientation_slider_lock_max_error_rad": float(
                _OUTWARD_ENTRY_MAX_ERROR_RAD
            ),
            "orientation_slider_lock_max_error_deg": float(
                np.rad2deg(_OUTWARD_ENTRY_MAX_ERROR_RAD)
            ),
            "orientation_slider_min_table_gap_m": float(
                _SLIDER_LATCH_MIN_TABLE_GAP_M
            ),
            "orientation_slider_latched": bool(
                self._as019_slider_orientation_latched
            ),
            "orientation_slider_latched_step": int(
                self._as019_slider_orientation_latched_step
            ),
            "orientation_slider_active_owners": tuple(
                self._as019_slider_orientation_active_owners
            ),
            "orientation_slider_contact_grace_steps": int(
                _SLIDER_CONTACT_GRACE_STEPS
            ),
            "orientation_slider_grace_owners": tuple(
                self._as019_slider_orientation_grace_owners
            ),
            "orientation_slider_owner_last_live_step": dict(
                self._as019_slider_owner_last_live_step
            ),
            "orientation_slider_wrist_locks_wxyz": {
                side: np.asarray(quaternion, dtype=np.float64).tolist()
                for side, quaternion in sorted(
                    self._as019_slider_hand_orientation_locks.items()
                )
            },
            "right_wrench_requires_live_opposed_tail_contact": bool(
                "right" not in self._as019_active_wrench_owners
                or self._as019_active_wrench_gate_snapshot.get(
                    "right_wrench_invariant", False
                )
            ),
            "right_wrench_gate_snapshot": dict(
                self._as019_active_wrench_gate_snapshot
            ),
            "left_wrench_requires_live_opposed_middle_contact": bool(
                "left" not in self._as019_active_wrench_owners
                or self._as019_active_wrench_gate_snapshot.get(
                    "left_wrench_invariant", False
                )
            ),
            "left_wrench_gate_snapshot": dict(
                self._as019_active_wrench_gate_snapshot
            ),
            "trajectory_state_write_count": int(self._trajectory_state_write_count),
            "screwdriver_mass_kg": float(self._as019_mass_kg),
            "screwdriver_friction": float(self._as019_friction),
            "reset_sampling_attempts": int(self._as019_reset_attempts),
            "automatic_acquire_gate_required": bool(
                self._as019_automatic_acquire_gate_required
            ),
            "automatic_acquire_gate_open": bool(
                self._as019_automatic_acquire_gate_open
            ),
            "automatic_arrival_gate": dict(
                self._as019_automatic_arrival_gate
            ),
            "as019_release_frame_trace": tuple(self._as019_release_frame_trace),
            "as019_release_frame_trace_count": len(
                self._as019_release_frame_trace
            ),
            "as019_release_frame_trace_dropped_frames": int(
                self._as019_release_trace_dropped_frames
            ),
            **{f"as019_{key}": value for key, value in metrics.items()},
            **{f"as019_{key}": value for key, value in release_telemetry.items()},
        }


class AS019HandsOnlyGuidanceEnv(
    AS019ScrewdriverHandoverEnv,
    HandsOnABCIsaacLabEnv,
):
    """AS019's exact USD/task contract on the shared two-floating-hand runtime.

    The class is selected only while recording guidance.  Its MRO keeps every
    AS019 contact/order/success override first, while ``super()`` enters the
    ABC scene lifecycle that spawns two detached Wuji hands and no arms.
    """


class AS019AutomaticPolicyWrapper(gym.Wrapper):
    """Unattended action-only policy used by the production recorder."""

    def __init__(self, env: gym.Env, raw_env: AS019ScrewdriverHandoverEnv):
        super().__init__(env)
        self.raw_env = raw_env
        self._zone_plan: dict[str, Any] | None = None
        self._right_clear_streak = 0
        self._placement_ready_step: int | None = None
        self._placement_progress = 0
        self._placement_progress_step: int | None = None
        self._zone_settle_step: int | None = None
        self._placement_contact_active = False
        self._placement_reacquire_progress = 0
        self._placement_reacquire_progress_step: int | None = None
        self._placement_reacquire_contact_streak = 0
        self._placement_contact_loss_streak = 0
        self._left_stage3_relatched = False
        self._stage3_regrasp_active = False
        self._stage3_regrasp_attempted = False
        self._stage3_regrasp_start_step: int | None = None
        self._stage3_regrasp_start_palm: np.ndarray | None = None
        self._stage3_regrasp_start_hand: np.ndarray | None = None
        self._stage3_regrasp_near_palm: np.ndarray | None = None
        self._stage3_regrasp_grasp_palm: np.ndarray | None = None
        self._stage3_regrasp_pinch_streak = 0
        self._placement_hold_palm: np.ndarray | None = None
        self._release_quiet_hold_palm: np.ndarray | None = None
        self._release_grasp_target: np.ndarray | None = None
        self._release_hold_palm: np.ndarray | None = None
        self._release_open_unload_palm: np.ndarray | None = None
        self._release_final_palm: np.ndarray | None = None
        self._release_contact_clear_latched = False
        self._release_contact_clear_step: int | None = None
        self._release_contact_clear_palm: np.ndarray | None = None
        self._release_ready_streak = 0
        self._release_ready_step: int | None = None
        self._release_open_progress = 0
        self._release_progress_step: int | None = None
        self._release_preclear_progress = 0
        self._release_preclear_progress_step: int | None = None
        self._release_preclear_start_palm: np.ndarray | None = None
        self._release_preclear_target_palm: np.ndarray | None = None
        self._release_retract_ready_streak = 0
        self._release_retract_step: int | None = None
        self._release_retract_progress = 0
        self._release_retract_progress_step: int | None = None
        self._receiver_contact_palm: np.ndarray | None = None
        self._receiver_preload_palm: np.ndarray | None = None
        self._receiver_contact_step: int | None = None
        self._receiver_thumb_contact_palm: np.ndarray | None = None
        self._receiver_thumb_contact_step: int | None = None
        self._receiver_thumb_contact_hand: np.ndarray | None = None
        self._stage2_receiver_clear_latched = False
        self._receiver_thumb_loss_counter = 0
        self._receiver_single_contact_reseat_step: int | None = None
        self._receiver_single_contact_start_hand: np.ndarray | None = None
        self._receiver_single_contact_hold_palm: np.ndarray | None = None
        self._receiver_retry_right_hold_palm: np.ndarray | None = None
        self._receiver_retry_thumb_palm: np.ndarray | None = None
        self._right_acquired_palm: np.ndarray | None = None
        self._right_acquired_step: int | None = None
        self._right_acquired_tool: np.ndarray | None = None
        self._right_acquired_anchor_step: int | None = None
        self._right_acquisition_progress = 0
        self._right_acquisition_progress_step: int | None = None
        self._right_approach_tool: np.ndarray | None = None
        self._right_pick_arrival_streak = 0
        self._right_pick_arrival_last_step: int | None = None
        self._right_pick_arrived_step: int | None = None
        self._right_acquire_arrival_streak = 0
        self._right_acquire_arrival_last_step: int | None = None
        self._right_acquire_arrived_step: int | None = None
        self._episode_right_home: np.ndarray | None = None
        self._episode_left_home: np.ndarray | None = None
        # Remember the raw episode that existed when the wrapper was created,
        # even though the physical palms remain lazily captured.  If callers
        # bypass wrapper.reset() and reset the raw env before the first action,
        # the serial change must still clear and re-arm every automatic gate.
        self._captured_reset_serial = int(
            getattr(self.raw_env, "_as019_reset_serial", 0)
        )
        self._right_entry_palm: np.ndarray | None = None
        self._right_entry_tool: np.ndarray | None = None
        self._right_contact_palm: np.ndarray | None = None
        self._right_dual_contact_latched = False
        self._left_stage0_prealign_tool: np.ndarray | None = None
        self._left_stage0_prealign_start_step: int | None = None
        self._left_stage0_prealign_start_palm: np.ndarray | None = None
        self._left_stage1_wait_palm: np.ndarray | None = None
        self._left_receiver_approach_start_palm: np.ndarray | None = None
        self._left_receiver_approach_start_step: int | None = None
        self._activate_automatic_acquire_gate()

    def _activate_automatic_acquire_gate(self) -> None:
        """Require fixed-pose arrival before automatic physical ownership."""

        self.raw_env._as019_automatic_acquire_gate_required = True
        self.raw_env._as019_automatic_acquire_gate_open = False
        self.raw_env._as019_automatic_arrival_gate = {
            "gate": "approach_not_started",
            "open": False,
        }

    def _physical_palm_pose(self, side: str) -> np.ndarray:
        """Read one finite physical palm pose without a nominal-home fallback."""

        if self.raw_env._launch_isaac:
            robot = self.raw_env._control_robot(side)
            body_index = self.raw_env._ee_body_indices[side]
            value = self.raw_env._robot_body_pose_matrix(robot, body_index)
        else:
            # Interface-only CPU tests intentionally have no articulation.
            value = self.raw_env._palm_matrix(side)
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"AS019 {side} physical palm pose is non-finite")
        pose = _matrix_pose_vector(matrix)
        if not np.all(np.isfinite(pose)):
            raise RuntimeError(f"AS019 {side} physical palm pose is non-finite")
        return pose

    def _capture_episode_home_palms(self) -> None:
        """Atomically bind the automatic plan to this reset's physical palms."""

        right = self._physical_palm_pose("right")
        left = self._physical_palm_pose("left")
        self._episode_right_home = right
        self._episode_left_home = left
        self._captured_reset_serial = int(
            getattr(self.raw_env, "_as019_reset_serial", 0)
        )

    def _ensure_episode_home_palms(self) -> None:
        reset_serial = int(getattr(self.raw_env, "_as019_reset_serial", 0))
        serial_changed = bool(
            self._captured_reset_serial >= 0
            and self._captured_reset_serial != reset_serial
        )
        if serial_changed:
            # Raw/manual reset paths intentionally bypass this wrapper.  Palm
            # recapture alone is insufficient: every cached tool transform,
            # contact waypoint, placement plan, and release latch belongs to
            # the previous episode and must be discarded atomically first.
            self._reset_plan_state()
        if (
            self._episode_right_home is None
            or self._episode_left_home is None
            or self._captured_reset_serial != reset_serial
        ):
            self._capture_episode_home_palms()

    def _reset_plan_state(self) -> None:
        """Clear every wrapper-owned value whose meaning is reset-relative."""

        self._zone_plan = None
        self._right_clear_streak = 0
        self._placement_ready_step = None
        self._placement_progress = 0
        self._placement_progress_step = None
        self._zone_settle_step = None
        self._placement_contact_active = False
        self._placement_reacquire_progress = 0
        self._placement_reacquire_progress_step = None
        self._placement_reacquire_contact_streak = 0
        self._placement_contact_loss_streak = 0
        self._left_stage3_relatched = False
        self._stage3_regrasp_active = False
        self._stage3_regrasp_attempted = False
        self._stage3_regrasp_start_step = None
        self._stage3_regrasp_start_palm = None
        self._stage3_regrasp_start_hand = None
        self._stage3_regrasp_near_palm = None
        self._stage3_regrasp_grasp_palm = None
        self._stage3_regrasp_pinch_streak = 0
        self._placement_hold_palm = None
        self._release_quiet_hold_palm = None
        self._release_grasp_target = None
        self._release_hold_palm = None
        self._release_open_unload_palm = None
        self._release_final_palm = None
        self._release_contact_clear_latched = False
        self._release_contact_clear_step = None
        self._release_contact_clear_palm = None
        self._release_ready_streak = 0
        self._release_ready_step = None
        self._release_open_progress = 0
        self._release_progress_step = None
        self._release_preclear_progress = 0
        self._release_preclear_progress_step = None
        self._release_preclear_start_palm = None
        self._release_preclear_target_palm = None
        self._release_retract_ready_streak = 0
        self._release_retract_step = None
        self._release_retract_progress = 0
        self._release_retract_progress_step = None
        self._receiver_contact_palm = None
        self._receiver_preload_palm = None
        self._receiver_contact_step = None
        self._receiver_thumb_contact_palm = None
        self._receiver_thumb_contact_step = None
        self._receiver_thumb_contact_hand = None
        self._stage2_receiver_clear_latched = False
        self._receiver_thumb_loss_counter = 0
        self._receiver_single_contact_reseat_step = None
        self._receiver_single_contact_start_hand = None
        self._receiver_single_contact_hold_palm = None
        self._receiver_retry_right_hold_palm = None
        self._receiver_retry_thumb_palm = None
        self._right_acquired_palm = None
        self._right_acquired_step = None
        self._right_acquired_tool = None
        self._right_acquired_anchor_step = None
        self._right_acquisition_progress = 0
        self._right_acquisition_progress_step = None
        self._right_approach_tool = None
        self._right_pick_arrival_streak = 0
        self._right_pick_arrival_last_step = None
        self._right_pick_arrived_step = None
        self._right_acquire_arrival_streak = 0
        self._right_acquire_arrival_last_step = None
        self._right_acquire_arrived_step = None
        self._episode_right_home = None
        self._episode_left_home = None
        self._captured_reset_serial = -1
        self._right_entry_palm = None
        self._right_entry_tool = None
        self._right_contact_palm = None
        self._right_dual_contact_latched = False
        self._left_stage0_prealign_tool = None
        self._left_stage0_prealign_start_step = None
        self._left_stage0_prealign_start_palm = None
        self._left_stage1_wait_palm = None
        self._left_receiver_approach_start_palm = None
        self._left_receiver_approach_start_step = None
        self.raw_env._as019_placement_assist_enabled = False
        self.raw_env._as019_placement_motion_started = False
        self.raw_env._as019_release_support_enabled = False
        self.raw_env._as019_receiver_retry_assist_tier = False
        self._activate_automatic_acquire_gate()

    def reset(self, **kwargs: Any):
        self._reset_plan_state()
        result = self.env.reset(**kwargs)
        # Raw reset clears the task-side gate so manual operation remains
        # unaffected.  Re-arm only after the automatic wrapper owns the new
        # episode.
        self._activate_automatic_acquire_gate()
        self._capture_episode_home_palms()
        return result

    @staticmethod
    def _smooth(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64).copy()
        # q and -q encode the same physical orientation.  Align their
        # hemispheres before normalized interpolation so a randomized
        # contact-anchor solve cannot command the nearly 360-degree long arc.
        if float(np.dot(start[3:7], end[3:7])) < 0.0:
            end[3:7] *= -1.0
        result = (1.0 - alpha) * start + alpha * end
        result[3:7] /= max(float(np.linalg.norm(result[3:7])), 1.0e-12)
        return result

    def _update_right_arrival_gate(
        self,
        *,
        gate: str,
        target: np.ndarray,
        start_step: int,
        timeout_steps: int,
    ) -> bool:
        """Latch one fixed right-palm target after consecutive live arrivals."""

        if gate not in {"pick", "acquire"}:
            raise ValueError(f"unsupported AS019 arrival gate: {gate}")
        step = int(self.raw_env.env_step)
        actual = self._physical_palm_pose("right")
        target = np.asarray(target, dtype=np.float64).reshape(7)
        target_matrix = _pose_vector_to_matrix(target)
        actual_matrix = _pose_vector_to_matrix(actual)
        position_error = float(
            np.linalg.norm(actual_matrix[:3, 3] - target_matrix[:3, 3])
        )
        rotation_error = float(
            np.linalg.norm(
                _rotation_error_vector(
                    target_matrix[:3, :3],
                    actual_matrix[:3, :3],
                )
            )
        )
        within = bool(
            position_error <= _RIGHT_ARRIVAL_POSITION_TOLERANCE_M
            and rotation_error <= _RIGHT_ARRIVAL_ROTATION_TOLERANCE_RAD
        )
        streak_name = f"_right_{gate}_arrival_streak"
        last_step_name = f"_right_{gate}_arrival_last_step"
        arrived_name = f"_right_{gate}_arrived_step"
        streak = int(getattr(self, streak_name))
        last_step = getattr(self, last_step_name)
        arrived_step = getattr(self, arrived_name)
        if arrived_step is None and last_step != step:
            if last_step is not None and step != int(last_step) + 1:
                streak = 0
            streak = streak + 1 if within else 0
            setattr(self, streak_name, streak)
            setattr(self, last_step_name, step)
            if streak >= _RIGHT_ARRIVAL_STABLE_STEPS:
                arrived_step = step
                setattr(self, arrived_name, step)
        deadline_step = int(start_step + timeout_steps - 1)
        timed_out = bool(arrived_step is None and step >= deadline_step)
        if timed_out and not self.raw_env._as019_failure_reason:
            self.raw_env._as019_failure_reason = (
                f"automatic_right_{gate}_arrival_timeout"
            )
        self.raw_env._as019_automatic_arrival_gate = {
            "gate": gate,
            "target_pose_wxyz": target.tolist(),
            "actual_pose_wxyz": actual.tolist(),
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
            "position_tolerance_m": _RIGHT_ARRIVAL_POSITION_TOLERANCE_M,
            "rotation_tolerance_rad": _RIGHT_ARRIVAL_ROTATION_TOLERANCE_RAD,
            "within_tolerance": within,
            "streak": int(streak),
            "required_streak": int(_RIGHT_ARRIVAL_STABLE_STEPS),
            "arrived_step": arrived_step,
            "sample_env_step": step,
            "start_step": int(start_step),
            "deadline_step": deadline_step,
            "timed_out": timed_out,
            "open": bool(self.raw_env._as019_automatic_acquire_gate_open),
        }
        return arrived_step is not None

    def _zone_motion_plan(
        self,
        tool: np.ndarray,
        zone: np.ndarray,
    ) -> dict[str, Any] | None:
        """Latch a contact-owned placement plan at the right-release event."""

        event_step = int(self.raw_env._as019_event_steps[2])
        payload = self.raw_env._as019_assist_anchor.get("left")
        metrics = self.raw_env._as019_last_metrics
        owner_available = bool(
            payload is not None
            and "hand_to_tool" in payload
            and metrics.get("contact_query_valid", False)
            and metrics.get("left_middle_physical", False)
            and _grasp_command_max(self.raw_env._left_hand)
            >= _HAND_CLOSE_THRESHOLD
        )
        if (
            self._zone_plan is not None
            and self._zone_plan["event_step"] == event_step
            and (self._zone_plan["anchor"] is not None or not owner_available)
        ):
            return self._zone_plan
        if event_step < 0:
            return None
        live_tool = np.asarray(tool, dtype=np.float64).reshape(7).copy()
        hover_tool, placed_tool = _zone_tool_targets(live_tool, zone)
        stable_placed_tool = placed_tool.copy()
        stable_placed_tool[3:7] = _stable_tool_quaternion(
            live_tool,
            _ZONE_PLACEMENT_ROLL_RAD,
        )
        stable_placed_tool[2] = (
            float(zone[2])
            - _MARKER_CENTER_ABOVE_TABLE_M
            + _tool_vertical_support_depth(stable_placed_tool[3:7])
            + _ZONE_SUPPORT_CLEARANCE_M
        )
        # Center laterally at hover height, then descend on the continuous,
        # contact-owned palm branch.  Longitudinal roll is preserved through
        # these free-space moves and corrected only after table support.
        lowered_tool = placed_tool.copy()
        x_aligned_tool = live_tool.copy()
        x_aligned_tool[0] = hover_tool[0]
        x_aligned_tool[2] = hover_tool[2]
        # Keep the interpolation structure available for diagnostic placement
        # roll offsets; the formal zero offset preserves the live quaternion.
        x_aligned_tool[3:7] = self._smooth(
            live_tool, hover_tool, 1.0 / 3.0
        )[3:7]
        hover_tool[3:7] = self._smooth(
            live_tool, hover_tool, 2.0 / 3.0
        )[3:7]
        right_start = _matrix_pose_vector(self.raw_env._palm_matrix("right"))
        right_clear, right_normal = _tool_normal_retreat_pose(
            live_tool, right_start, zone
        )
        anchor = None
        start_palm = None
        x_aligned_palm = None
        hover_palm = None
        lowered_palm = None
        placed_palm = None
        stable_placed_palm = None
        release_palm = None
        if owner_available:
            anchor = np.asarray(payload["hand_to_tool"], dtype=np.float64).reshape(4, 4)
            start_palm = _owner_palm_pose(live_tool, anchor)
            x_aligned_palm = _owner_palm_pose(x_aligned_tool, anchor)
            hover_palm = _owner_palm_pose(hover_tool, anchor)
            lowered_palm = _owner_palm_pose(lowered_tool, anchor)
            placed_palm = _owner_palm_pose(placed_tool, anchor)
            stable_placed_palm = _owner_palm_pose(stable_placed_tool, anchor)
            release_palm = stable_placed_palm.copy()
            # Slide the open receiver toward the blunt local -X end.  This
            # level path moves away from the authored +X sharp tip and avoids
            # rolling the table-supported cylinder with a radial withdrawal.
            release_palm[:3] += (
                _LEFT_RELEASE_CLEAR_DISTANCE_M
                * _tool_blunt_release_direction(stable_placed_tool)
            )
        self._zone_plan = {
            "event_step": event_step,
            "anchor": anchor,
            "start_tool": live_tool,
            "x_aligned_tool": x_aligned_tool,
            "hover_tool": hover_tool,
            "lowered_tool": lowered_tool,
            "placed_tool": placed_tool,
            "stable_placed_tool": stable_placed_tool,
            "start_palm": start_palm,
            "x_aligned_palm": x_aligned_palm,
            "hover_palm": hover_palm,
            "lowered_palm": lowered_palm,
            "placed_palm": placed_palm,
            "stable_placed_palm": stable_placed_palm,
            "release_palm": release_palm,
            "right_start": right_start,
            "right_clear": right_clear,
            "right_normal": right_normal,
        }
        self._right_clear_streak = 0
        self._placement_ready_step = None
        self._placement_progress = 0
        self._placement_progress_step = None
        self._zone_settle_step = None
        self._placement_contact_active = False
        self._placement_reacquire_progress = 0
        self._placement_reacquire_progress_step = None
        self._placement_reacquire_contact_streak = 0
        self._placement_contact_loss_streak = 0
        self.raw_env._as019_placement_assist_enabled = False
        self.raw_env._as019_placement_motion_started = False
        self._release_grasp_target = None
        self._release_ready_streak = 0
        self._release_ready_step = None
        self._release_open_progress = 0
        self._release_progress_step = None
        self._release_preclear_progress = 0
        self._release_preclear_progress_step = None
        self._release_preclear_start_palm = None
        self._release_preclear_target_palm = None
        self._release_retract_ready_streak = 0
        self._release_retract_step = None
        self._release_retract_progress = 0
        self._release_retract_progress_step = None
        self._release_final_palm = None
        self._release_contact_clear_latched = False
        self._release_contact_clear_step = None
        self._release_contact_clear_palm = None
        return self._zone_plan

    @staticmethod
    def _placement_command(
        plan: dict[str, Any], tool: np.ndarray
    ) -> np.ndarray:
        """Compensate the measured low-table arm endpoint error."""

        command = np.asarray(plan["placed_palm"], dtype=np.float64).copy()
        correction = (
            np.asarray(plan["placed_tool"], dtype=np.float64)[:3]
            - np.asarray(tool, dtype=np.float64)[:3]
        )
        # The low-table DLS branch retains a repeatable vertical endpoint
        # residual.  Compensate that robot error in the command so the tool is
        # physically table-supported before release; lateral centering stays
        # one-to-one and the total correction remains capped below.
        correction[2] *= _ZONE_FINAL_VERTICAL_COMMAND_CORRECTION_GAIN
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > _ZONE_FINAL_COMMAND_CORRECTION_MAX_M:
            correction *= _ZONE_FINAL_COMMAND_CORRECTION_MAX_M / correction_norm
        command[:3] += correction
        return command

    @staticmethod
    def _placement_tracking_command(
        command: np.ndarray,
        measured_palm: np.ndarray,
        contact_force_n: float = 0.0,
    ) -> np.ndarray:
        """Compensate a bounded DLS residual without overloading the grasp."""

        corrected = np.asarray(command, dtype=np.float64).copy()
        measured = np.asarray(measured_palm, dtype=np.float64).reshape(7)
        residual = corrected[:3] - measured[:3]
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > _AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_MAX_M:
            residual *= (
                _AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_MAX_M
                / residual_norm
            )
        force_fraction = float(
            np.clip(
                (
                    float(contact_force_n)
                    - _AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_TAPER_START_N
                )
                / (
                    _AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_TAPER_END_N
                    - _AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_TAPER_START_N
                ),
                0.0,
                1.0,
            )
        )
        tracking_gain = (
            _AUTOMATIC_RECEIVER_PLACEMENT_TRACKING_GAIN
            * (1.0 - force_fraction)
        )
        corrected[:3] += tracking_gain * residual
        return corrected

    @staticmethod
    def _placement_reacquire_pose(
        tool: np.ndarray, anchor_palm: np.ndarray
    ) -> np.ndarray:
        """Apply the verified 3 mm receiver preload toward the handle axis."""

        tool_matrix = _pose_vector_to_matrix(tool)
        palm_matrix = _pose_vector_to_matrix(anchor_palm)
        palm_local = (np.linalg.inv(tool_matrix) @ palm_matrix)[:3, 3]
        radial_local = np.asarray(
            [
                0.0,
                palm_local[1] - _HANDLE_LOCAL_CENTER[1],
                palm_local[2] - _HANDLE_LOCAL_CENTER[2],
            ],
            dtype=np.float64,
        )
        radial_norm = float(np.linalg.norm(radial_local))
        result = np.asarray(anchor_palm, dtype=np.float64).copy()
        if radial_norm > 1.0e-9:
            outward_world = tool_matrix[:3, :3] @ (
                radial_local / radial_norm
            )
            result[:3] -= (
                _LEFT_RECEIVER_CONTACT_PRELOAD_DISTANCE_M * outward_world
            )
        return result

    def _stage3_regrasp_command(
        self,
        *,
        step: int,
        tool: np.ndarray,
        metrics: dict[str, Any],
        left: np.ndarray,
        left_hand: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Recover a receiver without an opposed pinch on its live wrist."""

        stage = int(self.raw_env._as019_stage)
        two_sided = bool(
            self.raw_env._receiver_has_physical_pinch(metrics)
        )
        physical_left = bool(
            metrics.get("contact_query_valid", False)
            and metrics.get("left_middle_physical", False)
            and _grasp_command_max(self.raw_env._left_hand)
            >= _HAND_CLOSE_THRESHOLD
        )
        stage1_retry_ready = bool(
            stage == 1
            and int(self.raw_env._as019_event_steps[0]) >= 0
            and "right" in self.raw_env._as019_assist_anchor
            and "left" not in self.raw_env._as019_assist_anchor
            and self._receiver_single_contact_reseat_step is not None
            and step - self._receiver_single_contact_reseat_step
            >= (
                _LEFT_RESEAT_SINGLE_CONTACT_OPEN_STEPS
                + _LEFT_RESEAT_RETRY_RETURN_STEPS
                + _LEFT_RESEAT_DELAY_STEPS
            )
        )
        stage2_retry_ready = False
        stage3_retry_ready = bool(
            stage == 3
            and int(self.raw_env._as019_event_steps[2]) >= 0
            and self.raw_env._as019_mass_kg
            <= _AUTOMATIC_RECEIVER_LOW_INERTIA_MASS_KG
            and not self._stage3_regrasp_attempted
            and self._placement_ready_step is None
            and bool(metrics.get("right_clear", False))
            and not physical_left
        )
        should_start = bool(
            not two_sided
            and (stage1_retry_ready or stage2_retry_ready or stage3_retry_ready)
        )
        if should_start and not self._stage3_regrasp_active:
            if stage == 1:
                # The one-sided unload and return take dozens of frames.  Use
                # the giver's measured pose at the instant the full regrasp
                # begins, rather than the old pose cached at first contact
                # loss.  When contact is still authoritative, rebind its
                # compliant anchor on that same zero-jump frame.
                current_right_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("right")
                )
                self._receiver_retry_right_hold_palm = current_right_palm
                right_anchor = self.raw_env._as019_assist_anchor.get("right")
                if (
                    bool(metrics.get("right_tail_physical", False))
                    and right_anchor is not None
                    and "hand_to_tool" in right_anchor
                ):
                    right_anchor["hand_to_tool"] = (
                        np.linalg.inv(self.raw_env._palm_matrix("right"))
                        @ self.raw_env._tool_matrix()
                    )
                    right_anchor["attached_step"] = step
            low_y = bool(
                self._right_approach_tool is not None
                and float(self._right_approach_tool[1])
                <= _LEFT_RECEIVER_LOW_Y_THRESHOLD_M
            )
            grasp_local = np.asarray(
                _LEFT_RECEIVER_LOW_Y_GRASP_LOCAL
                if low_y
                else _LEFT_RECEIVER_GRASP_LOCAL,
                dtype=np.float64,
            ).copy()
            grasp_local += (
                _STAGE3_REGRASP_LOCAL_OFFSET
                if stage == 3
                else _STAGE1_REGRASP_LOCAL_OFFSET
            )
            near_local = grasp_local.copy()
            near_local[1] += _STAGE3_REGRASP_NEAR_LOCAL_Y_M
            local_quat = np.asarray(
                _LEFT_RECEIVER_LOW_Y_LOCAL_QUAT
                if low_y
                else _LEFT_RECEIVER_LOCAL_QUAT,
                dtype=np.float64,
            )
            start_palm = _matrix_pose_vector(
                self.raw_env._palm_matrix("left")
            )
            near_palm = _tool_frame_pose(tool, near_local, local_quat)
            grasp_palm = _tool_frame_pose(tool, grasp_local, local_quat)
            # The measured wrist orientation is the stable reachable DLS
            # branch.  Only translate it; rotating toward the authored branch
            # caused a discontinuous IK flip in live multi-pose sweeps.
            near_palm[3:7] = start_palm[3:7]
            grasp_palm[3:7] = start_palm[3:7]
            self._stage3_regrasp_active = True
            if stage == 3:
                self._stage3_regrasp_attempted = True
            self._stage3_regrasp_start_step = step
            self._stage3_regrasp_start_palm = start_palm
            self._stage3_regrasp_start_hand = np.asarray(
                self.raw_env._left_hand, dtype=np.float64
            ).copy()
            self._stage3_regrasp_near_palm = near_palm
            self._stage3_regrasp_grasp_palm = grasp_palm
            self._stage3_regrasp_pinch_streak = 0
        if not self._stage3_regrasp_active:
            return left, left_hand, False
        self.raw_env._as019_receiver_retry_assist_tier = True
        anchor = self.raw_env._as019_assist_anchor.get("left")
        if stage == 2:
            # Keep the giver closed and reapply the unchanged twelve-frame
            # receiver hold after this robot-only recovery.  A one-finger
            # collision sample must not consume that dwell or start release.
            self.raw_env._as019_receiver_hold_counter = 0
            self.raw_env._as019_receiver_hold_complete_step = -1
            self.raw_env._as019_right_release_counter = 0
            self.raw_env._as019_stage2_left_contact_active = False
        assert self._stage3_regrasp_start_step is not None
        assert self._stage3_regrasp_start_palm is not None
        assert self._stage3_regrasp_start_hand is not None
        assert self._stage3_regrasp_near_palm is not None
        assert self._stage3_regrasp_grasp_palm is not None
        elapsed = max(0, step - self._stage3_regrasp_start_step)
        start_palm = self._stage3_regrasp_start_palm
        start_hand = self._stage3_regrasp_start_hand
        near_palm = self._stage3_regrasp_near_palm
        grasp_palm = self._stage3_regrasp_grasp_palm
        open_hand = start_hand.copy()
        if stage == 1:
            # A stage-1 retry follows a lost thumb contact and an opposing
            # one-finger load.  Unload both sides of that failed pinch before
            # returning on the original collision-checked receiver height;
            # retaining the closed, separated thumb forced the reachable DLS
            # branch outward and could never form a new opposed grasp.
            open_hand[0:4] = 0.0
        open_hand[16:20] = 0.0
        grasp_hand = np.asarray(
            _AUTOMATIC_RECEIVER_GRASP_TARGET, dtype=np.float64
        )
        preshape_hand = (
            (1.0 - _STAGE3_REGRASP_PRESHAPE_FRACTION) * open_hand
            + _STAGE3_REGRASP_PRESHAPE_FRACTION * grasp_hand
        )
        open_end = _STAGE3_REGRASP_OPEN_STEPS
        move_end = open_end + _STAGE3_REGRASP_MOVE_STEPS
        preshape_end = move_end + _STAGE3_REGRASP_PRESHAPE_STEPS
        approach_end = preshape_end + _STAGE3_REGRASP_APPROACH_STEPS
        if elapsed < open_end:
            alpha = elapsed / float(max(open_end - 1, 1))
            command_palm = start_palm
            command_hand = (1.0 - alpha) * start_hand + alpha * open_hand
        elif elapsed < move_end:
            alpha = (elapsed - open_end) / float(
                max(_STAGE3_REGRASP_MOVE_STEPS - 1, 1)
            )
            command_palm = self._smooth(start_palm, near_palm, alpha)
            command_hand = open_hand
        elif elapsed < preshape_end:
            alpha = (elapsed - move_end) / float(
                max(_STAGE3_REGRASP_PRESHAPE_STEPS - 1, 1)
            )
            command_palm = near_palm
            command_hand = (
                (1.0 - alpha) * open_hand + alpha * preshape_hand
            )
        elif elapsed < approach_end:
            alpha = (elapsed - preshape_end) / float(
                max(_STAGE3_REGRASP_APPROACH_STEPS - 1, 1)
            )
            command_palm = self._smooth(near_palm, grasp_palm, alpha)
            command_hand = (
                (1.0 - alpha) * preshape_hand + alpha * grasp_hand
            )
        else:
            alpha = float(
                np.clip(
                    (elapsed - approach_end)
                    / float(max(_STAGE3_REGRASP_CLOSE_STEPS - 1, 1)),
                    0.0,
                    1.0,
                )
            )
            command_palm = grasp_palm.copy()
            command_hand = grasp_hand.copy()
            if alpha >= 1.0 and not self.raw_env._receiver_has_physical_pinch(
                metrics
            ):
                current_tool = self.raw_env._tool_matrix()
                current_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("left")
                )
                current_local = (
                    np.linalg.inv(current_tool)
                    @ _pose_vector_to_matrix(current_palm)
                )[:3, 3]
                correction = current_tool[:3, :3] @ (
                    _LEFT_RESEAT_TARGET_LOCAL - current_local
                )
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > _LEFT_RESEAT_MAX_CORRECTION_M:
                    correction *= _LEFT_RESEAT_MAX_CORRECTION_M / correction_norm
                command_palm = current_palm
                command_palm[:3] += correction
                command_palm[3:7] = grasp_palm[3:7]
        if self.raw_env._as019_receiver_pinch_target is not None:
            self.raw_env._as019_receiver_pinch_target = (
                self.raw_env._as019_receiver_pinch_target.new_tensor(
                    command_hand
                ).reshape(1, -1)
            )
        self.raw_env._left_hand[:] = command_hand
        self.raw_env._as019_placement_assist_enabled = False
        two_sided = bool(
            self.raw_env._receiver_has_physical_pinch(metrics)
        )
        self._stage3_regrasp_pinch_streak = (
            self._stage3_regrasp_pinch_streak + 1 if two_sided else 0
        )
        if (
            self._stage3_regrasp_pinch_streak
            >= _STAGE3_REGRASP_CONTACT_CONFIRM_STEPS
            and anchor is not None
            and "hand_to_tool" in anchor
        ):
            anchor["hand_to_tool"] = (
                np.linalg.inv(self.raw_env._palm_matrix("left"))
                @ self.raw_env._tool_matrix()
            )
            anchor["attached_step"] = step
            self._zone_plan = None
            self._placement_ready_step = None
            self._placement_progress = 0
            self._placement_progress_step = None
            self._placement_contact_active = True
            self._placement_reacquire_progress = 0
            self._placement_reacquire_progress_step = None
            self._placement_reacquire_contact_streak = 0
            self._placement_contact_loss_streak = 0
            self._right_clear_streak = 0
            self.raw_env._as019_placement_motion_started = False
            # The old pinch record belongs to the grasp that was physically
            # lost.  If it survives this rebind, the next stage-3 action opens
            # finger 5 back toward that stale target immediately after the new
            # pinch has passed its eight-frame confirmation.  Freeze the live
            # measured joints at that authoritative contact frame, rather than
            # the still-closing target.  Retain only a small fraction of the
            # intended close as preload: this is near-continuous at the first
            # normal placement frame but does not leave a light tool loose.
            robot = self.raw_env._scene["left_robot"]
            joint_ids = list(self.raw_env._hand_joint_ids["left"])
            measured_hand_tensor = robot.data.joint_pos[:, joint_ids].clone()
            measured_hand = (
                measured_hand_tensor[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            hold_hand = measured_hand + _STAGE3_REGRASP_HOLD_TARGET_BLEND * (
                np.asarray(command_hand, dtype=np.float64) - measured_hand
            )
            hold_hand_tensor = measured_hand_tensor.new_tensor(
                hold_hand
            ).reshape(1, -1)
            self.raw_env._as019_receiver_pinch_target = hold_hand_tensor
            self.raw_env._left_hand[:] = hold_hand
            command_hand = hold_hand.copy()
            pinch_record = dict(
                self.raw_env._as019_receiver_pinch_freeze
            )
            pinch_record.update(
                {
                    "env_step": int(step),
                    "substep": -1,
                    "normalization_applied": False,
                    "stage3_regrasp_latched": True,
                    "measured_joint_positions": measured_hand.tolist(),
                    "joint_positions": hold_hand.tolist(),
                }
            )
            self.raw_env._as019_receiver_pinch_freeze = pinch_record
            command_palm = _matrix_pose_vector(
                self.raw_env._palm_matrix("left")
            )
            self._stage3_regrasp_active = False
            self._stage3_regrasp_start_step = None
            self._stage3_regrasp_start_palm = None
            self._stage3_regrasp_start_hand = None
            self._stage3_regrasp_near_palm = None
            self._stage3_regrasp_grasp_palm = None
            self._stage3_regrasp_pinch_streak = 0
        return command_palm, command_hand, True

    def automatic_action(self) -> np.ndarray:
        self._ensure_episode_home_palms()
        step = int(self.raw_env.env_step)
        stage = int(self.raw_env._as019_stage)
        tool = self.raw_env._peg_ori_pose
        zone = self.raw_env._socket_ori_pose
        if self._right_approach_tool is None:
            self._right_approach_tool = np.asarray(
                tool, dtype=np.float64
            ).copy()
        quat = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        assert self._episode_right_home is not None
        assert self._episode_left_home is not None
        right_home = self._episode_right_home.copy()
        left_home = self._episode_left_home.copy()
        # Keep the palm orientation simple for an operator to reproduce.  The
        # closed-hand acquisition target is expressed in the live tool frame:
        # the canonical successful trajectory reached physical tail ownership
        # at this relative position, whereas a fixed world handoff target
        # shifted by the entire reset randomization envelope.
        # Follow only the authored reset-yaw perturbation.  The live tool
        # quaternion also contains small roll/pitch settling errors; feeding
        # those errors back to the wrist amplified them during acquisition.
        reset_yaw_quat = np.asarray(
            getattr(
                self.raw_env,
                "_as019_reset_yaw_quat",
                [1.0, 0.0, 0.0, 0.0],
            ),
            dtype=np.float64,
        )
        right_wrist_roll_quat = np.asarray(
            [
                math.cos(0.5 * _RIGHT_WRIST_ROLL_OFFSET_RAD),
                math.sin(0.5 * _RIGHT_WRIST_ROLL_OFFSET_RAD),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        canonical_tool_quat = _matrix_to_quat_wxyz(
            self.raw_env._as019_outward_reference_rotation
        )
        tool_aligned_quat = _quat_mul_wxyz(
            reset_yaw_quat,
            _quat_mul_wxyz(
                canonical_tool_quat,
                _quat_mul_wxyz(quat, right_wrist_roll_quat),
            ),
        )
        tool_aligned_quat /= max(
            float(np.linalg.norm(tool_aligned_quat)), 1.0e-12
        )
        right_pick_position = (
            _pose_vector_to_matrix(self._right_approach_tool)
            @ np.asarray([*_RIGHT_PICK_TARGET_LOCAL, 1.0], dtype=np.float64)
        )[:3]
        right_pick = np.concatenate(
            [right_pick_position, tool_aligned_quat]
        )
        right_acquire_position = (
            _pose_vector_to_matrix(self._right_approach_tool)
            @ np.asarray([*_RIGHT_ACQUIRE_TARGET_LOCAL, 1.0], dtype=np.float64)
        )[:3]
        right_acquire = np.concatenate(
            [right_acquire_position, tool_aligned_quat]
        )
        # Keep every wrist rotation and lateral move well above the table,
        # then descend vertically to the audited no-contact pick pose.  The
        # reset palm itself is still measured by the live canary; this first
        # segment moves monotonically upward from its established task home.
        right_high_home = right_home.copy()
        right_high_home[2] = max(
            float(right_home[2]), _RIGHT_APPROACH_HIGH_WORLD_Z_M
        )
        right_high_pick = right_pick.copy()
        right_high_pick[2] = right_high_home[2]
        right_handoff = np.asarray([-0.205, -0.035, 1.155, *quat], dtype=np.float64)
        low_y_receiver_branch = bool(
            float(self._right_approach_tool[1])
            <= _LEFT_RECEIVER_LOW_Y_THRESHOLD_M
        )
        self.raw_env._as019_low_y_receiver_branch = low_y_receiver_branch
        reset_yaw_rad = 2.0 * math.atan2(
            float(reset_yaw_quat[3]), float(reset_yaw_quat[0])
        )
        self.raw_env._as019_high_handover_assist_tier = bool(
            float(self._right_approach_tool[1])
            <= _HIGH_HANDOVER_ASSIST_TOOL_Y_THRESHOLD_M
            or abs(reset_yaw_rad)
            > _RESET_CONTROLLER_REACHABLE_ABS_YAW_MAX_RAD
            or self._receiver_single_contact_reseat_step is not None
        )
        self.raw_env._as019_receiver_retry_assist_tier = bool(
            self._receiver_single_contact_reseat_step is not None
        )
        receiver_pregrasp_local = (
            _LEFT_RECEIVER_LOW_Y_PREGRASP_LOCAL
            if low_y_receiver_branch
            else _LEFT_RECEIVER_PREGRASP_LOCAL
        )
        receiver_grasp_local = (
            _LEFT_RECEIVER_LOW_Y_GRASP_LOCAL
            if low_y_receiver_branch
            else _LEFT_RECEIVER_GRASP_LOCAL
        )
        receiver_local_quat = (
            _LEFT_RECEIVER_LOW_Y_LOCAL_QUAT
            if low_y_receiver_branch
            else _LEFT_RECEIVER_LOCAL_QUAT
        )
        left_handoff = _tool_frame_pose(
            tool,
            receiver_grasp_local,
            receiver_local_quat,
        )
        left_pregrasp = _tool_frame_pose(
            tool,
            receiver_pregrasp_local,
            _LEFT_RECEIVER_PREGRASP_LOCAL_QUAT,
        )
        right = right_home
        left = left_home
        right_hand = np.zeros(20, dtype=np.float64)
        left_hand = np.zeros(20, dtype=np.float64)

        right_owner = "right" in self.raw_env._as019_assist_anchor
        right_tail_physical = bool(
            self.raw_env._as019_last_metrics.get(
                "right_tail_physical", False
            )
        )
        if (
            stage == 0
            and self._right_acquired_palm is not None
            and not right_tail_physical
            and not right_owner
        ):
            # Once physical contact has established semantic ownership, keep
            # the measured robot plan across sub-threshold force flicker.  The
            # task wrench remains independently gated off by live contact in
            # ``_as019_pre_substep``; retaining this robot-only target cannot
            # drag a separated object and lets the closed fingers re-contact.
            self._right_acquired_palm = None
            self._right_acquired_step = None
            self._right_acquired_tool = None
            self._right_acquired_anchor_step = None
            self._right_acquisition_progress = 0
            self._right_acquisition_progress_step = None
        right_anchor = self.raw_env._as019_assist_anchor.get("right")
        right_anchor_step = (
            int(right_anchor.get("attached_step", -1))
            if right_anchor is not None
            else -1
        )
        if right_owner and right_tail_physical and (
            self._right_acquired_palm is None
            or self._right_acquired_anchor_step != right_anchor_step
        ):
            first_acquisition = self._right_acquired_palm is None
            self._right_acquired_palm = _matrix_pose_vector(
                self.raw_env._palm_matrix("right")
            )
            self._right_acquired_step = step
            self._right_acquired_tool = np.asarray(tool, dtype=np.float64).copy()
            self._right_acquired_anchor_step = right_anchor_step
            if first_acquisition:
                self._right_acquisition_progress = 0
                self._right_acquisition_progress_step = None
        if (
            stage == 0
            and self._right_acquired_palm is not None
            and right_tail_physical
            and self._right_acquisition_progress_step != step
        ):
            self._right_acquisition_progress += 1
            self._right_acquisition_progress_step = step
        right_tail_step = int(self.raw_env._as019_event_steps[0])
        if (
            stage >= 1
            and right_tail_step >= 0
            and self._right_entry_palm is None
        ):
            self._right_entry_palm = _matrix_pose_vector(
                self.raw_env._palm_matrix("right")
            )
            self._right_entry_tool = np.asarray(tool, dtype=np.float64).copy()
        right_transfer = (
            self._right_entry_palm.copy()
            if self._right_entry_palm is not None
            else right_handoff.copy()
        )
        right_entry_aligned = (
            self._right_entry_palm.copy()
            if self._right_entry_palm is not None
            else right_transfer.copy()
        )
        if self._right_entry_palm is not None:
            # Plan in tool space from the physical contact-owned anchor.  A
            # canonical palm quaternion does not imply a canonical tool axis
            # when randomized grasps latch different hand-to-tool transforms;
            # that old shortcut is what rotated otherwise valid grasps across
            # the 30-degree fail-closed boundary.
            if self._right_entry_tool is not None:
                aligned_entry_tool = self._right_entry_tool.copy()
                aligned_entry_tool[3:7] = _matrix_to_quat_wxyz(
                    _minimal_axis_aligned_rotation(
                        self.raw_env._as019_outward_reference_rotation[:, 0],
                        _quat_wxyz_to_matrix(self._right_entry_tool[3:7]),
                    )
                )
                transfer_tool = aligned_entry_tool.copy()
                transfer_tool[:3] = (
                    self._right_entry_tool[:3]
                    + _RIGHT_HANDOVER_TRANSLATION_WORLD
                )
                payload = self.raw_env._as019_assist_anchor.get("right")
                if payload is not None and "hand_to_tool" in payload:
                    anchor = np.asarray(
                        payload["hand_to_tool"], dtype=np.float64
                    )
                    right_entry_aligned = _owner_palm_pose(
                        aligned_entry_tool,
                        anchor,
                    )
                    right_transfer = _owner_palm_pose(
                        transfer_tool,
                        anchor,
                    )
                else:
                    right_transfer[:3] += (
                        transfer_tool[:3] - self._right_entry_tool[:3]
                    )
        if (
            self._right_entry_palm is not None
            and self._right_contact_palm is None
        ):
            if "left" in self.raw_env._as019_assist_anchor:
                # Stop the giver at its measured pose as soon as the receiver
                # physically owns the handle.  Continuing the remaining
                # transfer while the closed receiver follows through DLS winds
                # the cylinder between both hands before the unchanged dual-
                # contact debounce can complete.
                self._right_contact_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("right")
                )
            elif step > (
                right_tail_step
                + _RIGHT_HANDOVER_PRETRANSFER_HOLD_STEPS
                + _RIGHT_HANDOVER_STEPS
                + _RIGHT_HANDOVER_SETTLE_STEPS
            ):
                # Keep the verified contact-owned plan endpoint.  Latching the
                # measured DLS palm here reintroduced its residual Cartesian
                # error after a transfer had already completed.
                self._right_contact_palm = right_transfer.copy()
        right_hold = (
            self._right_contact_palm.copy()
            if self._right_contact_palm is not None
            else self._right_entry_palm.copy()
            if self._right_entry_palm is not None
            else self._right_acquired_palm.copy()
            if self._right_acquired_palm is not None
            else right_handoff
        )
        right_alignment = (
            self._right_acquired_palm.copy()
            if self._right_acquired_palm is not None
            else right_handoff.copy()
        )
        right_preload = right_alignment.copy()
        right_lift = right_alignment.copy()
        if (
            self._right_acquired_palm is not None
            and self._right_acquired_tool is not None
            and right_owner
        ):
            preload_delta = np.zeros(3, dtype=np.float64)
            approach = right_acquire[:3] - self._right_acquired_palm[:3]
            approach_norm = float(np.linalg.norm(approach))
            if approach_norm > 1.0e-9:
                preload_delta = (
                    _RIGHT_CONTACT_PRELOAD_DISTANCE_M
                    * approach
                    / approach_norm
                )
            right_preload = self._right_acquired_palm.copy()
            right_preload[:3] += preload_delta
            lifted_tool = self._right_acquired_tool.copy()
            lifted_tool[:3] += preload_delta
            lifted_tool[2] += _RIGHT_POST_GRASP_LIFT_DISTANCE_M
            payload = self.raw_env._as019_assist_anchor.get("right")
            if payload is not None and "hand_to_tool" in payload:
                right_lift = _owner_palm_pose(
                    lifted_tool, payload["hand_to_tool"]
                )
                right_alignment = right_lift.copy()
            else:
                right_lift = self._right_acquired_palm.copy()
                right_lift[:3] += preload_delta
                right_lift[2] += _RIGHT_POST_GRASP_LIFT_DISTANCE_M
                right_alignment = right_lift.copy()
        if step < _RIGHT_APPROACH_LIFT_END_STEP:
            phase = 0
            right = self._smooth(
                right_home,
                right_high_home,
                step / float(max(_RIGHT_APPROACH_LIFT_STEPS - 1, 1)),
            )
        elif step < _RIGHT_APPROACH_ALIGN_END_STEP:
            phase = 0
            right = self._smooth(
                right_high_home,
                right_high_pick,
                (step - _RIGHT_APPROACH_LIFT_END_STEP)
                / float(max(_RIGHT_APPROACH_ALIGN_STEPS - 1, 1)),
            )
        elif step < _RIGHT_GRASP_CLOSE_START_STEP:
            phase = 0
            right = self._smooth(
                right_high_pick,
                right_pick,
                (step - _RIGHT_APPROACH_ALIGN_END_STEP)
                / float(max(_RIGHT_APPROACH_DESCENT_STEPS - 1, 1)),
            )
        elif stage == 0 and self._right_pick_arrived_step is None:
            phase = 0
            right = right_pick
            right_hand.fill(0.0)
            self._update_right_arrival_gate(
                gate="pick",
                target=right_pick,
                start_step=_RIGHT_GRASP_CLOSE_START_STEP,
                timeout_steps=_RIGHT_PICK_ARRIVAL_TIMEOUT_STEPS,
            )
        elif (
            stage == 0
            and self._right_pick_arrived_step is not None
            and step
            < self._right_pick_arrived_step + _RIGHT_GRASP_CLOSE_STEPS
        ):
            phase = 1
            right = right_pick
            close_elapsed = max(0, step - self._right_pick_arrived_step)
            right_hand[:] = _AUTOMATIC_GIVER_GRASP_TARGET * (
                close_elapsed
                / float(_RIGHT_GRASP_CLOSE_STEPS - 1)
            )
        elif stage <= 1:
            phase = 2
            if self._right_contact_palm is not None:
                right = right_hold
            elif self._right_entry_palm is not None:
                handover_elapsed = max(0, step - right_tail_step)
                if handover_elapsed <= _RIGHT_HANDOVER_PRETRANSFER_HOLD_STEPS:
                    # The 15-degree event gate may accept a grasp
                    # before the giver finishes centering the tool axis.  Keep
                    # the contact-owned tool position fixed while completing
                    # that safe, operator-reproducible wrist alignment.
                    right = self._smooth(
                        self._right_entry_palm,
                        right_entry_aligned,
                        handover_elapsed
                        / _RIGHT_HANDOVER_PRETRANSFER_HOLD_STEPS,
                    )
                else:
                    transfer_progress = float(
                        np.clip(
                            (
                                handover_elapsed
                                - _RIGHT_HANDOVER_PRETRANSFER_HOLD_STEPS
                            )
                            / _RIGHT_HANDOVER_STEPS,
                            0.0,
                            1.0,
                        )
                    )
                    # Ease both endpoints so a newly acquired cylindrical
                    # handle is not hit by a discontinuous Cartesian velocity.
                    transfer_progress = (
                        transfer_progress
                        * transfer_progress
                        * (3.0 - 2.0 * transfer_progress)
                    )
                    right = self._smooth(
                        right_entry_aligned,
                        right_transfer,
                        transfer_progress,
                    )
            elif (
                self._right_acquired_palm is not None
                and self._right_acquired_step is not None
            ):
                # Preload the closed physical grasp, then correct its
                # contact-owned orientation slowly.  A separate pre-lift was
                # physically observed to clear the thin handle before any
                # correction could occur; reaching the 15-degree
                # entry gate naturally transitions to the upward handover.
                acquired_elapsed = int(self._right_acquisition_progress)
                if not right_tail_physical:
                    # Pause progress and seek the verified handle-relative
                    # grasp point from the measured palm.  The object remains
                    # ballistic while contact is absent; only the robot target
                    # moves until the authoritative force query recovers.
                    current_tool = self.raw_env._tool_matrix()
                    current_palm = _matrix_pose_vector(
                        self.raw_env._palm_matrix("right")
                    )
                    current_local = (
                        np.linalg.inv(current_tool)
                        @ _pose_vector_to_matrix(current_palm)
                    )[:3, 3]
                    correction_world = current_tool[:3, :3] @ (
                        _RIGHT_RESEAT_TARGET_LOCAL - current_local
                    )
                    correction_world *= _RIGHT_RESEAT_COMMAND_FEEDBACK_GAIN
                    correction_norm = float(np.linalg.norm(correction_world))
                    if correction_norm > _RIGHT_RESEAT_MAX_CORRECTION_M:
                        correction_world *= (
                            _RIGHT_RESEAT_MAX_CORRECTION_M / correction_norm
                        )
                    right = current_palm
                    right[:3] += correction_world
                    # Reuse the wrist pose measured at the authoritative
                    # contact frame.  Feeding the measured DLS pose back into
                    # itself while separated otherwise accumulates several
                    # degrees of error and changes which fingers meet the
                    # cylindrical handle even after position has converged.
                    right[3:7] = self._right_acquired_palm[3:7]
                elif acquired_elapsed <= _RIGHT_CONTACT_PRELOAD_STEPS:
                    right = self._smooth(
                        self._right_acquired_palm,
                        right_preload,
                        acquired_elapsed / _RIGHT_CONTACT_PRELOAD_STEPS,
                    )
                elif (
                    _RIGHT_POST_GRASP_LIFT_STEPS > 0
                    and acquired_elapsed
                    <= (
                        _RIGHT_CONTACT_PRELOAD_STEPS
                        + _RIGHT_POST_GRASP_LIFT_STEPS
                    )
                ):
                    lift_elapsed = (
                        acquired_elapsed - _RIGHT_CONTACT_PRELOAD_STEPS
                    )
                    right = self._smooth(
                        right_preload,
                        right_lift,
                        lift_elapsed / _RIGHT_POST_GRASP_LIFT_STEPS,
                    )
                else:
                    right = self._smooth(
                        right_lift,
                        right_alignment,
                        (
                            acquired_elapsed
                            - _RIGHT_CONTACT_PRELOAD_STEPS
                            - _RIGHT_POST_GRASP_LIFT_STEPS
                        )
                        / _RIGHT_ORIENTATION_ALIGNMENT_STEPS,
                    )
            else:
                assert self._right_pick_arrived_step is not None
                acquire_start_step = (
                    self._right_pick_arrived_step + _RIGHT_GRASP_CLOSE_STEPS
                )
                acquire_elapsed = max(0, step - acquire_start_step)
                right = self._smooth(
                    right_pick,
                    right_acquire,
                    acquire_elapsed / _RIGHT_ACQUIRE_APPROACH_STEPS,
                )
                if (
                    acquire_elapsed >= _RIGHT_ACQUIRE_APPROACH_STEPS
                    and self._right_acquire_arrived_step is None
                ):
                    self._update_right_arrival_gate(
                        gate="acquire",
                        target=right_acquire,
                        start_step=(
                            acquire_start_step
                            + _RIGHT_ACQUIRE_APPROACH_STEPS
                        ),
                        timeout_steps=_RIGHT_ACQUIRE_ARRIVAL_TIMEOUT_STEPS,
                    )
                if self._right_acquire_arrived_step is not None:
                    self.raw_env._as019_automatic_acquire_gate_open = True
                    self.raw_env._as019_automatic_arrival_gate["open"] = True
            if stage == 1 and self._right_entry_palm is not None:
                # The live sweep showed that translating toward a canonical
                # handoff pose made the receiver chase a moving cylinder.  A
                # measured entry hold with only a 5 mm lateral/downward ease
                # kept real giver contact stable and is straightforward for a
                # teleoperator to reproduce.
                hold_progress = np.clip(
                    max(0, step - right_tail_step) / _RIGHT_HANDOVER_STEPS,
                    0.0,
                    1.0,
                )
                hold_progress = (
                    hold_progress
                    * hold_progress
                    * (3.0 - 2.0 * hold_progress)
                )
                right = self._right_entry_palm.copy()
                right[:3] += hold_progress * _RIGHT_HANDOVER_TRANSLATION_WORLD
            left_anchor = self.raw_env._as019_assist_anchor.get("left")
            if left_anchor is None:
                right_hand[:] = _AUTOMATIC_GIVER_GRASP_TARGET
            else:
                regrasp_elapsed = max(
                    0, step - int(left_anchor.get("attached_step", step))
                )
                if regrasp_elapsed < _AUTOMATIC_HANDOVER_REGRASP_OPEN_STEPS:
                    open_progress = (
                        regrasp_elapsed
                        / _AUTOMATIC_HANDOVER_REGRASP_OPEN_STEPS
                    )
                    right_hand[:] = (
                        (1.0 - open_progress)
                        * _AUTOMATIC_GIVER_GRASP_TARGET
                        + open_progress * _AUTOMATIC_GIVER_REGRASP_OPEN_TARGET
                    )
                elif regrasp_elapsed < (
                    _AUTOMATIC_HANDOVER_REGRASP_OPEN_STEPS
                    + _AUTOMATIC_HANDOVER_REGRASP_CLOSE_STEPS
                ):
                    close_progress = (
                        regrasp_elapsed
                        - _AUTOMATIC_HANDOVER_REGRASP_OPEN_STEPS
                    ) / _AUTOMATIC_HANDOVER_REGRASP_CLOSE_STEPS
                    right_hand[:] = (
                        (1.0 - close_progress)
                        * _AUTOMATIC_GIVER_REGRASP_OPEN_TARGET
                        + close_progress * _AUTOMATIC_GIVER_GRASP_TARGET
                    )
                else:
                    right_hand[:] = _AUTOMATIC_GIVER_GRASP_TARGET
        elif stage == 2:
            phase = 5
            if not self._right_dual_contact_latched:
                # The demonstrator may have reached dual contact while its
                # collision-safe stage-1 hold differs from the older planned
                # transfer endpoint.  Re-latch the authoritative physical
                # palm once on stage entry so the giver release begins with a
                # zero-jump command instead of restoring stale plan state.
                self._right_contact_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("right")
                )
                self._right_dual_contact_latched = True
                right_hold = self._right_contact_palm.copy()
            right = right_hold
            dual_contact_step = int(self.raw_env._as019_event_steps[1])
            if (
                "left" in self.raw_env._as019_assist_anchor
                and dual_contact_step >= 0
                and self.raw_env._receiver_has_physical_pinch(
                    self.raw_env._as019_last_metrics
                )
                and _grasp_command_max(self.raw_env._left_hand)
                >= _HAND_CLOSE_THRESHOLD
                and (
                    self._receiver_contact_palm is None
                    or int(self._receiver_contact_step or -1)
                    < dual_contact_step
                )
            ):
                # A glancing approach contact can precede the authoritative
                # dual-contact event.  Re-latch the measured receiver palm at
                # that event instead of carrying the stale first-touch pose
                # into the required 12-frame support hold.
                self._receiver_contact_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("left")
                )
                self._receiver_contact_step = step
                self._receiver_preload_palm = self._receiver_contact_palm.copy()
                contact_direction = (
                    np.asarray(left_handoff, dtype=np.float64)[:3]
                    - self._receiver_contact_palm[:3]
                )
                contact_distance = float(np.linalg.norm(contact_direction))
                if contact_distance > 1.0e-9:
                    self._receiver_preload_palm[:3] += (
                        _LEFT_RECEIVER_CONTACT_PRELOAD_DISTANCE_M
                        * contact_direction
                        / contact_distance
                    )
            if self._receiver_contact_palm is not None:
                left = self._receiver_contact_palm.copy()
                left_anchor = self.raw_env._as019_assist_anchor.get("left")
                if (
                    left_anchor is not None
                    and "hand_to_tool" in left_anchor
                    and self.raw_env._receiver_has_physical_pinch(
                        self.raw_env._as019_last_metrics
                    )
                    and _grasp_command_max(self.raw_env._left_hand)
                    >= _HAND_CLOSE_THRESHOLD
                ):
                    left = _owner_palm_pose(tool, left_anchor["hand_to_tool"])
                if self._receiver_preload_palm is not None:
                    preload_progress = np.clip(
                        max(
                            0,
                            step - int(self._receiver_contact_step or step),
                        )
                        / _LEFT_RECEIVER_CONTACT_PRELOAD_STEPS,
                        0.0,
                        1.0,
                    )
                    left[:3] += preload_progress * (
                        self._receiver_preload_palm[:3]
                        - self._receiver_contact_palm[:3]
                    )
            else:
                left = left_handoff
            receiver_hold_complete_step = int(
                self.raw_env._as019_receiver_hold_complete_step
            )
            if receiver_hold_complete_step < 0:
                right_hand[:] = _AUTOMATIC_GIVER_GRASP_TARGET
            else:
                elapsed = max(0, step - receiver_hold_complete_step)
                thumb_progress = np.clip(
                    elapsed / _RIGHT_RELEASE_THUMB_STEPS, 0.0, 1.0
                )
                finger_progress = np.clip(
                    (elapsed - _RIGHT_RELEASE_THUMB_STEPS)
                    / _RIGHT_RELEASE_FINGER_STEPS,
                    0.0,
                    1.0,
                )
                right_hand[:] = _AUTOMATIC_GIVER_GRASP_TARGET
                right_hand[:4] *= 1.0 - thumb_progress
                right_hand[_RIGHT_FINGER_FLEXION_INDICES] *= (
                    1.0 - finger_progress
                )
                right_hand[_HAND_ABDUCTION_INDICES[1:]] *= (
                    1.0 - finger_progress
                )
                if elapsed > _RIGHT_RELEASE_THUMB_STEPS:
                    # Once the opposing thumb is fully open, begin a slow
                    # tool-normal unload while the support fingers finish
                    # opening.  Waiting until the final finger command reached
                    # zero left a one-frame wedged contact that rotated the tip
                    # before the old post-open retreat could begin.
                    unload_progress = np.clip(
                        (elapsed - _RIGHT_RELEASE_THUMB_STEPS)
                        / _RIGHT_POST_OPEN_UNLOAD_STEPS,
                        0.0,
                        1.0,
                    )
                    _, right_normal = _tool_normal_retreat_pose(
                        tool, right_hold, zone
                    )
                    right_unload = right_hold.copy()
                    right_unload[:3] += (
                        right_normal * _RIGHT_POST_OPEN_UNLOAD_DISTANCE_M
                    )
                    right = self._smooth(
                        right_hold, right_unload, unload_progress
                    )
            if self._receiver_thumb_contact_palm is not None:
                # Preserve the measured, two-sided receiver grasp while the
                # giver opens.  Following the compliant object anchor during
                # this overlap lets the redundant constraints translate the
                # receiver by centimetres before giver-clear.  Stage 3
                # re-latches the live grasp once and then owns placement.  A
                # cleared hand must not chase its own measured DLS drift or
                # re-enable that chase on a later weak finger brush; retain
                # the already collision-observed thumb pose for the rest of
                # the giver release instead.
                dual_contact_offset = (
                    float(
                        np.linalg.norm(
                            self._receiver_contact_palm[:3]
                            - self._receiver_thumb_contact_palm[:3]
                        )
                    )
                    if self._receiver_contact_palm is not None
                    else 0.0
                )
                receiver_contact_live = bool(
                    self.raw_env._as019_last_metrics.get(
                        "contact_query_valid", False
                    )
                    and self.raw_env._as019_last_metrics.get(
                        "left_middle_physical", False
                    )
                    and _grasp_command_max(self.raw_env._left_hand)
                    >= _HAND_CLOSE_THRESHOLD
                )
                if not receiver_contact_live:
                    self._stage2_receiver_clear_latched = True
                if (
                    receiver_contact_live
                    and not self._stage2_receiver_clear_latched
                    and dual_contact_offset
                    > _STAGE2_DUAL_CONTACT_LIVE_HOLD_MIN_M
                ):
                    left = _matrix_pose_vector(
                        self.raw_env._palm_matrix("left")
                    )
                else:
                    left = self._receiver_thumb_contact_palm.copy()
            left_hand[:] = _AUTOMATIC_RECEIVER_GRASP_TARGET
            left, left_hand, stage2_regrasp_handled = (
                self._stage3_regrasp_command(
                    step=step,
                    tool=tool,
                    metrics=self.raw_env._as019_last_metrics,
                    left=left,
                    left_hand=left_hand,
                )
            )
            if stage2_regrasp_handled:
                # Recovery is admitted only while the giver still owns real
                # support.  Do not let a stale hold-complete sample open it.
                right = right_hold
                right_hand[:] = _AUTOMATIC_GIVER_GRASP_TARGET
        elif stage == 3:
            phase = 6
            elapsed = max(0, step - int(self.raw_env._as019_event_steps[2]))
            metrics = self.raw_env._as019_last_metrics
            physical_left_contact = bool(
                metrics.get("contact_query_valid", False)
                and metrics.get("left_middle_physical", False)
                and _grasp_command_max(self.raw_env._left_hand)
                >= _HAND_CLOSE_THRESHOLD
            )
            if physical_left_contact:
                self._placement_contact_loss_streak = 0
            elif (
                self._placement_ready_step is not None
                and self._placement_contact_active
            ):
                self._placement_contact_loss_streak += 1
            else:
                self._placement_contact_loss_streak = 0
            placement_contact_debounced = bool(
                physical_left_contact
                or (
                    self._placement_ready_step is not None
                    and self._placement_contact_active
                    and self._placement_contact_loss_streak
                    <= _PLACEMENT_CONTACT_LOSS_DEBOUNCE_STEPS
                )
            )
            left_anchor = self.raw_env._as019_assist_anchor.get("left")
            if (
                not self._left_stage3_relatched
                and physical_left_contact
                and left_anchor is not None
            ):
                # The 15 mm receiver preload seats the handle after its first
                # touch.  Rebind once at stage entry so the placement branch
                # preserves that measured physical grasp instead of restoring
                # the earlier fingertip contact transform.
                left_anchor["hand_to_tool"] = (
                    np.linalg.inv(self.raw_env._palm_matrix("left"))
                    @ self.raw_env._tool_matrix()
                )
                left_anchor["attached_step"] = step
                self._zone_plan = None
                self._left_stage3_relatched = True
            plan = self._zone_motion_plan(tool, zone)
            if plan is not None:
                if elapsed <= _RIGHT_RETREAT_NORMAL_STEPS:
                    right = self._smooth(
                        plan["right_start"],
                        plan["right_clear"],
                        elapsed / _RIGHT_RETREAT_NORMAL_STEPS,
                    )
                else:
                    right = self._smooth(
                        plan["right_clear"],
                        right_home,
                        (elapsed - _RIGHT_RETREAT_NORMAL_STEPS)
                        / _RIGHT_RETREAT_HOME_STEPS,
                    )
                physically_right_clear = bool(metrics.get("right_clear", False))
                if self._stage3_regrasp_active:
                    # Do not let a one-sided brush start the placement path
                    # while the recovery is still building its physical pinch.
                    physically_right_clear = False
                grip_settled = bool(
                    elapsed >= _AUTOMATIC_RECEIVER_PREPLACEMENT_GRIP_STEPS
                )
                if elapsed >= _RIGHT_RETREAT_NORMAL_STEPS and grip_settled:
                    self._right_clear_streak = (
                        self._right_clear_streak + 1
                        if physically_right_clear
                        else 0
                    )
                else:
                    self._right_clear_streak = 0
                if (
                    self._placement_ready_step is None
                    and self._right_clear_streak >= _RIGHT_CLEAR_READY_STEPS
                    and left_anchor is not None
                    and physical_left_contact
                    and (
                        self._placement_reacquire_progress == 0
                        or self._placement_contact_active
                    )
                ):
                    # Longitudinal roll is deliberately free while the giver
                    # retreats.  Rebuild all contact-owned waypoints from the
                    # live tool pose when placement actually becomes legal;
                    # retaining the stage-entry plan while changing only its
                    # start palm can leave a centimetre-scale endpoint jump.
                    self._zone_plan = None
                    plan = self._zone_motion_plan(tool, zone)
                    assert plan is not None
                    self._placement_ready_step = step
                    # Begin at the live measured receiver pose.  The semantic
                    # hand-to-tool anchor intentionally survives small DLS
                    # tracking residuals; reconstructing the palm from that
                    # anchor here can therefore introduce a centimetre-scale
                    # first-frame jump exactly when placement becomes legal.
                    # The measured palm is already in authoritative physical
                    # contact and leaves every later endpoint unchanged.
                    plan["start_palm"] = _matrix_pose_vector(
                        self.raw_env._palm_matrix("left")
                    )
                    self._placement_progress = 0
                    self._placement_progress_step = step
                    self._placement_contact_active = True
                    self._placement_reacquire_progress = 0
                    self._placement_reacquire_progress_step = None
                    self._placement_reacquire_contact_streak = 0
                    self._placement_contact_loss_streak = 0
                    self.raw_env._as019_placement_motion_started = True
                    self.raw_env._as019_placement_assist_enabled = True
                if plan["anchor"] is None:
                    left = left_handoff
                elif self._placement_ready_step is None:
                    # Until placement assistance is legal, keep the closed
                    # receiver on its live contact-owned transform.  A fixed
                    # world palm lets the cylindrical handle roll out of the
                    # fingers and turns later wrench activation into an
                    # invisible pull.
                    anchor_seek = _owner_palm_pose(tool, plan["anchor"])
                    if physical_left_contact:
                        if self._placement_reacquire_progress > 0:
                            self._placement_reacquire_contact_streak += 1
                            grasp_seek = self._placement_reacquire_pose(
                                tool, anchor_seek
                            )
                            if (
                                self._placement_reacquire_contact_streak
                                < _PLACEMENT_REACQUIRE_CONTACT_STEPS
                            ):
                                left = self._smooth(
                                    anchor_seek,
                                    grasp_seek,
                                    self._placement_reacquire_progress
                                    / _PLACEMENT_REACQUIRE_STEPS,
                                )
                            else:
                                # The second recovery occurred before the
                                # placement-ready latch.  Rebind only after
                                # the same authoritative contact debounce used
                                # by the ordinary placement recovery, then
                                # rebuild the path on the next frame.
                                left_anchor["hand_to_tool"] = (
                                    np.linalg.inv(
                                        self.raw_env._palm_matrix("left")
                                    )
                                    @ self.raw_env._tool_matrix()
                                )
                                left_anchor["attached_step"] = step
                                self._zone_plan = None
                                self._placement_reacquire_progress = 0
                                self._placement_reacquire_progress_step = None
                                self._placement_reacquire_contact_streak = 0
                                self._right_clear_streak = 0
                                self.raw_env._as019_placement_motion_started = (
                                    False
                                )
                                left = _matrix_pose_vector(
                                    self.raw_env._palm_matrix("left")
                                )
                        else:
                            left = anchor_seek
                    else:
                        # A recovered grasp can clear again during the short
                        # right-clear readiness dwell.  That state has no
                        # placement-ready latch yet, so the ordinary recovery
                        # branch below is not reachable.  Continue the same
                        # robot-only handle entrance here instead of holding a
                        # separated semantic anchor forever.
                        self._placement_contact_active = False
                        self._placement_reacquire_contact_streak = 0
                        if not grip_settled or not physically_right_clear:
                            # Do not press the receiver inward while the giver
                            # is still unloading or before the post-release
                            # grip dwell has elapsed.
                            left = anchor_seek
                        else:
                            if self._placement_reacquire_progress_step != step:
                                self._placement_reacquire_progress += 1
                                self._placement_reacquire_progress_step = step
                            grasp_seek = self._placement_reacquire_pose(
                                tool, anchor_seek
                            )
                            left = self._smooth(
                                anchor_seek,
                                grasp_seek,
                                self._placement_reacquire_progress
                                / _PLACEMENT_REACQUIRE_STEPS,
                            )
                    self.raw_env._as019_placement_assist_enabled = False
                elif not placement_contact_debounced:
                    # Do not advance a contact-owned path after the tool has
                    # left the receiver.  Disable object assistance and move
                    # only the closed robot hand inward by the same 3 mm
                    # radial preload that established the original physical
                    # contact.  Returning to the nominal approach pose here
                    # can be centimetres away from a valid randomized grasp
                    # and sweep the table-supported tool across the workspace.
                    self.raw_env._as019_placement_assist_enabled = False
                    self._placement_contact_active = False
                    self._placement_reacquire_contact_streak = 0
                    self._placement_contact_loss_streak = 0
                    if self._placement_reacquire_progress_step != step:
                        self._placement_reacquire_progress += 1
                        self._placement_reacquire_progress_step = step
                    anchor_seek = _owner_palm_pose(tool, plan["anchor"])
                    grasp_seek = self._placement_reacquire_pose(
                        tool, anchor_seek
                    )
                    left = self._smooth(
                        anchor_seek,
                        grasp_seek,
                        self._placement_reacquire_progress
                        / _PLACEMENT_REACQUIRE_STEPS,
                    )
                else:
                    if not self._placement_contact_active:
                        self._placement_reacquire_contact_streak += 1
                        if (
                            self._placement_reacquire_progress > 0
                            and self._placement_reacquire_contact_streak
                            < _PLACEMENT_REACQUIRE_CONTACT_STEPS
                        ):
                            # A single collision sample is not yet a recovered
                            # grasp.  Continue the same robot-only entrance
                            # command through a short physical debounce so a
                            # one-frame touch cannot alternate the target
                            # between seek and hold poses.
                            anchor_seek = _owner_palm_pose(
                                tool, plan["anchor"]
                            )
                            grasp_seek = self._placement_reacquire_pose(
                                tool, anchor_seek
                            )
                            left = self._smooth(
                                anchor_seek,
                                grasp_seek,
                                self._placement_reacquire_progress
                                / _PLACEMENT_REACQUIRE_STEPS,
                            )
                            self.raw_env._as019_placement_assist_enabled = False
                        else:
                            # Rebind the recovered physical grasp and rebuild
                            # the path from the live tool pose only after the
                            # contact debounce is authoritative.  Resuming the
                            # stale pre-drop path would jump and eject it again.
                            left_anchor["hand_to_tool"] = (
                                np.linalg.inv(self.raw_env._palm_matrix("left"))
                                @ self.raw_env._tool_matrix()
                            )
                            left_anchor["attached_step"] = step
                            self._zone_plan = None
                            self._placement_ready_step = None
                            self._placement_progress = 0
                            self._placement_progress_step = None
                            self._placement_contact_active = True
                            self._placement_reacquire_progress = 0
                            self._placement_reacquire_progress_step = None
                            self._placement_reacquire_contact_streak = 0
                            self._placement_contact_loss_streak = 0
                            self._right_clear_streak = 0
                            self.raw_env._as019_placement_motion_started = False
                            self.raw_env._as019_placement_assist_enabled = False
                            left = _matrix_pose_vector(
                                self.raw_env._palm_matrix("left")
                            )
                    else:
                        if (
                            physical_left_contact
                            and self._placement_progress_step != step
                        ):
                            self._placement_progress += 1
                            self._placement_progress_step = step
                        # Keep the robot command at the current path sample
                        # during a debounced force gap.  The raw-environment
                        # substep gate independently disables every object
                        # wrench until physical contact is live again.
                        self.raw_env._as019_placement_assist_enabled = bool(
                            physical_left_contact
                        )
                        placement_elapsed = self._placement_progress
                        if placement_elapsed <= _ZONE_X_ALIGN_STEPS:
                            phase = 7
                            left = self._smooth(
                                plan["start_palm"],
                                plan["x_aligned_palm"],
                                placement_elapsed / _ZONE_X_ALIGN_STEPS,
                            )
                        elif placement_elapsed <= _ZONE_X_ALIGN_STEPS + _ZONE_Y_ALIGN_STEPS:
                            phase = 8
                            left = self._smooth(
                                plan["x_aligned_palm"],
                                plan["hover_palm"],
                                (placement_elapsed - _ZONE_X_ALIGN_STEPS)
                                / _ZONE_Y_ALIGN_STEPS,
                            )
                        elif (
                            placement_elapsed
                            <= _ZONE_X_ALIGN_STEPS
                            + _ZONE_Y_ALIGN_STEPS
                            + _ZONE_DESCENT_STEPS
                        ):
                            phase = 9
                            descent_progress = (
                                (
                                    placement_elapsed
                                    - _ZONE_X_ALIGN_STEPS
                                    - _ZONE_Y_ALIGN_STEPS
                                )
                                / _ZONE_DESCENT_STEPS
                            )
                            left = self._smooth(
                                plan["hover_palm"],
                                plan["lowered_palm"],
                                descent_progress,
                            )
                        else:
                            phase = 9
                            left = self._placement_command(plan, tool)
            else:
                # A right-release latch without the physical left-owner anchor
                # is inconsistent.  Hold the live handle pose rather than
                # guessing a fixed-world placement command.
                left = left_handoff
            pinch_record = self.raw_env._as019_receiver_pinch_freeze
            pinch_positions = pinch_record.get("joint_positions")
            receiver_start = (
                np.asarray(pinch_positions, dtype=np.float64)
                if pinch_positions is not None
                else _AUTOMATIC_RECEIVER_GRASP_TARGET
            )
            # A physical pinch can legitimately close at a different thumb
            # posture under reset randomization.  Only the already-normalized
            # (near-template) case may continue toward the nominal stage-3
            # target.  A distinct pinch strengthens along its own measured
            # closing trajectory instead of being driven toward that template.
            if (
                pinch_positions is None
                or bool(pinch_record.get("normalization_applied", False))
            ):
                receiver_end = _AUTOMATIC_RECEIVER_STAGE3_TARGET
            else:
                # A distinct pinch is already strengthened along its own
                # measured close trajectory at the physical contact substep.
                # Preserve that exact target after giver-clear; applying a
                # second close ramp torques otherwise stable reset poses.
                receiver_end = receiver_start
            close_progress = np.clip(
                (elapsed + 1) / _AUTOMATIC_RECEIVER_STAGE3_CLOSE_STEPS,
                0.0,
                1.0,
            )
            left_hand[:] = (
                (1.0 - close_progress) * receiver_start
                + close_progress * receiver_end
            )
            if self.raw_env._as019_receiver_pinch_target is not None:
                self.raw_env._as019_receiver_pinch_target = (
                    self.raw_env._as019_receiver_pinch_target.new_tensor(
                        left_hand
                    ).reshape(1, -1)
                )
                self.raw_env._left_hand[:] = left_hand
            left, left_hand, _regrasp_handled = self._stage3_regrasp_command(
                step=step,
                tool=tool,
                metrics=metrics,
                left=left,
                left_hand=left_hand,
            )
        elif stage == 4:
            phase = 10
            right = right_home
            if self._release_grasp_target is None:
                # Preserve the live stage-3 grasp at the zone-entry boundary.
                # Replacing this reset-specific physical pinch with the fixed
                # nominal template creates a large joint jump while the tool
                # is table-supported, preventing the unchanged quiet-release
                # gate from ever accumulating its 12 required frames.
                self._release_grasp_target = np.asarray(
                    self.raw_env._left_hand, dtype=np.float64
                ).reshape(-1).copy()
            release_grasp_target = self._release_grasp_target
            plan = self._zone_motion_plan(tool, zone)
            placement_total_steps = (
                _ZONE_X_ALIGN_STEPS
                + _ZONE_Y_ALIGN_STEPS
                + _ZONE_DESCENT_STEPS
            )
            metrics = self.raw_env._as019_last_metrics
            physical_left_contact = bool(
                metrics.get("contact_query_valid", False)
                and metrics.get("left_middle_physical", False)
                and _grasp_command_max(self.raw_env._left_hand)
                >= _HAND_CLOSE_THRESHOLD
            )
            left_anchor = self.raw_env._as019_assist_anchor.get("left")
            stage4_recovering = bool(
                self._release_ready_step is None
                and plan is not None
                and left_anchor is not None
                and not physical_left_contact
                and self._placement_progress < placement_total_steps
            )
            stage4_recovered = False
            stage4_confirming = False
            if stage4_recovering:
                # Table contact can unload a weak but valid pinch on the same
                # control frame that latches the in-zone event.  Keep the tool
                # ballistic and move only the still-closed receiver back
                # through the verified handle entrance.  Release may not begin
                # until this produces new authoritative physical contact.
                if self._placement_hold_palm is None:
                    self._placement_hold_palm = _matrix_pose_vector(
                        self.raw_env._palm_matrix("left")
                    )
                if self._placement_reacquire_progress_step != step:
                    self._placement_reacquire_progress += 1
                    self._placement_reacquire_progress_step = step
                self._placement_reacquire_contact_streak = 0
                assert self._placement_hold_palm is not None
                grasp_seek = _tool_frame_pose(
                    tool,
                    receiver_grasp_local,
                    receiver_local_quat,
                )
                left = self._smooth(
                    self._placement_hold_palm,
                    grasp_seek,
                    self._placement_reacquire_progress
                    / _PLACEMENT_REACQUIRE_STEPS,
                )
                self._zone_settle_step = None
                for key in (
                    "supported_roll_start_tool",
                    "supported_roll_start_palm",
                    "stable_placed_tool",
                    "stable_placed_palm",
                    "release_palm",
                ):
                    plan.pop(key, None)
                self.raw_env._as019_placement_assist_enabled = False
                self.raw_env._as019_release_support_enabled = False
            elif (
                self._placement_reacquire_progress > 0
                and physical_left_contact
                and left_anchor is not None
                and self._placement_hold_palm is not None
            ):
                self._placement_reacquire_contact_streak += 1
                if (
                    self._placement_reacquire_contact_streak
                    < _PLACEMENT_REACQUIRE_CONTACT_STEPS
                ):
                    assert self._placement_hold_palm is not None
                    grasp_seek = _tool_frame_pose(
                        tool,
                        receiver_grasp_local,
                        receiver_local_quat,
                    )
                    left = self._smooth(
                        self._placement_hold_palm,
                        grasp_seek,
                        self._placement_reacquire_progress
                        / _PLACEMENT_REACQUIRE_STEPS,
                    )
                    stage4_confirming = True
                    self.raw_env._as019_placement_assist_enabled = False
                    self.raw_env._as019_release_support_enabled = False
                else:
                    # Rebind only after contact remains physical for the full
                    # debounce.  Rebuild the supported target from the live
                    # tabletop pose, then hold one zero-jump frame before the
                    # stable-facet/release sequence resumes.
                    left_anchor["hand_to_tool"] = (
                        np.linalg.inv(self.raw_env._palm_matrix("left"))
                        @ self.raw_env._tool_matrix()
                    )
                    left_anchor["attached_step"] = step
                    preserved_release_grasp_target = release_grasp_target.copy()
                    self._zone_plan = None
                    plan = self._zone_motion_plan(tool, zone)
                    self._release_grasp_target = preserved_release_grasp_target
                    release_grasp_target = preserved_release_grasp_target
                    self._placement_progress = placement_total_steps
                    self._placement_progress_step = step
                    self._placement_reacquire_progress = 0
                    self._placement_reacquire_progress_step = None
                    self._placement_reacquire_contact_streak = 0
                    self._placement_contact_active = True
                    self._zone_settle_step = None
                    self._placement_hold_palm = _matrix_pose_vector(
                        self.raw_env._palm_matrix("left")
                    )
                    left = self._placement_hold_palm.copy()
                    stage4_recovered = True
                    self.raw_env._as019_placement_assist_enabled = False
                    self.raw_env._as019_release_support_enabled = True
            if (
                not stage4_recovering
                and not stage4_recovered
                and not stage4_confirming
                and plan is not None
                and plan["anchor"] is not None
                and self._placement_progress < placement_total_steps
                and float(
                    np.linalg.norm(
                        np.asarray(plan["placed_tool"], dtype=np.float64)[:3]
                        - np.asarray(tool, dtype=np.float64)[:3]
                    )
                )
                <= _ZONE_SUPPORT_READY_GAP_M
            ):
                # Tests and teleoperation can enter stage 4 only after the
                # physical endpoint has already been reached.  In that case
                # there is no unfinished path to preserve.
                self._placement_progress = placement_total_steps
                self._placement_progress_step = step
            continuing_placement = bool(
                not stage4_recovering
                and not stage4_recovered
                and not stage4_confirming
                and plan is not None
                and plan["anchor"] is not None
                and self._placement_progress < placement_total_steps
            )
            if stage4_recovering or stage4_recovered or stage4_confirming:
                left_hand[:] = release_grasp_target
            elif continuing_placement:
                # Entering the semantic zone can occur while the continuous
                # center-and-lower trajectory is still in progress.  Preserve
                # that already-safe path across the stage boundary instead of
                # jumping directly to the final endpoint correction, which can
                # roll a firmly held cylindrical handle back up off the table.
                if self._placement_progress_step != step:
                    self._placement_progress += 1
                    self._placement_progress_step = step
                placement_elapsed = self._placement_progress
                if placement_elapsed <= _ZONE_X_ALIGN_STEPS:
                    phase = 7
                    left = self._smooth(
                        plan["start_palm"],
                        plan["x_aligned_palm"],
                        placement_elapsed / _ZONE_X_ALIGN_STEPS,
                    )
                elif placement_elapsed <= (
                    _ZONE_X_ALIGN_STEPS + _ZONE_Y_ALIGN_STEPS
                ):
                    phase = 8
                    left = self._smooth(
                        plan["x_aligned_palm"],
                        plan["hover_palm"],
                        (placement_elapsed - _ZONE_X_ALIGN_STEPS)
                        / _ZONE_Y_ALIGN_STEPS,
                    )
                else:
                    phase = 9
                    left = self._smooth(
                        plan["hover_palm"],
                        plan["lowered_palm"],
                        (
                            placement_elapsed
                            - _ZONE_X_ALIGN_STEPS
                            - _ZONE_Y_ALIGN_STEPS
                        )
                        / _ZONE_DESCENT_STEPS,
                    )
                left_hand[:] = release_grasp_target
                self.raw_env._as019_placement_assist_enabled = True
                self.raw_env._as019_release_support_enabled = False
            elif plan is not None and plan["anchor"] is not None:
                if self._zone_settle_step is None:
                    self._zone_settle_step = step
                stable_roll_elapsed = max(
                    0, step - self._zone_settle_step
                )
                stable_roll_latch_steps = _ZONE_STABLE_ROLL_LATCH_DELAY_STEPS
                if (
                    physical_left_contact
                    and float(metrics.get("left_force", 0.0))
                    >= _ZONE_STABLE_ROLL_EARLY_MIN_CONTACT_FORCE_N
                    and float(metrics.get("outward_alignment", 1.0))
                    < _ZONE_STABLE_ROLL_EARLY_MAX_OUTWARD_ALIGNMENT
                ):
                    # A high-load, visibly pitched placement is already
                    # pushing the long tool toward an edge-balanced support
                    # pose.  Begin the same physical stable-facet motion while
                    # that grasp is still authoritative; ordinary low-load
                    # placements retain the verified twelve-frame delay.
                    stable_roll_latch_steps = (
                        _ZONE_STABLE_ROLL_EARLY_LATCH_STEPS
                    )
                if (
                    "supported_roll_start_tool" not in plan
                    and stable_roll_elapsed
                    >= stable_roll_latch_steps
                ):
                    supported_roll_start_tool = np.asarray(
                        plan["placed_tool"], dtype=np.float64
                    ).reshape(7).copy()
                    supported_roll_start_tool[3:7] = np.asarray(
                        tool, dtype=np.float64
                    ).reshape(7)[3:7]
                    stable_placed_tool = supported_roll_start_tool.copy()
                    stable_placed_tool[:2] = np.asarray(
                        plan["placed_tool"], dtype=np.float64
                    )[:2]
                    # The replacement wooden handle has four stable radial
                    # faces.  Snap only after tabletop support to the nearest
                    # 90-degree face; otherwise an edge-balanced release can
                    # rock after the hand clears.  The existing supported-roll
                    # interpolation applies the bounded correction gradually
                    # while the receiver still owns a physical opposed pinch.
                    stable_placed_tool[3:7] = _stable_tool_quaternion(
                        supported_roll_start_tool
                    )
                    stable_placed_tool[2] = (
                        float(zone[2])
                        - _MARKER_CENTER_ABOVE_TABLE_M
                        + _tool_vertical_support_depth(
                            stable_placed_tool[3:7]
                        )
                        + _ZONE_SUPPORT_CLEARANCE_M
                    )
                    # The cylindrical handle may roll a few degrees inside
                    # the closed receiver after it first touches the table.
                    # The original hand-to-tool anchor was measured before
                    # that slip, so reusing it here commands the *entire*
                    # observed slip as a sudden palm displacement.  Start
                    # from the already continuous placement command and
                    # apply only the small live-to-nearest-facet rotation.
                    # At stable_roll_progress == 0 this therefore reproduces
                    # the preceding stage-4 command exactly.
                    previous_action = np.asarray(
                        self.raw_env._as019_last_policy_action,
                        dtype=np.float64,
                    ).reshape(-1)
                    if (
                        self.raw_env._as019_last_policy_action_step == step - 1
                        and previous_action.shape == (54,)
                    ):
                        # Stage 4 may be entered one frame before the final
                        # continuous placement command.  Seed the supported
                        # roll from the exact preceding wrist command so the
                        # semantic stage edge cannot introduce an orientation
                        # jump that opens an otherwise physical pinch.
                        supported_roll_start_palm = previous_action[7:14].copy()
                    else:
                        supported_roll_start_palm = self._placement_command(
                            plan, tool
                        )
                    live_rotation = _quat_wxyz_to_matrix(
                        supported_roll_start_tool[3:7]
                    )
                    stable_rotation = _quat_wxyz_to_matrix(
                        stable_placed_tool[3:7]
                    )
                    stable_delta = stable_rotation @ live_rotation.T
                    stable_placed_palm = supported_roll_start_palm.copy()
                    stable_placed_palm[:3] = (
                        stable_placed_tool[:3]
                        + stable_delta
                        @ (
                            supported_roll_start_palm[:3]
                            - supported_roll_start_tool[:3]
                        )
                    )
                    stable_placed_palm[3:7] = _matrix_to_quat_wxyz(
                        stable_delta
                        @ _quat_wxyz_to_matrix(
                            supported_roll_start_palm[3:7]
                        )
                    )
                    release_palm = stable_placed_palm.copy()
                    release_palm[:3] += (
                        _LEFT_RELEASE_CLEAR_DISTANCE_M
                        * _tool_blunt_release_direction(stable_placed_tool)
                    )
                    if (
                        not physical_left_contact
                        or float(metrics.get("left_force", 0.0))
                        < _ZONE_STABLE_ROLL_MIN_CONTACT_FORCE_N
                    ):
                        # A few weak-grasp poses naturally unload the hand as
                        # the handle reaches the tabletop.  Do not send an
                        # already-clear or sub-newton closed hand back through
                        # the tool just to command a convex-facet roll.  The
                        # live tool is already in the accepted zone and will
                        # settle on its physical hull; keep the exact preceding
                        # hand command and let the quiet gate observe that pose.
                        supported_roll_start_tool = np.asarray(
                            tool, dtype=np.float64
                        ).reshape(7).copy()
                        stable_placed_tool = supported_roll_start_tool.copy()
                        if (
                            self.raw_env._as019_last_policy_action_step
                            == step - 1
                            and previous_action.shape == (54,)
                        ):
                            supported_roll_start_palm = (
                                previous_action[7:14].copy()
                            )
                        else:
                            supported_roll_start_palm = _matrix_pose_vector(
                                self.raw_env._palm_matrix("left")
                            )
                        stable_placed_palm = (
                            supported_roll_start_palm.copy()
                        )
                        release_palm = stable_placed_palm.copy()
                        release_palm[:3] += (
                            _LEFT_RELEASE_CLEAR_DISTANCE_M
                            * _tool_blunt_release_direction(
                                stable_placed_tool
                            )
                        )
                        stable_roll_elapsed = (
                            _ZONE_STABLE_ROLL_LATCH_DELAY_STEPS
                            + _ZONE_STABLE_ROLL_STEPS
                        )
                    plan["supported_roll_start_tool"] = (
                        supported_roll_start_tool
                    )
                    plan["supported_roll_latch_steps"] = int(
                        stable_roll_latch_steps
                    )
                    plan["supported_roll_start_palm"] = (
                        supported_roll_start_palm
                    )
                    plan["stable_placed_tool"] = stable_placed_tool
                    plan["stable_placed_palm"] = stable_placed_palm
                    plan["release_palm"] = release_palm
                supported_roll_latched = bool(
                    "supported_roll_start_tool" in plan
                )
                supported_roll_latch_steps = int(
                    plan.get(
                        "supported_roll_latch_steps",
                        _ZONE_STABLE_ROLL_LATCH_DELAY_STEPS,
                    )
                )
                stable_roll_progress = (
                    float(
                        np.clip(
                            (
                                stable_roll_elapsed
                                - supported_roll_latch_steps
                            )
                            / _ZONE_STABLE_ROLL_STEPS,
                            0.0,
                            1.0,
                        )
                    )
                    if supported_roll_latched
                    else 0.0
                )
                if self._release_ready_step is None:
                    # Continue the measured endpoint correction after the
                    # zone-entry event.  Freezing its first command left the
                    # rolled, off-origin handle 9--12 mm in the air even
                    # though the unchanged zone predicate was already true.
                    if self._release_quiet_hold_palm is not None:
                        self._placement_hold_palm = (
                            self._release_quiet_hold_palm.copy()
                        )
                    elif supported_roll_latched:
                        stable_tool = self._smooth(
                            plan["supported_roll_start_tool"],
                            plan["stable_placed_tool"],
                            stable_roll_progress,
                        )
                        stable_palm = self._smooth(
                            plan["supported_roll_start_palm"],
                            plan["stable_placed_palm"],
                            stable_roll_progress,
                        )
                        self._placement_hold_palm = self._placement_command(
                            {
                                "placed_tool": stable_tool,
                                "placed_palm": stable_palm,
                            },
                            tool,
                        )
                    else:
                        self._placement_hold_palm = self._placement_command(
                            plan, tool
                        )
                    release_offset = (
                        self._placement_hold_palm[:3]
                        - np.asarray(
                            plan["stable_placed_palm"], dtype=np.float64
                        )[:3]
                    )
                    self._release_hold_palm = np.asarray(
                        plan["release_palm"], dtype=np.float64
                    ).copy()
                    self._release_hold_palm[:3] += release_offset
                    self._release_open_unload_palm = self._placement_hold_palm.copy()
                    open_unload_direction = (
                        self._release_hold_palm[:3]
                        - self._placement_hold_palm[:3]
                    )
                    open_unload_direction_norm = float(
                        np.linalg.norm(open_unload_direction)
                    )
                    if open_unload_direction_norm > 1.0e-9:
                        self._release_open_unload_palm[:3] += (
                            _LEFT_RELEASE_OPEN_UNLOAD_DISTANCE_M
                            * open_unload_direction
                            / open_unload_direction_norm
                        )
                assert self._placement_hold_palm is not None
                left = self._placement_hold_palm
                metrics = self.raw_env._as019_last_metrics
                left_owner = "left" in self.raw_env._as019_assist_anchor
                left_contact = bool(
                    metrics.get("contact_query_valid", False)
                    and metrics.get("left_middle_physical", False)
                )
                left_hand_closed = bool(
                    float(np.max(np.abs(self.raw_env._left_hand)))
                    >= _HAND_CLOSE_THRESHOLD
                )
                right_clear = bool(metrics.get("right_clear", False))
                release_ready = bool(
                    supported_roll_latched
                    and stable_roll_progress >= 1.0
                    and metrics.get("contact_query_valid", False)
                    and metrics.get("zone_inside", False)
                    and float(metrics.get("linear_speed", float("inf")))
                    <= _ZONE_RELEASE_MAX_LINEAR_SPEED_MPS
                    and float(metrics.get("angular_speed", float("inf")))
                    <= _ZONE_RELEASE_MAX_ANGULAR_SPEED_RADPS
                    and _tool_table_gap_m(tool, zone)
                    <= _ZONE_SUPPORT_READY_GAP_M
                    and right_clear
                    and left_owner
                    and left_contact
                    and left_hand_closed
                )
                # Once the contact-owned placement path and stable-facet move
                # are complete, remove every pose-restoring term.  Retain only
                # the lower contact-gated velocity-damping tier until a fresh
                # quiet window is measured; it cannot pull a separated tool or
                # bias its tabletop pose.
                self.raw_env._as019_placement_assist_enabled = False
                self.raw_env._as019_release_support_enabled = bool(
                    supported_roll_latched
                    and stable_roll_progress >= 1.0
                    and self._release_ready_step is None
                    and left_owner
                    and left_contact
                    and left_hand_closed
                )
                if self._release_ready_step is None:
                    quiet_pose_valid = bool(
                        supported_roll_latched
                        and stable_roll_progress >= 1.0
                        and metrics.get("contact_query_valid", False)
                        and metrics.get("zone_inside", False)
                        and _tool_table_gap_m(tool, zone)
                        <= _ZONE_SUPPORT_READY_GAP_M
                        and right_clear
                        and left_owner
                        and left_contact
                        and left_hand_closed
                    )
                    if (
                        self._release_quiet_hold_palm is not None
                        and not quiet_pose_valid
                    ):
                        self._release_quiet_hold_palm = None
                    if (
                        release_ready
                        and self._release_quiet_hold_palm is None
                    ):
                        # The first frame satisfying the unchanged physical
                        # release gates proves a reachable, quiet wrist pose.
                        # Hold that measured pose while accumulating the
                        # existing continuous readiness dwell, instead of
                        # exciting the supported tool with the last few
                        # millimetres of Cartesian endpoint residual.
                        self._release_quiet_hold_palm = _matrix_pose_vector(
                            self.raw_env._palm_matrix("left")
                        )
                        self._placement_hold_palm = (
                            self._release_quiet_hold_palm.copy()
                        )
                        left = self._placement_hold_palm.copy()
                    self._release_ready_streak = (
                        self._release_ready_streak + 1 if release_ready else 0
                    )
                    if self._release_ready_streak >= _ZONE_RELEASE_READY_STEPS:
                        self._release_ready_step = step
                elif self._release_open_progress == 0 and not release_ready:
                    # Readiness is a continuous physical precondition until
                    # the first opening command, not a stale one-shot latch.
                    # If the supported handle starts moving during the safety
                    # dwell, resume closed-hand support and acquire a fresh
                    # quiet window before release.
                    self._release_ready_step = None
                    self._release_ready_streak = 0
                if (
                    self._release_ready_step is None
                    or step - self._release_ready_step
                    < _LEFT_RELEASE_OPEN_DWELL_STEPS
                ):
                    left_hand[:] = release_grasp_target
                else:
                    phase = 11
                    # Readiness confirms the tool is already quietly supported
                    # in the zone.  From the first opening frame onward the
                    # release must be purely physical: an anchor-following
                    # support wrench would otherwise drag the tool along with
                    # the withdrawing palm while semantic ownership is still
                    # latched for lineage auditing.
                    self.raw_env._as019_placement_assist_enabled = False
                    self.raw_env._as019_release_support_enabled = False
                    contact_clear = bool(
                        metrics.get("contact_query_valid", False)
                        and metrics.get("left_clear", False)
                        and not left_contact
                    )
                    preclear_started = bool(
                        self._release_preclear_start_palm is not None
                        and self._release_preclear_progress
                        < _LEFT_RELEASE_PRECLEAR_RETREAT_STEPS
                    )
                    preclear_retracting = bool(
                        not self._release_contact_clear_latched
                        and self._release_preclear_progress
                        < _LEFT_RELEASE_PRECLEAR_RETREAT_STEPS
                        and (contact_clear or preclear_started)
                    )
                    preclear_command: np.ndarray | None = None
                    if preclear_retracting:
                        if self._release_preclear_start_palm is None:
                            self._release_preclear_start_palm = np.asarray(
                                self._placement_hold_palm, dtype=np.float64
                            ).copy()
                            tool_rotation = _quat_wxyz_to_matrix(tool[3:7])
                            tool_axis = tool_rotation[:, 0]
                            radial = (
                                self._release_preclear_start_palm[:3]
                                - np.asarray(tool[:3], dtype=np.float64)
                            )
                            radial -= tool_axis * float(
                                np.dot(radial, tool_axis)
                            )
                            radial_norm = float(np.linalg.norm(radial))
                            if radial_norm <= 1.0e-9:
                                radial = tool_rotation[:, 1].copy()
                                radial_norm = 1.0
                            self._release_preclear_target_palm = (
                                self._release_preclear_start_palm.copy()
                            )
                            self._release_preclear_target_palm[:3] += (
                                _LEFT_RELEASE_PRECLEAR_RETREAT_DISTANCE_M
                                * radial
                                / radial_norm
                            )
                        if self._release_preclear_progress_step != step:
                            self._release_preclear_progress += 1
                            self._release_preclear_progress_step = step
                        assert self._release_preclear_target_palm is not None
                        preclear_command = self._smooth(
                            self._release_preclear_start_palm,
                            self._release_preclear_target_palm,
                            self._release_preclear_progress
                            / _LEFT_RELEASE_PRECLEAR_RETREAT_STEPS,
                        )
                        if (
                            self._release_preclear_progress
                            >= _LEFT_RELEASE_PRECLEAR_RETREAT_STEPS
                        ):
                            self._placement_hold_palm = (
                                self._release_preclear_target_palm.copy()
                            )
                            # This path is entered only after authoritative
                            # physical clearance.  Open in place at the radial
                            # retreat target; resuming the normal longitudinal
                            # unload would sweep the now-straight fingers back
                            # across the handle even though the closed hand had
                            # already separated safely.
                            self._release_open_unload_palm = (
                                self._release_preclear_target_palm.copy()
                            )
                            self._release_hold_palm = (
                                self._release_preclear_target_palm.copy()
                            )
                    release_total_steps = (
                        _LEFT_RELEASE_PREPEEL_STEPS
                        + _LEFT_RELEASE_OPEN_STEPS
                    )
                    prepeel_target = release_grasp_target.copy()
                    prepeel_target[
                        _LEFT_RELEASE_PREPEEL_START_INDEX:
                        _LEFT_RELEASE_PREPEEL_END_INDEX
                    ] = _AUTOMATIC_RECEIVER_RELEASE_TARGET[
                        _LEFT_RELEASE_PREPEEL_START_INDEX:
                        _LEFT_RELEASE_PREPEEL_END_INDEX
                    ]
                    if _LEFT_RELEASE_PREPEEL_STEPS > 0:
                        current_prepeel_progress = float(
                            np.clip(
                                self._release_open_progress
                                / _LEFT_RELEASE_PREPEEL_STEPS,
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        current_prepeel_progress = 1.0
                    current_main_open_progress = float(
                        np.clip(
                            (
                                self._release_open_progress
                                - _LEFT_RELEASE_PREPEEL_STEPS
                            )
                            / _LEFT_RELEASE_OPEN_STEPS,
                            0.0,
                            1.0,
                        )
                    )
                    current_open_command = (
                        (1.0 - current_prepeel_progress)
                        * release_grasp_target
                        + current_prepeel_progress * prepeel_target
                    )
                    if self._release_open_progress >= _LEFT_RELEASE_PREPEEL_STEPS:
                        current_open_command = (
                            (1.0 - current_main_open_progress) * prepeel_target
                            + current_main_open_progress
                            * _AUTOMATIC_RECEIVER_RELEASE_TARGET
                        )
                    current_open_command_max = float(
                        np.max(np.abs(current_open_command))
                    )
                    if (
                        not self._release_contact_clear_latched
                        and contact_clear
                        and current_open_command_max <= _HAND_OPEN_THRESHOLD
                    ):
                        self._release_contact_clear_latched = True
                        self._release_contact_clear_step = step
                    if (
                        (
                            not preclear_retracting
                            or self._release_open_progress > 0
                        )
                        and self._release_progress_step != step
                        and self._release_open_progress < release_total_steps
                    ):
                        # First peel the opposing thumb while the four fingers
                        # still support the table-resting tool.  Then relax the
                        # full grasp as the palm unloads radially.  A transient
                        # first clear frame may occur on the penultimate command;
                        # always finish the remaining open steps so the fingers
                        # cannot rebound around the supported tool.
                        self._release_open_progress = min(
                            release_total_steps,
                            self._release_open_progress + 1,
                        )
                        self._release_progress_step = step
                    main_open_elapsed = max(
                        0,
                        self._release_open_progress
                        - _LEFT_RELEASE_PREPEEL_STEPS,
                    )
                    thumb_open_progress = np.clip(
                        main_open_elapsed
                        / _LEFT_RELEASE_THUMB_CONTACT_UNLOAD_STEPS,
                        0.0,
                        1.0,
                    )
                    thumb_finish_progress = np.clip(
                        (
                            main_open_elapsed
                            - _LEFT_RELEASE_THUMB_CONTACT_UNLOAD_STEPS
                        )
                        / _LEFT_RELEASE_THUMB_FINISH_OPEN_STEPS,
                        0.0,
                        1.0,
                    )
                    finger_open_progress = np.clip(
                        (
                            main_open_elapsed
                            - _LEFT_RELEASE_THUMB_OPEN_STEPS
                        )
                        / _LEFT_RELEASE_FINGER_OPEN_STEPS,
                        0.0,
                        1.0,
                    )
                    simultaneous_open_progress = np.clip(
                        main_open_elapsed / _LEFT_RELEASE_OPEN_STEPS,
                        0.0,
                        1.0,
                    )
                    left_hand[:] = (
                        (1.0 - current_prepeel_progress)
                        * release_grasp_target
                        + current_prepeel_progress * prepeel_target
                    )
                    if main_open_elapsed > 0:
                        left_hand[:] = (
                            (1.0 - simultaneous_open_progress) * prepeel_target
                            + simultaneous_open_progress
                            * _AUTOMATIC_RECEIVER_RELEASE_TARGET
                        )
                    if (
                        self._release_contact_clear_latched
                        and not _LEFT_RELEASE_SIMULTANEOUS_OPEN
                    ):
                        # The straight receiver fingertips point directly at
                        # the handle in the verified side grasp.  Peel them to
                        # their open target on the first physically clear frame
                        # instead of waiting behind the remaining thumb ramp,
                        # which allowed immediate re-contact.
                        peel_elapsed = max(
                            1,
                            step - int(self._release_contact_clear_step or step) + 1,
                        )
                        peel_progress = np.clip(
                            peel_elapsed / _LEFT_RELEASE_FINGER_PEEL_STEPS,
                            0.0,
                            1.0,
                        )
                        left_hand[4:] = (
                            (1.0 - peel_progress)
                            * release_grasp_target[4:]
                            + peel_progress
                            * _AUTOMATIC_RECEIVER_RELEASE_TARGET[4:]
                        )
                    if self._release_open_unload_palm is not None:
                        left = self._smooth(
                            self._placement_hold_palm,
                            self._release_open_unload_palm,
                            thumb_open_progress,
                        )
                    if (
                        self._release_contact_clear_latched
                        and self._release_contact_clear_palm is None
                    ):
                        self._release_contact_clear_palm = np.asarray(
                            left, dtype=np.float64
                        ).copy()
                    # Do not retreat while the hand is still physically
                    # supporting the handle.  Once the first clear frame is
                    # observed, move 15 mm continuously so the opening fingers
                    # cannot rebound around the tool, then hold.  Extending the
                    # motion while contact remained dragged the supported tool
                    # out of its zone instead of letting the fingers settle.
                    if self._release_contact_clear_latched:
                        self._release_retract_ready_streak = (
                            self._release_retract_ready_streak + 1
                        )
                        if (
                            self._release_retract_step is None
                            and self._release_retract_ready_streak
                            >= _LEFT_RELEASE_RETRACT_READY_STEPS
                        ):
                            self._release_retract_step = step
                        if self._release_retract_step is not None:
                            minimum_escape_progress = int(
                                np.ceil(
                                    _LEFT_RELEASE_CLEAR_ESCAPE_DISTANCE_M
                                    / _LEFT_RELEASE_CLEAR_DISTANCE_M
                                    * _LEFT_RELEASE_CONTACT_RETRACT_STEPS
                                )
                            )
                            if (
                                self._release_retract_progress
                                < minimum_escape_progress
                                and self._release_retract_progress_step != step
                            ):
                                self._release_retract_progress = min(
                                    _LEFT_RELEASE_CONTACT_RETRACT_STEPS,
                                    self._release_retract_progress + 1,
                                )
                                self._release_retract_progress_step = step
                            left = self._smooth(
                                self._release_contact_clear_palm
                                if self._release_contact_clear_palm is not None
                                else (
                                    self._release_open_unload_palm
                                    if self._release_open_unload_palm is not None
                                    else self._placement_hold_palm
                                ),
                                self._release_hold_palm,
                                self._release_retract_progress
                                / _LEFT_RELEASE_CONTACT_RETRACT_STEPS,
                            )
                            if self._release_retract_progress > 0:
                                phase = 12
                    if preclear_retracting:
                        # Once the hand is physically clear, finish the same
                        # radial retreat even if opening has already begun.
                        # Mid-open clearance must be sticky: cancelling on a
                        # transient finger brush lets the longitudinal unload
                        # sweep the straightening hand back through the tool.
                        assert preclear_command is not None
                        left = preclear_command
                    self._release_final_palm = np.asarray(
                        left, dtype=np.float64
                    ).copy()
                    self.raw_env._as019_placement_assist_enabled = False
                    self.raw_env._as019_release_support_enabled = False
            elif not continuing_placement:
                left = left_home
        else:
            phase = 13
            right = right_home
            self.raw_env._as019_placement_assist_enabled = False
            self.raw_env._as019_release_support_enabled = False
            plan = self._zone_plan
            complete_step = int(self.raw_env._as019_event_steps[4])
            if (
                plan is not None
                and self._release_hold_palm is not None
                and complete_step >= 0
            ):
                retreat_elapsed = max(0, step - complete_step)
                left = self._smooth(
                    self._release_final_palm
                    if self._release_final_palm is not None
                    else self._release_hold_palm,
                    left_home,
                    retreat_elapsed / _LEFT_RELEASE_RETRACT_STEPS,
                )
            else:
                left = left_home

        # Keep the receiver at home until the giver event.  Moving either a
        # low or overhead pre-stage through the shared workspace perturbs the
        # giver arm before physical ownership is established.  After the event,
        # reuse the v74 collision-free pregrasp after the giver finishes its
        # wrist-stabilized transfer.  Its live-entry-relative endpoint matches
        # the physically successful giver wrist branch without forcing reset
        # variation to one fixed world position.
        if stage <= 1:
            right_tail_step = int(self.raw_env._as019_event_steps[0])
            nominal_approach_start = (
                right_tail_step + _LEFT_RECEIVER_APPROACH_DELAY_STEPS
            )
            right_owner_ready = bool(
                stage == 1
                and "right" in self.raw_env._as019_assist_anchor
                and right_tail_step >= 0
            )
            if right_owner_ready and self._left_stage1_wait_palm is None:
                # Stage-0 prealignment may have moved the receiver far from its
                # reset palm.  Hold the physical stage-entry pose through the
                # short delay instead of snapping back to episode home.
                self._left_stage1_wait_palm = self._physical_palm_pose("left")
            if (
                right_owner_ready
                and self._left_receiver_approach_start_palm is None
                and step >= nominal_approach_start
            ):
                # Latch again at actual motion start.  If a wrapper is attached
                # late, the full approach starts here rather than jumping into
                # the middle of the old event-relative timeline.
                self._left_receiver_approach_start_palm = (
                    self._physical_palm_pose("left")
                )
                self._left_receiver_approach_start_step = step
            approach_start = self._left_receiver_approach_start_step
            approach_end = (
                approach_start + _LEFT_RECEIVER_APPROACH_STEPS
                if approach_start is not None
                else None
            )
            metrics = self.raw_env._as019_last_metrics
            physical_middle_contact = bool(
                right_owner_ready
                and "left" in self.raw_env._as019_assist_anchor
                and metrics.get("contact_query_valid", False)
                and metrics.get("left_middle", False)
                and not metrics.get("tip_left_palm", False)
                and _grasp_command_max(self.raw_env._left_hand)
                >= _HAND_CLOSE_THRESHOLD
            )
            if self._receiver_contact_palm is None and physical_middle_contact:
                self._receiver_contact_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("left")
                )
                self._receiver_contact_step = step
                self._receiver_preload_palm = self._receiver_contact_palm.copy()
                contact_direction = (
                    np.asarray(left_handoff, dtype=np.float64)[:3]
                    - self._receiver_contact_palm[:3]
                )
                contact_distance = float(np.linalg.norm(contact_direction))
                if contact_distance > 1.0e-9:
                    self._receiver_preload_palm[:3] += (
                        _LEFT_RECEIVER_CONTACT_PRELOAD_DISTANCE_M
                        * contact_direction
                        / contact_distance
                    )
            if right_owner_ready:
                # Keep the collision-checked preshape fixed during the entire
                # approach.  Closing begins only after the outside thumb has
                # physically touched the handle, avoiding blind penetration.
                left_hand[:] = _AUTOMATIC_RECEIVER_APPROACH_TARGET
                if (
                    self._left_stage1_wait_palm is not None
                    and approach_start is None
                ):
                    left = self._left_stage1_wait_palm.copy()
            if (
                approach_start is not None
                and self._left_receiver_approach_start_palm is not None
                and step >= approach_start
            ):
                phase = 3
                approach_progress = np.clip(
                    (step - approach_start) / _LEFT_RECEIVER_APPROACH_STEPS,
                    0.0,
                    1.0,
                )
                left = self._smooth(
                    self._left_receiver_approach_start_palm,
                    left_pregrasp,
                    approach_progress,
                )
            if (
                right_owner_ready
                and approach_end is not None
                and step >= approach_end
            ):
                phase = 4
                receiver_elapsed = step - approach_end
                left = self._smooth(
                    left_pregrasp,
                    left_handoff,
                    receiver_elapsed / _LEFT_RECEIVER_DESCENT_STEPS,
                )
                left_hand[:] = _AUTOMATIC_RECEIVER_APPROACH_TARGET
                # Use only validated detailed contacts inside the receiver's
                # wooden-handle role.  Aggregate or distal-link force proxies
                # can stay nonzero across a visual gap and must never trigger
                # thumb-loss recovery or a receiver owner transition.
                receiver_thumb_force = float(
                    metrics.get("left_thumb_middle_force", 0.0)
                )
                receiver_opposing_force = float(
                    metrics.get("left_opposing_middle_force", 0.0)
                )
                receiver_two_sided_pinch = bool(
                    metrics.get("contact_query_valid", False)
                    and metrics.get("left_middle_physical", False)
                )
                receiver_single_opposing_contact = bool(
                    metrics.get("contact_query_valid", False)
                    and not receiver_two_sided_pinch
                    and receiver_thumb_force < _CONTACT_FORCE_N
                    and receiver_opposing_force >= _CONTACT_FORCE_N
                )
                if (
                    self._receiver_thumb_contact_step is not None
                    and self._receiver_contact_palm is None
                    and "left" not in self.raw_env._as019_assist_anchor
                    and receiver_single_opposing_contact
                ):
                    self._receiver_thumb_loss_counter += 1
                else:
                    self._receiver_thumb_loss_counter = 0
                if (
                    self._receiver_thumb_loss_counter
                    >= _LEFT_RECEIVER_THUMB_LOSS_DEBOUNCE_STEPS
                ):
                    # The original thumb touch has cleared while a closing
                    # opposing finger is loaded.  Stop treating that stale
                    # touch as a grasp immediately and reuse the existing
                    # open-before-reseat recovery; continuing the close would
                    # push the screwdriver around the giver contact.
                    self._receiver_retry_thumb_palm = (
                        self._receiver_thumb_contact_palm.copy()
                        if self._receiver_thumb_contact_palm is not None
                        else None
                    )
                    self._receiver_thumb_contact_step = None
                    self._receiver_thumb_contact_palm = None
                    self._receiver_thumb_contact_hand = None
                    self._receiver_single_contact_reseat_step = step
                    self._receiver_single_contact_start_hand = np.asarray(
                        self.raw_env._left_hand, dtype=np.float64
                    ).reshape(-1).copy()
                    self._receiver_single_contact_hold_palm = (
                        _matrix_pose_vector(
                            self.raw_env._palm_matrix("left")
                        )
                    )
                    if self._receiver_retry_right_hold_palm is None:
                        self._receiver_retry_right_hold_palm = (
                            _matrix_pose_vector(
                                self.raw_env._palm_matrix("right")
                            )
                        )
                    self._receiver_thumb_loss_counter = 0
                failed_close_elapsed = (
                    _AUTOMATIC_RECEIVER_CLOSE_STEPS
                    + _AUTOMATIC_RECEIVER_FALLBACK_DELAY_STEPS
                    + _AUTOMATIC_RECEIVER_FALLBACK_CLOSE_STEPS
                )
                if (
                    self._receiver_thumb_contact_step is not None
                    and step - self._receiver_thumb_contact_step
                    >= failed_close_elapsed
                    and self._receiver_contact_palm is None
                    and "left" not in self.raw_env._as019_assist_anchor
                    and not receiver_two_sided_pinch
                    and receiver_thumb_force < _CONTACT_FORCE_N
                ):
                    # A transient thumb brush is not a permanent grasp plan.
                    # If both nominal and bounded fallback closure finish
                    # without a live two-sided pinch, release that stale palm
                    # latch and let the one-sided recovery below unload the
                    # opposing finger before trying again.
                    self._receiver_retry_thumb_palm = (
                        self._receiver_thumb_contact_palm.copy()
                        if self._receiver_thumb_contact_palm is not None
                        else None
                    )
                    self._receiver_thumb_contact_step = None
                    self._receiver_thumb_contact_palm = None
                    self._receiver_thumb_contact_hand = None
                    self._receiver_single_contact_reseat_step = step
                    self._receiver_single_contact_start_hand = np.asarray(
                        self.raw_env._left_hand, dtype=np.float64
                    ).reshape(-1).copy()
                    self._receiver_single_contact_hold_palm = (
                        _matrix_pose_vector(
                            self.raw_env._palm_matrix("left")
                        )
                    )
                    if self._receiver_retry_right_hold_palm is None:
                        self._receiver_retry_right_hold_palm = (
                            _matrix_pose_vector(
                                self.raw_env._palm_matrix("right")
                            )
                        )
                if (
                    self._receiver_single_contact_reseat_step is None
                    and receiver_single_opposing_contact
                    and self._receiver_thumb_contact_step is None
                    and self._receiver_contact_palm is None
                    and "left" not in self.raw_env._as019_assist_anchor
                ):
                    self._receiver_retry_thumb_palm = None
                    self._receiver_single_contact_reseat_step = step
                    self._receiver_single_contact_start_hand = np.asarray(
                        self.raw_env._left_hand, dtype=np.float64
                    ).reshape(-1).copy()
                    self._receiver_single_contact_hold_palm = (
                        _matrix_pose_vector(
                            self.raw_env._palm_matrix("left")
                        )
                    )
                    if self._receiver_retry_right_hold_palm is None:
                        self._receiver_retry_right_hold_palm = (
                            _matrix_pose_vector(
                                self.raw_env._palm_matrix("right")
                            )
                        )
                single_contact_open_progress: float | None = None
                retry_thumb_return_progress: float | None = None
                if self._receiver_single_contact_reseat_step is not None:
                    # Unload the one-sided finger first.  Starting from the
                    # actual failed-close command avoids jumping back through
                    # the wide approach preshape while the finger is loaded.
                    single_contact_open_progress = float(
                        np.clip(
                            (
                                step
                                - self._receiver_single_contact_reseat_step
                                + 1
                            )
                            / _LEFT_RESEAT_SINGLE_CONTACT_OPEN_STEPS,
                            0.0,
                            1.0,
                        )
                    )
                    single_contact_start_hand = (
                        self._receiver_single_contact_start_hand
                        if self._receiver_single_contact_start_hand is not None
                        else _AUTOMATIC_RECEIVER_APPROACH_TARGET
                    )
                    left_hand[16:20] = (
                        1.0 - single_contact_open_progress
                    ) * single_contact_start_hand[16:20]
                    if (
                        single_contact_open_progress >= 1.0
                        and self._receiver_retry_thumb_palm is not None
                    ):
                        retry_thumb_return_progress = float(
                            np.clip(
                                (
                                    step
                                    - self._receiver_single_contact_reseat_step
                                    - _LEFT_RESEAT_SINGLE_CONTACT_OPEN_STEPS
                                    + 1
                                )
                                / _LEFT_RESEAT_RETRY_RETURN_STEPS,
                                0.0,
                                1.0,
                            )
                        )
                if (
                    receiver_elapsed
                    >= _LEFT_RECEIVER_DESCENT_STEPS
                    + _LEFT_RESEAT_DELAY_STEPS
                    and self._receiver_thumb_contact_step is None
                    and self._receiver_contact_palm is None
                    and "left" not in self.raw_env._as019_assist_anchor
                    and not receiver_two_sided_pinch
                ):
                    # A fixed tool-relative command cannot remove the DLS
                    # endpoint residual because feeding that command through
                    # the controller reproduces the same residual.  Command
                    # the measured palm plus its bounded local error instead.
                    # Keep the measured wrist so translation does not select
                    # another IK orientation branch near the handle.  A
                    # one-sided finger brush is deliberately allowed here:
                    # only a real opposed two-finger pinch should stop reseating.
                    current_tool = self.raw_env._tool_matrix()
                    current_palm = _matrix_pose_vector(
                        self.raw_env._palm_matrix("left")
                    )
                    current_local = (
                        np.linalg.inv(current_tool)
                        @ _pose_vector_to_matrix(current_palm)
                    )[:3, 3]
                    correction_world = current_tool[:3, :3] @ (
                        _LEFT_RESEAT_TARGET_LOCAL - current_local
                    )
                    correction_world *= _LEFT_RESEAT_COMMAND_FEEDBACK_GAIN
                    correction_norm = float(np.linalg.norm(correction_world))
                    if correction_norm > _LEFT_RESEAT_MAX_CORRECTION_M:
                        correction_world *= (
                            _LEFT_RESEAT_MAX_CORRECTION_M / correction_norm
                        )
                    left = current_palm
                    if (
                        single_contact_open_progress is not None
                        and single_contact_open_progress < 1.0
                        and self._receiver_single_contact_hold_palm is not None
                    ):
                        # Freeze the measured receiver wrist while the loaded
                        # opposing finger peels away.  Re-sampling the lagging
                        # DLS endpoint every frame lets the wrist drift toward
                        # the giver and can rotate the singly-held tool even
                        # after the finger's force view has cleared.
                        left = self._receiver_single_contact_hold_palm.copy()
                    elif (
                        retry_thumb_return_progress is not None
                        and self._receiver_single_contact_hold_palm is not None
                        and self._receiver_retry_thumb_palm is not None
                    ):
                        # Return along the short, collision-observed path to
                        # this episode's first real thumb contact.  A generic
                        # tool-local reseat target can lie on an unreachable
                        # DLS branch after the giver has moved; the cached pose
                        # is both reachable and already penetration-checked by
                        # the authoritative contact view.
                        left = self._smooth(
                            self._receiver_single_contact_hold_palm,
                            self._receiver_retry_thumb_palm,
                            retry_thumb_return_progress,
                        )
                    elif (
                        (
                            single_contact_open_progress is None
                            or single_contact_open_progress >= 1.0
                        )
                        and float(metrics.get("outward_alignment", -1.0))
                        >= _LEFT_OWNER_AXIS_ALIGNMENT_ACTIVATION_COS
                    ):
                        left[:3] += correction_world
            if (
                right_owner_ready
                and self._receiver_thumb_contact_step is None
            ):
                if (
                    bool(metrics.get("contact_query_valid", False))
                    and float(metrics.get("left_thumb_middle_force", 0.0))
                    >= _CONTACT_FORCE_N
                ):
                    self._receiver_thumb_contact_step = step
                    self._receiver_thumb_contact_palm = _matrix_pose_vector(
                        self.raw_env._palm_matrix("left")
                    )
                    self._receiver_thumb_contact_hand = left_hand.copy()
                    self._receiver_thumb_loss_counter = 0
                    self._receiver_single_contact_reseat_step = None
                    self._receiver_single_contact_start_hand = None
                    self._receiver_single_contact_hold_palm = None
                    self._receiver_retry_thumb_palm = None
            if (
                right_owner_ready
                and self._receiver_thumb_contact_step is not None
                and self._receiver_thumb_contact_palm is not None
            ):
                phase = 4
                thumb_preload_palm = self._receiver_thumb_contact_palm.copy()
                thumb_preload_direction = (
                    np.asarray(left_handoff, dtype=np.float64)[:3]
                    - self._receiver_thumb_contact_palm[:3]
                )
                thumb_preload_distance = float(
                    np.linalg.norm(thumb_preload_direction)
                )
                if thumb_preload_distance > 1.0e-9:
                    thumb_preload_palm[:3] += (
                        _LEFT_RECEIVER_CONTACT_PRELOAD_DISTANCE_M
                        * thumb_preload_direction
                        / thumb_preload_distance
                    )
                thumb_preload_progress = np.clip(
                    (
                        step - self._receiver_thumb_contact_step + 1
                    )
                    / _LEFT_RECEIVER_CONTACT_PRELOAD_STEPS,
                    0.0,
                    1.0,
                )
                # Keep the first live thumb contact seated while the opposing
                # fingers finish their measured close.  This is the same 3 mm
                # collision-checked preload already used after dual contact,
                # applied to the robot palm only over eight control frames.
                left = self._smooth(
                    self._receiver_thumb_contact_palm,
                    thumb_preload_palm,
                    thumb_preload_progress,
                )
                close_progress = np.clip(
                    (
                        step - self._receiver_thumb_contact_step + 1
                    ) / _AUTOMATIC_RECEIVER_CLOSE_STEPS,
                    0.0,
                    1.0,
                )
                receiver_close_start_hand = (
                    self._receiver_thumb_contact_hand
                    if self._receiver_thumb_contact_hand is not None
                    else _AUTOMATIC_RECEIVER_APPROACH_TARGET
                )
                left_hand[:] = (
                    (1.0 - close_progress)
                    * receiver_close_start_hand
                    + close_progress * _AUTOMATIC_RECEIVER_GRASP_TARGET
                )
                if (
                    close_progress >= 1.0
                    and self._receiver_contact_palm is None
                ):
                    fallback_progress = np.clip(
                        (
                            step
                            - self._receiver_thumb_contact_step
                            - _AUTOMATIC_RECEIVER_CLOSE_STEPS
                            - _AUTOMATIC_RECEIVER_FALLBACK_DELAY_STEPS
                            + 1
                        )
                        / _AUTOMATIC_RECEIVER_FALLBACK_CLOSE_STEPS,
                        0.0,
                        1.0,
                    )
                    left_hand[:] = (
                        (1.0 - fallback_progress)
                        * _AUTOMATIC_RECEIVER_GRASP_TARGET
                        + fallback_progress
                        * _AUTOMATIC_RECEIVER_FALLBACK_GRASP_TARGET
                    )
            if self._receiver_contact_palm is not None:
                phase = 4
                # Ownership is admitted only for a simultaneous live pinch;
                # the substep hook freezes that measured hand target until the
                # giver has cleared.
                pass
            left, left_hand, _stage1_regrasp_handled = (
                self._stage3_regrasp_command(
                    step=step,
                    tool=tool,
                    metrics=self.raw_env._as019_last_metrics,
                    left=left,
                    left_hand=left_hand,
                )
            )
        if (
            stage == 0
            and self._left_stage0_prealign_tool is None
            and self._right_acquired_step is not None
            and right_tail_physical
            and step - self._right_acquired_step
            >= _LEFT_STAGE0_PREALIGN_DELAY_STEPS
            and self.raw_env._outward_alignment() < _OUTWARD_ENTRY_COS
        ):
            self._left_stage0_prealign_tool = np.asarray(
                tool, dtype=np.float64
            ).copy()
            self._left_stage0_prealign_start_step = step
            self._left_stage0_prealign_start_palm = (
                self._physical_palm_pose("left")
            )
        if (
            stage == 0
            and self._left_stage0_prealign_tool is not None
            and self._left_stage0_prealign_start_step is not None
            and self._left_stage0_prealign_start_palm is not None
        ):
            # Keep the open hand on a reset-relative, collision-bounded path.
            # The first physical finger touch can align an 8--12 degree reset
            # without closing around the sharp end or moving either root pose.
            phase = 14
            low = _tool_frame_pose(
                self._left_stage0_prealign_tool,
                _LEFT_STAGE0_PREALIGN_LOCAL,
                _LEFT_RECEIVER_PREGRASP_LOCAL_QUAT,
            )
            high = low.copy()
            high[2] += _LEFT_STAGE0_PREALIGN_WORLD_LIFT_M
            pushed_local = _LEFT_STAGE0_PREALIGN_LOCAL.copy()
            pushed_local[1] += _LEFT_STAGE0_PREALIGN_PUSH_LOCAL_Y_M
            pushed = _tool_frame_pose(
                self._left_stage0_prealign_tool,
                pushed_local,
                receiver_local_quat,
            )
            elapsed = step - self._left_stage0_prealign_start_step
            if elapsed < _LEFT_STAGE0_PREALIGN_APPROACH_STEPS:
                left = self._smooth(
                    self._left_stage0_prealign_start_palm,
                    high,
                    elapsed
                    / max(_LEFT_STAGE0_PREALIGN_APPROACH_STEPS - 1, 1),
                )
            elif elapsed < (
                _LEFT_STAGE0_PREALIGN_APPROACH_STEPS
                + _LEFT_STAGE0_PREALIGN_DESCENT_STEPS
            ):
                left = self._smooth(
                    high,
                    low,
                    (
                        elapsed - _LEFT_STAGE0_PREALIGN_APPROACH_STEPS
                    )
                    / max(_LEFT_STAGE0_PREALIGN_DESCENT_STEPS - 1, 1),
                )
            else:
                left = self._smooth(
                    low,
                    pushed,
                    (
                        elapsed
                        - _LEFT_STAGE0_PREALIGN_APPROACH_STEPS
                        - _LEFT_STAGE0_PREALIGN_DESCENT_STEPS
                    )
                    / max(_LEFT_STAGE0_PREALIGN_PUSH_STEPS - 1, 1),
                )
            left_hand.fill(0.0)
        if stage == 1 and self._receiver_retry_right_hold_palm is not None:
            # Receiver recovery removes the only opposing support from a
            # marginal cylindrical giver grasp.  Freeze the giver at its live
            # measured pose instead of continuing to pull a roughly 2 mm DLS
            # residual while the receiver unloads.  Stage 2 independently
            # re-latches the authoritative giver pose on dual contact.
            right = self._receiver_retry_right_hold_palm.copy()
        # Once the giver has reached its measured handover entry pose, the
        # contact-owned bounded wrench is the stable safety-axis controller.
        # Re-enabling wrist DLS feedback during a receiver retry amplified the
        # retreat transient even after the opposing finger had cleared.
        right_axis_feedback_active = bool(
            stage <= 1
            and right_owner
            and not (
                stage == 0
                and right_owner
                and self._right_acquired_palm is not None
            )
            and not (
                stage == 1
                and self._right_entry_palm is not None
            )
        )
        if right_axis_feedback_active:
            current_tool = self.raw_env._tool_matrix()
            right[3:7] = _matrix_to_quat_wxyz(
                _bounded_axis_feedback_rotation(
                    self.raw_env._as019_outward_reference_rotation[:, 0],
                    current_tool[:3, 0],
                    _quat_wxyz_to_matrix(right[3:7]),
                    gain=_RIGHT_AXIS_COMMAND_FEEDBACK_GAIN,
                    max_angle_rad=_RIGHT_AXIS_COMMAND_FEEDBACK_MAX_RAD,
                )
            )
        if (
            stage == 0
            and self._right_acquire_arrived_step is not None
            and step > self._right_acquire_arrived_step
            and int(self.raw_env._as019_event_steps[0]) < 0
            and not bool(
                self.raw_env._as019_last_metrics.get(
                    "right_tail_physical", False
                )
            )
        ):
            assert self._right_approach_tool is not None
            frozen_tool = _pose_vector_to_matrix(self._right_approach_tool)
            current_palm = self.raw_env._palm_matrix("right")
            current_local = (
                np.linalg.inv(frozen_tool)
                @ current_palm
            )[:3, 3]
            correction_world = frozen_tool[:3, :3] @ (
                _RIGHT_RESEAT_TARGET_LOCAL - current_local
            )
            correction_world *= _RIGHT_RESEAT_COMMAND_FEEDBACK_GAIN
            correction_norm = float(np.linalg.norm(correction_world))
            if correction_norm > _RIGHT_RESEAT_MAX_CORRECTION_M:
                correction_world *= (
                    _RIGHT_RESEAT_MAX_CORRECTION_M / correction_norm
                )
            # The complete pick->acquire path above remains tied to the reset
            # tool pose through its endpoint.  Only from endpoint+1 may this
            # bounded terminal correction compensate the measured DLS residual.
            # Preserve the calibrated wrist orientation until real contact.
            right[:3] = current_palm[:3, 3] + correction_world
        left_owner = "left" in self.raw_env._as019_assist_anchor
        left_owner_contact = bool(
            left_owner
            and self.raw_env._as019_last_metrics.get(
                "contact_query_valid", False
            )
            and self.raw_env._as019_last_metrics.get(
                "left_middle_physical", False
            )
            and _grasp_command_max(self.raw_env._left_hand)
            >= _HAND_CLOSE_THRESHOLD
        )
        left_axis_feedback_active = bool(
            stage in (2, 3)
            or (stage == 4 and self._release_open_progress == 0)
        ) and not (
            stage == 2
            and self._receiver_thumb_contact_palm is not None
            and float(
                self.raw_env._as019_last_metrics.get("right_force", 0.0)
            )
            >= _CONTACT_FORCE_N
        ) and not (
            stage == 4 and self._release_quiet_hold_palm is not None
        )
        if left_owner_contact and left_axis_feedback_active:
            current_tool = self.raw_env._tool_matrix()
            left[3:7] = _matrix_to_quat_wxyz(
                _bounded_axis_feedback_rotation(
                    self.raw_env._as019_outward_reference_rotation[:, 0],
                    current_tool[:3, 0],
                    _quat_wxyz_to_matrix(left[3:7]),
                    gain=(
                        _LEFT_AXIS_COMMAND_FEEDBACK_STAGE4_GAIN
                        if stage == 4
                        else _LEFT_AXIS_COMMAND_FEEDBACK_GAIN
                    ),
                    max_angle_rad=_LEFT_AXIS_COMMAND_FEEDBACK_MAX_RAD,
                )
            )
        if (
            left_owner_contact
            and stage in (2, 3)
        ):
            metrics = self.raw_env._as019_last_metrics
            physical_left_contact = bool(
                metrics.get("contact_query_valid", False)
                and metrics.get("left_middle_physical", False)
                and _grasp_command_max(self.raw_env._left_hand)
                >= _HAND_CLOSE_THRESHOLD
            )
            payload = self.raw_env._as019_assist_anchor.get("left")
            if (
                physical_left_contact
                and payload is not None
                and float(metrics.get("outward_alignment", -1.0))
                < _LEFT_OWNER_AXIS_ALIGNMENT_ACTIVATION_COS
            ):
                live_tool = np.asarray(tool, dtype=np.float64).copy()
                aligned_tool = live_tool.copy()
                aligned_tool[3:7] = _matrix_to_quat_wxyz(
                    _minimal_axis_aligned_rotation(
                        self.raw_env._as019_outward_reference_rotation[:, 0],
                        _quat_wxyz_to_matrix(live_tool[3:7]),
                    )
                )
                target_palm = _owner_palm_pose(
                    aligned_tool, payload["hand_to_tool"]
                )
                measured_palm = _matrix_pose_vector(
                    self.raw_env._palm_matrix("left")
                )
                corrected_palm = self._smooth(
                    measured_palm,
                    target_palm,
                    (
                        (
                            _LEFT_OWNER_AXIS_ALIGNMENT_STAGE3_GAIN
                            if (
                                not self.raw_env._as019_receiver_pinch_freeze
                                or bool(
                                    self.raw_env._as019_receiver_pinch_freeze.get(
                                        "normalization_applied", False
                                    )
                                )
                            )
                            else _LEFT_OWNER_AXIS_ALIGNMENT_DISTINCT_STAGE3_GAIN
                        )
                        if stage == 3
                        else (
                            _LEFT_OWNER_AXIS_ALIGNMENT_GAIN
                            if (
                                not self.raw_env._as019_receiver_pinch_freeze
                                or bool(
                                    self.raw_env._as019_receiver_pinch_freeze.get(
                                        "normalization_applied", False
                                    )
                                )
                            )
                            else _LEFT_OWNER_AXIS_ALIGNMENT_DISTINCT_GAIN
                        )
                    ),
                )
                if stage == 2 or self.raw_env._as019_placement_assist_enabled:
                    # Preserve receiver preload and active placement
                    # translations; only the safety-axis orientation is fed
                    # back into those motion branches.
                    left[3:7] = corrected_palm[3:7]
                else:
                    # Before placement, the stage-3 anchor is the measured
                    # seated grasp and its full contact-owned pose is safe.
                    left = corrected_palm
        if (
            stage == 3
            and self.raw_env._as019_placement_motion_started
            and self.raw_env._as019_placement_assist_enabled
        ):
            # Compensate only the measured Cartesian endpoint residual of the
            # low-table DLS branch.  Half-gain keeps the physical tool and the
            # time-parameterized lateral hand path synchronized while still
            # overcoming the residual that otherwise stalls a weak but valid
            # pinch.  Taper that gain only when measured finger force rises,
            # so a loaded grasp is not driven across the safety boundary.  The
            # correction remains capped at 12 mm and never writes object state.
            left = self._placement_tracking_command(
                left,
                _matrix_pose_vector(self.raw_env._palm_matrix("left")),
                contact_force_n=float(
                    self.raw_env._as019_last_metrics.get("left_force", 0.0)
                ),
            )
        # The calibrated thin-handle grasp intentionally includes thumb/index
        # abduction.  Do not overwrite those values after the phase-specific
        # close/regrasp/release commands have been constructed.
        self.raw_env._as019_automatic_phase = phase
        action = np.concatenate([right, left, right_hand, left_hand]).astype(
            np.float32
        )
        self.raw_env._as019_last_policy_action = action.copy()
        self.raw_env._as019_last_policy_action_step = step
        return action

    def step(self, _action):
        return self.env.step(self.automatic_action())
