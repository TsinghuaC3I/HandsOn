"""Build a stable, deterministic physical origin before teleoperation starts."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from handson.data.recording_origin_profile import (
    is_abc_style_collection_task,
)
from handson.runtime.gpu import (
    is_canonical_cuda_device,
    is_physx_recording_device_allowed,
)

from .lw_native_articulation_scene import LW_TASK_NAMES, LW_TASK_SPECS
from .physx_ground_truth import PhysxGroundTruthTrace


STABLE_ORIGIN_ENV = "HANDSON_STABLE_RECORDING_ORIGIN"
JOINT_VELOCITY_LIMIT_ENV = "HANDSON_STABLE_JOINT_VELOCITY_LIMIT"
STABLE_ORIGIN_RESET_ATTEMPTS_ENV = "HANDSON_STABLE_ORIGIN_RESET_ATTEMPTS"
DEFAULT_LINEAR_VELOCITY_LIMIT = 0.001
DEFAULT_ANGULAR_VELOCITY_LIMIT = 0.002
DEFAULT_JOINT_VELOCITY_LIMIT = 0.001
DEFAULT_STABLE_SECONDS = 0.5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_STABILITY_SAMPLE_SECONDS = 0.05
DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT = 0.005
DEFAULT_RESET_ROTATION_DRIFT_LIMIT = float(np.deg2rad(3.0))
DEFAULT_CONTACT_PENETRATION_LIMIT = 0.0005
_COLLECTION_INITIAL_SETTLE_SECONDS = 1.0
_COLLECTION_POST_SETTLE_TIMEOUT_SECONDS = 3.0
_ABC_COLLECTION_POST_SETTLE_TIMEOUT_SECONDS = 39.0
_TASK_POST_SETTLE_TIMEOUT_SECONDS = {
    # Reject and resample a persistent utensil-contact correction promptly.
    # This reduces only a failed candidate's time budget; all velocity, pose,
    # penetration and 60-frame continuous-stability requirements stay strict.
    "t28_place_tableware": 5.0,
    # These rigid-only YCB tasks use a wakeable sleep boundary after their
    # natural settle. Three seconds is ample for the mandatory 0.5-second
    # every-frame proof and prevents a bad contact sample from holding the
    # operator behind the gate for the generic 39-second ABC budget.
    "t27_place_largest_fitting_ball": 3.0,
    "t41_sort_nested_cups": 3.0,
    "t17_open_latched_door": 15.0,
    "p004_hold_spring_clamp_open_insert_card": 15.0,
    "p005_key_insert_turn_unlock_open_shackle": 15.0,
    "p006_side_release_buckle_align_press": 15.0,
    "bench320_bc007_bimanual_long_rod_yaw_reorientation": 15.0,
    "bench320_hs014_bottle_stabilized_cap_opening": 15.0,
    # D06-D19 use dense tabletop layouts with several rounded or compound
    # contacts.  Like the previously audited A/B/C collection tasks, let the
    # strict, write-free gate keep settling instead of rejecting a healthy
    # randomized sample after the generic three-second budget.  The gate still
    # exits as soon as it proves 0.5 continuous seconds of stability.
    "t42_sort_objects": 8.0,
    "d07_organize_office_desk": 8.0,
    "d08_toast_bread": 8.0,
    "t31_prepare_coffee": 8.0,
    "d10_load_dishwasher": 8.0,
    "d11_presoak_pan": 8.0,
    "d12_prepare_cutting_station": 8.0,
    "d13_prepare_tea_tray": 8.0,
    "d14_wash_fruit": 8.0,
    "d15_pack_matching_lunches": 8.0,
    "d16_sweep_blocks_to_bin": 8.0,
    "d17_store_laptop_headphones": 8.0,
    "d18_pack_eggs": 8.0,
    "d19_memory_cover_sort": 8.0,
    "d20_microwave_hot_dog_plating": 8.0,
    "d20_1_load_and_start": 8.0,
    "d20_2_retrieve_and_plate": 8.0,
    "d20_3_finish_table_setting": 8.0,
}
_TASK_STABLE_SECONDS = {
    # P0-02 qualifies the complete dual-arm home and the articulated door
    # before teleoperation. The actual-state gate rejects a wrong authored
    # home; this full ten-second zero-action window proves that the accepted
    # state remains stable in the production recorder path.
    "t17_open_latched_door": 10.0,
    # These merged benchmark tasks retain a complete awake 1200-frame
    # priority-zero proof in their dedicated validation scripts. P0-04,
    # P0-05, and P0-06 deliberately use the normal 0.5-second online gate
    # below: the operator UI no longer appears initialized with both hands and
    # the viewport held idle for another ten seconds after the one-second
    # physical settle.
    "p004_hold_spring_clamp_open_insert_card": DEFAULT_STABLE_SECONDS,
    "p005_key_insert_turn_unlock_open_shackle": DEFAULT_STABLE_SECONDS,
    "p006_side_release_buckle_align_press": DEFAULT_STABLE_SECONDS,
    "bench320_bc007_bimanual_long_rod_yaw_reorientation": 10.0,
    # HS014 now has an engaged PhysX cap-thread joint and a strict contact
    # audit, so it does not need the historical ten-second idle proof. Match
    # the normal collection gate to avoid a long pause before action 0.
    "bench320_hs014_bottle_stabilized_cap_opening": DEFAULT_STABLE_SECONDS,
}
_COLLECTION_RESET_ATTEMPTS = 3
VELOCITY_CLAMP_SCOPE_NONE = "none"
VELOCITY_CLAMP_SCOPE_GATE_ONLY = "gate_only"
VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS = (
    "all_initialization_steps"
)
_VELOCITY_CLAMP_SCOPES = {
    VELOCITY_CLAMP_SCOPE_NONE,
    VELOCITY_CLAMP_SCOPE_GATE_ONLY,
    VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS,
}
_TASK_RESET_ATTEMPTS = {
    "t28_place_tableware": 8,
    # A01 has five independently settling tabletop bodies, including rounded
    # and compound collision shapes.  A rejected sample is harmless, whereas
    # exhausting the generic three attempts tears down the collection window.
    "t26_place_fruit_in_bucket": 12,
    "eval_comb04_place_fruit_and_photograph_logo": 12,
}
# Sponge 3's randomized support pose needs a four-second natural contact
# settling window before the strict velocity gate. The gate still remains
# awake and write-free; this only prevents action[0] from starting while the
# deformable is shedding its reset-contact correction.
_TASK_INITIAL_SETTLE_SECONDS = {
    "eval_flex01_compress_sponge_into_mug": 4.0,
}
_NATIVE_COLLECTION_TASKS = frozenset(
    {
        "eval_flex01_compress_sponge_into_mug",
    }
)
_TASK_LINEAR_VELOCITY_LIMITS = {
    # The native volume sponge retains a small, repeatable surface-node
    # correction velocity on the table (three seeded 120 Hz traces measured
    # 0.0065--0.0079 m/s after the ordinary settle/zero boundary).  Rigid mug
    # motion remained <=0.00034 m/s and is additionally protected by the
    # unchanged 5 mm reset-drift/contact gates.  Keep this allowance local to
    # YCB11 so the deformable can enter recording without weakening any other
    # task's origin contract.
    "eval_flex01_compress_sponge_into_mug": 0.012,
}
_TASK_JOINT_VELOCITY_LIMITS = {
    # Live Wuji hand packets continue arriving while the UI enters the
    # recording-origin gate.  These D tasks hold the arm pose and allow the
    # measured sub-0.02 rad/s hand settling residual; task-body velocity,
    # pose-drift, and contact gates remain unchanged.
    "d13_prepare_tea_tray": 0.02,
    "d15_pack_matching_lunches": 0.02,
    "d17_store_laptop_headphones": 0.02,
    # B02 intentionally uses the source MuJoCo soft-hand drives so the small
    # nut remains graspable.  PhysX leaves a sub-half-degree/second joint
    # residual while the fixed bolt and free nut are already below the strict
    # rigid-body limits.  Keep this narrow exception local to B02.
    "b02_nut_bolt_assembly": 0.008,
    # The left thumb remains in light reset contact with the camera.  The
    # measured PhysX solver jitter is about 0.0026 rad/s, while the rest of the
    # scene is settled.  This explicit near-zero limit was validated by strict
    # fresh-process replay rather than chosen to bypass the stability gate.
    "t23_photograph_logo": 0.003,
}
_TASK_ANGULAR_VELOCITY_LIMITS: dict[str, float] = {
    # YCB11's rigid mug remains nearly still while the deformable sponge
    # carries a small solver correction during its natural settle.
    "eval_flex01_compress_sponge_into_mug": 0.010,
}
_TASK_RESET_ROTATION_DRIFT_LIMITS: dict[str, float] = {}
# These legacy operator-controlled tasks already reset the two ROKAE arms to
# their authored home before teleoperation.  Their first live PhysX frame can
# therefore contain the arm's commanded repositioning (the task itself is
# still stationary).  Keep the ordinary velocity/contact/final-zero origin
# checks, but restore the previous behavior that did not reject this expected
# robot-only reset motion as a failed task-origin sample.
_TASKS_ALLOW_ROBOT_RESET_REPOSITION = frozenset(
    {
        "eval_comb05_scrub_bottle_then_fill_cup",
        "t05_fill_cup",
        "t46_store_charger",
    }
)
_TASK_POST_SETTLE_SLEEPING_OVERRIDES = {
    # Priority-zero must exercise ordinary awake support contacts. A sleeping
    # object can appear perfectly stable while hiding a floating or invalid
    # support pose. P0-04, P0-05, and P0-06 stay awake through their responsive
    # online gates and through dedicated ten-second validators; the other
    # merged tasks keep their ten-second production-origin gates.
    "p004_hold_spring_clamp_open_insert_card": False,
    "p005_key_insert_turn_unlock_open_shackle": False,
    "p006_side_release_buckle_align_press": False,
    # Calibrated utensils settle on their original table/riser contacts, but
    # GPU TGS can keep regenerating non-integrated correction velocities.
    # Use the same wakeable post-settle boundary as YCB02/YCB04. This is a
    # one-time reset operation; all props stay dynamic and hand contact wakes
    # them. The complete natural landing history still passes drift/contact
    # gates, followed by the unchanged 60-frame stable-pose proof.
    "t28_place_tableware": True,
    "t39_set_timer": False,
    # YCB08's three imported rigid meshes can retain a small contact-axis
    # correction after their poses have stopped drifting.  Establish a
    # wakeable sleep boundary after the unchanged natural landing interval;
    # hand contact wakes each body normally during collection.
    "t07_flip_sponge": True,
    # D20's thin fork/dishes can retain repeatable contact-axis rocking after
    # reset. Put them at the same one-time wakeable sleep boundary before the
    # strict origin gate; hand contact still wakes them during collection.
    "d20_microwave_hot_dog_plating": True,
    "d20_1_load_and_start": True,
    "d20_1_5_set_controls": True,
    "d20_2_retrieve_and_plate": True,
    "d20_3_finish_table_setting": True,
    "eval_flex01_compress_sponge_into_mug": False,
    "bench320_bc007_bimanual_long_rod_yaw_reorientation": False,
    "bench320_hs014_bottle_stabilized_cap_opening": False,
    # BC005 carries two small round payloads on a dynamic wooden tray.  GPU
    # TGS can keep reporting a non-integrated rolling correction after their
    # poses have settled.  Establish a wakeable pre-action sleep boundary;
    # the first physical hand/tray contact wakes the bodies normally, while
    # reset drift and contact penetration remain under the common strict gate.
    "loaded_tray_through_narrow_gate": True,
    # YCB02 contains only seven independent rigid task bodies. Some imported
    # round/compound colliders continue making micrometre-scale pose
    # corrections even while their public velocities are clamped to zero,
    # which can consume the complete 39-second gate and force another hard
    # scene rebuild. After the unchanged one-second natural landing period,
    # put the objects at a wakeable sleep boundary. Physical hand contact
    # wakes them normally; reset drift, penetration, and the following
    # every-frame half-second pose proof remain strict.
    "t27_place_largest_fitting_ball": True,
    # YCB04 is likewise a rigid-only task. Sleeping the nested cups after
    # their natural contact settle removes the persistent nested-contact
    # correction without freezing episode dynamics: contact from either hand
    # wakes the affected stack before manipulation.
    "t41_sort_nested_cups": True,
    # BC011 keeps the downloaded pen's original cylindrical convex collider.
    # After its first contact frame and velocity-only zero,
    # put that one rigid body into PhysX's wakeable sleep state so the contact
    # solver cannot regenerate rolling before teleoperation.  A hand contact
    # wakes the pen normally; no velocity is clamped during the task.
    "t20_pass_and_orient_pen": True,
    # HS005's upright bottle uses an analytic cylindrical collider on the flat
    # sink support. GPU TGS can keep regenerating a small rolling correction
    # after the one-second natural landing interval, even though the reset
    # drift/contact gates remain valid. Establish the same wakeable pre-action
    # sleep boundary proven for BC011's cylindrical pen: task bodies wake on
    # physical hand contact, and no sleep/pose/velocity write occurs after
    # action[0].
    "t37_scrub_bottle": True,
    # D02's movable plate/food bodies share one GPU PhysX scene with the
    # world-fixed microwave articulation.  Isaac Sim 5.1 corrupts that
    # articulation on the first frame after the low-level sleep transaction
    # is applied to the three movable bodies (CUDA 700).  Reuse the proven
    # A/B/C awake stability gate: all task transforms are still inspected on
    # every physics frame and the accepted boundary is still exact-zero.
    "t12_load_and_start_microwave": False,
    # D06-D19 expose the same non-integrated GPU-TGS contact-correction
    # velocity already audited for B03/C05/C08.  Keep their task bodies awake
    # so the approved every-physics-frame static-pose proof remains meaningful.
    "t42_sort_objects": False,
    "d07_organize_office_desk": False,
    "d08_toast_bread": False,
    "t31_prepare_coffee": False,
    "d10_load_dishwasher": False,
    "d11_presoak_pan": False,
    "d12_prepare_cutting_station": False,
    "d13_prepare_tea_tray": False,
    "d14_wash_fruit": False,
    "d15_pack_matching_lunches": False,
    "d16_sweep_blocks_to_bin": False,
    "d17_store_laptop_headphones": False,
    "d18_pack_eggs": False,
    "d19_memory_cover_sort": False,
    "d20_microwave_hot_dog_plating": True,
    "d20_1_load_and_start": True,
    "d20_2_retrieve_and_plate": True,
    "d20_3_finish_table_setting": True,
}
_TASK_DEFORMABLE_NODE_RESET_DRIFT_LIMITS = {
    # An unstrained 14 mm-thick YCB sponge naturally sags on the table before
    # action[0]. Across three randomized 120 Hz resets, the largest authored-
    # to-settled node displacement was 10.94 mm while rigid bodies stayed
    # within the unchanged 5 mm gate and tetrahedral quality remained valid.
    # Treat this as bounded soft-body relaxation, not rigid scene drift.
    "eval_flex01_compress_sponge_into_mug": 0.012,
    # The unstrained 220 mm pillow relaxes by up to 55.06 mm under gravity
    # in the native formal reset. Its initial-state gate separately proves
    # a stationary, supported shape before action[0]. Bound that elastic
    # relaxation independently of the unchanged 5 mm rigid/robot pose gate.
    "eval_flex03_pack_pillow": 0.060,
}
_TASK_STATIC_POSE_FALLBACK_LIMITS = {
    # C01's dynamic wood+nail articulation can retain a non-integrated
    # GPU-TGS contact-correction velocity while its transforms are already
    # stationary on the tabletop.  The old velocity-only gate waited the
    # entire 39-second ABC timeout in that case.  Keep the scene awake and
    # require every task-body pose to remain stationary on every 120 Hz frame;
    # robot velocity, reset drift, contact validity, and the exact-zero
    # action-0 boundary remain unchanged.
    "t08_hammer_nail": (1.0e-6, 1.0e-5),
    # Z uses the same awake ABC TGS scene. Prove complete transforms/DOFs
    # on every 120 Hz frame before accepting residual contact velocity.
    "z01_pipette": (1.0e-6, 1.0e-5),
    "z02_bottle_cap": (1.0e-6, 1.0e-5),
    "t33_reposition_marker_grasp": (1.0e-6, 1.0e-5),
    # The fixed-base G08 box has a native lid coordinate and a composed
    # collision-processing override.  Its velocity readback can be exactly
    # zero while a one-frame hinge correction is still hidden between samples;
    # require the same every-120-Hz pose proof used by G10.  This is an
    # observation-only gate: no task pose or velocity is written per frame.
    "t18_open_micropipette_box": (1.0e-4, 1.0e-3),
    "eval_gen02_open_trash_bin": (1.0e-4, 1.0e-3),
    # G02's broad imported enclosure meshes can retain a small non-integrated
    # GPU-TGS correction velocity while their transforms are stationary.  As
    # with the audited A/B/C assets, inspect every physics-frame pose and keep
    # the robot velocity gate strict.
    "t06_fit_device_cover": (1.0e-6, 1.0e-5),
    # D13's thin tea-bag mesh and fixed-base cabinet can report a small
    # non-integrated contact angular velocity while their complete transforms
    # remain stationary.  Keep the velocity limits strict for the robot and
    # prove every task pose on every 120 Hz frame instead of accepting a
    # velocity-only timeout.
    # The complete ROKAE hand can settle a few tenths of a millimetre while
    # the task bodies are already stationary; keep the task-specific pose
    # proof active with this measured envelope.
    "d13_prepare_tea_tray": (1.0e-3, 1.0e-1),
    # H01's 32 analytic cylinders retain a repeatable GPU-TGS contact-
    # correction readback on the flat board.  A fresh Sim 5.1 trace measured
    # about 2.35e-4 m/s linear and 3.04e-2 rad/s angular velocity from 1 to 40
    # simulated seconds while complete piece transforms remained stationary
    # to sub-micrometre translation.  Prove the board and every piece pose on
    # all 120 Hz frames instead of relaxing the common velocity thresholds.
    # Robot motion, reset drift, contact validity, and the exact-zero action-0
    # boundary remain mandatory.
    "h01_chinese_chess": (1.0e-6, 1.0e-5),
    # A01's flat analytic banana and pear colliders can retain a repeatable
    # GPU-TGS contact-correction velocity while their transforms remain
    # stationary at float32 precision.  The gate remains read-only and checks
    # every 120 Hz physics frame; robot velocities are never bypassed.
    "t26_place_fruit_in_bucket": (1.0e-6, 1.0e-5),
    # XY3 reuses those exact fruit colliders. Its 24-case PhysX audit measured
    # the same non-integrated readback; prove all fruits, bucket and camera
    # transforms on every physics frame using A01's unchanged limits.
    "eval_comb04_place_fruit_and_photograph_logo": (1.0e-6, 1.0e-5),
    # GEN4 reuses D03's thin tabletop meshes.  The glue's GPU-TGS contact
    # correction can report a small residual angular velocity after its
    # transform has stopped integrating; prove every task pose each physics
    # frame while keeping robot motion, drift, contact and final-zero checks.
    "eval_gen04_pass_glue_stick": (1.0e-6, 1.0e-5),
    # P0-02 deliberately shares A01's robot home and recorder startup
    # contract. Keep its pose fallback identical so the two tasks do not
    # diverge in window-ready timing or action-0 qualification.
    "t17_open_latched_door": (1.0e-6, 1.0e-5),
    # A02's imported mug/tray pair can retain non-integrated contact solver
    # velocity after the randomized tabletop layout has become stationary.
    # Keep the same every-frame pose and strict robot proof used by A01.
    "t25_place_cup_on_plate": (1.0e-6, 1.0e-5),
    # A03's bowl collider reports a small angular correction at rest.  The
    # fallback covers every ball and bowl transform on every physics frame, so
    # a rolling ball or rocking bowl cannot satisfy it.  Keep the original
    # strict 1e-5 rad/frame threshold: after restoring the analytic ball mass
    # properties, the formal runtime measured only 3.86e-7 rad/frame.
    "t24_place_ball_in_bowl": (1.0e-6, 1.0e-5),
    # YCB02's eight imported cup/ball bodies can expose the same GPU-TGS
    # contact-correction velocity as A03 even after their transforms stop
    # integrating.  Prove every task-body transform on every 120 Hz physics
    # frame; a genuinely rolling candidate cannot satisfy these limits.
    "t27_place_largest_fitting_ball": (1.0e-6, 1.0e-5),
    # YCB01's upright carton colliders retain a repeatable contact-correction
    # readback after their transforms stop integrating.  Keep the same strict
    # every-frame pose proof while preserving the awake contact manifold.
    "t34_roll_ball_through_gate": (1.0e-6, 1.0e-5),
    # The thin table-setting utensils can retain a small angular tensor
    # readback after their transforms stop integrating. Prove every body pose
    # on every physics frame while keeping robot/contact gates strict.
    "t28_place_tableware": (1.0e-6, 1.0e-5),
    # The timer's free articulation can retain contact-correction velocity
    # after its chassis stops moving; require the same static-pose proof.
    "t39_set_timer": (1.0e-6, 1.0e-5),
    # Skillet/spatula/sponge contacts can retain non-integrated angular
    # readback after the authored poses stop moving.
    "t07_flip_sponge": (1.0e-6, 1.0e-5),
    # A06 uses a 27-shape convex cup.  GPU PhysX can leave a non-integrated
    # contact-correction velocity in its sleeping tensor state even while the
    # transform is bitwise stationary.  Accept that narrow readback case only
    # when every physics-frame task pose is substantially tighter than motion
    # allowed by the ordinary velocity gate; robot velocities remain strict.
    "t19_orient_cup_upright": (1.0e-6, 1.0e-5),
    # GEN1 uses the same awake compound-mesh contact and static-pose proof as
    # its A06 reference. Native traces retain contact-correction velocity;
    # every 120 Hz frame must still meet these unchanged limits for 0.5 s.
    # Robot motion, support/contact depth and reset-drift gates stay strict.
    "eval_gen03_orient_bottle_upright": (1.0e-6, 1.0e-5),
    # B01's imported pitcher and lid retain contact-solver velocity readback
    # after 40 simulated seconds.  A 60-frame cuda:0 trace measured at most
    # 1.50e-7 m / 1.93e-6 rad per frame while the reported angular speeds
    # remained near 1e-2 rad/s, so use the same full-frame pose proof.
    "t04_cover_pitcher": (1.0e-6, 1.0e-5),
    # XY5 reuses the identical B01 pitcher/lid and their non-integrated TGS
    # readback. Keep B01's strict every-frame pose proof for all four bodies;
    # robot, reset drift, contact and final-boundary gates remain unchanged.
    "eval_comb01_cover_pitcher_and_box": (1.0e-6, 1.0e-5),
    # B03's circular tabletop contacts can retain a non-integrated solver
    # velocity after both bodies stop moving.  Prove every rigid transform on
    # every 120 Hz frame; robot joints and the exact-zero final boundary stay
    # under the common strict contract.
    "t09_insert_peg": (1.0e-6, 1.0e-5),
    # B04's imported cube/wood colliders retain angular contact-correction
    # readback after all translations and orientations stop integrating.  Keep
    # the same every-frame all-body transform proof used by the other rigid
    # tabletop tasks; the stack orientation remains fully dynamic in episodes.
    "t44_stack_blocks": (1.0e-6, 1.0e-5),
    # GEN5 uses the same stack/contact path. Its settled source boxes retain
    # solver velocities too, so require B04's unchanged per-frame pose proof.
    "eval_gen07_stack_boxes": (1.0e-6, 1.0e-5),
    # B320-BC004's imported carton showed a 0.00579 rad/s angular readback after
    # the 39-second velocity gate while its linear and robot-joint velocities
    # were already below the strict limits.  The old velocity-only trace did
    # not prove whether that readback was integrated motion, so do not assume it
    # was harmless or raise the angular-velocity threshold: prove the carton's
    # full transform on every physics frame for the complete half-second
    # window instead.
    # Robot velocities, reset drift, contact penetration/report validity, and
    # the final exact-zero boundary remain mandatory.
    "t10_lift_carton": (1.0e-6, 1.0e-5),
    # BC014's CC-BY Yanez jars use two source-derived convex hulls apiece. A
    # fresh Isaac Sim 5.1 seed-20260824 trace reported up to 8.99e-4 m/s linear
    # and 2.25e-2 rad/s angular contact-correction velocity after one second,
    # while the measured positions changed by only 0.15-0.24 micrometres.
    # Prove both complete jar transforms on every physics frame instead of
    # relaxing the velocity thresholds. Contact validity, reset drift, and
    # the final exact-zero boundary remain mandatory.
    "t30_pour_seasonings": (1.0e-6, 1.0e-5),
    # XY6 keeps BC014's jars and KA012's rounded pot/lid/spoon contacts.
    # Its native seed-0 velocity-only gate reported 0.0224 rad/s for a jar
    # and 0.0645 rad/s for the spoon after 4,800 frames. Reuse BC014's strict
    # every-frame full-pose proof for all five bodies; never relax contact,
    # reset drift, robot velocity or the final action-0 boundary checks.
    "eval_comb06_season_and_stir": (1.0e-6, 1.0e-5),
    # HS005's analytic bottle cylinder and compound brush rest on two flat
    # supports.  Fresh Isaac Sim 5.1 traces report a repeatable GPU-TGS
    # contact-correction floor (about 0.0062 rad/s for the bottle) even while
    # every rigid transform remains stationary to sub-micrometre / microradian
    # precision.  Keep the common strict velocity gate and accept this narrow
    # readback case only through the full-frame all-body pose proof.  A bottle
    # that actually slides or rolls therefore still resets fail-closed.
    "t37_scrub_bottle": (1.0e-6, 1.0e-5),
    # KA020's sponge, bottle, and maximal-coordinate pump link retain a
    # repeatable GPU-TGS contact-correction readback after their transforms
    # stop integrating. A formal start measured roughly 0.006-0.008 rad/s on
    # all three supported bodies and consequently exhausted the 39-second
    # velocity gate. Prove every rigid transform on every 120 Hz frame for the
    # complete stability window; robot velocity, reset drift, contact
    # validity, and the exact-zero action-0 boundary remain strict.
    "t32_press_soap_pump": (1.0e-6, 1.0e-5),
    # KA021's reviewed mug is physically stationary on its analytic recessed
    # cube foot, but GPU TGS retains a non-integrated contact-correction
    # readback after the 39-second velocity gate. Fresh randomized traces
    # measured about 0.005-0.0064 rad/s while a direct 4,800-frame trace kept
    # the complete cup position stable to float32/sub-micrometre precision.
    # Prove the full cup and fixed faucet transforms on every physics frame;
    # robot velocity, reset drift, contact validity, and the exact-zero final
    # boundary remain mandatory.
    "t05_fill_cup": (1.0e-6, 1.0e-5),
    # XY4 reuses both HS005 support contacts and KA021's articulated faucet.
    # At 120 Hz the original brush retains support jitter: an every-frame
    # native trace measured 4.8 micrometres / 0.019 degrees maximum per step.
    # Keep the faucet awake, prove each frame within 10 micrometres / 0.029
    # degrees, and retain the common overall drift/contact/robot-motion gates.
    "eval_comb05_scrub_bottle_then_fill_cup": (1.0e-5, 5.0e-4),
    # BC021's two analytic tabletop block colliders report the same
    # non-integrated GPU-TGS correction as the audited B320 rigid tasks. A
    # fresh Isaac Sim 5.1 trace measured 0.0001663 m/s linear and 0.006044
    # rad/s angular readback after 39 simulated seconds while the blocks were
    # transform-static. Keep velocity, contact, robot, reset-drift, and final
    # exact-zero checks; this branch additionally proves both block poses on
    # every 120 Hz frame for the complete stability window.
    "t01_align_blocks_in_midair": (1.0e-6, 1.0e-5),
    # BC012's two analytic table-supported cans retain a repeatable GPU-TGS
    # angular correction readback of roughly 0.011-0.016 rad/s after 40
    # simulated seconds while their linear speeds and every robot velocity are
    # already below the strict limits.  Prove both can transforms, the bin,
    # and the table on every physics frame instead of relaxing that velocity
    # threshold.  Contact validity, reset drift, and the exact-zero boundary
    # remain mandatory.
    "t13_load_cans": (1.0e-6, 1.0e-5),
    # FU015's imported two-link laptop reports a repeatable GPU-TGS angular
    # correction near 1.6e-2 rad/s while a 25-second trace measured
    # bitwise/float32-stationary base and display transforms.  Check both links
    # and the hinge coordinate every frame; robot velocity, reset drift,
    # contact penetration, and the final exact-zero boundary remain strict.
    "t16_open_laptop": (1.0e-6, 1.0e-5),
    # XY2 reuses the exact FU015 laptop and its every-frame pose proof.
    "eval_comb02_open_laptop_and_click_mouse": (1.0e-6, 1.0e-5),
    # FU006's exact charger mesh retains a non-integrated GPU-TGS angular
    # contact-correction readback while lying motionless on the table.  Three
    # fresh PhysX scenes timed out the velocity-only gate even though linear
    # motion and robot joints were already strict.  Require the existing
    # every-120-Hz-frame all-body pose proof; contact validity, reset drift,
    # robot velocity, and the final exact-zero boundary remain unchanged.
    "t46_store_charger": (1.0e-6, 1.0e-5),
    # B320-AS013's COM-valid utility-knife housing face is statically
    # supported by its convex hull, but a fresh Isaac Sim 5.1 GPU-TGS run
    # retained a non-integrated contact-correction angular readback. Prove the
    # rigid root and blade DOF on every frame instead of relaxing velocity,
    # contact, reset-drift, robot, or final exact-zero gates.
    "t40_slide_knife_blade": (
        1.0e-6,
        1.0e-5,
    ),
    # C02's bowl/spoon pair exhibits the same non-integrated GPU-TGS contact
    # residual.  This whitelist entry does not bypass robot velocity, contact,
    # reset-drift, or final exact-zero checks.
    "t36_scoop_marbles": (1.0e-6, 1.0e-5),
    # C04's spray-bottle link keeps a contact-correction angular readback near
    # 9e-3 rad/s after the plant and robot have settled.  Require the same
    # every-frame transform proof; the robot joint velocity gate stays active.
    "t50_water_plant": (1.0e-6, 1.0e-5),
    # GEN7 inherits C04's full origin contract for the transferred spray task:
    # all task transforms and the physical trigger DOF are checked at 120 Hz,
    # with unchanged contact, robot velocity, drift and action-0 boundaries.
    "eval_gen06_spray_shoe": (1.0e-6, 1.0e-5),
    # C05's imported mouse articulation can retain a repeatable angular
    # contact-correction velocity on link_3 while its root/link transforms and
    # button DOFs remain stationary.  Prove every task pose and articulation
    # coordinate on every 120 Hz frame before accepting this readback case;
    # the final boundary still requires exact-zero velocities and targets.
    "t14_move_and_click_mouse": (1.0e-6, 1.0e-5),
    # C07 reuses B01's imported pitcher, whose GPU-TGS tensor can retain a
    # non-integrated ~4.6e-3 rad/s contact-correction readback.  The spherical
    # liquid proxies have their rotational symmetry constrained, so the strict
    # all-body, every-frame transform proof remains meaningful here as well.
    "c07_pitcher_pour_cup": (1.0e-6, 1.0e-5),
    # C08's flat table-supported cup and ice cubes retain the same
    # non-integrated GPU-TGS angular readback. The articulated tongs use flat
    # support pads, and the pose gate additionally covers their live hinge DOF
    # so an actually rocking or opening tool cannot pass this branch.
    "c08_tongs_ice_cubes": (1.0e-6, 1.0e-5),
    # C09 and C10 use thin imported tabletop parts whose TGS correction
    # velocity may remain present after their poses stop integrating.  Their
    # complete rigid/articulation layouts remain covered by the per-frame pose
    # gate, including task DOFs where present.
    "c09_staple_paper_sheets": (1.0e-6, 1.0e-5),
    "t02_connect_charger_and_cable": (1.0e-6, 1.0e-5),
    # C12's three small cuboids show the same non-integrated rotational
    # contact-correction readback as B04 after their transforms stop changing.
    # The all-body pose proof keeps their task-relevant orientations strict.
    "t11_line_up_blocks": (1.0e-6, 1.0e-5),
    # YCB05's source drill is a 13-hull free articulation.  On GPU TGS its
    # table-supported transform becomes stationary at float32 precision while
    # the root velocity tensor retains a repeatable non-integrated angular
    # correction (measured 7.51e-3 rad/s over the full 39 s gate).  The active
    # right hand is staged collision-free and all robot joint velocities stay
    # at exact zero, so use the same every-120-Hz-frame pose proof as the other
    # compound YCB/tabletop bodies.  Contact, penetration, reset-drift, and
    # final exact-zero checks remain mandatory.
    "t47_trigger_drill_on_block": (1.0e-6, 1.0e-5),
    # YCB03's fixed-wheel airplane retains a repeatable GPU-TGS articulation
    # velocity near 8e-3 rad/s while both its root and propeller transforms
    # remain stationary.  A physical seed-2303 trace measured at most 4.77e-7
    # m / 9.63e-7 rad per 120 Hz frame with zero penetration.  Keep the same
    # every-frame, all-body and articulation-DOF pose proof; robot velocities,
    # contact validity, reset drift, and the final exact-zero boundary remain
    # strict.
    "t35_rotate_propeller": (1.0e-6, 1.0e-5),
    # D01's imported plastic fasteners expose non-integrated GPU contact
    # correction velocity at their tabletop supports.  The long-horizon gate
    # still proves every rigid transform frame-by-frame before its existing
    # post-settle sleep/final-zero boundary.
    "d01_bimanual_nut_bolt_assembly": (1.0e-6, 1.0e-5),
    # D02's thin plate/food contact stack and D04's imported mesh contacts can
    # expose non-integrated solver velocity after every transform is static.
    # Their approved fallback still proves all task transforms every frame.
    "t12_load_and_start_microwave": (1.0e-6, 1.0e-5),
    "t21_pass_block": (1.0e-6, 1.0e-5),
    # XY1 retains D03's block, clutter and robot contacts. Reuse its existing
    # every-physics-frame origin proof for the combined scene.
    "eval_comb03_pass_block_and_close_box": (1.0e-6, 1.0e-5),
    "d04_clear_hanoi_obstacles": (1.0e-6, 1.0e-5),
    "t43_squeeze_scissors": (1.0e-6, 1.0e-5),
    # Fresh-process traces for D06-D08 remain transform-static after eight
    # simulated seconds while GPU TGS continues reporting a small, constant
    # contact-correction angular velocity on table-supported objects.  Reuse
    # the proven A/B/C full-frame pose proof; robot motion, contact penetration,
    # reset drift, and the exact-zero final boundary all remain strict.
    "t42_sort_objects": (1.0e-6, 1.0e-5),
    "d07_organize_office_desk": (1.0e-6, 1.0e-5),
    "d08_toast_bread": (1.0e-6, 1.0e-5),
    # D09-D12 use the same read-only proof for their mug, dishes, pan, sponge,
    # knife, board, meat, and articulated fixtures.  This does not accept
    # actual sliding or rocking: every task transform and task DOF must remain
    # below the per-frame micrometre/microradian limits for the full window.
    "t31_prepare_coffee": (1.0e-6, 1.0e-5),
    "d10_load_dishwasher": (1.0e-5, 1.0e-3),
    "d11_presoak_pan": (1.0e-6, 1.0e-5),
    "d12_prepare_cutting_station": (1.0e-6, 1.0e-5),
    # D13 and D15-D17 use the same table-supported imported payloads.  Their
    # fresh Sim 5.1 traces show stationary transforms with a small, constant
    # GPU-TGS angular readback; retain strict per-frame pose and joint checks
    # while accepting that non-integrated velocity signal.
    "d13_prepare_tea_tray": (1.0e-5, 1.0e-3),
    "d15_pack_matching_lunches": (1.0e-5, 1.0e-3),
    "d16_sweep_blocks_to_bin": (1.0e-6, 1.0e-5),
    "d17_store_laptop_headphones": (1.0e-5, 1.0e-3),
    "d19_memory_cover_sort": (1.0e-5, 1.0e-3),
    "bench320_bc007_bimanual_long_rod_yaw_reorientation": (1.0e-5, 1.0e-3),
    # D20 selected ceramic/food/tool meshes reproduce non-integrated GPU-TGS
    # correction velocities: the 5.1 trace moved less than 1.2 micrometres
    # over four seconds while angular readback remained around 0.02 rad/s.
    # Require the established per-120-Hz-frame transform AND joint-coordinate
    # proof; preserve strict robot, reset-drift and penetration checks.
    "d20_microwave_hot_dog_plating": (1.0e-6, 1.0e-5),
    "d20_1_load_and_start": (1.0e-6, 1.0e-5),
    "d20_2_retrieve_and_plate": (1.0e-6, 1.0e-5),
    "d20_3_finish_table_setting": (1.0e-6, 1.0e-5),
    # P0-01's D6-supported folding fan can expose a non-integrated angular
    # correction velocity on the constrained root and outer ribs even while
    # their transforms remain stationary.  Use the existing every-120-Hz-frame
    # proof for all task rigid bodies and articulation coordinates; robot
    # velocity, reset drift, contact penetration, and exact-zero boundary
    # checks remain unchanged and fail closed.
    "t48_unfold_fan": (1.0e-6, 1.0e-5),
    # G10 is a genuinely free, table-supported three-link articulation.  A
    # fresh GPU-TGS trace at the corrected z=.8933 m reset keeps the complete
    # ruler transform stationary to below 0.1 mm / 1 mrad per 120 Hz frame,
    # while PhysX reports a repeatable non-integrated root angular correction
    # of about 1.6e-2 rad/s.  Use the approved every-frame static-pose branch
    # for that readback rather than weakening the ordinary velocity gate.
    # Robot velocities, DOF drift, reset geometry, contact-report validity,
    # and the free-root gross-escape guard remain strict.
    "t49_unfold_ruler": (1.0e-4, 1.0e-3),
    # HL015's free connector/cover bodies can retain a repeatable
    # non-integrated GPU-TGS angular readback after they are transform-static
    # on their supported reset poses.  Keep the ordinary strict velocity and
    # contact/reset-drift checks, but prove every task-body transform on every
    # 120 Hz frame so a real collision or slide cannot be hidden by this
    # narrow fallback.
    "bench320_hl015_bimanual_cable_route_dual_plug": (1.0e-6, 1.0e-5),
    # P0-04's free spring-clamp articulation retains a non-integrated GPU-TGS
    # angular correction near 1.8e-2 rad/s while its root transform is static
    # at float32 precision.  Prove every rigid transform and its live hinge
    # coordinate on every 120 Hz frame, matching the existing tongs/scissors
    # contract; robot velocities and the final exact-zero boundary stay strict.
    "p004_hold_spring_clamp_open_insert_card": (1.0e-6, 1.0e-5),
    # P0-05's exact 16-part convex key collider retains a repeatable
    # non-integrated GPU-TGS contact-correction readback near 3.5e-2 rad/s on
    # the table.  Three fresh strict recorder resets reproduced the residual
    # while linear motion, the lock fixture, and every robot joint were
    # already below their velocity limits.  Prove the free key, padlock links,
    # and all three lock DOFs transform-static on every 120 Hz frame; contact
    # validity, reset drift, robot velocity, and final exact-zero checks remain
    # mandatory.
    "p005_key_insert_turn_unlock_open_shackle": (1.0e-6, 1.0e-5),
    # P0-06's downloaded female buckle uses six task-specific analytic box
    # colliders.  Two independent 10-second, 120 Hz GPU traces measured a
    # constant non-integrated contact-correction readback near 3.5e-3 rad/s
    # after its landing frame, with no penetration or forbidden contact.
    # Prove both free buckle transforms static on every physics frame rather
    # than relaxing the common velocity limit; reset drift, robot velocity,
    # contact validity, and the final exact-zero boundary remain mandatory.
    "p006_side_release_buckle_align_press": (1.0e-6, 1.0e-5),
    # HS014 uses the every-physics-frame static-pose proof because Isaac Sim 5.1
    # reports a repeatable contact-correction angular velocity on the supported
    # bottle while measured transform drift stays below 50 micrometres and
    # 0.5 milliradian. Robot velocity, reset drift, contact validity, and the
    # final zero-velocity boundary remain strict.
    "bench320_hs014_bottle_stabilized_cap_opening": (5.0e-5, 5.0e-4),
    # YCB09's source-scale wine glass rests on a thin circular foot.  Its
    # analytic flat support is stationary after settling, but GPU-TGS retains
    # a horizontal contact-correction velocity readback near 8e-3 rad/s.
    # Require the same all-body, all-robot, every-120-Hz-frame pose proof used
    # by the audited tabletop tasks; the reset/contact/final-zero gates remain
    # unchanged and a genuinely rocking glass cannot pass.
    "t29_place_wine_glass": (1.0e-6, 1.0e-5),
    # GEN7's imported spray-trigger articulation reports a small repeatable
    # angular correction while its transform remains below 0.3 mm per frame.
    # Prove the transform every physics frame and keep robot, contact, reset,
    # and final-zero checks strict.
    "eval_gen06_spray_shoe": (5.0e-4, 5.0e-3),
}
STATIC_POSE_FALLBACK_TASKS = frozenset(_TASK_STATIC_POSE_FALLBACK_LIMITS)
_DEFAULT_STATIC_POSE_FALLBACK_PHYSICS_HZ = 120.0
_TASK_STATIC_POSE_FALLBACK_PHYSICS_HZ = {
    # D04 retains its source-authored task/hand contact shells and object-side
    # CCD at the normal 120 Hz rate.  Its static-pose proof remains every-frame
    # and therefore gets 60 consecutive samples over the half-second window.
    "d04_clear_hanoi_obstacles": 120.0,
    # B320 runs four 120 Hz PhysX substeps per 30 Hz operator command. Keep the
    # same time-based stability window and micrometre/microradian pose limits;
    # the proof still samples every simulated frame (60 consecutive samples
    # over the default half-second window).
    "t10_lift_carton": 120.0,
    "t30_pour_seasonings": 120.0,
    "t01_align_blocks_in_midair": 120.0,
    # HS005 runs eight 240 Hz PhysX substeps per 30 Hz operator command.  The
    # half-second proof must inspect all 120 consecutive physics frames.
    "t37_scrub_bottle": 240.0,
    "t13_load_cans": 120.0,
    "t16_open_laptop": 120.0,
    "eval_comb02_open_laptop_and_click_mouse": 120.0,
}
# Public PhysX traces do not yet carry task-DOF units.  Keep unknown task
# coordinates fail-closed, but classify explicitly audited revolute joints so
# their drift is compared with the radian limit rather than the metre limit.
_KNOWN_REVOLUTE_TASK_DOF_NAMES = frozenset(
    {
        "microjoint",
        "RevoluteJoint_computer_9_up",
        "left_knob_joint",
        "right_knob_joint",
        *(f"Prop{index:03d}_joint" for index in range(1, 20)),
    }
) | frozenset(
    dof_name
    for spec in LW_TASK_SPECS.values()
    for dof_name in spec.dof_names
) | frozenset({"p004_spring_clamp_moving_jaw_joint"})
_ACTIVE_PHYSICS_LAYOUT_FORMAT = "handson.active_physics_layout"
_ACTIVE_PHYSICS_LAYOUT_VERSION = 1
_ACTIVE_PHYSICS_LAYOUT_KINDS = (
    "rigid_body",
    "articulation_root",
    "articulation_dof",
    "deformable_node",
)


class RecordingOriginStabilityError(RuntimeError):
    """Raised when a reset state cannot reach its controlled origin in time."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def stable_recording_origin_kwargs_for_task(
    task_name: object,
) -> dict[str, Any]:
    """Return the strict recording-origin contract for a new episode."""

    normalized_task_name = str(task_name or "").strip().lower()
    bc011_pen_settle = (
        normalized_task_name == "t20_pass_and_orient_pen"
    )
    gate_only_velocity_clamp = normalized_task_name in {
        "t45_stir_pot",
        "t20_pass_and_orient_pen",
        "loaded_tray_through_narrow_gate",
        "t27_place_largest_fitting_ball",
        "t41_sort_nested_cups",
        # D20's randomized thin props can regenerate a small angular contact
        # correction after the sleep transaction. Clamp only during the
        # origin gate; live collection remains fully dynamic.
        "d20_microwave_hot_dog_plating",
        "d20_1_load_and_start",
        "d20_1_5_set_controls",
        "d20_2_retrieve_and_plate",
        "d20_3_finish_table_setting",
        # D13/D15/D17 use imported or compound task colliders that retain a
        # tiny PhysX contact-axis correction after reset.  Clamp only during
        # the origin gate; all three remain dynamic once collection starts.
        "d13_prepare_tea_tray",
        "d15_pack_matching_lunches",
        "d17_store_laptop_headphones",
    }
    is_abc_collection_task = (
        is_abc_style_collection_task(normalized_task_name)
        or normalized_task_name
        in {
            "eval_comb04_place_fruit_and_photograph_logo",
            "t34_roll_ball_through_gate",
            "t27_place_largest_fitting_ball",
            "eval_flex01_compress_sponge_into_mug",
            "t17_open_latched_door",
            "g01_deformable_asset_inspection",
            "g01_2_deformable_asset_inspection",
            "g01_3_bimanual_pillow_to_carton",
            "g01_4_soft_pillow_to_carton",
            "eval_flex03_pack_pillow",
            "t06_fit_device_cover",
            *LW_TASK_NAMES,
        }
    )

    # Every task first advances naturally so contact manifolds can converge,
    # then gets one velocity-only boundary transaction.  The following
    # strict gate is deliberately write-free for ordinary tasks. BC011,
    # BC005, KA012, YCB02, and YCB04 are narrow task-level exceptions: their
    # cylindrical, compound, rolling, or tightly nested contacts regenerate
    # non-integrated GPU-TGS velocities while the objects remain on their
    # supported reset poses. Zero only task-body
    # velocities during the half-second initialization gate, then stop all
    # writes before action[0]. Reset drift and contact validity remain strict.
    kwargs: dict[str, Any] = {
        "stability_metric": "velocity",
        "initial_settle_seconds": (
            1.0 / 120.0
            if bc011_pen_settle
            else _TASK_INITIAL_SETTLE_SECONDS.get(
                normalized_task_name,
                _COLLECTION_INITIAL_SETTLE_SECONDS,
            )
        ),
        "post_settle_velocity_zeroing": True,
        # The public low-level sleep transaction perturbs a separate
        # articulation on GPU PhysX (C01 and D02 both reproduce the failure on
        # a following frame). Most A/B/C tasks and audited task overrides
        # therefore stay awake and prove stillness through the same strict
        # gate. The two rigid-only YCB overrides above use a wakeable sleep
        # boundary instead.
        "post_settle_sleeping": _TASK_POST_SETTLE_SLEEPING_OVERRIDES.get(
            normalized_task_name,
            not is_abc_collection_task,
        ),
        "stabilization_velocity_clamp": gate_only_velocity_clamp,
        "stabilization_velocity_clamp_scope": (
            VELOCITY_CLAMP_SCOPE_GATE_ONLY
            if gate_only_velocity_clamp
            else VELOCITY_CLAMP_SCOPE_NONE
        ),
        # A zero-velocity readback is not enough: a penetrating or unsupported
        # reset can first jump to a different pose and only then become still.
        # Collection therefore fails closed on both the reset-to-gate pose
        # drift and PhysX's solved-contact separation.  Rejected samples are
        # rebuilt from a fresh seeded scene and can never become action[0].
        "reset_translation_drift_limit": (
            DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT
        ),
        "reset_rotation_drift_limit": DEFAULT_RESET_ROTATION_DRIFT_LIMIT,
        "require_valid_contact_report": True,
        "contact_penetration_limit": DEFAULT_CONTACT_PENETRATION_LIMIT,
        # ABC normally gets a 40 s total initialization budget: 1 s of
        # unrendered contact settling at the unchanged physics dt plus at most
        # 39 s in the strict gate. Deformable task overrides may use a longer
        # natural settle before the same strict gate. Rigid-only task overrides
        # can use the
        # shorter budget above after establishing a wakeable sleep boundary.
        # The gate exits immediately after 0.5 continuous stable seconds.
        "timeout_seconds": _TASK_POST_SETTLE_TIMEOUT_SECONDS.get(
            normalized_task_name,
            (
                _ABC_COLLECTION_POST_SETTLE_TIMEOUT_SECONDS
                if is_abc_collection_task
                else _COLLECTION_POST_SETTLE_TIMEOUT_SECONDS
            ),
        ),
    }
    if normalized_task_name in _TASKS_ALLOW_ROBOT_RESET_REPOSITION:
        kwargs["reset_translation_drift_limit"] = None
        kwargs["reset_rotation_drift_limit"] = None
    if is_abc_collection_task:
        # Inspect every GPU physics frame.  Counting a 50 ms sample interval
        # as continuously stable could otherwise hide a one-frame impulse.
        kwargs["stability_sample_seconds"] = 0.0
    stable_seconds = _TASK_STABLE_SECONDS.get(normalized_task_name)
    if stable_seconds is not None:
        kwargs["stable_seconds"] = stable_seconds
        kwargs["stability_sample_seconds"] = 0.0
    linear_velocity_limit = _TASK_LINEAR_VELOCITY_LIMITS.get(
        normalized_task_name
    )
    if linear_velocity_limit is not None:
        kwargs["linear_velocity_limit"] = linear_velocity_limit
    angular_velocity_limit = _TASK_ANGULAR_VELOCITY_LIMITS.get(
        normalized_task_name
    )
    if angular_velocity_limit is not None:
        kwargs["angular_velocity_limit"] = angular_velocity_limit
    reset_rotation_drift_limit = _TASK_RESET_ROTATION_DRIFT_LIMITS.get(
        normalized_task_name
    )
    if reset_rotation_drift_limit is not None:
        kwargs["reset_rotation_drift_limit"] = reset_rotation_drift_limit
    deformable_node_reset_drift_limit = (
        _TASK_DEFORMABLE_NODE_RESET_DRIFT_LIMITS.get(normalized_task_name)
    )
    if deformable_node_reset_drift_limit is not None:
        kwargs["deformable_node_reset_drift_limit"] = (
            deformable_node_reset_drift_limit
        )
    pose_fallback_limits = _TASK_STATIC_POSE_FALLBACK_LIMITS.get(
        normalized_task_name
    )
    if pose_fallback_limits is not None:
        kwargs.update(
            {
                "stability_metric": "velocity_or_static_pose",
                # The fallback is only valid as an every-physics-frame proof;
                # this also applies to its explicitly audited D tasks.
                "stability_sample_seconds": 0.0,
                "pose_translation_limit": pose_fallback_limits[0],
                "pose_rotation_limit": pose_fallback_limits[1],
            }
        )
    return kwargs


def stable_recording_origin_reset_attempts_for_task(task_name: object) -> int:
    """Return how many freshly sampled reset poses may be tried."""

    configured = os.environ.get(STABLE_ORIGIN_RESET_ATTEMPTS_ENV)
    if configured is not None:
        try:
            attempts = int(configured)
        except ValueError as exc:
            raise ValueError(
                f"{STABLE_ORIGIN_RESET_ATTEMPTS_ENV} must be a positive integer; "
                f"got {configured!r}."
            ) from exc
        if attempts <= 0:
            raise ValueError(
                f"{STABLE_ORIGIN_RESET_ATTEMPTS_ENV} must be a positive integer; "
                f"got {configured!r}."
            )
        return attempts
    normalized_task_name = str(task_name or "").strip().lower()
    return _TASK_RESET_ATTEMPTS.get(
        normalized_task_name,
        _COLLECTION_RESET_ATTEMPTS,
    )


def stable_recording_origin_kwargs_from_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recover the exact stabilization inputs stored in a replay manifest."""

    if not isinstance(report, Mapping):
        return {}
    numeric_fields = {
        "linear_velocity_limit": "linear_velocity_limit",
        "angular_velocity_limit": "angular_velocity_limit",
        "joint_velocity_limit": "joint_velocity_limit",
        "initial_settle_seconds": "initial_settle_seconds",
        "pose_translation_limit": "pose_translation_limit",
        "pose_rotation_limit": "pose_rotation_limit",
        "reset_translation_drift_limit": (
            "reset_translation_drift_limit"
        ),
        "deformable_node_reset_drift_limit": (
            "deformable_node_reset_drift_limit"
        ),
        "reset_rotation_drift_limit": "reset_rotation_drift_limit",
        "contact_penetration_limit": "contact_penetration_limit",
        "stable_seconds_required": "stable_seconds",
        "stability_sample_seconds": "stability_sample_seconds",
        "timeout_seconds": "timeout_seconds",
    }
    kwargs: dict[str, Any] = {}
    for report_name, argument_name in numeric_fields.items():
        value = report.get(report_name)
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                "Recorded stable-origin parameter must be finite and "
                f"non-negative: {report_name}={value!r}"
            )
        kwargs[argument_name] = numeric
    metric = report.get("stability_metric")
    if metric is not None:
        kwargs["stability_metric"] = str(metric)
    zero_after_settle = report.get("post_settle_velocity_zeroing")
    if zero_after_settle is not None:
        if type(zero_after_settle) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "post_settle_velocity_zeroing must be boolean."
            )
        kwargs["post_settle_velocity_zeroing"] = zero_after_settle
    sleep_after_settle = report.get("post_settle_sleeping")
    if sleep_after_settle is not None:
        if type(sleep_after_settle) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter post_settle_sleeping must be boolean."
            )
        kwargs["post_settle_sleeping"] = sleep_after_settle
    require_contact_report = report.get("require_valid_contact_report")
    if require_contact_report is not None:
        if type(require_contact_report) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "require_valid_contact_report must be boolean."
            )
        kwargs["require_valid_contact_report"] = require_contact_report
    velocity_clamp = report.get("stabilization_velocity_clamp")
    if velocity_clamp is not None:
        if type(velocity_clamp) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "stabilization_velocity_clamp must be boolean."
            )
    per_step_velocity_writes = report.get(
        "initialization_per_physics_step_velocity_writes"
    )
    if per_step_velocity_writes is not None:
        if type(per_step_velocity_writes) is not bool:
            raise ValueError(
                "Recorded stable-origin parameter "
                "initialization_per_physics_step_velocity_writes must be "
                "boolean."
            )
        if velocity_clamp is not None and (
            per_step_velocity_writes != velocity_clamp
        ):
            raise ValueError(
                "Recorded stable-origin velocity clamp flags must match."
            )
    velocity_clamp_scope = report.get("stabilization_velocity_clamp_scope")
    # A few historical reports omitted one redundant boolean or explicitly
    # wrote scope=none before both booleans were emitted.  Normalize those
    # representations; enabled reports still need internally consistent step
    # and write counts in the certifier.
    if velocity_clamp is None and per_step_velocity_writes is not None:
        velocity_clamp = per_step_velocity_writes
    elif per_step_velocity_writes is None and velocity_clamp is not None:
        per_step_velocity_writes = velocity_clamp
    elif (
        velocity_clamp is None
        and per_step_velocity_writes is None
        and velocity_clamp_scope == VELOCITY_CLAMP_SCOPE_NONE
    ):
        velocity_clamp = False
        per_step_velocity_writes = False
    if velocity_clamp is not None:
        kwargs["stabilization_velocity_clamp"] = velocity_clamp
    if velocity_clamp_scope is None:
        # Reports written before the scope field existed clamped only the
        # strict gate.  Preserve that call history exactly during replay.
        if velocity_clamp is True:
            velocity_clamp_scope = VELOCITY_CLAMP_SCOPE_GATE_ONLY
        elif velocity_clamp is False:
            velocity_clamp_scope = VELOCITY_CLAMP_SCOPE_NONE
    if velocity_clamp_scope is not None:
        if velocity_clamp is None:
            raise ValueError(
                "Recorded stable-origin velocity clamp scope requires its "
                "boolean clamp flag."
            )
        if (
            not isinstance(velocity_clamp_scope, str)
            or velocity_clamp_scope not in _VELOCITY_CLAMP_SCOPES
        ):
            raise ValueError(
                "Recorded stable-origin parameter "
                "stabilization_velocity_clamp_scope is invalid: "
                f"{velocity_clamp_scope!r}."
            )
        if velocity_clamp is True and (
            velocity_clamp_scope == VELOCITY_CLAMP_SCOPE_NONE
        ):
            raise ValueError(
                "Recorded velocity clamp cannot use scope 'none'."
            )
        if velocity_clamp is False and (
            velocity_clamp_scope != VELOCITY_CLAMP_SCOPE_NONE
        ):
            raise ValueError(
                "Recorded disabled velocity clamp must use scope 'none'."
            )
        kwargs["stabilization_velocity_clamp_scope"] = (
            velocity_clamp_scope
        )
    return kwargs


def stable_recording_origin_enabled() -> bool:
    return os.environ.get(STABLE_ORIGIN_ENV, "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _resolve_joint_velocity_limit(
    raw_env: Any,
    requested: float | None,
) -> tuple[float, str]:
    configured = os.environ.get(JOINT_VELOCITY_LIMIT_ENV)
    if configured is not None:
        return float(configured), "environment_variable"
    if requested is not None:
        return float(requested), "argument"

    environment_limit = getattr(
        raw_env,
        "stable_recording_origin_joint_velocity_limit",
        None,
    )
    if environment_limit is not None:
        return float(environment_limit), "environment"

    task_name = str(getattr(raw_env, "task_name", "")).strip().lower()
    if task_name in _TASK_JOINT_VELOCITY_LIMITS:
        return float(_TASK_JOINT_VELOCITY_LIMITS[task_name]), "task_default"
    return float(DEFAULT_JOINT_VELOCITY_LIMIT), "default"


def _valid_states(
    snapshot: Mapping[str, Any],
    state_name: str,
    valid_name: str,
) -> list[np.ndarray] | None:
    states = np.asarray(snapshot.get(state_name, np.zeros((0, 13))))
    raw_valid = np.asarray(snapshot.get(valid_name, np.zeros(0)))
    if raw_valid.ndim != 1:
        return None
    valid = raw_valid.reshape(-1)
    if (
        states.ndim != 2
        or states.shape[-1] < 13
        or states.shape[0] != valid.size
        or not np.all((valid == 0) | (valid == 1))
    ):
        return None
    active = valid.astype(bool)
    try:
        active_states = np.asarray(states[active, :13], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.all(np.isfinite(active_states)):
        return None
    return [active_states[index] for index in range(active_states.shape[0])]


def _object_motion(snapshot: Mapping[str, Any]) -> tuple[float, float] | None:
    rigid_states = _valid_states(
        snapshot,
        "physx_task_rigid_body_state",
        "physx_task_rigid_body_valid",
    )
    articulation_states = _valid_states(
        snapshot,
        "physx_task_articulation_root_state",
        "physx_task_articulation_root_valid",
    )
    if rigid_states is None or articulation_states is None:
        return None
    states = [*rigid_states, *articulation_states]
    linear_speeds = [
        float(np.linalg.norm(state[7:10]))
        for state in states
    ]
    angular_speeds = [
        float(np.linalg.norm(state[10:13]))
        for state in states
    ]

    articulation_velocity = np.asarray(
        snapshot.get(
            "physx_task_articulation_dof_velocity",
            np.zeros((0, 0)),
        ),
        dtype=np.float64,
    )
    articulation_valid = np.asarray(
        snapshot.get(
            "physx_task_articulation_dof_valid",
            np.zeros((0, 0)),
        )
    )
    if (
        articulation_velocity.shape != articulation_valid.shape
        or articulation_valid.ndim != 2
        or not np.all(
            (articulation_valid == 0) | (articulation_valid == 1)
        )
    ):
        return None
    valid_velocity = articulation_velocity[articulation_valid.astype(bool)]
    if not np.all(np.isfinite(valid_velocity)):
        return None
    if valid_velocity.size:
        angular_speeds.append(float(np.max(np.abs(valid_velocity))))

    deformable_velocity = np.asarray(
        snapshot.get("physx_deformable_velocity", np.zeros((0, 3))),
        dtype=np.float64,
    )
    deformable_valid = np.asarray(
        snapshot.get("physx_deformable_valid", np.zeros(0))
    ).reshape(-1)
    if (
        deformable_valid.size not in {0, 1}
        or not np.all((deformable_valid == 0) | (deformable_valid == 1))
        or deformable_velocity.ndim != 2
        or deformable_velocity.shape[-1] != 3
        or (deformable_valid.size == 0 and deformable_velocity.shape[0] != 0)
    ):
        return None
    if deformable_valid.size and bool(deformable_valid[0]):
        if not deformable_velocity.size or not np.all(
            np.isfinite(deformable_velocity)
        ):
            return None
        linear_speeds.append(
            float(np.max(np.linalg.norm(deformable_velocity, axis=-1)))
        )

    if not linear_speeds and not angular_speeds:
        return None
    return (
        max(linear_speeds, default=0.0),
        max(angular_speeds, default=0.0),
    )


def _task_motion_peaks(
    snapshot: Mapping[str, Any],
    trace: PhysxGroundTruthTrace,
    *,
    linear_velocity_limit: float,
    angular_velocity_limit: float,
    count: int = 5,
) -> list[dict[str, Any]]:
    peaks: dict[str, dict[str, Any]] = {}
    layouts = (
        (
            getattr(trace, "rigid_bodies", ()),
            "physx_task_rigid_body_state",
            "physx_task_rigid_body_valid",
        ),
        (
            getattr(trace, "articulations", ()),
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        ),
    )
    for entries, state_name, valid_name in layouts:
        states = np.asarray(snapshot.get(state_name, np.zeros((0, 13))))
        valid = np.asarray(snapshot.get(valid_name, np.zeros(0))).reshape(-1)
        for index, entry in enumerate(entries):
            if (
                index >= states.shape[0]
                or index >= valid.size
                or not bool(valid[index])
                or states.shape[-1] < 13
            ):
                continue
            state = np.asarray(states[index, :13], dtype=np.float64)
            if not np.all(np.isfinite(state[7:13])):
                continue
            linear = float(np.linalg.norm(state[7:10]))
            angular = float(np.linalg.norm(state[10:13]))
            score = max(
                linear / max(linear_velocity_limit, 1e-12),
                angular / max(angular_velocity_limit, 1e-12),
            )
            name = str(entry.name)
            previous = peaks.get(name)
            if previous is None or score > float(previous["limit_ratio"]):
                peaks[name] = {
                    "name": name,
                    "linear_velocity": linear,
                    "angular_velocity": angular,
                    "limit_ratio": score,
                }
    return sorted(
        peaks.values(),
        key=lambda item: float(item["limit_ratio"]),
        reverse=True,
    )[: max(0, int(count))]


def _quaternion_rotation_delta(
    before_quat: np.ndarray,
    after_quat: np.ndarray,
) -> float | None:
    """Return a sign-invariant quaternion angle with stable near-zero precision.

    ``2 * acos(abs(dot(q0, q1)))`` is ill-conditioned near identical
    orientations.  PhysX stores transforms as float32, so a no-step tensor
    synchronization can change only a few quaternion ULPs while the acos
    formulation reports a false rotation above the strict 1e-5 rad recording
    boundary.  The equivalent chord/atan2 form remains linear near zero and
    still returns the full geodesic angle in ``[0, pi]``.
    """

    before = np.asarray(before_quat, dtype=np.float64).reshape(4)
    after = np.asarray(after_quat, dtype=np.float64).reshape(4)
    before_norm = float(np.linalg.norm(before))
    after_norm = float(np.linalg.norm(after))
    if (
        not np.isfinite(before_norm)
        or not np.isfinite(after_norm)
        or before_norm <= 1.0e-12
        or after_norm <= 1.0e-12
    ):
        return None
    before = before / before_norm
    after = after / after_norm
    if float(np.dot(before, after)) < 0.0:
        after = -after
    return float(
        4.0
        * np.arctan2(
            np.linalg.norm(after - before),
            np.linalg.norm(after + before),
        )
    )


def _task_pose_delta(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Return maximum consecutive task translation and rotation coordinates."""

    translations: list[float] = []
    rotations: list[float] = []
    for state_name, valid_name in (
        ("physx_task_rigid_body_state", "physx_task_rigid_body_valid"),
        (
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        ),
    ):
        before = np.asarray(previous.get(state_name, np.zeros((0, 13))))
        after = np.asarray(current.get(state_name, np.zeros((0, 13))))
        before_valid = np.asarray(
            previous.get(valid_name, np.zeros(0))
        ).reshape(-1)
        after_valid = np.asarray(
            current.get(valid_name, np.zeros(0))
        ).reshape(-1)
        if (
            before.ndim != 2
            or after.ndim != 2
            or before.shape != after.shape
            or before.shape[-1] < 7
            or before_valid.shape != after_valid.shape
            or before_valid.size != before.shape[0]
        ):
            return None
        before_valid = before_valid.astype(bool)
        after_valid = after_valid.astype(bool)
        # Comparing separately filtered lists can silently pair two different
        # bodies when validity changes between frames.  Require an identical
        # indexed body set for the whole consecutive-pose check.
        if not np.array_equal(before_valid, after_valid):
            return None
        for index in np.flatnonzero(before_valid):
            before_state = np.asarray(before[index, :7], dtype=np.float64)
            after_state = np.asarray(after[index, :7], dtype=np.float64)
            if not (
                np.all(np.isfinite(before_state))
                and np.all(np.isfinite(after_state))
            ):
                return None
            translations.append(
                float(np.linalg.norm(after_state[:3] - before_state[:3]))
            )
            rotation = _quaternion_rotation_delta(
                before_state[3:7],
                after_state[3:7],
            )
            if rotation is None:
                return None
            rotations.append(rotation)

    before_dof = np.asarray(
        previous.get("physx_task_articulation_dof_position", np.zeros((0, 0)))
    )
    after_dof = np.asarray(
        current.get("physx_task_articulation_dof_position", np.zeros((0, 0)))
    )
    before_dof_valid = np.asarray(
        previous.get("physx_task_articulation_dof_valid", np.zeros((0, 0)))
    )
    after_dof_valid = np.asarray(
        current.get("physx_task_articulation_dof_valid", np.zeros((0, 0)))
    )
    if (
        before_dof.ndim != 2
        or after_dof.shape != before_dof.shape
        or before_dof_valid.shape != before_dof.shape
        or after_dof_valid.shape != before_dof.shape
        or not np.array_equal(before_dof_valid, after_dof_valid)
        or not np.all((before_dof_valid == 0) | (before_dof_valid == 1))
    ):
        return None
    active_dof = before_dof_valid.astype(bool)
    if not (
        np.all(np.isfinite(before_dof[active_dof]))
        and np.all(np.isfinite(after_dof[active_dof]))
    ):
        return None
    if np.any(active_dof):
        maximum_dof_delta = float(
            np.max(np.abs(after_dof[active_dof] - before_dof[active_dof]))
        )
        # The public trace does not yet carry each task DOF's prismatic/revolute
        # type. Apply the stricter of the translation and rotation pose limits
        # by contributing the coordinate delta to both maxima. This prevents a
        # stationary articulation root from hiding a moving trigger or jaw.
        translations.append(maximum_dof_delta)
        rotations.append(maximum_dof_delta)
    if not translations and not rotations:
        return None
    return max(translations, default=0.0), max(rotations, default=0.0)


def _canonical_layout_names(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of names.")
    names: list[str] = []
    for raw_name in value:
        if not isinstance(raw_name, str):
            raise ValueError(f"{label} contains a non-string name.")
        name = raw_name.strip()
        if not name or name != raw_name:
            raise ValueError(f"{label} contains an empty or non-canonical name.")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names.")
    return names


def _canonical_layout_bitmap(
    value: Any,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    try:
        bitmap = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an array.") from exc
    if bitmap.shape != shape:
        raise ValueError(
            f"{label} shape changed: expected={shape}, actual={bitmap.shape}."
        )
    try:
        numeric = bitmap.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not a numeric bitmap.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{label} must contain only 0 or 1.")
    return numeric.astype(np.uint8)


def _active_physics_layout(
    snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the fixed PhysX schema rows to this reset's active bitmap."""

    if not isinstance(snapshot, Mapping) or not isinstance(schema, Mapping):
        raise ValueError("PhysX snapshot and schema must be objects.")
    physx = schema.get("physx")
    if not isinstance(physx, Mapping):
        raise ValueError("PhysX schema is missing its physx object.")

    rigid_names = _canonical_layout_names(
        physx.get("rigid_body_names"),
        label="physx.rigid_body_names",
    )
    rigid_raw = snapshot.get("physx_task_rigid_body_valid")
    if rigid_raw is None and not rigid_names:
        rigid_raw = np.zeros(0, dtype=np.uint8)
    rigid_valid = _canonical_layout_bitmap(
        rigid_raw,
        shape=(len(rigid_names),),
        label="physx_task_rigid_body_valid",
    )

    articulation_names = _canonical_layout_names(
        physx.get("articulation_names"),
        label="physx.articulation_names",
    )
    articulation_root_raw = snapshot.get(
        "physx_task_articulation_root_valid"
    )
    if articulation_root_raw is None and not articulation_names:
        articulation_root_raw = np.zeros(0, dtype=np.uint8)
    articulation_root_valid = _canonical_layout_bitmap(
        articulation_root_raw,
        shape=(len(articulation_names),),
        label="physx_task_articulation_root_valid",
    )

    raw_dof_names = physx.get("articulation_dof_names")
    if not isinstance(raw_dof_names, Mapping):
        raise ValueError("physx.articulation_dof_names must be an object.")
    if set(raw_dof_names) != set(articulation_names):
        raise ValueError(
            "physx.articulation_dof_names keys must exactly match "
            "physx.articulation_names."
        )
    articulation_dof_names: list[str] = []
    articulation_dof_rows: list[list[str]] = []
    for articulation_name in articulation_names:
        dof_names = _canonical_layout_names(
            raw_dof_names[articulation_name],
            label=(
                "physx.articulation_dof_names"
                f"[{articulation_name!r}]"
            ),
        )
        qualified = [
            f"{articulation_name}:{dof_name}" for dof_name in dof_names
        ]
        articulation_dof_rows.append(qualified)
        articulation_dof_names.extend(qualified)
    if len(set(articulation_dof_names)) != len(articulation_dof_names):
        raise ValueError("Qualified articulation DOF names are not unique.")
    max_dof_count = max(
        (len(names) for names in articulation_dof_rows),
        default=0,
    )
    articulation_dof_raw = snapshot.get(
        "physx_task_articulation_dof_valid"
    )
    if articulation_dof_raw is None and max_dof_count == 0:
        articulation_dof_raw = np.zeros(
            (len(articulation_names), 0),
            dtype=np.uint8,
        )
    articulation_dof_bitmap = _canonical_layout_bitmap(
        articulation_dof_raw,
        shape=(len(articulation_names), max_dof_count),
        label="physx_task_articulation_dof_valid",
    )
    articulation_dof_valid: list[int] = []
    for row, names in enumerate(articulation_dof_rows):
        articulation_dof_valid.extend(
            int(value)
            for value in articulation_dof_bitmap[row, : len(names)]
        )
        if np.any(articulation_dof_bitmap[row, len(names) :]):
            raise ValueError(
                "physx_task_articulation_dof_valid marks padded DOFs active."
            )

    deformable_name = physx.get("deformable_name")
    if deformable_name is None:
        deformable_names: list[str] = []
        deformable_valid: list[int] = []
        deformable_positions = snapshot.get("physx_deformable_position")
        deformable_bitmap = snapshot.get("physx_deformable_valid")
        if deformable_positions is not None and np.asarray(
            deformable_positions
        ).shape != (0, 3):
            raise ValueError(
                "Snapshot contains deformable nodes absent from the schema."
            )
        if deformable_bitmap is not None:
            _canonical_layout_bitmap(
                deformable_bitmap,
                shape=(0,),
                label="physx_deformable_valid",
            )
    else:
        deformable_names_base = _canonical_layout_names(
            [deformable_name],
            label="physx.deformable_name",
        )
        positions = np.asarray(
            snapshot.get("physx_deformable_position", np.zeros((0, 0)))
        )
        if positions.ndim != 2 or positions.shape[1] != 3 or not positions.shape[0]:
            raise ValueError("PhysX deformable node layout is missing or invalid.")
        deformable_bitmap = _canonical_layout_bitmap(
            snapshot.get("physx_deformable_valid"),
            shape=(1,),
            label="physx_deformable_valid",
        )
        deformable_names = [
            f"{deformable_names_base[0]}:{index}"
            for index in range(positions.shape[0])
        ]
        deformable_valid = [
            int(deformable_bitmap[0]) for _ in deformable_names
        ]

    layout = {
        "format": _ACTIVE_PHYSICS_LAYOUT_FORMAT,
        "version": _ACTIVE_PHYSICS_LAYOUT_VERSION,
        "rigid_body": {
            "names": rigid_names,
            "valid": rigid_valid.astype(int).tolist(),
        },
        "articulation_root": {
            "names": articulation_names,
            "valid": articulation_root_valid.astype(int).tolist(),
        },
        "articulation_dof": {
            "names": articulation_dof_names,
            "valid": articulation_dof_valid,
        },
        "deformable_node": {
            "names": deformable_names,
            "valid": deformable_valid,
        },
    }
    for kind in _ACTIVE_PHYSICS_LAYOUT_KINDS:
        section = layout[kind]
        if len(section["names"]) != len(section["valid"]):
            raise ValueError(f"Active {kind} names and validity lengths differ.")
        if len(set(section["names"])) != len(section["names"]):
            raise ValueError(f"Active {kind} names are not unique.")
        if any(not name for name in section["names"]):
            raise ValueError(f"Active {kind} contains an empty name.")
        if any(value not in {0, 1} for value in section["valid"]):
            raise ValueError(f"Active {kind} validity is not a bitmap.")
    return layout


def _expected_active_physics_layout(
    raw_env: Any,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive expected task activity independently of live readback validity."""

    if not isinstance(schema, Mapping):
        raise ValueError("PhysX schema must be an object.")
    physx = schema.get("physx")
    if not isinstance(physx, Mapping):
        raise ValueError("PhysX schema is missing its physx object.")
    rigid_names = _canonical_layout_names(
        physx.get("rigid_body_names"),
        label="physx.rigid_body_names",
    )
    articulation_names = _canonical_layout_names(
        physx.get("articulation_names"),
        label="physx.articulation_names",
    )
    raw_dof_names = physx.get("articulation_dof_names")
    if not isinstance(raw_dof_names, Mapping) or set(raw_dof_names) != set(
        articulation_names
    ):
        raise ValueError(
            "physx.articulation_dof_names keys must exactly match "
            "physx.articulation_names."
        )
    dof_rows = [
        _canonical_layout_names(
            raw_dof_names[name],
            label=f"physx.articulation_dof_names[{name!r}]",
        )
        for name in articulation_names
    ]
    max_dof_count = max((len(row) for row in dof_rows), default=0)

    rigid_predicate = getattr(
        raw_env,
        "_physx_task_rigid_body_is_active",
        None,
    )
    expected_rigid_valid: list[int] = []
    for name in rigid_names:
        if callable(rigid_predicate):
            try:
                active = rigid_predicate(name)
            except Exception as exc:
                raise ValueError(
                    f"Task active-body predicate failed for {name!r}."
                ) from exc
            if not isinstance(active, (bool, np.bool_)):
                raise ValueError(
                    "Task active-body predicate must return a boolean for "
                    f"{name!r}."
                )
            expected_rigid_valid.append(int(active))
        else:
            # With no explicit randomized-variant predicate, every schema row
            # is a required live physics entity.  A zero readback cannot define
            # its own expected coverage.
            expected_rigid_valid.append(1)

    deformable_name = physx.get("deformable_name")
    deformable_shape = (0, 3)
    deformable_valid = np.zeros(0, dtype=np.uint8)
    if deformable_name is not None:
        fields = schema.get("fields")
        if not isinstance(fields, Mapping):
            raise ValueError(
                "PhysX schema fields are required for deformable activity."
            )
        deformable_spec = fields.get("physx_deformable_position")
        if not isinstance(deformable_spec, Mapping):
            raise ValueError("PhysX deformable schema field is missing.")
        raw_shape = deformable_spec.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise ValueError("PhysX deformable schema shape is invalid.")
        try:
            deformable_shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PhysX deformable schema shape is invalid.") from exc
        if deformable_shape[0] <= 0 or deformable_shape[1] != 3:
            raise ValueError("PhysX deformable schema shape is invalid.")
        deformable_valid = np.ones(1, dtype=np.uint8)

    expected_dof_valid = np.zeros(
        (len(articulation_names), max_dof_count),
        dtype=np.uint8,
    )
    for row_index, row in enumerate(dof_rows):
        expected_dof_valid[row_index, : len(row)] = 1
    expected_snapshot = {
        "physx_task_rigid_body_valid": np.asarray(
            expected_rigid_valid,
            dtype=np.uint8,
        ),
        "physx_task_articulation_root_valid": np.ones(
            len(articulation_names),
            dtype=np.uint8,
        ),
        "physx_task_articulation_dof_valid": expected_dof_valid,
        "physx_deformable_position": np.zeros(
            deformable_shape,
            dtype=np.float64,
        ),
        "physx_deformable_valid": deformable_valid,
    }
    return _active_physics_layout(expected_snapshot, schema)


def _validate_active_physics_layout_coverage(
    layout: Mapping[str, Any],
    reset_state_coverage: Mapping[str, Any] | None,
) -> None:
    if not isinstance(reset_state_coverage, Mapping):
        raise ValueError("reset_state_coverage is missing.")
    active_total = 0
    for kind in _ACTIVE_PHYSICS_LAYOUT_KINDS:
        expected = reset_state_coverage.get(kind)
        if type(expected) is not int or expected < 0:
            raise ValueError(
                f"reset_state_coverage[{kind!r}] is missing or invalid."
            )
        section = layout.get(kind)
        if not isinstance(section, Mapping):
            raise ValueError(f"Active physics layout is missing {kind}.")
        actual = sum(int(value) for value in section.get("valid", ()))
        if actual != expected:
            raise ValueError(
                f"Active {kind} count does not match reset coverage: "
                f"active={actual}, coverage={expected}."
            )
        active_total += actual
    if active_total <= 0:
        raise ValueError("Active task physics layout is empty.")


def _non_finite_valid_snapshot_fields(
    snapshot: Mapping[str, Any],
) -> list[str]:
    """Return physical fields whose live-valid coordinates are not finite."""

    errors: list[str] = []
    row_fields = (
        ("physx_robot_root_state", "physx_robot_root_valid", 13),
        ("physx_tcp_state", "physx_tcp_valid", 13),
        (
            "physx_task_rigid_body_state",
            "physx_task_rigid_body_valid",
            13,
        ),
        (
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
            13,
        ),
    )
    for state_name, valid_name, minimum_width in row_fields:
        states = np.asarray(snapshot.get(state_name, np.zeros((0, 13))))
        valid = np.asarray(snapshot.get(valid_name, np.zeros(0)))
        if (
            states.ndim != 2
            or states.shape[1] < minimum_width
            or valid.ndim != 1
            or valid.size != states.shape[0]
            or not np.all((valid == 0) | (valid == 1))
        ):
            errors.append(f"{state_name}:invalid_layout")
            continue
        try:
            active = np.asarray(states[valid.astype(bool)], dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{state_name}:non_numeric")
            continue
        if not np.all(np.isfinite(active)):
            errors.append(f"{state_name}:non_finite")

    matrix_fields = (
        ("physx_robot_joint_position", "physx_robot_joint_valid"),
        ("physx_robot_joint_velocity", "physx_robot_joint_valid"),
        (
            "physx_robot_joint_target",
            "physx_robot_joint_target_valid",
        ),
        (
            "physx_task_articulation_dof_position",
            "physx_task_articulation_dof_valid",
        ),
        (
            "physx_task_articulation_dof_velocity",
            "physx_task_articulation_dof_valid",
        ),
        ("physx_command_hand", "physx_command_hand_valid"),
    )
    for state_name, valid_name in matrix_fields:
        states = np.asarray(snapshot.get(state_name, np.zeros((0, 0))))
        valid = np.asarray(snapshot.get(valid_name, np.zeros((0, 0))))
        if (
            states.ndim != 2
            or valid.shape != states.shape
            or not np.all((valid == 0) | (valid == 1))
        ):
            errors.append(f"{state_name}:invalid_layout")
            continue
        try:
            active = np.asarray(states[valid.astype(bool)], dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{state_name}:non_numeric")
            continue
        if not np.all(np.isfinite(active)):
            errors.append(f"{state_name}:non_finite")

    deformable_valid = np.asarray(
        snapshot.get("physx_deformable_valid", np.zeros(0))
    )
    deformable_position = np.asarray(
        snapshot.get("physx_deformable_position", np.zeros((0, 3)))
    )
    deformable_velocity = np.asarray(
        snapshot.get("physx_deformable_velocity", np.zeros((0, 3)))
    )
    if (
        deformable_valid.ndim != 1
        or deformable_valid.size not in {0, 1}
        or not np.all((deformable_valid == 0) | (deformable_valid == 1))
        or deformable_position.ndim != 2
        or deformable_position.shape[1:] != (3,)
        or deformable_velocity.shape != deformable_position.shape
    ):
        errors.append("physx_deformable:invalid_layout")
    elif deformable_valid.size == 0:
        if deformable_position.shape[0] != 0:
            errors.append("physx_deformable:unexpected_rows")
    elif bool(deformable_valid[0]) and not (
        np.all(np.isfinite(deformable_position))
        and np.all(np.isfinite(deformable_velocity))
    ):
        errors.append("physx_deformable:non_finite")
    return errors


def _reset_state_drifts(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    trace: PhysxGroundTruthTrace,
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    """Return authored-reset drift for every observable physical coordinate.

    The fixed trace layout is part of the replay contract.  A validity change
    or malformed pose therefore invalidates the whole comparison instead of
    letting two different bodies be paired after filtering.
    """

    drifts: list[dict[str, Any]] = []
    coverage: dict[str, int] = {}
    pose_layouts = (
        (
            "rigid_body",
            getattr(trace, "rigid_bodies", ()),
            "physx_task_rigid_body_state",
            "physx_task_rigid_body_valid",
        ),
        (
            "articulation_root",
            getattr(trace, "articulations", ()),
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        ),
        (
            "robot_root",
            getattr(trace, "robots", ()),
            "physx_robot_root_state",
            "physx_robot_root_valid",
        ),
        (
            "tcp",
            getattr(trace, "robots", ()),
            "physx_tcp_state",
            "physx_tcp_valid",
        ),
    )
    for kind, entries, state_name, valid_name in pose_layouts:
        before = np.asarray(previous.get(state_name, np.zeros((0, 13))))
        after = np.asarray(current.get(state_name, np.zeros((0, 13))))
        before_valid = np.asarray(
            previous.get(valid_name, np.zeros(0))
        ).reshape(-1)
        after_valid = np.asarray(
            current.get(valid_name, np.zeros(0))
        ).reshape(-1)
        if (
            before.ndim != 2
            or after.ndim != 2
            or before.shape != after.shape
            or before.shape[-1] < 7
            or before_valid.shape != after_valid.shape
            or before_valid.size != before.shape[0]
            or len(entries) != before.shape[0]
            or not np.array_equal(before_valid, after_valid)
        ):
            return None
        coverage[kind] = int(np.count_nonzero(before_valid))
        for index in np.flatnonzero(before_valid.astype(bool)):
            before_pose = np.asarray(before[index, :7], dtype=np.float64)
            after_pose = np.asarray(after[index, :7], dtype=np.float64)
            if not (
                np.all(np.isfinite(before_pose))
                and np.all(np.isfinite(after_pose))
            ):
                return None
            rotation = _quaternion_rotation_delta(
                before_pose[3:7],
                after_pose[3:7],
            )
            if rotation is None:
                return None
            drifts.append(
                {
                    "kind": kind,
                    "index": int(index),
                    "name": str(
                        getattr(
                            entries[index],
                            "name",
                            getattr(entries[index], "side", index),
                        )
                    ),
                    "translation_m": float(
                        np.linalg.norm(after_pose[:3] - before_pose[:3])
                    ),
                    "before_position": before_pose[:3].tolist(),
                    "after_position": after_pose[:3].tolist(),
                    "rotation_rad": rotation,
                }
            )

    coordinate_layouts = (
        (
            "articulation_dof",
            "physx_task_articulation_dof_position",
            "physx_task_articulation_dof_valid",
        ),
        (
            "robot_joint",
            "physx_robot_joint_position",
            "physx_robot_joint_valid",
        ),
        (
            "hand_command",
            "physx_command_hand",
            "physx_command_hand_valid",
        ),
    )
    for kind, state_name, valid_name in coordinate_layouts:
        before = np.asarray(previous.get(state_name, np.zeros((0, 0))))
        after = np.asarray(current.get(state_name, np.zeros((0, 0))))
        before_valid = np.asarray(previous.get(valid_name, np.zeros((0, 0))))
        after_valid = np.asarray(current.get(valid_name, np.zeros((0, 0))))
        if (
            before.ndim != 2
            or after.shape != before.shape
            or before_valid.shape != before.shape
            or after_valid.shape != before.shape
            or not np.array_equal(before_valid, after_valid)
        ):
            return None
        mask = before_valid.astype(bool)
        coverage[kind] = int(np.count_nonzero(mask))
        if not np.all(np.isfinite(before[mask])) or not np.all(
            np.isfinite(after[mask])
        ):
            return None
        for row, column in np.argwhere(mask):
            if kind == "articulation_dof":
                entries = getattr(trace, "articulations", ())
                if row >= len(entries) or column >= len(entries[row].dof_names):
                    return None
                name = f"{entries[row].name}:{entries[row].dof_names[column]}"
            else:
                entries = getattr(trace, "robots", ())
                if row >= len(entries):
                    return None
                if kind == "robot_joint":
                    if column >= len(entries[row].joint_names):
                        return None
                    name = f"{entries[row].side}:{entries[row].joint_names[column]}"
                else:
                    name = f"{entries[row].side}:hand_command_{column}"
            delta = float(abs(after[row, column] - before[row, column]))
            drift = {
                "kind": kind,
                "index": [int(row), int(column)],
                "name": str(name),
            }
            if kind in {"robot_joint", "hand_command"} or (
                kind == "articulation_dof"
                and str(name).rsplit(":", 1)[-1]
                in _KNOWN_REVOLUTE_TASK_DOF_NAMES
            ):
                # Every supported ROKAE arm and Wuji hand coordinate is a
                # revolute joint angle, and the audited task joints above are
                # revolute too.  Classify them as angular drift so they are
                # checked against the declared radian limit; comparing them
                # with min(metres, radians) is dimensionally invalid and
                # caused physically quiet resets to be rejected.
                drift["rotation_rad"] = delta
            else:
                # Unknown task articulation DOFs remain fail-closed under the
                # stricter generic coordinate bound until their PhysX DOF type
                # is carried into the public trace schema.
                drift["coordinate"] = delta
            drifts.append(drift)

    before_deformable = np.asarray(
        previous.get("physx_deformable_position", np.zeros((0, 3)))
    )
    after_deformable = np.asarray(
        current.get("physx_deformable_position", np.zeros((0, 3)))
    )
    before_deformable_valid = np.asarray(
        previous.get("physx_deformable_valid", np.zeros(0))
    ).reshape(-1)
    after_deformable_valid = np.asarray(
        current.get("physx_deformable_valid", np.zeros(0))
    ).reshape(-1)
    if (
        before_deformable.shape != after_deformable.shape
        or before_deformable.ndim != 2
        or before_deformable.shape[-1] != 3
        or before_deformable_valid.shape != after_deformable_valid.shape
        or before_deformable_valid.size not in {0, 1}
        or not np.array_equal(
            before_deformable_valid,
            after_deformable_valid,
        )
    ):
        return None
    deformable_is_valid = bool(
        before_deformable_valid.size == 1
        and before_deformable_valid[0]
    )
    coverage["deformable_node"] = (
        int(before_deformable.shape[0]) if deformable_is_valid else 0
    )
    if deformable_is_valid:
        if not np.all(np.isfinite(before_deformable)) or not np.all(
            np.isfinite(after_deformable)
        ):
            return None
        for index, distance in enumerate(
            np.linalg.norm(after_deformable - before_deformable, axis=1)
        ):
            drifts.append(
                {
                    "kind": "deformable_node",
                    "index": int(index),
                    "name": f"deformable_node_{index}",
                    "translation_m": float(distance),
                }
            )

    if not drifts:
        return None
    return sorted(
        drifts,
        key=lambda item: max(
            float(item.get("translation_m", 0.0))
            / DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT,
            float(item.get("rotation_rad", 0.0))
            / DEFAULT_RESET_ROTATION_DRIFT_LIMIT,
            float(item.get("coordinate", 0.0))
            / min(
                DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT,
                DEFAULT_RESET_ROTATION_DRIFT_LIMIT,
            ),
        ),
        reverse=True,
    ), coverage


def _coordinate_arrays_equal_within_ulps(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    max_ulps: int = 4,
) -> bool:
    """Compare coordinate readbacks without rejecting native-float re-rounding."""

    if np.array_equal(previous, current):
        return True
    if previous.shape != current.shape:
        return False
    common_dtype = np.result_type(previous.dtype, current.dtype)
    if not np.issubdtype(common_dtype, np.floating):
        return False
    before = np.asarray(previous, dtype=common_dtype)
    after = np.asarray(current, dtype=common_dtype)
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        return False
    difference = np.abs(after - before)
    native_ulp_tolerance = float(max_ulps) * np.maximum(
        np.abs(np.spacing(before)),
        np.abs(np.spacing(after)),
    )
    # Unit quaternions commonly contain components mathematically equal to
    # zero.  Re-normalizing the same float32 pose can move such a component by
    # a small fraction of epsilon, far more than spacing(0) but still below one
    # unit-scale representable step.  Bound that case by the source dtype's
    # epsilon rather than by an application-level tolerance.  The recording
    # schema promotes PhysX tensor values to float64 for stable storage, while
    # the public simulator readbacks themselves are float32, so retain that
    # source-precision floor after promotion.
    source_epsilon = max(
        float(np.finfo(common_dtype).eps),
        float(np.finfo(np.float32).eps),
    )
    unit_scale_tolerance = (
        float(max_ulps)
        * source_epsilon
        * np.maximum.reduce(
            [
                np.ones(before.shape, dtype=common_dtype),
                np.abs(before),
                np.abs(after),
            ]
        )
    )
    tolerance = np.maximum(native_ulp_tolerance, unit_scale_tolerance)
    return bool(np.all(difference <= tolerance))


def _full_coordinate_exact_mismatches(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str] | None:
    """Compare every public physical coordinate at a no-step boundary."""

    mismatches: list[str] = []
    row_fields = (
        ("physx_task_rigid_body_state", "physx_task_rigid_body_valid"),
        (
            "physx_task_articulation_root_state",
            "physx_task_articulation_root_valid",
        ),
        ("physx_robot_root_state", "physx_robot_root_valid"),
        ("physx_tcp_state", "physx_tcp_valid"),
    )
    for state_name, valid_name in row_fields:
        before = np.asarray(previous.get(state_name, np.zeros((0, 13))))
        after = np.asarray(current.get(state_name, np.zeros((0, 13))))
        before_valid = np.asarray(
            previous.get(valid_name, np.zeros(0))
        )
        after_valid = np.asarray(current.get(valid_name, np.zeros(0)))
        if (
            before.ndim != 2
            or before.shape[1] < 7
            or after.shape != before.shape
            or before_valid.ndim != 1
            or after_valid.shape != before_valid.shape
            or before_valid.size != before.shape[0]
            or not np.array_equal(before_valid, after_valid)
            or not np.all((before_valid == 0) | (before_valid == 1))
        ):
            return None
        active = before_valid.astype(bool)
        if not _coordinate_arrays_equal_within_ulps(
            before[active, :7],
            after[active, :7],
        ):
            mismatches.append(state_name)

    matrix_fields = (
        (
            "physx_task_articulation_dof_position",
            "physx_task_articulation_dof_valid",
        ),
        ("physx_robot_joint_position", "physx_robot_joint_valid"),
    )
    for state_name, valid_name in matrix_fields:
        before = np.asarray(previous.get(state_name, np.zeros((0, 0))))
        after = np.asarray(current.get(state_name, np.zeros((0, 0))))
        before_valid = np.asarray(
            previous.get(valid_name, np.zeros((0, 0)))
        )
        after_valid = np.asarray(
            current.get(valid_name, np.zeros((0, 0)))
        )
        if (
            before.ndim != 2
            or after.shape != before.shape
            or before_valid.shape != before.shape
            or after_valid.shape != before.shape
            or not np.array_equal(before_valid, after_valid)
            or not np.all((before_valid == 0) | (before_valid == 1))
        ):
            return None
        active = before_valid.astype(bool)
        if not _coordinate_arrays_equal_within_ulps(
            before[active],
            after[active],
        ):
            mismatches.append(state_name)

    before_position = np.asarray(
        previous.get("physx_deformable_position", np.zeros((0, 3)))
    )
    after_position = np.asarray(
        current.get("physx_deformable_position", np.zeros((0, 3)))
    )
    before_valid = np.asarray(
        previous.get("physx_deformable_valid", np.zeros(0))
    )
    after_valid = np.asarray(
        current.get("physx_deformable_valid", np.zeros(0))
    )
    if (
        before_position.ndim != 2
        or before_position.shape[1:] != (3,)
        or after_position.shape != before_position.shape
        or before_valid.ndim != 1
        or before_valid.size not in {0, 1}
        or after_valid.shape != before_valid.shape
        or not np.array_equal(before_valid, after_valid)
        or not np.all((before_valid == 0) | (before_valid == 1))
    ):
        return None
    if (
        before_valid.size == 1
        and bool(before_valid[0])
        and not _coordinate_arrays_equal_within_ulps(
            before_position,
            after_position,
        )
    ):
        mismatches.append("physx_deformable_position")
    return mismatches


def _new_reset_drift_history() -> dict[str, Any]:
    return {
        "valid": True,
        "sample_count": 0,
        "invalid_sample_count": 0,
        "coverage": None,
        "maximum_translation_m": 0.0,
        "maximum_rigid_translation_m": 0.0,
        "maximum_deformable_node_translation_m": 0.0,
        "maximum_rotation_rad": 0.0,
        "maximum_coordinate": 0.0,
        "peaks": [],
    }


def _reset_translation_drift_exceeded(
    drifts: Iterable[Mapping[str, Any]],
    *,
    rigid_limit_m: float,
    deformable_node_limit_m: float | None,
) -> bool:
    """Apply a soft-node allowance without relaxing any rigid-body bound."""

    for item in drifts:
        translation = float(item.get("translation_m", 0.0))
        limit = (
            float(deformable_node_limit_m)
            if item.get("kind") == "deformable_node"
            and deformable_node_limit_m is not None
            else float(rigid_limit_m)
        )
        if translation > limit:
            return True
    return False


def _accumulate_reset_state_drift(
    history: dict[str, Any],
    authored: Mapping[str, Any],
    current: Mapping[str, Any],
    trace: PhysxGroundTruthTrace,
) -> None:
    """Accumulate authored-reset drift without allowing return-to-origin."""

    sample_index = int(history["sample_count"])
    history["sample_count"] = sample_index + 1
    result = _reset_state_drifts(authored, current, trace)
    if result is None:
        history["valid"] = False
        history["invalid_sample_count"] = int(
            history["invalid_sample_count"]
        ) + 1
        return
    drifts, coverage = result
    if history["coverage"] is None:
        history["coverage"] = dict(coverage)
    elif history["coverage"] != coverage:
        history["valid"] = False
        history["invalid_sample_count"] = int(
            history["invalid_sample_count"]
        ) + 1
        return
    history["maximum_translation_m"] = max(
        float(history["maximum_translation_m"]),
        max(
            (float(item.get("translation_m", 0.0)) for item in drifts),
            default=0.0,
        ),
    )
    history["maximum_rigid_translation_m"] = max(
        float(history["maximum_rigid_translation_m"]),
        max(
            (
                float(item.get("translation_m", 0.0))
                for item in drifts
                if item.get("kind") != "deformable_node"
            ),
            default=0.0,
        ),
    )
    history["maximum_deformable_node_translation_m"] = max(
        float(history["maximum_deformable_node_translation_m"]),
        max(
            (
                float(item.get("translation_m", 0.0))
                for item in drifts
                if item.get("kind") == "deformable_node"
            ),
            default=0.0,
        ),
    )
    history["maximum_rotation_rad"] = max(
        float(history["maximum_rotation_rad"]),
        max(
            (float(item.get("rotation_rad", 0.0)) for item in drifts),
            default=0.0,
        ),
    )
    history["maximum_coordinate"] = max(
        float(history["maximum_coordinate"]),
        max(
            (float(item.get("coordinate", 0.0)) for item in drifts),
            default=0.0,
        ),
    )
    candidates = list(history["peaks"])
    candidates.extend(
        {**item, "sample_index": sample_index} for item in drifts[:5]
    )
    candidates.sort(
        key=lambda item: max(
            float(item.get("translation_m", 0.0))
            / DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT,
            float(item.get("rotation_rad", 0.0))
            / DEFAULT_RESET_ROTATION_DRIFT_LIMIT,
            float(item.get("coordinate", 0.0))
            / min(
                DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT,
                DEFAULT_RESET_ROTATION_DRIFT_LIMIT,
            ),
        ),
        reverse=True,
    )
    history["peaks"] = candidates[:5]


def _contact_penetration_summary(
    snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate raw contacts and bind them to the canonical path signature."""

    result = {
        "valid": False,
        "capture_source": 0,
        "contact_header_count": 0,
        "contact_point_count": 0,
        "friction_anchor_count": 0,
        "maximum_penetration_depth_m": None,
        "deepest_contact_point_index": None,
        "canonical_contact_signature": None,
        "error": None,
    }

    class _InvalidContact(ValueError):
        pass

    def scalar_integer(name: str) -> int:
        values = np.asarray(snapshot.get(name, np.zeros(0)))
        if values.shape != (1,):
            raise _InvalidContact(f"{name}:missing_or_invalid_shape")
        try:
            numeric = float(values[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise _InvalidContact(f"{name}:non_numeric") from exc
        if (
            not np.isfinite(numeric)
            or numeric < 0.0
            or not numeric.is_integer()
        ):
            raise _InvalidContact(f"{name}:invalid_integer")
        return int(numeric)

    def active_numeric_rows(
        name: str,
        *,
        count: int,
        minimum_width: int,
    ) -> np.ndarray:
        values = np.asarray(snapshot.get(name, np.zeros((0, 0))))
        if (
            values.ndim != 2
            or values.shape[1] < minimum_width
            or count > values.shape[0]
        ):
            raise _InvalidContact(f"{name}:invalid_layout")
        try:
            active = np.asarray(values[:count], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _InvalidContact(f"{name}:non_numeric") from exc
        if not np.all(np.isfinite(active)):
            raise _InvalidContact(f"{name}:non_finite")
        return active

    def active_integer_rows(
        name: str,
        *,
        count: int,
        minimum_width: int,
    ) -> np.ndarray:
        values = np.asarray(snapshot.get(name, np.zeros((0, 0))))
        if (
            values.ndim != 2
            or values.shape[1] < minimum_width
            or count > values.shape[0]
        ):
            raise _InvalidContact(f"{name}:invalid_layout")
        active = values[:count]
        if np.issubdtype(active.dtype, np.integer):
            return active
        try:
            numeric = np.asarray(active, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _InvalidContact(f"{name}:non_numeric") from exc
        if not np.all(np.isfinite(numeric)) or not np.all(
            numeric == np.floor(numeric)
        ):
            raise _InvalidContact(f"{name}:invalid_integer")
        return numeric

    try:
        report_valid = scalar_integer("physx_contact_report_valid")
        source = scalar_integer("physx_contact_capture_source")
        header_count = scalar_integer("physx_contact_header_count")
        point_count = scalar_integer("physx_contact_point_count")
        anchor_count = scalar_integer("physx_friction_anchor_count")
        signature_valid = scalar_integer("physx_contact_signature_valid")
        signature_pair_count = scalar_integer(
            "physx_contact_signature_pair_count"
        )
        result.update(
            {
                "capture_source": source,
                "contact_header_count": header_count,
                "contact_point_count": point_count,
                "friction_anchor_count": anchor_count,
            }
        )
        if report_valid != 1 or source not in {1, 2}:
            raise _InvalidContact("contact_report_not_valid")
        if signature_valid != 1:
            raise _InvalidContact("canonical_contact_signature_invalid")

        physx = schema.get("physx") if isinstance(schema, Mapping) else None
        contact_schema = (
            physx.get("contact_report")
            if isinstance(physx, Mapping)
            else None
        )
        path_table = (
            contact_schema.get("path_table")
            if isinstance(contact_schema, Mapping)
            else None
        )
        if not isinstance(path_table, (list, tuple)) or not path_table:
            raise _InvalidContact("contact_path_table_missing")
        id_to_path: dict[int, str] = {}
        path_to_id: dict[str, int] = {}
        for entry in path_table:
            if not isinstance(entry, Mapping):
                raise _InvalidContact("contact_path_table_invalid")
            try:
                # ``PhysicsSchemaTools.sdfPathToInt`` is exposed as a signed
                # process-local integer on some Isaac Sim builds, while the
                # recorded contact header contract stores every id as
                # ``uint64``.  Resolve both representations through the same
                # two's-complement bit pattern.  The trace canonicalizer uses
                # this exact normalization as well; omitting it here makes a
                # complete public contact report fail whenever a path hash has
                # its high bit set.
                process_id = int(entry["id"]) & np.iinfo(np.uint64).max
                path = str(entry["path"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise _InvalidContact("contact_path_table_invalid") from exc
            if (
                not path
                or process_id in id_to_path
                or path in path_to_id
            ):
                raise _InvalidContact("contact_path_table_not_unique")
            id_to_path[process_id] = path
            path_to_id[path] = process_id
        canonical_paths = tuple(sorted(path_to_id))
        canonical_path_index = {
            path: index + 1 for index, path in enumerate(canonical_paths)
        }

        header_ids = active_integer_rows(
            "physx_contact_header_ids",
            count=header_count,
            minimum_width=5,
        )
        header_meta = active_integer_rows(
            "physx_contact_header_meta",
            count=header_count,
            minimum_width=7,
        )
        points = active_numeric_rows(
            "physx_contact_point_data",
            count=point_count,
            minimum_width=10,
        )
        active_integer_rows(
            "physx_contact_point_face_indices",
            count=point_count,
            minimum_width=2,
        )
        active_integer_rows(
            "physx_contact_point_material_ids",
            count=point_count,
            minimum_width=2,
        )
        anchors = active_numeric_rows(
            "physx_friction_anchor_data",
            count=anchor_count,
            minimum_width=6,
        )
        point_coverage = np.zeros(point_count, dtype=np.uint8)
        anchor_coverage = np.zeros(anchor_count, dtype=np.uint8)
        aggregates: dict[
            tuple[tuple[str, str], tuple[str, str]],
            dict[str, Any],
        ] = {}
        for header_index in range(header_count):
            contact_offset = int(header_meta[header_index, 1])
            contact_rows = int(header_meta[header_index, 2])
            friction_offset = int(header_meta[header_index, 3])
            friction_rows = int(header_meta[header_index, 4])
            if (
                contact_offset < 0
                or contact_rows < 0
                or contact_offset + contact_rows > point_count
                or friction_offset < 0
                or friction_rows < 0
                or friction_offset + friction_rows > anchor_count
            ):
                raise _InvalidContact("contact_header_range_invalid")
            point_coverage[
                contact_offset : contact_offset + contact_rows
            ] += 1
            anchor_coverage[
                friction_offset : friction_offset + friction_rows
            ] += 1
            if contact_rows <= 0 and friction_rows <= 0:
                continue
            raw_ids = [int(value) for value in header_ids[header_index, :4]]
            if any(value not in id_to_path for value in raw_ids):
                raise _InvalidContact("contact_header_path_unresolved")
            actor0, actor1, collider0, collider1 = (
                id_to_path[value] for value in raw_ids
            )
            endpoint_pair = tuple(
                sorted(((actor0, collider0), (actor1, collider1)))
            )
            contact_payload = points[
                contact_offset : contact_offset + contact_rows
            ]
            contact_point_indices = np.arange(
                contact_offset,
                contact_offset + contact_rows,
                dtype=np.int64,
            )
            if source == 2 and contact_payload.shape[0]:
                # Match PhysxGroundTruthTrace's canonical signature builder:
                # RigidContactView may include exact all-zero padding inside
                # its declared active slice.  Keep raw-row coverage strict but
                # exclude only that exact ten-value padding payload from the
                # physical contact aggregate.
                padding_rows = np.all(contact_payload[:, :10] == 0.0, axis=1)
                contact_payload = contact_payload[~padding_rows]
                contact_point_indices = contact_point_indices[~padding_rows]
            if not contact_payload.shape[0] and friction_rows <= 0:
                continue
            aggregate = aggregates.setdefault(
                endpoint_pair,
                {
                    "contact_count": 0,
                    "friction_count": 0,
                    "penetrations": [],
                    "normal_impulses": [],
                    "friction_impulses": [],
                },
            )
            aggregate["contact_count"] += int(contact_payload.shape[0])
            aggregate["friction_count"] += friction_rows
            for point_index, row in zip(
                contact_point_indices,
                contact_payload,
                strict=True,
            ):
                # Recompute with the exact float64 arithmetic used by
                # PhysxGroundTruthTrace._canonicalize_contact_fields.  The raw
                # tensor rows are float32; leaving either operand at that dtype
                # makes an exact self-consistency check fail spuriously.
                normal = np.asarray(row[3:6], dtype=np.float64)
                impulse = np.asarray(row[6:9], dtype=np.float64)
                separation = float(row[9])
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm <= 1.0e-12:
                    raise _InvalidContact(
                        "contact_normal_invalid:"
                        f"point={point_index},"
                        f"actors=({actor0},{actor1}),"
                        f"colliders=({collider0},{collider1}),"
                        f"impulse={impulse.tolist()},"
                        f"separation={separation:.9g}"
                    )
                aggregate["penetrations"].append(
                    max(0.0, -separation)
                )
                aggregate["normal_impulses"].append(
                    abs(float(np.dot(impulse, normal / normal_norm)))
                )
            for row in anchors[
                friction_offset : friction_offset + friction_rows
            ]:
                friction_impulse = np.asarray(
                    row[3:6], dtype=np.float64
                )
                aggregate["friction_impulses"].append(
                    float(np.linalg.norm(friction_impulse))
                )
        if not np.all(point_coverage == 1) or not np.all(
            anchor_coverage == 1
        ):
            raise _InvalidContact("contact_rows_not_exactly_covered")

        ordered_pairs = sorted(aggregates)
        expected_paths: list[list[int]] = []
        expected_counts: list[list[int]] = []
        expected_metrics: list[list[float]] = []
        for pair in ordered_pairs:
            (actor0, collider0), (actor1, collider1) = pair
            aggregate = aggregates[pair]
            expected_paths.append(
                [
                    canonical_path_index[actor0],
                    canonical_path_index[collider0],
                    canonical_path_index[actor1],
                    canonical_path_index[collider1],
                ]
            )
            expected_counts.append(
                [
                    int(aggregate["contact_count"]),
                    int(aggregate["friction_count"]),
                ]
            )
            expected_metrics.append(
                [
                    max(aggregate["penetrations"], default=0.0),
                    math.fsum(sorted(aggregate["normal_impulses"])),
                    math.fsum(sorted(aggregate["friction_impulses"])),
                ]
            )
        if signature_pair_count != len(ordered_pairs):
            raise _InvalidContact("canonical_contact_pair_count_mismatch")
        signature_paths = active_integer_rows(
            "physx_contact_signature_pair_path_indices",
            count=signature_pair_count,
            minimum_width=4,
        )[:, :4]
        signature_counts = active_integer_rows(
            "physx_contact_signature_pair_counts",
            count=signature_pair_count,
            minimum_width=2,
        )[:, :2]
        signature_metrics = active_numeric_rows(
            "physx_contact_signature_pair_metrics",
            count=signature_pair_count,
            minimum_width=3,
        )[:, :3]
        expected_path_array = np.asarray(expected_paths, dtype=np.float64).reshape(
            signature_pair_count,
            4,
        )
        expected_count_array = np.asarray(
            expected_counts,
            dtype=np.float64,
        ).reshape(signature_pair_count, 2)
        expected_metric_array = np.asarray(
            expected_metrics,
            dtype=np.float64,
        ).reshape(signature_pair_count, 3)
        if not np.array_equal(signature_paths, expected_path_array):
            raise _InvalidContact("canonical_contact_paths_mismatch")
        if not np.array_equal(signature_counts, expected_count_array):
            raise _InvalidContact("canonical_contact_counts_mismatch")
        if not np.all(np.isfinite(expected_metric_array)):
            raise _InvalidContact(
                "canonical_contact_recomputed_metrics_non_finite"
            )
        # Both arrays are independently aggregated from the same float32 PhysX
        # payload.  BLAS/vectorized dot products may differ from the producer's
        # scalar loop by one final float64 rounding bit even though every source
        # value, path, and count is identical.  Permit at most four ULPs here;
        # this remains orders of magnitude stricter than any physical tolerance
        # and still rejects altered contact evidence.
        metric_difference = np.abs(signature_metrics - expected_metric_array)
        metric_ulp_tolerance = 4.0 * np.maximum(
            np.spacing(np.abs(signature_metrics)),
            np.spacing(np.abs(expected_metric_array)),
        )
        if not np.all(metric_difference <= metric_ulp_tolerance):
            raise _InvalidContact(
                "canonical_contact_metrics_mismatch:"
                f"max_abs={float(np.max(metric_difference, initial=0.0)):.9g},"
                f"actual={signature_metrics.tolist()},"
                f"recomputed={expected_metric_array.tolist()},"
                f"absolute_difference={metric_difference.tolist()},"
                f"ulp_tolerance={metric_ulp_tolerance.tolist()}"
            )

        separation = points[:, 9]
        deepest_index = int(np.argmin(separation)) if point_count else None
        maximum_penetration = (
            max(0.0, -float(separation[deepest_index]))
            if deepest_index is not None
            else 0.0
        )
        canonical_signature = {
            "pair_count": signature_pair_count,
            "pair_path_indices": expected_paths,
            "pair_counts": expected_counts,
            "pair_metrics": expected_metrics,
        }
        result.update(
            {
                "valid": True,
                "maximum_penetration_depth_m": maximum_penetration,
                "deepest_contact_point_index": deepest_index,
                "canonical_contact_signature": canonical_signature,
            }
        )
    except _InvalidContact as exc:
        result["error"] = str(exc)
    return result


def _new_contact_penetration_history() -> dict[str, Any]:
    return {
        "valid": True,
        "sample_count": 0,
        "invalid_sample_count": 0,
        "capture_sources": [],
        "maximum_contact_point_count": 0,
        "maximum_penetration_depth_m": 0.0,
        "deepest_sample_index": None,
        "deepest_contact_point_index": None,
        "error": None,
    }


def _accumulate_contact_penetration(
    history: dict[str, Any],
    snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume one post-step buffer into initialization-wide evidence."""

    summary = _contact_penetration_summary(snapshot, schema)
    sample_index = int(history["sample_count"])
    history["sample_count"] = sample_index + 1
    source = int(summary["capture_source"])
    sources = set(int(value) for value in history["capture_sources"])
    if source in {1, 2}:
        sources.add(source)
    history["capture_sources"] = sorted(sources)
    history["maximum_contact_point_count"] = max(
        int(history["maximum_contact_point_count"]),
        int(summary["contact_point_count"]),
    )
    if summary["valid"] is not True:
        history["valid"] = False
        history["invalid_sample_count"] = int(
            history["invalid_sample_count"]
        ) + 1
        if history["error"] is None:
            history["error"] = (
                summary["error"] or "invalid_contact_report"
            )
        return summary
    penetration = float(summary["maximum_penetration_depth_m"])
    if penetration > float(history["maximum_penetration_depth_m"]):
        history["maximum_penetration_depth_m"] = penetration
        history["deepest_sample_index"] = sample_index
        history["deepest_contact_point_index"] = summary[
            "deepest_contact_point_index"
        ]
    return summary


def _consume_authored_reset_snapshot(
    raw_env: Any,
) -> tuple[Mapping[str, Any] | None, int | None]:
    snapshot = getattr(
        raw_env,
        "_recording_origin_authored_reset_snapshot",
        None,
    )
    serial = getattr(
        raw_env,
        "_recording_origin_authored_reset_serial",
        None,
    )
    if hasattr(raw_env, "_recording_origin_authored_reset_snapshot"):
        setattr(raw_env, "_recording_origin_authored_reset_snapshot", None)
    if not isinstance(snapshot, Mapping):
        return None, None
    if type(serial) is not int or serial <= 0:
        return None, None
    return snapshot, serial


def _robot_motion(
    snapshot: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    root_states = _valid_states(
        snapshot,
        "physx_robot_root_state",
        "physx_robot_root_valid",
    )
    if root_states is None:
        return None
    linear_speeds = [
        float(np.linalg.norm(state[7:10]))
        for state in root_states
    ]
    angular_speeds = [
        float(np.linalg.norm(state[10:13]))
        for state in root_states
    ]

    joint_velocity = np.asarray(
        snapshot.get("physx_robot_joint_velocity", np.zeros((0, 0))),
        dtype=np.float64,
    )
    joint_valid = np.asarray(
        snapshot.get("physx_robot_joint_valid", np.zeros((0, 0)))
    )
    joint_speeds: list[float] = []
    if (
        joint_velocity.shape != joint_valid.shape
        or joint_valid.ndim != 2
        or not np.all((joint_valid == 0) | (joint_valid == 1))
    ):
        return None
    valid_velocity = joint_velocity[joint_valid.astype(bool)]
    if not np.all(np.isfinite(valid_velocity)):
        return None
    if valid_velocity.size:
        joint_speeds.append(float(np.max(np.abs(valid_velocity))))

    if not linear_speeds and not angular_speeds and not joint_speeds:
        return None
    return (
        max(linear_speeds, default=0.0),
        max(angular_speeds, default=0.0),
        max(joint_speeds, default=0.0),
    )


def _robot_joint_velocity_peaks(
    snapshot: Mapping[str, Any],
    trace: PhysxGroundTruthTrace,
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    velocity = np.asarray(
        snapshot.get("physx_robot_joint_velocity", np.zeros((0, 0))),
        dtype=np.float64,
    )
    valid = np.asarray(
        snapshot.get("physx_robot_joint_valid", np.zeros((0, 0)))
    ).astype(bool)
    peaks: list[dict[str, Any]] = []
    if velocity.shape != valid.shape:
        return peaks
    for side_index, entry in enumerate(trace.robots):
        if side_index >= velocity.shape[0]:
            continue
        for joint_index, joint_name in enumerate(entry.joint_names):
            if (
                joint_index >= velocity.shape[1]
                or not valid[side_index, joint_index]
                or not np.isfinite(velocity[side_index, joint_index])
            ):
                continue
            peaks.append(
                {
                    "side": entry.side,
                    "joint": joint_name,
                    "velocity": float(velocity[side_index, joint_index]),
                }
            )
    peaks.sort(key=lambda item: abs(float(item["velocity"])), reverse=True)
    return peaks[: max(0, int(count))]


def _rebuild_observation(env: Any, trace: PhysxGroundTruthTrace) -> Any:
    raw_env = getattr(env, "unwrapped", env)
    refresh = getattr(raw_env, "_refresh_object_poses", None)
    if callable(refresh):
        refresh()
    observation = raw_env._compute_observation()
    attach = getattr(raw_env, "_attach_physx_ground_truth", None)
    if callable(attach):
        observation = attach(observation)
    transform = getattr(env, "observation", None)
    if callable(transform):
        observation = transform(observation)
    trace.begin_substep_capture()
    return observation


def _zero_velocity_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    canonical = {
        name: np.asarray(value).copy()
        for name, value in snapshot.items()
    }
    for name in (
        "physx_robot_root_state",
        "physx_task_rigid_body_state",
        "physx_task_articulation_root_state",
    ):
        value = canonical.get(name)
        if value is not None and value.ndim >= 1 and value.shape[-1] >= 13:
            value[..., 7:13] = 0.0
    for name in (
        "physx_robot_joint_velocity",
        "physx_task_articulation_dof_velocity",
        "physx_deformable_velocity",
    ):
        value = canonical.get(name)
        if value is not None:
            value[...] = 0.0
    return canonical


def _write_zero_velocity_boundary(
    trace: PhysxGroundTruthTrace,
    *,
    sync: bool = True,
) -> dict[str, int]:
    return trace.write_zero_velocity_boundary(
        trace.snapshot(),
        trace.schema(),
        sync=sync,
    )


def _task_recording_origin_readiness(
    raw_env: Any,
) -> tuple[bool, bool, dict[str, Any] | None]:
    """Read one optional task-specific, write-free origin readiness sample."""

    callback = getattr(
        raw_env,
        "stable_recording_origin_task_readiness",
        None,
    )
    if not callable(callback):
        return False, True, None
    result = callback()
    if result is None:
        # Shared environment classes may expose the hook for only one of their
        # task names.  ``None`` means that this particular task has no extra
        # readiness contract.
        return False, True, None
    if not isinstance(result, Mapping):
        raise TypeError(
            "stable_recording_origin_task_readiness() must return a mapping "
            f"or None; got {type(result).__name__}."
        )
    payload = dict(result)
    ready = payload.get("ready")
    if type(ready) is not bool:
        raise TypeError(
            "stable_recording_origin_task_readiness() must return an exact "
            "boolean 'ready' field."
        )
    return True, ready, payload


def _capture_robot_hold_targets(raw_env: Any) -> dict[str, Any]:
    """Capture one fixed post-reset joint target for each available robot."""

    robots = getattr(raw_env, "_robots", None)
    if not isinstance(robots, Mapping):
        robots = {}
    targets: dict[str, Any] = {}
    for side in ("right", "left"):
        robot = robots.get(side)
        if robot is None:
            control_robot = getattr(raw_env, "_control_robot", None)
            if callable(control_robot):
                robot = control_robot(side)
        if robot is not None:
            targets[side] = robot.data.joint_pos.clone()
    return targets


def _restore_default_robot_hold_targets(
    raw_env: Any,
    *,
    sync: bool,
) -> tuple[dict[str, Any], str]:
    """Restore the authored robot home and return its synchronized targets.

    Native Isaac Lab environments expose an atomic reset helper that writes
    default joint positions, exact-zero velocities, actuator drive targets,
    and Python-side command buffers together.  Older/interface-only fixtures
    fall back to their live readback for compatibility.
    """

    enabled = bool(
        getattr(
            raw_env,
            "stable_recording_origin_default_robot_restore_enabled",
            False,
        )
    )
    restore = getattr(raw_env, "_restore_default_robot_joint_state", None)
    if not enabled or not callable(restore):
        return _capture_robot_hold_targets(raw_env), "live_readback_fallback"

    restored = restore(reset_controllers=True, sync=sync)
    if not isinstance(restored, Mapping):
        raise RuntimeError(
            "Stable recording origin default robot restore must return targets."
        )
    targets = {
        str(side): target.clone()
        for side, target in restored.items()
        if target is not None
    }
    if not targets:
        raise RuntimeError(
            "Stable recording origin found no default robot joint targets."
        )
    return targets, "default_joint_state"


def _hold_robot_at_targets(
    raw_env: Any,
    targets: Mapping[str, Any],
    *,
    reset_controllers: bool = False,
) -> dict[str, int]:
    """Hold the robot at one fixed reset readback while physics settles."""

    robots = getattr(raw_env, "_robots", None)
    if not isinstance(robots, Mapping):
        robots = {}
    joint_targets = getattr(raw_env, "_joint_targets", None)
    hand_joint_ids = getattr(raw_env, "_hand_joint_ids", {})
    held_joints = 0
    held_hands = 0

    for side in ("right", "left"):
        target = targets.get(side)
        if target is None:
            continue
        robot = robots.get(side)
        if robot is None:
            control_robot = getattr(raw_env, "_control_robot", None)
            if callable(control_robot):
                robot = control_robot(side)
        if robot is None:
            continue

        robot.set_joint_position_target(target)
        set_velocity_target = getattr(
            robot,
            "set_joint_velocity_target",
            None,
        )
        if callable(set_velocity_target):
            velocity_target = target.clone()
            zero_ = getattr(velocity_target, "zero_", None)
            if callable(zero_):
                zero_()
            else:
                velocity_target[...] = 0.0
            set_velocity_target(velocity_target)
        write_data = getattr(robot, "write_data_to_sim", None)
        if callable(write_data):
            write_data()
        if isinstance(joint_targets, dict):
            joint_targets[side] = target.clone()
        held_joints += int(target.shape[-1])

        ids = list(hand_joint_ids.get(side, ()))
        hand_command = getattr(raw_env, f"_{side}_hand", None)
        if ids and isinstance(hand_command, np.ndarray):
            values = (
                target[:, ids]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            count = min(hand_command.size, values.size)
            hand_command[:count] = values[:count]
            held_hands += count

        setattr(raw_env, f"_{side}_pose_commanded", False)

    filters = getattr(raw_env, "_arm_pose_filters", None)
    if isinstance(filters, dict):
        filters.update({"right": None, "left": None})
    if reset_controllers:
        controllers = getattr(raw_env, "_ik_controllers", None)
        if isinstance(controllers, Mapping):
            for controller in controllers.values():
                reset = getattr(controller, "reset", None)
                if callable(reset):
                    reset()

    return {
        "robot_joints": held_joints,
        "hand_commands": held_hands,
    }


def stabilize_recording_origin(
    env: Any,
    observation: Any,
    info: Mapping[str, Any] | None,
    *,
    linear_velocity_limit: float = DEFAULT_LINEAR_VELOCITY_LIMIT,
    angular_velocity_limit: float = DEFAULT_ANGULAR_VELOCITY_LIMIT,
    joint_velocity_limit: float | None = None,
    initial_settle_seconds: float = 0.0,
    post_settle_velocity_zeroing: bool = False,
    post_settle_sleeping: bool = False,
    stabilization_velocity_clamp: bool = False,
    stabilization_velocity_clamp_scope: str | None = None,
    stability_metric: str = "velocity",
    pose_translation_limit: float = 0.0005,
    pose_rotation_limit: float = 0.01,
    reset_translation_drift_limit: float | None = None,
    deformable_node_reset_drift_limit: float | None = None,
    reset_rotation_drift_limit: float | None = None,
    require_valid_contact_report: bool = False,
    contact_penetration_limit: float | None = None,
    stable_seconds: float = DEFAULT_STABLE_SECONDS,
    stability_sample_seconds: float = DEFAULT_STABILITY_SAMPLE_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a physically settled recording boundary before record or replay."""

    output_info = dict(info or {})
    raw_env = getattr(env, "unwrapped", env)
    if not stable_recording_origin_enabled() or not bool(
        getattr(raw_env, "stable_recording_origin_gate_enabled", True)
    ):
        return observation, output_info

    task_name = str(getattr(raw_env, "task_name", "")).strip().lower()
    isaac_device = getattr(raw_env, "_isaac_device", "")
    cpu_exception = getattr(
        raw_env,
        "_physx_cpu_ground_truth_exception",
        "",
    )
    if not is_physx_recording_device_allowed(
        isaac_device,
        cpu_exception=cpu_exception,
    ):
        raise RuntimeError(
            "Stable recording origin requires a canonical CUDA PhysX device "
            "or a schema-bound exact-asset CPU contract; "
            f"got {isaac_device!r}."
        )
    scene_generation = getattr(raw_env, "_physx_scene_generation", 0)
    if type(scene_generation) is not int or scene_generation < 0:
        raise RuntimeError(
            "Stable recording origin requires a nonnegative integer PhysX "
            "scene generation."
        )
    scene_rebuilt_before_reset = bool(scene_generation > 0)
    if type(stabilization_velocity_clamp) is not bool:
        raise ValueError("stabilization_velocity_clamp must be boolean.")
    if type(require_valid_contact_report) is not bool:
        raise ValueError("require_valid_contact_report must be boolean.")
    if stabilization_velocity_clamp_scope is None:
        # Direct legacy callers used the clamp only during the strict gate.
        # New collection profiles pass the all-step scope explicitly.
        stabilization_velocity_clamp_scope = (
            VELOCITY_CLAMP_SCOPE_GATE_ONLY
            if stabilization_velocity_clamp
            else VELOCITY_CLAMP_SCOPE_NONE
        )
    stabilization_velocity_clamp_scope = str(
        stabilization_velocity_clamp_scope
    ).strip().lower()
    if stabilization_velocity_clamp_scope not in _VELOCITY_CLAMP_SCOPES:
        raise ValueError(
            "unsupported stabilization velocity clamp scope: "
            f"{stabilization_velocity_clamp_scope!r}"
        )
    if stabilization_velocity_clamp and (
        stabilization_velocity_clamp_scope == VELOCITY_CLAMP_SCOPE_NONE
    ):
        raise ValueError("Enabled velocity clamp cannot use scope 'none'.")
    if not stabilization_velocity_clamp and (
        stabilization_velocity_clamp_scope != VELOCITY_CLAMP_SCOPE_NONE
    ):
        raise ValueError("Disabled velocity clamp must use scope 'none'.")
    stability_metric = str(stability_metric).strip().lower()
    if stability_metric not in {
        "velocity",
        "pose",
        "velocity_or_static_pose",
    }:
        raise ValueError(
            "unsupported recording-origin stability metric: "
            f"{stability_metric}"
        )
    if (
        stability_metric == "velocity_or_static_pose"
        and task_name not in STATIC_POSE_FALLBACK_TASKS
    ):
        approved_tasks = ", ".join(sorted(STATIC_POSE_FALLBACK_TASKS))
        raise ValueError(
            "velocity_or_static_pose recording-origin stability is restricted "
            f"to approved tasks ({approved_tasks}); got {task_name!r}."
        )
    if stability_metric == "velocity_or_static_pose" and (
        task_name
        in {
            "t26_place_fruit_in_bucket",
            "eval_comb04_place_fruit_and_photograph_logo",
            "t24_place_ball_in_bowl",
            "t10_lift_carton",
            "t30_pour_seasonings",
        }
        and (stabilization_velocity_clamp or post_settle_sleeping)
    ):
        raise ValueError(
            "Approved static-pose stability requires awake, unclamped physics."
        )
    joint_velocity_limit, joint_velocity_limit_source = (
        _resolve_joint_velocity_limit(raw_env, joint_velocity_limit)
    )
    step_physics = getattr(raw_env, "_step_isaaclab", None)
    apply_targets = getattr(raw_env, "_apply_joint_targets", None)
    if not callable(step_physics) or not callable(apply_targets):
        raise RuntimeError(
            "Stable recording origin requires native Isaac physics stepping."
        )

    physics_dt = float(getattr(raw_env, "_physics_dt", 0.0))
    if not np.isfinite(physics_dt) or physics_dt <= 0.0:
        raise RuntimeError("Stable recording origin requires a positive physics dt.")

    trace = getattr(raw_env, "_physx_ground_truth_trace", None)
    if trace is None:
        trace = PhysxGroundTruthTrace(raw_env)
    ground_truth_schema = trace.schema()
    try:
        expected_active_physics_layout = _expected_active_physics_layout(
            raw_env,
            ground_truth_schema,
        )
    except (TypeError, ValueError) as exc:
        raise RecordingOriginStabilityError(
            "Stable recording origin could not derive the expected active "
            f"physics layout: {exc}",
            diagnostics={
                "failure_phase": "expected_active_physics_layout",
                "geometry_failures": [
                    "expected_active_physics_layout_invalid"
                ],
            },
        ) from exc

    scalar_limits = {
        "linear_velocity_limit": linear_velocity_limit,
        "angular_velocity_limit": angular_velocity_limit,
        "joint_velocity_limit": joint_velocity_limit,
        "initial_settle_seconds": initial_settle_seconds,
        "pose_translation_limit": pose_translation_limit,
        "pose_rotation_limit": pose_rotation_limit,
        "stable_seconds": stable_seconds,
        "stability_sample_seconds": stability_sample_seconds,
        "timeout_seconds": timeout_seconds,
    }
    optional_limits = {
        "reset_translation_drift_limit": reset_translation_drift_limit,
        "deformable_node_reset_drift_limit": (
            deformable_node_reset_drift_limit
        ),
        "reset_rotation_drift_limit": reset_rotation_drift_limit,
        "contact_penetration_limit": contact_penetration_limit,
    }
    if any(
        not np.isfinite(float(value)) or float(value) < 0.0
        for value in scalar_limits.values()
    ):
        raise ValueError(
            "Stable recording-origin limits and durations must be finite and "
            f"non-negative: {scalar_limits}"
        )
    if any(
        value is not None
        and (not np.isfinite(float(value)) or float(value) < 0.0)
        for value in optional_limits.values()
    ):
        raise ValueError(
            "Stable recording-origin optional geometry limits must be finite "
            f"and non-negative: {optional_limits}"
        )
    if require_valid_contact_report and contact_penetration_limit is None:
        raise ValueError(
            "A required contact report must have a penetration-depth limit."
        )
    if stable_seconds <= 0.0:
        raise ValueError("stable_seconds must be positive.")
    if stability_metric == "velocity_or_static_pose":
        approved_pose_limits = _TASK_STATIC_POSE_FALLBACK_LIMITS[task_name]
        approved_physics_hz = _TASK_STATIC_POSE_FALLBACK_PHYSICS_HZ.get(
            task_name,
            _DEFAULT_STATIC_POSE_FALLBACK_PHYSICS_HZ,
        )
        if (
            float(pose_translation_limit) > approved_pose_limits[0]
            or float(pose_rotation_limit) > approved_pose_limits[1]
            or float(linear_velocity_limit) > DEFAULT_LINEAR_VELOCITY_LIMIT
            or float(angular_velocity_limit) > DEFAULT_ANGULAR_VELOCITY_LIMIT
            or (
                float(joint_velocity_limit) > DEFAULT_JOINT_VELOCITY_LIMIT
                and task_name
                not in {
                    "d13_prepare_tea_tray",
                    "d15_pack_matching_lunches",
                    "d17_store_laptop_headphones",
                }
            )
            or float(stable_seconds) < DEFAULT_STABLE_SECONDS
            or not np.isclose(
                physics_dt,
                1.0 / approved_physics_hz,
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                f"{task_name} static-pose fallback exceeds its approved "
                f"{approved_physics_hz:g} Hz stability contract."
            )

    stable_steps_required = max(
        1, int(np.ceil(float(stable_seconds) / physics_dt))
    )
    initial_settle_steps = max(
        0, int(np.ceil(float(initial_settle_seconds) / physics_dt))
    )
    # Approved task fallbacks exist only for non-integrated PhysX velocity
    # readback.
    # Inspect every simulated frame so an actual transform change can never be
    # hidden between the ordinary 50 ms velocity samples.
    stability_sample_steps = (
        1
        if stability_metric == "velocity_or_static_pose"
        else max(
            1,
            int(np.ceil(float(stability_sample_seconds) / physics_dt)),
        )
    )
    stability_sample_interval_seconds = stability_sample_steps * physics_dt
    timeout_steps = max(
        1, int(np.ceil(float(timeout_seconds) / physics_dt))
    )
    if timeout_steps < stable_steps_required:
        raise ValueError(
            "timeout_seconds must allow at least one full stable interval: "
            f"stable_seconds={stable_seconds}, timeout_seconds={timeout_seconds}"
        )
    progress_interval_steps = max(1, int(np.ceil(1.0 / physics_dt)))

    def publish_progress(
        phase: str,
        *,
        stable_steps: int,
        object_motion: tuple[float, float] | None = None,
        robot_motion: tuple[float, float, float] | None = None,
    ) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {
            "phase": str(phase),
            "physics_steps": int(physics_steps),
            "physics_step_budget": int(initial_settle_steps + timeout_steps),
            "simulated_seconds": float(physics_steps * physics_dt),
            "stable_steps_observed": int(stable_steps),
            "stable_steps_required": int(stable_steps_required),
        }
        if object_motion is not None:
            payload.update(
                {
                    "task_linear_velocity": float(object_motion[0]),
                    "task_angular_velocity": float(object_motion[1]),
                }
            )
        if robot_motion is not None:
            payload.update(
                {
                    "robot_linear_velocity": float(robot_motion[0]),
                    "robot_angular_velocity": float(robot_motion[1]),
                    "robot_joint_velocity": float(robot_motion[2]),
                }
            )
        progress_callback(payload)

    # The reset call advances one PhysX frame before returning its observation.
    # Never promote that solver-displaced live q (or an old drive target) to the
    # recording origin.  Restore the authored joint home, exact-zero qd, and
    # matching control/drive targets first.  The following canonical write is
    # then an atomic guard for every task and robot velocity before settling.
    robot_hold_targets, robot_hold_target_source = (
        _restore_default_robot_hold_targets(raw_env, sync=True)
    )
    default_joint_state_restores = int(
        robot_hold_target_source == "default_joint_state"
    )
    authored_reset_snapshot, authored_reset_serial = (
        _consume_authored_reset_snapshot(raw_env)
    )
    reset_pose_drift_required = bool(
        reset_translation_drift_limit is not None
        or deformable_node_reset_drift_limit is not None
        or reset_rotation_drift_limit is not None
    )
    if reset_pose_drift_required and authored_reset_snapshot is None:
        raise RecordingOriginStabilityError(
            "Stable recording origin has no authored-reset snapshot from "
            "before the first PhysX step.",
            diagnostics={
                "failure_phase": "authored_reset_snapshot",
                "geometry_failures": ["authored_reset_snapshot_missing"],
            },
        )
    # The live reset snapshot is already one solver frame later.  It remains
    # the state used for the first zero-velocity write, while all drift is
    # measured from the separately captured authored state above.
    live_reset_snapshot = trace.snapshot()
    try:
        if authored_reset_snapshot is None:
            raise ValueError("Authored reset snapshot is missing.")
        authored_active_physics_layout = _active_physics_layout(
            authored_reset_snapshot,
            ground_truth_schema,
        )
        live_reset_active_physics_layout = _active_physics_layout(
            live_reset_snapshot,
            ground_truth_schema,
        )
    except (TypeError, ValueError) as exc:
        raise RecordingOriginStabilityError(
            "The authored/live reset physics layout is invalid: "
            f"{exc}",
            diagnostics={
                "failure_phase": "initial_active_physics_layout",
                "geometry_failures": ["active_physics_layout_invalid"],
            },
        ) from exc
    initial_layout_failures: list[str] = []
    if authored_active_physics_layout != expected_active_physics_layout:
        initial_layout_failures.append(
            "authored_active_physics_layout_expected_mismatch"
        )
    if live_reset_active_physics_layout != expected_active_physics_layout:
        initial_layout_failures.append(
            "live_reset_active_physics_layout_expected_mismatch"
        )
    if initial_layout_failures:
        raise RecordingOriginStabilityError(
            "The authored/live reset hid an expected physics entity: "
            f"failures={initial_layout_failures}.",
            diagnostics={
                "failure_phase": "initial_active_physics_layout",
                "geometry_failures": initial_layout_failures,
                "expected_active_physics_layout": (
                    expected_active_physics_layout
                ),
                "authored_active_physics_layout": (
                    authored_active_physics_layout
                ),
                "live_reset_active_physics_layout": (
                    live_reset_active_physics_layout
                ),
            },
        )
    reset_drift_history = _new_reset_drift_history()
    if authored_reset_snapshot is not None:
        _accumulate_reset_state_drift(
            reset_drift_history,
            authored_reset_snapshot,
            live_reset_snapshot,
            trace,
        )
    contact_history = _new_contact_penetration_history()
    reset_contact_snapshot = (
        observation
        if isinstance(observation, Mapping)
        and "physx_contact_report_valid" in observation
        else live_reset_snapshot
    )
    _accumulate_contact_penetration(
        contact_history,
        reset_contact_snapshot,
        ground_truth_schema,
    )
    first_write = trace.write_zero_velocity_boundary(
        live_reset_snapshot,
        ground_truth_schema,
        sync=True,
    )
    hold_target_sync = _hold_robot_at_targets(
        raw_env,
        robot_hold_targets,
        reset_controllers=True,
    )
    hold_target_sync["passes"] = 1
    stable_steps_observed = 0
    physics_steps = 0
    last_object_motion: tuple[float, float] | None = None
    last_robot_motion: tuple[float, float, float] | None = None
    last_snapshot: Mapping[str, Any] | None = None
    last_pose_delta: tuple[float, float] | None = None
    velocity_clamp_steps = 0
    velocity_clamp_write = None

    def reject_irrecoverable_reset_drift() -> None:
        # These are history maxima: once exceeded, later stillness cannot
        # make this reset valid. Opt in only for tasks audited with this path.
        if not getattr(raw_env, "fail_fast_recording_origin_geometry", False):
            return
        failures = []
        drifts = (
            {"translation_m": reset_drift_history["maximum_rigid_translation_m"]},
            {
                "kind": "deformable_node",
                "translation_m": reset_drift_history["maximum_deformable_node_translation_m"],
            },
        )
        if (
            reset_translation_drift_limit is not None
            and _reset_translation_drift_exceeded(
                drifts,
                rigid_limit_m=float(reset_translation_drift_limit),
                deformable_node_limit_m=deformable_node_reset_drift_limit,
            )
        ):
            failures.append("reset_translation_drift_exceeded")
        if (
            reset_rotation_drift_limit is not None
            and reset_drift_history["maximum_rotation_rad"] > float(reset_rotation_drift_limit)
        ):
            failures.append("reset_rotation_drift_exceeded")
        if failures:
            raise RecordingOriginStabilityError(
                "Reset already violated the physical-origin geometry contract; "
                f"resample immediately: failures={failures}, steps={physics_steps}.",
                diagnostics={
                    "failure_phase": "initial_geometry_gate",
                    "early_rejection": True,
                    "physics_steps": physics_steps,
                    "geometry_failures": failures,
                    "reset_pose_drift_peaks": list(reset_drift_history["peaks"]),
                },
            )

    reject_irrecoverable_reset_drift()
    # Let reset contacts dissipate before applying the strict recording gate.
    # These frames are neither returned as observations nor recorded.
    for _ in range(initial_settle_steps):
        _hold_robot_at_targets(raw_env, robot_hold_targets)
        hold_target_sync["passes"] += 1
        apply_targets()
        step_physics()
        physics_steps += 1
        if stabilization_velocity_clamp_scope == (
            VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
        ):
            velocity_clamp_write = trace.zero_task_velocities()
            velocity_clamp_steps += 1
        initialization_step_snapshot = trace.snapshot()
        _accumulate_contact_penetration(
            contact_history,
            initialization_step_snapshot,
            ground_truth_schema,
        )
        if authored_reset_snapshot is not None:
            _accumulate_reset_state_drift(
                reset_drift_history,
                authored_reset_snapshot,
                initialization_step_snapshot,
                trace,
            )
        if (
            physics_steps % progress_interval_steps == 0
            or physics_steps == initial_settle_steps
        ):
            publish_progress("initial_settle", stable_steps=0)
        reject_irrecoverable_reset_drift()
    post_settle_write = None
    post_settle_sleep = None
    put_to_sleep = None
    post_settle_robot_hold_target_source = robot_hold_target_source
    if post_settle_velocity_zeroing:
        # A-series reset objects can keep rolling while contacts find their
        # resting manifold.  The authored robot home can likewise have a tiny
        # contact-constrained equilibrium offset.  Advance unrendered physics
        # at the unchanged dt first, capture that physically reached q as both
        # state and drive target, then zero velocities without rewriting q or
        # task poses.  Capturing after the reset's single PhysX frame prevents
        # the drive from chasing a stale
        # or unverified joint pose during collection.
        robot_hold_targets = _capture_robot_hold_targets(raw_env)
        post_settle_robot_hold_target_source = (
            "settled_after_default_joint_state"
            if robot_hold_target_source == "default_joint_state"
            else "settled_live_readback"
        )
        _hold_robot_at_targets(
            raw_env,
            robot_hold_targets,
            reset_controllers=True,
        )
        hold_target_sync["passes"] += 1
        post_settle_write = _write_zero_velocity_boundary(trace)
    if post_settle_sleeping:
        put_to_sleep = getattr(trace, "put_task_bodies_to_sleep", None)
        if not callable(put_to_sleep):
            raise RuntimeError(
                "Stable recording origin requires task-body sleep support."
            )
        post_settle_sleep = put_to_sleep()
    pose_reference = (
        trace.snapshot()
        if stability_metric in {"pose", "velocity_or_static_pose"}
        else None
    )
    steps_since_sample = 0
    stability_samples = 0
    task_velocity_stable = False
    task_pose_stable = False
    robot_velocity_stable = False
    gate_task_velocity_stable = False
    gate_task_static_pose_stable = False
    gate_robot_velocity_stable = False
    gate_static_pose_fallback_samples = 0
    gate_pose_samples = 0
    gate_max_task_translation_delta: float | None = None
    gate_max_task_rotation_delta: float | None = None
    gate_max_fallback_translation_delta: float | None = None
    gate_max_fallback_rotation_delta: float | None = None
    velocity_streak_steps = 0
    pose_streak_steps = 0
    pose_streak_task_velocity_stable = True
    pose_streak_fallback_samples = 0
    pose_streak_samples = 0
    pose_streak_max_translation_delta: float | None = None
    pose_streak_max_rotation_delta: float | None = None
    pose_streak_max_fallback_translation_delta: float | None = None
    pose_streak_max_fallback_rotation_delta: float | None = None
    # LW native articulations have task-relevant hinge coordinates that must
    # be proven stationary on every PhysX frame, even when the aggregate
    # velocity tensor happens to fall below the ordinary threshold.  Keep the
    # velocity branch available for other approved fallback tasks, but do not
    # let it bypass the full-frame pose evidence for G08/G10.
    require_lw_pose_gate = bool(
        stability_metric == "velocity_or_static_pose"
        and task_name in {
            "t18_open_micropipette_box",
            "t49_unfold_ruler",
            "eval_gen02_open_trash_bin",
        }
    )
    task_readiness_required = False
    task_readiness_ready = True
    task_readiness_samples = 0
    task_readiness_ready_samples = 0
    task_readiness_consecutive_steps = 0
    last_task_readiness: dict[str, Any] | None = None
    for timeout_step in range(timeout_steps):
        _hold_robot_at_targets(raw_env, robot_hold_targets)
        hold_target_sync["passes"] += 1
        apply_targets()
        step_physics()
        physics_steps += 1
        steps_since_sample += 1
        if stabilization_velocity_clamp:
            velocity_clamp_write = trace.zero_task_velocities()
            velocity_clamp_steps += 1
        initialization_step_snapshot = trace.snapshot()
        _accumulate_contact_penetration(
            contact_history,
            initialization_step_snapshot,
            ground_truth_schema,
        )
        if authored_reset_snapshot is not None:
            _accumulate_reset_state_drift(
                reset_drift_history,
                authored_reset_snapshot,
                initialization_step_snapshot,
                trace,
            )
        (
            readiness_required_now,
            task_readiness_ready,
            task_readiness_sample,
        ) = _task_recording_origin_readiness(raw_env)
        reject_irrecoverable_reset_drift()
        if readiness_required_now:
            task_readiness_required = True
            task_readiness_samples += 1
            task_readiness_ready_samples += int(task_readiness_ready)
            task_readiness_consecutive_steps = (
                task_readiness_consecutive_steps + 1
                if task_readiness_ready
                else 0
            )
            last_task_readiness = task_readiness_sample
        elif task_readiness_required:
            raise RecordingOriginStabilityError(
                "Task-specific recording-origin readiness disappeared during "
                "the continuous physical gate.",
                diagnostics={
                    "failure_phase": "task_readiness_contract",
                    "physics_steps": int(physics_steps),
                    "task_readiness": last_task_readiness,
                },
            )
        if (
            steps_since_sample < stability_sample_steps
            and timeout_step + 1 < timeout_steps
        ):
            continue
        last_snapshot = initialization_step_snapshot
        stability_samples += 1
        last_object_motion = _object_motion(last_snapshot)
        last_robot_motion = _robot_motion(last_snapshot)
        if last_object_motion is None:
            if task_name == "eval_gen05_set_mixer_knobs":
                # GEN6's fixed mixer fixture is represented by articulation
                # handles owned by its dedicated B320 runtime and therefore
                # has no generic rigid-body trace entry.  Keep the robot and
                # reset gates strict while allowing startup to proceed.
                last_object_motion = (0.0, 0.0)
            else:
                raise RuntimeError(
                    "Stable recording origin found no task-object state readback."
                )
        if last_robot_motion is None:
            raise RuntimeError(
                "Stable recording origin found no robot velocity readback."
            )
        if stability_metric in {"pose", "velocity_or_static_pose"}:
            assert pose_reference is not None
            last_pose_delta = _task_pose_delta(pose_reference, last_snapshot)
            pose_reference = last_snapshot
            task_pose_stable = bool(
                last_pose_delta is not None
                and last_pose_delta[0] <= pose_translation_limit
                and last_pose_delta[1] <= pose_rotation_limit
            )
        task_velocity_stable = bool(
            last_object_motion[0] <= linear_velocity_limit
            and last_object_motion[1] <= angular_velocity_limit
        )
        robot_velocity_stable = bool(
            last_robot_motion[0] <= linear_velocity_limit
            and last_robot_motion[1] <= angular_velocity_limit
            and last_robot_motion[2] <= joint_velocity_limit
        )
        if stability_metric == "velocity_or_static_pose":
            # Once task velocities are explicitly zeroed after the simulated
            # frame they are no longer independent evidence that the task pose
            # was stationary.  Preserve the approved tasks' per-frame pose
            # protection by disabling the velocity-only acceptance branch
            # under the global initialization clamp; robot velocities remain
            # strict.
            velocity_stable_now = bool(
                task_readiness_ready
                and not stabilization_velocity_clamp
                and robot_velocity_stable
                and task_velocity_stable
            )
            pose_stable_now = bool(
                task_readiness_ready
                and robot_velocity_stable
                and task_pose_stable
            )
            if velocity_stable_now:
                velocity_streak_steps += steps_since_sample
                if task_readiness_required:
                    velocity_streak_steps = min(
                        velocity_streak_steps,
                        task_readiness_consecutive_steps,
                    )
            else:
                velocity_streak_steps = 0
            if pose_stable_now:
                assert last_pose_delta is not None
                if pose_streak_steps == 0:
                    pose_streak_task_velocity_stable = True
                    pose_streak_fallback_samples = 0
                    pose_streak_samples = 0
                    pose_streak_max_translation_delta = None
                    pose_streak_max_rotation_delta = None
                    pose_streak_max_fallback_translation_delta = None
                    pose_streak_max_fallback_rotation_delta = None
                pose_streak_steps += steps_since_sample
                if task_readiness_required:
                    pose_streak_steps = min(
                        pose_streak_steps,
                        task_readiness_consecutive_steps,
                    )
                pose_streak_samples += 1
                pose_streak_task_velocity_stable &= task_velocity_stable
                pose_streak_max_translation_delta = max(
                    pose_streak_max_translation_delta or 0.0,
                    last_pose_delta[0],
                )
                pose_streak_max_rotation_delta = max(
                    pose_streak_max_rotation_delta or 0.0,
                    last_pose_delta[1],
                )
                if stabilization_velocity_clamp or not task_velocity_stable:
                    pose_streak_fallback_samples += 1
                    pose_streak_max_fallback_translation_delta = max(
                        pose_streak_max_fallback_translation_delta or 0.0,
                        last_pose_delta[0],
                    )
                    pose_streak_max_fallback_rotation_delta = max(
                        pose_streak_max_fallback_rotation_delta or 0.0,
                        last_pose_delta[1],
                    )
            else:
                pose_streak_steps = 0

            # The two criteria have independent continuous windows.  Never
            # combine a moving-but-slow frame with a static-but-noisy frame to
            # manufacture one accepted interval.
            if (
                velocity_streak_steps >= stable_steps_required
                and not require_lw_pose_gate
            ):
                stable_steps_observed = velocity_streak_steps
                gate_task_velocity_stable = True
                gate_robot_velocity_stable = True
                publish_progress(
                    "stability_gate",
                    stable_steps=stable_steps_observed,
                    object_motion=last_object_motion,
                    robot_motion=last_robot_motion,
                )
                break
            if pose_streak_steps >= stable_steps_required:
                stable_steps_observed = pose_streak_steps
                gate_task_velocity_stable = (
                    False
                    if stabilization_velocity_clamp
                    else pose_streak_task_velocity_stable
                )
                gate_task_static_pose_stable = True
                gate_robot_velocity_stable = True
                gate_static_pose_fallback_samples = (
                    pose_streak_fallback_samples
                )
                gate_pose_samples = pose_streak_samples
                gate_max_task_translation_delta = (
                    pose_streak_max_translation_delta
                )
                gate_max_task_rotation_delta = (
                    pose_streak_max_rotation_delta
                )
                gate_max_fallback_translation_delta = (
                    pose_streak_max_fallback_translation_delta
                )
                gate_max_fallback_rotation_delta = (
                    pose_streak_max_fallback_rotation_delta
                )
                publish_progress(
                    "stability_gate",
                    stable_steps=stable_steps_observed,
                    object_motion=last_object_motion,
                    robot_motion=last_robot_motion,
                )
                break
            stable_steps_observed = max(
                velocity_streak_steps,
                pose_streak_steps,
            )
        else:
            stable_now = (
                task_pose_stable
                if stability_metric == "pose"
                else bool(task_velocity_stable and robot_velocity_stable)
            )
            stable_now = bool(task_readiness_ready and stable_now)
            if stable_now:
                stable_steps_observed += steps_since_sample
                if task_readiness_required:
                    stable_steps_observed = min(
                        stable_steps_observed,
                        task_readiness_consecutive_steps,
                    )
                if stable_steps_observed >= stable_steps_required:
                    gate_task_velocity_stable = task_velocity_stable
                    gate_task_static_pose_stable = task_pose_stable
                    gate_robot_velocity_stable = robot_velocity_stable
                    publish_progress(
                        "stability_gate",
                        stable_steps=stable_steps_observed,
                        object_motion=last_object_motion,
                        robot_motion=last_robot_motion,
                    )
                    break
            else:
                stable_steps_observed = 0
        if (
            physics_steps % progress_interval_steps == 0
            or timeout_step + 1 == timeout_steps
        ):
            publish_progress(
                "stability_gate",
                stable_steps=stable_steps_observed,
                object_motion=last_object_motion,
                robot_motion=last_robot_motion,
            )
        steps_since_sample = 0
    else:
        assert last_object_motion is not None
        assert last_robot_motion is not None
        assert last_snapshot is not None
        task_motion_peaks = _task_motion_peaks(
            last_snapshot,
            trace,
            linear_velocity_limit=linear_velocity_limit,
            angular_velocity_limit=angular_velocity_limit,
        )
        raise RecordingOriginStabilityError(
            "The robot and task objects did not remain physically stable "
            "before the "
            "recording-origin timeout: "
            f"linear={max(last_object_motion[0], last_robot_motion[0]):.6g} m/s "
            f"(limit {linear_velocity_limit:.6g}), "
            "angular="
            f"{max(last_object_motion[1], last_robot_motion[1]):.6g} rad/s "
            f"(limit {angular_velocity_limit:.6g}), "
            f"robot_joint={last_robot_motion[2]:.6g} rad/s "
            f"(limit {joint_velocity_limit:.6g}), "
            f"steps={physics_steps}, timeout={timeout_seconds:.6g}s, "
            f"metric={stability_metric}, pose_delta={last_pose_delta}, "
            f"task_readiness={last_task_readiness}."
            f" Fastest task bodies: {task_motion_peaks}."
            f" Fastest joints: "
            f"{_robot_joint_velocity_peaks(last_snapshot, trace)}",
            diagnostics={
                "failure_phase": "stability_timeout",
                "physics_steps": int(physics_steps),
                "stability_samples": int(stability_samples),
                "hold_target_sync_passes": int(
                    hold_target_sync["passes"]
                ),
                "task_readiness_required": bool(
                    task_readiness_required
                ),
                "task_readiness": last_task_readiness,
                "task_readiness_consecutive_steps": int(
                    task_readiness_consecutive_steps
                ),
            },
        )

    assert last_object_motion is not None
    assert last_robot_motion is not None
    assert last_snapshot is not None
    gate_snapshot = last_snapshot
    gate_object_motion = last_object_motion
    gate_robot_motion = last_robot_motion

    reset_drift_expected_sample_count = int(physics_steps + 1)
    reset_drift_all_physics_frames_sampled = bool(
        int(reset_drift_history["sample_count"])
        == reset_drift_expected_sample_count
        and reset_drift_history["valid"] is True
        and int(reset_drift_history["invalid_sample_count"]) == 0
    )
    if not reset_drift_all_physics_frames_sampled:
        reset_pose_drifts = None
        reset_state_coverage = None
        reset_max_coordinate_drift = None
        reset_max_translation_drift = None
        reset_max_rigid_translation_drift = None
        reset_max_deformable_node_translation_drift = None
        reset_max_rotation_drift = None
    else:
        reset_pose_drifts = list(reset_drift_history["peaks"])
        reset_state_coverage = dict(reset_drift_history["coverage"])
        reset_max_coordinate_drift = float(
            reset_drift_history["maximum_coordinate"]
        )
        reset_max_translation_drift = float(
            reset_drift_history["maximum_translation_m"]
        )
        reset_max_rigid_translation_drift = float(
            reset_drift_history["maximum_rigid_translation_m"]
        )
        reset_max_deformable_node_translation_drift = float(
            reset_drift_history[
                "maximum_deformable_node_translation_m"
            ]
        )
        reset_max_rotation_drift = float(
            reset_drift_history["maximum_rotation_rad"]
        )
    reset_coordinate_drift_limit = (
        min(
            float(reset_translation_drift_limit),
            float(reset_rotation_drift_limit),
        )
        if reset_translation_drift_limit is not None
        and reset_rotation_drift_limit is not None
        else (
            float(reset_translation_drift_limit)
            if reset_translation_drift_limit is not None
            else (
                float(reset_rotation_drift_limit)
                if reset_rotation_drift_limit is not None
                else None
            )
        )
    )
    gate_contact_summary = _contact_penetration_summary(
        gate_snapshot,
        ground_truth_schema,
    )
    gate_active_physics_layout = None
    active_physics_layout_error = None
    try:
        gate_active_physics_layout = _active_physics_layout(
            gate_snapshot,
            ground_truth_schema,
        )
        _validate_active_physics_layout_coverage(
            expected_active_physics_layout,
            reset_state_coverage,
        )
        _validate_active_physics_layout_coverage(
            gate_active_physics_layout,
            reset_state_coverage,
        )
    except (TypeError, ValueError) as exc:
        active_physics_layout_error = str(exc)
    contact_summary = dict(contact_history)
    contact_summary["expected_sample_count"] = int(physics_steps + 1)
    contact_summary["all_physics_frames_sampled"] = bool(
        int(contact_summary["sample_count"]) == physics_steps + 1
    )
    geometry_failures: list[str] = []
    if gate_active_physics_layout is None:
        geometry_failures.append("active_physics_layout_invalid")
    elif authored_active_physics_layout != expected_active_physics_layout:
        geometry_failures.append(
            "authored_active_physics_layout_expected_mismatch"
        )
    elif gate_active_physics_layout != expected_active_physics_layout:
        geometry_failures.append(
            "gate_active_physics_layout_expected_mismatch"
        )
    if reset_pose_drift_required and reset_pose_drifts is None:
        geometry_failures.append("reset_pose_layout_not_comparable")
    if (
        reset_coordinate_drift_limit is not None
        and reset_max_coordinate_drift is not None
        and reset_max_coordinate_drift > reset_coordinate_drift_limit
    ):
        geometry_failures.append("reset_coordinate_drift_exceeded")
    if (
        reset_translation_drift_limit is not None
        and reset_max_rigid_translation_drift is not None
        and reset_max_deformable_node_translation_drift is not None
        and _reset_translation_drift_exceeded(
            (
                {
                    "kind": "rigid_or_articulated",
                    "translation_m": reset_max_rigid_translation_drift,
                },
                {
                    "kind": "deformable_node",
                    "translation_m": (
                        reset_max_deformable_node_translation_drift
                    ),
                },
            ),
            rigid_limit_m=float(reset_translation_drift_limit),
            deformable_node_limit_m=deformable_node_reset_drift_limit,
        )
    ):
        geometry_failures.append("reset_translation_drift_exceeded")
    if (
        reset_rotation_drift_limit is not None
        and reset_max_rotation_drift is not None
        and reset_max_rotation_drift > float(reset_rotation_drift_limit)
    ):
        geometry_failures.append("reset_rotation_drift_exceeded")
    if require_valid_contact_report and (
        contact_summary["valid"] is not True
        or contact_summary["all_physics_frames_sampled"] is not True
    ):
        geometry_failures.append("physx_contact_report_invalid")
    maximum_penetration = contact_summary[
        "maximum_penetration_depth_m"
    ]
    if (
        contact_penetration_limit is not None
        and float(maximum_penetration) > float(contact_penetration_limit)
    ):
        geometry_failures.append("contact_penetration_exceeded")
    if geometry_failures:
        contact_schema = (
            (ground_truth_schema.get("physx") or {}).get("contact_report")
            if isinstance(ground_truth_schema, Mapping)
            else None
        )
        contact_path_table = (
            list(contact_schema.get("path_table") or [])
            if isinstance(contact_schema, Mapping)
            else []
        )
        contact_signature = gate_contact_summary.get(
            "canonical_contact_signature"
        )
        contact_pairs: list[tuple[str, str, str, str]] = []
        if isinstance(contact_signature, Mapping):
            for raw_pair in contact_signature.get("pair_path_indices", ()):
                try:
                    paths = tuple(
                        str(contact_path_table[int(index) - 1]["path"])
                        for index in raw_pair
                    )
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
                if len(paths) == 4:
                    contact_pairs.append(paths)  # type: ignore[arg-type]
        raise RecordingOriginStabilityError(
            "The reset became quiet only after violating the physical-origin "
            "geometry contract: "
            f"failures={geometry_failures}, "
            f"coordinate_drift={reset_max_coordinate_drift}, "
            f"translation_drift={reset_max_translation_drift}, "
            f"rotation_drift={reset_max_rotation_drift}, "
            f"penetration={maximum_penetration}, "
            f"contact_pairs={contact_pairs[:20]}, "
            f"contact_error={contact_summary.get('error')!r}, "
            "peaks="
            f"{None if reset_pose_drifts is None else reset_pose_drifts[:5]}.",
            diagnostics={
                "failure_phase": "initial_geometry_gate",
                "geometry_failures": list(geometry_failures),
                "reset_max_translation_drift_m": (
                    reset_max_translation_drift
                ),
                "reset_max_rotation_drift_rad": reset_max_rotation_drift,
                "reset_max_coordinate_drift": reset_max_coordinate_drift,
                "reset_state_coverage": reset_state_coverage,
                "reset_drift_sample_count": int(
                    reset_drift_history["sample_count"]
                ),
                "reset_drift_expected_sample_count": (
                    reset_drift_expected_sample_count
                ),
                "reset_drift_all_physics_frames_sampled": (
                    reset_drift_all_physics_frames_sampled
                ),
                "reset_pose_drift_peaks": (
                    None
                    if reset_pose_drifts is None
                    else reset_pose_drifts[:5]
                ),
                "contact_summary": dict(contact_summary),
                "gate_contact_summary": dict(gate_contact_summary),
                "contact_path_table": (
                    list(contact_schema.get("path_table") or [])
                    if isinstance(contact_schema, Mapping)
                    else None
                ),
                "active_physics_layout_error": (
                    active_physics_layout_error
                ),
                "gate_active_physics_layout": (
                    gate_active_physics_layout
                ),
                "authored_active_physics_layout": (
                    authored_active_physics_layout
                ),
                "expected_active_physics_layout": (
                    expected_active_physics_layout
                ),
                "physics_steps": int(physics_steps),
                "stability_samples": int(stability_samples),
                "hold_target_sync_passes": int(
                    hold_target_sync["passes"]
                ),
            },
        )

    # The stability gate necessarily ends after a simulated frame, which can
    # leave a small gravity/contact residual even though every threshold was
    # satisfied.  Establish the actual pre-action boundary only after the gate:
    # preserve the accepted poses/contact manifold while setting all public
    # task and joint velocities to exactly zero, then read the boundary back
    # without advancing physics.  No state write or
    # physics step is allowed after this point and before action[0].
    # Keep the joint pose that actually passed the physical gate.  Teleporting
    # back to the authored home here would create a new, unverified state that
    # could move again on action[0].  The velocity-only transaction synchronizes
    # derived body/TCP caches without advancing physics and aligns every drive
    # target to that accepted q, while deliberately avoiding pose/DOF-position
    # teleportation.
    final_robot_hold_targets = _capture_robot_hold_targets(raw_env)
    final_hold_target_sync = _hold_robot_at_targets(
        raw_env,
        final_robot_hold_targets,
        reset_controllers=True,
    )
    final_boundary_write = _write_zero_velocity_boundary(trace, sync=True)
    final_boundary_sleep = None
    if post_settle_sleeping:
        assert callable(put_to_sleep)
        final_boundary_sleep = put_to_sleep()
    final_snapshot = trace.snapshot()
    assert gate_active_physics_layout is not None
    final_boundary_failures: list[str] = []
    (
        final_task_readiness_required,
        final_task_readiness_ready,
        final_task_readiness,
    ) = _task_recording_origin_readiness(raw_env)
    if task_readiness_required and not final_task_readiness_required:
        final_boundary_failures.append(
            "final_task_readiness_contract_missing"
        )
    if final_task_readiness_required:
        task_readiness_required = True
        if not final_task_readiness_ready:
            final_boundary_failures.append(
                "final_task_readiness_unavailable"
            )
    final_active_physics_layout = None
    final_active_physics_layout_error = None
    try:
        final_active_physics_layout = _active_physics_layout(
            final_snapshot,
            ground_truth_schema,
        )
        _validate_active_physics_layout_coverage(
            final_active_physics_layout,
            reset_state_coverage,
        )
    except (TypeError, ValueError) as exc:
        final_active_physics_layout_error = str(exc)
        final_boundary_failures.append("active_physics_layout_invalid")
    if (
        final_active_physics_layout is not None
        and final_active_physics_layout != gate_active_physics_layout
    ):
        final_boundary_failures.append(
            "active_physics_layout_gate_final_mismatch"
        )
    if (
        final_active_physics_layout is not None
        and final_active_physics_layout != expected_active_physics_layout
    ):
        final_boundary_failures.append(
            "final_active_physics_layout_expected_mismatch"
        )
    gate_robot_target_valid = np.asarray(
        gate_snapshot.get(
            "physx_robot_joint_target_valid",
            np.zeros((0, 0)),
        )
    )
    final_robot_target_valid = np.asarray(
        final_snapshot.get(
            "physx_robot_joint_target_valid",
            np.zeros((0, 0)),
        )
    )
    if not np.array_equal(
        gate_robot_target_valid,
        final_robot_target_valid,
    ):
        final_boundary_failures.append(
            "final_robot_target_validity_mismatch"
        )

    final_reset_drifts = None
    final_reset_state_coverage = None
    final_reset_max_translation_drift = None
    final_reset_max_rotation_drift = None
    final_reset_max_coordinate_drift = None
    if authored_reset_snapshot is not None:
        final_reset_result = _reset_state_drifts(
            authored_reset_snapshot,
            final_snapshot,
            trace,
        )
    else:
        final_reset_result = None
    if final_reset_result is None:
        final_boundary_failures.append(
            "final_authored_reset_layout_not_comparable"
        )
    else:
        final_reset_drifts, final_reset_state_coverage = final_reset_result
        final_reset_max_translation_drift = max(
            (
                float(item.get("translation_m", 0.0))
                for item in final_reset_drifts
            ),
            default=0.0,
        )
        final_reset_max_rotation_drift = max(
            (
                float(item.get("rotation_rad", 0.0))
                for item in final_reset_drifts
            ),
            default=0.0,
        )
        final_reset_max_coordinate_drift = max(
            (
                float(item.get("coordinate", 0.0))
                for item in final_reset_drifts
            ),
            default=0.0,
        )
        if final_reset_state_coverage != reset_state_coverage:
            final_boundary_failures.append(
                "final_reset_state_coverage_mismatch"
            )

    if task_name in _TASKS_ALLOW_ROBOT_RESET_REPOSITION:
        final_reset_translation_limit = None
        final_reset_rotation_limit = None
        final_reset_coordinate_limit = None
    else:
        final_reset_translation_limit = (
            float(reset_translation_drift_limit)
            if reset_translation_drift_limit is not None
            else DEFAULT_RESET_TRANSLATION_DRIFT_LIMIT
        )
        final_reset_rotation_limit = (
            float(reset_rotation_drift_limit)
            if reset_rotation_drift_limit is not None
            else DEFAULT_RESET_ROTATION_DRIFT_LIMIT
        )
        final_reset_coordinate_limit = min(
            final_reset_translation_limit,
            final_reset_rotation_limit,
        )
        if (
            final_reset_max_coordinate_drift is not None
            and final_reset_max_coordinate_drift > final_reset_coordinate_limit
        ):
            final_boundary_failures.append(
                "final_reset_coordinate_drift_exceeded"
            )
        if (
            final_reset_drifts is not None
            and _reset_translation_drift_exceeded(
                final_reset_drifts,
                rigid_limit_m=final_reset_translation_limit,
                deformable_node_limit_m=deformable_node_reset_drift_limit,
            )
        ):
            final_boundary_failures.append(
                "final_reset_translation_drift_exceeded"
            )
        if (
            final_reset_max_rotation_drift is not None
            and final_reset_max_rotation_drift > final_reset_rotation_limit
        ):
            final_boundary_failures.append(
                "final_reset_rotation_drift_exceeded"
            )

    gate_final_translation_limit = float(pose_translation_limit)
    gate_final_rotation_limit = float(pose_rotation_limit)
    if final_reset_translation_limit is not None:
        gate_final_translation_limit = min(
            gate_final_translation_limit,
            final_reset_translation_limit,
        )
    if final_reset_rotation_limit is not None:
        gate_final_rotation_limit = min(
            gate_final_rotation_limit,
            final_reset_rotation_limit,
        )
    gate_final_rotation_readback_floor = 0.0
    if str(getattr(raw_env, "task_name", "")).strip().lower() == (
        "t46_store_charger"
    ):
        # FU006's fixed, scaled drawer articulation is recomputed by
        # ``sim.forward()`` during the final velocity-only synchronization.
        # On Isaac Sim 5.1 GPU PhysX that no-step readback differs by exactly
        # float32 transform ULPs (1.5256e-5 rad on the production RTX 4090),
        # even though the preceding every-120-Hz-frame pose gate passed.  Keep
        # the physical gate at 1e-5 rad and admit only this bounded cache-sync
        # representation noise at the final no-step boundary.
        gate_final_rotation_readback_floor = float(
            512.0 * np.finfo(np.float32).eps
        )
        gate_final_rotation_limit = max(
            gate_final_rotation_limit,
            gate_final_rotation_readback_floor,
        )
    gate_final_coordinate_limit = min(
        gate_final_translation_limit,
        gate_final_rotation_limit,
    )
    gate_final_result = _reset_state_drifts(
        gate_snapshot,
        final_snapshot,
        trace,
    )
    gate_final_drifts = None
    gate_final_state_coverage = None
    gate_final_coordinate_mismatches = _full_coordinate_exact_mismatches(
        gate_snapshot,
        final_snapshot,
    )
    if gate_final_result is None or gate_final_coordinate_mismatches is None:
        gate_final_pose_delta = None
        gate_final_max_coordinate_delta = None
        final_boundary_failures.append(
            "gate_final_full_coordinates_not_comparable"
        )
    else:
        gate_final_drifts, gate_final_state_coverage = gate_final_result
        gate_final_pose_delta = (
            max(
                (
                    float(item.get("translation_m", 0.0))
                    for item in gate_final_drifts
                ),
                default=0.0,
            ),
            max(
                (
                    float(item.get("rotation_rad", 0.0))
                    for item in gate_final_drifts
                ),
                default=0.0,
            ),
        )
        gate_final_max_coordinate_delta = max(
            (
                float(item.get("coordinate", 0.0))
                for item in gate_final_drifts
            ),
            default=0.0,
        )
        if gate_final_state_coverage != reset_state_coverage:
            final_boundary_failures.append(
                "gate_final_state_coverage_mismatch"
            )
        if gate_final_coordinate_mismatches:
            final_boundary_failures.append(
                "gate_final_coordinates_changed_without_physics_step"
            )
        if gate_final_pose_delta[0] > gate_final_translation_limit:
            final_boundary_failures.append(
                "gate_final_translation_delta_exceeded"
            )
        if gate_final_pose_delta[1] > gate_final_rotation_limit:
            final_boundary_failures.append(
                "gate_final_rotation_delta_exceeded"
            )
        if gate_final_max_coordinate_delta > gate_final_coordinate_limit:
            final_boundary_failures.append(
                "gate_final_coordinate_delta_exceeded"
            )

    final_non_finite_fields = _non_finite_valid_snapshot_fields(
        final_snapshot
    )
    if final_non_finite_fields:
        final_boundary_failures.append(
            "final_valid_physics_rows_non_finite"
        )

    final_contact_summary = _contact_penetration_summary(
        final_snapshot,
        ground_truth_schema,
    )
    final_contact_penetration_limit = min(
        DEFAULT_CONTACT_PENETRATION_LIMIT,
        (
            float(contact_penetration_limit)
            if contact_penetration_limit is not None
            else DEFAULT_CONTACT_PENETRATION_LIMIT
        ),
    )
    if final_contact_summary["valid"] is not True:
        final_boundary_failures.append("final_contact_report_invalid")
    if (
        gate_contact_summary.get("canonical_contact_signature")
        != final_contact_summary.get("canonical_contact_signature")
    ):
        final_boundary_failures.append(
            "gate_final_contact_signature_mismatch"
        )
    final_penetration = final_contact_summary[
        "maximum_penetration_depth_m"
    ]
    if (
        final_penetration is not None
        and float(final_penetration) > final_contact_penetration_limit
    ):
        final_boundary_failures.append(
            "final_contact_penetration_exceeded"
        )
    if final_boundary_failures:
        raise RecordingOriginStabilityError(
            "The canonical pre-action restore changed or hid physical state: "
            f"failures={final_boundary_failures}, "
            f"gate_final_pose_delta={gate_final_pose_delta}.",
            diagnostics={
                "failure_phase": "final_boundary_physics_gate",
                "geometry_failures": list(final_boundary_failures),
                "active_physics_layout_error": (
                    final_active_physics_layout_error
                ),
                "gate_active_physics_layout": (
                    gate_active_physics_layout
                ),
                "final_active_physics_layout": (
                    final_active_physics_layout
                ),
                "expected_active_physics_layout": (
                    expected_active_physics_layout
                ),
                "final_reset_state_coverage": (
                    final_reset_state_coverage
                ),
                "final_reset_max_translation_drift_m": (
                    final_reset_max_translation_drift
                ),
                "final_reset_max_rotation_drift_rad": (
                    final_reset_max_rotation_drift
                ),
                "final_reset_max_coordinate_drift": (
                    final_reset_max_coordinate_drift
                ),
                "gate_final_pose_delta": gate_final_pose_delta,
                "gate_final_rotation_readback_floor": (
                    gate_final_rotation_readback_floor
                ),
                "gate_final_drifts": gate_final_drifts,
                "gate_final_state_coverage": gate_final_state_coverage,
                "gate_final_coordinate_mismatches": (
                    gate_final_coordinate_mismatches
                ),
                "gate_final_max_coordinate_delta": (
                    gate_final_max_coordinate_delta
                ),
                "final_non_finite_fields": final_non_finite_fields,
                "final_contact_summary": dict(final_contact_summary),
                "final_task_readiness": final_task_readiness,
                "physics_steps": int(physics_steps),
            },
        )
    final_object_motion = _object_motion(final_snapshot)
    final_robot_motion = _robot_motion(final_snapshot)
    if final_object_motion is None:
        raise RuntimeError(
            "Stable recording origin found no final task-object readback."
        )
    if final_robot_motion is None:
        raise RuntimeError(
            "Stable recording origin found no final robot velocity readback."
        )
    final_position = np.asarray(
        final_snapshot.get("physx_robot_joint_position", np.zeros((0, 0))),
        dtype=np.float64,
    )
    final_target = np.asarray(
        final_snapshot.get("physx_robot_joint_target", np.zeros((0, 0))),
        dtype=np.float64,
    )
    final_position_valid = np.asarray(
        final_snapshot.get("physx_robot_joint_valid", np.zeros((0, 0)))
    ).astype(bool)
    final_target_valid = np.asarray(
        final_snapshot.get(
            "physx_robot_joint_target_valid",
            np.zeros((0, 0)),
        )
    ).astype(bool)
    comparable = bool(
        final_position.shape == final_target.shape
        and final_position.shape == final_position_valid.shape
        and final_position.shape == final_target_valid.shape
    )
    valid_targets = (
        final_position_valid & final_target_valid
        if comparable
        else np.zeros((0, 0), dtype=bool)
    )
    if (
        not comparable
        or not np.any(valid_targets)
        or not np.array_equal(final_position_valid, final_target_valid)
    ):
        raise RecordingOriginStabilityError(
            "Stable recording origin found missing or mismatched final robot "
            "drive-target validity.",
            diagnostics={
                "failure_phase": "final_boundary_physics_gate",
                "geometry_failures": [
                    "final_robot_target_validity_mismatch"
                ],
                "physics_steps": int(physics_steps),
            },
        )
    final_robot_position_target_error = float(
        np.max(np.abs(final_position[valid_targets] - final_target[valid_targets]))
    )
    if (
        final_object_motion[0] != 0.0
        or final_object_motion[1] != 0.0
        or final_robot_motion[0] != 0.0
        or final_robot_motion[1] != 0.0
        or final_robot_motion[2] != 0.0
        or final_robot_position_target_error != 0.0
    ):
        raise RecordingOriginStabilityError(
            "The final pre-action recording boundary did not read back exact "
            "zero task/joint velocity and an exactly aligned robot target "
            "without a physics step: "
            f"task_linear={final_object_motion[0]:.9g}, "
            f"task_angular={final_object_motion[1]:.9g}, "
            f"robot_linear={final_robot_motion[0]:.9g}, "
            f"robot_angular={final_robot_motion[1]:.9g}, "
            f"robot_joint={final_robot_motion[2]:.9g}, "
            f"robot_target_error={final_robot_position_target_error:.9g}.",
            diagnostics={
                "failure_phase": "final_boundary_readback",
                "physics_steps": int(physics_steps),
                "stability_samples": int(stability_samples),
                "hold_target_sync_passes": int(
                    hold_target_sync["passes"]
                ),
            },
        )
    last_snapshot = final_snapshot
    last_object_motion = final_object_motion
    last_robot_motion = final_robot_motion
    final_max_linear_velocity = max(
        last_object_motion[0],
        last_robot_motion[0],
    )
    final_max_angular_velocity = max(
        last_object_motion[1],
        last_robot_motion[1],
    )
    raw_env.env_step = 0
    for counter_name in ("_success_counter", "_failure_counter"):
        if hasattr(raw_env, counter_name):
            setattr(raw_env, counter_name, 0)

    report = {
        "strategy": "continuous_current_target_all_velocity_settle",
        "stability_metric": stability_metric,
        "physics_steps": physics_steps,
        "physics_dt": physics_dt,
        "initial_settle_seconds": float(initial_settle_seconds),
        "initial_settle_steps": initial_settle_steps,
        "velocity_zeroing_passes": (
            3 if post_settle_write is not None else 2
        ),
        "post_settle_velocity_zeroing": bool(
            post_settle_velocity_zeroing
        ),
        "post_settle_state_write": post_settle_write,
        "post_settle_sleeping": bool(post_settle_sleeping),
        "post_settle_sleep": post_settle_sleep,
        "boundary_state_write_mode": "velocity_only",
        "contact_warm_start_preserved": bool(not post_settle_sleeping),
        "gate_max_task_linear_velocity": float(gate_object_motion[0]),
        "gate_max_task_angular_velocity": float(gate_object_motion[1]),
        "gate_max_robot_linear_velocity": float(gate_robot_motion[0]),
        "gate_max_robot_angular_velocity": float(gate_robot_motion[1]),
        "gate_max_robot_joint_velocity": float(gate_robot_motion[2]),
        "gate_task_velocity_stable": bool(gate_task_velocity_stable),
        "gate_task_static_pose_stable": bool(
            gate_task_static_pose_stable
        ),
        "gate_robot_velocity_stable": bool(gate_robot_velocity_stable),
        "gate_static_pose_fallback_samples": (
            gate_static_pose_fallback_samples
        ),
        "gate_pose_samples": gate_pose_samples,
        "gate_max_task_translation_delta": gate_max_task_translation_delta,
        "gate_max_task_rotation_delta": gate_max_task_rotation_delta,
        "gate_max_fallback_translation_delta": (
            gate_max_fallback_translation_delta
        ),
        "gate_max_fallback_rotation_delta": (
            gate_max_fallback_rotation_delta
        ),
        "gate_robot_joint_velocity_peaks": _robot_joint_velocity_peaks(
            gate_snapshot,
            trace,
        ),
        "reset_pose_drift_checked": reset_pose_drift_required,
        "reset_translation_drift_limit": (
            None
            if reset_translation_drift_limit is None
            else float(reset_translation_drift_limit)
        ),
        "deformable_node_reset_drift_limit": (
            None
            if deformable_node_reset_drift_limit is None
            else float(deformable_node_reset_drift_limit)
        ),
        "reset_rotation_drift_limit": (
            None
            if reset_rotation_drift_limit is None
            else float(reset_rotation_drift_limit)
        ),
        "reset_max_translation_drift_m": reset_max_translation_drift,
        "reset_max_rigid_translation_drift_m": (
            reset_max_rigid_translation_drift
        ),
        "reset_max_deformable_node_translation_drift_m": (
            reset_max_deformable_node_translation_drift
        ),
        "reset_max_rotation_drift_rad": reset_max_rotation_drift,
        "reset_coordinate_drift_limit": reset_coordinate_drift_limit,
        "reset_max_coordinate_drift": reset_max_coordinate_drift,
        "reset_state_coverage": reset_state_coverage,
        "active_physics_layout": gate_active_physics_layout,
        "expected_active_physics_layout": expected_active_physics_layout,
        "authored_active_physics_layout": authored_active_physics_layout,
        "gate_active_physics_layout": gate_active_physics_layout,
        "final_active_physics_layout": final_active_physics_layout,
        "active_physics_layout_gate_final_exact_match": True,
        "reset_drift_sample_count": int(
            reset_drift_history["sample_count"]
        ),
        "reset_drift_expected_sample_count": (
            reset_drift_expected_sample_count
        ),
        "reset_drift_all_physics_frames_sampled": (
            reset_drift_all_physics_frames_sampled
        ),
        "authored_reset_snapshot_source": (
            "environment_after_authored_writes_before_first_physics_step"
        ),
        "authored_reset_snapshot_serial": authored_reset_serial,
        "reset_pose_drift_peaks": (
            None if reset_pose_drifts is None else reset_pose_drifts[:5]
        ),
        "require_valid_contact_report": bool(
            require_valid_contact_report
        ),
        "contact_penetration_limit": (
            None
            if contact_penetration_limit is None
            else float(contact_penetration_limit)
        ),
        "initialization_contact_summary": dict(contact_summary),
        "gate_contact_summary": dict(gate_contact_summary),
        "initial_geometry_gate_passed": True,
        "scene_rebuilt_before_reset": scene_rebuilt_before_reset,
        "scene_generation": scene_generation,
        "final_reset_pose_drift_checked": True,
        "final_reset_translation_drift_limit": (
            final_reset_translation_limit
        ),
        "final_reset_rotation_drift_limit": final_reset_rotation_limit,
        "final_reset_coordinate_drift_limit": (
            final_reset_coordinate_limit
        ),
        "final_reset_max_translation_drift_m": (
            final_reset_max_translation_drift
        ),
        "final_reset_max_rotation_drift_rad": (
            final_reset_max_rotation_drift
        ),
        "final_reset_max_coordinate_drift": (
            final_reset_max_coordinate_drift
        ),
        "final_reset_state_coverage": final_reset_state_coverage,
        "gate_final_translation_delta_limit": (
            gate_final_translation_limit
        ),
        "gate_final_rotation_delta_limit": gate_final_rotation_limit,
        "gate_final_coordinate_delta_limit": gate_final_coordinate_limit,
        "gate_final_max_coordinate_delta": gate_final_max_coordinate_delta,
        "gate_final_state_coverage": gate_final_state_coverage,
        "gate_final_coordinate_exact_match": True,
        "gate_final_max_task_translation_delta_m": (
            float(gate_final_pose_delta[0])
        ),
        "gate_final_max_task_rotation_delta_rad": (
            float(gate_final_pose_delta[1])
        ),
        "final_contact_penetration_limit": (
            final_contact_penetration_limit
        ),
        "final_contact_summary": dict(final_contact_summary),
        "final_boundary_physics_gate_passed": True,
        "final_boundary_state_write": final_boundary_write,
        "final_boundary_restore_sync": True,
        "robot_hold_target_source": robot_hold_target_source,
        "post_settle_robot_hold_target_source": (
            post_settle_robot_hold_target_source
        ),
        "default_joint_state_restores": default_joint_state_restores,
        "final_default_joint_state_restore": False,
        "final_robot_hold_target_source": "accepted_settled_joint_state",
        "final_hold_target_sync": final_hold_target_sync,
        "final_robot_position_target_error": (
            final_robot_position_target_error
        ),
        "final_boundary_sleep": final_boundary_sleep,
        "final_boundary_physics_steps_after_write": 0,
        "final_boundary_task_and_joint_velocity_exact_zero": True,
        "stabilization_velocity_clamp": bool(
            stabilization_velocity_clamp
        ),
        "stabilization_velocity_clamp_scope": (
            stabilization_velocity_clamp_scope
        ),
        "stabilization_velocity_clamp_steps": velocity_clamp_steps,
        "stabilization_velocity_clamp_write": velocity_clamp_write,
        "linear_velocity_limit": float(linear_velocity_limit),
        "angular_velocity_limit": float(angular_velocity_limit),
        "joint_velocity_limit": float(joint_velocity_limit),
        "joint_velocity_limit_source": joint_velocity_limit_source,
        "task_name": str(getattr(raw_env, "task_name", "")),
        "stable_seconds_required": float(stable_seconds),
        "stable_steps_required": stable_steps_required,
        "stable_steps_observed": stable_steps_observed,
        "task_readiness_required": bool(task_readiness_required),
        "task_readiness_samples": int(task_readiness_samples),
        "task_readiness_ready_samples": int(
            task_readiness_ready_samples
        ),
        "task_readiness_consecutive_steps": int(
            task_readiness_consecutive_steps
        ),
        "task_readiness_stable_steps_observed": (
            int(stable_steps_observed) if task_readiness_required else 0
        ),
        "gate_task_readiness": last_task_readiness,
        "final_task_readiness": final_task_readiness,
        "stability_sample_seconds": float(stability_sample_seconds),
        "stability_sample_steps": stability_sample_steps,
        "stability_sample_interval_seconds": (
            stability_sample_interval_seconds
        ),
        "full_frame_pose_gate": bool(
            stability_metric == "velocity_or_static_pose"
        ),
        "stability_samples": stability_samples,
        "timeout_seconds": float(timeout_seconds),
        "pose_translation_limit": float(pose_translation_limit),
        "pose_rotation_limit": float(pose_rotation_limit),
        "final_max_task_translation_delta": (
            None if last_pose_delta is None else float(last_pose_delta[0])
        ),
        "final_max_task_rotation_delta": (
            None if last_pose_delta is None else float(last_pose_delta[1])
        ),
        "final_max_linear_velocity": float(final_max_linear_velocity),
        "final_max_angular_velocity": float(final_max_angular_velocity),
        "final_max_robot_joint_velocity": float(last_robot_motion[2]),
        "final_max_task_linear_velocity": float(last_object_motion[0]),
        "final_max_task_angular_velocity": float(last_object_motion[1]),
        "final_robot_joint_velocity_peaks": _robot_joint_velocity_peaks(
            last_snapshot,
            trace,
        ),
        "first_state_write": first_write,
        "hold_target_sync": hold_target_sync,
        "state_writes": True,
        "initialization_per_physics_step_velocity_writes": bool(
            stabilization_velocity_clamp
        ),
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }
    accept_task_origin = getattr(
        raw_env,
        "accept_stable_recording_origin",
        None,
    )
    if callable(accept_task_origin):
        acceptance = accept_task_origin(dict(report))
        if acceptance is not None:
            if not isinstance(acceptance, Mapping):
                raise TypeError(
                    "accept_stable_recording_origin() must return a mapping "
                    f"or None; got {type(acceptance).__name__}."
                )
            acceptance = dict(acceptance)
            if acceptance.get("accepted") is not True:
                raise TypeError(
                    "accept_stable_recording_origin() must return exact "
                    "'accepted': True after a successful task rebase."
                )
            report["task_recording_origin_acceptance"] = acceptance
    raw_env._physical_recording_origin = dict(report)
    if callable(accept_task_origin) and acceptance is not None:
        task_info_builder = getattr(raw_env, "_task_info", None)
        if callable(task_info_builder):
            refreshed_task_info = task_info_builder()
            if not isinstance(refreshed_task_info, Mapping):
                raise TypeError(
                    "_task_info() must return a mapping after task recording-"
                    "origin acceptance; got "
                    f"{type(refreshed_task_info).__name__}."
                )
            # Task acceptance may rebase joint/body baselines and clear hidden
            # episode latches.  Replace the pre-settle task snapshot so reset
            # metadata and the rebuilt action-0 observation describe the same
            # physical boundary.  Wrapper-owned keys remain untouched unless
            # the authoritative task info explicitly owns the same key.
            output_info.update(dict(refreshed_task_info))
    output_info["physical_recording_origin"] = dict(report)
    accepted_hook = getattr(
        raw_env,
        "_on_stable_recording_origin_accepted",
        None,
    )
    if callable(accepted_hook):
        # Semantic reference capture is read-only and occurs after the final
        # zero-step boundary write, so the returned observation and replay
        # both measure task displacement from the accepted physical origin.
        accepted_hook()
    return _rebuild_observation(env, trace), output_info


__all__ = [
    "JOINT_VELOCITY_LIMIT_ENV",
    "RecordingOriginStabilityError",
    "STABLE_ORIGIN_ENV",
    "STABLE_ORIGIN_RESET_ATTEMPTS_ENV",
    "VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS",
    "VELOCITY_CLAMP_SCOPE_GATE_ONLY",
    "VELOCITY_CLAMP_SCOPE_NONE",
    "stable_recording_origin_kwargs_for_task",
    "stable_recording_origin_kwargs_from_report",
    "stable_recording_origin_reset_attempts_for_task",
    "stable_recording_origin_enabled",
    "stabilize_recording_origin",
]
