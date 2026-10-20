# B320 collection contract

This contract is the acceptance source for the ten tasks listed below.  They
were originally specified with spreadsheet task IDs (B320-BC004, B320-KA012,
…); HandsOn maps each to one deliberate lower-snake-case runtime name.  The
mapping is given per row in the table that follows, and the rows whose runtime
name is a Core-50 task use the same identifier as the manuscript.  Later
registered B320 variants reuse the family-wide teleoperation and formal-UI
clause below while retaining their own task-specific physical contract.

## Identity, runtime, and operator contract

| Spreadsheet ID | HandsOn canonical name | Runtime specialization | Nearest proven reference | Time limit |
|---|---|---|---|---:|
| B320-BC004 | `t10_lift_carton` | ABC rigid-body USD specialization | Current D08 initialization/render/control baseline + D03 dual-hand role truth | 22 s |
| B320-BC011 | `t20_pass_and_orient_pen` | ABC rigid-body USD specialization | D03 ordered handoff + A06 signed orientation | 32 s |
| B320-BC014 | `t30_pour_seasonings` | ABC two-rigid-body USD specialization | D03 role truth + C12 multi-object state | 16 s |
| B320-BC017 | `t15_open_drawers` | Clean-room dedicated six-drawer articulation | Task-local CC0 Metal Office Desk + physical handle/contact truth | 30 s |
| B320-BC021 | `t01_align_blocks_in_midair` | Clean-room dedicated two-rigid-body runtime; final asset pending | D01 contact/state truth + G02 physical alignment truth | 60 s |
| B320-BC025 | `b320_towel_corner_synchronized_spread` | Existing E01 native surface-deformable lifecycle, task specialization | E01 detached Wuji control + G01 corrected Towl011 surface | 45 s |
| B320-KA012 | `t45_stir_pot` | ABC rigid tool specialization | C07 staged tool state + G02 hand-role contact | 40 s |
| B320-KA020 | `t32_press_soap_pump` | ABC rigid tool plus physical proxy DOF | C09 physical press + G02 hand-role contact | 30 s |
| B320-KA021 | `t05_fill_cup` | ABC articulation plus rigid payload specialization | D02 fixed articulation + D14 faucet state | 22 s |
| B320-FU001 | `b320_synchronized_double_cabinet_open` | ABC task-local two-DOF fixture specialization | D02 fixed fixture + D07 articulated furniture | 32 s |

BC017 uses the task-local CC0 Metal Office Desk runtime with three articulated
candidate drawers (`top`, `middle`, and `bottom`) on each side. Every reset
independently samples one left target and one right target; the selected left
drawer receives a cyan outline and the selected right drawer receives an
orange outline. Only this selected pair contributes to success. Contact with or
motion of a nonselected drawer remains recorded diagnostic truth and does not
terminate the episode by itself.

BC021 no longer uses the retired bottom-origin unmarked square-block asset or
its root-height/contact-history success heuristic. Its dedicated 120 Hz
runtime and procedural construction scaffold are testable, but the scaffold is
explicitly provisional and cannot report formal success. BC021 remains
unverified until the operator-selected visible source, source/license/hash
manifest, measured geometry, derived colliders, camera/contact audit, and
fresh-process episode replay are complete.

BC021 has a task-local reset home: both Wuji palm planes are parallel to the
table and to each other, both palm normals face world `-Z`, and both finger
directions face world `+X`. The physical palm origins are mirrored at
`(-0.32, +/-0.32, 1.15)` m. Block A/B reset around
`(-0.32, +/-0.20, 0.9275)` m with at most 45 mm total X displacement, 45 mm Y
displacement, and 30 degrees yaw. Every accepted sample must remain between
0.42 m and 0.68 m from its assigned arm base, preserving a 0.10 m margin to
the controller's 0.78 m physical workspace limit; the complete reset-home
robot collider envelope remains an additional rejection gate.

All tasks use Isaac Sim/IsaacLab, dual ROKAE arms, Wuji Hand 1, 30 Hz
operator control, 120 Hz physics, and the dual-arm Vive/Manus wrapper.  They
are Isaac-only tasks.  Raw 54D operator actions must be preserved.  A single
hand performing both roles, even in the correct final geometry, is a failure.

The shared operator sequence is: calibrate both trackers; establish the
left/right roles named in the spreadsheet; perform the two actions in parallel
or in the required handoff order; hold the physical success state through the
task's stability window where one is required; release only where the task
requires release. BC011 instead latches success on the first complete
final-state control sample.

BC014 keeps two 100 x 100 mm target zones and assigned hands, but is a
sprinkling task rather than a jar-placement task. At every randomized reset,
the salt jar and cyan target are sampled jointly inside the left hand's central
comfort band, while the pepper jar and orange target are sampled inside the
right hand's central comfort band. Both bands retain about 70 mm of radial
margin inside the physical arm limit, keep the full target square on the table,
and enforce at least 120 mm between each jar and its matching target. Each held
jar emits a render-only D09-style blue water-stream proxy when its cap points
downward.
The cap ray must hit the matching zone for 0.6 cumulative seconds per jar, and
both rays must hit together for 0.4 continuous seconds. The proxy stands in for
salt/pepper particles and has no collider or rigid-body effect; jar position,
upright pose, spacing, and release are not success predicates.
Each jar also has a per-episode cumulative pouring budget of 4.0 seconds. The
budget advances whenever that held jar's downward blue stream is visible,
whether or not it hits its target. If either budget exceeds 4.0 seconds before
both individual-hit latches and the synchronized-overlap latch complete, the
episode fails with a side-specific timeout reason and the collection runtime
resets the scene.

### Family-wide D08 teleoperation/UI amendment (2026-08-23)

Every registered B320 task inherits the same shared teleoperation object as
D08 through `B320TaskConfigBase.teleop`.  In formal UI/collection launches the
child environment receives the D08 real-time input profile: 30 Hz control,
`alpha=1`, zero palm translation/rotation deadbands, no sustained palm or Wuji
joint step ceiling, and D08's packet-scoped tracker-jump rejection.  The UI
also applies the ordinary A-D display profile to every B320 task: an 800 x 500
operator viewport, `HANDSON_ISAACLAB_DIRECT_VIEWPORT_DIVISOR=1`, low-cost live
display, and canonical three-camera post-render outside the online control
loop.  No B320 task may reintroduce the former family-wide two-wrist hold/rebase
or `divisor=2` profile through a task-local configuration or a later UI
environment override.

Rigid B320 tasks also keep the source/D08 ROKAE follower configuration instead
of applying the former B320-only `570/450/160` stiffness override, and they do
not enter the extra B320 palm/workspace target limiter. The independent D03
contact-sensitive target limiter remains task-local to D03.

This common input/UI alignment does not weaken or replace task-specific
physical safety.  Each task keeps its authored reset and reach bounds,
collider/contact validity gates, articulation limits, task-frame role checks,
force and velocity failure predicates, stability windows, and physical replay
state.  In particular, BC004 keeps its carton-specific force/velocity barrier
and 12 N formal failure ceiling.  That barrier evaluates each hand
independently using the hand's actual contact patch. Current force, recent
positive force rate, and requested inward displacement predict the next-frame
load; if it would exceed 10 N, the complete compressive horizontal velocity
component is rejected while lift, tangent motion, rotation, outward release,
and the other hand remain unchanged. BC004 also retains the collision-clear
lateral arm home for the 0.47 m-wide carton and the existing eight-partition
B320 contact solve.

BC025 is an explicit end-to-end exception to any stronger equivalence claim.
Its task config and formal UI launch receive the common D08 teleoperation and
display values, but its E01-derived detached-Wuji surface-deformable runtime,
controller/solver lifecycle, reset truth, and replay scope remain distinct.
Therefore BC025 is not claimed to have D08-identical action-to-physics or
rendering behavior merely because the shared configuration values match.

Beyond the family-wide input/UI profile, BC004 separately retains its proven
D08-aligned rigid-runtime baseline: 120 Hz TGS physics, absolute full-pose DLS
IK (`lambda=0.04`), enhanced determinism, deferred-arm collection, source
camera/light framing, and the 0.8 x 1.6 m neutral worktable appearance/material
baseline. Dataset camera framing remains identical to D08, but the three
dataset cameras and complete robot visuals are generated after online
collection so they cannot stall the 30 Hz control loop. Its prepared USD uses
the task-local `d08_aligned_v1` cache namespace so a deployment cannot silently
reuse the former common-B320 scene.

The carton has one authoritative spawn/tuning specification: mass 0.75 kg,
static/dynamic friction 1.25/1.0, linear/angular damping 0.40/2.00, sleep
threshold 0.020, 32/4 body solver iterations, 1 mm contact offset, speculative
CCD, and a 12 N inclusive side-force ceiling. Runtime post-compose authoring
must reproduce these values instead of replacing them with a second set.

### BC011 grip and final-orientation amendment (2026-08-24)

BC011 binds a task-local max-combine pen material with static/dynamic friction
`2.5/2.0`, matching the proven thin-object grasp profile without changing the
shared hand material or any other B320 object. The ordered handoff must start
with a right-only physical touch, pass through a simultaneous physical
two-hand touch, and then reach a left-only physical touch after right-hand
contact reaches zero. Every hand-contact predicate aggregates contact over the
complete pen collider; any positive readable contact counts, with no minimum
force, permitted/forbidden contact region, or pen-speed gate. The final state
requires positive left-hand contact while the pen is
lifted at least 10 mm and pen-table contact is at most 0.02 N. The selected
asset's authored pen-tip axis must be at 90 +/- 2 degrees from the world
horizontal plane, equivalently no more than 2 degrees from world `+Z`. The
black pen's tip is local `-X`; The Pen's tip is local `+X`, and reset-time
variant selection determines which axis is evaluated. Once simultaneous
two-hand contact has been latched and the pen has become airborne, two
consecutive positive pen-table-contact 120 Hz substeps latch
`bc011_pen_dropped`; the episode terminates and resets. The first control frame
satisfying the complete final state terminates successfully; there is no pose
snap, scripted freeze, force-region gate, speed gate, or multi-frame hold.

## Asset provenance

Most legacy source directories were copied byte-for-byte from host `hp-15`
under `/mnt/pfs/luotianwei/dexverse-scenesmith/data`. BC014 and KA021 instead
use the publicly licensed downloads identified below. Each
aggregate digest is SHA-256 over the sorted list of `sha256sum` rows for every
file in that source directory.

| Asset key | Local bundle | Aggregate SHA-256 |
|---|---|---|
| `dv_carton` | `sim/envs/usd/b320_bimanual_benchmark/assets/dv_carton` | `5d7d738d184acbf32080023021ebc7d79df1c86611fd1bacbd66ea48868574b6` |
| `dexverse_pen` | `.../assets/dexverse_pen` | `70cd998340b0c2b2c0092741b275c1678faa7d17559b506d82b29f00b4d6086b` |
| `cc_by_anthony_yanez_salt_pepper_shakers` | `.../assets/cc_by_anthony_yanez_salt_pepper_shakers` | `728bbba4705b8d671941a59aee09cea01add190a843b44a582c802f362bcefac` |
| `dv_dutch_oven` | `.../assets/dv_dutch_oven` | `b11acb4260bb822584234eedd3f1ea80d2fcada3ee112829660fddd70edea727` |
| `dexverse_authored_spoon` | `.../assets/dexverse_authored_spoon` | `d3cdb721ea12f82fbcb342aeb0ee36d449d60729538104d491bd9745c44bb555` |
| `dexverse_authored_sponge` | `.../assets/dexverse_authored_sponge` | `1cdf56103878aea6f42dfbd1df4eec489aebce2ce55cfbe4302db171c475f14d` |
| `dexverse_dish_soap` | `.../assets/dexverse_dish_soap` | `3159190e10f8612799543706f0d96df27d6184e7ba51bd46975ca7014b85c646` |
| `cc_ka021_hs005_kitchen_anilai_white_mug` | `.../assets/cc_ka021_hs005_kitchen_anilai_white_mug` | `f7f415b67a23ab51e5b83100b43ef52f17332b0a599e7ceab46a53fb5c29cec2` |
| `dv_double_cabinet_44781` | `.../assets/dv_double_cabinet_44781` | `04fa5e601c0149c526c949e4c6464a20f999dfb156c930f8a7138f0bc07f57dd` |

KA021 reuses the complete realistic kitchen composition reviewed for HS005:
Visthétique's CC BY 4.0 **Modern Apartment** kitchen, its textured open sink,
high-arc faucet and drain. HS005's source open-basin triangle-mesh collision is
retained for provenance but disabled in KA021 because it overlaps both Wuji
hand home regions. A low analytic basin-floor collider supports the randomized
cup during filling; a separate collision-enabled counter pad on the left
counter, outside the visible basin and drainboard, supports the post-fill
 placement so a released mug cannot fall through the visual sink. The pad and
 its highlighted marker are raised to the HS005 counter surface height. The
 source faucet is a single connected mesh, so the KA021 builder
deterministically partitions its original vertices, faces, and UVs into fixed
body and moving side-lever derivatives. HandsOn adds only the fixed-base
articulation, revolute joint, physical lever collider, outlet metric and
render-only water stream. The manipulated cup is Ani 3D's CC BY 4.0
**Ceramic White Mug** from Sketchfab; its original glTF mesh and full
base-color, normal, metallic-roughness, and emissive texture closure are
preserved without a color override. The mug has an open-cavity analytic wall
collider set (four thin wall segments, three handle capsules, plus the recessed
flat foot), so the gripper can reach into the cup instead of colliding with a
solid cylinder while the mug handle remains physically present.
KA021 lifts only the two initial hand TCPs by 80 mm using
the HS005-verified collision-clear arm home; scene and object height rules are
unchanged. The mug starts in the sink basin and keeps the formal reset
randomization of ±35 mm in XY and ±12 degrees in yaw. Exact external HS005
paths, source hashes, attribution and the authorship boundary are recorded in
`asset_manifest.json`.

The regenerated mug has one coordinate contract across rendering, collision,
reset and success. Its asset root and scene reset are identity transforms; the
source mesh's +90-degree X correction remains on a visual child, while the
analytic open wall segments and recessed foot are authored directly along local
+Z. The mug is upright exactly when its root local +Z is within 15 degrees, inclusive,
of world +Z. Both the Visthétique kitchen and Ani 3D mug sources are licensed
under CC BY 4.0; the replacement faucet visual is the BlenderKit “Kitchen
Faucet” derivative used under the BlenderKit Royalty Free license, with its
source UUID, version, hash, and attribution recorded in the runtime manifest.

The faucet starts closed at `q = 0`. Its downloaded BlenderKit handle follows
the source `Empty`/HandleBase hinge: the revolute joint keeps the `Y` axis token
but rotates its local joint frame with the model yaw, so the visual lever and
its cartridge remain coaxial. Negative travel from 0 to -45 degrees opens
toward the robots with a strong world -X component. Open fraction is
`clip(-q / radians(45), 0, 1)`; the 70-percent success threshold is therefore
-31.5 degrees. Left-right lever motion is a temperature adjustment, not opening
travel, and cannot satisfy the flow gate.
The fill phase requires the faucet to remain open and all cup, contact, flow
and joint conditions to hold for 15 consecutive 30 Hz control frames (0.5
seconds). The terminal phase exposes two independent conditions: the upright
mug must be parked on the explicit counter support outside the sink, and the
faucet joint must separately return to its closed range. Closing the faucet
cannot satisfy the fill condition or the parking condition.

The towel task deliberately reuses the already vendored and corrected
`g01_deformable_asset_inspection/Towl011` bundle.  The supplier's closed thick
mesh is visual provenance only; simulation uses the existing 41x41
mid-surface cloth because a volumetric rubber-pad representation is physically
wrong for spreading and folding.

The spoon and sponge GLBs receive deterministic derived USD meshes. Their
source GLBs remain in the bundle and the converter records each source hash.
Derived geometry is not allowed to change scale, topology, or material color
silently.

BC021's reviewed final-asset candidate is Quaternius **Stone Block** under
CC0-1.0 ([source](https://poly.pizza/m/ZKvgpQwJiY)). It is not a vendored or
formal source until the user downloads and selects the exact GLTF/FBX file;
only that selected file may establish the final geometry and asset digest.

BC014 uses Anthony Yanez's **Salt And Pepper Shakers** under CC BY 4.0
([source](https://sketchfab.com/3d-models/salt-and-pepper-shakers-5f2adebdda754e40915d1330a61fa70c)).
The bundle retains the complete glTF closure, all four downloaded textures,
the original license, attribution, conversion metadata, and exact hashes. Its
Isaac Sim 5.1 wrappers split the source `Salt` and `Peeper` nodes into two
independent 96 mm rigid objects. Per jar, the visible bottle and cap meshes
provide two convex-hull colliders; the visible contents remain render-only.
No collider or geometry from the retired white placeholder jars is reused.

The retained PartNet furniture bundle was inspected rather than guessed:

- 44781's two real door DOFs are `target_joint_revolute` and `joint_11`,
  both with 0-90 degree source travel.  Dummy canonical slots are locked.

## Shared physical contract

- The first PhysX state is authored before `SimulationContext.reset()` with
  every dynamic body supported, separated from both complete hand envelopes,
  and inside the recording workspace and camera frusta.
- Tabletop tasks use a non-moving work surface. BC004 authors it as a
  kinematic rigid body so Isaac 5.1 can expose a GPU-filtered carton/table
  contact channel; the other rigid tabletop tasks keep their static surface.
  The integrated furniture task uses a task-local kinematic fixture root with
  exact source visuals, simplified validated collision proxies, and only the
  two named physical DOFs.
- The eight integrated rigid/articulated tasks use only `self._random`. BC004 carton XY
  jitter is bounded by 25/20 mm with fixed authored yaw; other tabletop XY
  jitter is bounded by 35 mm, yaw by 12 degrees, and target-joint initial
  positions by 10-30% of safe travel. Fixed furniture roots do not move;
  their active joints provide the seeded variation. Other task samplers reject
  overlap, unsupported, out-of-reach, and already-solved states. BC004 instead
  uses its fixed conservative 25/20 mm box, whose extrema remain fully on the
  D08-sized tabletop and inside the collision-clear spread-arm workspace.
- The R3 towel runtime deliberately keeps its complete nodal physical reset
  deterministic because the surface solver cannot yet restore a randomized
  nodal state with formal replay equivalence.  Requested randomization still
  changes visual materials, lighting, and cameras through a seed-derived RNG.
- Reset clears contact latches, ordered stages, release/threshold frame
  indices, force peaks, cumulative stir angle, coverage/tension metrics,
  assists, success counters, failure flags, DOF velocity, body velocity, and
  cached reference poses.
- Every dynamic rigid body and task articulation root/DOF is in reset and
  PhysX physical truth.  Goal markers, table, and fixture visual roots are
  static/kinematic and registered separately.
- Missing hand-role contact views, missing colliders, missing named DOFs, or a
  source/derived-asset hash mismatch must fail closed before collection.

## Interface contract

All integrated rigid/articulated B320 tasks expose these state keys with fixed shape and
`float64` dtype: `tcp_pose(14)`, `gripper_pose(32)`,
`left_role_pose(7)`, `right_role_pose(7)`, `fixture_pose(7)`,
`goal_pose(7)`, `task_metrics(24)`, `contact_metrics(12)`,
`stage_state(8)`, `articulation_q(4)`, and `table_delta_height(1)`.
Validity bits live in `contact_metrics`/`stage_state`; unavailable physical
contact is never interpreted as zero-force success.

The towel specialization exposes the same robot keys plus fixed-cardinality
towel corner/keypoint, AABB, target-frame, tension, coverage, contact-role,
release, and stability fields.  Surface nodes and hidden contact-assist state
remain in the recording schema.

## Task-specific success and failure

| Task | Required physical success | Mandatory false-positive rejection |
|---|---|---|
| Carton carry | Physical carton pose and carton-table contact channels are readable; the lowest of all eight transformed carton bounding-box corners is at least 10 mm above the fixed table top; no carton-table contact exceeds 0.02 N; both hands support opposite lower side zones; both side-force peaks are <=12 N; every 120 Hz physics substep satisfies the predicate for 3.0 continuous seconds (360 substeps / 90 control frames) | tabletop/horizontal motion, root-height-only false positive, one-hand or same-side support, unreadable contact truth, any invalid/touchdown substep resetting the timer, force >12 N, drop; tilt remains diagnostic and is not a hidden success gate |
| Pen handover | contact may occur anywhere on the pen and any positive readable force counts; right-only physical touch occurs first; simultaneous left/right physical touch occurs second with no speed gate; left-only physical touch with zero right contact occurs third; finally the left hand remains in positive physical contact while the pen is lifted >=10 mm and pen-table contact is <=0.02 N; the selected asset's authored tip axis is 90 +/- 2 degrees from horizontal (<=2 degrees from world +Z); the first complete control sample latches success | simultaneous or left-first initial contact, no overlap, no left-only transfer, wrong order, unknown asset-tip mapping, unreadable contact truth, zero left contact, any positive right residual contact, table-supported pen, horizontal/downward tip, axis error >2 degrees; only after simultaneous two-hand contact and airborne pickup, positive pen-table contact lasting two 120 Hz substeps latches `bc011_pen_dropped` and resets the episode; releasing both hands in midair alone is not a drop |
| Salt/pepper | left handles salt and right handles pepper; each cap-directed proxy ray hits its current randomized matching 100 mm square zone for at least 0.6 cumulative seconds; both rays hit together for at least 0.4 continuous seconds; complete before either jar accumulates more than 4.0 seconds of pouring | wrong-hand ownership, cap not directed downward, either ray misses its current matching zone, insufficient individual duration, insufficient synchronized overlap, or either per-jar pouring budget exceeding 4.0 seconds before completion; jar placement and release are not predicates |
| Dual drawers | Every episode independently selects one of the top, middle, or bottom drawers per side; left uses the cyan-outlined target and right uses the orange-outlined target; only the selected pair counts; each selected drawer crosses strictly beyond 60% with assigned-hand causal handle contact; the two qualifying 120 Hz crossings are <=144 substeps (1.2 s) apart; both remain beyond threshold for two consecutive samples | unreadable selected indices, selected DOF travel, or hand/handle contact truth; wrong/cross-hand contact; exactly 60%; qualifying crossings outside 1.2 s; a nonselected drawer satisfying a selected target; nonselected motion by itself remains diagnostic and is not a terminal condition |
| Midair blocks (asset pending) | current left-A and right-B physical contacts are readable and crossed contacts are clear; the lowest transformed point of each block is at least 80 mm above the table; after left-A is airborne and level (<=10 degrees tilt), its pose becomes the reference and the left hand command is locked; the reference maximum translation stays strictly below 20 mm; marked-face normals oppose within 5 degrees, lateral error is strictly below 5 mm, signed axial gap is -2 to +3 mm, and actual target-patch AB contact is present; both blocks satisfy absolute and relative velocity limits for 48 consecutive 120 Hz substeps (0.4 s) | provisional/unreadable geometry or pose/velocity/contact truth, tabletop or near-miss mating, wrong/cross hand ownership, target plus non-target contact, non-target face/edge impact, common high-speed motion, deep penetration, drop, workspace exit, or reference excursion >=20 mm |
| Towel spread | hands have contacted different top-corner regions; end distance remains within +/-10% target and stretch <=15%; at least 3/4 corners enter the goal; projected target coverage >80%; both hands release within 1.0 s; stable 0.5 s | same-end grasp, overstretch, self/scene penetration, numerical divergence, dropped corner |
| Dutch oven stir | left hand continuously stabilizes the left handle; right hand owns spoon; spoon head crosses six ordered angular sectors and accumulates >650 degrees; pot translation <25 mm; stable completion | left role missing, spoon dangerous-area grasp, checkpoint skipping/backtracking, pot slip |
| Sponge/soap | left hand stabilizes sponge; right owns bottle; nozzle projection lies in sponge AABB and is 30-100 mm above it; physical dispense proxy reaches threshold; stable hold | sponge drift >20 mm, no role contact, nozzle outside target, proxy not actuated |
| Cup/faucet | the normalized mug asset, its analytic body/foot colliders, reset pose, and success predicate all use local +Z as up; the mug starts at identity orientation with its collider foot on the 0.89 m basin floor; current left-mug and right-lever contacts are both readable and >=0.10 N; mug opening is within 30 mm of the outlet projection; local +Z is within 15 degrees (inclusive) of world +Z; outlet is 40-190 mm above the mug rim; the faucet starts closed and its single flow DOF follows the downloaded handle's yawed source hinge (negative 45-degree travel with a world -X opening component), reaching >=70% of 45 degrees; the visible stream intersects the mug opening; all conditions hold continuously for 0.5 s | historical-only contact, one hand only, tipped/misaligned/upside-down mug, invalid outlet height, inward or left-right temperature motion, unreadable or wrong joint, open joint with stream missing the mug, oversized fixture collision blocking the source lever grip, or scene-jammed actuation |
| Double cabinet | left/right contact assigned doors; both named DOFs reach >=80%; first threshold times differ by <=1.0 s; stable 0.4 s | wrong hand/door, serial completion, rollback, limit violation |

`succeed`, `terminated`, `truncated`, the task time limit, failure, and manual
reset remain separate. Time-limit expiry truncates; physical invalidity may
terminate; success is debounced by the stated consecutive window except BC011,
which terminates on its first complete final-state sample. BC004 remains the
substep-qualified exception: a bad 120 Hz sample resets its airborne timer, but
does not permanently prevent a later fresh three-second attempt.
Recorded-drive and global-IK replay own only the pre-substep command writer;
BC004's read-only post-substep physical observer remains active during replay.

## Cameras, recording, and replay

All tasks require three distinct canonical streams: `main`, `wrist_left`, and
`wrist_right`, in live and deferred-arm modes.  Main framing must include both
role objects and goal/fixture over multiple seeds.  The UI uses explicit
task labels, the repository logo only where no task-specific thumbnail exists,
and task-specific operator steps.

Record raw actions, exact low-level drive targets where supported, current and
next state, task rigid/articulation truth, seed/reset ordinal, asset hashes,
contact-role latches, stage/release/threshold frame indices, force peaks,
cumulative stir sectors/angle, towel corner/coverage/tension state, success
counter, terminal timing, and camera files.

The integrated rigid/articulated tasks target formal fresh-process
full-trajectory physical replay, but are not marked certified until a saved
episode and certification report bind the exact code/asset hashes. BC017 uses
that ordinary replay path and records both selected target indices so replay
preserves the episode's independently sampled pair. The towel task supports
collection and diagnostic/state-guided replay; formal
rigid-body replay is not claimed for surface nodes unless the deformable
certifier proves the full nodal state and solver lifecycle.

BC004 is registered and enabled for formal collection. Its first accepted
Isaac Sim 5.1 / IsaacLab v2.3.2 episode must still prove the 360th continuous
substep terminal frame, and publication remains subject to fresh-process replay
and the normal episode verification gates below.

## Required evidence before production collection/replay certification

- Registration audit PASS for every canonical task and the all-task audit.
- Syntax/import, focused UI/identity, interface, deterministic reset,
  multi-seed layout, observation/action, false/true/stability semantics,
  camera, recording-schema, and replay-contract tests.
- `launch_isaac=False` reset/step for all ten tasks.
- Real-asset Isaac headless construction/reset/step for all ten tasks,
  including exact collider/contact/DOF fail-closed logs.
- Repeated same-seed equality and multiple-seed support/reach/visibility
  checks.
- One complete recorded episode per task with manifest/schema/Zarr/cameras
  inspected; fresh-process certification for each rigid/articulated task
  before its formal replay flag or upload gate is enabled.
