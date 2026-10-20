# Stable PhysX recording origin

This document describes the recording boundary used by persistent Isaac Sim
collection and fresh-process physical replay. The goal is to make `state[0]`
the first state that the recorded action sequence actually observes, after the
robot and task state pass strict velocity, geometry, contact, and coverage
gates. The accepted boundary is then written once with exact-zero velocity;
there are no task-state writes after `action[0]` begins.

## Why reset completion is not a recording boundary

An environment reset can finish while:

- an actuator is still pursuing a stale position target;
- a finger is oscillating in contact;
- a task object is falling or sliding;
- an articulation or deformable object still has velocity;
- PhysX contacts are still converging.

Equal positions with unequal velocities are unequal physical initial states.
Waiting for a fixed wall-clock interval also does not help when a stale target
continues to drive a joint.

## Reset and settling sequence

`stabilize_recording_origin()` performs this sequence when
`HANDSON_STABLE_RECORDING_ORIGIN=1`:

1. Capture the exact public task-state setter arguments before PhysX's first
   reset step. This avoids treating a stale asynchronous GPU tensor readback
   as the authored reset state.
2. Read the live reset state through `PhysxGroundTruthTrace`, restore the
   authored default robot joint state when supported, align drive/command
   targets, and write exact-zero velocities once.
3. During the initial-settle phase, keep the fixed robot target and execute
   native PhysX steps with `render=false`, no wall-clock throttle, and the
   configured physics `dt` unchanged. Sample every physics frame for
   authored-state drift, contact validity, penetration, and active-entity
   coverage; no per-frame pose or velocity clamp is applied.
4. After the initial settle, capture the physically reached robot joint pose
   once as the next fixed hold target, reset the controllers at that phase
   boundary, and write only exact-zero velocities. A/B/C bodies remain awake,
   so the validated contact manifold and solver warm-start state are preserved;
   the existing D-series path retains its explicit sleep transaction.
5. During the strict-gate phase, keep that captured target fixed and continue
   normal, unclamped PhysX steps. Require real task and robot velocities to
   remain within the gate for a continuous interval. A01, A06, B01, B04, C02,
   C04, C07, C08, and C12 additionally permit a task-only static-pose branch:
   every task pose and valid task-articulation DOF must remain within the
   approved coordinate deltas on every physics frame while robot velocities
   remain strict. This branch is read-only; it never writes a pose, clamps
   velocity, or puts a body to sleep.
6. Capture the accepted robot pose as the final fixed target, reset the
   controllers once more, then align the drive target and perform one final
   velocity-only write and exact-zero readback without an intervening physics
   step. Never rewrite poses or joint positions at either settled boundary.
   Revalidate the
   complete active layout, all task/robot coordinates, contact signature,
   penetration, and finite values at this boundary.
7. Reset the environment step and success/failure counters, rebuild the
   observation from the settled readback, and use it as `state[0]`.
8. Start `action[0]`.

Current A/B/C production records
`stabilization_velocity_clamp = false` and
`stabilization_velocity_clamp_scope = none`. Compatibility code can still
describe historical clamped initialization, but it is not the path used for
new A/B/C samples. D-series keeps its existing sleep transaction and three-
second post-settle timeout, but D-series changes are outside this checkpoint.
Any task assist that directly writes task or robot state during an action frame
is journaled in the origin report and makes the sample fail publication.

## Motion included in the gate

The controlled-origin report covers:

- robot root linear and angular velocity;
- every valid robot joint velocity;
- every valid task rigid-body linear and angular velocity;
- every valid task articulation root and DOF velocity;
- every valid deformable node velocity when a deformable object is present;
- A01, A06, B01, B04, C02, C04, C07, C08, and C12 full-frame task-pose and
  task-articulation-DOF deltas.

If a robot value exceeds its limit, the continuous-stability counter returns
to zero. The approved static-pose counter also returns to zero on any excessive
pose delta.

## Current limits

| Value | Default |
|---|---:|
| Root/rigid/deformable linear speed | `0.001 m/s` |
| Root/rigid/articulation angular speed | `0.002 rad/s` |
| Robot joint speed | `0.001 rad/s` |
| Continuous stable interval | `0.5 s` |
| Generic API timeout | `30 s` |
| Collection initial settle | `1 s` |
| A/B/C collection post-settle timeout | `39 s` |
| A/B/C total initialization maximum | `40 s` (`1 s + 39 s`) |
| D-series post-settle timeout | `3 s` |

The task-only static-pose branch is restricted to A01
(`t26_place_fruit_in_bucket`), A06 (`t19_orient_cup_upright`), B01
(`t04_cover_pitcher`), B04 (`t44_stack_blocks`), C02 (`t36_scoop_marbles`),
C04 (`t50_water_plant`), C07 (`c07_pitcher_pour_cup`), C08
(`c08_tongs_ice_cubes`), and C12 (`t11_line_up_blocks`). Each uses at most
`1e-6 m` consecutive translation and `1e-5 rad` consecutive rotation, sampled
on all 60 physics frames in the required `0.5 s` window at 120 Hz. C07 also
constrains the rotationally symmetric proxy spheres while leaving all three
translation axes dynamic and wakeable. C08 includes its live tongs hinge DOF,
so a stationary root cannot hide an opening or closing jaw. Its authored
cradle pose and load-bearing box contacts are a measured passive equilibrium;
the glass uses a flat physical base at a tangent reset height. Action safety is
checked against the exact oriented collider boxes and their swept corners, not
an axis-aligned approximation. These profiles handle measured, non-integrated
GPU-TGS contact-correction velocity readbacks while retaining a full-frame
transform proof. Approval of the pose fallback does not bypass the robot-
velocity, reset-drift, contact, penetration, finite-value, or final exact-zero
checks.

At 120 Hz, the default stable interval requires 60 consecutive physics
substeps. A/B/C inspects every physics frame and exits immediately after those
60 continuous stable frames; the 40-second value is a fail-closed maximum, not
a mandatory wait. These substeps run unrendered and without a real-time sleep,
so they advance as quickly as the GPU and scene allow. This is simulated time,
not a promised 40x wall-clock multiplier, and the physics `dt` is never enlarged
to make initialization appear faster. B02 retains its explicit configured-rate
override; the same configured `dt` must be recorded and replayed exactly.

The checked-in task-specific limits inherited by this branch are:

- B02 (`b02_nut_bolt_assembly`): robot joint speed `0.008 rad/s`;
- C05 (`t14_move_and_click_mouse`): articulation angular speed `0.003 rad/s`;
- C06 (`t23_photograph_logo`): robot joint speed `0.003 rad/s`.

C06's left thumb remains in light reset contact with the camera, and the
measured solver jitter is approximately `0.0026 rad/s`. A `0.001 rad/s` joint
limit did not pass after 30 simulated seconds; the explicit near-zero C06 limit
did. These exceptions do not weaken the final action-0 boundary, which must
still read back exact-zero task and robot-joint velocity.

The resolution order for the robot joint limit is:

1. `HANDSON_STABLE_JOINT_VELOCITY_LIMIT`;
2. a function argument;
3. `env.stable_recording_origin_joint_velocity_limit`;
4. a validated task default;
5. the global default.

The selected value and its source are saved in
`physical_recording_origin`.

Exact robot `qvel == 0` is deliberately not required while accumulating the
continuous stability window: floating-point contact, friction, and constraint
solvers can retain tiny numerical velocities in a visually static scene. The
final velocity-only action-0 boundary is stricter and must read back exact-zero
task velocity and exact-zero robot joint velocity without another physics step.

## Failure behavior

Collection fails before `action[0]` when:

- native Isaac physics stepping is unavailable;
- no robot or task-object velocity readback is available;
- an expected active entity is missing, hidden, duplicated, or non-finite;
- authored-reset drift or solved-contact penetration exceeds its limit;
- the final contact signature is invalid or changes at the action-0 boundary;
- a limit or duration is invalid;
- motion does not remain below the limits before the timeout.

Timeout errors include the final maximum linear, angular, and robot joint
speeds plus the fastest robot joints.

## Replay contract

### Source collection and publication lifecycle

Source collection and certification intentionally use different lifecycle
contracts:

1. The collection UI keeps one Isaac Sim/Kit process alive for a batch so RTX,
   device input, and application startup remain warm. The first reset uses the
   newly constructed scene. Before every later reset, the recorder releases all
   camera/tensor/articulation views, clears the Isaac Lab `SimulationContext`,
   creates a new USD stage and PhysX `PxScene`, and reconstructs the complete
   task environment in the same Python object. It then performs the normal
   reset and stable-origin sequence with a unique deterministic seed. This
   removes contact manifolds, friction anchors, warm-start impulses, island and
   sleep history without paying the cost of restarting Kit. The journal and
   replay manifest bind the rebuild marker and monotonically increasing scene
   generation to the reset ordinal. Certification repeats those same rebuild
   boundaries. The source process is still `persistent_process`; it must not be
   described as a fresh process. Each candidate also records the
   `collection_run_seed`, reset ordinal, `episode_reset_seed`, complete RNG
   state, raw commands, per-substep drive targets, complete declared physical
   trajectory, and initialization history.
2. A completed candidate is written under the hidden
   `.physical_replay_pending` staging area, not the published dataset.
3. After the source recorder has fully exited, three mutually independent,
   newly started certification processes each reconstruct the episode from its
   recorded seed and manifest. E/F candidates additionally wait for any
   replay-Zarr-mutating offline postprocess to finish first. All three must pass
   the full control-frame and physics-substep comparison before the candidate
   is atomically promoted into the published dataset.
4. For an ordinary successful A-D Isaac episode, the collection service
   finishes deferred full-arm rendering and the final v6 all-camera/marker
   repair, then automatically uploads the complete, visible episode; no manual
   quality-review click is required to start that upload. Automatic
   eligibility requires an exact collection-metadata/replay-manifest binding
   for task, backend, positive integer step count, certification requirement,
   and deferred-certification mode. The uploader re-runs the physical-replay
   and non-physical postprocess gates immediately before packaging the
   episode. E/F collection keeps its separate postprocess/review policy.
5. Any failed, missing, inconsistent, or stale proof leaves the candidate in
   pending/quarantine. Pending transaction paths are invisible to dataset
   discovery, and the upload gate rejects them even when automatic upload is
   enabled.

The detached certification launcher is idempotent. Recorder/worker crashes and
UI restarts are recovered by scanning the hidden pending root after collection
becomes idle. A candidate may receive one automatic worker-crash recovery; a
second crash records `recovery_exhausted` and stops retrying, so recovery cannot
turn into a restart loop. A genuine physical-replay failure is terminal and is
never retried automatically. Malformed or temporarily unreadable recovery
inputs use a fingerprinted 60-second backoff instead of being relaunched on
every maintenance tick.

For A-D detached-hand collection, `isaac_arm_render_deferred` is repeated in
the certified replay manifest and collection metadata. New data requires those
declarations to agree. Legacy deferred manifests that predate the field are
handled fail-closed as deferred: they still need a completed arm-render status
and can never be interpreted as full-arm live video merely because the field is
absent. A `complete` status is valid only when it contains the non-empty size
and SHA-256 proof for exactly `videos/main.mp4`, `videos/wrist_left.mp4`, and
`videos/wrist_right.mp4`, and all three current files still match that proof.
Missing or changed artifacts are rendered again and remain upload-blocked.
The subsequent v6 camera pass deliberately replaces those same three canonical
files. It therefore refreshes the arm artifact proof while holding the episode
lock; a valid newer v6 camera proof is also accepted as the final-file proof so
an older worker cannot create an arm-render/camera-repair loop.

Every uploadable episode also has a sibling-side cross-process publication
lock. Deferred arm rendering, wrist-camera replacement, the final upload gate,
archive construction, all cloud PUTs, and the local cloud-ack write serialize
on that lock. Camera replacement first publishes a fail-closed `replacing`
marker and clears it only after all canonical videos have been installed. An
uploader that cannot acquire the lock performs zero PUTs and retries later, so
it cannot package a mixture of old and newly replaced videos. The arm worker
passes the exact held lock descriptor to its renderer with `pass_fds`; the
child verifies the descriptor and lock-file inode instead of trusting a PID or
a generic "lock busy" observation.

Before the first canonical-video replacement, any previous cloud
acknowledgement is marked stale. New upload records include SHA-256 identities,
and the automatic scanner compares the current manifest, proofs, metadata, and
three canonical videos with the acknowledged cloud-file identities. A changed
final artifact is re-uploaded to the same episode prefix instead of being
skipped merely because an older `status=uploaded` record exists.

The legacy `fresh_process_certified_collection.py` supervisor remains available
as a fallback that uses one fresh source process per attempt. It is useful for
isolating source-process-history issues, but it is not the default UI path.

Fresh-process articulated replay must:

1. use the same code, assets, configuration, reset seed, and replay-relevant
   environment variables;
2. run the same stable-origin sequence;
3. stop at the freshly rebuilt action-0 boundary without restoring the
   recorded t0 state, because that restore would create a different
   contact/cache history;
4. install each recorded raw command without rerunning IK and apply the
   recorded public robot drive targets at every PhysX substep;
5. allow PhysX to produce the task-object and contact trajectory without
   per-frame state restoration;
6. compare the source and replay physical fields at every recorded state.

This drive-target path applies only when the recorded schema declares target
replay support. E01 currently remains an explicit legacy raw-action exception
and must report its acceptance result separately.

Per-frame task-object or robot-state writeback after action execution begins is
not physical action replay. Legacy reports retain their explicit initialization
scope: a missing scope on an enabled clamp means `gate_only`, while all new
collection reports must use `false / false / none` for the clamp flag,
per-step-write flag, and scope, with zero clamp steps and no clamp-write record.
Enabled clamps remain loadable for historical diagnostics but cannot pass v4
publication certification.
Comparing two generated replays to each other without comparing either to the
source is only a repeatability check.

## Two independent acceptance results

Physical replay and task outcome answer different questions:

- **Physical replay:** did the same initial state and actions produce the same
  physical trajectory?
- **Task success:** did that trajectory complete the task?

A failed source trajectory can be reproduced exactly. That proves
deterministic execution, but it is not a successful training demonstration.

A production success sample should pass all of these checks:

1. stable origin completed;
2. source trajectory satisfied the task success predicate;
3. fresh-process replay also satisfied the task success predicate;
4. all required physical fields passed the per-state tolerance;
5. video, action, observation, and timestamp axes are aligned.

Passing candidates must additionally have three independent fresh-process
proofs before atomic publication. “100% physical replay” refers only to the
non-empty published set after this gate; it is not a claim that every candidate
from the persistent source process passes naturally.

## Historical validation performed on 2026-07-31

The stable-origin sequence was tested in separate source and replay processes
on four C04 water-plant trajectories and three C06 bimanual-photograph
trajectories.

| Sample | Strict states | Maximum absolute error |
|---|---:|---:|
| C04 demo19 | `306/306` | `0.0` |
| C04 demo10 | `216/216` | `0.0` |
| C04 demo20 | `178/178` | `0.0` |
| C04 demo2 | `287/287` | `0.0` |
| C06 demo91 | `256/256` | `0.0` |
| C06 demo40 | `319/319` | `0.0` |
| C06 demo85 | `414/414` | `0.0` |

Aggregate result: `1976/1976` strict states passed with a `1e-4` audit
tolerance, and the observed maximum error was `0.0`.

That experiment predates the current natural-settle, velocity-only awake
boundary. It must not be presented as an end-to-end validation of the current
contract; the current contract needs its own Isaac task-matrix run.

These validation trajectories reused actions from legacy recordings after
changing the initial state. The newly generated source trajectories therefore
failed their tasks, and their independent replays reproduced those failures
exactly. They are determinism/regression fixtures, not certified successful
training samples.

## Dataset-generation boundary

Current fresh-scene PhysX collection data is identified by an explicit,
redundant contract. A current episode directory has this form:

```text
<task>_demo_<index>_<timestamp>__physx_v3/
```

It also contains:

```text
DATASET_PHYSX_V3.json
```

The same generation is stored in:

- `replay.zarr` root attributes;
- `replay_manifest.json`;
- `collection_meta.json`;
- `DATASET_PHYSX_V3.json`.

The canonical generation value is `physx_fresh_scene_v3`. The marker is
emitted only when the recorded stable-origin report uses
`continuous_current_target_all_velocity_settle`, has no per-frame state
writes, uses a velocity-only boundary, satisfies the task-family sleep/warm-
start contract, checks every A/B/C physics frame, shows that the full
consecutive-stability interval passed, and the
recorder has enabled PhysX ground-truth capture, complete deferred session
history, and an in-process fresh PhysX Scene rebuild for every later reset.

The older `__stable_v2` / `DATASET_STABLE_V2.json` generation remains readable
but is deliberately not renamed or promoted to `__physx_v3`: it predates the
fresh-scene/session-history boundary.

Only episodes carrying the `__physx_v3` suffix and matching marker may be
treated as current fresh-scene candidates. Collection time or file
modification time must not be used to infer the generation.

The new marker establishes the data generation; it does not by itself certify
the episode. New episodes remain
`new_requires_physical_certification` until the independent task-success and
physical-replay checks pass. New candidates are staged beneath
`.physical_replay_pending`; only candidates with three valid independent proofs
are atomically promoted to the published directory.

## Data policy

- Preserve legacy recordings only as quarantined source material; they are
  invalid for the current training generation.
- Do not mix legacy and newly certified samples without a version label.
- Quarantine recordings with action/state misalignment or dropped frames.
- Use failed stable-origin experiments only for regression or explicitly
  designed negative-data training.
- Recollect successful demonstrations from the stable origin before declaring
  the new collection version training-ready.

The current evidence covers the tested software/hardware environment. Exact
replay across different GPUs, drivers, Isaac/PhysX versions, or machines must
be validated separately.
