# Replaying a released expert episode

An EEF52 state contains measured palm poses and finger joints. It does not
contain object poses, velocities, full robot/controller state, or task history.
Likewise, a seed identifies a random draw only within the corresponding task
implementation. A current `env.reset(seed=...)` is not a historical scene snapshot.

## Included example: e015401

The repository now includes an initial-state bundle and the 209 public EEF52
commands for `e015401` (`t34_roll_ball_through_gate`). The command below restores
the recorded initial state once, then executes the public commands using native
physics. It does not write recorded scene states during action execution.

Use the [PhysX installation](installation.md): Python 3.11, Isaac Sim 5.1.0.0,
Isaac Lab v2.3.2, and an NVIDIA GPU. No separate dataset download is needed for
this bundled example. First check the bundle and implementation identities on CPU:

```bash
python -m handson.replay --episode e015401 --check
```

Then run in the simulator environment, using a new output directory:

```bash
python -m handson.replay --episode e015401 --output outputs/e015401
```

Progress is written to `runtime.log` and `episode/PROGRESS.json`. Outputs include
three-view PNG captures, `INITIAL_STATE.json`, `T0_RESTORE.json`, per-action
`trajectory.jsonl`, `physical_readback.npz`, `FINAL_STATE.json`, and `RESULT.json`
under `episode/`.
The parent command requires a completed result receipt; a zero Kit exit code
alone is not treated as a completed replay. Default render stride is 10 actions
and the worker wall-clock limit is 1,200 seconds. These affect capture/runtime
management, not the 120 Hz physics / 30 Hz action clock.

## What the example establishes

The original episode was collected with source commit
`e199dbf4a82fbc6af97a976925f8ad6c5a13da42`. The archived collection code and
all 900 trained-model paper evaluation receipts for t34 use the same task
semantics. The initial public release had drifted to a different ball/gate
reset distribution and a weaker success predicate. This revision restores
the historical t34 sampling and event predicates, including right-hand contact,
passage through the opening, table support, and avoidance of gate-box contact.
The full-ball far-side threshold is again 0.072967 m.

On an RTX 5090 on 2026-11-01, all 209 public commands executed exactly and native
success first occurred at action 209, matching the original recording. With
one full initial-state restore, the initial ball position error was zero; ball
trajectory error was 1.39 mm RMS and 13.09 mm maximum. Initial EEF52 readback
error was below 1.5e-7. Physics ran at 120 Hz with four substeps per 30 Hz action.
The seeded reset's ball position also matched the recorded initial position
within 1e-7 m before the full-state restore. This last observation concerns the
ball only, not the entire robot, controller, or settled simulator state.

These measurements apply to this episode and implementation. The original
recording does not contain a bitwise PhysX checkpoint, including contact/solver
caches. Task-predicate restoration and a successful replay do not certify every
historical dependency, other episodes, or pixel-identical rendering. Results
therefore retain `historical_checker_certified: false` and
`formal_benchmark_trial: false`.

The manifest binds the initialization bundle, source identifiers, and qualified
runtime file checksums. Changing those files requires revalidation. Earlier
measurements (action-208 success and the 0.0299465 m threshold) describe the
superseded public implementation and do not describe this repaired revision.

For a diagnostic that finishes all recorded commands after native success:

```bash
python -m handson.replay --episode e015401 \
    --complete-recording --output outputs/e015401-all-actions
```

This option still stops on failure, reset requests, and timeout. It records the
first native success separately from the subsequent diagnostic actions. The
historical far-side geometric check is reported separately; that check alone
is not the complete historical success test.

The bundle also carries the published resolved room appearance (material,
texture placement, lighting and context) and camera jitter. Replay applies that
profile directly instead of re-sampling it under a renamed task or renderer.
An appearance receipt verifies that the physical scene and contact materials
are unchanged. Pixel-identical rendering remains uncertified.

## Included legacy device-cover example: e013673

This 720-action episode uses the historical left-hand-body/right-hand-cover
variant of `t06_fit_device_cover`, source commit
`59d128d30062a25e3283da344e4de6fe777de2d5`. The bundle selects that variant
explicitly and includes its initial physical/task state, published appearance,
camera jitter, public EEF52 commands, and recovered original motor targets.
No separate dataset download is required.

To replay the unchanged public EEF52 commands:

```bash
python -m handson.replay --episode e013673 --check
python -m handson.replay --episode e013673 --output outputs/e013673-eef52
```

**This mode completed all 720 commands but did not reach native success.**
The same commands also failed in the unmodified original source, with
bitwise-equal joint, palm and object readback. Restoring the historical scene
and code therefore does not make this exported action sequence reproduce the
original collection execution.

For a separate diagnostic, explicitly select the recovered motor targets:

```bash
python -m handson.replay --episode e013673 \
    --control recorded_joint_targets --check
python -m handson.replay --episode e013673 \
    --control recorded_joint_targets --complete-recording \
    --output outputs/e013673-recorded-targets
```

On RTX 5090, this executed 720 actions / 2,880 physics substeps and first
reached native success at action 479. Every applied motor target matched its
recorded value; full joint, palm and object arrays matched the corresponding
original-source control. Both modes restore scene state only once, before the
first action. The second mode writes motor targets, never historical object or
robot poses during execution.

The motor targets are additional original execution records, absent from the
released EEF52 columns. This mode is not a policy evaluation and does not show
that EEF52-only replay succeeded. It also retains historical position errors:
3.97 mm palm RMS and 13.30 mm object RMS. The recording's success label is set
only on the final frame and does not establish a matching first-success time.
`RESULT.json` records the chosen control mode, substep checks and measured
errors. Exact historical-state reproduction remains uncertified.

## Other episodes and evaluation

Only `e015401` and `e013673` currently have packaged, simulator-tested entries.
Other episodes require their own initial object/robot/controller/task state and
their recorded layout. Do not reuse this bundle for another episode or infer
that all 18,000 demonstrations already support this command. The additional
initialization data are bundled here, not retroactively present in the original
LeRobot export on Hugging Face.

The older `scripts/platform_independent_episode_replay.py` and certification
scripts expect original recording bundles and matching source versions; they
are not drop-in players for public Parquet files.

Use `python -m handson.evaluation` for policy evaluation with the current task
protocol. Replay does not override the evaluator's
reset, scoring, or denominator. The t34 source repair applies to both entry points. Exact historical certification additionally
requires the original version's complete task/controller implementation and
success checker; the included example is explicitly a current-runtime replay
from the historical initial state.

## Supplemental command activation

Some recordings begin with hand commands before native arm-pose control has
been activated. The native controller retains an inactive stored pose target;
EEF52 alone cannot express that inactive state. Do not infer the missing flag
from a small action value, replace it with a measured pose, or change the policy
evaluator to guess it.

For an independently verified recording bundle, the lower-level
`handson.replay.control.RecordedPoseControl` helper accepts the original
`next_physx_pose_commanded` flags alongside each unchanged public action. Its
initial flags must match the controller state already restored at t0:

```python
from handson.replay.control import RecordedPoseControl

control = RecordedPoseControl(env, initial_snapshot["physx_pose_commanded"])
for action, flags in zip(public_actions, source_pose_commanded, strict=True):
    observation, reward, terminated, truncated, info = control.step(action, flags)
    if terminated or truncated or info.get("failed") or info.get("manual_reset"):
        break
```

This fragment assumes the matched native environment, task settings, complete
initialization and source flags have already been recovered and verified. Those
inputs are not all present in the original LeRobot export. The three diagnostic
bundles using this helper are not yet supported by the packaged CLI. See
[the alignment audit](trajectory-alignment.md) for measured outcomes and the
separate controller contract used by 56 policy-knot recordings.
