# Paper evaluator restoration and validation

The public evaluator restores the task implementations, launch settings and
assets used by the archived Aliyun RTX 5090 paper deployment. The compact
[per-task validation record](trajectory-alignment.json) covers Core-50 and
Transfer-16. A successful task startup is not evidence of full trajectory or
policy-score equivalence.

Historical demonstration replay needs additional scene and controller state.
It is tracked separately in [expert replay](expert-replay.md). The released
LeRobot state/action arrays are not complete simulator snapshots.

## Archived Core trial inputs

For Core-50, `--paper-trial` selects a retained input for the requested model:
accepted reset candidate, goal, instruction, task setting and source variant.
Indices are 0–99. Inspect an input without starting the simulator:

```bash
python -m handson.evaluation --model openwam --task t24_place_ball_in_bowl \
    --paper-trial 0 --inspect-paper-trial
```

Use the same `--paper-trial` selection with your normal policy connection to
run that input. Seed-only mode remains available. The public single-episode
entry does not reproduce the campaign's queue, recovery or seed-replacement
scheduler. It does not silently replace an archived rejected candidate.

The input manifests contain no success outcomes. All 45,000 retained Core
trained-policy job/result identities were checked against their recorded input,
loaded-module and asset hashes. This excludes GPT and does not rerun scores.
Transfer source checks cover 6,234 non-DP scoring-snapshot candidates; the
historical table builder used 6,232 declared-limit traces. The two excluded
candidates were not individually identified by this audit.

## Restored runtime contracts

- Historical task factories, native reset/settling lifecycle, EEF52 control,
  goal conditioning and native success checks are selected explicitly.
- The action clock is 30 Hz in physical time. Scrub-bottle retains its native
  240 Hz integration with eight substeps; the other checked routes use
  120 Hz with four substeps. Demonstration-derived duration budgets and the
  three Transfer deformable-task exceptions are retained.
- Recorded task names remain the input to historical RNG derivation. Ten early
  Core tasks retain the original visual helper's reset-RNG draw sequence.
- Core observation cameras use two recorded profiles: 43,193 trials have
  independent wrist calibration; 1,807 have the earlier shared calibration
  (900 fill-cup, 900 drawer and seven legacy device-cover trials). Hammer's
  reset visibility projection retains its own original camera geometry.
- The task surface loader installs the 46 recorded Core visual layers before
  physics initialization. Four Core tasks and all Transfer tasks retain native
  task-object surfaces. Geometry, UVs, material subsets and texture dependencies
  are checked by hash. Portable USD encoding and relative asset paths preserve
  the original authored values.
- The worker clears external `HANDSON_` variables before applying recorded
  control settings, preventing shell overrides from changing the paper route.

Some tasks have genuine historical variants. The seven legacy device-cover
trials use `stabilize_body`; the dominant task implementation remains the
default. Largest-fitting-ball retains the recorded hand-only/full-arm reset
selection. `--paper-trial` resolves these from the archived input. Variants
are never selected using observed success.

The documented campaign decisions remain explicit: latched-door's source/text
waiver and carton's force-observation-only scoring are not presented as repaired
source agreement or low-force certification. See the
[policy contract](custom_policy_integration.md) for those exceptions.

## Original/public controls

The r54 baseline was tested on RTX 5090 with Isaac Sim 5.1.0.0 and Isaac Lab
v2.3.2. Each control used 90 identical original-origin hold commands per side,
without restoring a historical scene during evaluation.

| Scope | Observed agreement | Remaining distinction |
|---|---|---|
| 57 Core source cohorts across 50 tasks | Exposed physical profiles, robot/body and non-contact action states, actual camera poses/intrinsics and appearance draws match | Six raw friction-anchor arrays differ in row order; active multisets match at every action |
| 16 Transfer tasks | Physical profiles, appearance parameters and measured cameras match; 13 action-array sets match exactly | Comb03 has near-zero floating-point differences; Flex01/Flex03 have native soft-body/settling variability |

All 342 Core videos (1,368 frames) and 1,539 PNGs decoded. All 15 Core main-view
initial/final comparison sheets were inspected. Transfer media and comparison
sheets were checked separately. These checks establish neither pixel identity
nor success-rate equivalence for trained policies.

Microwave contact counts were additionally checked with both recorder
algorithms on each live environment. The old friction getter overwrites the
normal-contact count/start buffers, recording six contacts where the corrected
reader preserves twelve. Corrected readers on original/public environments
produce equal active contact and friction multisets before and after 90 actions;
all compared action states also match. The lossless recorder is retained. This
is a recording difference, not a modification of physics or native scoring;
it is not an all-task contact-cache certification.

## Packaged entry checks

The r54 wheel's 8,231 entries and 967 task-surface resources were checked against
installed bytes. Five CLI checks and 32 installed tests with 155 subtests passed.
Four real installed-wheel GPU entries exited normally:

| Entry | Result |
|---|---|
| `e015401`, public EEF52 | 209 actions; first native success at 209 |
| `e013673`, recorded joint targets | 720 actions; first native success at 479; 2,880 recorded targets applied exactly |
| `t46_store_charger`, pi05 paper input 0 | 90 diagnostic hold actions; initial cameras and appearance match the reference |
| Transfer Flex01, native input | 90 diagnostic hold actions; native task-object surfaces retained |

The legacy joint-target example is not an EEF52-only reconstruction. Six videos
and 492 PNGs decoded, and all four entries' three-view initial/final images were
inspected. These diagnostics are not formal benchmark trials.

## Validation after shared-runtime cleanup

The shared runtime in [runtime maintenance](runtime-maintenance.md) was checked
again against the retained original controls on RTX 5090. All 57 Core source
cohorts and all 16 Transfer tasks completed the same 90-action controls. Physical
profiles, measured camera poses/intrinsics and appearance draws match. Core
non-contact arrays match, and the six friction-anchor ordering differences have
equal active multisets at every action.

For Transfer, 13 action-array sets match exactly. Comb03 differs only near zero
(maximum effort difference 6.25e-14). Flex01's maximum soft-body position
variation is 0.265 mm, with a 0.2084-second settling-origin offset; Flex03's
maximum position variation is 0.115 mm. These are measured differences, not
claims of exact trajectory identity. All 342 Core and 96 Transfer videos decoded;
all 1,539 Core and 384 Transfer PNGs decoded. All 19 main-view initial/final
comparison sheets were visually inspected. Native appearance was preserved,
including the magenta mousepad visible in both Comb02 controls; matching the
historical rendering is not a general cosmetic-quality certification.

The source controls used revision r57. Final revision r58 changes only trailing
blank lines in three runtime modules, with matching ASTs, plus integrity
metadata and test fixtures. The r58 Linux CPU suite has 2,100 passes and 189
subtests: 2,099 passed in the complete run, then the sole Git-environment failure
passed after supplying a compatible isolated Git binary.

The final r58 wheel contains 8,233 entries, all verified against installed bytes.
Its five CLI checks and 49 installed tests with 155 subtests passed. All four
installed-wheel GPU entries in the table above were rerun using r58, with the
same reported action counts, first-success steps and exact recorded-target
application. The archived Core input also retains exact measured initial
cameras and appearance draws. Six videos and 492 PNGs decoded, and all four
entries' three-view initial/final images were inspected.

The per-task JSON records the fresh controls separately from the retained r54
baseline. This cleanup does not certify trained-policy success-rate equivalence,
all historical demonstration trajectories, or pixel identity.

## Scope and evidence

The source mapping, input/asset manifests and tests ship with the repository.
The per-task JSON retains result hashes, disagreement categories and verification
scope. Full operational logs, intermediate failures and raw diagnostics are
retained in the authors' audit archive; local cluster paths are not runtime
dependencies or public reproduction instructions.

No paper results or released training labels were changed. Untested seeds,
policy-dependent contacts, exact pixels and historical demonstrations remain
outside these bounded checks. Some collection cohorts lack uncommitted source
changes or complete controller/solver history. Neither a seed nor an EEF52
robot pose alone restores that missing state.
