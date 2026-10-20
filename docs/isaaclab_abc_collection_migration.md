# IsaacLab A/B/C Collection Migration

This checklist records the A/B/C tasks migrated to the shared native IsaacLab
collection environment. Every task listed below is enabled in
`ISAACSIM_COLLECTION_TASKS` and uses the formal Gym/recorder path rather than
the generic preview-only importer.

## 2026-08-04 stability checkpoint

Today's delivery scope is A/B/C plus the shared physical-replay gate published
to `release/collection`.

- Reset layouts and support geometry were tightened for the previously
  unstable A/B/C cases: A01, A06, B01, B04, B05, C01, C02, C04, C07, C08,
  and C12.
- The reset boundary journals exact GPU setter inputs before the first PhysX
  step, then validates every subsequent readback without a CPU fallback or a
  per-frame velocity clamp.
- A/B/C initialization uses `cuda:N` only: one second of initial settle plus
  at most 39 seconds for the strict gate. The gate samples every physics frame
  and exits as soon as 0.5 continuous seconds are stable. Native PhysX steps
  run with `render=false`, no real-time sleep, and the configured `dt`
  unchanged; 40 seconds is a simulated-time ceiling, not a 40x wall-clock or
  enlarged-timestep mode.
- Post-settle and action-0 boundary writes are velocity-only. They align the
  robot drive target but never rewrite task poses or DOF positions, and A/B/C
  bodies remain awake to preserve the contact manifold and PhysX warm start.
- Per-frame clamp histories remain readable as diagnostics but cannot pass v4
  publication certification. The certifiable contract is exactly
  `stabilization_velocity_clamp=false`,
  `initialization_per_physics_step_velocity_writes=false`, and scope `none`.
- Publication is fail-closed on active-entity coverage, full-coordinate drift,
  finite values, canonical path-resolved contact signatures, replay errors,
  runtime/schema/Git identity, and scene-generation history.
- D01-D12 code and live regression work are deliberately excluded from this
  checkpoint. The local D02 GPU result (five measured refreshes, five
  first-attempt acceptances) is diagnostic only and will be revalidated with
  the final common gate before the next D-series checkpoint is committed.

### Measured GPU checkpoint

The final A/B/C fixes below were exercised on `cuda:0` with an RTX 4090,
driver `580.173.02`, Isaac Sim `5.1.0.0`, IsaacLab `0.54.2`, and a fixed
physics step of `1/120 s`. Each listed task accepted its final bootstrap on
the first physical attempt. This is a targeted regression checkpoint, not a
statistical claim about untested runtimes or seeds.

| Task | Accepted origins | Stable frames | Peak sampled penetration |
| --- | ---: | ---: | ---: |
| A01 | `1/1` | `60` | `0.123 mm` |
| A06 | `1/1` | `60` | `<0.001 mm` |
| B01 | `1/1` | `60` | `<0.001 mm` |
| B04 | `1/1` | `60` | `0.001 mm` |
| B05 | `1/1` | `60` | `0.002 mm` |
| C01 | `1/1` | `60` | `0.001 mm` |
| C02 | `1/1` | `60` | `<0.001 mm` |
| C04 | `1/1` | `60` | `<0.001 mm` |
| C07 | `1/1` | `60` | `0.002 mm` |
| C08 | `3/3` (bootstrap + two refreshes) | `60` each | `0.463 mm` |
| C12 | `1/1` | `60` | `0.001 mm` |

For C08, both measured refreshes were accepted on their first attempt, so the
observed refresh instability rate was `0/2` and did not exceed 10%. Its final
support geometry uses load-bearing box contacts for the tongs, the measured
unclamped cradle equilibrium as the authored pose, and a flat cup base with
tangent reset placement. The action safety check uses the exact oriented
bounding boxes and swept corners of those colliders. The three accepted C08
origins stayed below `1e-6 m` consecutive translation, `1e-5 rad` consecutive
rotation, and `0.5 mm` penetration without pose writes, sleeping, or per-frame
velocity clamps. The machine-readable audit is
`~/.cache/fix_origin_live/abc_c08_seed0_v132_flat_cup_continuous3.json`.

## Status Terms

- `collection-ready`: the recorder has been smoke-tested and the task can be
  added to the collection allowlist.
- `shared ABC env`: task-specific scene, reset, observations, success and
  camera behavior are implemented by `HandsOnABCIsaacLabEnv`, while robot
  control is inherited from `BimanualAssemblyIsaacLabEnv`.

## A Series

| Task | Source scene | Reuse group | Status |
| --- | --- | --- | --- |
| `t26_place_fruit_in_bucket` | `arena_arm_hand_ycb_tasks_rokae_wuji1.xml` | YCB table object manipulation | collection-ready, shared ABC env |
| `t25_place_cup_on_plate` | `arena_arm_hand_25_place_cup_on_plate_rokae_wuji1.xml` | tray placement | collection-ready, shared ABC env |
| `t24_place_ball_in_bowl` | `arena_arm_hand_ball_to_bowl_rokae_wuji1.xml` | bowl placement | collection-ready, shared ABC env |
| `t19_orient_cup_upright` | `arena_arm_hand_ycb_tasks_rokae_wuji1.xml` | cup upright-orientation task | collection-ready, shared ABC env |

## B Series

| Task | Source scene | Reuse group | Status |
| --- | --- | --- | --- |
| `t04_cover_pitcher` | `arena_arm_hand_ycb_tasks_rokae_wuji1.xml` | YCB table object manipulation | collection-ready, shared ABC env |
| `b02_nut_bolt_assembly` | `arena_arm_hand_b02_nut_bolt_assembly_rokae_wuji1.xml` | threaded assembly | collection-ready, shared ABC env |
| `t09_insert_peg` | `arena_arm_hand_09_insert_peg_rokae_wuji1.xml` | insertion assembly | collection-ready, shared ABC env |
| `t44_stack_blocks` | `arena_arm_hand_ycb_tasks_rokae_wuji1.xml` | YCB table object manipulation | collection-ready, shared ABC env |
| `t03_cover_box` | `arena_arm_hand_ycb_tasks_rokae_wuji1.xml` | YCB table object manipulation | collection-ready, shared ABC env |

## C Series

| Task | Source scene | Reuse group | Status |
| --- | --- | --- | --- |
| `t08_hammer_nail` | `arena_arm_hand_hammer_nail_rokae_wuji1.xml` | tool use | collection-ready, shared ABC env |
| `t36_scoop_marbles` | YCB spoon/marbles scene | scooping | collection-ready, shared ABC env |
| `t50_water_plant` | `arena_arm_hand_plant_rokae_wuji1.xml` | spray/water plant | collection-ready, shared ABC env |
| `t14_move_and_click_mouse` | `arena_arm_hand_monitor_mouse_rokae_wuji1.xml` | mouse interaction | collection-ready, shared ABC env |
| `t23_photograph_logo` | `arena_arm_hand_bimanual_photograph_rokae_wuji1.xml` | bimanual photo | collection-ready, shared ABC env |

## Per-Task Done Criteria

For future changes to any task, continue to follow
`$handson-isaaclab-task-migration`:

1. Create `<task>_scene.py` and `<task>_env.py` or extend a shared IsaacLab
   task-family env only when the observation/action/success contracts match.
2. Preserve the MuJoCo reset distributions, including table height and target
   object placement.
3. Use real `CameraCfg` RGB streams for main, `wrist_left`, and `wrist_right`.
4. Keep the default IK as IsaacLab DLS unless a task requires a custom backend.
5. Run syntax tests, UI tests, Isaac runtime reset/render smoke tests, and a
   short GUI recorder test.
6. Add the task to `ISAACSIM_COLLECTION_TASKS` only after the recorder test
   produces usable videos and no camera fallback warnings.

## Production Camera Profile

Native Isaac collection uses the packaged HandsOn calibration by default.
The head sensor renders 854×480 with the measured lens intrinsics, principal
point and native OpenCV pinhole distortion schema. Both wrist sensors render
848×480 with side-specific rectified intrinsics and calibrated palm-relative
poses. Do not apply fisheye rectification to wrist frames again.

Each camera has its own exposure and capture pass. Only visual obstructions
are hidden for wrist capture; contact geometry is unchanged. See
[Default rendering](rendering.md) for asset checks and appearance behavior.

C06's `photograph_target_region_visual` is removed only from the generated
Isaac scene, so the green helper block is not recorded. The source MuJoCo
scene and C06 success logic are unchanged.

C01 accepts a hammer strike only when its existing face alignment, nail travel,
and contact conditions pass and the hammer has at least `0.20 m/s` downward
speed. A slow touch cannot latch success.

Run a formal Isaac collection task from the repository root. The supervisor
uses a fresh simulator process per candidate and publishes only episodes that
pass three independent physical replay certifications:

```bash
python scripts/fresh_process_certified_collection.py \
  --exp_name=t08_hammer_nail \
  --sim_backend=isaacsim \
  --successes_needed=1 \
  --show_sim_cameras=True \
  --out_dir=./demos
```

Do not invoke `record_demos_zarr.py` directly for formal Isaac data; that path
is reserved for workers started by the supervisor. Disabling PhysX recording
does not bypass this rule and is allowed only for preview/debug runs.
