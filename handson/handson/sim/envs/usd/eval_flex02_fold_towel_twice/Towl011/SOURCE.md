# Towl011 source and debug delta

## Intake and provenance

- User-provided date: `2026-08-25`.
- User-provided directory: `/home/user/下载/Towl Assets/`.
- GitHub baseline repository: `https://github.com/llltttwww/handson.git`, remote
  branch `release/collection` (local checkout branch `collection/release`).
- GitHub baseline commit: `6f0c5169add5f5bacf81c4228d0a52d4525494ba`.
- Baseline asset path: `handson/handson/sim/envs/usd/g01_deformable_asset_inspection/Towl011/Towl011.usd`.
- Baseline Git blob OID (the LFS pointer):
  `f51c1972f04701d1ea56b66ab01e39f6fa02f078`.
- Baseline Git LFS object OID:
  `sha256:c56bec3db1aaec0e83ae73d61ec854dc1d9b77132738a9afdd2b85b7b151c16e`.
- The user supplied no independent license or provenance metadata with the
  debug package. This copy therefore preserves the existing G01 provenance
  chain only: source archive `ZRYZ_LW_Assets_20260807.zip`, SHA-256
  `90798bdad84d8b1de38eb625984f7fefa4ae3b56c7297fb4769bd35151189306`.
  That inheritance is a provenance record, not a new license grant.

No `.DS_Store` file was imported.

## Imported-file SHA-256

| File | SHA-256 |
| --- | --- |
| `Towl011.usd` | `e2e84e9f22ce21550bc2ddecd995de7dbe08004a1f6bb001ee6fd2162b26e327` |
| `texture/T_Towl011_BC001.png` | `fc43cd8f058c3413f75813b9a0d9d8ee114e9a30b046ac805648cfacf95921e0` |
| `texture/T_Towl011_N001.png` | `28b07d6caf7c8289b0d0d8e594bb232b95868499b58ec915e7784dae2326cea9` |
| `texture/T_Towl011_ORM001.png` | `2606180f241830b1ca600ad1c7391b3250e106639737f6f68960d55fd49b23a9` |

All three PNG files are byte-for-byte identical to the current G01 GitHub
baseline. Only `Towl011.usd` differs; the baseline USD SHA-256 is
`c56bec3db1aaec0e83ae73d61ec854dc1d9b77132738a9afdd2b85b7b151c16e`.

## Debugged USD differences from the GitHub baseline

- Deformable-material dynamic friction is newly authored as `3.0`.
- Collision `contactOffset` changes from `0.0005 m` to `0.0020 m`, and
  `restOffset` changes from `0.0003 m` to `0.0010 m`.
- `simulationHexahedralResolution` changes from `16` to `32`, and
  `solverPositionIterationCount=32` is newly authored. The regenerated volume
  data grows from 522 to 2,030 simulation points and from 1,103 to 4,677
  collision points; the simulation/collision index arrays and tetrahedral CRC
  change with that rebake.
- The source visual mesh translation changes from `(0, 0, 0)` to
  `(0.0002228022, -0.0001075715, -0.0457976162) m`. Its 3,048 visual points
  retain the same topology; 2,978 points differ only by float-rounding noise,
  with a maximum component delta of `2.98023e-08 m`.
- The debug file adds a nested `/root/PhysicsScene` authored at 200 physics
  steps per second with `solveArticulationContactLast=true`. It also carries
  invalid source-local gravity (`direction=(0,0,0)`, `magnitude=-inf`), so this
  scene must not control the task stage.
- Authoring-only viewport state changed: the saved Hydra texture resolution is
  3840x2160 instead of 2048x1080, and the saved Perspective camera moved.

## Runtime migration contract

The task runtime disables the nested source `PhysicsScene` and the hidden
supplier volume-deformable object. It does not run that volume alongside the
task cloth. Only validated material/contact/solver tuning from this debugged
asset is migrated onto the task-owned surface-deformable towel; the shared
Isaac Sim 5.1 stage remains authoritative for gravity and physics rate.
