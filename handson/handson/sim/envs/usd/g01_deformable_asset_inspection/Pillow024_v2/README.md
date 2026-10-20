# Pillow024 v2 normalized asset

This directory is a deterministic derivative of the reviewed supplier file
`/home/user/下载/Pillow024/Pillow024.usd` with SHA-256
`8b08e24ba0e8bc359f8802fce393fddb6234b31ca60a8e51a85404638cc80305`.
The download source is never modified or copied into this repository.
It did not include independent license metadata.  The source inherits the
repository's existing supplier-archive provenance record for
`ZRYZ_LW_Assets_20260807.zip` (SHA-256
`90798bdad84d8b1de38eb625984f7fefa4ae3b56c7297fb4769bd35151189306`),
but that provenance record is not a new license grant or authorization.

Build with the single collection runtime:

```bash
"$HANDSON_ISAACSIM_PYTHON" scripts/prepare_g01_2_pillow_asset.py
```

The builder requires Isaac Sim 5.1.0.0 and IsaacLab v2.3.2, builds twice,
requires byte-identical USD output, and validates the committed result again.
It performs only these asset-local normalizations:

- restores visual `points` from the deformable bind points and recomputes the
  visual extent;
- preserves and re-authors the supplier face-varying normals byte-for-byte,
  while checking them against a smooth bind-surface reference;
- restores simulation `points` from `restShapePoints`, zeroes velocities, and
  recomputes the simulation extent;
- removes the visual child `PhysicsMassAPI` / `physics:mass`, retaining the
  deformable root mass of 0.8 kg;
- removes the visual child physics-material binding while retaining the
  physics-material bindings on the deformable root and simulation TetMesh;
- explicitly hides the simulation TetMesh;
- removes `/Render` editor prims and the unused `primvars:______` data;
- reuses the byte-identical textures in `../Pillow024/texture/` rather than
  copying them.

The supplier TetMesh has 39 boundary-surface edges with incidence four.  This
known non-manifold risk is detected and recorded, but the builder deliberately
does not remesh or change the tetrahedral topology.  The preserved normals are
validated against the initial bind surface; live deformable rendering must
still be checked in Isaac Sim 5.1 to prove runtime normal updates after
deformation.

Companion repository changes wire this asset into the G01-2 scene,
environment, and task registration.  This builder itself remains asset-local
and does not edit those integration files.  No formal physical-replay claim is
made.
