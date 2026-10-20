# GEN5 source assets

Source: [Cardboard Boxes](https://sketchfab.com/3d-models/cardboard-boxes-b0e935d3008b4c2f866e1a8fffe2bc1f), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The original downloaded license credits **Isaac Rivera (https://sketchfab.com/silverpirate)**. The current model page credits **JesterArts_ (@JesterArtist_)**. Both credits are retained; no reason for that difference is inferred. The unmodified archive and its original license are in `source/`. SHA256: `422418b7bd9396b28a63723d73f4731bf9d87dc7dc6c87c13b7077c1893541a1`.

This work is based on “Cardboard Boxes” by Isaac Rivera, licensed under CC BY 4.0; the current source page displays JesterArts_. Modifications: extract the existing `Box_big__CardboardBoxes__0` closed component, center its original Z-up coordinates, uniformly scale and export to OBJ. Source faces, UV coordinates and normal data are preserved. No box faces, texture artwork or other visible geometry were created or repaired.

| Role | Dimensions (mm) | Mass (g) | Uniform scale (m/source unit) |
|---|---|---:|---:|
| Base | 90 × 90 × 76.6875 | 180 | 0.001875 |
| Each of three small boxes | 36 × 36 × 30.675 | 30 | 0.00075 |

The source contains two closed convex boxes and one open box. GEN5 instances the near-square closed box at two scales; the elongated and open components are excluded to keep grasping and four-layer support close to B04. The small boxes remain three independently named dynamic bodies. Total nominal stack height is 168.713 mm.

Invisible collision meshes use the same closed source surface, welding only duplicated vertices at UV/normal seams. Each has 44 triangles. The original bevel is 0.375 mm on the base and 0.150 mm on small boxes. The large top support face is 89.25 × 89.25 mm; small support faces are 35.7 × 35.7 mm. Geometry is neither inflated nor replaced by a visible primitive.

`geometry.json` records dimensions, source transformation, exact inertia tensor and physical parameters. Inertia is integrated over the closed mesh with uniform effective density normalized to the stated mass. The task keeps B04's friction, damping, contact offsets and success semantics. The original albedo PNG is referenced directly with the source color factor. The original normal and metallic/roughness maps are retained in the archive; the MJCF material currently uses albedo.

Reproduction commands, from the dedicated GEN5 checkout:

```sh
PYTHONPATH=handson:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python taskdev/tasks/eval_gen07_stack_boxes/integrate_assets.py
PYTHONPATH=handson:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python taskdev/tasks/eval_gen07_stack_boxes/audit_cpu_assets.py
```

CPU evidence is in `taskdev/tasks/eval_gen07_stack_boxes/evidence/cpu_asset_audit.json`: source equality, watertight/convex surfaces, UV data, inertia, ten orientations, compiled scene, twenty seeds and four boundary layouts, mesh-distance preflight and all six upper orders. It does **not** establish live PhysX collision clearance, stability, actual collection, camera/replay correctness or operator comfort; those gates require the approved subsequent checks.

The CPU mesh-distance audit retains ten ambiguous raw zero results together with positive actual-vertex separating-axis witnesses. These are CPU preflight evidence only, not PhysX narrow-phase results.
