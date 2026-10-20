# Soap Dispenser — GEN1 asset

This notice describes the retained historical soap-dispenser source files.
The released EVAL-Gen03 scene uses the yellow YCB mustard bottle from
`lightwheel_ycb/mustard_bottle` for visual and collision meshes; `geometry.json`
records that substitution under `legacy_visual_source` and
`legacy_collision_source`. The internal `gen1_soap_bottle` body identifier is
retained for compatibility and does not describe the rendered object.

“Soap Dispenser” by [Anshul Star](https://sketchfab.com/anshulstar) is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

Source: https://sketchfab.com/3d-models/soap-dispenser-46d8d2bc470640ad95ce473304518cd6

Downloaded through the author's official Sketchfab download on 2026-09-07,
with the user's authorization. The official GLB file is retained at
`taskdev/tasks/eval_gen03_orient_bottle_upright/source/soap_dispenser.glb`.
SHA256: `97aa710e4dc0377da973a65155f4ca788ba092aa62b480f2a26696c5d15327dd`.
Its embedded metadata also identifies the author, source and license.

Changes: removed the photography background node; applied the source transforms,
rotated the bottle axis to +Z and its spout toward +X, centered the rigid frame at
half the source bottle height, and uniformly scaled the bottle body to 65 mm wide.
The source product meshes and material factors are retained. The pump components
are fixed to the same rigid body. No visible geometry has been modeled.

Invisible collision consists of source support-point hulls plus CoACD pieces for
the curved spout and pump cap. The enclosed pump-stem cavity is filled inside the
rigid assembly; its exposed surface is checked against the complete source model.
A small nonclosed source cap detail remains visible and is covered by adjoining
solid collision components. Collision is a task approximation, not a pump-fluid model.
See `geometry.json` and `taskdev/tasks/eval_gen03_orient_bottle_upright/evidence/source_geometry_audit.json`.

Mass is 100 g. COM and inertia come from uniform density over the collider union,
using 1 mm volume quadrature; no stabilizing COM bias is applied. These values are
simulation assumptions rather than measured product specifications. The build
scripts are `prepare_asset.py` and `decompose_pump.py` in the task development directory.

The authored side-rest pose has a -3.6 degree roll refinement around the bottle
axis, selected from native settling trials. This changes only initial placement;
the visible mesh, colliders, mass, COM, inertia and dynamic parameters are unchanged.
