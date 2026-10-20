# B320-KA005-2 asset provenance

This directory is the runtime visual closure for
`t38_set_dual_knobs`. It was built only from the two source
archives supplied on 2026-08-22; the previous dual-knob task and its assets were
not used.

## Rotary Knob

- Title: **Rotary Knob**
- Author: **0x.anas**
- Source: <https://sketchfab.com/3d-models/rotary-knob-d3dd68a3c5ae4b3dbe861f2bcffa376e>
- License: **CC BY 4.0**
- Original archive SHA-256:
  `9f65d6d81959812851dc12205d3412c0fbe3f4958e0fa8cb4c17bb4451e9b87b`
- Selected nodes: `Knob base_0`, `Grip_1`, `Top face_2`, and
  `Index line_3`. The other two knob styles and all three showcase labels are
  excluded.

The geometry is converted from glTF Y-up to USD Z-up, re-centered on its source
axis, uniformly scaled to a 60 mm moving-grip diameter, and rotated around Z so
the index line points along +X at q=0. The fixed bezel and moving grip are kept
in separate visual-only USD layers.

## Project Boxy

- Title: **Project Boxy**
- Author: **abusetech**
- Source: <https://github.com/abusetech/project-boxy>
- License: **MIT**
- Original archive SHA-256:
  `0726041e66f680b6b5dc61b0dcb5c76219bcf42f83b5825bd7913681635e6299`

The download contains parameterized OpenSCAD source rather than an exported
mesh. `build_assets.py` creates a simplified rounded body/lid visual derivative
from that design: a deterministic 210 x 490 x 50 mm enclosure with a 12 mm
corner radius, lid seam, and four screw heads.

## Physics boundary

These USD layers contain no rigid-body, mass, collision, joint, or articulation
schemas. The Isaac Sim 5.1 runtime derives one analytic 210 x 490 x 50 mm base
collider and two analytic 60 mm cylindrical grip colliders from the normalized
visual bounds, then authors two named +Z revolute joints. This keeps downloaded
mesh topology out of dynamic triangle-mesh collision.

Exact generated-file and dependency hashes are recorded in
`asset_manifest.json`. The upstream Rotary Knob credit/license record and full
Project Boxy MIT text are in `licenses/`.
