# GEN6 source-derived asset bundle

[Audio Mixer by Rayan](https://sketchfab.com/3d-models/audio-mixer-40cccd93ff904e2f9d5f2e02bbbbe92b),
CC BY 4.0. See `ATTRIBUTION.md` and `SOURCE_LICENSE.txt` for credit and license.
Source files and conversion evidence are retained in
`taskdev/tasks/eval_gen05_set_mixer_knobs/`.

All layers use meters, Z-up and default prim `/Asset`:

- `mixer_fixed_visual.usda`: source enclosure and retained fixed controls.
- `knob_visual.usda`: original Cylinder098 in its normalized joint frame.
- `mixer_fixed_collision.usda`: 178 invisible source-derived convex pieces,
  including 57 enclosure pieces that preserve the fader slots/display recess.
- `knob_collision.usda`: one invisible 130-vertex hull, instanced per moving link.
- `textures/`: unchanged material-complete glTF textures.
- `asset_manifest.json`: source identity, archives and dependency hashes,
  scaling/axes, geometry, masses/inertias and source pocket partitions.

The runtime creates two passive revolute joints and uses measured frames from
the manifest. There is no training-knob fallback or new visible proxy model.

CPU source/dependency checks, sampled complete-robot PhysX clearance, nominal
physical operation, engineering recording/replay and three-camera rendering have
passed. Human teleoperation and ordinary hardware collection acceptance remain
pending. See the task directory's `HANDOFF.md` for evidence and measured limits.

To rebuild, use the scoped CPU environment with the task directory's pinned
`asset_build_requirements.txt` and run its `build_assets.py`, `audit_assets.py`
and `render_cpu_preview.py`. This build environment is not the Isaac runtime;
collection remains Isaac Sim 5.1 / IsaacLab 2.3.2.
