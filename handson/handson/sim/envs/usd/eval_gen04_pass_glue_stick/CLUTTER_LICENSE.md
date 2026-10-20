# GEN4 source-derived clutter collision attribution

The ten D03 distractor bodies reuse nine asset identities from Lightwheel-YCB
by LightwheelAI: flat_screwdriver, phillips_screwdriver, spoon, plastic_bolt,
a_marbles, b_marbles (two instances), hammer, a_Nails and bowl.

Source: https://github.com/LightwheelAI/Lightwheel-YCB
License: Creative Commons Attribution-NonCommercial 4.0 International.
https://creativecommons.org/licenses/by-nc/4.0/

Lightwheel-YCB builds on the YCB Benchmarks Object and Model Set.
The existing visual meshes and textures remain in the repository's
`sim/envs/xmls/lightwheel_ycb/` directories. `clutter_collisions.usd` is a
derivative of their existing collision meshes, except the marble package hull
which follows the original visual package exterior. Some objects receive a
shared visual/collision support-frame rotation; small source convex hulls are
partitioned to meet PhysX GPU cooking limits without changing their exterior.
Exact inputs, hashes and modifications are recorded in `asset_geometry.json`.

These non-commercial terms differ from the CC BY 4.0 glue and tray licenses.
