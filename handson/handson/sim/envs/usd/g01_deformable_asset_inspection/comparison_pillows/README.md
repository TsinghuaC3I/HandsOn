# G01 original sewn pillows — revision sewn-pillow-v2

G01-4 and G01-5 both use the same bimanual carton-placement task. The beige
soft pillow and blue resilient foam pillow are original editable Blender
meshes, with separate live Isaac Sim 5.1 / IsaacLab 2.3.2 FEM assets. Neither
uses baked Blender motion, a hidden rigid core, nor interaction-time node clamps.

| Profile | Mass | Young's modulus | Poisson ratio | Elasticity damping |
| --- | --- | --- | --- | --- |
| G01-4 soft filling | 0.55 kg | 30 kPa | 0.45 | 0.25 |
| G01-5 resilient latex-foam approximation | 2.8081 kg (55 kg/m³) | 400 kPa | 0.33 | 0.18 |

Both are equivalent homogeneous continua. Qualitative video references informed
the rounded crown, thin sewn perimeter, gathered edges and contrasting support.
They do not establish measured force/displacement curves, feather migration,
viscoelastic memory or temperature response. See VIDEO_REFERENCES.md.
G01-5 additionally uses published latex/foam references and measured native-hand
tests; see [LATEX_REFERENCE.md](LATEX_REFERENCE.md). Its modulus is an equivalent
simulator calibration, not a measured commercial product specification.

Three meshes serve different purposes:

| Profile | Solver nodes / tets | Contact nodes / tets | Contact triangles | Render triangles |
| --- | --- | --- | --- | --- |
| soft | 721 / 2,832 | 16,489 / 77,125 | 15,112 | 69,664 |
| foam | 753 / 3,000 | 19,110 / 89,409 | 17,484 | 79,152 |

The regular PhysX TetMaker voxel mesh provides uniform solver cells. The fine
TetGen mesh follows the original sewn envelope and supplies collision. The
main cloth surface exactly subdivides that boundary; the separate visual
piping has a 0.8 mm radius and follows the same deformable body. Cotton weave
maps are original procedural textures. G01-4 remains 80 × 42.22 × 18.65 cm;
G01-5 is rebuilt at 80 × 42.22 × 22.0 cm so it reads as a pillow and gives
the hands a fuller side wall, while keeping the same reachable footprint and
carton.

G01-4 requests 128 position iterations. G01-5 requests 255 and its scene cap
honors that request; the previous 128-iteration solve retained excessive
compliance under native hand loading. This costs additional simulation time.
G01-5 also allows a bounded 100 mm measured-palm target offset for the 2.81 kg
loaded grasp; shared joint slew, jump and workspace guards remain active.
G01-3/G01-4 retain 35 mm. Both use speculative CCD, self collision,
6 mm contact offset / 1 mm rest offset, and dynamic friction 0.65 before a
grasp. For G01-5, a local inward deformation of at least 6 mm switches
the foam material to static/dynamic friction 1.80/1.45; a grazing touch stays
at the baseline material, and unloading restores the baseline. PhysX 107.3
ignores deformable static friction; the dynamic value is verified by live
material readback. The native
42 palm/finger rigid bodies and 52 hand colliders receive the CCD flag and
4 mm / 0.5 mm contact/rest offsets. NVIDIA documents that rigid-body CCD is
unsupported in Direct GPU API mode, so that flag is not an effective guarantee
of hand collision protection. Live contact measurements are the acceptance
evidence. Their source geometry is retained. Contact
offsets are solver margins, not thicker replacement objects. Support and
containment predicates read the live collision surface, not the coarser solver
boundary. Inverted FEM cells cannot count as a stable ready state.

Physics advances at 120 Hz with 30 Hz control targets and shared deferred
main/left-wrist/right-wrist cameras. These are simulated rates, not a claim of
achieved wall-clock throughput. Ordered physical contact, lift, containment,
release and stability success semantics remain enabled. Formal deformable
physical replay is still explicitly uncertified by the runtime.

Rebuild from the repository root (MeshPy is needed only for the first step):

```sh
python3 scripts/prepare_g01_comparison_pillows.py --geometry-only --output-dir work/pillow-geometry
/path/to/isaacsim-smooth-python scripts/voxelize_g01_comparison_pillows.py --geometry-dir work/pillow-geometry --resolution 16
blender --background --python scripts/prepare_g01_comparison_pillows.py -- --geometry-dir work/pillow-geometry --output-dir handson/handson/sim/envs/usd/g01_deformable_asset_inspection/comparison_pillows
```

The build and runtime check positive volumes, boundary topology and
visual/contact correspondence. `asset_manifest.json` records material and mesh
parameters and the exact USD SHA-256; task metadata records that same asset
revision and hash. Validation results belong to the matching USD hashes.

Physics references:
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_beta.html
- https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html

Rigid-body GPU limitation: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-continuous-collision-detection-on-a-rigid-body

Validation scope is recorded in `validation_summary.json`. Individual and
simultaneous native hand press/release and 40 mm sphere indentation are tested.
The separate 60 mm concentrated-load challenge can invert internal FEM cells
even when the collision surface does not penetrate the indenter. It is not a
guarantee for arbitrary deep squeezes. A completed boxing episode and formal
fresh-process physical replay are not yet verified.
