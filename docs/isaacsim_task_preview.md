# Isaac Lab task previews

HandsOn can import all collection-UI tasks into a native Isaac Sim 5.1 / Isaac
Lab 2.3 PhysX scene. The preview migration covers the source arena, ROKAE arms,
Wuji hands, task assets, textures, materials, initial object layouts, driven
joint state, gravity, and contacts.

## Supported tasks

The generic launcher supports:

```text
t26_place_fruit_in_bucket          t03_cover_box
t25_place_cup_on_plate           water_plant
t24_place_ball_in_bowl             hammer_nail
t19_orient_cup_upright       fold_glasses
t04_cover_pitcher              pinch_tongs
b02_nut_bolt_assembly        click_mouse
t09_insert_peg            pick_bucket
t44_stack_blocks           bimanual_hanoi
bimanual_microwave_cook      bimanual_photograph
bimanual_unlock_ipad         d01_bimanual_nut_bolt_assembly
t12_load_and_start_microwave  t21_pass_block
e01_fold_garment             e02_tie_rope_knot
e03_untie_random_rope_knot   e04_cats_cradle_cycle
e05_fold_cloth_wipe_table
```

`bimanual_assembly` additionally uses the dedicated
`preview_isaaclab_bimanual_assembly.py` implementation.

E01--E05 are Isaac-only because they use native PhysX Surface Deformables.
E03 samples an initially verified rope knot, E04 samples one legal transition
from the recorded Cat's Cradle sequence, and E05 requires folding before
wiping only the C01-derived arm-reachable table mask.

## Interactive launch

Run a generic task from the repository root:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/miniconda3/envs/isaacsim/bin/python \
  scripts/preview_isaaclab_task.py \
  --task_name=click_mouse
```

The collection UI exposes the same launcher. Start the UI, select any task,
choose **Isaac Sim**, then click **Preview task**:

```bash
./scripts/run_collection_ui.sh
```

Open `http://127.0.0.1:8090/`.

## Headless validation

Use a finite run for a reproducible smoke test:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/miniconda3/envs/isaacsim/bin/python \
  scripts/preview_isaaclab_task.py \
  --task_name=water_plant \
  --headless --device=cpu --max_steps=100 --status_interval=25
```

A successful run reports `preview PASSED` and exits with status 0. E03--E05
were also validated through their complete native environments on CUDA,
including live Surface Deformable tensors, reset write/readback, robot control
steps, and all three RGB camera streams.

The launcher defaults to `cuda:0`; pass `--device=cpu` explicitly on systems
without a compatible CUDA/PyTorch build.

## Migration behavior

Generated MJCF and USD files live under `~/.cache/handson/isaaclab/`, or under
`HANDSON_ISAACLAB_CACHE` when set. The conversion layer:

- resolves and flattens nested MJCF includes for the Isaac 5.1 importer
- preserves source mesh and texture paths without modifying source assets
- isolates duplicate meshes and same-named textures in the generated cache
- repairs missing OBJ material sidecars and nested USD material bindings
- hides render copies of collision-only meshes
- rebuilds fixed roots whose imported anchors do not preserve body poses
- restores MuJoCo task layouts and the collection UI's tabletop robot reset
- adds valid inertias to fixed frame bodies required by PhysX

## Current boundary

The generic preview validates scene construction and stepping. Tasks listed in
`handson.runtime.collection_support.ISAACSIM_COLLECTION_TASKS` additionally
provide the native HandsOn action/observation contract, task success metrics,
simulation cameras, reset randomization, Vive/hand teleoperation wrappers, and
Zarr demo recording. Tasks outside that capability list remain preview-only on
Isaac Sim and must use MuJoCo for collection.
