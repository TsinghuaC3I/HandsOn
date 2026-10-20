# Isaac Lab `bimanual_assembly` task preview

For the all-task launcher and complete supported task list, see
[`isaacsim_task_preview.md`](isaacsim_task_preview.md). This document describes
the dedicated `bimanual_assembly` implementation.

This preview migrates the HandsOn ROKAE + Wuji `bimanual_assembly` scene to a
native Isaac Lab/PhysX scene. It is intentionally limited to correct task and
asset preview; teleoperation, external devices, policy observations, and data
collection are not part of this stage.

## What is simulated

- two 7-DoF ROKAE arms, each with a 20-DoF Wuji hand, as PhysX articulations
- the original detailed robot and hand visual meshes
- simplified Wuji collision geometry already shipped with HandsOn, to keep
  PhysX collision cooking and startup time practical
- the original peg and socket assets as dynamic rigid bodies
- the original six-wall arena, tiled floor, worktable, metal legs, central
  robot body/head housing, and task lighting
- source arena textures converted to projected OmniPBR materials (floor tile,
  plaster walls, ceramic desktop, and brushed-steel legs)
- the original MuJoCo arm, Wuji hand, peg, socket, and center-housing material
  colors, including a runtime repair for Isaac Sim 5.1's nested
  `DefaultMaterial` mesh bindings
- a collision-enabled kinematic worktable and invisible PhysX collision meshes
- real joint-position drives and gravity/contact stepping; objects and robots
  are not moved by USD Xform updates during the simulation loop

The scene structure and configuration follow the native Isaac Lab patterns used
by DexVerse, while all task assets come from this HandsOn checkout.

## Environment

The tested setup is:

- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- Python 3.11

Install Isaac Lab into the Python environment that contains Isaac Sim. For the
local checkout used during migration:

```bash
conda activate isaacsim
cd /home/user/panting/IsaacLab
./isaaclab.sh --install none
```

## Interactive preview

Run from the HandsOn repository root:

```bash
cd /home/user/handson
OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/miniconda3/envs/isaacsim/bin/python \
  scripts/preview_isaaclab_bimanual_assembly.py
```

The default `--max_steps 0` keeps the preview open until the Isaac Sim window
is closed. The script prints both 27-joint articulations and the live peg/socket
state after startup.

## Headless validation

Use a finite run for a reproducible physics smoke test:

```bash
cd /home/user/handson
OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/miniconda3/envs/isaacsim/bin/python \
  scripts/preview_isaaclab_bimanual_assembly.py \
  --headless --max_steps 600 --status_interval 300
```

A successful run ends with:

```text
[HandsOn/IsaacLab] Preview PASSED after 600 steps
```

Useful options:

| Option | Default | Description |
| --- | ---: | --- |
| `--num_envs` | `1` | Number of tiled task scenes. |
| `--max_steps` | `0` | Physics steps before exit; `0` runs until the window closes. |
| `--status_interval` | `240` | Interval for live physics diagnostics; `0` disables them. |
| `--device` | `cpu` on this workstation | Simulation device, for example `cuda:0`. |
| `--headless` | off | Run without the Isaac Sim window. |

The current workstation defaults to CPU because its installed PyTorch CUDA
build does not include RTX 5090 (`sm_120`) kernels. After installing a matching
CUDA 12.8 PyTorch build, pass `--device cuda:0` to opt into GPU simulation.

## Generated asset cache

On first launch, the MJCF importer creates Isaac-oriented MJCF/USD derivatives
under:

```text
~/.cache/handson/isaaclab/
```

Set `HANDSON_ISAACLAB_CACHE` to use another cache directory. Delete the cache to
force a clean asset conversion after modifying source MJCF or mesh assets.

The default desktop uses the shipped `table_arena/textures/ceramic.png`. Set
`HANDSON_ISAACLAB_TABLE_TEXTURE` to a path relative to `table_arena/textures`
to select one of the original randomization textures, for example:

```bash
HANDSON_ISAACLAB_TABLE_TEXTURE=desktop_random_textures/warm_wood_grain_2.png \
  OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/miniconda3/envs/isaacsim/bin/python \
  scripts/preview_isaaclab_bimanual_assembly.py
```

## Current boundary

This milestone validates high-fidelity scene/asset conversion, articulation
topology, initial task layout, contacts, and stable PhysX stepping. The source
MuJoCo scene's invisible tablecloth/randomization clutter is intentionally not
spawned. It does not yet implement the HandsOn task action/observation API,
insertion success metrics, camera observations, domain randomization scheduling,
policy evaluation, teleoperation, or other external-device integration.

The older raw Isaac Sim compatibility scaffold remains in
`handson/sim/isaacsim/`, but it is not the implementation used by this native
preview.
