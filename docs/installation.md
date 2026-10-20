# Installation and testing

Run commands from the repository root. The default PhysX simulator requires Linux x86_64,
Python 3.11, Isaac Sim 5.1.0.0 and Isaac Lab v2.3.2. The Lab tag reports Python
package version 0.54.2. Inference servers use separate environments described
in [paper models](paper-models.md).

For Newton, use the separate Python 3.12 / Isaac Sim 6 environment and
commands in [Newton evaluation](newton-evaluation.md). Do not apply the PhysX
constraints below to that environment.

## Simulator environment

Install the NVIDIA runtime following the instructions for those exact versions,
then activate it and run:

```bash
python -m pip check
python -m pip install -c requirements/simulator-constraints.txt -e ./handson
python -m pip check
python -m handson.evaluation --check
python -m handson.robots --check
python -m handson.sim.isaaclab.render_assets --packaged-only
```

Native rollouts use the bundled ROKAE AR5-08/Wuji robot and prepared arm render
layers. Run `python -m handson.sim.isaaclab.render_assets` to verify the full
bundle. Source/license evidence is in [robot assets](robot-assets.md).
Tianji is a separate optional [local import](robot-assets.md#external-assets).

The constraints align HandsOn with the simulator's NumPy and typing-extensions
requirements and Lab's Gymnasium requirement. They also bound OpenCV (newer
releases require NumPy 2), Zarr codecs, and Tyro. They are not a complete NVIDIA
environment lock. Record any pre-existing `pip check` errors before installing
HandsOn.

Isaac Sim 5.1 and the full Isaac Lab v2.3.2 dependency declarations themselves
conflict: the kernel pins FastAPI 0.115.7, while Lab pins Starlette 0.49.1, outside
that FastAPI version's supported range. HandsOn does not use that web server,
but this remains an upstream packaging conflict. Do not treat a successful
GPU startup as a clean dependency resolution, or hide it with a global
`--no-deps` install. The repository has not certified installation of the full
NVIDIA stack from an empty machine. The CPU lock below does not install it.

Install `ffmpeg` on `PATH`. Each GPU diagnostic requires a fresh output directory:

```bash
python -m handson.evaluation --task t12_load_and_start_microwave \
  --diagnostic-steps 90 --seed 0 --output outputs/microwave
```

`--check` constructs all 66 CPU factories. It does not initialize physics,
capture images, run a checkpoint, or certify success predicates in simulation.
GPU diagnostics use held measured actions and report no model success score.

## CPU regression environment

Use a separate environment so optional test tools cannot change the simulator.
Git must be on `PATH` for the source-provenance tests. Mesh-cache tests copy
assets into temporary directories; put `TMPDIR` on a volume with several GB
of free space rather than a small container system disk:

```bash
python3.11 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements/cpu-test-lock.txt
python -m pip install --no-deps -e './handson[test]'
python -m pip check
mkdir -p outputs/test-tmp
export TMPDIR="$PWD/outputs/test-tmp"
python -m pytest handson/tests -q
```

The lock records the verified Linux/Python 3.11 CPU packages. Here `--no-deps`
only installs the local package after its dependencies have been installed from
the lock; `pip check` verifies that they satisfy its metadata. Kit boundaries in
unit tests use explicit mocks. Optional Torch/GPU checks may skip in this CPU
environment; a green result is not a GPU or trained-policy certification.
Tests loading ROKAE kinematics and meshes verify the bundled files. Missing or
modified assets fail integrity checks rather than being skipped.

The suite covers the released task registry, native task semantics, assets,
replay, and policy interfaces. Tests for removed prototype tasks, removed MuJoCo
environments, private collection UI/SDKs and archival acceptance evidence have
been retired from the maintained release suite. Historical `scripts/tests/`
files can still reference unshipped collection tools and are outside that suite.
This changes the test population; compare coverage and test IDs,
not raw pass counts against older releases.

## Known GPU limitation

An external RTX 4090 / driver 580.178.04 run of `t03_cover_box`, seed 1, hit
PhysX CUDA error 700 twice during initialization. The same source and 90-action
command completed on an RTX 5090 / driver 570.153.02. The 4090 problem remains
unresolved; no reset threshold, collision geometry or success condition was
changed to bypass it. See [release validation](release-validation.md).
