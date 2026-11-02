# Maintaining the restored evaluation runtime

The evaluator must preserve the historical task behavior while sharing code
whose behavior is the same. Task/source selection belongs in the registry and
explicit manifests; a new collection implementation must not silently replace
a paper evaluation route.

## Runtime organization

- `tasks/registry.py` selects each task's environment, camera, trace and evaluation
  adapter. `evaluation/paper_trials.py` selects archived inputs;
  `paper_variants.py` handles the explicitly named device-cover variant.
- `evaluation/task_contracts.py` installs task controls and the accepted recording
  origin. `camera_contracts.py` scopes the two historical camera profiles.
- `sim/isaaclab/bimanual_runtime.py` contains common robot runtime support.
  `collection_runtime.py` contains common collection/environment methods.
  Their mixins own no task configuration or separate environment state.
- `core_*_base.py` and `core_*_env.py` retain differing initialization, control,
  geometry and native task rules. Some remain large because the historical
  variants have substantial differences; they are not interchangeable aliases.
- `default_appearance.py` installs robot/cables, task surfaces and room assets
  before physics. `task_surfaces.py` verifies the portable task-specific layers.
- `replay/` supplies the two explicit episode initialization bundles. This is
  separate from ordinary benchmark reset and policy evaluation.

The shared-runtime extraction removes repeated methods only when their syntax
and imported library bindings match. Methods using module-specific constants,
dynamic module globals or `super()` are retained. A locally different method
continues to override the mixin. Existing class names, task factories and public
imports remain stable.

`tests/fixtures/runtime_extraction.json` records method fingerprints from the
verified pre-refactor implementation. Its regression checks resolve inherited
and overridden methods for every affected environment. Do not regenerate this
fixture merely to make a changed implementation pass: a deliberate semantic
change needs its own reviewed protocol and evidence.

## Checks before changing a paper route

Install the CPU environment from the repository root, matching CI:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/cpu-test-lock.txt
python -m pip install --no-deps -e './handson[test]'
python -m pip check
python -m pytest handson/tests -q
python -m handson.evaluation --check
python -m handson.robots --check
python -m handson.sim.isaaclab.render_assets --packaged-only
```

CPU tests cover task routing, reset/success contracts, asset integrity and the
shared implementation. They do not replace simulator validation. For runtime
changes, use the supported Isaac stack and compare against the retained verified
baseline with identical inputs. Keep physical profiles, clocks, camera readbacks,
state arrays and raw differences; keep policy scores separate from diagnostics.
Build and test the installed wheel as well as the source tree.

When a runtime file changes, update its replay integrity dependency explicitly,
including any newly imported shared module. Preserve historical source hashes
and previous restoration identities in provenance records. Changing a checksum
alone is not validation.
