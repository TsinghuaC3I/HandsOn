# Default rendering

HandsOn uses one default appearance for previews, native environments and
replay. It includes detailed ROKAE/Wuji1 meshes, palm-side tread, routed wrist
leads, a CAD station body, and a textured laboratory with area lights. Source
task objects and their native material hooks remain task-owned.

ROKAE AR5-08 geometry and converted arm layers are bundled with the other
rendering resources; see [source and license evidence](robot-assets.md#rokae-ar5-08-provenance).
The paper appearance and all rendering asset hashes are preserved.

The assets live in `handson/assets/rendering` inside the Python package. The
root manifest and every dependency are checked by SHA256. No download, internal
filesystem path or alternate appearance profile is used at runtime. An optional
`HANDSON_RENDER_ASSETS` path must contain the identical checked bundle. Missing
or changed assets stop startup with a concrete error instead of a visual fallback.

Robot references, body and room are attached before the first physics reset.
The original collection visibility mask is restored before inserting the
visual bundle, while native control flags and physical state are preserved.
The adapter checks physical schemas, collision geometry and effective physics
materials before and after insertion. It verifies all robot references again
on reuse. After physics starts, cable vertices follow current measured body
poses; the adapter does not replace robot references or write physical state.

The default renderer uses path tracing, 32 samples per pixel, eight bounces,
manual exposure, and the packaged color/normal/roughness maps. The operator
preview is 1280×960; changing that viewport does not change policy camera poses.
The default main observation is 854×480; wrist observations are 848×480.
The head camera uses the packaged native lens schema, including its calibrated
principal point and distortion. The three policy sensors have independent
exposure settings. A preview is an
environment diagnostic, not a policy evaluation or task-success result.

The preview writes a `.render.json` file next to `HANDSON_LIVE_PREVIEW_PATH`.
Keep this receipt with the image: it contains the actual bundle checksums,
robot references, room bindings, hidden helpers and physics-invariance results.

AmbientCG material maps retain their CC0 attribution and source URLs in
`materials/ASSETS.json`. `station/BODY_ASSET.json` and `robot/manifest.json`
identify the geometry and its dependencies. Wrist cables are a visual routing
approximation; they do not add cable contact dynamics.

The room material and lighting sample is deterministic for each task and seed;
it stays fixed within an episode. Requesting RGB image observations enables
live sensors by default. Explicit offline recording remains available through
the collection interface.

## Evaluation validation

The same RTX appearance and calibrated three-view capture are used by the
optional [Newton evaluator](newton-evaluation.md). Backend selection changes
physics and reset semantics; it does not select a simplified appearance.

Use the documented entry to exercise native reset, held measured EEF52 actions,
and all three policy cameras:

```bash
python -m handson.evaluation --check
python -m handson.evaluation --task eval_flex02_fold_towel_twice \
  --diagnostic-steps 90 --seed 0 --output outputs/towel
```

The first command checks all 66 factories without Isaac Sim. The second starts
a real simulator and writes three videos, PNGs, camera calibration and exposure
receipts, measured state/action readback, actual physics clocks, and asset hashes.
Video cadence is one frame per 30 actions by default; each sampled view receives
six render passes without advancing physics. Use a new output directory each time.

The evaluator preserves native initialization rejection. An origin rejection,
missing resource or incomplete worker exit produces an error and cannot become
a policy failure or a completed score. Diagnostic results have `success: null`
and `counts_as_formal_trial: false`. A successful held-action diagnostic does
not establish that a learned policy can solve a task, or certify every possible
reset seed and success path.

For a rendering diagnostic rejected by native initialization, preserve its
output directory and select the next seed before running a new attempt. For
example, the supplementary box-cover check uses `--seed 1` after seed 0 was
rejected. The command records a separate attempt; it does not change the
default seed or silently retry. Model-evaluation comparisons must use the same
predeclared seed and reset-admission protocol across models, and must never
replace a completed policy failure with another seed.

The packaged `evaluation/source_provenance.json` identifies the original and
adapted code hashes for transplanted components. The task-specific origin
and contact serializer manifests preserve their paired implementations;
contact serialization changes can invalidate the original exact-value checks.
See [policy integration](custom_policy_integration.md) for action units,
checkpoint metadata, language sampling and the two disclosed scoring waivers.

The latest installed-package run is documented in [release validation](release-validation.md), with per-task diagnostics and three-view samples.
