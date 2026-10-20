# Adopted ROKAE / Wuji1 runtime visual assets

Source: sim-bench-3:/data/luotianwei/blender-isaac-real-a03-20260907/asset-v11
Reference: EP-20260831-1412 (ROKAE AR5-5_08, Wuji1).
Adopted by Blender-Isaac-优化过程-2026-09-08.md: robot asset-v11. The v12 object materials are separately scoped to A03 and are not loaded here.

The unmodified manifest, 60 link payloads and four declared dependencies are included and SHA256-verified. Project robot CAD provenance remains in manifest.json; this migration does not assign a new license to project-derived assets.

Only visual children are replaced. No colliders, mass/inertia, contacts, joints or drives are added. Glove thickness/grip bumps and cable routes are photo estimates, not measured contact geometry. The cable profile is copied from the adopted realism-v11 version. The runtime preserves the nominal route when reachable and reports a visual extension when another task's wrist pose exceeds the estimated source-clip cable length. It never applies a cable force.

V13 camera/head candidates and A03 scene/object overlays are excluded. The relocated runtime package has no dependency on the original machine path. The editable Blender file remains at the source path; it is not a runtime dependency. Its currently observed hash differs from the historical manifest entry and it has not been repackaged as a verified authoring source.

The task_visual_appearance module loads this package at fresh construction for tasks owning a private visual_appearance.py. HANDSON_ISAACSIM_TASK_VISUAL_PROFILE=source selects the previous GEN baseline for fresh-process A/B. Formal online recording remains state/action-only with three-camera post-rendering. The reference laminate and light values derive from the adopted capture pipeline; each task retains its own object provenance. No whole-scene metric calibration is claimed.
