# FU001 selected asset provenance

The selected candidate is Poly Haven's “Painted Wooden Cabinet”, authored by
Kirill Sannikov / Poly Haven: <https://polyhaven.com/a/painted_wooden_cabinet>.
The upstream license is CC0 1.0 as documented by Poly Haven at
<https://polyhaven.com/license>. The downloaded ZIP did not contain a license
file, so the external license URL and the reviewed archive SHA-256 are bound
in `asset_manifest.json` and `LICENSE.txt`.

Reviewed download: `painted_wooden_cabinet_4k.gltf.zip`

`dec8c52b933705c3332619faf6ec8618f5e8ca4ff57c24f25016e9876eba0ddf`

The source is glTF 2.0, meter convention, Y-up, with independent cabinet body,
`door_01`, and `door_02` nodes. The measured body envelope is approximately
1.1947 m wide × 1.1819 m high × 0.5465 m deep. The asset converter first maps
source `(x,y,z)` to intermediate Z-up runtime `(x,-z,y)` (+90° about X), so
the source `+Y` hinge becomes intermediate `+Z`. The canonical B320 scene then
applies `Rz(-90°)` with `(w,x,y,z)=(0.5,0.5,-0.5,-0.5)`; the composed
source-to-world map is `(x,y,z) -> (-z,-x,y)`, and the hinge remains world `+Z`.
Conversion settings and per-file hashes are in `asset_manifest.json`.

The three runtime USD components preserve the original mesh/material topology.
Collision APIs are authored on those same source meshes and request PhysX
`convexDecomposition`; no box or invisible placeholder is used as the cabinet
or door collision. The canonical B320 runtime composes these components with a
uniform `0.55` scale and `(w,x,y,z)=(0.5,0.5,-0.5,-0.5)` frame transform. The
runtime copy is under
`handson/handson/sim/envs/usd/b320_bimanual_benchmark/assets/` while this task
directory remains the source/provenance record.
