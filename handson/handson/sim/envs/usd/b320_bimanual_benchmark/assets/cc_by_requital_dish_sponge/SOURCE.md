# Sponge For Washing Dishes

- Author: Requital (Azamkhon)
- Source: https://sketchfab.com/3d-models/sponge-for-washing-dishes-7ec3822721dd416aacc6308a7ed46c2c
- Author page: https://sketchfab.com/requital_0111
- License: CC BY 4.0 (commercial use allowed with attribution)
- Downloaded archive: `sponge_for_washing_dishes.zip`
- Archive SHA-256: `2003f17f0913bb7383d48a369fa345fb5ea18f405ea4a6ecb9f51646e852a914`
- Download timestamp: `2026-08-26 21:14:45 +0800`

The complete downloaded glTF closure and embedded `license.txt` are retained
unchanged under `source/`. The source combines the sponge and its black holder
in one mesh, so B320-KA020 does not reference the full conversion.

`derived/sponge_only.gltf` deterministically retains only the five connected
sponge shells (1,905 vertices / 3,180 triangles), removes all 45,508 holder
triangles, and recenters the result on a Y-up support frame.
`dish_sponge_only_visual.usd` is the render-only Isaac Sim 5.1 conversion of
that derivative. The converted default root is identity and the normalized
transform is baked into its child, so task composition cannot reintroduce the
source holder offset or double-apply its rotation. Its measured
94.9 x 53.4 x 16.5 mm bounds also define the single collision box; no
bowl/holder visual or collision is part of the KA020 dynamic body. The exact
derivation is reproducible with
`taskdev/tasks/bench320_ka020_sponge_stabilized_soap_dispense/prepare_sponge_only_asset.py`.
