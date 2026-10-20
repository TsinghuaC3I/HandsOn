# Old Fashioned Wooden Blocks

- Title: **Old Fashioned Wooden Blocks**
- Author: [CurlyFryWhy](https://sketchfab.com/CurlyFryWhy)
- Source: [Sketchfab model page](https://sketchfab.com/3d-models/old-fashioned-wooden-blocks-8331752cf38243638b0681f5ea230c28)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Original download: `old_fashioned_wooden_blocks.zip`
- Archive SHA-256: `983de66500f8ab94997929442d355cec21c9afa66d0fa27af7860202a760c6b8`

The complete downloaded glTF closure is preserved in `source/`, including the
original `license.txt`, binary buffer, and all source textures. The source
scene contains four independent block meshes. BC021 selects the broad square
source mesh `Object_2/Object_0` for each of its two rigid bodies, deactivates
the other nodes, and keeps that selected visible mesh as the collision source.

`source/converted_scene.usd` was generated with the pinned Isaac Sim 5.1
`omni.kit.asset_converter` 5.0.17. Each block wrapper applies one uniform
scale so its largest transformed extent is 75 mm and places the rigid-body
origin at the measured source-mesh center on all three axes. This matches the
BC021 reset contract, whose poses are body-center poses; the resulting local
visual and collision bounds are symmetric around that origin. The wrapper
authors one PhysX `convexHull` collider on that same visible mesh; there is no
rendered proxy or reused placeholder collider.

The downloaded archive contains three zero-byte optional texture maps
(metallic/roughness, normal, and specular). They remain unchanged in the
preserved source closure. The derived USD clears only those unusable optional
shader file inputs, retaining the valid wood base-color maps so RTX does not
fall back to gray materials or emit missing-texture errors.

The required attribution is:

> This work is based on "Old Fashioned Wooden Blocks" (https://sketchfab.com/3d-models/old-fashioned-wooden-blocks-8331752cf38243638b0681f5ea230c28) by CurlyFryWhy (https://sketchfab.com/CurlyFryWhy) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
