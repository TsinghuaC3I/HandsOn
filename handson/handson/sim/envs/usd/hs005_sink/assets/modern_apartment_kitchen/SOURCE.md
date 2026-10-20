# Modern Apartment kitchen subset

This directory contains a task-local kitchen subset derived from the user's
downloaded `modern_apartment.zip` archive.

- Title: `Modern Apartment`
- Author: Visthétique
- Source: https://sketchfab.com/3d-models/modern-apartment-1fbb649cd6624f2bb7b7d6e30c6533a5
- License: CC BY 4.0; the copied license text is in `LICENSE.txt`
- Original archive: `/home/user/下载/modern_apartment.zip`
- Archive mtime at selection: `2026-08-26 19:21:11 +0800`
- Archive SHA256: `d1b8f506f69fd66ccf1b816d651a884eab999d9db2f4a1ad9f7f491d0044a2a6`

## Selection and conversion

The original apartment contains about three million triangles.  HS005 keeps
only the named kitchen/background groups `Sink`, `Plane.002`, `Plane.007`,
`Dishwasher.001`, `Hobs.001`, `Hood`, `Oven.001`, `Refrigator.001`,
`kitchenblind`, and `Coffee+Maker`.  The resulting visual subset has 42 mesh
nodes and 100,807 triangles.

The selected nodes were exported as a temporary GLB with `trimesh 4.5.1`, then
converted with Isaac Sim 5.1.0's `omni.kit.asset_converter` using Z-up stage
metadata, meter units, separate meshes, embedded material extraction, and no
animation/camera/light import.  Extracted textures were reduced to a maximum
of 1024 pixels per side with Pillow/Lanczos to keep the task-local payload
bounded.  No geometry was decimated or rescaled.

`sink_collision_source.ply` is the exact world-space triangle mesh from the
source node `Sink_Sink4Metal.001_0` (2,634 vertices, 3,753 triangles).  The
generated task layer uses this same source geometry as an invisible static
triangle-mesh collider, preserving the real basin cavity instead of sealing it
with a convex hull or stale box proxy.

## Output hashes

- `modern_apartment_kitchen.usd`: `12ac915b9d442399f10af5b302d9c6ef887e32c8aa1d0f6addd3a1dbc5f58524`
- `sink_collision_source.ply`: `46839123013a8785fcc70b9744fae0745972561e8c399c14c06b394b4f774082`
