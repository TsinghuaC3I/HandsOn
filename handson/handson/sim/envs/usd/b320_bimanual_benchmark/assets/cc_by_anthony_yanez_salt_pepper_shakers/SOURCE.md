# Anthony Yanez salt and pepper shakers

- Title: `Salt And Pepper Shakers`
- Author: [Anthony Yanez](https://sketchfab.com/paulyanez)
- Source: [Sketchfab model page](https://sketchfab.com/3d-models/salt-and-pepper-shakers-5f2adebdda754e40915d1330a61fa70c)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Original download: `salt_and_pepper_shakers.zip`
- Download SHA-256: `a60fe7e5289422527695254baf4c6d98463fd71a3c5c5cd0c6e88696dcb9594d`
- Source `scene.gltf` SHA-256: `769ce0c35612ed8e4741e42a914c27cf775ce302c5b962dd7c072a3328b145cb`

The complete downloaded glTF closure is preserved in `source/`: the glTF,
external binary buffer, four color/normal textures, and the original
Sketchfab-generated `license.txt`. The archive in the user's download folder
was not modified.

`source/converted_scene.usd` was generated with the repository's pinned Isaac
Sim 5.1.0.0 `omni.kit.asset_converter` 5.0.17. Conversion preserves separate
meshes, PreviewSurface materials, transparency, and external textures. The
converter declares a Z-up stage but retains the glTF's Y-up point coordinates,
so each task wrapper owns an explicit, auditable Y-up-to-Z-up transform.

The source scene contains separate `Peeper` (the source spelling) and `Salt`
groups. `pepper_jar.usda` and `salt_jar.usda` each reference the complete
converted scene so `/Looks` and texture bindings remain intact, then deactivate
the other jar. Both wrappers recenter the selected jar at local XY zero, put its
lowest point at local Z zero, and uniformly normalize it to 0.096 m high. The
resulting pepper and salt widths are approximately 0.05676 m and 0.05589 m,
respectively, which fit inside BC014's 0.075 m target markers.

Each wrapper authors collision only on the downloaded outer bottle mesh and
metal cap mesh, using PhysX `convexHull`. Those same meshes remain the visible
surfaces; there is no rendered proxy or old DexVerse collider. The internal
salt/pepper content mesh is visual-only. Runtime code owns the single rigid-body
root, mass, damping, friction, CCD, and semantic labels.
