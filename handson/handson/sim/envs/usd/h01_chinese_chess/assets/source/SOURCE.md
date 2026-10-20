# H01 Chinese Chess asset provenance status

## Packaged legacy archive (not used by the active H01 MJCF)

- Title: Chinese Chess
- Author: [chung_the_artist](https://sketchfab.com/chung_the_artist)
- Source: [Sketchfab model](https://sketchfab.com/3d-models/chinese-chess-7ad0f4f0ebbc455d9e9f829c956dda80)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Archive: `chinese_chess.zip` (SHA-256 `65469cb29af91124baaf414e4c94b0063bf08a8a3b9526573aef414c00d87604`)
- License file: `LICENSE.txt`

This is the only upstream archive retained in this source directory. Its mesh
fingerprints correspond to the older numeric-suffixed piece OBJ family and the
legacy `h01_chess_board.obj`. Those legacy OBJ files are not referenced by the
active `h01_chinese_chess_objects.xml`. This archive and its `LICENSE.txt` are
not provenance or license evidence for the active H01 assets.

## Active checked-in derived OBJ assets (upstream closure unresolved)

The active MJCF asset table references 44 checked-in derived OBJ files under
`sim/envs/xmls/assets/h01_chinese_chess/`: the `h01_chess_board_068.obj` board
visual, the compact `h01_chess_board_support.obj` audit mesh, 14 complete piece
geometry audit meshes, and 28 wood/text visual submeshes. The active runtime
uses transparent analytic cylinders inscribed in those 14 complete piece
meshes, whose rounded source bottoms previously produced persistent PhysX
micro-rocking. These checked-in bytes are the only currently auditable inputs
to the active runtime.

Historical build metadata names the following intended upstream inputs:

- Board candidate: “Chinese chess board - Xiangqi” by `elsa2297`, Sketchfab
  model `068525be5e4e44aaa10d34c01f72f545`; expected archive
  `chinese_chess_board_-_xiangqi.zip`, documentary SHA-256
  `905c47a5e310cd5589f3014bdd6d6b9f87879914a628fc2ec41822045bfe2223`.
- Piece candidate: “Chinese chess - 218414” by `mming1106`; expected archives
  `Chinese chess - 218414 - part 1 of 2.zip` and
  `Chinese chess - 218414 - part 2 of 2.zip`, documentary SHA-256 values
  `d3d92f434feef045c489b580f0445810fc470b42c16be6e5ab3935c53a3bc2aa`
  and `17e67e249495910ad4c65a1e33d770079af878791483c68cf37bac4a36927641`.

Those three archive bytes and their exact license bytes are absent from the
repository. The current builder also does not bind both documented piece
archives to an output manifest. The active OBJ files therefore have unresolved
upstream provenance and license closure. The documentary names, URLs, and
hashes above must not be presented as verified attribution or as a commercial
release license until the exact inputs and licenses are restored and checked.

## Collision audit boundary

The active board collider is an analytic box. Its X/Y bounds are derived from
the opposite straight-edge intersections in the checked-in compact support OBJ,
and its Z bounds span that checked-in OBJ's top and bottom planes. This proves
internal geometric consistency between the active derived bytes, MJCF collider,
table support, and initial piece placement. It does not prove that those bytes
match either missing upstream archive, and it does not close their licenses.

The active board visual and support mesh are scaled by `0.873 0.873 1` in their
MJCF asset declarations (the authored `0.97` scale multiplied by `0.90`), and
the logical grid spacing is 55.872 mm. Each active piece mesh is scaled by
`1.485 1.485 1` (the authored `1.65` scale multiplied by `0.90`), producing an
approximately 44.25 mm diameter by 15 mm thick piece—about 79% of the grid
spacing. Each active collider is a transparent 22.1265 mm radius,
7.5 mm half-height cylinder with the original 32 g mass. The cylinder is
strictly inside the scaled piece bounds and shares their 15 mm height,
preserving the visible grasp size while giving every initially seated piece a
planar support face. The 14 complete piece meshes remain checked-in audit
inputs, not active PhysX collision meshes.

The unresolved upstream provenance remains a redistribution concern, but it no
longer blocks local H01 collection. The active checked-in geometry, analytic
colliders, and Isaac Sim 5.1 collection path are the runtime inputs.
