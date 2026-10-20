# LINGO-Space Hanoi assets

- Source project: LINGO-Space
- Source repository: https://github.com/rirolab/LINGO-Space
- Source revision: `7cca0c9b89faa0b3f2cbb145f4f5b326dbe1f33d`
- Source path: `dataset_generation/environments/assets/hanoi/`
- Copyright: 2024 Robust Intelligence and Robotics Laboratory
- License: MIT

The three ring meshes are geometry-identical renamed copies of the upstream
files:

- `disk_small.obj` from `disk1.obj` (70 x 70 x 10 source units)
- `disk_medium.obj` from `disk2.obj` (80 x 80 x 10 source units)
- `disk_large.obj` from `disk3.obj` (90 x 90 x 10 source units)

The upstream `disk1.urdf`, `disk2.urdf`, `disk3.urdf`, and `stand.urdf` are
retained here as the runtime source of dimensions, colors, masses, visual
meshes, and collision geometry. `stand.urdf` is whitespace-compacted only;
its URDF values are unchanged.

The dangling `obj.mtl` declaration and `usemtl` statements were removed from
the OBJ copies because that material file is not distributed upstream. Vertex
and face data are unchanged. D04 uses the URDF scale `0.001 0.001 0.001`
without resizing, uses the same upstream OBJ for visual and collision as the
URDF declares, and enables PhysX SDF collision on that mesh so the annular
hole remains non-convex and physical. The full three-part stand is flattened
from the upstream fixed-link URDF into the D04 task body without changing any
primitive dimensions or poses.

The `collision_wedge_*.obj` files are unused legacy HandsOn derivatives from
the superseded partial integration. They remain only to keep old commits and
cached scene references inspectable; the current D04 scene does not reference
them.

## MIT License

Copyright (c) 2024 Robust Intelligence and Robotics Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
