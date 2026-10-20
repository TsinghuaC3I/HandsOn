"""CPU audit of the actual Isaac MJCF import, including its face-corner UVs.

Pass the source cold run's cached eval_gen06_spray_shoe_scene_isaac.usd.
This reads the converter's cache before normal runtime preparation. It does
not apply an appearance to that still-instanced cache or save any USD asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from handson.tasks.eval_gen06_spray_shoe import visual_appearance as appearance


def audit(stage):
    rows = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if not prim.IsA(UsdGeom.Mesh) or name not in appearance.source_materials()['materials']:
            continue
        obj = appearance.ASSETS.parent / 'meshes' / f'{name}.obj'
        vertices, texture_coordinates, faces = [], [], []
        for line in obj.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == 'vt':
                texture_coordinates.append([float(x) for x in fields[1:3]])
            elif fields[0] == 'f':
                faces.append([[int(i) - 1 if i else None for i in c.split('/')]
                              for c in fields[1:]])
        mesh = UsdGeom.Mesh(prim)
        corners = np.asarray(mesh.GetPointsAttr().Get())[np.asarray(mesh.GetFaceVertexIndicesAttr().Get())]
        original_corners = np.array([vertices[i[0]] for face in faces for i in face])
        np.testing.assert_allclose(corners, original_corners, atol=1e-7)
        row = {'name': name, 'corner_count': len(corners),
               'points_corner_max_delta': float(np.max(np.abs(corners - original_corners))),
               'authored_tangent_attributes': [a.GetName() for a in prim.GetAttributes()
                                               if 'tangent' in a.GetName().lower()]}
        if texture_coordinates:
            uv = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar('st').Get())
            original_uv = np.array([texture_coordinates[i[1]] for face in faces for i in face])
            np.testing.assert_allclose(uv, original_uv, atol=1e-7)
            flipped = original_uv.copy()
            flipped[:, 1] = 1 - flipped[:, 1]
            row.update(usd_matches_obj_uv_max_delta=float(np.max(np.abs(uv - original_uv))),
                       usd_matches_reflipped_uv_max_delta=float(np.max(np.abs(uv - flipped))),
                       uv_sha256=hashlib.sha256(uv.tobytes()).hexdigest())
        rows.append(row)
    assert len(rows) == 14
    return {'passed': True, 'source_mesh_count': len(rows), 'meshes': rows,
            'usd_stage_saved': False, 'audit_scope': 'Raw MJCF converter geometry and UVs'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.usd))
    result = audit(stage)
    result['source_usd'] = str(args.usd)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': result['passed'], 'source_mesh_count': result['source_mesh_count']}))


if __name__ == '__main__':
    main()
