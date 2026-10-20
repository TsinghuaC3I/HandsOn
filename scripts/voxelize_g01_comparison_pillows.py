#!/usr/bin/env python3
"""Bake regular FEM solver cells around the original detailed pillow envelope.

Use the project's Isaac Sim 5.1 interpreter. This is an offline asset build
step; it adds no fixtures, proxies, or animation to a collection task.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', required=True, type=Path)
    parser.add_argument('--resolution', type=int, default=32)
    args = parser.parse_args()
    if not 8 <= args.resolution <= 64:
        parser.error('--resolution must be between 8 and 64')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 320, 'height': 240})
    from pxr import Gf
    from omni.physx.scripts.deformableUtils import compute_voxel_tetrahedral_mesh
    try:
        for profile in ('soft', 'foam'):
            source = np.load(args.geometry_dir / (profile + '-geometry.npz'))
            points, indices = compute_voxel_tetrahedral_mesh(
                [Gf.Vec3f(*map(float, point)) for point in source['points']],
                source['tets'].ravel().tolist(), Gf.Vec3f(1), args.resolution,
            )
            points = np.asarray(points, dtype=np.float64)
            tets = np.asarray(indices, dtype=np.int32).reshape(-1, 4)
            determinants = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])
            negative = determinants < 0
            tets[negative, 1], tets[negative, 2] = tets[negative, 2].copy(), tets[negative, 1].copy()
            incidences, oriented = Counter(), {}
            for a,b,c,d in tets:
                for face in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
                    key = tuple(sorted(map(int, face)))
                    incidences[key] += 1
                    oriented[key] = face
            faces = np.asarray([oriented[k] for k,n in incidences.items() if n == 1], dtype=np.int32)
            surface_ids, inverse = np.unique(faces, return_inverse=True)
            volumes = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) / 6
            boundary_volume = np.einsum('ij,ij->i', points[faces[:,0]], np.cross(points[faces[:,1]],points[faces[:,2]])).sum()/6
            nonmanifold_faces = sum(count > 2 for count in incidences.values())
            if nonmanifold_faces or volumes.min() <= 1e-10 or not np.isclose(volumes.sum(),boundary_volume,rtol=1e-6):
                np.savez_compressed(args.geometry_dir / (profile + '-invalid-simulation.npz'), points=points,tets=tets,faces=faces)
                raise RuntimeError(f'Invalid voxel mesh: nonmanifold_faces={nonmanifold_faces}, min={volumes.min()}, sum={volumes.sum()}, boundary={boundary_volume}, nodes={len(points)}, tets={len(tets)}')
            report = dict(profile=profile,mesher=f'PhysX TetMaker voxel resolution {args.resolution}',
                node_count=len(points),tet_count=len(tets),surface_vertex_count=len(surface_ids),
                surface_face_count=len(faces),minimum_tet_volume_m3=float(volumes.min()),
                volume_m3=float(volumes.sum()),boundary_volume_m3=float(boundary_volume),
                bounds_min=points.min(0).tolist(),bounds_max=points.max(0).tolist())
            np.savez_compressed(args.geometry_dir/(profile+'-simulation.npz'),points=points,tets=tets,
                faces=faces,skin=points[surface_ids],skin_faces=inverse.reshape(-1,3))
            (args.geometry_dir/(profile+'-simulation-report.json')).write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    except BaseException:
        # Kit's close can exit the interpreter before an exception propagates.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()


if __name__ == '__main__':
    main()
