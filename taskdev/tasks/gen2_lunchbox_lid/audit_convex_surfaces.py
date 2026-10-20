"""Quantify source convex-hull concavity fill, separate from envelope excess.

Samples hull vertices, edge midpoints and face centroids against the identical
visible mesh. This describes source convex geometry, not PhysX simplifier error.
"""
import argparse,json
from pathlib import Path
import numpy as np,trimesh
from pxr import Usd,UsdGeom
p=argparse.ArgumentParser();p.add_argument('scene',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=Usd.Stage.Open(str(a.scene));rows=[];transforms=UsdGeom.XformCache()
for prim in Usd.PrimRange(s.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
    path=str(prim.GetPath())
    if '/HandsOnRobots/' not in path or '/collisions/' not in path or not prim.IsA(UsdGeom.Mesh):continue
    mesh=UsdGeom.Mesh(prim);v=np.array(mesh.GetPointsAttr().Get());indices=np.array(mesh.GetFaceVertexIndicesAttr().Get());counts=np.array(mesh.GetFaceVertexCountsAttr().Get())
    owner=s.GetPrimAtPath(path.split('/collisions/')[0])
    matrix=np.array(transforms.GetLocalToWorldTransform(prim)*transforms.GetLocalToWorldTransform(owner).GetInverse())
    v=np.c_[v,np.ones(len(v))].dot(matrix)[:,:3]
    assert np.all(counts==3)
    obj=trimesh.Trimesh(v,indices.reshape(-1,3),process=False)
    hull=obj.convex_hull;tri=hull.triangles
    samples=np.concatenate([hull.vertices,tri.mean(1),(tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2])
    dist=[]
    for part in np.array_split(samples,max(1,len(samples)//256)):
        _,d,_=trimesh.proximity.closest_point(obj,part);dist.extend(d.tolist())
    row=dict(collider=path,source_vertices=len(v),source_faces=len(counts),hull_vertices=len(hull.vertices),samples=len(samples),max_sampled_concavity_fill_m=max(dist),p95_sampled_concavity_fill_m=float(np.percentile(dist,95)))
    rows.append(row);print(prim.GetName(),row['max_sampled_concavity_fill_m'],flush=True)
    a.out.write_text(json.dumps(dict(rows=rows,method=__doc__),indent=2)+'\n')
