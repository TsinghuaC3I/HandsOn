#!/usr/bin/env python3
"""Deterministic source-mesh versus compound-collision exterior distances."""
from pathlib import Path
import numpy as np
import trimesh
import json
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, UsdPhysics
from handson.sim.isaaclab.gen4_scene import ASSET_DIR


def main():
    stage=Usd.Stage.Open(str(ASSET_DIR/'glue_stick.usd'));visuals=[];hulls=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        usd=UsdGeom.Mesh(prim)
        mesh=trimesh.Trimesh(vertices=np.asarray(usd.GetPointsAttr().Get()),faces=np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
        if prim.HasAPI(UsdPhysics.CollisionAPI): hulls.append((str(prim.GetPath()),mesh))
        else: visuals.append(mesh)
    visual=trimesh.util.concatenate(visuals)
    query_visual=visual.copy(); query_visual.apply_scale(1000.0)
    equations=[ConvexHull(mesh.vertices).equations for _,mesh in hulls]
    results=[]
    for i,(name,mesh) in enumerate(hulls):
        points=np.concatenate([mesh.vertices,mesh.triangles_center]);inside=np.zeros(len(points),dtype=bool)
        for j,eq in enumerate(equations):
            if i!=j: inside |= np.all(points@eq[:,:3].T+eq[:,3]<=1e-6,axis=1)
        exterior=points[~inside]
        if not len(exterior): continue
        _,distance,_=trimesh.proximity.closest_point(query_visual,exterior*1000.0)
        distance/=1000.0
        results.append({'collider':name,'vertices':len(mesh.vertices),'exterior_samples':len(exterior),
            'max_surface_distance_m':float(distance.max()),'p99_surface_distance_m':float(np.quantile(distance,.99)),
            'worst_sample_m':exterior[distance.argmax()].tolist()})
    bounds=np.asarray([np.concatenate([m.vertices for _,m in hulls]).min(0),np.concatenate([m.vertices for _,m in hulls]).max(0)])
    report={'method':'Every hull vertex and face centroid on the compound exterior versus original visual triangles; deterministic CPU mesh distance',
        'results':results,'max_surface_distance_m':max(r['max_surface_distance_m'] for r in results),
        'visual_bounds_m':visual.bounds.tolist(),'collision_bounds_m':bounds.tolist(),
        'actual_hand_surface_sweep':'pending'}
    out=Path(__file__).resolve().parent/'evidence/glue_proxy_surface_cpu.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':main()
