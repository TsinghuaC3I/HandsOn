#!/usr/bin/env python3
"""Compare first contact from six sides, including the recessed base opening."""
from pathlib import Path
import json
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics
from handson.sim.isaaclab.gen4_scene import ASSET_DIR


def first_hits(mesh, origins, directions):
    locations, ray_ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    depths=np.full(len(origins),np.inf)
    if len(ray_ids): np.minimum.at(depths,ray_ids,np.einsum('ij,ij->i',locations-origins[ray_ids],directions[ray_ids]))
    return depths


def main():
    stage=Usd.Stage.Open(str(ASSET_DIR/'glue_stick.usd'));groups=[[],[]]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        u=UsdGeom.Mesh(prim)
        m=trimesh.Trimesh(np.asarray(u.GetPointsAttr().Get()),np.asarray(u.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
        groups[int(prim.HasAPI(UsdPhysics.CollisionAPI))].append(m)
    visual,collision=[trimesh.util.concatenate(ms) for ms in groups]
    # Millimetres avoid numerical rejection of the source's tiny triangles.
    visual.apply_scale(1000.0); collision.apply_scale(1000.0)
    bounds=np.asarray([np.minimum(visual.bounds[0],collision.bounds[0]),np.maximum(visual.bounds[1],collision.bounds[1])])
    results=[]
    for axis in range(3):
        axes=[i for i in range(3) if i!=axis]
        u,v=np.meshgrid(*[np.linspace(bounds[0,i]+.2,bounds[1,i]-.2,64) for i in axes])
        for sign in [-1,1]:
            origins=np.zeros((u.size,3));origins[:,axes[0]]=u.ravel();origins[:,axes[1]]=v.ravel()
            origins[:,axis]=bounds[0 if sign==1 else 1,axis]-sign*10.0
            directions=np.zeros_like(origins);directions[:,axis]=sign
            a,b=first_hits(visual,origins,directions),first_hits(collision,origins,directions)
            both=np.isfinite(a)&np.isfinite(b);delta=(a[both]-b[both])/1000.0
            occupied=np.isfinite(b)
            first_points=origins[occupied]+directions[occupied]*b[occupied,None]
            _,distance,_=trimesh.proximity.closest_point(visual,first_points)
            distance/=1000.0
            results.append({'axis':axis,'direction':sign,'rays':len(a),'common_hits':int(both.sum()),
                'max_outward_m':float(max(0,delta.max())),'p99_outward_m':float(max(0,np.quantile(delta,.99))),
                'max_inward_m':float(max(0,-delta.min())),
                'first_contact_surface_max_distance_m':float(distance.max()),
                'first_contact_surface_p99_distance_m':float(np.quantile(distance,.99)),
                'surface_worst_point_m':(first_points[distance.argmax()]/1000.0).tolist(),
                'collision_only_rays':int((~np.isfinite(a)&np.isfinite(b)).sum()),
                'visual_only_rays':int((np.isfinite(a)&~np.isfinite(b)).sum())})
    report={'method':'First triangle hit on 64x64 orthographic rays from each of six sides; visual and collision geometry from generated USD',
        'results':results,'actual_hand_surface_sweep':'pending'}
    (Path(__file__).resolve().parent/'evidence/glue_first_contact_rays_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
