"""Final USD geometry audit; ignore importer extent caches and transform vertices.

A positive collider excess is a proved protrusion beyond even the visible AABB.
Zero excess alone is not a tight-surface certificate for convex-cooked meshes.
"""
from pathlib import Path
import argparse,json,itertools
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf

p=argparse.ArgumentParser();p.add_argument('scene',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=Usd.Stage.Open(str(a.scene));cache=UsdGeom.XformCache();rows=[]
def points(prim):
    if prim.IsA(UsdGeom.Mesh):return np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
    if prim.IsA(UsdGeom.Cube):return np.array(list(itertools.product((-1,1),repeat=3)))*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
    if prim.IsA(UsdGeom.Sphere):
        r=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get());h=0;axis=2
    elif prim.IsA(UsdGeom.Capsule):
        sh=UsdGeom.Capsule(prim);r=float(sh.GetRadiusAttr().Get());h=float(sh.GetHeightAttr().Get());axis='XYZ'.index(sh.GetAxisAttr().Get())
    else:return None
    z=np.linspace(-1,1,33);theta=np.linspace(0,2*np.pi,65)
    pts=np.array([[np.sqrt(max(0,1-v*v))*np.cos(t),np.sqrt(max(0,1-v*v))*np.sin(t),v] for v in z for t in theta])*r
    pts[:,2]+=np.sign(pts[:,2])*h/2
    pts[:,[axis,2]]=pts[:,[2,axis]]
    return pts

def relative_points(prim,owner):
    pts=points(prim)
    if pts is None:return None
    m=np.array(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(owner).GetInverse())
    return np.c_[pts,np.ones(len(pts))].dot(m)[:,:3]

prims=list(Usd.PrimRange(s.GetPseudoRoot(),Usd.TraverseInstanceProxies()))
for c in prims:
    if not c.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(c).GetCollisionEnabledAttr().Get() is False:continue
    owner=c
    while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
    if not owner:continue
    cp=relative_points(c,owner)
    if cp is None:continue
    visuals=[v for v in prims if str(v.GetPath()).startswith(str(owner.GetPath())+'/visuals/') and v.IsA(UsdGeom.Gprim)]
    vp=[relative_points(v,owner) for v in visuals];vp=[v for v in vp if v is not None and len(v)]
    if not vp:continue
    vp=np.concatenate(vp);cb=np.array([cp.min(0),cp.max(0)]);vb=np.array([vp.min(0),vp.max(0)])
    excess=np.maximum(np.r_[vb[0]-cb[0],cb[1]-vb[1]],0)
    missing=np.maximum(np.r_[cb[0]-vb[0],vb[1]-cb[1]],0)
    exact_matches=[]
    for v in visuals:
        vpts=relative_points(v,owner)
        if vpts is not None and vpts.shape==cp.shape and np.allclose(vpts,cp,atol=1e-8,rtol=0):exact_matches.append(str(v.GetPath()))
    rows.append(dict(collider=str(c.GetPath()),owner=str(owner.GetPath()),type=c.GetTypeName(),
        visible=str(UsdGeom.Imageable(c).ComputeVisibility()),collider_bounds=cb.tolist(),visual_union_bounds=vb.tolist(),
        proved_excess_beyond_visual_aabb_m=float(excess.max()),visual_aabb_outside_this_shape_m=float(missing.max()),
        identical_visible_geometry=exact_matches,mesh_approximation=c.GetAttribute('physics:approximation').Get()))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(scene=str(a.scene),rows=rows,notes=__doc__),indent=2)+'\n')
print(json.dumps(dict(shapes=len(rows),exact_geometry_matches=sum(bool(r['identical_visible_geometry']) for r in rows),max_proved_excess_m=max(r['proved_excess_beyond_visual_aabb_m'] for r in rows))))
