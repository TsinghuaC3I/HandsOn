"""Read-only GEN7 collision queries using cooked shapes and GPU body poses.

PhysX's CPU scene-query actor poses can lag direct GPU tensor resets. These
queries preserve the actual cooked convex geometry and source triangle meshes,
but obtain every moving body transform from the same tensors as observations.
They never write a pose, joint, velocity, or drive target to the simulator.
"""
from __future__ import annotations

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=np.float64)


def pose_matrix(pose):
    """PhysX tensor pose layout is XYZ / XYZW."""
    from .bimanual_assembly_env import _quat_wxyz_to_matrix
    p=array(pose);matrix=np.eye(4)
    matrix[:3,:3]=_quat_wxyz_to_matrix(p[[6,3,4,5]])
    matrix[:3,3]=p[:3]
    return matrix


def physical_body_matrices(env):
    matrices={}
    views=[robot.root_physx_view for robot in env._robots.values()]
    views.extend(state[1] for state in env._task_reset_articulation_states)
    for view in views:
        poses=array(view.get_link_transforms())
        for paths,body_poses in zip(view.link_paths,poses):
            for path,pose in zip(paths,body_poses):
                matrices[str(path)]=pose_matrix(pose)
    for state in env._task_reset_body_states:
        view=state[1]
        for path,pose in zip(view.prim_paths,array(view.get_transforms())):
            matrices[str(path)]=pose_matrix(pose)
    return matrices


def polygon_triangles(indices,counts):
    faces=[];offset=0
    for count in counts:
        polygon=indices[offset:offset+count];offset+=count
        faces.extend([polygon[0],polygon[i],polygon[i+1]] for i in range(1,count-1))
    return np.asarray(faces,dtype=np.int32).reshape(-1,3)


def convex_ray_distance(origin,direction,planes,maximum):
    a=planes[:,:3]@origin+planes[:,3]
    b=planes[:,:3]@direction
    parallel=np.abs(b)<1e-12
    if np.any(parallel & (a>1e-8)):
        return float('inf')
    entering=b < -1e-12;leaving=b > 1e-12
    lower=max(0.,float(np.max(-a[entering]/b[entering])) if np.any(entering) else 0.)
    upper=min(float(maximum),float(np.min(-a[leaving]/b[leaving])) if np.any(leaving) else float(maximum))
    return lower if lower<=upper+1e-9 else float('inf')


def sphere_ray_distance(origin, direction, radius, center=None):
    offset=np.asarray(origin)-(np.zeros(3) if center is None else center)
    a=float(direction@direction);b=float(offset@direction)
    c=float(offset@offset-radius*radius)
    if c<=0:return 0.
    discriminant=b*b-a*c
    if discriminant<0 or a<=0:return float('inf')
    distance=(-b-np.sqrt(discriminant))/a
    return float(distance) if distance>=0 else float('inf')


def capsule_ray_distance(origin, direction, radius, height, axis):
    axis={'X':0,'Y':1,'Z':2}[axis];half=height/2.
    radial=[i for i in range(3) if i!=axis]
    closest=np.zeros(3);closest[axis]=np.clip(origin[axis],-half,half)
    if np.sum((origin-closest)**2)<=radius*radius:return 0.
    candidates=[]
    for sign in (-1,1):
        center=np.zeros(3);center[axis]=sign*half
        candidates.append(sphere_ray_distance(origin,direction,radius,center))
    o=origin[radial];d=direction[radial]
    a=float(d@d);b=float(o@d);c=float(o@o-radius*radius)
    discriminant=b*b-a*c
    if a>1e-15 and discriminant>=0:
        for distance in ((-b-np.sqrt(discriminant))/a,(-b+np.sqrt(discriminant))/a):
            if distance>=0 and abs(origin[axis]+distance*direction[axis])<=half+1e-12:
                candidates.append(float(distance))
    return min(candidates)


class PhysicsGeometry:
    def __init__(self,env):
        from pxr import Usd,UsdGeom,UsdPhysics,PhysicsSchemaTools
        from omni.physx import get_physx_cooking_interface
        import omni.usd
        stage=env._stage;stage_id=omni.usd.get_context().get_stage_id()
        cooking=get_physx_cooking_interface();self.shapes=[]
        for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            parent=prim;body=None
            while parent.IsValid() and not parent.IsPseudoRoot():
                if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    body=parent;break
                parent=parent.GetParent()
            if body and UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is False:
                continue
            mesh_world=array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            body_world=array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).T if body else np.eye(4)
            local=np.linalg.inv(body_world)@mesh_world
            common={'path':str(prim.GetPath()),'body':str(body.GetPath()) if body else None,
                    'body_local':local,'static_world':mesh_world}
            if prim.GetTypeName()=='Plane':
                self.shapes.append({**common,'kind':'plane','axis':str(UsdGeom.Plane(prim).GetAxisAttr().Get())})
                continue
            if prim.IsA(UsdGeom.Cube):
                import itertools
                from scipy.spatial import ConvexHull
                half=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2.
                vertices=np.asarray(list(itertools.product([-half,half],repeat=3)))
                planes=np.c_[np.r_[np.eye(3),-np.eye(3)],np.full(6,-half)]
                self.shapes.append({**common,'kind':'convex','vertices':vertices,
                                    'faces':ConvexHull(vertices).simplices,'planes':planes})
                continue
            if prim.IsA(UsdGeom.Sphere):
                self.shapes.append({**common,'kind':'sphere','radius':float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())})
                continue
            if prim.IsA(UsdGeom.Capsule):
                capsule=UsdGeom.Capsule(prim)
                self.shapes.append({**common,'kind':'capsule','radius':float(capsule.GetRadiusAttr().Get()),
                                    'height':float(capsule.GetHeightAttr().Get()),'axis':str(capsule.GetAxisAttr().Get())})
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(f'GEN7 collision query has an unexamined shape: {prim.GetPath()} ({prim.GetTypeName()})')
            mesh=UsdGeom.Mesh(prim)
            approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation=='none':
                vertices=array(mesh.GetPointsAttr().Get())
                faces=polygon_triangles(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),np.asarray(mesh.GetFaceVertexCountsAttr().Get()))
                self.shapes.append({**common,'kind':'triangles','vertices':vertices,'faces':faces,'triangles':vertices[faces]})
                continue
            cooked=[]
            def receive(result,convexes):
                if int(result)!=0:
                    raise RuntimeError(f'GEN7 cannot read cooked collision shape {prim.GetPath()}: {result}')
                for convex in convexes:
                    vertices=np.asarray([tuple(v) for v in convex.vertices],dtype=np.float64)
                    faces=[]
                    for polygon in convex.polygons:
                        indices=convex.indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
                        faces.extend([indices[0],indices[i],indices[i+1]] for i in range(1,len(indices)-1))
                    cooked.append({'vertices':vertices,'faces':np.asarray(faces,dtype=np.int32),
                                   'planes':np.asarray([tuple(p.plane) for p in convex.polygons],dtype=np.float64)})
            cooking.request_convex_collision_representation(stage_id,PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),False,receive)
            if not cooked:
                raise RuntimeError(f'GEN7 cooked collision representation is empty: {prim.GetPath()}')
            self.shapes.extend({**common,'kind':'convex',**part} for part in cooked)
        paths=sorted({shape['body'] for shape in self.shapes if shape['body']})
        simulation=next(iter(env._robots.values()))._physics_sim_view
        self.body_view=simulation.create_rigid_body_view(paths)
        if set(self.body_view.prim_paths)!=set(paths):
            raise RuntimeError('GEN7 GPU collision body view does not cover every enabled collider owner')

    def prepare(self,env):
        bodies={str(path):pose_matrix(pose) for path,pose in zip(self.body_view.prim_paths,array(self.body_view.get_transforms()))}
        prepared=[]
        for shape in self.shapes:
            body=shape['body']
            if body is None:
                world=shape['static_world']
            elif body in bodies:
                world=bodies[body]@shape['body_local']
            else:
                raise RuntimeError(f'GEN7 collision query has no GPU pose for rigid body {body}')
            prepared.append((shape,np.linalg.inv(world)))
        return QuerySnapshot(prepared)


class QuerySnapshot:
    def __init__(self,shapes):
        self.shapes=shapes

    def raycast_closest(self,origin,direction,maximum):
        from .eval_gen06_spray_shoe_scene import ray_triangle_distance
        origin=np.asarray(origin,dtype=np.float64);direction=np.asarray(direction,dtype=np.float64)
        best=float(maximum);hit=None
        for shape,inverse in self.shapes:
            o=inverse[:3,:3]@origin+inverse[:3,3];d=inverse[:3,:3]@direction
            if shape['kind']=='convex':
                distance=convex_ray_distance(o,d,shape['planes'],best)
            elif shape['kind']=='sphere':
                distance=sphere_ray_distance(o,d,shape['radius'])
            elif shape['kind']=='capsule':
                distance=capsule_ray_distance(o,d,shape['radius'],shape['height'],shape['axis'])
            elif shape['kind']=='triangles':
                distance=ray_triangle_distance(o,d,shape['triangles'])
            else:
                axis={'X':0,'Y':1,'Z':2}[shape['axis']]
                distance=-o[axis]/d[axis] if abs(d[axis])>1e-12 else float('inf')
                if distance<0:distance=float('inf')
            if distance<=best:
                best=float(distance);hit=shape
        return ({'hit':True,'distance':best,'collision':hit['path'],'rigidBody':hit['body']} if hit else {'hit':False})
