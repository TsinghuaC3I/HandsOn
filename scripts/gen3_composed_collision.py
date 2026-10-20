"""Exact cooked convex/primitive checks using live body poses for GEN3 audits."""
from __future__ import annotations

import itertools
import numpy as np
import fcl
from pxr import UsdGeom


def pose_matrix(pose):
    p=np.asarray(pose,dtype=float).reshape(7)
    w,x,y,z=p[3:]/np.linalg.norm(p[3:])
    m=np.eye(4)
    m[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    m[:3,3]=p[:3]
    return m


class ComposedCollisionAudit:
    def __init__(self, stage, inventory, cooked):
        self.inventory=inventory
        self.shapes=[]
        self.authored_body_world={}
        for path,row in inventory.items():
            prim=stage.GetPrimAtPath(path)
            body=stage.GetPrimAtPath(row['body'])
            body_world=np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).T
            self.authored_body_world[row['body']]=body_world
            world=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            relative=np.linalg.inv(body_world)@world
            scale=np.linalg.norm(relative[:3,:3],axis=0)
            relative[:3,:3]/=scale
            if np.linalg.det(relative[:3,:3])<.999:
                raise ValueError(f'Unsupported reflected collider transform: {path}')
            parts=[]
            if prim.IsA(UsdGeom.Mesh):
                for piece in cooked['shapes'][path]['pieces']:
                    vertices=np.asarray(piece['vertices'])*scale
                    faces=[]
                    for poly in piece['polygons']:
                        start,count=poly['index_base'],poly['num_vertices']
                        faces.extend([count,*piece['indices'][start:start+count]])
                    shape=fcl.Convex(vertices,len(piece['polygons']),faces)
                    parts.append((shape,np.stack([vertices.min(0),vertices.max(0)])))
            elif prim.IsA(UsdGeom.Cube):
                size=scale*UsdGeom.Cube(prim).GetSizeAttr().Get()
                parts.append((fcl.Box(*size),np.stack([-size/2,size/2])))
            elif prim.IsA(UsdGeom.Sphere):
                if np.ptp(scale)>1e-8:raise ValueError(f'Non-uniform sphere: {path}')
                radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get()*scale[0]
                parts.append((fcl.Sphere(radius),np.array([[-radius]*3,[radius]*3])))
            elif prim.IsA(UsdGeom.Capsule):
                cap=UsdGeom.Capsule(prim);axis=cap.GetAxisAttr().Get();i={'X':0,'Y':1,'Z':2}[axis]
                if np.ptp(np.delete(scale,i))>1e-8:raise ValueError(f'Elliptic capsule: {path}')
                radius=cap.GetRadiusAttr().Get()*np.delete(scale,i)[0]
                height=cap.GetHeightAttr().Get()*scale[i]
                relative[:3,:3]=relative[:3,:3]@{
                    'X':np.array([[0,0,1],[0,1,0],[-1,0,0]]),
                    'Y':np.array([[1,0,0],[0,0,1],[0,-1,0]]),'Z':np.eye(3)}[axis]
                parts.append((fcl.Capsule(radius,height),np.array([[-radius,-radius,-height/2-radius],[radius,radius,height/2+radius]])))
            elif row['group']=='ground_support':
                continue  # Ground is checked by the minimum actual shape Z.
            else:
                raise ValueError(f'Uninspected collider type {prim.GetTypeName()}: {path}')
            for shape,bounds in parts:
                corners=np.array(list(itertools.product(*zip(bounds[0],bounds[1]))))
                self.shapes.append((path,row['body'],shape,relative.copy(),corners))

    def run(self, raw, *, before_physics):
        poses={key:value.copy() for key,value in self.authored_body_world.items()}
        if not before_physics:
            for side,robot in raw._robots.items():
                robot.update(0.)
                values=robot.data.body_state_w[0].detach().cpu().numpy()
                by_name={name:pose_matrix(values[i,:7]) for i,name in enumerate(robot.body_names)}
                for body in poses:
                    if f'/{side.title()}Robot/' in body:
                        poses[body]=by_name[body.rsplit('/',1)[-1]]
            for name in (raw._lw_scene_spec.root_body_name,raw._lw_scene_spec.right_body_name):
                pose=raw._physics_body_pose(name)
                if pose is None or not np.all(np.isfinite(pose)):
                    raise RuntimeError(f'Missing live task-body tensor pose: {name}')
                poses[raw._lw_scene_spec.body_path(name)]=pose_matrix(pose)
        objects=[];bounds=[]
        for path,body,shape,relative,corners in self.shapes:
            world=poses[body]@relative
            objects.append(fcl.CollisionObject(shape,fcl.Transform(world[:3,:3],world[:3,3])))
            points=corners@world[:3,:3].T+world[:3,3]
            bounds.append([points.min(0),points.max(0)])
        bounds=np.asarray(bounds)
        broad=np.all(bounds[:,None,0]<=bounds[None,:,1],axis=2)&np.all(bounds[None,:,0]<=bounds[:,None,1],axis=2)
        overlaps=[]
        for i,j in zip(*np.where(np.triu(broad,1))):
            first,second=self.shapes[i],self.shapes[j]
            if first[1]==second[1]:continue
            result=fcl.CollisionResult()
            fcl.collide(objects[i],objects[j],fcl.CollisionRequest(num_max_contacts=64,enable_contact=True),result)
            depth=max((c.penetration_depth for c in result.contacts),default=0.)
            if result.is_collision and depth>1e-7:
                overlaps.append({'paths':sorted([first[0],second[0]]),'depth_m':float(depth),
                    'contact_points_m':[np.asarray(c.pos).tolist() for c in result.contacts]})
        return {'method':'FCL convex/box/capsule/sphere intersection on actual PhysX cooked shapes',
            'pose_source':'authored before first physics' if before_physics else 'live robot body_state_w and task rigid-body get_transforms tensors',
            'minimum_world_shape_z_m':float(bounds[:,0,2].min()),
            'body_world_matrices':{p:m.tolist() for p,m in poses.items()},'overlaps':overlaps}
