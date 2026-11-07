"""Ten connected visual knit covers following the native articulated fingers."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
import wuji2_cloth_bridges_v12 as previous

sha=previous.sha
smooth_normals=previous.smooth_normals

def finger_points(finger,poses):
    pieces=[]
    for piece in finger['pieces']:
        if piece['kind']=='joint':
            joint=piece['joint']
            points=previous.deform(joint,poses[joint['parent']],poses[joint['child']])
            points=points[:piece['ring_count']*finger['segments']]
        else:
            points=previous.transform_points(piece['points'],poses[piece['body']])
        pieces.append(points)
    pieces.append(previous.transform_points([finger['cap_point']],poses[finger['cap_body']]))
    return np.concatenate(pieces)

class JointCloth(previous.JointCloth):
    def __init__(self,stage,profile_path,side=None):
        from pxr import Gf,Sdf,UsdGeom,UsdShade
        super().__init__(stage,profile_path,side=side)
        for mesh in self.meshes:stage.RemovePrim(mesh.GetPath())
        self.fingers=[f for f in self.profile['fingers'] if side is None or f['side']==side]
        self.meshes=[]
        mat=UsdShade.Material.Get(stage,self.root+'/Knit')
        for finger in self.fingers:
            mesh=UsdGeom.Mesh.Define(stage,self.root+'/'+finger['name'])
            faces=np.asarray(finger['faces'])
            mesh.CreateFaceVertexCountsAttr([3]*len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*x) for x in finger['uv']])
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            self.meshes.append(mesh)

    def update(self,poses):
        from pxr import Gf
        for mesh,finger in zip(self.meshes,self.fingers):
            points=finger_points(finger,poses)
            normals=smooth_normals(points,np.asarray(finger['faces']))
            mesh.GetPointsAttr().Set([Gf.Vec3f(*x) for x in points])
            mesh.GetNormalsAttr().Set([Gf.Vec3f(*x) for x in normals])
            mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
        self.updates+=1

def install(env,profile_path,report_path=None):
    from pxr import Usd
    cloth=JointCloth(env._stage,profile_path)
    indexes={side:{str(name):i for i,name in enumerate(robot.body_names)} for side,robot in env._robots.items()}
    for joint in cloth.profile['joints']:
        for name in (joint['parent'],joint['child']):assert name in indexes[joint['side']]
    def update():
        poses={}
        for side,robot in env._robots.items():
            source=robot.data.body_pose_w[0].detach().cpu().numpy()
            for name,index in indexes[side].items():
                if '_finger' not in name:continue
                value=source[index];quat=value[3:7] if env._isaaclab_uses_xyzw else value[[4,5,6,3]]
                matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(quat).as_matrix();matrix[:3,3]=value[:3]
                poses[name]=matrix
        cloth.update(poses)
    native_render=env.render
    def render(*args,**kwargs):
        update()
        return native_render(*args,**kwargs)
    update();env.render=render;env._wuji2_joint_cloth_candidate=cloth
    assert not any('Physics' in s or 'Physx' in s for p in Usd.PrimRange(env._stage.GetPrimAtPath(cloth.root)) for s in p.GetAppliedSchemas())
    if report_path:Path(report_path).write_text(json.dumps({'profile_sha256':sha(profile_path),'module_sha256':sha(__file__),'sleeve_count':len(cloth.meshes),'articulated_joint_count':30,'continuous_fingers':10,'first_update_passed':True,'physics_schemas':[],'material':cloth.material_report,'candidate':True},indent=2)+'\n')
    return cloth
