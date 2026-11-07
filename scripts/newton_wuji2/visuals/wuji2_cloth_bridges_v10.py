"""Candidate knitted joint sleeves, deformed by the recorded Wuji2 rigid poses.

This is display geometry, not a cloth simulation. It adds no physics schemas,
changes no robot values, and uses the articulated body's data rather than stale
USD transforms when Fabric is active. Keep disabled until rendered validation.
"""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_transform(body):
    matrix = np.eye(4)
    matrix[:3, 3] = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
    q = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return matrix


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def outward_radii(equations, origin, directions):
    """Farthest positive intersection of each ray with a convex shell."""
    numerator = -(equations[:, :3] @ origin + equations[:, 3])
    denominator = directions @ equations[:, :3].T
    safe = np.where(np.abs(denominator) > 1e-10, denominator, 1)
    ratios = numerator[None, :] / safe
    upper = np.where(denominator > 1e-10, ratios, np.inf).min(axis=1)
    lower = np.where(denominator < -1e-10, ratios, -np.inf).max(axis=1)
    parallel_miss = ((np.abs(denominator) <= 1e-10) & (numerator[None, :] < 0)).any(axis=1)
    valid = (upper >= np.maximum(lower, 0)) & ~parallel_miss & np.isfinite(upper)
    return np.where(valid, upper, np.nan)


def shell_points(asset):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(asset))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/Visual/SourceFrame/FingerGlove'))
    if not mesh:
        raise RuntimeError(f'Expected FingerGlove in {asset}')
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())).T
    return transform_points(np.asarray(mesh.GetPointsAttr().Get()), matrix)


def make_profile(source, bundle, output, segments=64, rings=25, half_length=.011):
    source, bundle, output = Path(source), Path(bundle), Path(output)
    manifest = json.loads((bundle/'manifest.json').read_text())
    by_name = {x['link_name']: x for x in manifest['links']}
    profile = dict(schema='handson.wuji2_joint_cloth.v10_candidate', visual_only=True,
        bundle=str(bundle), bundle_manifest_sha256=sha(bundle/'manifest.json'),
        source_hashes={}, sleeve_count=30, segments=segments, rings=rings,
        half_length_m=half_length, candidate_only=True, joints=[],
        limitations=['estimated knitted deformation, not measured or simulated cloth',
            'open sleeve ends deliberately embedded in existing glove shells',
            'not yet GPU-rendered; bent-pose self-intersection needs visual review'])
    u = np.linspace(0, 1, rings)
    angles = np.arange(segments) * (2*math.pi/segments)
    radial = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    faces = []
    for i in range(rings-1):
        for j in range(segments):
            a, b = i*segments+j, i*segments+(j+1)%segments
            faces.extend([[a, b, b+segments], [a, b+segments, a+segments]])
    for side in ('left', 'right'):
        path = source/f'rokae_wuji2_{side}.xml'
        profile['source_hashes'][str(path)] = sha(path)
        tree = ET.parse(path)
        bodies = {x.get('name'): x for x in tree.findall('.//body')}
        for finger in range(1, 6):
            for link in (2, 3, 4):
                child = f'{side}_finger{finger}_link{link}'
                parent = f'{side}_finger{finger}_link{link-1}'
                body = bodies[child]
                bind = body_transform(body)
                pivot = bind[:3, 3]
                parent_points = shell_points(bundle/by_name[parent]['asset'])
                child_points = transform_points(shell_points(bundle/by_name[child]['asset']), bind)
                equations = [ConvexHull(p).equations for p in (parent_points, child_points)]
                rotations = Rotation.from_rotvec(u[:, None]*Rotation.from_matrix(bind[:3,:3]).as_rotvec()).as_matrix()
                directions = np.einsum('rij,sj->rsi', rotations, radial)
                centers = pivot + np.einsum('rij,j->ri', rotations, [0., 0., -1.]) * ((2*u-1)*half_length)[:,None]
                radius = []
                for center, ds in zip(centers, directions):
                    candidates = np.stack([outward_radii(e, center, ds) for e in equations])
                    if np.isnan(candidates).all(axis=0).any():
                        raise RuntimeError(f'Joint {child} has an unsupported ray')
                    radius.append(np.nanmax(candidates, axis=0))
                radius = np.asarray(radius)
                if not (.002 < radius.min() and radius.max() < .024):
                    raise RuntimeError(f'Unexpected finger radius {child}: {radius.min(), radius.max()}')
                # A small envelope over the native covered joint reduces the
                # sharp seam; ends lie beneath the already-visible knit shell.
                # Round each longitudinal transition over roughly3mm rather
                # than inheriting the sharp unions of two rigid link hulls.
                # Keep its tiny fabric allowance bounded to1.1mm; the original
                # hardware and existing glove remain untouched below it.
                smoothed = radius.copy()
                for _ in range(4):
                    smoothed = np.maximum(radius, gaussian_filter1d(smoothed,2.,axis=0,mode='nearest'))
                    smoothed = np.maximum(radius, gaussian_filter1d(smoothed,.7,axis=1,mode='wrap'))
                envelope = np.sin(np.pi*u)**2
                offset = -.00012 + .00038*envelope
                # No perfectly repeated accordion ridges in the rest pose.
                seed=sum(map(ord,child))
                phase=(seed%19)*.173
                fold=(.000035*np.sin(5.3*np.pi*u+phase)+.00002*np.sin(8.7*np.pi*u-phase))*envelope
                allowance=np.clip(smoothed-radius,0,.0008)*envelope[:,None]
                radius_final = radius + allowance + (offset+fold)[:,None]
                points = centers[:,None,:] + directions*radius_final[:,:,None]
                points = points.reshape(-1,3)
                local_faces = np.asarray(faces, dtype=np.int32)
                # Orient the tube outward in bind pose; fixed indices persist
                # under deformation, while smooth normals are refreshed.
                tri = points[local_faces]
                norm = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                ring_indices = local_faces[:,0]//segments
                outward = tri.mean(axis=1)-centers[ring_indices]
                if np.median(np.einsum('ij,ij->i',norm,outward)) < 0:
                    local_faces = local_faces[:,::-1]
                arc = radius.mean()*angles
                st = np.stack(np.meshgrid(arc/.006, (centers[:,2]-pivot[2])/.006),axis=-1).reshape(-1,2)
                joint = body.find('joint')
                profile['joints'].append(dict(side=side,parent=parent,child=child,
                    bind=bind.tolist(),pivot=pivot.tolist(),points=points.tolist(),
                    weights=np.repeat(u*u*(3-2*u),segments).tolist(),
                    u=np.repeat(u,segments).tolist(),directions=directions.reshape(-1,3).tolist(),
                    angles=np.tile(angles,rings).tolist(),wrinkle_phase=phase,
                    faces=local_faces.tolist(),uv=st.tolist(),
                    axis=np.fromstring(joint.get('axis'),sep=' ').tolist(),
                    range=np.fromstring(joint.get('range'),sep=' ').tolist(),
                    radius_range_m=[float(radius_final.min()),float(radius_final.max())],
                    max_envelope_above_source_shell_m=float((radius_final-radius).max()),
                    old_span_m=.014,new_span_m=2*half_length,
                    source_axial_radius_curvature=float(np.abs(np.diff(radius,2,axis=0)).mean()),
                    candidate_axial_radius_curvature=float(np.abs(np.diff(radius_final,2,axis=0)).mean())))
    output.write_text(json.dumps(profile,separators=(',',':'))+'\n')
    return profile


def deform(joint, parent_world, child_world):
    """Quaternion bend interpolation; end rings follow exact adjacent links."""
    bind = np.asarray(joint['bind'])
    pivot = np.asarray(joint['pivot'])
    points = np.asarray(joint['points'])
    relative_rotation = parent_world[:3,:3].T @ child_world[:3,:3] @ bind[:3,:3].T
    delta = Rotation.from_matrix(relative_rotation).as_rotvec()
    weights = np.asarray(joint['weights'])
    rotated = Rotation.from_rotvec(weights[:,None]*delta).apply(points-pivot)
    if 'directions' in joint:
        # Local compression selects the inside of a bend. Only this region
        # receives subtle, phase-varied folds; the stretched dorsal side stays
        # smooth. These are deterministic display wrinkles, not simulated cloth.
        directions=np.asarray(joint['directions'])
        bend=np.linalg.norm(delta)
        if bend>1e-9:
            inward=np.cross(delta/bend,np.array([0.,0.,-1.]))
            projection=np.maximum(directions@inward,0.)**1.5
            u=np.asarray(joint['u']);a=np.asarray(joint['angles'])
            e=np.sin(np.pi*u)**2
            phase=joint['wrinkle_phase']+.45*np.sin(2*a)+.18*np.cos(5*a)
            wave=.65*np.sin(7.2*np.pi*u+phase)+.35*np.sin(12.3*np.pi*u-phase)
            magnitude=.00020*min(bend/.9,1.)*projection*e*wave
            rd=Rotation.from_rotvec(weights[:,None]*delta).apply(directions)
            rotated+=rd*magnitude[:,None]
    # The small pivot discrepancy check catches wrong body frames / COM poses.
    expected_pivot = transform_points(pivot[None,:], parent_world)[0]
    if np.linalg.norm(expected_pivot-child_world[:3,3]) > .00015:
        raise RuntimeError(f'Joint pose mismatch for {joint["child"]}')
    return transform_points(rotated+pivot, parent_world)


def smooth_normals(points, faces):
    tri = points[faces]
    normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    result = np.zeros_like(points)
    for i in range(3):
        np.add.at(result, faces[:,i], normals)
    length = np.linalg.norm(result,axis=1)
    if (length<1e-12).any():
        raise RuntimeError('Degenerate joint cloth normal')
    return result/length[:,None]


class JointCloth:
    def __init__(self, stage, profile_path, side=None):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        self.stage = stage
        self.profile = json.loads(Path(profile_path).read_text())
        if side is not None:
            self.profile['joints'] = [j for j in self.profile['joints'] if j['side'] == side]
        bundle = Path(self.profile['bundle'])
        if sha(bundle/'manifest.json') != self.profile['bundle_manifest_sha256']:
            raise RuntimeError('Cloth profile does not match glove bundle')
        self.root = '/World/Wuji2JointClothCandidate'
        if stage.GetPrimAtPath(self.root):
            raise RuntimeError('Cloth candidate is already installed')
        UsdGeom.Xform.Define(stage, self.root)
        mat = UsdShade.Material.Define(stage,self.root+'/Knit')
        shader = UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.88)
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
        reader = UsdShade.Shader.Define(stage,str(mat.GetPath())+'/ST')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        for kind,filename,dest,space in [('Color','wuji2_knit_detail_color.png','diffuseColor','sRGB'),('Normal','wuji2_knit_detail_normal.png','normal','raw')]:
            texture=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/'+kind)
            texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(bundle/'textures'/filename)))
            texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(space)
            texture.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
            texture.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
            texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
            if kind=='Normal':
                texture.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
                texture.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
            shader.CreateInput(dest,Sdf.ValueTypeNames.Normal3f if kind=='Normal' else Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        self.meshes=[]
        for joint in self.profile['joints']:
            mesh=UsdGeom.Mesh.Define(stage,self.root+'/'+joint['child'])
            faces=np.asarray(joint['faces'])
            mesh.CreateFaceVertexCountsAttr([3]*len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*x) for x in joint['uv']])
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            self.meshes.append(mesh)
        self.updates=0

    def update(self, poses):
        from pxr import Gf
        for mesh,joint in zip(self.meshes,self.profile['joints']):
            points=deform(joint,poses[joint['parent']],poses[joint['child']])
            normals=smooth_normals(points,np.asarray(joint['faces']))
            mesh.GetPointsAttr().Set([Gf.Vec3f(*x) for x in points])
            mesh.GetNormalsAttr().Set([Gf.Vec3f(*x) for x in normals])
            mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(axis=0)),Gf.Vec3f(*points.max(axis=0))])
        self.updates+=1


def install(env, profile_path, report_path=None):
    """Install after native Wuji2 creation; wrap only this env's render method."""
    from pxr import Usd
    cloth=JointCloth(env._stage,profile_path)
    indexes={}
    for side,robot in env._robots.items():
        indexes[side]={str(n):i for i,n in enumerate(robot.body_names)}
    for joint in cloth.profile['joints']:
        for name in (joint['parent'],joint['child']):
            if name not in indexes[joint['side']]:
                raise RuntimeError(f'Native cloth body is missing: {name}')
    def update():
        poses={}
        for side,robot in env._robots.items():
            source=robot.data.body_pose_w[0].detach().cpu().numpy()
            for name,index in indexes[side].items():
                if '_finger' not in name: continue
                p=source[index]
                q=p[3:7] if env._isaaclab_uses_xyzw else p[[4,5,6,3]]
                matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(q).as_matrix();matrix[:3,3]=p[:3]
                poses[name]=matrix
        cloth.update(poses)
    native_render=env.render
    def render(*args,**kwargs):
        update()
        return native_render(*args,**kwargs)
    update()
    env.render=render
    env._wuji2_joint_cloth_candidate=cloth
    if report_path:
        Path(report_path).write_text(json.dumps(dict(profile_sha256=sha(profile_path),
            sleeve_count=len(cloth.meshes),candidate=True,first_update_passed=True,
            physics_schemas=[]),indent=2)+'\n')
    assert not any('Physics' in schema or 'Physx' in schema
        for prim in Usd.PrimRange(env._stage.GetPrimAtPath(cloth.root))
        for schema in prim.GetAppliedSchemas())
    return cloth


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    p=make_profile(args.source,args.bundle,args.output)
    print('JOINT_CLOTH_PROFILE',len(p['joints']),max(j['max_envelope_above_source_shell_m'] for j in p['joints']))
