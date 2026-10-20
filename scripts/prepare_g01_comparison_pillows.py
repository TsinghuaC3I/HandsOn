#!/usr/bin/env python3
"""Author original, conforming FEM pillows and editable Blender source.

Run with Blender --background --python THIS_FILE -- --output-dir DIRECTORY.
The visible triangle boundary is exactly the boundary of the physical tets.
USD uses the Isaac Sim 5.1 deformable beta schema, not Blender baked motion.
Mesh generation requires MeshPy/TetGen; Blender can load the exported NPZ.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REVISION = "sewn-pillow-v2"
PROFILES = {
    "soft": dict(label="Soft filling pillow", youngs_modulus=30000.0,
                 poissons_ratio=0.45, mass=0.55, damping=0.25,
                 color=(0.82, 0.79, 0.71), seam_half_thickness_m=0.009),
    "foam": dict(label="Resilient foam pillow", youngs_modulus=100000.0,
                 poissons_ratio=0.45, mass=0.80, damping=0.18,
                 color=(0.24, 0.41, 0.48), seam_half_thickness_m=0.016),
}


def pillow_geometry(profile: str):
    """Tetrahedralize a closed sewn envelope; never warp interior grid cells."""
    import meshpy.tet as mt
    nx, ny = 65,37
    u, v = np.meshgrid(np.linspace(-1,1,nx),np.linspace(-1,1,ny),indexing='ij')
    x=.4*u*(1-.065*v**8)
    y=.2111*v*(1-.065*u**8)
    fullness=PROFILES[profile]['seam_half_thickness_m']
    crown=((1-u*u)*(1-v*v)).clip(0)**(.62 if profile=='soft' else .48)
    top=fullness+(.09325-fullness)*crown
    bottom=-fullness-(.09325-fullness)*(1-u**4)*(1-v**4)
    amount=.0018 if profile=='soft' else .0007
    gathers=amount*(np.sin(15*np.pi*u+3*v)*v**8*(1-u*u)
                         +.7*np.sin(11*np.pi*v-2*u)*u**8*(1-v*v))
    top+=gathers
    points=np.concatenate([np.stack([x,y,top],-1).reshape(-1,3),
                           np.stack([x,y,bottom],-1).reshape(-1,3)])
    n=nx*ny; faces=[]
    for i,j in itertools.product(range(nx-1),range(ny-1)):
        a=i*ny+j;b=a+ny;c=b+1;d=a+1
        faces.extend([(a,b,c),(a,c,d),(a+n,c+n,b+n),(a+n,d+n,c+n)])
    # Boundary loops are counterclockwise seen from above.
    ring=[i*ny for i in range(nx)]+[(nx-1)*ny+j for j in range(1,ny)]
    ring += [i*ny+ny-1 for i in range(nx-2,-1,-1)]+[j for j in range(ny-2,0,-1)]
    for a,b in zip(ring,ring[1:]+ring[:1]):
        faces.extend([(a,a+n,b+n),(a,b+n,b)])
    surface=np.asarray(faces,dtype=np.int32)
    # TetGen's constrained Delaunay mesh fills this envelope, preserving the
    # source surface and quality-refining the volume, including its thin rim.
    info=mt.MeshInfo();info.set_points(points.tolist());info.set_facets(surface.tolist())
    volume=mt.build(info,options=mt.Options('pq1.7/8a0.000003f'))
    points=np.asarray(volume.points,dtype=np.float64)
    tets=np.asarray(volume.elements,dtype=np.int32)
    det=np.linalg.det(points[tets[:,1:]]-points[tets[:,:1]])
    reverse=det<0
    tets[reverse,1],tets[reverse,2]=tets[reverse,2].copy(),tets[reverse,1].copy()
    face_map={}; incidences=Counter()
    for a,b,c,d in tets:
        for face in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
            key=tuple(sorted(map(int,face)));incidences[key]+=1;face_map[key]=face
    faces=np.asarray([face_map[f] for f,k in incidences.items() if k==1],dtype=np.int32)
    surface_ids,inverse=np.unique(faces,return_inverse=True)
    skin=points[surface_ids];skin_faces=inverse.reshape(-1,3).astype(np.int32)
    edge_counts=Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,-1)))
    assert set(edge_counts.values())=={2} and max(incidences.values())==2
    assert np.min(np.abs(det))/6>1e-11
    assert len(surface_ids)-len(edge_counts)+len(faces)==2
    # Oriented boundary volume must equal the sum of all tetrahedron volumes.
    # A folded/overlapping grid can pass positive-volume tests after swapping
    # its vertex order, but cannot pass this independent envelope check.
    volume_sum=float(np.abs(det).sum()/6)
    envelope_volume=float(np.einsum('ij,ij->i',points[faces[:,0]],np.cross(points[faces[:,1]],points[faces[:,2]])).sum()/6)
    assert np.isclose(volume_sum,envelope_volume,rtol=1e-7)
    lengths=np.linalg.norm(points[faces[:,[0,1,2]]]-points[faces[:,[1,2,0]]],axis=2)
    report=dict(profile=profile,revision=REVISION,mesher='TetGen constrained Delaunay q1.7/8',
        node_count=len(points),tet_count=len(tets),surface_vertex_count=len(skin),surface_face_count=len(faces),
        boundary_nonmanifold_edges=0,minimum_tet_volume_m3=float(np.abs(det).min()/6),
        volume_m3=volume_sum,boundary_volume_m3=envelope_volume,
        boundary_edge_p95_m=float(np.quantile(lengths,.95)),
        bounds_min=points.min(0).tolist(),bounds_max=points.max(0).tolist(),
        visual_collision_max_error_m=0.,parameters=PROFILES[profile])
    return points,tets,faces,skin,skin_faces,report


def detail_skin(points, faces):
    """Four render triangles per physical boundary triangle, exact alignment."""
    vertices=points.tolist(); mids={};result=[]
    for a,b,c in faces:
        ids=[]
        for i,j in ((a,b),(b,c),(c,a)):
            key=tuple(sorted((int(i),int(j))))
            if key not in mids:
                mids[key]=len(vertices);vertices.append(((points[i]+points[j])*.5).tolist())
            ids.append(mids[key])
        ab,bc,ca=ids;result.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
    return np.asarray(vertices),np.asarray(result,dtype=np.int32)


def vertex_normals(points,faces):
    n=np.zeros_like(points)
    fn=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    for col in range(3):np.add.at(n,faces[:,col],fn)
    return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)


def seam_geometry():
    # Piping is a 1.6 mm textile cord sewn to the physical envelope's rim.
    u=np.concatenate([np.linspace(-1,1,193)[:-1],np.ones(96),np.linspace(1,-1,193)[:-1],-np.ones(96)])
    v=np.concatenate([-np.ones(192),np.linspace(-1,1,97)[:-1],np.ones(192),np.linspace(1,-1,97)[:-1]])
    center=np.stack([.4*u*(1-.065*v**8),.2111*v*(1-.065*u**8),np.zeros_like(u)],-1)
    tangent=np.roll(center,-1,0)-np.roll(center,1,0)
    normal=np.stack([tangent[:,1],-tangent[:,0],np.zeros_like(u)],-1)
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    a=np.arange(8)*np.pi/4
    pts=(center[:,None,:]+.0008*(normal[:,None,:]*np.cos(a)[None,:,None]+np.array([0,0,1])[None,None,:]*np.sin(a)[None,:,None])).reshape(-1,3)
    faces=[]
    for i in range(len(center)):
        for j in range(8):
            p=i*8+j;q=((i+1)%len(center))*8+j;r=((i+1)%len(center))*8+(j+1)%8;s=i*8+(j+1)%8
            faces.extend([(p,q,r),(p,r,s)])
    return pts,np.asarray(faces,dtype=np.int32)

def _array(values):
    a = np.asarray(values)
    if a.ndim == 1:
        return "[" + ", ".join(str(int(x)) for x in a) + "]"
    return "[" + ", ".join("(" + ", ".join(
        str(int(x)) if np.issubdtype(a.dtype, np.integer) else format(float(x), ".9g")
        for x in row) + ")" for row in a) + "]"


def write_usd(path, geometry, profile, simulation_geometry=None):
    points,tets,faces,skin,skin_faces,report = geometry
    collision_points,collision_tets,collision_faces=points,tets,faces
    if simulation_geometry is not None:
        points,tets,faces=simulation_geometry[:3]
        report['collision_node_count']=len(collision_points)
        report['collision_tet_count']=len(collision_tets)
        report['node_count']=len(points);report['tet_count']=len(tets)
        report['minimum_tet_volume_m3']=simulation_geometry[-1]['minimum_tet_volume_m3']
    skin,skin_faces = detail_skin(skin,skin_faces)
    cord,cord_faces = seam_geometry()
    report['render_face_count'] = len(skin_faces)+len(cord_faces)
    report['render_vertex_count'] = len(skin)+len(cord)
    report['piping_radius_m'] = .0008
    p = PROFILES[profile]
    report['parameters']=dict(p)
    report['simulation_mesher']=simulation_geometry[-1]['mesher']
    report['simulation_volume_m3']=simulation_geometry[-1]['volume_m3']
    bounds = _array(np.asarray([points.min(0), points.max(0)]))
    # No auto-remesh API: the authored tets are the solver and collision truth.
    text = f'''#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "root" {{
    def Xform "Pillow024" (
        prepend apiSchemas = ["OmniPhysicsDeformableBodyAPI", "PhysxBaseDeformableBodyAPI", "MaterialBindingAPI"]
    ) {{
        float omniphysics:mass = {p['mass']}
        bool physxDeformableBody:enableSpeculativeCCD = true
        bool physxDeformableBody:selfCollision = true
        float physxDeformableBody:selfCollisionFilterDistance = 0.012
        uint physxDeformableBody:solverPositionIterationCount = 128
        float physxDeformableBody:linearDamping = 0.3
        float physxDeformableBody:maxDepenetrationVelocity = 1.0
        rel material:binding:physics = </root/Physics/PhysicsMaterial>
        def TetMesh "simulation_mesh" (
            prepend apiSchemas = ["OmniPhysicsVolumeDeformableSimAPI", "OmniPhysicsDeformablePoseAPI:default", "MaterialBindingAPI"]
        ) {{
            point3f[] points = {_array(points)}
            point3f[] omniphysics:restShapePoints = {_array(points)}
            int4[] tetVertexIndices = {_array(tets)}
            int4[] omniphysics:restTetVtxIndices = {_array(tets)}
            int3[] surfaceFaceVertexIndices = {_array(faces)}
            vector3f[] velocities = {_array(np.zeros_like(points))}
            token orientation = "rightHanded"
            float3[] extent = {bounds}
            token visibility = "invisible"
            token[] deformablePose:default:omniphysics:purposes = ["bindPose"]
            point3f[] deformablePose:default:omniphysics:points = {_array(points)}
            rel material:binding:physics = </root/Physics/PhysicsMaterial>
        }}
        def TetMesh "collision_mesh" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "OmniPhysicsDeformablePoseAPI:default"]
        ) {{
            point3f[] points = {_array(collision_points)}
            int4[] tetVertexIndices = {_array(collision_tets)}
            int3[] surfaceFaceVertexIndices = {_array(collision_faces)}
            token orientation = "rightHanded"
            token visibility = "invisible"
            bool physics:collisionEnabled = true
            float physxCollision:contactOffset = 0.006
            float physxCollision:restOffset = 0.001
            token[] deformablePose:default:omniphysics:purposes = ["bindPose"]
            point3f[] deformablePose:default:omniphysics:points = {_array(collision_points)}
        }}
        def Xform "Visuals" {{
            def Mesh "Pillow024" (
                prepend apiSchemas = ["OmniPhysicsDeformablePoseAPI:default", "MaterialBindingAPI"]
            ) {{
                point3f[] points = {_array(skin)}
                int[] faceVertexCounts = {_array(np.full(len(skin_faces),3))}
                int[] faceVertexIndices = {_array(skin_faces.ravel())}
                normal3f[] normals = {_array(vertex_normals(skin,skin_faces))} (interpolation = "vertex")
                texCoord2f[] primvars:st = {_array(skin[:,:2]/.02)} (interpolation = "vertex")
                token subdivisionScheme = "none"
                bool doubleSided = true
                float3[] extent = {bounds}
                token[] deformablePose:default:omniphysics:purposes = ["bindPose"]
                point3f[] deformablePose:default:omniphysics:points = {_array(skin)}
                rel material:binding = </root/Looks/Textile>
            }}
            def Mesh "SewnPiping" (
                prepend apiSchemas = ["OmniPhysicsDeformablePoseAPI:default", "MaterialBindingAPI"]
            ) {{
                point3f[] points = {_array(cord)}
                int[] faceVertexCounts = {_array(np.full(len(cord_faces),3))}
                int[] faceVertexIndices = {_array(cord_faces.ravel())}
                normal3f[] normals = {_array(vertex_normals(cord,cord_faces))} (interpolation = "vertex")
                texCoord2f[] primvars:st = {_array(cord[:,:2]/.02)} (interpolation = "vertex")
                token subdivisionScheme = "none"
                token[] deformablePose:default:omniphysics:purposes = ["bindPose"]
                point3f[] deformablePose:default:omniphysics:points = {_array(cord)}
                rel material:binding = </root/Looks/Textile>
            }}
        }}
    }}
    def Scope "Physics" {{
        def Material "PhysicsMaterial" (
            prepend apiSchemas = ["OmniPhysicsBaseMaterialAPI", "OmniPhysicsDeformableMaterialAPI", "PhysxDeformableMaterialAPI"]
        ) {{
            float omniphysics:youngsModulus = {p['youngs_modulus']}
            float omniphysics:poissonsRatio = {p['poissons_ratio']}
            float omniphysics:density = {p['mass']/report['volume_m3']}
            float omniphysics:staticFriction = 0.75
            float omniphysics:dynamicFriction = 0.65
            float physxDeformableMaterial:elasticityDamping = {p['damping']}
        }}
    }}
    def Scope "Looks" {{
        def Material "Textile" {{
            token outputs:surface.connect = </root/Looks/Textile/Shader.outputs:surface>
            def Shader "Shader" {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </root/Looks/Textile/Cotton.outputs:rgb>
                normal3f inputs:normal.connect = </root/Looks/Textile/Weave.outputs:rgb>
                float inputs:roughness = 0.88
                float inputs:metallic = 0
                float inputs:specular = 0.18
                token outputs:surface
            }}
            def Shader "UV" {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "Cotton" {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @cotton_albedo.png@
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float2 inputs:st.connect = </root/Looks/Textile/UV.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "Weave" {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @cotton_normal.png@
                token inputs:sourceColorSpace = "raw"
                float4 inputs:scale = (2,2,2,1)
                float4 inputs:bias = (-1,-1,-1,0)
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float2 inputs:st.connect = </root/Looks/Textile/UV.outputs:result>
                float3 outputs:rgb
            }}
        }}
    }}
}}
'''
    path.write_text(text)
    report['usd_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def make_blender(path, geometry, profile):
    import bpy
    _,_,_,points,faces,_ = geometry
    points,faces=detail_skin(points,faces)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    mesh=bpy.data.meshes.new('ConformingPillowBoundary')
    mesh.from_pydata(points.tolist(), [], faces.tolist())
    mesh.update()
    obj=bpy.data.objects.new(PROFILES[profile]['label'], mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth=True
    mat=bpy.data.materials.new('Woven cotton')
    mat.diffuse_color=(*PROFILES[profile]['color'],1)
    mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value=mat.diffuse_color
    bsdf.inputs['Roughness'].default_value=.88
    noise=mat.node_tree.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value=380
    bump=mat.node_tree.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value=.13
    bump.inputs['Distance'].default_value=.0004
    mat.node_tree.links.new(noise.outputs['Fac'],bump.inputs['Height'])
    mat.node_tree.links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
    mesh.materials.append(mat)
    uv=mesh.uv_layers.new(name='st')
    for loop in mesh.loops: uv.data[loop.index].uv=points[loop.vertex_index,:2]/.02
    cord,cord_faces=seam_geometry()
    cord_mesh=bpy.data.meshes.new('SewnPiping');cord_mesh.from_pydata(cord.tolist(),[],cord_faces.tolist())
    piping=bpy.data.objects.new('Sewn piping - 1.6 mm',cord_mesh);bpy.context.collection.objects.link(piping)
    cord_mesh.materials.append(mat)
    for polygon in cord_mesh.polygons:polygon.use_smooth=True
    obj['physics_runtime']='Isaac Sim 5.1 volume FEM, see companion USD'
    for key,value in PROFILES[profile].items(): obj[key]=value
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'
    scene.unit_settings.scale_length=1
    bpy.context.view_layer.objects.active=obj
    obj.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    # Original seamless woven cotton maps, also used by the live USD material.
    yy,xx=np.mgrid[:512,:512]/512
    yarn=np.sin(2*np.pi*32*xx)*np.sin(2*np.pi*32*yy)
    noise=np.random.default_rng(47).normal(0,.008,(512,512))
    rgb=np.asarray(PROFILES[profile]['color'])[None,None,:]**(1/2.2)
    color=np.clip(rgb*(.96+.035*yarn+noise)[:,:,None],0,1)
    nx=.15*np.sin(2*np.pi*32*xx);ny=.15*np.sin(2*np.pi*32*yy)
    normal=np.stack([nx,ny,np.ones_like(nx)],-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    for name,data in [('cotton_albedo',color),('cotton_normal',normal*.5+.5)]:
        img=bpy.data.images.new(name,width=512,height=512)
        img.colorspace_settings.name='Non-Color'
        img.pixels.foreach_set(np.concatenate([data,np.ones((512,512,1))],-1).astype(np.float32).ravel())
        img.filepath_raw=str(path.parent/(name+'.png'));img.file_format='PNG';img.save()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--geometry-dir',type=Path)
    parser.add_argument('--geometry-only',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    if not args.geometry_only and not args.geometry_dir:
        parser.error('Build geometry with --geometry-only, run voxelize_g01_comparison_pillows.py in Isaac Sim 5.1, then pass --geometry-dir to Blender.')
    for profile in PROFILES:
        directory=args.output_dir/profile
        directory.mkdir(parents=True,exist_ok=True)
        if args.geometry_dir:
            data=np.load(args.geometry_dir/(profile+'-geometry.npz'))
            report=json.loads((args.geometry_dir/(profile+'-report.json')).read_text())
            geometry=(*[data[key] for key in ('points','tets','faces','skin','skin_faces')],report)
        else:
            geometry=pillow_geometry(profile)
        if args.geometry_only:
            np.savez(args.output_dir/(profile+'-geometry.npz'),**dict(zip(('points','tets','faces','skin','skin_faces'),geometry[:-1])))
            (args.output_dir/(profile+'-report.json')).write_text(json.dumps(geometry[-1],indent=2))
            continue
        simulation_geometry=None
        if args.geometry_dir:
            data=np.load(args.geometry_dir/(profile+'-simulation.npz'))
            report=json.loads((args.geometry_dir/(profile+'-simulation-report.json')).read_text())
            simulation_geometry=(*[data[key] for key in ('points','tets','faces','skin','skin_faces')],report)
        write_usd(directory/'Pillow.usda',geometry,profile,simulation_geometry)
        np.savez_compressed(directory/'geometry.npz',**dict(zip(('points','tets','faces','skin','skin_faces'),geometry[:-1])),
            simulation_points=simulation_geometry[0],simulation_tets=simulation_geometry[1],simulation_faces=simulation_geometry[2])
        make_blender(directory/'Pillow.blend',geometry,profile)
        (directory/'asset_manifest.json').write_text(json.dumps(geometry[-1],indent=2)+'\n')
        print(json.dumps(geometry[-1]),flush=True)


if __name__ == '__main__':
    main()
