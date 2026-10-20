#!/usr/bin/env python3
"""Measured GEN4 asset/layout checks; never claims a live PhysX overlap PASS."""
from __future__ import annotations
from itertools import product, combinations
from pathlib import Path
import hashlib
import json
import os
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.gen4-local/matplotlib'))
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics
from handson.tasks.eval_gen04_pass_glue_stick import contract as c
from handson.tasks.eval_gen04_pass_glue_stick.semantics import transformed_points, tray_geometry
from handson.sim.isaaclab.gen4_scene import ASSET_DIR, TRAY_ASSET, load_geometry, sample_layout

OUT = Path(__file__).resolve().parent / 'evidence'
XML = ROOT / 'handson/handson/sim/envs/xmls'


def corners(bounds):
    return np.asarray(list(product(*np.asarray(bounds).T)))


def pose(row):
    angle=np.radians(row['yaw_degrees'])/2
    return np.asarray([*row['xy'], row['z'], np.cos(angle), 0, 0, np.sin(angle)])


def usd_inventory(path):
    stage=Usd.Stage.Open(str(path))
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy','guide'], False, True)
    meshes=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable): continue
        mesh=UsdGeom.Mesh(prim) if prim.IsA(UsdGeom.Mesh) else None
        bound=cache.ComputeRelativeBound(prim, stage.GetDefaultPrim()).ComputeAlignedRange()
        meshes.append({'path':str(prim.GetPath()), 'collision':prim.HasAPI(UsdPhysics.CollisionAPI),
            'type':prim.GetTypeName(), 'visibility':str(UsdGeom.Imageable(prim).ComputeVisibility()),
            'vertices':len(mesh.GetPointsAttr().Get()) if mesh else 0,
            'triangles':len(mesh.GetFaceVertexCountsAttr().Get()) if mesh else 0,
            'bounds_m':[list(bound.GetMin()),list(bound.GetMax())]})
    return stage,meshes


def main():
    OUT.mkdir(exist_ok=True)
    g=load_geometry()
    glue_stage, glue_meshes=usd_inventory(ASSET_DIR/'glue_stick.usd')
    tray_stage,tray_meshes=usd_inventory(TRAY_ASSET)
    clutter_stage,clutter_meshes=usd_inventory(ASSET_DIR/'clutter_collisions.usd')
    inventories={'glue':glue_meshes,'tray':tray_meshes,'clutter_collision':clutter_meshes}
    for meshes in inventories.values():
        assert all(m['visibility']=='invisible' for m in meshes if m['collision'])
        assert all(m['visibility']=='inherited' for m in meshes if not m['collision'])
    assert sum(m['collision'] for m in glue_meshes)==len(g['glue']['collision_paths'])
    assert sum(m['collision'] for m in tray_meshes)==17
    assert sum(m['triangles'] for m in glue_meshes if not m['collision'])==23740
    assert np.all(np.linalg.eigvalsh(g['glue']['inertia_tensor_kg_m2'])>0)
    source=ASSET_DIR/'source'/g['source_file']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==g['source_sha256']
    # Actual source texture pixels and converted maps must agree, not just names.
    scene=trimesh.load(source,force='scene',process=False)
    from PIL import Image
    maps=[];uv_checks=[]
    for i,node in enumerate(sorted(scene.graph.nodes_geometry)):
        _,name=scene.graph[node]
        material=scene.geometry[name].visual.material
        source_uv=np.asarray(scene.geometry[name].visual.uv,dtype=np.float32)
        usd_uv=np.asarray(UsdGeom.PrimvarsAPI(glue_stage.GetPrimAtPath(f'/GlueStick/Visual/Part{i}')).GetPrimvar('st').Get())
        assert np.array_equal(source_uv,usd_uv), f'Part{i}: imported lower-left UVs changed'
        uv_checks.append({'part':i,'vertices':len(source_uv),'source_import_uv_exact':True})
        for attr,label in [('baseColorTexture','BaseColor'),('metallicRoughnessTexture','MetallicRoughness'),('normalTexture','Normal'),('occlusionTexture','Occlusion')]:
            original=np.asarray(getattr(material,attr))
            actual=np.asarray(Image.open(ASSET_DIR/'textures'/f'{label}_{i}.png'))
            assert np.array_equal(original,actual)
            maps.append({'name':f'{label}_{i}','size':list(actual.shape),'pixel_exact':True})
    # D03 assets are retained. Inspect both their authored visual bounds and
    # analytical source colliders; report visual support separately.
    r=ET.parse(XML/'d03_handoff_objects.xml').getroot()
    definitions={m.get('name'):m for m in r.findall('asset/mesh')}
    bounds={c.GLUE_BODY:np.asarray(g['glue']['bounds_m']),c.TRAY_BODY:np.asarray(g['tray']['bounds_m'])}
    visual_bounds={}
    for body in r.findall('worldbody/body'):
        name=body.get('name')
        if name not in c.DISTRACTORS:continue
        collision_points=[];visual_points=[]
        for geom in body.findall('geom'):
            if geom.get('mesh') in definitions:
                m=trimesh.load(XML/definitions[geom.get('mesh')].get('file'),force='mesh',process=False)
                if m.bounds is None:raise ValueError(f'{name}: source mesh is an unhydrated LFS pointer')
                rotation=np.asarray(g['distractors'][name].get('source_normalization_rotation',np.eye(3)))
                visual_points.extend(np.asarray(m.vertices)@rotation.T)
            if geom.get('class')!='d03_collision':continue
            kind=geom.get('type');size=np.fromstring(geom.get('size'),sep=' ')
            half=size if kind=='box' else np.repeat(size[0],3) if kind=='sphere' else np.asarray([size[0],size[0],size[1]])
            center=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
            collision_points.extend(corners([center-half,center+half]))
        collision_points=np.asarray(collision_points);visual_points=np.asarray(visual_points)
        bounds[name]=np.asarray(g["distractors"][name]["bounds_m"])
        visual_bounds[name]=np.asarray([visual_points.min(0),visual_points.max(0)])
    first=sample_layout(np.random.RandomState(0),randomize=False,geometry=g)
    rows=[('authored',first)]+[(f'seed_{i}',sample_layout(np.random.RandomState(i),randomize=True,geometry=g)) for i in range(20)]
    assert rows[18][1]==sample_layout(np.random.RandomState(17),randomize=True,geometry=g)
    for dx,dy,yaw in product((-0.03,0.03),(-0.03,0.03),(-15,15)):
        layout=json.loads(json.dumps(first))
        layout[c.GLUE_BODY]['xy']=(np.asarray(c.GLUE_XY)+[dx,dy]).tolist()
        layout[c.GLUE_BODY]['yaw_degrees']=yaw
        rows.append((f'boundary_{dx}_{dy}_{yaw}',layout))
    observations=[]
    for label,layout in rows:
        world={name:transformed_points(corners(bounds[name]),pose(row)) for name,row in layout.items()}
        table_overhang=[];support={};visual_support={}
        for name,points in world.items():
            low,high=points.min(0),points.max(0)
            if np.any(low[:2]<[-.55,-.8]) or np.any(high[:2]>[.25,.8]):table_overhang.append(name)
            support[name]=float(low[2]-c.TABLE_TOP_M)
            if name in visual_bounds:
                visual_support[name]=float(transformed_points(corners(visual_bounds[name]),pose(layout[name]))[:,2].min()-c.TABLE_TOP_M)
        possible_pairs=[]
        for a,b in combinations(world,2):
            overlap=np.minimum(world[a].max(0),world[b].max(0))-np.maximum(world[a].min(0),world[b].min(0))
            if np.all(overlap>1e-6):possible_pairs.append([a,b,overlap.tolist()])
        observations.append({'sample':label,'layout':layout,'table_overhang':table_overhang,'collision_floor_gaps_m':support,'visual_floor_gaps_m':visual_support,'aabb_candidate_pairs':possible_pairs})
    spans={name:np.ptp([row[name]['xy'] for _,row in rows[1:21]],axis=0).tolist() for name in (c.GLUE_BODY,*c.DISTRACTORS)}
    assert all(all(x>0 for x in v) for v in spans.values())
    # Actual occupied vertices: standing, lying in both tray axes and diagonal.
    from scipy.spatial.transform import Rotation
    containment=[]
    tray_pose=pose(first[c.TRAY_BODY])
    for label,euler in [('standing',[0,0,0]),('lying_long',[0,90,0]),('lying_cross',[90,0,0]),('lying_diagonal',[0,90,45])]:
        q=Rotation.from_euler('xyz',euler,degrees=True).as_quat()[[3,0,1,2]]
        target=np.asarray([*c.TRAY_XY,0,*q])
        target[2]=tray_pose[2]+g['tray']['floor_z_m']-transformed_points(g['glue']['occupied_vertices_m'],target)[:,2].min()
        result=tray_geometry(g['glue']['occupied_vertices_m'],target,tray_pose,g['tray']['interior_xy_m'],g['tray']['floor_z_m'])
        assert result['at_floor'] and result['footprint_inside']
        containment.append({'pose':label,'target_pose_wxyz':target.tolist(),**result})
    report={'task_id':'GEN4','method':'CPU source geometry, USD composition, texture equality and conservative layout bounds',
        'source_sha256':g['source_sha256'],'glue_size_m':g['glue']['size_m'],'glue_inertia_kg_m2':g['glue']['diagonal_inertia_kg_m2'],
        'prim_inventory':inventories,'texture_maps':maps,'uv_checks':uv_checks,'same_seed_repeatable':True,'different_seed_xy_span_m':spans,
        'samples':observations,'containment_poses':containment,
        'final_physx_overlap_gate':'pending; includes complete arms/palms/fingers and final composed collision',
        'live_support_reach_camera_stability':'pending'}
    (OUT/'assets_and_layout_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Rectangle
    fig,ax=plt.subplots(figsize=(8,9));ax.add_patch(Rectangle((-.55,-.8),.8,1.6,facecolor='#ece9e1',edgecolor='black'))
    for name,row in first.items():
        points=transformed_points(corners(bounds[name]),pose(row))[:,:2]
        from scipy.spatial import ConvexHull
        xy=points[ConvexHull(points).vertices]
        color='#55aa66' if name==c.GLUE_BODY else '#b78656' if name==c.TRAY_BODY else '#92a9bc'
        ax.add_patch(Polygon(xy,facecolor=color,alpha=.7,edgecolor='black'))
        ax.text(*np.asarray(row['xy']),name.replace('d03_','').replace('gen4_',''),fontsize=7,ha='center')
    for _,row in rows[1:21]:ax.plot(*row[c.GLUE_BODY]['xy'],'k.',markersize=3)
    ax.set(xlim=(-.62,.32),ylim=(-.88,.88),xlabel='World X (m)',ylabel='World Y (m)',title='GEN4 source-derived layout bounds (CPU)\nRobot collision, live support and reach still pending')
    ax.set_aspect('equal');fig.tight_layout();fig.savefig(OUT/'layout_cpu.png',dpi=150);plt.close(fig)
    print(json.dumps({'samples':len(rows),'texture_maps_pixel_exact':len(maps),'table_overhang_samples':[r['sample'] for r in observations if r['table_overhang']],
        'aabb_candidate_samples':[r['sample'] for r in observations if r['aabb_candidate_pairs']],
        'visual_floor_gaps_authored_m':observations[0]['visual_floor_gaps_m'],'containment_poses':len(containment),'live_gates':'pending'},indent=2))

if __name__=='__main__':main()
