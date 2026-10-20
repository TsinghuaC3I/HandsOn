#!/usr/bin/env python3
"""CPU FCL audit of GEN3 source-derived collision, cavity and hinge sweep.

This asset-only check supplements, but cannot replace, final composed-stage
robot/scene PhysX overlap and contact checks before physics and after settling.
"""
from pathlib import Path
import itertools
import hashlib
import json
import sys
import argparse
import numpy as np
import trimesh
import fcl
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, UsdPhysics

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'handson'))
from handson.sim.isaaclab.gen3_hinged_trash_bin_scene import GEN3_ASSET_DIRECTORY
from handson.sim.isaaclab.lw_native_articulation_contract import sample_reset

SOURCE_COLLIDER_SURFACE_LIMIT_M = .001


def geometry(prim):
    mesh=UsdGeom.Mesh(prim)
    return trimesh.Trimesh(vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64),faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)


def convex(mesh):
    return fcl.Convex(mesh.vertices,len(mesh.faces),np.column_stack([np.full(len(mesh.faces),3),mesh.faces]).ravel())


def bvh(mesh):
    model=fcl.BVHModel();model.beginModel(len(mesh.vertices),len(mesh.faces));model.addSubModel(mesh.vertices,mesh.faces);model.endModel();return model


def rotation_x(degrees):
    a=np.radians(degrees);return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])


def intersection(first,second):
    result=fcl.CollisionResult()
    fcl.collide(first,second,fcl.CollisionRequest(num_max_contacts=64,enable_contact=True),result)
    return result


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cooked-json',type=Path)
    args=parser.parse_args()
    cooked=json.loads(args.cooked_json.read_text()) if args.cooked_json else None
    stage=Usd.Stage.Open(str(GEN3_ASSET_DIRECTORY/'runtime/trash_bin.usda'))
    hinge=np.asarray(UsdGeom.Xform(stage.GetPrimAtPath('/TrashBin/trash_bin_lid')).GetLocalTransformation().ExtractTranslation())
    visuals={};colliders=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=geometry(prim);path=str(prim.GetPath());moving='/trash_bin_lid/' in path
        if '/Visuals/' in path:visuals[prim.GetName()]=(mesh,moving)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api=UsdPhysics.FilteredPairsAPI(prim)
            filters=[str(p) for p in api.GetFilteredPairsRel().GetTargets()] if api else []
            meshes=[mesh]
            if cooked is not None:
                live_path=path.replace('/TrashBin','/World/Task/LWNativeArticulation/trash_bin',1)
                meshes=[]
                for piece in cooked['shapes'][live_path]['pieces']:
                    faces=[]
                    for poly in piece['polygons']:
                        indices=piece['indices'][poly['index_base']:poly['index_base']+poly['num_vertices']]
                        faces.extend([[indices[0],indices[i],indices[i+1]] for i in range(1,len(indices)-1)])
                    meshes.append(trimesh.Trimesh(vertices=piece['vertices'],faces=faces,process=False))
                if not meshes:raise ValueError(f'Missing actual cooked collision: {live_path}')
            for mesh in meshes:
                colliders.append({'path':path,'source':prim.GetCustomDataByKey('handson:sourceNode'),'mesh':mesh,'moving':moving,'shape':convex(mesh),'filters':filters})
    base=[c for c in colliders if not c['moving']];lid=[c for c in colliders if c['moving']]
    source_geometry=[]
    # Nearest source face and its outward normal distinguish material-interior
    # samples from outward proxy expansion, including concave cavity walls.
    for source,(visual,moving) in visuals.items():
        visual.update_faces(visual.nondegenerate_faces(height=1e-12))
        welded=visual.copy();welded.merge_vertices()
        labels=trimesh.graph.connected_component_labels(welded.face_adjacency,node_count=len(welded.faces))
        solids=[]
        for label in np.unique(labels):
            part=trimesh.Trimesh(vertices=welded.vertices,faces=welded.faces[labels==label],process=False)
            part.remove_unreferenced_vertices()
            if part.is_watertight:solids.append(part)
        group=[c for c in colliders if c['source']==source]
        max_outside=0.;count=0
        for c in group:
            mesh=c['mesh'];points=np.concatenate([mesh.vertices,mesh.triangles.mean(1),mesh.vertices[mesh.edges_unique].mean(1)])
            closest,distance,face=trimesh.proximity.closest_point(visual,points)
            if not np.all(np.isfinite(closest)) or not np.all(np.isfinite(distance)):
                raise ValueError(f"Non-finite source distance: {source}")
            signed=np.einsum('ij,ij->i',points-closest,visual.face_normals[face])
            outside=signed>1e-8
            # The barrel includes a decorative underside seam inside its
            # closed solid wall. Nearest-face normals alone misclassify
            # material inside that solid as empty space.
            for solid in solids:
                selected=np.flatnonzero(outside)
                outside[selected[solid.contains(points[selected])]]=False
            max_outside=max(max_outside,float(np.max(np.where(outside,distance,0))))
            count+=len(points)
        source_points=np.concatenate([visual.vertices,visual.triangles.mean(1),visual.vertices[visual.edges_unique].mean(1)])
        covered=np.zeros(len(source_points),dtype=bool)
        for c in group:
            planes=ConvexHull(c['mesh'].vertices).equations
            covered|=np.all(source_points@planes[:,:3].T+planes[:,3]<=1e-6,axis=1)
        missing=source_points[~covered]
        max_missing=0.
        if len(missing):
            combined=trimesh.util.concatenate([c['mesh'] for c in group])
            for start in range(0,len(missing),128):
                distance=trimesh.proximity.closest_point(combined,missing[start:start+128])[1]
                if not np.all(np.isfinite(distance)):
                    raise ValueError(f"Non-finite collision coverage distance: {source}")
                max_missing=max(max_missing,float(distance.max()))
        source_geometry.append({'source':source,'collider_count':len(group),'surface_samples':count,
            'max_outward_distance_m':max_outside,'source_surface_samples':len(source_points),
            'max_uncovered_source_surface_distance_m':max_missing,
            'surface_limit_m':SOURCE_COLLIDER_SURFACE_LIMIT_M,
            'surface_fidelity_passed':max(max_outside,max_missing)<=SOURCE_COLLIDER_SURFACE_LIMIT_M})
    swept=[];unapproved=[]
    # Actual source cover vs. actual barrel wall. The downloaded underside
    # seam does not prevent a triangle intersection test.
    barrel=fcl.CollisionObject(bvh(visuals['SourceMesh00'][0]))
    cover_shape=bvh(visuals['SourceMesh01'][0])
    source_cover_wall_intersections=[]
    for angle in range(0,91,5):
        r=rotation_x(-angle);pose=fcl.Transform(r,hinge);pairs=[]
        source_result=intersection(fcl.CollisionObject(cover_shape,pose),barrel)
        if source_result.is_collision:source_cover_wall_intersections.append(angle)
        for l in lid:
            lc=np.array(list(itertools.product(*zip(l['mesh'].bounds[0],l['mesh'].bounds[1]))))@r.T+hinge
            lmin,lmax=lc.min(0),lc.max(0)
            for b in base:
                bmin,bmax=b['mesh'].bounds
                if np.any(lmin>bmax) or np.any(bmin>lmax):continue
                result=intersection(fcl.CollisionObject(l['shape'],pose),fcl.CollisionObject(b['shape']))
                if not result.is_collision:continue
                depth=max((c.penetration_depth for c in result.contacts),default=0.)
                if depth<=1e-7:continue
                intended=(b['source']=='SourceMesh06' and l['source'] in {'SourceMesh01','SourceMesh02'} and b['filters']==['/TrashBin/trash_bin_lid'])
                row={'lid':l['path'],'base':b['path'],'depth_m':float(depth),'intended_source_hinge_assembly':intended}
                pairs.append(row)
                if not intended:unapproved.append({'opening_deg':angle,**row})
        swept.append({'opening_deg':angle,'pairs':pairs,'unapproved_pair_count':sum(not p['intended_source_hinge_assembly'] for p in pairs)})
    # A vertical ray through the open cavity must not hit a collider above its
    # existing bottom. Multiple off-center rays detect accidental outer hulls.
    base_mesh=trimesh.util.concatenate([c['mesh'] for c in base])
    cavity=[]
    for x,y in itertools.product([-.035,0.,.035],repeat=2):
        hits,_,_=base_mesh.ray.intersects_location([[x,y,.5]],[[0,0,-1]],multiple_hits=True)
        highest=float(hits[:,2].max()) if len(hits) else None
        cavity.append({'xy_m':[x,y],'highest_collider_hit_z_m':highest,'open_to_bottom':highest is not None and highest<.035})
    # All required reset samples are enumerated. This proves sampled bounds
    # and table support only; robot reach/clearance remains a live-stage gate.
    samples=[]
    for seed in range(20):
        sample=sample_reset('GEN3',seed=seed,randomize=True)
        assert sample==sample_reset('GEN3',seed=seed,randomize=True)
        samples.append({'kind':'seed',**sample.as_dict()})
    for x,y,yaw in itertools.product([-.4,-.28,-.16],[-.12,0,.12],[-30,0,30]):
        samples.append({'kind':'boundary','root_x_m':x,'root_y_m':y,'root_yaw_deg':yaw})
    base_vertices=np.concatenate([c['mesh'].vertices for c in base])
    support=[]
    for sample in samples:
        a=np.radians(sample['root_yaw_deg']);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        v=base_vertices@r.T+[sample['root_x_m'],sample['root_y_m'],.89025]
        inside=bool(np.all(v[:,:2].min(0)>[-.64,-.5]) and np.all(v[:,:2].max(0)<[.08,.5]))
        support.append({**sample,'minimum_support_gap_m':float(v[:,2].min()-.89),'inside_table_footprint':inside})
    report={'scope':'CPU asset-only source/collision audit; final composed robot/PhysX scene not covered',
            'runtime_usda_sha256':hashlib.sha256((GEN3_ASSET_DIRECTORY/'runtime/trash_bin.usda').read_bytes()).hexdigest(),
            'asset_manifest_sha256':hashlib.sha256((GEN3_ASSET_DIRECTORY/'manifest.json').read_bytes()).hexdigest(),
            'source_to_collider':source_geometry,'hinge_sweep':swept,'unapproved_overlaps':unapproved,
            'source_cover_wall_intersection_angles_deg':source_cover_wall_intersections,
            'cavity_rays':cavity,'reset_samples':support,
            'same_seed_repeatability':True,'random_seed_count':20,'boundary_pose_count':27,
            'live_first_PhysX_and_settled_stage_gate':'pending',
            'robot_arm_palm_finger_overlap_gate':'pending','dynamic_contact_gate':'pending'}
    if cooked is not None:
        report['scope']='CPU geometry audit of actual Isaac Sim 5.1 cooked task colliders captured before first physics; robot scene audited separately'
        report['cooked_representation_sha256']=hashlib.sha256(args.cooked_json.read_bytes()).hexdigest()
        report['physics_steps_at_capture']=cooked['physics_steps']
    name='cooked_asset_geometry_audit.json' if cooked is not None else 'asset_geometry_audit.json'
    out=REPO/'taskdev/tasks/eval_gen02_open_trash_bin/evidence'/name
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'unapproved_overlap_count':len(unapproved),'cover_wall_intersections':source_cover_wall_intersections,'cavity_clear':all(x['open_to_bottom'] for x in cavity),'table_bounds_clear':all(x['inside_table_footprint'] for x in support),'source_to_collider':source_geometry},indent=2),flush=True)
    if (unapproved or source_cover_wall_intersections
            or not all(x['surface_fidelity_passed'] for x in source_geometry)
            or not all(x['open_to_bottom'] for x in cavity)
            or not all(x['inside_table_footprint'] for x in support)):
        raise SystemExit(1)

if __name__=='__main__':run()
