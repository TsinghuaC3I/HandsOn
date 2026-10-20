"""Audit exported cooked PhysX shapes at GPU-readback poses with FCL.

All reported distances are geometric surface distances, without contact offsets.
The report retains every cross-body overlap before intended-pair review.
"""
from pathlib import Path
import argparse
import itertools
import json

import fcl
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def object_for(row, arrays):
    matrix=np.asarray(row['world_matrix']);linear=matrix[:3,:3];position=matrix[:3,3]
    scale=np.linalg.norm(linear,axis=0);rotation=linear/scale
    kind=row['kind'];index=row['index']
    if kind in ('convex','triangles'):
        vertices=arrays[f'vertices_{index}']@linear.T+position
        faces=arrays[f'faces_{index}']
        if kind=='convex':
            hull=ConvexHull(vertices);faces=hull.simplices.copy()
            for i,face in enumerate(faces):
                a,b,c=vertices[face]
                if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[::-1]
            geometry=fcl.Convex(vertices,len(faces),np.c_[np.full(len(faces),3),faces].ravel())
        else:
            geometry=fcl.BVHModel();geometry.beginModel(len(vertices),len(faces))
            geometry.addSubModel(vertices,faces);geometry.endModel()
        return fcl.CollisionObject(geometry)
    if kind=='plane':
        normal=rotation[:,{'X':0,'Y':1,'Z':2}[row['axis']]]
        return fcl.CollisionObject(fcl.Plane(normal,float(normal@position)))
    if not np.allclose(scale,scale[0],atol=1e-6):raise ValueError('Nonuniform primitive scale needs exact handling: '+row['path'])
    if kind=='sphere':geometry=fcl.Sphere(row['radius']*scale[0])
    else:
        geometry=fcl.Capsule(row['radius']*scale[0],row['height']*scale[0])
        axis=row['axis']
        if axis=='X':rotation=rotation@Rotation.from_euler('y',90,degrees=True).as_matrix()
        elif axis=='Y':rotation=rotation@Rotation.from_euler('x',-90,degrees=True).as_matrix()
    return fcl.CollisionObject(geometry,fcl.Transform(rotation,position))


def run(directory):
    arrays=np.load(directory/'cooked_shapes.npz');reports=[]
    filters=json.loads((directory/'collision_filters.json').read_text()) if (directory/'collision_filters.json').exists() else []
    disabled_robot_roots=[row['path'] for row in filters if '/HandsOnRobots/' in row['path']
        and row['attributes'].get('physxArticulation:enabledSelfCollisions')=='False']

    for path in sorted(directory.glob('*_gpu_shapes.json')):
        rows=json.loads(path.read_text());objects=[object_for(row,arrays) for row in rows]
        overlaps=[];nearest=[]
        for i,j in itertools.combinations(range(len(rows)),2):
            a,b=rows[i],rows[j]
            if a['body']==b['body']:continue
            # Keep both arms, task links and all supports; only exact same-body
            # compound geometry is excluded before physics-filter review.
            if a['kind']=='plane' or b['kind']=='plane':
                plane,shape=(a,b) if a['kind']=='plane' else (b,a)
                pm=np.asarray(plane['world_matrix']);sm=np.asarray(shape['world_matrix'])
                normal=pm[:3,{'X':0,'Y':1,'Z':2}[plane['axis']]];normal/=np.linalg.norm(normal)
                offset=float(normal@pm[:3,3]);kind=shape['kind']
                if kind in ('convex','triangles'):
                    vertices=arrays[f"vertices_{shape['index']}"]@sm[:3,:3].T+sm[:3,3]
                    minimum=float(np.min(vertices@normal-offset))
                else:
                    minimum=float(normal@sm[:3,3]-offset)-shape['radius']*np.linalg.norm(sm[:3,0])
                    if kind=='capsule':minimum-=abs(normal@sm[:3,{'X':0,'Y':1,'Z':2}[shape['axis']]])*shape['height']/2
                if minimum < -1e-7:overlaps.append({'a':a['path'],'b':b['path'],'max_depth_m':-minimum,'points':[]})
                continue
            result=fcl.CollisionResult()
            fcl.collide(objects[i],objects[j],fcl.CollisionRequest(num_max_contacts=8,enable_contact=True),result)
            if result.is_collision:
                depths=[float(contact.penetration_depth) for contact in result.contacts]
                overlaps.append({'a':a['path'],'b':b['path'],'max_depth_m':max(depths,default=0),
                                 'points':[contact.pos.tolist() for contact in result.contacts]})
        for overlap in overlaps:
            a,b=overlap['a'],overlap['b']
            root=next((root for root in disabled_robot_roots if a.startswith(root+'/') and b.startswith(root+'/')),None)
            if root:
                overlap['classification']='existing_robot_self_collision_disabled'
                overlap['filter_source']=root
            elif ('/table/table/' in a or '/table/table/' in b) and any(
                    token in a or token in b for token in ('/gen7_bottle_collision_0/','/gen7_shoe_collision_1/')):
                overlap['classification']='intended_table_support' if overlap['max_depth_m']<=.0005 else 'excessive_support_penetration'
            else:overlap['classification']='unapproved_overlap'
        unapproved=[x for x in overlaps if x['classification'] in ('unapproved_overlap','excessive_support_penetration')]
        reports.append({'pose_file':path.name,'shape_count':len(rows),'cross_body_overlaps':overlaps,
                        'unapproved_overlap_count':len(unapproved)})
    report={'method':'FCL on actual PhysX-cooked convexes, exact analytic primitives and source triangle colliders, transformed by GPU rigid-body readback',
            'robot_filter_source':'collision_filters.json; existing canonical robot self-collision flags',
            'geometric_overlap_check_pass':bool(filters) and all(r['unapproved_overlap_count']==0 for r in reports),
            'full_task_acceptance_pass':False,'poses':reports}
    (directory/'geometric_overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pose_count':len(reports),'unapproved_overlap_count':sum(r['unapproved_overlap_count'] for r in reports),
                      'geometric_overlap_check_pass':report['geometric_overlap_check_pass']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);run(p.parse_args().directory)
