"""Supplementary CPU geometry audit, never a replacement for live PhysX."""
from pathlib import Path
from types import SimpleNamespace
import itertools
import hashlib
import json

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from handson.sim.isaaclab import eval_gen06_spray_shoe_scene as g

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=Path(__file__).parent/'evidence'
XML=ROOT/'handson/handson/sim/envs/xmls/arena_arm_hand_eval_gen06_spray_shoe_rokae_wuji1.xml'
ARM_HOME={
    'right':[-1.5707963,1.5707963,1.5707963,1.5707963,-1.5707963,0,0],
    'left':[-.441741,.721729,-.239304,1.982058,.346778,-.925025,-.925025],
}


def compiled_mesh(model,data,index):
    mesh_id=model.geom_dataid[index]
    va=model.mesh_vertadr[mesh_id]; vn=model.mesh_vertnum[mesh_id]
    fa=model.mesh_faceadr[mesh_id]; fn=model.mesh_facenum[mesh_id]
    vertices=model.mesh_vert[va:va+vn].astype(float)
    # Compiled mesh coordinates are in the geom's principal frame. mj_forward
    # already includes MuJoCo's mesh normalization in geom_xmat/geom_xpos.
    vertices=vertices@data.geom_xmat[index].reshape(3,3).T+data.geom_xpos[index]
    return trimesh.Trimesh(vertices,model.mesh_face[fa:fa+fn],process=False)


def run():
    model=mujoco.MjModel.from_xml_path(str(XML));data=mujoco.MjData(model)
    for side,angles in ARM_HOME.items():
        for i,q in enumerate(angles,1):data.joint(f'joint{i}_{side}').qpos[0]=q
        data.joint(f'{side}_finger1_joint1').qpos[0]=.1
    mujoco.mj_forward(model,data)
    robot=trimesh.collision.CollisionManager(); robot_meshes={}
    for i in range(model.ngeom):
        body=model.body(model.geom_bodyid[i]).name
        if not any(side in body for side in ARM_HOME) or not model.geom_contype[i]:continue
        if model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Unexamined robot shape')
        mesh=compiled_mesh(model,data,i)
        name=f'{body}:geom{i}';robot.add_object(name,mesh);robot_meshes[name]=mesh
    points={side:tuple(data.xpos[i].copy() for i in range(model.nbody) if side in model.body(i).name) for side in ARM_HOME}
    env=SimpleNamespace(randomize=True,_random=np.random.default_rng(0),_reset_robot_body_points=lambda side:points[side])
    layouts=[('nominal',{'spray':g.BOTTLE_XY.copy(),g.SHOE_NAME:g.SHOE_XY.copy()},{'spray':0.,g.SHOE_NAME:0.})]
    for i,(sx,sy,angle) in enumerate(itertools.product([-1,1],repeat=3)):
        layouts.append((f'boundary_{i}',{'spray':g.BOTTLE_XY+g.XY_JITTER*np.array([sx,sy]),g.SHOE_NAME:g.SHOE_XY-g.XY_JITTER*np.array([sx,sy])},{'spray':angle*g.YAW_DEGREES,g.SHOE_NAME:-angle*g.YAW_DEGREES}))
    parameters=[]
    for seed in range(24):
        env._random=np.random.default_rng(seed);xy,_=g.sample_layout(env)
        original=env._gen7_reset_parameters.copy()
        env._random=np.random.default_rng(seed);g.sample_layout(env)
        np.testing.assert_array_equal(original,env._gen7_reset_parameters)
        layouts.append((f'seed_{seed}',xy,env._gen7_reset_yaws.copy()));parameters.append(original)
    meshes=g.ASSET_DIR/'meshes'
    object_meshes={
        'spray':[trimesh.load(p,force='mesh') for p in sorted(meshes.glob('gen7_bottle_collision_*.obj'))+sorted(meshes.glob('gen7_trigger_collision_*.obj'))],
        g.SHOE_NAME:[trimesh.load(p,force='mesh') for p in sorted(meshes.glob('gen7_shoe_collision_*.obj'))],
    }
    rows=[]
    for label,xy,yaw in layouts:
        collisions=[]; distance=10.; bodies={}
        for name,parts in object_meshes.items():
            transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('z',yaw[name],degrees=True).as_matrix()
            transform[:3,3]=[*xy[name],.8905 if name=='spray' else .89]
            bodies[name]=trimesh.collision.CollisionManager()
            for i,mesh in enumerate(parts):
                key=f'{name}:{i}';bodies[name].add_object(key,mesh,transform=transform)
                hit,pairs=robot.in_collision_single(mesh,transform=transform,return_names=True)
                if hit:collisions.extend(f'{key}/{pair}' for pair in pairs)
                distance=min(distance,robot.min_distance_single(mesh,transform=transform))
        hit,pairs=bodies['spray'].in_collision_other(bodies[g.SHOE_NAME],return_names=True)
        if hit:collisions.extend(f'{a}/{b}' for a,b in pairs)
        rows.append({'layout':label,'xy':{k:v.tolist() for k,v in xy.items()},'yaw_deg':yaw,
                     'sampler_margin_m':g.layout_margin(env,xy,yaw),'robot_minimum_distance_m':distance,'unapproved_pairs':collisions})
    table=trimesh.creation.box(extents=[.8,1.6,.06],transform=trimesh.transformations.translation_matrix([-.15,0,.86]))
    robot_table_hit,robot_table_pairs=robot.in_collision_single(table,return_names=True)
    report={'scope':'CPU source-MJCF FK plus FCL triangle mesh geometry; live imported PhysX first-step and settled audits still required',
            'layout_count':len(rows),'clear_layout_count':sum(not r['unapproved_pairs'] for r in rows),
            'robot_collision_meshes':len(robot_meshes),'robot_table_pairs':sorted(robot_table_pairs),
            'seed_repeatability':True,'bottle_xy_span_m':np.ptp(np.array(parameters)[:,:2],axis=0).tolist(),
            'right_palm_xyz':data.body('right_palm_link').xpos.tolist(),'left_palm_xyz':data.body('left_palm_link').xpos.tolist(),
            'rows':rows}
    report['pass']=not robot_table_hit and all(not r['unapproved_pairs'] for r in rows)
    report['object_xml_sha256']=hashlib.sha256((XML.parent/'eval_gen06_spray_shoe_objects.xml').read_bytes()).hexdigest()
    fixed=trimesh.collision.CollisionManager()
    for path in sorted(meshes.glob('gen7_bottle_collision_*.obj')):
        fixed.add_object(path.stem,trimesh.load(path,force='mesh'))
    trigger_parts=[trimesh.load(path,force='mesh') for path in sorted(meshes.glob('gen7_trigger_collision_*.obj'))]
    sweep=[];manifest=g.asset_manifest()
    for q in np.linspace(0,.45,91):
        transform=trimesh.transformations.rotation_matrix(q,manifest['hinge_axis'],manifest['hinge_pivot'])
        pairs=[];gap=10.
        for i,part in enumerate(trigger_parts):
            hit,names=fixed.in_collision_single(part,transform=transform,return_names=True)
            if hit:pairs.extend(f'{i}:{name}' for name in names)
            gap=min(gap,fixed.min_distance_single(part,transform=transform))
        sweep.append({'q_rad':float(q),'colliding_pairs':pairs,'minimum_distance_m':gap})
    sweep_report={'scope':'CPU source-derived collider sweep; live trigger actuation remains required',
                  'object_xml_sha256':report['object_xml_sha256'],'poses':len(sweep),
                  'clear_poses':sum(not row['colliding_pairs'] for row in sweep),
                  'minimum_distance_m':min(row['minimum_distance_m'] for row in sweep),
                  'collider_counts':[manifest['bottle_collider_count'],manifest['trigger_collider_count']],
                  'rows':sweep}
    sweep_report['status']='passed' if sweep_report['clear_poses']==len(sweep) else 'failed'
    (EVIDENCE/'bottle_collider_sweep.json').write_text(json.dumps(sweep_report,indent=2)+'\n')
    report['trigger_sweep_pass']=sweep_report['status']=='passed'
    report['pass'] &= report['trigger_sweep_pass']
    (EVIDENCE/'cpu_layout_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    return report


if __name__=='__main__':
    raise SystemExit(0 if run()['pass'] else 1)
