#!/usr/bin/env python3
"""CPU native source-robot FK clearance screen, not the final Isaac overlap gate."""
from pathlib import Path
import json
import mujoco
import numpy as np
from audit_assets_cpu import ROOT, OUT, corners, pose
from handson.sim.envs.hand_specs import ROKAE_LEFT_HOME, ROKAE_RIGHT_HOME
from handson.sim.isaaclab.gen4_scene import load_geometry, sample_layout
from handson.tasks.eval_gen04_pass_glue_stick.semantics import transformed_points
from handson.tasks.eval_gen04_pass_glue_stick import contract as c


def main():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml'))
    d=mujoco.MjData(m)
    for side,home in [('left',ROKAE_LEFT_HOME),('right',ROKAE_RIGHT_HOME)]:
        for i,q in enumerate(home): d.qpos[m.joint(f'joint{i+1}_{side}').qposadr]=q
    mujoco.mj_forward(m,d)
    robot=[]
    for i in range(m.ngeom):
        if m.geom_contype[i]==0 and m.geom_conaffinity[i]==0: continue
        b=m.geom_aabb[i]
        world=corners([b[:3]-b[3:],b[:3]+b[3:]])@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
        robot.append({'body':m.body(int(m.geom_bodyid[i])).name,'geom':m.geom(i).name or str(i),'bounds_m':[world.min(0).tolist(),world.max(0).tolist()]})
    g=load_geometry();potential=[]
    for seed in range(20):
        row=sample_layout(np.random.RandomState(seed),randomize=True,geometry=g)
        for name,r in row.items():
            bounds=g['glue']['bounds_m'] if name==c.GLUE_BODY else g['tray']['bounds_m'] if name==c.TRAY_BODY else g['distractors'][name]['bounds_m']
            points=transformed_points(corners(bounds),pose(r));lo,hi=points.min(0),points.max(0)
            for rob in robot:
                low,high=np.asarray(rob['bounds_m']);overlap=np.minimum(hi,high)-np.maximum(lo,low)
                if np.all(overlap>0): potential.append({'seed':seed,'robot':rob['body'],'object':name,'aabb_overlap_m':overlap.tolist()})
    record={'method':'CPU native MuJoCo robot FK at D03 home, complete enabled source collision AABBs; not final Isaac collision proof',
        'robot_collision_geoms':len(robot),'robot_geometry':robot,'samples':20,'candidate_pairs':potential,
        'final_Isaac_before_reset_and_settled_overlap':'pending'}
    (OUT/'native_home_clearance_cpu.json').write_text(json.dumps(record,indent=2)+'\n')
    print({'robot_collision_geoms':len(robot),'candidate_pairs':potential,'final_Isaac_gate':'pending'})
    assert not potential

if __name__=='__main__':main()
