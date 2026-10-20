"""Audit the source/prepared MJCF without Kit; not a final PhysX overlap gate."""
from pathlib import Path
import itertools
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'handson'),str(ROOT/'handson/tests')]
from test_isaaclab_task_layout import _load_task_preview_module
from test_b03_newton_robot_mjcf_cache import _load_robot_mjcf_generator
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.tasks.eval_gen01_cover_lunchbox import geometry as g

OUT=ROOT/'taskdev/tasks/eval_gen01_cover_lunchbox/evidence'
cache=OUT/'prepared_cpu'
cache.mkdir(parents=True,exist_ok=True)
mirror=OUT/'xml_mirror'
preview=_load_task_preview_module(cache)
preview._XML_DIR=mirror
preview._PACKAGED_XML_DIR=mirror
# Only Kit entry points are stubbed. Actual source XML transformations are run.
generator=_load_robot_mjcf_generator(cache,xml_dir=mirror)
preview._prepare_robot_mjcf=generator._prepare_robot_mjcf
report=dict(scope='CPU source and actual MJCF transformations, not final USD/PhysX',
            mujoco_version=mujoco.__version__,compositions=[])
for first in (False,True):
    path=preview.prepare_task_mjcf(g.TASK_NAME,first_physx_safe=first)
    m=mujoco.MjModel.from_xml_path(str(path))
    d=mujoco.MjData(m)
    # Apply the preview's actual configured arm and hand defaults for source review.
    for side,home in preview._task_arm_home(g.TASK_NAME).items():
        for i,q in enumerate(home,1):
            joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}_{side}')
            if joint>=0:d.qpos[m.jnt_qposadr[joint]]=q
        for finger in range(1,6):
            for j in range(1,5):
                jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_finger{finger}_joint{j}')
                if jid>=0:d.qpos[m.jnt_qposadr[jid]]=.1 if finger==1 and j==1 else 0.
    mujoco.mj_forward(m,d)
    free=[m.body(int(m.jnt_bodyid[j])).name for j in range(m.njnt) if m.jnt_type[j]==0]
    assert set(free)==set(g.BODY_NAMES)
    source_contacts=[]
    for contact in d.contact:
        if contact.dist<0:
            source_contacts.append(dict(pair=[m.geom(int(i)).name for i in contact.geom],distance_m=float(contact.dist)))
    report['compositions'].append(dict(first_physx_safe=first,path=str(path.relative_to(ROOT)),
        bodies=m.nbody,geoms=m.ngeom,free_bodies=free,source_contacts=source_contacts))
    for name in g.BODY_NAMES:
        assert np.allclose(d.body(name).xpos[:2],g.DEFAULT_XY[name])
        assert d.body(name).xpos[2]+g.SUPPORT_MIN_Z[name]==np.float64(.891)
# Seeded sampler evidence and declared boundaries, including both robot point sets.
e= HandsOnABCIsaacLabEnv(g.TASK_NAME,launch_isaac=False,image_obs=False,randomize=True,seed=0)
points=(*e._reset_robot_body_points('right'),*e._reset_robot_body_points('left'))
records=[]
for seed in range(20):
    e._random=np.random.RandomState(seed)
    a=e._task_specific_reset_xy_positions()
    e._random=np.random.RandomState(seed)
    b=e._task_specific_reset_xy_positions()
    assert all(np.array_equal(a[n],b[n]) for n in g.BODY_NAMES)
    records.append(dict(seed=seed,positions={n:a[n].tolist() for n in g.BODY_NAMES},
                        prescreen_margin_m=g.reset_layout_margin(e,a,points)))
report['seeds']=records
report['position_span_m']={n:np.ptp([r['positions'][n] for r in records],axis=0).tolist() for n in g.BODY_NAMES}
report['boundaries']=[]
for sx,sy,opposed in itertools.product((-1,1),repeat=3):
    layout={g.BOX:g.DEFAULT_XY[g.BOX]+np.array([sx,sy])*.025,
            g.LID:g.DEFAULT_XY[g.LID]+np.array([opposed*sx,-sy])*.025}
    margin=g.reset_layout_margin(e,layout,points)
    assert margin>=0
    report['boundaries'].append(dict(positions={n:p.tolist() for n,p in layout.items()},prescreen_margin_m=margin))
e.close()
report['final_physx_overlap_status']='pending GPU validation'
(OUT/'cpu_scene.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('seeds','boundaries')},indent=2))
