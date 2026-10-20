"""Actual-source CPU checks; supplemental MuJoCo settling is not a PhysX gate."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from handson.sim.isaaclab import eval_gen03_orient_bottle_upright as gen1

ROOT=Path(__file__).resolve().parents[3]
TASK=Path(__file__).resolve().parent
shape=gen1.geometry()
assert shape.mass_kg == .1
assert np.all(shape.diagonal_inertia>0)
assert max(shape.diagonal_inertia) <= sum(shape.diagonal_inertia)-max(shape.diagonal_inertia)
root=ET.parse(gen1.ASSET_DIR/'bottle.xml').getroot()
for mesh in root.findall('./asset/mesh'):
 mesh.set('file',str(gen1.ASSET_DIR/mesh.get('file')))
ET.SubElement(root,'option',timestep=str(1/120),gravity='0 0 -9.81',iterations='50')
ET.SubElement(root.find('worldbody'),'geom',name='table',type='box',pos='-.15 0 .86',size='.4 .8 .03',friction='.8 .005 .0001')
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
data=mujoco.MjData(model)
body=model.body(gen1.BODY_NAME).id
poses=[('boundary_'+str(i),p) for i,p in enumerate(gen1.boundary_reset_poses(shape,.89))]
poses += [('seed_'+str(i),gen1.sample_reset_pose(np.random.default_rng(i),shape,.89,randomize=True)) for i in range(20)]
result={'engine':'MuJoCo 3.4 supplemental CPU only; actual PhysX acceptance remains pending','cases':[]}
for name,pose in poses:
 mujoco.mj_resetData(model,data);data.qpos[:7]=pose;mujoco.mj_forward(model,data)
 initial_max_overlap=max([max(0.,-c.dist) for c in data.contact],default=0.)
 for i in range(480):mujoco.mj_step(model,data)
 final=np.r_[data.xpos[body],data.xquat[body]]
 vertices=shape.collision_vertices@gen1.rotation_matrix(final[3:]).T+final[:3]
 checks={'finite':bool(np.isfinite(data.qpos).all()),'initial_overlap_zero':initial_max_overlap==0,'within_table_x':bool(vertices[:,0].min()>=-.55 and vertices[:,0].max()<=.25),'within_table_y':bool(vertices[:,1].min()>=-.8 and vertices[:,1].max()<=.8),'support_error_below_1mm':bool(abs(vertices[:,2].min()-.89)<.001)}
 result['cases'].append({'name':name,'initial_pose_wxyz':pose.tolist(),'settled_pose_wxyz':final.tolist(),'initial_overlap_m':initial_max_overlap,'xy_drift_m':float(np.linalg.norm(final[:2]-pose[:2])),'linear_speed_m_s':float(np.linalg.norm(data.qvel[:3])),'angular_speed_rad_s':float(np.linalg.norm(data.qvel[3:])),'lowest_vertex_z_m':float(vertices[:,2].min()),'checks':checks,'passed':all(checks.values())})
from hydrate_cpu_scene import build
scene=build()
complete=mujoco.MjModel.from_xml_path(str(scene))
result['full_mjcf_compile']={'body_count':complete.nbody,'geometry_count':complete.ngeom,'mesh_count':complete.nmesh}
result['max_xy_drift_m']=max(r['xy_drift_m'] for r in result['cases'])
result['max_final_linear_speed_m_s']=max(r['linear_speed_m_s'] for r in result['cases'])
result['max_final_angular_speed_rad_s']=max(r['angular_speed_rad_s'] for r in result['cases'])
result['passed']=all(r['passed'] for r in result['cases'])
(TASK/'evidence/source_cpu_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
print('CPU source cases passed:',sum(r['passed'] for r in result['cases']),'/',len(poses))
assert result['passed'],'Supplemental MuJoCo checks failed; inspect per-case evidence. This is not a PhysX result.'
