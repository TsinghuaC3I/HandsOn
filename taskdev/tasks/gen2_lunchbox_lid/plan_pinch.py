"""Offline finger IK using unchanged source hand kinematics and tip meshes."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,mujoco,trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
base=Path(__file__).parent/'evidence/xml_mirror'
m=mujoco.MjModel.from_xml_path(str(base/'wuji_hand1/mjcf/right_primitive.xml'));d=mujoco.MjData(m)
source=ET.parse(base/'rokae_wuji1_right.xml').getroot()
centers={}
for i in (1,2):
 name=f'right_finger{i}_tip_link'
 mesh=source.find(f"./asset/mesh[@name='{name}']")
 t=trimesh.load_mesh(base/mesh.get('file'));center=t.bounds.mean(0)
 geom=source.find(f".//geom[@name='{name}_vis']");offset=np.fromstring(geom.get('pos'),sep=' ')
 centers[i]=center+offset
rot=Rotation.from_euler('y',100,degrees=True)
palm=np.array([.010,.080,.080])
goals={1:np.array([.125,.124,.001]),2:np.array([.153,.100,.001])}
q=np.zeros(20);q[0]=.1
results={}
for finger in (1,2):
 ids=np.arange((finger-1)*4,finger*4);body=m.body(f'right_finger{finger}_link4').id
 def position(x):
  d.qpos[:]=q;d.qpos[ids]=x;mujoco.mj_forward(m,d)
  return rot.apply(d.xpos[body]+d.xmat[body].reshape(3,3)@centers[finger])+palm
 def objective(x):return np.r_[(position(x)-goals[finger])*100, .015*(x-np.array([.8,.3,.5,.5] if finger==1 else [.5,0,.5,.5]))]
 best=None
 for seed in range(10):
  rng=np.random.default_rng(seed);guess=rng.uniform(m.jnt_range[ids,0]+1e-5,m.jnt_range[ids,1]-1e-5)
  sol=least_squares(objective,guess,bounds=(m.jnt_range[ids,0],m.jnt_range[ids,1]),max_nfev=500)
  if best is None or sol.cost<best.cost:best=sol
 q[ids]=best.x
 results[finger]=dict(joints=best.x.tolist(),position=position(best.x).tolist(),goal=goals[finger].tolist(),error_m=float(np.linalg.norm(position(best.x)-goals[finger])))
print(json.dumps(dict(palm=palm.tolist(),rotation_wxyz=[rot.as_quat()[3],*rot.as_quat()[:3]],hand=q.tolist(),results=results),indent=2))
