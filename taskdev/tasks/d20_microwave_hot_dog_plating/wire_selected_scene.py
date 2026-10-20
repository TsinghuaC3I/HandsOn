#!/usr/bin/env python3
"""Author only the D20 sections of the shared MJCF from measured asset bounds."""
from pathlib import Path
import json, math, re
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'handson/handson/sim/envs/xmls'
ASSETS=ROOT/'assets/d20_selected'
M=json.loads((ASSETS/'manifest.json').read_text())
TABLE_Z=.89
RESET_XY={'hot_dog':[-.29,-.29],'tongs':[-.37,-.69],'plate':[-.29,-.29],
          'fork':[-.29,-.085],'mustard':[-.22,-.48],'mug':[-.39,-.48]}
YAW={'hot_dog':90,'tongs':-22,'plate':0,'fork':90,'mustard':0,'mug':0}
TRAY_XY=[-.29,-.29];MICROWAVE_XY=[.05,.14]


def supported_rotation(name):
    key='tongs_fixed' if name=='tongs' else name
    pieces=[trimesh.load(ASSETS/p,process=False) for p in M['assets'][key]['collision']]
    com=np.mean(M['assets'][key]['bounds'],axis=0)
    if name=='tongs':
        moving=[trimesh.load(ASSETS/p,process=False) for p in M['assets']['tongs_moving']['collision']]
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',.32).as_matrix();t[:3,3]=M['positions']['tongs_hinge']
        for part in moving:part.apply_transform(t)
        moving_com=t[:3,:3]@np.mean(M['assets']['tongs_moving']['bounds'],axis=0)+t[:3,3]
        com=(.11*com+.055*moving_com)/.165
        pieces+=moving
    mesh=trimesh.util.concatenate(pieces)
    rest=np.eye(3)
    if name in ('hot_dog','fork','tongs'):
        poses,probability=mesh.compute_stable_poses(center_mass=com,threshold=.001)
        candidates=[(p,t) for p,t in zip(probability,poses) if t[2,2]>math.cos(math.radians(15))]
        _,pose=max(candidates,key=lambda item:item[0])
        rest=pose[:3,:3]
        # Preserve the source's horizontal heading; only seat its support face.
        rest=Rotation.from_euler('z',-math.atan2(rest[1,0],rest[0,0])).as_matrix()@rest
    return rest,float(np.min((mesh.vertices@rest.T)[:,2]))


def vec(v):return ' '.join(f'{float(x):.9g}' for x in v)


def geoms(body,key,mass=0):
    a=M['assets'][key]
    ET.SubElement(body,'geom',name=f'd20_{key}_visual',type='mesh',mesh=f'd20_{key}_mesh',
        material=f'd20_{key}_material',contype='0',conaffinity='0',density='0',group='1')
    assert a['collision'],key
    for i,_ in enumerate(a['collision']):
        ET.SubElement(body,'geom',name=f'd20_{key}_collision_{i:03d}',type='mesh',
            mesh=f'd20_{key}_collision_{i:03d}',contype='1',conaffinity='1',
            group='3',rgba='0 0 0 0',mass=str(mass/len(a['collision'])),
            friction='1 .005 .0001',condim='6')


def inertial(body,key,mass):
    b=np.asarray(M['assets'][key]['bounds']);ext=b[1]-b[0];center=b.mean(0)
    inertia=mass/12*np.array([ext[1]**2+ext[2]**2,ext[0]**2+ext[2]**2,ext[0]**2+ext[1]**2])
    ET.SubElement(body,'inertial',pos=vec(center),mass=str(mass),diaginertia=vec(inertia))


def main():
    asset=ET.Element('asset')
    for key,a in M['assets'].items():
        mat=M['materials'][key];kw={}
        if 'baseColor' in mat:
            tex=f'd20_{key}_texture';ET.SubElement(asset,'texture',type='2d',name=tex,file=f'assets/d20_selected/{mat["baseColor"]}')
            kw['texture']=tex
        else:kw['rgba']=vec(mat.get('baseColorFactor',[1,1,1,1]))
        ET.SubElement(asset,'material',name=f'd20_{key}_material',specular='.4',shininess='.3',**kw)
        ET.SubElement(asset,'mesh',name=f'd20_{key}_mesh',file=f'assets/d20_selected/{a["visual"]}')
        for i,path in enumerate(a['collision']):
            ET.SubElement(asset,'mesh',name=f'd20_{key}_collision_{i:03d}',file=f'assets/d20_selected/{path}')
    world=ET.Element('worldbody')
    microwave_z=TABLE_Z-M['assets']['microwave_shell']['bounds'][0][2]
    oven=ET.SubElement(world,'body',name='d20_microwave',pos=vec([*MICROWAVE_XY,microwave_z]),quat='.707106781 0 0 -.707106781')
    geoms(oven,'microwave_shell');geoms(oven,'control_panel')
    door=ET.SubElement(oven,'body',name='d20_microwave_door')
    inertial(door,'microwave_door',.65)
    ET.SubElement(door,'joint',name='d20_door_joint',type='hinge',axis='0 0 1',
        pos=vec(M['positions']['door_hinge']),range='-1.6 0',limited='true',damping='.15',frictionloss='.015',armature='.003')
    geoms(door,'microwave_door',.65)
    for key,joint,limit in [('power_knob','power',math.pi),('timer_knob','timer',math.radians(307))]:
        body=ET.SubElement(oven,'body',name=f'd20_{key}',pos=vec(M['positions'][key]))
        inertial(body,key,.015)
        ET.SubElement(body,'joint',name=f'd20_{joint}_joint',type='hinge',axis='0 1 0',
            range=vec([0,limit]),limited='true',damping='.002',frictionloss='0',armature='.000002')
        geoms(body,key,.015)
    tray=ET.SubElement(world,'body',name='d20_serving_tray',pos=vec([*TRAY_XY,TABLE_Z]))
    geoms(tray,'tray')
    # The tray's large horizontal interior facet is measured at 13.58238 mm.
    support=TABLE_Z+.01358238
    masses={'hot_dog':.12,'tongs':.11,'plate':.28,'fork':.035,'mustard':.25,'mug':.24}
    reset={}
    for name,xy in RESET_XY.items():
        key='tongs_fixed' if name=='tongs' else name
        rest,bottom=supported_rotation(name)
        support_z=TABLE_Z if name=='tongs' else support
        if name=='hot_dog':
            # Plate root has its own 0.1 mm reset clearance over the tray.
            plate_bottom=min(b[0][2] for b in M['assets']['plate']['collision_bounds'])
            support_z=support-plate_bottom+.0001+M['notes']['plate_food_support_local_z']
        z=support_z-bottom+.0001
        q=Rotation.from_matrix(Rotation.from_euler('z',YAW[name],degrees=True).as_matrix()@rest).as_quat()
        body=ET.SubElement(world,'body',name=f'd20_{name}',pos=vec([*xy,z]),quat=vec(q[[3,0,1,2]]))
        ET.SubElement(body,'freejoint',name=f'd20_{name}_root')
        inertial(body,key,masses[name]);geoms(body,key,masses[name])
        if name=='tongs':
            child=ET.SubElement(body,'body',name='d20_tongs_moving_arm',pos=vec(M['positions']['tongs_hinge']))
            inertial(child,'tongs_moving',.055)
            ET.SubElement(child,'joint',name='d20_tongs_joint',type='hinge',axis='0 0 1',
                range='0 .5',limited='true',springref='.32',stiffness='8',damping='.20',frictionloss='.012',armature='.001')
            geoms(child,'tongs_moving',.055)
        reset[f'd20_{name}']={'xy':xy,'z':z,'yaw_degrees':YAW[name], 'rest_rotation':rest.tolist(), 'quat_xyzw':q.tolist()}
    def fragment(element):
        ET.indent(element,space='  ')
        return '\n'.join(ET.tostring(element,encoding='unicode').splitlines()[1:-1])
    p=ROOT/'d_long_horizon_objects.xml';text=p.read_text()
    start=text.index('    <!-- D20:',text.index('<asset>'));end=text.index('  </asset>',start)
    text=text[:start]+'    <!-- D20: user-selected realistic meshes; generated by wire_selected_scene.py. -->\n'+fragment(asset)+'\n'+text[end:]
    start=text.index('    <!-- D20:',text.index('<worldbody>'));end=text.index('  </worldbody>',start)
    text=text[:start]+'    <!-- D20: measured link frames, source-derived collision, no visible proxies. -->\n'+fragment(world)+'\n'+text[end:]
    text=re.sub(r'    <!-- D20: the start button.*?</contact>', '''    <!-- D20 knob stems are mounted inside their fixed panel; the spring
         tongs bend is internal to the articulation. All hand contacts remain. -->
    <exclude body1="d20_microwave" body2="d20_power_knob" />
    <exclude body1="d20_microwave" body2="d20_timer_knob" />
    <exclude body1="d20_tongs" body2="d20_tongs_moving_arm" />
  </contact>''',text,flags=re.S)
    p.write_text(text)
    M['reset']=reset;M['fixture_poses']={'microwave':[*MICROWAVE_XY,microwave_z],'tray':[*TRAY_XY,TABLE_Z]}
    M['tray_support_z']=support
    (ASSETS/'manifest.json').write_text(json.dumps(M,ensure_ascii=False,indent=2)+'\n')
    print('Wired D20 assets; reset:',json.dumps(reset))


if __name__=='__main__':main()
