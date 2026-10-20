"""D08-2: user-authored Blender visuals on the shared physical toast workflow."""
from __future__ import annotations

from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

TASK_NAME = 'd08_2_realistic_toast'
BASE_TASK_NAME = 'd08_toast_bread'
TOAST_TASKS = frozenset((BASE_TASK_NAME, TASK_NAME))
ASSETS = Path(__file__).resolve().parents[1] / 'envs/xmls/assets/d08_2_realistic_toast'
SLOT_CENTERS = (-.02952, .02952)
BREAD_RAISED_CENTER_Z = .147
LEVER_ORIGIN = (-.11808, .02376, .12240)
CYCLE_SECONDS = 8.0

def family_name(task_name: str) -> str:
    return BASE_TASK_NAME if task_name == TASK_NAME else task_name

def initial_control_state_report(env):
    """Check the actual selected robot: full arms or the shared floating hands."""
    from .priority_zero_robot_audit import initial_robot_state_report, pose_error_wxyz
    if not env._direct_hand_control:
        return initial_robot_state_report(env)
    import numpy as np
    def array(value):
        if hasattr(value, 'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value)
    result={}
    for side in ('right','left'):
        robot=env._robots[side];index=env._ee_body_indices[side]
        pose=np.concatenate((array(robot.data.body_pos_w)[0,index],array(robot.data.body_quat_w)[0,index]))
        velocity=array(robot.data.body_vel_w)[0,index]
        q=array(robot.data.joint_pos)[0];qd=array(robot.data.joint_vel)[0]
        target=np.asarray(env._direct_motion_target_pose[side])
        finite,error,angle=pose_error_wxyz(pose,target)
        hand_error=float(np.max(np.abs(q[np.asarray(env._hand_joint_ids[side])]-env._direct_motion_target_hand[side])))
        checks={
            'no_hidden_arm':len(env._arm_joint_ids[side])==0,
            'twenty_hand_joints':len(q)==20,
            'finite':bool(finite and np.isfinite(velocity).all() and np.isfinite(q).all() and np.isfinite(qd).all()),
            'palm_home':error<=.003 and angle<=np.deg2rad(1),
            'hand_home':hand_error<=.01,
            'stationary':float(np.linalg.norm(velocity[:3]))<=.001 and float(np.linalg.norm(velocity[3:]))<=.002 and float(np.max(np.abs(qd)))<=.001,
            'above_table':float(pose[2])>.90,
            'correct_side':float(pose[1])<0 if side=='right' else float(pose[1])>0,
        }
        checks={key:bool(value) for key,value in checks.items()}
        result[side]={'passed':all(checks.values()),'checks':checks,'physical_servo':bool(env._direct_physics_servo_enabled),'actual_palm_pose_wxyz':pose.tolist(),
            'expected_palm_pose_wxyz':target.tolist(),'actual_palm_velocity':velocity.tolist(),
            'actual_hand_joint_position_rad':q.tolist(),'actual_hand_joint_velocity_rad_s':qd.tolist(),
            'position_error_m':error,'rotation_error_deg':float(np.rad2deg(angle)),'hand_home_error_rad':hand_error}
    return all(row['passed'] for row in result.values()),result

def asset_digest() -> str:
    h=hashlib.sha256(Path(__file__).read_bytes())
    for file in sorted(ASSETS.rglob('*')):
        if file.is_file():h.update(file.name.encode());h.update(file.read_bytes())
    return h.hexdigest()[:20]

def configure_source(root: ET.Element) -> None:
    """Author the final simple colliders before the first PhysX state exists."""
    bodies={b.get('name'):b for b in root.findall('.//body')}
    machine=bodies['d08_toaster'];lever=bodies['d08_toaster_button']
    for name in ['d08_toaster','d08_toaster_button','d08_bread_basket','d08_plate','d08_bread_0','d08_bread_1']:
        for geom in list(bodies[name].findall('geom')):bodies[name].remove(geom)
    def box(body,name,half,pos=(0,0,0),mass=None,kind='box'):
        attrs=dict(name=name,type=kind,size=' '.join(map(str,half)),pos=' '.join(map(str,pos)),
                   rgba='0 0 0 0',group='3',friction='0.9 .01 .001',condim='4')
        if mass is not None:attrs['mass']=str(mass)
        ET.SubElement(body,'geom',attrs)
    box(machine,'d08_toaster_collision_base',(.083,.055,.010),(0,0,.018))
    box(machine,'d08_toaster_collision_side_n',(.079,.006,.052),(0,-.060,.081))
    box(machine,'d08_toaster_collision_side_p',(.079,.006,.052),(0,.060,.081))
    box(machine,'d08_toaster_collision_front',(.006,.048,.052),(-.090,0,.081))
    box(machine,'d08_toaster_collision_back',(.006,.048,.052),(.090,0,.081))
    box(machine,'d08_toaster_collision_divider',(.079,.017,.058),(0,0,.094))
    for i,x in enumerate((-.0792,.0792)):
        for j,y in enumerate((-.04824,.04824)):
            box(machine,f'd08_2_corner_{i}_{j}',(.0175,.050),(x,y,.081),kind='cylinder')
    # The outer top shoulders cover the sheet metal beside each real slot.
    for sign in (-1,1):
        box(machine,f'd08_2_top_outer_{sign}',(.079,.0085,.004),(0,sign*.050,.152))
        box(machine,f'd08_2_top_end_{sign}',(.006,.049,.004),(sign*.075,0,.152))
    for i,y in enumerate(SLOT_CENTERS):
        for suffix,sign in [('n',-1),('p',1)]:
            box(machine,f'd08_slot_{i}_guide_{suffix}',(.062,.0015,.047),(0,y+sign*.0105,.101))
        box(machine,f'd08_slot_{i}_guide_front',(.0015,.009,.047),(-.063,y,.101))
        box(machine,f'd08_slot_{i}_guide_back',(.0015,.009,.047),(.063,y,.101))
        box(machine,f'd08_toaster_low_stop_slot_{i}',(.057,.007,.002),(0,y,.04936))
        box(machine,f'd08_toaster_rack_slot_{i}',(.057,.007,.002),(0,y,.09936))
    lever.set('pos',' '.join(map(str,LEVER_ORIGIN)))
    box(lever,'d08_toaster_lever_collision',(.0108,.0118,.0046),(0,0,0),.025)
    box(lever,'d08_toaster_moving_rack_collision',(.057,.0365,.0018),(.11808,-.02376,-.02304),.010)
    board=bodies['d08_bread_basket']
    box(board,'d08_bread_tray_floor',(.082,.118,.009),(0,0,.011))
    for index in range(2):
        bread=bodies[f'd08_bread_{index}'];pos=bread.get('pos').split();pos[2]='.9273';bread.set('pos',' '.join(pos))
        box(bread,f'd08_bread_{index}_collision',(.0355,.0062,.027),(0,0,-.016),.025)
        box(bread,f'd08_bread_{index}_crown_collision',(.040,.0062,.014),(0,0,.025),.010)
    box(bodies['d08_plate'],'d08_2_plate_collision',(.127,.008),(0,0,0),kind='cylinder')

def compose_visuals(stage,task_root: str) -> dict[str,str]:
    """Reference PBR USD under the real bodies, retaining their PhysX identity."""
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
    from isaaclab import sim as sim_utils
    sim_utils.make_uninstanceable(task_root,stage=stage)
    root=stage.GetPrimAtPath(task_root)
    body_files={'d08_toaster':'toaster','d08_toaster_button':'lever_racks','d08_bread_0':'bread',
                'd08_bread_1':'bread','d08_bread_basket':'board','d08_plate':'plate'}
    matches={}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in body_files:
            # The MJCF link Xform is the visible body's transform; only one path may match.
            if prim.GetName() not in matches or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                matches[prim.GetName()]=prim
    missing=set(body_files)-set(matches)
    if missing:raise RuntimeError(f'D08-2 missing imported body roots: {sorted(missing)}')
    paths={}
    for name,file in body_files.items():
        parent=matches[name]
        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(parent).CreateThresholdAttr(0.0)
        for prim in Usd.PrimRange(parent):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
        path=str(parent.GetPath())+'/BlenderVisual'
        if not stage.GetPrimAtPath(path):
            prim=UsdGeom.Xform.Define(stage,path).GetPrim()
            prim.GetReferences().AddReference(str(ASSETS/(file+'.usdc')))
        paths[name]=path
    return paths

def set_appearance(stage,paths: dict[str,str],progress: float,heating: bool) -> None:
    from pxr import Usd,UsdShade,Sdf,Gf
    level=max(0,min(4,int(progress*4+1e-6)))
    for body in ('d08_bread_0','d08_bread_1'):
        root=stage.GetPrimAtPath(paths[body])
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdShade.Shader):continue
            shader=UsdShade.Shader(prim)
            if shader.GetIdAttr().Get()=='UsdUVTexture':
                file=shader.GetInput('file');value=file.Get()
                if value and any(f'bread_{i}.png' in value.path for i in range(5)):
                    file.Set(Sdf.AssetPath(str(ASSETS/f'bread_{level}.png')))
    # The exported nichrome material is shared by all visible heater wires.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(paths['d08_toaster'])):
        if prim.IsA(UsdShade.Shader) and 'Nichrome' in str(prim.GetPath()):
            shader=UsdShade.Shader(prim)
            if shader.GetIdAttr().Get()=='UsdPreviewSurface':
                shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(5,.22,.008) if heating else Gf.Vec3f(0))
