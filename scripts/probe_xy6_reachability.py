"""Source-FK reach screen adapted from XY5; no physical grasp/path claim."""
import argparse,ast,importlib,json,os,sys,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'handson')]

def prepare_source_scene(cache: Path):
    """Execute real MJCF preparation helpers without GPU-only scene configs.

    Only Isaac imports and the declarative GPU scene configuration class are
    omitted from the assembly module. Mesh/XML preparation code is unchanged.
    No simulator, rendering or physics implementation is replaced or claimed.
    """
    os.environ["HANDSON_ISAACLAB_CACHE"] = str(cache)
    name = "handson.sim.isaaclab.bimanual_assembly_scene"
    source = ROOT / "handson/handson/sim/isaaclab/bimanual_assembly_scene.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab")
        or isinstance(node, ast.Import) and any(a.name.startswith("isaaclab") for a in node.names)
        or isinstance(node, ast.ClassDef) and node.name == "BimanualAssemblySceneCfg"
    )]
    module = types.ModuleType(name)
    module.__file__ = str(source)
    module.__package__ = "handson.sim.isaaclab"
    sys.modules[name] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    lab = types.ModuleType("isaaclab")
    lab.sim = types.ModuleType("isaaclab.sim")
    sys.modules["isaaclab"] = lab
    sys.modules["isaaclab.sim"] = lab.sim
    preview = importlib.import_module("handson.sim.isaaclab.task_preview")
    from handson.sim.isaaclab.eval_comb06_season_and_stir_scene import TASK_NAME
    return {side:module._prepare_robot_mjcf(side,preserve_base_pose=True,ensure_proximal_finger_collisions=True) for side in ('left','right')}, preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    mjcf,preview=prepare_source_scene(args.output.parent/'reach-scene-cache')
    import mujoco
    from scipy.spatial.transform import Rotation
    from probe_loaded_tray_gate_reachability import _arm_indices,_refine_pose
    from handson.sim.isaaclab.eval_comb06_season_and_stir_scene import TASK_NAME,rotation_wxyz
    from handson.sim.isaaclab.b320_bimanual_env import _KA012_LEFT_HANDLE_LOCAL_OFFSET_M
    models={side:mujoco.MjModel.from_xml_path(str(path)) for side,path in mjcf.items()}
    datas={side:mujoco.MjData(model) for side,model in models.items()}
    layouts=json.loads((ROOT/'artifacts/xy6/gpu/tracking-04/report.json').read_text())['rows']
    native=layouts[0]['reset']['robots']
    homes={side:[dict(zip(native[side]['names'],native[side]['q']))[f'joint{i}_{side}'] for i in range(1,8)] for side in ('left','right')}
    for side in ('left','right'):
        model,data=models[side],datas[side]
        for i,q in enumerate(homes[side],1):data.qpos[model.joint(f'joint{i}_{side}').qposadr[0]]=q
        for i in range(model.njnt):
            if model.joint(i).name.startswith(side+'_finger'):
                data.qpos[model.jnt_qposadr[i]]=np.clip(.8,*model.jnt_range[i])
        mujoco.mj_forward(model,data)
    frames={};source_homes={}
    for side in ('left','right'):
        model,data=models[side],datas[side]
        palm=model.body(side+'_palm_link').id;R=data.xmat[palm].reshape(3,3).copy()
        offset=np.mean([data.xpos[model.body(f'{side}_finger{i}_link4').id]-data.xpos[palm] for i in (1,2)],axis=0)
        frames[side]=(R,R.T@offset)
        source_homes[side]=data.xpos[palm].copy()
    home_errors={side:float(np.linalg.norm(source_homes[side]-np.asarray(layouts[0]['reset']['robots'][side]['tcp_actual'])[:3])) for side in ('left','right')}
    assert max(home_errors.values())<.0001,home_errors
    results=[]
    for row in layouts:
        poses={n:np.asarray(p) for n,p in row['sampled_layout'].items()}
        pot=poses['dutch_oven'];Rpot=rotation_wxyz(pot[3:])
        targets=[]
        for side,jar in (('left','salt_jar'),('right','pepper_jar')):
            for phase,dz in (('jar_pick',.065),('jar_lift',.145),('jar_release',.065)):
                targets.append((side,phase,poses[jar][:3]+[0,0,dz],0.))
            # A tilted grasp-centre above the mouth is a reach screen only;
            # cap placement and contact timers are checked separately.
            targets.append((side,'tilted_pour_region',pot[:3]+[-.015,.08 if side=='left' else -.08,.22],110. if side=='left' else -110.))
        targets.append(('left','pot_handle',pot[:3]+Rpot@np.asarray(_KA012_LEFT_HANDLE_LOCAL_OFFSET_M),0.))
        spoon=poses['spoon'];targets.append(('right','spoon_handle',spoon[:3]+rotation_wxyz(spoon[3:])@np.array([0,0,.025])+[0,0,.01],0.))
        for sector in range(6):
            angle=2*np.pi*sector/6
            targets.append(('right',f'stir_sector_{sector}',pot[:3]+[.045*np.cos(angle),.045*np.sin(angle),.20],0.))
        for side,phase,grip,tilt in targets:
            model,data=models[side],datas[side]
            Rhome,offset=frames[side];joint_ids,qids=_arm_indices(model,side)
            low,high=model.jnt_range[joint_ids,0]+.03,model.jnt_range[joint_ids,1]-.03
            qhome=np.clip(homes[side],low+1e-6,high-1e-6)
            candidates=[]
            for yaw in (0.,-30.,30.):
                R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@Rotation.from_euler('x',tilt,degrees=True).as_matrix()@Rhome
                target=grip-R@offset
                answer=_refine_pose(model,data,side,target,R,qhome,low,high,qids)
                candidates.append((answer,target,R,yaw))
                if answer[1]<.005 and answer[2]<5:break
            answer,target,R,yaw=min(candidates,key=lambda x:x[0][-1])
            q,pos,ang,margin,score=answer
            results.append({'layout':row['label'],'side':side,'phase':phase,'grasp_region':grip.tolist(),'palm_target':target.tolist(),
                'palm_rotation':R.tolist(),'position_error_m':pos,'orientation_error_deg':ang,'joint_margin_rad':margin+.03,'q':q.tolist(),
                'feasible_screen':pos<.015 and ang<10.})
    report={'scope':'source-FK bounded IK screen; no physical grasp, collision-free path or task success claim','source_scene':{side:str(path) for side,path in mjcf.items()},'native_home_position_errors_m':home_errors,'targets':results}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    bad=[r for r in results if not r['feasible_screen']]
    print(json.dumps({'targets':len(results),'failed':len(bad),'max_position_error_m':max(r['position_error_m'] for r in results),'failed_examples':bad[:3]}))

if __name__=='__main__':main()
