"""Exact shared D08-2 child: first GPU reset, actual state, seeded 120 Hz holds."""
from pathlib import Path
import sys,os,json,traceback,importlib.metadata,time
import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'handson'))
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON']='0'
os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER']='1'
os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY']='0'
os.environ['HANDSON_ISAACLAB_ARM_IK_BACKEND']='haorui'

def arr(v):
    if hasattr(v,'detach'):v=v.detach().cpu().numpy()
    return np.asarray(v)
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
report={'task':'d08_2_realistic_toast','isaacsim':importlib.metadata.version('isaacsim'),'seeds':{},'passed':False}
def save():
    (OUT/'priority_zero_live.json').write_text(json.dumps(safe(report),indent=2,ensure_ascii=False))

from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,enable_cameras=False,device='cuda:0')
app=launcher.app
from isaaclab.sim import SimulationContext
from handson.sim.isaaclab.d_long_horizon_env import HandsOnDLongHorizonIsaacLabEnv
from handson.sim.isaaclab.priority_zero_robot_audit import initial_robot_state_report,RobotStabilityAudit,priority_zero_runtime_contract

env=HandsOnDLongHorizonIsaacLabEnv.__new__(HandsOnDLongHorizonIsaacLabEnv)
original_reset=SimulationContext.reset
reset_counter=0
def first_reset(context,*args,**kwargs):
    global reset_counter
    reset_counter+=1
    result=original_reset(context,*args,**kwargs)
    if reset_counter==1:
        snapshot={}
        for side,robot in env._robots.items():
            ids=robot.find_joints([f'joint{i}_{side}(_1)?' for i in range(1,8)])[0]
            palm=robot.find_bodies(f'{side}_palm_link')[0][0]
            snapshot[side]={'root':arr(robot.data.root_state_w)[0],
                'arm_q':arr(robot.data.joint_pos)[0,ids],'arm_qd':arr(robot.data.joint_vel)[0,ids],
                'palm_position':arr(robot.data.body_pos_w)[0,palm],'palm_quaternion':arr(robot.data.body_quat_w)[0,palm]}
        report['first_simulation_reset']=snapshot;save()
        assert all(np.all(np.isfinite(x)) for row in snapshot.values() for x in row.values())
        print('D08_2_FIRST_GPU_RESET_RETURNED',flush=True)
    return result
SimulationContext.reset=first_reset

try:
    env.__init__('d08_2_realistic_toast',seed=8021,randomize=True,image_obs=False,headless=True,
        render_mode='rgb_array',hz=30,physics_dt=1/120,control_dt=1/30,physics_backend='physx',isaac_device='cuda:0',
        existing_simulation_app=app,existing_app_launcher=launcher)
    report['runtime']=priority_zero_runtime_contract(env,requested_device='cuda:0')
    report['construction_initial']=getattr(env,'_d08_2_initial_state_report',{})
    body_names=('d08_toaster','d08_toaster_button','d08_bread_basket','d08_plate','d08_bread_0','d08_bread_1')
    def body_state():
        return {n:{'pose':env._pose(n).copy(),'velocity':env._physics_body_velocity(n)} for n in body_names}
    physics=next(iter(env._robots.values()))._physics_sim_view
    contact_views={}
    for name in ('d08_bread_0','d08_bread_1'):
        prim=env._find_rigid_body_prim(name)
        support=env._find_rigid_body_prim('d08_bread_basket')
        contact_views[name]=physics.create_rigid_contact_view(str(prim.GetPath()),filter_patterns=[str(support.GetPath())],max_contact_data_count=32)
    def penetration():
        worst=0.;force=0.
        for view in contact_views.values():
            f,points,normals,d,counts,starts=view.get_contact_data(env._physics_dt)
            fs=arr(f).reshape(-1);ds=arr(d).reshape(-1)
            for count,start in zip(arr(counts).reshape(-1),arr(starts).reshape(-1)):
                if count:
                    span=slice(int(start),int(start+count));worst=max(worst,float(max(0,-ds[span].min())));force+=float(np.abs(fs[span]).sum())
        return worst,force
    repeated=None
    for seed in (8021,8021,8022,8023):
        obs,info=env.reset(seed=seed)
        initial_pass,initial=initial_robot_state_report(env)
        row={'initial_robot_passed':initial_pass,'initial_robot':initial,'initial_bodies':body_state(),'steps':0}
        assert initial_pass,'Robot actual home failed'
        baseline={n:env._pose(n).copy() for n in body_names}
        if repeated is None:repeated={n:p.copy() for n,p in baseline.items()}
        elif seed==8021:
            row['same_seed_pose_error_m']=max(float(np.linalg.norm(baseline[n][:3]-repeated[n][:3])) for n in body_names)
            assert row['same_seed_pose_error_m']<.003
        robot_audit=RobotStabilityAudit.start(env)
        max_drift={n:0. for n in body_names};max_speed=0.;max_spin=0.;max_pen=0.
        original_step=env._step_isaaclab
        def audit_step(*args,**kwargs):
            global max_speed,max_spin,max_pen
            value=original_step(*args,**kwargs);robot_audit.sample(env)
            env._refresh_object_poses()
            for n in body_names:
                max_drift[n]=max(max_drift[n],float(np.linalg.norm(env._pose(n)[:3]-baseline[n][:3])))
            for n in ('d08_bread_0','d08_bread_1'):
                v=env._physics_body_velocity(n);assert v is not None and np.all(np.isfinite(v))
                max_speed=max(max_speed,float(np.linalg.norm(v[:3])));max_spin=max(max_spin,float(np.linalg.norm(v[3:])))
            pen,_=penetration();max_pen=max(max_pen,pen)
            row['steps']+=1
            return value
        env._step_isaaclab=audit_step
        for i in range(300):
            # Zero action means no new command: retain reset's canonical arm/hand
            # targets. An all-zero command packet actively opens the Wuji thumb.
            obs,reward,terminated,truncated,info=env.step_with_current_joint_targets()
            assert not terminated and not truncated,f'spontaneous terminal at {i}: {info}'
        env._step_isaaclab=original_step
        row.update(robot_stability=robot_audit.to_dict(),max_drift_m=max_drift,max_bread_speed_m_s=max_speed,max_bread_spin_rad_s=max_spin,max_penetration_m=max_pen,final_bodies=body_state())
        # Evaluate task-specific declared thresholds alongside the shared strict robot report.
        row['shared_robot_stability_passed']=robot_audit.passed(physics_steps=row['steps'],physics_hz=120)
        row['passed']=bool(row['shared_robot_stability_passed'] and robot_audit.base_max_drift_m<=.001 and robot_audit.palm_max_drift_m<=.003 and robot_audit.arm_max_drift_rad<=.01 and max(max_drift.values())<=.003 and max_speed<=.01 and max_spin<=.10 and max_pen<=.0005)
        report['seeds'][str(seed)+('_repeat' if seed==8021 and '8021' in report['seeds'] else '')]=row;save()
        print('D08_2_SEED',seed,'PASS',row['passed'],'DRIFT',max_drift,'SPEED',max_speed,'PENETRATION',max_pen,flush=True)
        assert row['passed'],'Zero-action drift/speed/penetration failed'
    report['passed']=True;save()
    print('D08_2_PRIORITY_ZERO_PASS',flush=True)
except Exception:
    report['error']=traceback.format_exc();save();print(report['error'],flush=True)
    raise
finally:
    # Kit may retain ROS/RTX worker threads on failure; this disposable evidence child owns no recordings.
    save()
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0 if report['passed'] else 1)
