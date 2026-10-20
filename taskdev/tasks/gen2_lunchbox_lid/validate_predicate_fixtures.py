"""Synthetic PhysX predicate fixtures. NOT manipulation or collection evidence.

Only this test sets rigid state to construct positive/negative conditions.
Production task code and robot-motion tests contain no such state writes.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation


def run(env,out):
    out=Path(out);rows=[]
    def put(name,pose,velocity=None):
        view=env._task_body_views[name];t=view.get_transforms().clone();v=view.get_velocities().clone()
        t[0,:3]=env._torch.as_tensor(pose[:3],dtype=t.dtype,device=t.device)
        t[0,3:]=env._torch.as_tensor([*pose[4:],pose[3]],dtype=t.dtype,device=t.device)
        v[0]=env._torch.as_tensor(np.zeros(6) if velocity is None else velocity,dtype=v.dtype,device=v.device)
        ids=env._torch.arange(view.count,dtype=env._torch.long,device=env._isaac_device)
        view.set_velocities(v,ids);view.set_transforms(t,ids);env._refresh_object_poses()
    def ready():
        env.reset(seed=77)
        origin=env._physics_body_pose('gen2_lid').copy();box=env._physics_body_pose('gen2_lunchbox').copy()
        box[3:]=[1,0,0,0];put('gen2_lunchbox',box)
        lid=box.copy();lid[2]+=.118;put('gen2_lid',lid)
        env._lid_start_pose=origin;env._lid_was_moved=False;env._success_counter=0
        return box,lid
    def record(tag,expected):
        env._refresh_object_poses();value=env._compute_success_now()
        row=dict(case=tag,expected=expected,actual=bool(value),metrics=dict(env._last_b05_lid_success_metrics),
                 lid=env._physics_body_pose('gen2_lid').tolist(),box=env._physics_body_pose('gen2_lunchbox').tolist())
        rows.append(row);out.write_text(json.dumps(dict(method=__doc__,cases=rows),indent=2)+'\n')
        assert bool(value)==expected,(tag,row)
        print('[GEN2] Predicate fixture',tag,'expected',expected,'actual',value,flush=True)
    box,lid=ready();record('aligned_supported_and_released_fixture',True)
    for tag,edit in [('coverage_below_90_percent',lambda p:p.__setitem__(0,p[0]+.040)),
                     ('height_above_tolerance',lambda p:p.__setitem__(2,p[2]+.030)),
                     ('relative_tilt_8_degrees',lambda p:p.__setitem__(slice(3,7),[np.cos(np.deg2rad(4)),np.sin(np.deg2rad(4)),0,0]))]:
        box,lid=ready();edit(lid);put('gen2_lid',lid);record(tag,False)
    box,lid=ready();env._lid_start_pose=lid.copy();env._lid_was_moved=False;record('aligned_but_unmoved',False)
    box,lid=ready();put('gen2_lid',lid,[.03,0,0,0,0,0]);record('lid_moving_30_mm_per_second',False)
    box,lid=ready();put('gen2_lunchbox',box,[0,0,0,0,0,.4]);record('box_rotating_0p4_radians_per_second',False)
    box,lid=ready();box[3:]=[np.sqrt(.5),np.sqrt(.5),0,0];put('gen2_lunchbox',box);record('box_on_side_fixture',False)
    # Recovery and the consecutive window run actual GPU physics/control steps.
    box,lid=ready()
    for i in range(120):env._step_isaaclab()
    env._success_counter=0;required=env._success_steps_required();window=[]
    for i in range(required):
        obs,reward,term,trunc,info=env.step(None)
        window.append(dict(frame=i+1,counter=int(env._success_counter),terminated=bool(term),truncated=bool(trunc),success_now=bool(env._last_b05_lid_success_metrics.get('b05_success_now'))))
    assert all(not r['terminated'] for r in window[:-1]) and window[-1]['terminated'],window
    env.reset(seed=77)
    hidden=env._compute_observation()['gen2_hidden_state']
    assert env._success_counter==0 and not env._lid_was_moved
    assert not env._compute_success_now()
    result=dict(method=__doc__,cases=rows,live_window=window,required=required,reset_hidden_state=hidden.tolist(),
                passed=True,held_by_actual_grasp='pending human/physical grasp validation',not_a_manipulation_trial=True)
    out.write_text(json.dumps(result,indent=2)+'\n');return result
