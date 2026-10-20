"""Physical edge-pinch feasibility trial. Writes robot controls only.

This diagnostic may fail. It is never a grasp assist or a task success override.
"""
from pathlib import Path
import json
import numpy as np
from collision_snapshot import capture,array


def run(env,directory,observe):
    from pxr import PhysicsSchemaTools
    out=Path(directory);out.mkdir(exist_ok=True)
    robot=env._robots['right'];idx=robot.body_names.index('right_palm_link')
    def palm():return np.r_[array(robot.data.body_pos_w)[0,idx],array(robot.data.body_quat_w)[0,idx]]
    initial_lid=env._physics_body_pose('gen2_lid').copy()
    initial_box=env._physics_body_pose('gen2_lunchbox').copy()
    current=np.r_[palm(),env._right_hand.copy()]
    opened=env._right_hand.copy()
    closed=np.array([1.37048,.422886,.0449085,.041658,.570005,-.023237,.465768,.465371]+[0.]*12)
    dt=env.control_dt;rows=[];phases=[];contacts={};step=0;success_reached=False
    def save():
        (out/'motion.json').write_text(json.dumps(dict(initial_lid=initial_lid.tolist(),initial_box=initial_box.tolist(),phases=phases,contacts=list(contacts.values()),frames=rows),indent=2)+'\n')
    def segment(tag,target,seconds):
        nonlocal current,step,success_reached
        if success_reached:return
        start=current.copy();n=max(1,int(np.ceil(seconds/dt)))
        for i in range(1,n+1):
            alpha=(i/n)**2*(3-2*i/n);action=start+(target-start)*alpha
            action[3:7]/=np.linalg.norm(action[3:7])
            obs,reward,terminated,truncated,info=env.step(action)
            step+=1
            row=dict(step=step,phase=tag,action=action.tolist(),palm=palm().tolist(),
                lid=env._physics_body_pose('gen2_lid').tolist(),box=env._physics_body_pose('gen2_lunchbox').tolist(),
                lid_velocity=env._physics_body_velocity('gen2_lid').tolist(),terminated=bool(terminated),truncated=bool(truncated))
            fields=env._physx_ground_truth_trace._sample_tensor_contact_fields()
            row['contacts']=[]
            for j in range(int(fields['physx_contact_header_count'][0])):
                ids=fields['physx_contact_header_ids'][j];meta=fields['physx_contact_header_meta'][j]
                names=sorted(str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in ids[:2])
                if not any('gen2_' in name for name in names):continue
                points=fields['physx_contact_point_data'][int(meta[1]):int(meta[1]+meta[2])]
                pair=' | '.join(names);entry=contacts.setdefault(pair,dict(pair=names,steps=[],min_separation_m=0.))
                entry['steps'].append(step)
                if len(points):entry['min_separation_m']=min(entry['min_separation_m'],float(points[:,9].min()))
                row['contacts'].append(pair)
            rows.append(row)
            if terminated and env._last_b05_lid_success_metrics.get('b05_success_now'):
                success_reached=True
                break
            if terminated or truncated:raise RuntimeError('Unexpected episode end during physical trial '+tag)
            if not np.all(np.isfinite(np.r_[row['palm'],row['lid'],row['box']])):raise RuntimeError('Nonfinite motion')
        current=target.copy()
        phase=dict(tag=tag,final=observe(env),palm=palm().tolist(),snapshot=capture(env,out/'snapshots',tag),success_metrics=dict(env._last_b05_lid_success_metrics))
        phases.append(phase);save()
        print('[GEN2] Motion',tag,'lid',env._physics_body_pose('gen2_lid')[:3].tolist(),'palm error',float(np.linalg.norm(palm()[:3]-target[:3])),flush=True)
    target=current.copy();segment('synchronize_at_physical_pose',target,1.)
    target[2]+=.14;target[3:7]=[0.6427876096865394, 0.0, 0.766044443118978, 0.0];segment('raise_open_hand',target,2.)
    target[:2]=initial_lid[:2]+[.010,0.];segment('approach_above_edge',target,2.)
    target[2]=initial_lid[2]+.085;segment('lower_open_hand',target,2.)
    target[7:]=closed;segment('close_edge_pinch',target,2.)
    segment('sustain_pinch',target,1.2)
    target[2]+=.14;segment('lift',target,2.)
    segment('sustain_lift',target,1.2)
    lifted=float(env._physics_body_pose('gen2_lid')[2]-initial_lid[2])
    if lifted<.02:
        target[7:]=opened;segment('release_failed_pinch_in_clearance',target,1.5)
        save();return dict(passed=False,reason='Lid did not lift 20 mm',lift_m=lifted,evidence=str(out/'motion.json'))
    target[:2]+=initial_box[:2]-env._physics_body_pose('gen2_lid')[:2]
    desired_height=initial_box[2]+.118+.04
    target[2]+=desired_height-env._physics_body_pose('gen2_lid')[2]
    segment('transport_over_box',target,3.)
    target[2]-=.04;segment('place',target,2.)
    segment('sustain_box_support',target,1.2)
    target[7:]=opened;segment('release',target,2.)
    target[2]+=.18;segment('withdraw',target,2.)
    segment('released_stability',target,1.2)
    success=bool(env._last_b05_lid_success_metrics.get('b05_success_now'))
    save();return dict(passed=success,reason='Physical trial completed',lift_m=lifted,evidence=str(out/'motion.json'))
