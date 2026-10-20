#!/usr/bin/env python3
"""Grade recorded GEN4 live resets; never substitutes for operation/teleop tests."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    p=args.directory
    matrix=json.loads((p/'matrix.json').read_text())
    cases={c['name']:json.loads((p/c['report']).read_text()) for c in matrix['cases']}
    expected_shapes=matrix['expected_shape_count']
    failures=[]
    def require(condition,message):
        if not condition:failures.append(message)
    require(matrix.get('completed') and len(cases)==29,'The complete 20-seed + 8-boundary + repeated-seed matrix is required')
    phases=[r for d in cases.values() for r in d['phases']]
    require(all(r['shape_count']==expected_shapes and r['callback_count']>0 and not r['unapproved_pairs'] and not r['query_errors'] for r in phases),'Final shape query failed or lost geometry coverage')
    require(all(not d['exceptions'] for d in cases.values()),'A live case raised an exception')
    quaternion_norm_errors=[]
    for name,d in cases.items():
        require(np.isclose(d['timing']['measured_idle_duration_s'],2,atol=1e-5),f'{name}: invalid physical-time window')
        require(len(d['motion'])==60 and not any(m['terminated'] or m['truncated'] for m in d['motion']),f'{name}: incomplete idle window')
        require(d.get('reset_hidden_state')==[0]*10,f'{name}: stale task progress')
        for body,row in d['reset_layout'].items():
            physical=np.array(d['reset_physical_poses'][body])
            require(np.max(np.abs(physical[:3]-[*row['xy'],row['z']]))<=2e-7,f'{name}/{body}: requested position was not written')
            angle=np.deg2rad(row['yaw_degrees'])/2
            quat=np.array([np.cos(angle),0,0,np.sin(angle)])
            # PhysX float32 readback can deviate a few ULPs from unit length.
            # Compare the represented rotation, retaining and bounding norm
            # error separately; raw poses remain in every input report.
            norm=float(np.linalg.norm(physical[3:]))
            quaternion_norm_errors.append(abs(norm-1))
            require(abs(norm-1)<=1e-6,f'{name}/{body}: invalid quaternion norm')
            normalized=physical[3:]/norm
            require(min(np.max(np.abs(normalized-quat)),np.max(np.abs(normalized+quat)))<=2e-7,f'{name}/{body}: requested absolute orientation was not written')
    contacts=[c for d in cases.values() for c in d['contacts']]
    depth=max((c.get('penetration_depth_m',0) for c in contacts),default=float('inf'))
    require(depth<=.002,'Intended support penetration exceeds 2 mm')
    require(all(m['contact_signature']['valid']==1 and m['contact_signature']['pair_count']>=11 for d in cases.values() for m in d['motion']),'Incomplete live contact readback')
    sampling={}
    nominal=[cases[f'seed_{i:02}'] for i in range(20)]
    for body in nominal[0]['reset_layout']:
        xyz=np.array([d['reset_physical_poses'][body][:3] for d in nominal])
        span=np.ptp(xyz,axis=0)
        sampling[body]={'min_xyz_m':xyz.min(0).tolist(),'max_xyz_m':xyz.max(0).tolist(),'span_xyz_m':span.tolist()}
        if body!='gen4_tray':require(np.all(span[:2]>0),f'{body}: no sampled XY translation spread')
    original,repeat=cases['seed_00'],cases['seed_00_repeat']
    require(original['reset_layout']==repeat['reset_layout'],'Same-seed declared layout did not reproduce exactly')
    repeat_errors={}
    for body,pose in original['reset_physical_poses'].items():
        repeat_errors[body]=float(np.max(np.abs(np.array(pose)-repeat['reset_physical_poses'][body])))
    for side,state in original['reset_robot_states'].items():
        for key,values in state.items():
            repeat_errors[side+'/'+key]=float(np.max(np.abs(np.array(values)-repeat['reset_robot_states'][side][key])))
    require(max(repeat_errors.values())<=1e-6,'Same-seed physical/robot reset differs by more than numerical readback tolerance')
    stability={}
    for name,d in cases.items():
        bodies={}
        for body in d['motion'][0]['poses']:
            xyz=np.array([m['poses'][body][:3] for m in d['motion']])
            velocity=np.array([m['velocities'][body] for m in d['motion'][-30:]])
            row={'xyz_span_m':np.ptp(xyz,axis=0).tolist(),
                'last_second_max_linear_speed_m_s':float(np.linalg.norm(velocity[:,:3],axis=1).max()),
                'last_second_max_angular_speed_rad_s':float(np.linalg.norm(velocity[:,3:],axis=1).max())}
            bodies[body]=row
            require(np.ptp(xyz,axis=0).max()<=.01,f'{name}/{body}: more than 10 mm idle drift')
            require(row['last_second_max_linear_speed_m_s']<=.08 and row['last_second_max_angular_speed_rad_s']<=1.8,f'{name}/{body}: persistent idle motion')
        robot_vel=max(r['joint_velocity_max'] for m in d['motion'][-30:] for r in m['robots'].values())
        robot_err=max(r['joint_target_error_max'] for m in d['motion'][-30:] for r in m['robots'].values())
        require(robot_vel<=.01 and robot_err<=.01,f'{name}: robot did not hold its home')
        stability[name]={'bodies':bodies,'last_second_max_robot_speed_rad_s':robot_vel,'last_second_max_robot_target_error_rad':robot_err}
    log=(p/'child.log').read_text()
    errors=[line for line in log.splitlines() if '[Error]' in line or 'Traceback' in line]
    require(not errors,'Actual child log contains errors')
    result={'task':'GEN4','scope':'live idle/reset matrix only; operation contact, cameras, collection and replay remain separate',
        'cases':len(cases),'shape_query_phases':len(phases),'shape_count':expected_shapes,'unapproved_overlaps':sum(len(x['unapproved_pairs']) for x in phases),
        'contact_pair_samples':len(contacts),'max_support_penetration_m':depth,'sampled_positions':sampling,
        'max_float32_quaternion_norm_error':max(quaternion_norm_errors),
        'same_seed_readback_max_abs_error':repeat_errors,'stability':stability,'child_log_errors':errors,
        'evidence_sha256':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'matrix.json',p/'child.log',p/'source_manifest.json',*[p/c['report'] for c in matrix['cases']]]},
        'failures':failures,'passed':not failures}
    (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['cases','shape_query_phases','unapproved_overlaps','contact_pair_samples','max_support_penetration_m','failures','passed']},indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__=='__main__':main()
