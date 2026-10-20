"""Diagnostic normal-control contact trial; never counts as human teleoperation."""
from __future__ import annotations

import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from gen3_composed_collision import pose_matrix
from validate_lw_native_articulation_live import _json_ready, _reset_and_accept_origin


def contacts(raw, observation=None):
    from pxr import PhysicsSchemaTools
    fields = raw._physx_ground_truth_trace.latest_substep_fields((
        "physx_contact_report_valid", "physx_contact_capture_source",
        "physx_contact_header_count", "physx_contact_header_ids",
        "physx_contact_header_meta", "physx_contact_point_data"))
    rows=[]
    for i in range(int(fields.get("physx_contact_header_count", [0])[0])):
        ids=fields["physx_contact_header_ids"][i]
        meta=fields["physx_contact_header_meta"][i]
        points=fields["physx_contact_point_data"][int(meta[1]):int(meta[1]+meta[2])]
        rows.append({"actor_paths":[str(PhysicsSchemaTools.intToSdfPath(int(x))) for x in ids[:2]],
            "collider_paths":[str(PhysicsSchemaTools.intToSdfPath(int(x))) for x in ids[2:4]],
            "points":[{"position":p[:3],"normal":p[3:6],"separation_m":float(p[9]),"impulse":p[6:9]} for p in points]})
    substeps=[]
    if observation is not None:
        count=int(observation.get('physx_substep_count',[0])[0])
        for step in range(count):
            point_count=int(observation['physx_substep_contact_point_count'][step,0])
            points=observation['physx_substep_contact_point_data'][step,:point_count]
            substeps.append({'valid':bool(observation['physx_substep_contact_report_valid'][step,0]),
                'point_count':point_count,'max_penetration_depth_m':max(0.,float(-points[:,9].min())) if point_count else 0.,
                'points':points,'header_count':int(observation['physx_substep_contact_header_count'][step,0]),
                'header_ids':observation['physx_substep_contact_header_ids'][step,:int(observation['physx_substep_contact_header_count'][step,0])],
                'header_meta':observation['physx_substep_contact_header_meta'][step,:int(observation['physx_substep_contact_header_count'][step,0])]})
            substeps[-1]['actor_paths']=[
                [str(PhysicsSchemaTools.intToSdfPath(int(value))) for value in ids[:2]]
                for ids in substeps[-1]['header_ids']]
        if count!=raw._n_substeps:
            raise RuntimeError(f'Incomplete diagnostic physical contact sampling: {count}/{raw._n_substeps}')
    return {"valid":fields.get("physx_contact_report_valid"),"pairs":rows,"all_substeps":substeps}


def snapshot(raw, output, label):
    from PIL import Image
    if len(raw._cameras)!=3:
        raise RuntimeError("Diagnostic preview did not construct three actual cameras")
    for _ in range(12):
        raw._sim.render()
    frames=raw.render()
    for key,frame in zip(("main","wrist_left","wrist_right"),frames):
        if not isinstance(frame,np.ndarray) or frame.std()<2:
            raise RuntimeError(f"Empty diagnostic camera: {key}")
        Image.fromarray(frame).save(output/f"{label}_{key}.png")
    geometry=raw._gen3_composed_audit.run(raw,before_physics=False)
    (output/f'{label}_composed_geometry.json').write_text(json.dumps(_json_ready(geometry),indent=2)+'\n')


def run_contact_probe(env, raw, output, result, *, pose=(-.28,0.,0.)):
    """Try a slow fingertip approach and friction lift of the existing lid edge."""
    raw.boundary_pose=pose
    result["contact_origin"]=_reset_and_accept_origin(env,raw,0)
    rows=[]
    result["contact_probe"]={"diagnostic_only":True,"human_trial_count":0,"frames":rows,"composed_motion_samples":[]}
    root=pose_matrix(raw._read_lw_truth()["root_pose"])
    side='left' if (root@np.array([-.32,0.,.248,1.]))[1]>1e-6 else 'right'
    side_y=.03 if side=='left' else -.03
    result['contact_probe']['manipulating_side']=side
    result['contact_probe']['robot_joint_names']=list(raw._robots[side].joint_names)
    pose=getattr(raw,f'_{side}_tcp_pose').copy()
    initial_hand=getattr(raw,f'_{side}_hand').copy()
    command=pose.copy()
    last=None
    fixed_support_bodies={row['body'] for row in raw._gen3_composed_audit.inventory.values()
        if row['group'] in {'bin_base','table','ground_support'}}
    def send(target,phase):
        nonlocal command,last
        command=np.asarray(target).copy()
        transition=raw.step({side:np.r_[command,initial_hand]})
        truth=raw._read_lw_truth()
        actual=raw._pose_matrix_to_vector(raw._robot_body_pose_matrix(raw._robots[side],raw._ee_body_indices[side]))
        row={"step":len(rows),"phase":phase,"command":command,"actual_palm":actual,
            "truth":truth,"contacts":contacts(raw,transition[0]),"succeed":bool(transition[4].get('succeed')),
            "terminated":bool(transition[2]),"truncated":bool(transition[3]),"failure_reason":raw._lw_state.failure_reason,
            "robot_joint_positions":raw._robots[side].data.joint_pos[0].detach().cpu().numpy()}
        row['hand_lid_normal_force_n']=sum(abs(float(np.dot(point['normal'],point['impulse'])))/raw._sim_dt
            for pair in row['contacts']['pairs']
            if raw._lw_scene_spec.right_body_path in pair['actor_paths']
            and any(f'/{side.title()}Robot/' in path for path in pair['actor_paths'])
            for point in pair['points'])
        rows.append(row);last=row
        if len(rows)%15==0:
            result['contact_probe']['composed_motion_samples'].append({
                'step':len(rows)-1,'phase':phase,'geometry':raw._gen3_composed_audit.run(raw,before_physics=False)})
        if len(rows)%15==0:
            print(json.dumps({"GEN3_contact_step":len(rows),"phase":phase,"opening_deg":float(-np.degrees(truth['q'][0])),
                "manipulating_side":side,"contact_force_n":row['hand_lid_normal_force_n'],"tcp_position_error_m":float(np.linalg.norm(actual[:3]-command[:3]))}),flush=True)
            (output/'contact_probe_progress.json').write_text(json.dumps(_json_ready(result['contact_probe']),indent=2)+'\n')
        if row['failure_reason'] or row['truncated']:
            raise RuntimeError(f"Contact probe ended unexpectedly: {row['failure_reason']}")
        bad_contact=[p for p in row['contacts']['pairs']
            if any(f'/{side.title()}Robot/' in a for a in p['actor_paths'])
            and any(a in fixed_support_bodies for a in p['actor_paths'])
            and any(np.linalg.norm(pnt['impulse'])>1e-6 for pnt in p['points'])]
        if bad_contact:
            snapshot(raw,output,'unexpected_contact')
            raise RuntimeError('Diagnostic path contacted the fixed bin/support; inspect recorded pairs')
        return row

    def move(target,steps,phase):
        start=command.copy()
        rotations=Rotation.from_quat(np.stack([start[3:],target[3:]])[:,[1,2,3,0]])
        interpolator=Slerp([0.,1.],rotations)
        for t in np.linspace(0.,1.,steps+1)[1:]:
            r=interpolator(t).as_quat()[[3,0,1,2]]
            send(np.r_[start[:3]+t*(target[:3]-start[:3]),r],phase)

    def local_pose(position,rotation):
        m=root.copy();m[:3,3]=root[:3,3]+root[:3,:3]@np.asarray(position)
        m[:3,:3]=root[:3,:3]@rotation.as_matrix()
        return np.r_[m[:3,3],Rotation.from_matrix(m[:3,:3]).as_quat()[[3,0,1,2]]]

    snapshot(raw,output,'contact_reset')
    lift=command.copy();lift[2]=1.22
    move(lift,90,'clearance_lift')
    rotation=Rotation.from_euler('y',90,degrees=True)
    move(local_pose([-.32,side_y,.248],rotation),150,'orient_outside_bin')
    move(local_pose([-.30,side_y,.248],rotation),60,'approach_lid_edge')
    snapshot(raw,output,'before_edge_contact')
    touched=False
    for x in np.linspace(-.30,-.23,211)[1:]:
        row=send(local_pose([x,side_y,.248],rotation),'seek_lid_contact')
        body_contact=[p for p in row['contacts']['pairs'] if any('/trash_bin_base' in a for a in p['actor_paths']) and any(f'/{side.title()}Robot/' in a for a in p['actor_paths'])]
        if body_contact:
            snapshot(raw,output,'unexpected_base_contact')
            raise RuntimeError('The approach touched the bin body before an isolated lid grasp; inspect the retained contact pairs')
        if row['hand_lid_normal_force_n']>.2:
            touched=True
            break
    snapshot(raw,output,'first_edge_contact' if touched else 'missed_edge')
    if not touched:
        raise RuntimeError('No physical lid contact was obtained by the diagnostic approach')
    contact_command=command.copy()
    contact_command[:3]+=root[:3,:3]@np.array([.003,0.,0.])
    move(contact_command,10,'light_contact_preload')
    for _ in range(35):
        send(contact_command,'sustained_lid_contact')
    snapshot(raw,output,'sustained_edge_contact')
    initial=pose_matrix(contact_command)
    hinge=np.eye(4);hinge[:3,3]=[.0005588702871450912,.09082636969920814,.2444057715000594]
    lid_now=pose_matrix(raw._physics_body_pose(raw._lw_scene_spec.right_body_name))
    physical_points=[point for pair in last['contacts']['pairs']
        if raw._lw_scene_spec.right_body_path in pair['actor_paths']
        for point in pair['points'] if np.linalg.norm(point['impulse'])>1e-6]
    if not physical_points:
        raise RuntimeError('The sustained contact ended before a physical lift anchor could be measured')
    anchor=np.asarray(max(physical_points,key=lambda p:np.linalg.norm(p['impulse']))['position'])
    anchor_local=np.linalg.inv(lid_now)@np.r_[anchor,1.]
    command_offset=anchor-contact_command[:3]
    result['contact_probe']['lift_anchor_lid_local']=anchor_local[:3]
    result['contact_probe']['lift_palm_rotation_fraction']=1.
    for degrees in np.linspace(0.,85.,271)[1:]:
        turn=np.eye(4);turn[:3,:3]=Rotation.from_euler('x',-degrees,degrees=True).as_matrix()
        target=root@hinge@turn@np.linalg.inv(hinge)@np.linalg.inv(root)@initial
        row=send(np.r_[target[:3,3],Rotation.from_matrix(target[:3,:3]).as_quat()[[3,0,1,2]]],'contact_driven_lift')
        if row['succeed']:
            break
    snapshot(raw,output,'after_contact_lift')
    success_observed=any(row['succeed'] for row in rows)
    result['contact_probe']['physical_success_observed']=success_observed
    result['contact_probe']['max_opening_deg']=max(float(-np.degrees(r['truth']['q'][0])) for r in rows)
    result['contact_probe']['final_composed_geometry']=raw._gen3_composed_audit.run(raw,before_physics=False)
    (output/'contact_probe_complete.json').write_text(json.dumps(_json_ready(result['contact_probe']))+'\n')
    if not success_observed:
        raise RuntimeError('The physical contact probe did not reach the unchanged 75-degree success threshold')
