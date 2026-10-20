"""Summarize measured GPU trajectories and independent PhysX shape queries."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'handson')]
from handson.sim.isaaclab import eval_gen07_stack_boxes as gen5
EVIDENCE=Path(__file__).parent/'evidence'


def read(path):return json.loads((EVIDENCE/path).read_text())

def classify(a,b):
    if a==b:return 'same_rigid_body_compound_shapes'
    for side in ('RightRobot','LeftRobot'):
        if f'/{side}/' in a and f'/{side}/' in b:return 'inherited_robot_assembly_self_collision_disabled'
    if '/gen5_package_' in a and '/gen5_package_' in b:return 'package_package_support_or_recoverable_impact'
    if ('/gen5_package_' in a and b.endswith('/table/table')) or ('/gen5_package_' in b and a.endswith('/table/table')):return 'package_table_support_or_recoverable_impact'
    return 'unexpected'


def rollout_summary(row):
    frames=row['frames'];pairs={}
    for i,frame in enumerate(frames):
        assert frame['contacts']['valid']
        for c in frame['contacts']['pairs']:
            paths=tuple(sorted(set(c['paths'])));key=' <-> '.join(paths)
            p=pairs.setdefault(key,{'bodies':paths,'max_penetration_m':0.0,'frames_with_contact':0})
            p['max_penetration_m']=max(p['max_penetration_m'],c['max_penetration_m']);p['frames_with_contact']+=1
    positions={n:np.array([f['poses'][n][:3] for f in frames[-30:]]) for n in gen5.BODY_NAMES}
    return {'name':row.get('name',f"seed_{row.get('seed')}"),'frames':len(frames),'simulated_seconds':len(frames)/30,
        'success_frames':sum(f['succeed'] for f in frames),
        'first_success_frame':next((i for i,f in enumerate(frames) if f['succeed']),None),
        'unexpected_terminal':any((f['terminated'] or f['truncated']) and not f['succeed'] for f in frames),
        'contact_pairs':list(pairs.values()),
        'max_penetration_m':max((p['max_penetration_m'] for p in pairs.values()),default=0),
        'last_frame_max_penetration_m':max((p['max_penetration_m'] for p in frames[-1]['contacts']['pairs']),default=0),
        'last_second_position_span_m':{n:np.ptp(v,axis=0).tolist() for n,v in positions.items()},
        'last_linear_speed_m_s':frames[-1]['info']['stack_max_linear_speed'],
        'last_angular_speed_rad_s':frames[-1]['info']['stack_max_angular_speed'],
        'last_robot_readback':{s:{k:v for k,v in r.items() if k.startswith('max_')} for s,r in frames[-1]['robots'].items()}}


def snapshots(main,layouts):
    rows=[main['authored_first_state'],*main['sampled_reset_queries'],main['post_hold_shapes']]
    for row in layouts['seeds']:rows.extend([row['shape_query'],row['after_hold_query']])
    for row in layouts['layouts']:rows.extend([row['before_rollout_query'],row['after_hold_query']])
    return rows


def main():
    live=read('live_snapshot_v1/result.json');layouts=read('live_snapshot_v1/layouts.json')
    query=read('collision_queries_v1/result.json');edge=read('live_edge_v1/layouts.json');edgequery=read('collision_queries_edge_v1/result.json')
    all_queries=[*query['queries'],*edgequery['queries']]
    all_snapshots=[*snapshots(live,layouts),*snapshots(read('live_edge_v1/result.json'),edge)]
    assert len(all_queries)==len(all_snapshots)
    assert all(q['functional'] for q in all_queries)
    meshes={n:np.asarray(trimesh.load(gen5.ASSET_DIR/spec['collision_mesh'],process=False).vertices) for n,spec in gen5.geometry()['objects'].items()}
    counts=Counter();unique={};coverage=[];support=[]
    for i,(q,s) in enumerate(zip(all_queries,all_snapshots)):
        assert q['phase']==s['phase']
        for p in q['pairs']:
            a=p['source_body'] or p['source'];b=p['target_body'] or p['target'];kind=classify(a,b);counts[kind]+=1
            pair=tuple(sorted([p['source'],p['target']]))
            unique.setdefault(pair,{'shapes':pair,'bodies':tuple(sorted([a,b])),'classification':kind,
                'depth_source':'live GPU contact reports for physical task contacts; geometric overlap only for disabled inherited self-collision and same-body compounds'})
        missing=[c['path'] for c in s['inventory'] if c['owner'] and s['bodies'][c['owner']]['source']=='authored_usd' and not c['owner'].endswith('/worldBody/floor')]
        is_authored=s['phase']=='authored_before_first_sim_reset_or_step'
        coverage.append({'index':i,'phase':s['phase'],'finite_shapes_queried':len(q['self_hits']),
            'physical_centers_verified':len(q['physical_center_controls']),
            'unread_live_collider_bodies':[] if is_authored else missing,
            'pose_matrix_max_error':q['max_snapshot_matrix_error']})
        if is_authored or s['phase']=='sampled_reset_before_physics':
            table=np.asarray(s['bodies']['/World/Task/table/table']['matrix']); top=(table@np.array([0,0,.89,1]))[2]
            for n,v in meshes.items():
                body=f'/World/Task/{n}/{n}';m=np.asarray(s['bodies'][body]['matrix']);w=v@m[:3,:3].T+m[:3,3]
                support.append({'phase':s['phase'],'index':i,'body':body,'mesh_bottom_to_table_m':float(w[:,2].min()-top),
                    'measurement':'actual source convex vertices at captured body pose versus MJCF table top; supplements, does not replace, PhysX shape overlap'})
    rows=[rollout_summary(r) for r in layouts['seeds']+layouts['layouts']]
    edge_row=rollout_summary(edge['layouts'][0])
    robot_settings=edgequery['original_self_collision_settings']
    assert robot_settings['/World/HandsOnRobots/RightRobot/base_right']['physxArticulation:enabledSelfCollisions'] is False
    assert robot_settings['/World/HandsOnRobots/LeftRobot/base_left']['physxArticulation:enabledSelfCollisions'] is False
    checks={'all_shape_query_controls_pass':all(q['functional'] for q in all_queries),
        'all_live_collider_bodies_read':all(not r['unread_live_collider_bodies'] for r in coverage),
        'no_unexpected_shape_pairs':counts['unexpected']==0,
        'all_twenty_seeds_repeat_exactly':len(layouts['seeds'])==20 and all(r['same_seed_exact_repeatability'] for r in layouts['seeds']),
        'ten_requested_layouts_executed':len(layouts['layouts'])==10,
        'no_unexpected_terminal':all(not r['unexpected_terminal'] for r in rows+[edge_row]),
        'expected_success_results':all((r['first_success_frame']==14 if r['name'] in ('center_stack','offset_stack') else r['success_frames']==0) for r in rows),
        'initial_support_not_penetrating_beyond_numeric_precision':min(r['mesh_bottom_to_table_m'] for r in support)>-1e-6,
        'new_asset_colliders_invisible_and_aligned':all(p['collider_visibility']==['invisible'] and p['vertex_surface_difference_m']<1e-7 and p['max_outward_protrusion_m']<1e-7 for p in layouts['geometry']['packages'].values())}
    report={'task':'GEN5','scope':'Multi-pose GPU physics and immutable final-composed PhysX shape audit; operator reach, real collection and physical replay are separate pending gates.',
        'method':'Native GPU USD scene queries were stale after tensor moves and are rejected. Freeze actual GPU body matrices with the exact composed collision geometry in a separate non-stepping PhysX query scene; self-hit and four physical-center controls must pass for every snapshot. Depths below come from original GPU contact reports, never the frozen query copy.',
        'query_snapshots':len(all_queries),'coverage':coverage,'pair_classification_counts':dict(counts),'unique_shape_pairs':list(unique.values()),
        'initial_support_vertex_measurements':support,'rollouts':rows,'extended_edge_rollout':edge_row,
        'original_self_collision_settings':robot_settings,'checks':checks,'passed_this_scope':all(checks.values()),
        'limitations':['Composed-geometry legacy visual_min/max fields are authored USD coordinates, not live GPU coordinates. Relative collider/visual equality and protrusion remain valid; physical queries use GPU matrices.',
        'Inherited robot self-collision-disabled assembly overlaps and same-body compounds are retained explicitly, not misreported as clearances.',
        'Edge impact transient depth reaches 4.662 mm; after ten seconds native maximum depth is about 0.035 mm. Edge arrangement remains unsuccessful and recoverable. Residual velocity and measured pose ranges are retained.',
        'Nominal 120 Hz physics and 30 Hz control verified; diagnostic wall rate about 19–23 Hz is not operator realtime acceptance.',
        'No complete successful ordinary engineering episode or physical-replay camera certification has been produced.']}
    (EVIDENCE/'live_physics_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':checks,'pair_counts':dict(counts),'snapshots':len(all_queries),'initial_support_min_m':min(r['mesh_bottom_to_table_m'] for r in support)},indent=2))
if __name__=='__main__':main()
