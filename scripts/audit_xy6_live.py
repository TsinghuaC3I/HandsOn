"""Audit XY6 native evidence; missing/failed stages cannot count as a pass."""
import argparse
import json
from pathlib import Path
import numpy as np
from xy6_native_geometry import BODY_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--completed-layouts-only',action='store_true')
    args=parser.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    if args.completed_layouts_only:
        report['rows']=[r for r in report['rows'] if 'settled' in r and any(e['label']==r['label']+'_settled' for e in report['events'])]
        labels={r['label'] for r in report['rows']}
        report['events']=[e for e in report['events'] if 'before_first' in e['label'] or any(e['label'].startswith(label+'_') for label in labels)]
        assert report['rows'],'No completed layouts'
    else:
        assert report.get('completed_diagnostic'),report.get('error','Native diagnostic did not complete')
    unknown=[];pairs={};max_depth=0.;summaries=[]
    for event in report['events']:
        q=event['overlap'];assert q['positive_control_passed'] and not q['errors'],event['label']
        unknown.extend(event['unapproved'])
        for r in event['approved']:pairs[tuple(r['paths'])]=r['reason']
        inventory=event['inventory']
        for side in ('left','right'):
            for finger in range(1,6):
                for link in range(1,5):
                    assert any(r['owner'] and r['owner'].endswith(f'/{side}_finger{finger}_link{link}') for r in inventory)
        # Source pot/lid meshes are deliberately both visible and physical.
        # Robot collision-only derivatives must remain invisible.
        assert all(r['visibility']=='invisible' for r in inventory if '/HandsOnRobots/' in r['path'])
    for row in report['rows']:
        assert row['repeatable']
        if 'recording_origin' in row:
            assert row['recording_origin']['initial_geometry_gate_passed']
            assert row['recording_origin']['final_boundary_physics_gate_passed']
        states=[row['reset'],row['settled'],*[x['physics'] for x in row['samples']]]
        if 'accepted_origin' in row:states.append(row['accepted_origin'])
        for state in states:
            assert not state.get('robot_static_contacts'),state.get('robot_static_contacts')
            for contact in state.get('support_static_contacts',[]):
                max_depth=max(max_depth,contact['max_penetration'])
        assert len({x['sensor'] for x in row['settled']['support_static_contacts']})==len(BODY_NAMES),'All five bodies must be supported'
        target_errors={side:max(float(np.max(np.abs(np.asarray(s['robots'][side]['q'])-s['robots'][side]['drive_target']))) for s in states) for side in ('left','right')}
        tracking=[]
        for sample in row['samples']:
            if sample['step']==35:
                tracking=[{'side':side,'palm_error_m':float(np.linalg.norm(np.asarray(sample['physics']['robots'][side]['tcp_actual'])[:3]-np.asarray(sample['physics']['robots'][side]['tcp_command'])[:3])),
                           'joint_target_error_rad':sample['physics']['robots'][side]['max_target_error']} for side in ('left','right')]
        summaries.append({'label':row['label'],'layout':row['sampled_layout'],'max_target_error_rad':target_errors,
                          'hold_tracking':tracking,'origin':row.get('recording_origin'),'frames':row.get('frames',[])})
    seeds=[r['layout'] for r in summaries if r['label'].startswith('seed_')]
    spans={name:np.ptp([r[name][:2] for r in seeds],axis=0).tolist() for name in BODY_NAMES} if len(seeds)>1 else {}
    if len(seeds)>=20:assert all(min(v)>.005 for v in spans.values())
    result={'layout_count':len(summaries),'query_count':len(report['events']),'shape_count':len(report['events'][0]['inventory']),
            'max_support_penetration_m':max_depth,'unapproved_overlaps':unknown,'translation_spans_m':spans,
            'pair_reasons':[{'paths':list(k),'reason':v} for k,v in sorted(pairs.items())],
            'rows':summaries,'completed_layouts_only':args.completed_layouts_only,'source_run_error':report.get('error'),
            'operator_verified':False,'full_task_trajectories_verified':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    assert not unknown,unknown
    assert max_depth<=.0005,max_depth
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pair_reasons')}))

if __name__=='__main__':
    main()
