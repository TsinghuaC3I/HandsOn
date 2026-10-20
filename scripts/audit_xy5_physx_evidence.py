"""Audit full native collider coverage, intended pairs and measured contact depths."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

def reason(a, b, inventory):
    ra, rb = inventory[a], inventory[b]
    oa, ob = ra['owner'], rb['owner']
    if '/HandsOnRobots/' in a or '/HandsOnRobots/' in b:
        for side in ('right', 'left'):
            root = f'/World/HandsOnRobots/{side.capitalize()}Robot/'
            if not (a.startswith(root) and b.startswith(root)):
                continue
            na, nb = oa.rsplit('/', 1)[-1], ob.rsplit('/', 1)[-1]
            if oa == ob:
                return 'source compound fingertip on one rigid link'
            names = {na, nb}
            if names in ({f'attachment_{side}', f'{side}_palm_link'},
                         {f'link7_{side}', f'{side}_palm_link'}):
                return 'source wrist/palm mounting junction'
            fingers = [re.fullmatch(rf'{side}_finger(\d)_link(\d)', n) for n in (na, nb)]
            if all(fingers) and fingers[0][1] == fingers[1][1] and abs(int(fingers[0][2])-int(fingers[1][2])) == 1:
                return 'source adjacent finger joint shells'
            if f'{side}_palm_link' in names and any(m and int(m[2]) <= 2 for m in fingers):
                return 'source proximal finger/palm housing junction'
            links = [re.fullmatch(rf'link(\d)_{side}', n) for n in (na, nb)]
            if all(links) and (abs(int(links[0][1])-int(links[1][1])) == 1 or {links[0][1], links[1][1]} == {'5', '7'}):
                return 'source arm joint/wrist housing junction'
        return None
    if oa and oa == ob:
        return 'source compound colliders on one rigid body'
    table = '/World/Task/table/table'
    if oa == table or ob == table:
        other = ob if oa == table else oa
        if other and other.rsplit('/', 1)[-1] in {'pitcher_lid', '_019_pitcher_base', 'Box_lid', 'Clear_Box'}:
            return 'intended tabletop support; native contact depth checked separately'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'report.json').read_text())
    assert report.get('completed_diagnostic'), report.get('error')
    unknown, pair_reasons, summaries = [], {}, []
    max_depth = 0.
    for event in report['events']:
        label, inventory, query = event['label'], event['inventory'], event['overlap']
        assert query['positive_control_passed'] and not query['errors'], label
        assert all(x['visibility']=='invisible' for x in inventory), label
        by_path={x['path']:x for x in inventory}
        for side in ('right','left'):
            for finger in range(1,6):
                for link in range(1,5):
                    assert any(x['owner'] and x['owner'].endswith(f'/{side}_finger{finger}_link{link}') for x in inventory)
        for a,b in query['pairs']:
            why=reason(a,b,by_path)
            pair_reasons[(a,b)]=why
            if why is None:unknown.append({'query':label,'paths':[a,b]})
            if 'before_first' in label and why and 'tabletop support' in why:
                table, obj=(by_path[a],by_path[b]) if by_path[a]['owner'].endswith('/table') else (by_path[b],by_path[a])
                assert table['bounds'][1][2]-obj['bounds'][0][2] <= .0005
    for row in report['rows']:
        assert row['repeatable']
        if 'recording_origin' in row:
            assert row['recording_origin']['initial_geometry_gate_passed']
            assert row['recording_origin']['final_boundary_physics_gate_passed']
        states=[row['reset'],row['settled'],*[s['physics'] for s in row['samples']]]
        if 'accepted_origin' in row:states.append(row['accepted_origin'])
        for state in states:
            assert not state.get('robot_static_contacts'), state.get('robot_static_contacts')
            for contact in state.get('support_static_contacts',[]):
                max_depth=max(max_depth,contact['max_penetration'])
        assert len({x['sensor'] for x in row['settled']['support_static_contacts']})==4, 'Every task body must have table support'
        def joint_error(state,side):
            r=state['robots'][side]
            return float(np.max(np.abs(np.asarray(r['q'])-r['drive_target'])))
        summaries.append({'label':row['label'],'sampled_layout':row['sampled_layout'],
                          'origin':row.get('recording_origin'),
                          'max_robot_target_error_rad':{side:max(joint_error(x,side) for x in states) for side in ('right','left')},
                          'frames':row.get('frames',[])})
    seeds=[x['sampled_layout'] for x in summaries if x['label'].startswith('seed_')]
    spans={name:np.ptp([x[name][:2] for x in seeds],axis=0).tolist() for name in seeds[0]} if seeds else {}
    if seeds:assert len(seeds)>=12 and all(min(v)>.005 for v in spans.values())
    result={'layout_count':len(summaries),'query_count':len(report['events']),
            'shape_count':len(report['events'][0]['inventory']),'max_support_penetration_m':max_depth,
            'unapproved_overlaps':unknown,'seed_xy_spans_m':spans,
            'pair_reasons':[{'paths':list(k),'reason':v} for k,v in sorted(pair_reasons.items())],
            'samples':summaries,'scope':'native geometry, reset/settle and short tracking; full human manipulation pending'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    assert max_depth<=.0005, max_depth
    assert not unknown, unknown
    print(json.dumps({k:v for k,v in result.items() if k not in ('pair_reasons','samples')}))

if __name__=='__main__':
    main()
