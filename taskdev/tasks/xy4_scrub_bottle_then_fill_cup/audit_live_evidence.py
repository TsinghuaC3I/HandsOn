"""Summarize native query/contact evidence; reject unexplained overlap pairs."""
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
    if '/Sink/Collider' in a or '/Sink/Collider' in b:
        other = b if '/Sink/Collider' in a else a
        if any(t in other for t in ('/ceramic_mug/', '/Bottle/Colliders/body', '/Brush/Colliders/')):
            return 'intended basin support; depth checked from static contact tensors'
    if oa is None and ob is None:
        return 'source fixed kitchen construction junction'
    if any('/E_body_1/column_collision/' in p for p in (a, b)) and any('/SceneCollision/' in p for p in (a, b)):
        return 'source fixed faucet column mounted into countertop'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'report.json').read_text())
    authored = json.loads((args.run/'before-first-physics.json').read_text())
    assert report.get('completed_diagnostic'), report.get('error')
    checks = [('authored-before-first-step', authored['overlap'], authored['inventory'])]
    all_pairs, unknown, samples, support_depth = {}, [], [], 0.
    for row in report['rows']:
        checks.extend((row['layout']+'-'+phase, row[phase+'_overlap'], row[phase+'_shapes'])
                      for phase in ('reset', 'settled'))
        assert row['recording_origin']['initial_geometry_gate_passed']
        assert row['recording_origin']['final_boundary_physics_gate_passed']
        for state in (row['reset'], row['accepted_origin'], row['settled'],
                      *[s['physics'] for s in row['samples']]):
            assert not state['robot_static_contacts'], state['robot_static_contacts']
            for contact in state['support_contacts']:
                if contact['separations']:
                    support_depth = max(support_depth, -float(np.min(contact['separations'])))
        samples.append({'layout': row['layout'], 'sampled_layout': row['sampled_layout'],
                        'park_position_w': row.get('park_position_w'),
                        'origin_steps': row['recording_origin']['physics_steps'],
                        'peak_body_linear_speed_m_s': row['peak_body_linear_speed'],
                        'peak_body_angular_speed_rad_s': row['peak_body_angular_speed'],
                        'peak_robot_joint_speed_rad_s': row['peak_robot_joint_speed'],
                        'previews': row.get('preview_frames', [])})
    for label, query, inventory in checks:
        assert query['positive_control_passed'] and not query['errors'], label
        assert all(x['visibility'] == 'invisible' for x in inventory), label
        by_path = {x['path']: x for x in inventory}
        for side in ('right', 'left'):
            for finger in range(1, 6):
                for link in range(1, 5):
                    assert any(x['owner'] and x['owner'].endswith(f'/{side}_finger{finger}_link{link}') for x in inventory)
        for a, b in query['pairs']:
            why = reason(a, b, by_path)
            if why is None:
                unknown.append({'query': label, 'pair': [a, b]})
            all_pairs[(a, b)] = why
    seeds = [s['sampled_layout'] for s in samples if s['layout'].startswith('seed_')]
    spans = {name: np.ptp([s[name][:2] for s in seeds], axis=0).tolist() for name in seeds[0]} if seeds else {}
    assert all(min(span) > .005 for span in spans.values())
    result = {'query_count': len(checks), 'shape_count': len(authored['inventory']),
              'layout_count': len(samples), 'seed_position_span_xy_m': spans,
              'max_support_penetration_m': support_depth, 'unapproved_overlaps': unknown,
              'pair_reasons': [{'paths': list(pair), 'reason': why} for pair, why in sorted(all_pairs.items())],
              'samples': samples,
              'scope': 'authored geometry plus live tensor-pose resets/settles, not full manipulation sweeps'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    assert support_depth <= .0005
    assert not unknown, unknown
    print(json.dumps({k: v for k, v in result.items() if k not in ('pair_reasons', 'samples')}))


if __name__ == '__main__':
    main()
