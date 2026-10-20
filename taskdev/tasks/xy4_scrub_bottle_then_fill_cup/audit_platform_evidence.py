"""Verify real recorder logs and sealed reset segments; no successful-demo claim."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'handson')]
from handson.data.physx_session_history import load_session_history, load_discarded_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    assert report.get('completed'), report.get('error')
    result = {'scope': 'ordinary platform lifecycle and real discarded state/action segments',
              'successful_episode_verified': False, 'human_teleoperation_verified': False,
              'fresh_process_replay_verified': False, 'postrender_video_verified': False,
              'rounds': [], 'sessions': []}
    for row in report['rounds']:
        assert all(row.get(key) for key in ('initial_reset_complete', 'manual_reset_complete', 'stopped'))
        command = row['session']['command']
        assert command[0].endswith('/isaacsim-5.1-handson/bin/python')
        assert row['session']['isaac_arm_render_mode'] == 'live'
        path = Path(row['log_path'])
        local = args.run / path.name
        log = (local if local.exists() else path).read_text(errors='replace')
        assert 'Traceback (most recent call last)' not in log
        assert '[Error]' not in log and 'CUDA error' not in log
        stable = [line for line in log.splitlines() if '[Recorder] Stable physical frame 0:' in line]
        assert len(stable) >= 2
        assert all('steps=180' in line and 'attempt=1/3' in line and
                   'gate_joint=0.000000rad/s' in line and 'target_error=0.000000000rad' in line
                   for line in stable)
        ready = [line for line in log.splitlines() if 'Native ABC collection ready:' in line]
        assert ready and all('simulated_arms=2' in line and 'deferred_camera_render=True' in line
                             and 'deferred_arm_render=False' in line for line in ready)
        assert 'without placeholder RGB encoding' in log
        result['rounds'].append({'index': row['index'], 'command': command,
            'stable_origin_log_lines': stable, 'native_ready_log_lines': ready,
            'stopped': True, 'second_segment_sealed': row.get('second_segment_sealed', False),
            'observed_wall_control_hz': [float(x) for x in re.findall(r'actual=([\d.]+)Hz', log)]})
    for finalized in sorted((args.run / 'demos').rglob('FINALIZED.json')):
        path = finalized.parent
        history = load_session_history(path)  # Recomputes file hashes and event chain.
        session = {'id': path.name, 'hash_chain_verified': True, 'events': []}
        for index, event in enumerate(history['events']):
            payload = event['payload']
            entry = {'kind': event['kind'], 'ordinal': index}
            if event['kind'] == 'trajectory':
                assert payload['accepted'] is False
                data = load_discarded_trajectory(path, index)
                if data:
                    assert np.isfinite(data['action']).all()
                    assert not np.asarray(data['succeed']).any()
                    for key in ('physx_sim_time', 'next_physx_sim_time',
                                'next_physx_substep_robot_joint_target',
                                'next_physx_robot_joint_target'):
                        assert key in data
                    dt = np.asarray(data['next_physx_sim_time']) - np.asarray(data['physx_sim_time'])
                    assert np.allclose(dt, 1 / 30, atol=1e-5)
                    entry.update(steps=len(data['action']), fields=len(data),
                        action_shape=list(data['action'].shape),
                        dt_seconds_min=float(dt.min()), dt_seconds_max=float(dt.max()),
                        all_success_false=True,
                        hidden_state_in_compact_discard_journal='xy4_hidden_state' in data,
                        field_names=sorted(data))
            session['events'].append(entry)
        result['sessions'].append(session)
    assert len(result['sessions']) == len(result['rounds'])
    assert all(any(e.get('steps', 0) > 0 for e in s['events']) for s in result['sessions'])
    result['passed'] = True
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'rounds': len(result['rounds']),
                      'sessions': len(result['sessions']),
                      'segment_steps': [e['steps'] for s in result['sessions'] for e in s['events'] if 'steps' in e]}))


if __name__ == '__main__':
    main()
