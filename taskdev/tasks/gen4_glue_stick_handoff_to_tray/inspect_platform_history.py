#!/usr/bin/env python3
"""Inspect actual ordinary GEN4 reset journals; never manufacture an episode."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from handson.data.physx_session_history import load_session_history, load_discarded_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    directory = args.directory.resolve()
    platform = json.loads((directory / 'platform.json').read_text())
    journal_root = root / '.gen4-local/platform-episodes/.physical_replay_pending/.physx_session_history'
    headers = {json.loads(p.read_text())['session_id']: p.parent
               for p in journal_root.glob('*/session.json')}
    failures, sessions = [], []
    accepted_episodes = 0
    def require(condition, message):
        if not condition:
            failures.append(message)
    require(platform['completed'] and not platform['exceptions'], 'Ordinary platform cycles incomplete')
    require(len(platform['cycles']) == 3, 'Three ordinary load/start/stop cycles required')
    for cycle in platform['cycles']:
        sid = cycle['start']['session']['session_id']
        path = headers[sid]
        loaded = load_session_history(path)  # Verifies the actual immutable hash chain.
        row = {'session_id': sid, 'path': str(path.relative_to(root)),
               'finalized': (path / 'FINALIZED.json').is_file(), 'origins': [], 'segments': []}
        require(row['finalized'], f'{sid}: journal not finalized')
        for event in loaded['events']:
            payload = event['payload']
            if event['kind'] == 'trajectory' and payload['accepted']:
                accepted_episodes += 1
            if event['kind'] == 'reset_attempt':
                origin = payload.get('stable_origin') or {}
                origin_keys = (
                    'stable_steps_observed', 'stable_steps_required', 'physics_steps', 'physics_dt',
                    'final_max_task_linear_velocity', 'final_max_task_angular_velocity',
                    'final_max_robot_joint_velocity', 'final_robot_position_target_error',
                    'reset_max_translation_drift_m', 'reset_max_rotation_drift_rad',
                    'pose_translation_limit', 'pose_rotation_limit', 'contact_penetration_limit',
                    'post_settle_sleep', 'trajectory_per_frame_state_writes',
                    'final_boundary_physics_gate_passed', 'final_contact_summary',
                )
                row['origins'].append({'accepted': payload['accepted'], 'seed': payload['seed'],
                    'generation': payload['scene_generation'],
                    **{k: origin.get(k) for k in origin_keys}})
                require(payload['accepted'], f'{sid}: origin rejected')
                require(origin.get('final_boundary_physics_gate_passed') is True, f'{sid}: final origin gate failed')
            elif event['kind'] == 'trajectory' and not payload['accepted'] and payload['step_count']:
                data = load_discarded_trajectory(path, event['ordinal'])
                segment_dir = path / event['event_directory']
                schema = json.loads((segment_dir / 'recording_schema.json').read_text())
                names = ('action', 'physx_robot_joint_target', 'next_physx_robot_joint_target',
                         'physx_substep_robot_joint_target', 'next_physx_substep_robot_joint_target',
                         'gen4_handoff_state', 'next_gen4_handoff_state')
                shapes = {name: list(data[name].shape) for name in names if name in data}
                require(set(shapes) == set(names), f'{sid}: missing recorded state/action/target field')
                require(shapes.get('action') == [payload['step_count'], 54], f'{sid}: action shape mismatch')
                for name in ('gen4_handoff_state', 'next_gen4_handoff_state'):
                    require(shapes.get(name) == [payload['step_count'], 10], f'{sid}: hidden state shape mismatch')
                    if name in data:
                        require(np.isfinite(data[name]).all(), f'{sid}: nonfinite hidden state')
                require(not any('image' in name for name in data), f'{sid}: unexpected online camera field')
                row['segments'].append({'ordinal': event['ordinal'], 'reason': payload['reason'],
                    'steps': payload['step_count'], 'shapes': shapes,
                    'task_state_schema': schema.get('environment_schema', {}).get('task_state'),
                    'schema_sha256': hashlib.sha256((segment_dir / 'recording_schema.json').read_bytes()).hexdigest(),
                    'hidden_current_next_equal': bool(np.array_equal(data.get('gen4_handoff_state'), data.get('next_gen4_handoff_state'))),
                    'observation': 'Idle commands are expected to keep the stage at zero; changing-state storage is separately regression tested.'})
        sessions.append(row)
    require(platform['cycles'][0].get('manual_reset_completed'), 'Manual reset completion was not observed')
    require([x['generation'] for x in sessions[0]['origins']] == [0, 1], 'Manual reset did not create a fresh scene')
    require(any(x['segments'] for x in sessions), 'No actual discarded trajectory to inspect')
    result = {'task': 'GEN4', 'scope': 'ordinary start/stop, manual reset and discarded trajectory storage',
              'sessions': sessions, 'successful_saved_episode_count': accepted_episodes,
              'successful_episode_videos_and_replay': 'pending actual physical completion',
              'failures': failures, 'passed': not failures}
    (directory / 'history_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('scope', 'successful_saved_episode_count', 'failures', 'passed')}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
