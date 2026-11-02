"""Load checked replay data without importing Isaac Sim or executing source files."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


EPISODES = Path(__file__).with_name('episodes')
PACKAGE = Path(__file__).resolve().parents[1]
FILES = {'commands.npz', 'initial_snapshot.npz', 'task_state.npz',
         'physics_schema.json', 'camera_jitter.json', 'appearance.json'}
BUNDLES = {
    'e015401': ('t34_roll_ball_through_gate', None, 'handson.episode_replay.v1'),
    'e013673': ('t06_fit_device_cover', 'stabilize_body', 'handson.episode_replay.v2'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    for name, value in arrays.items():
        require(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(),
                f'Nonfinite or nonnumeric replay data: {name}')
    return arrays


@dataclass
class Episode:
    manifest: dict
    actions: np.ndarray
    states: np.ndarray
    timestamps: np.ndarray
    task_state: dict
    snapshot: dict
    physics_schema: dict
    camera_jitter: dict
    appearance: dict
    joint_targets: dict | None = None
    reference: dict | None = None

    def require_control(self, control):
        require(control in {'eef52', 'recorded_joint_targets'}, 'Unknown replay control mode')
        require(control != 'recorded_joint_targets' or self.joint_targets is not None,
                'This episode has no bundled recorded joint targets')


def load_episode(episode_id, *, root=EPISODES):
    require(episode_id in BUNDLES, 'No bundled initialization for this episode')
    folder = Path(root) / episode_id
    manifest = json.loads((folder / 'manifest.json').read_text())
    task, variant, version = BUNDLES[episode_id]
    require(manifest['format'] == version, 'Unsupported replay bundle format')
    require(manifest['episode'] == episode_id and manifest['task'] == task
            and manifest.get('paper_variant') == variant,
            'Episode/task identity mismatch')
    extra = {'joint_targets.npz', 'reference.npz'} if version.endswith('.v2') else set()
    require(set(manifest['files']) == FILES | extra, 'Incomplete replay bundle file contract')
    if variant is not None:
        from ..evaluation.paper_variants import load
        require(load(task, variant)['source_commit'] == manifest['recorded_source_commit'],
                'Replay source and environment cohort disagree')
    for name, expected in manifest['files'].items():
        require(hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected,
                f'Replay payload checksum mismatch: {name}')
    for name, expected in manifest['runtime_files'].items():
        path = (PACKAGE / name).resolve()
        require(PACKAGE in path.parents, 'Runtime identity path escapes package')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'Unvalidated runtime implementation: {name}')
    data = _arrays(folder / 'commands.npz')
    actions, states, timestamps = data['actions'], data['states'], data['timestamp']
    count = manifest['action_count']
    require(actions.shape == states.shape == (count, 52), 'Expected aligned (N,52) state/actions')
    require(timestamps.shape == (count,), 'Invalid replay timestamps')
    # LeRobot stores these timestamps as float32. Differencing rounded values
    # can exceed 1e-6 at 24 seconds even for an exact exported 30 Hz grid.
    expected_time = (np.arange(count, dtype=np.float64) / 30).astype(np.float32)
    require(np.array_equal(timestamps, expected_time),
            'Recorded timestamps must match the exported 30 Hz grid')
    task_state = _arrays(folder / 'task_state.npz')
    require({key: list(value.shape) for key, value in task_state.items()} == manifest['task_state_shapes'],
            'Historical task-state layout mismatch')
    snapshot = _arrays(folder / 'initial_snapshot.npz')
    require({key: list(value.shape) for key, value in snapshot.items()} == manifest['snapshot_shapes'],
            'Initial physical-state layout mismatch')
    require(np.array_equal(snapshot['physx_control_step'], [0]), 'Snapshot must precede action zero')
    schema = json.loads((folder / 'physics_schema.json').read_text())
    require(abs(schema['physics_dt'] - 1 / 120) < 1e-12 and schema['n_substeps'] == 4,
            'Unsupported recorded physics clock')
    from handson.eef52 import encode_measured
    measured = encode_measured(snapshot['physx_tcp_state'][:, :7],
                               snapshot['physx_robot_joint_position'], schema['robot_joint_names'])
    require(np.allclose(measured, states[0], atol=1e-6, rtol=0),
            'Initial snapshot and public episode do not match')
    targets = reference = None
    if extra:
        targets = _arrays(folder / 'joint_targets.npz')
        expected = {'next_physx_substep_count', 'next_physx_substep_robot_joint_target',
                    'next_physx_substep_robot_joint_target_valid'}
        require(set(targets) == expected, 'Invalid substep target fields')
        require(targets['next_physx_substep_count'].shape == (count, 1)
                and np.all(targets['next_physx_substep_count'] == schema['n_substeps']),
                'Recorded target substep count differs from the physics clock')
        values = targets['next_physx_substep_robot_joint_target']
        valid = targets['next_physx_substep_robot_joint_target_valid']
        require(values.shape == valid.shape == (count, 4, 2, 27)
                and np.isin(valid, [0, 1]).all() and valid.any(axis=(2, 3)).all(),
                'Invalid recorded joint target layout or validity mask')
        reference = _arrays(folder / 'reference.npz')
        shapes = {'next_physx_tcp_state': (count, *snapshot['physx_tcp_state'].shape),
                  'next_physx_task_rigid_body_state': (count, *snapshot['physx_task_rigid_body_state'].shape)}
        require({k: v.shape for k, v in reference.items()} == shapes,
                'Historical state reference layout differs from the initialization')
    return Episode(manifest, actions, states, timestamps, task_state, snapshot, schema,
                   json.loads((folder / 'camera_jitter.json').read_text()),
                   json.loads((folder / 'appearance.json').read_text()), targets, reference)


def restore_initial_state(env, episode):
    """One pre-action restore; no per-step scene writes or source-code patching."""
    require(env.task_name == episode.manifest['task'], 'Replay environment task mismatch')
    require(getattr(env, 'paper_variant', None) == episode.manifest.get('paper_variant'),
            'Replay environment source cohort mismatch')
    require(env.env_step == 0, 'Cannot restore initial state after action execution')
    require(not getattr(env, '_episode_replay_restore_started', False),
            'This environment already attempted a replay-state restore')
    require(abs(env._physics_dt - 1 / 120) < 1e-12 and env._n_substeps == 4,
            'Replay requires 120 Hz physics and four substeps per action')
    env._episode_replay_restore_started = True
    env.restore_recorded_initial_state(episode.task_state)
    trace = env._physx_ground_truth_trace
    counts = trace.restore(episode.snapshot, {'physx': episode.physics_schema}, sync=True)
    if episode.manifest['put_task_bodies_to_sleep']:
        trace.put_task_bodies_to_sleep()
    actual = trace.snapshot()
    errors = {}
    for key in ('physx_robot_joint_position', 'physx_robot_joint_velocity',
                'physx_robot_root_state', 'physx_tcp_state', 'physx_task_rigid_body_state'):
        expected = episode.snapshot[key]
        value = np.asarray(actual[key])
        require(value.shape == expected.shape and np.isfinite(value).all(),
                f'Invalid initial-state readback: {key}')
        errors[key] = float(np.max(np.abs(value - expected)))
        require(errors[key] <= 1e-5, f'Initial-state restore mismatch: {key} ({errors[key]})')
    return dict(counts=counts, max_abs_errors=errors, source='bundled historical initial state',
                restore_count=1, writes_after_action_start=False)
