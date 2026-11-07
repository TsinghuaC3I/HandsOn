#!/usr/bin/env python3
"""Select exactly 200 complete source episodes per task, without changing them."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import zipfile


def inspect_episode(args):
    root, item = args
    task_id = str(item['task_id']).zfill(2)
    episode_id = item['prefix'].rstrip('/').split('/')[-1]
    relative = f'tasks/{task_id}/{episode_id}'
    path = root / relative
    meta_bytes = (path / 'collection_meta.json').read_bytes()
    meta = json.loads(meta_bytes)
    duration = Decimal(str(meta['duration_seconds']))
    steps, fps = int(meta['step_count']), Decimal(str(meta['data_fps']))
    assert duration.is_finite() and duration > 0 and steps > 0 and fps > 0, relative
    assert abs(duration - Decimal(steps) / fps) <= Decimal('0.000002'), relative
    assert meta['hand_type'] == 'wuji2', relative
    assert meta['sim_backend'] == 'isaacsim_newton', relative
    with zipfile.ZipFile(path / 'replay.zarr.zip') as archive:
        action_shape = json.loads(archive.read('data/action/.zarray'))['shape']
        assert action_shape[0] == steps, (relative, steps, action_shape)
    return {
        'task_id': task_id, 'task_name': meta['task_name'],
        'episode_id': episode_id, 'source_relative_path': relative,
        'duration_seconds': float(duration), 'step_count': steps, 'fps': float(fps),
        'dataset_generation': meta['dataset_generation'],
        'task_outcome': meta.get('task_outcome'),
        'source_bytes': item['bytes'], 'source_file_count': item['object_count'],
        'collection_meta_sha256': hashlib.sha256(meta_bytes).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    inventory = json.loads((args.source / 'episodes.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect_episode, ((args.source, item) for item in inventory)))
    assert len({r['source_relative_path'] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['task_id']].append(row)
    assert sorted(groups) == [f'{i:02d}' for i in range(1, 51)]
    selected, task_stats = [], []
    for task_id, candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: (Decimal(str(r['duration_seconds'])), r['episode_id']))
        assert len(candidates) >= args.per_task, task_id
        chosen = candidates[:args.per_task]
        for rank, row in enumerate(candidates, 1):
            row['duration_rank_in_task'] = rank
            row['selected'] = rank <= args.per_task
        selected.extend(chosen)
        task_stats.append({'task_id': task_id, 'task_name': chosen[0]['task_name'],
            'available': len(candidates), 'selected': len(chosen),
            'min_duration_seconds': chosen[0]['duration_seconds'],
            'max_selected_duration_seconds': chosen[-1]['duration_seconds'],
            'first_excluded_duration_seconds': candidates[args.per_task]['duration_seconds'] if len(candidates) > args.per_task else None,
            'selected_frames': sum(r['step_count'] for r in chosen)})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('selection.jsonl', selected), ('candidates.jsonl', sorted(rows, key=lambda r: (r['task_id'], r['duration_rank_in_task'])))]:
        (args.output / name).write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in data))
    summary = {
        'format': 'handson.newton_wuji2_shortest_selection.v1',
        'selection_rule': f'Within each task sort by (collection_meta.duration_seconds, episode_id) ascending; keep first {args.per_task}. Preserve complete episodes.',
        'duration_validation': 'duration_seconds == step_count / data_fps within 2 microseconds; replay action header length == step_count',
        'source_episodes': len(rows), 'selected_episodes': len(selected),
        'task_count': len(groups), 'episodes_per_task': args.per_task,
        'selected_frames': sum(r['step_count'] for r in selected),
        'selected_source_bytes': sum(r['source_bytes'] for r in selected),
        'selected_source_files': sum(r['source_file_count'] for r in selected),
        'generation_counts': dict(Counter(r['dataset_generation'] for r in selected)),
        'outcome_counts': dict(Counter(r['task_outcome'] for r in selected)),
        'selection_sha256': hashlib.sha256((args.output / 'selection.jsonl').read_bytes()).hexdigest(),
        'tasks': task_stats,
    }
    (args.output / 'SELECTION.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'tasks'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
