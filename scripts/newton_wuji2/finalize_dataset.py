#!/usr/bin/env python3
"""Verify all 200 remote shards and publish release manifests and dataset card."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--readme', type=Path, required=True)
    p.add_argument('--publish-card', action='store_true')
    a = p.parse_args()
    folder = a.release / 'package/newton_wuji2'
    selected_bytes = (folder / 'selection.jsonl').read_bytes()
    rows = [json.loads(line) for line in selected_bytes.splitlines()]
    candidates = [json.loads(line) for line in (folder / 'candidates.jsonl').read_text().splitlines()]
    summary = json.loads((folder / 'SELECTION.json').read_text())
    shards = json.loads((folder / 'SHARDS.json').read_text())['shards']
    complete = json.loads((a.release / 'UPLOAD_COMPLETE.json').read_text())
    assert complete['episodes'] == len(rows) == 10000
    assert complete['shards'] == len(shards) == 200
    assert len(candidates) == 11678
    assert hashlib.sha256(selected_bytes).hexdigest() == summary['selection_sha256']
    groups = defaultdict(list)
    for row in candidates:
        groups[row['task_id']].append(row)
    assert sorted(groups) == [f'{i:02d}' for i in range(1, 51)]
    wanted = {}
    for task, items in groups.items():
        for item in sorted(items, key=lambda r: (r['duration_seconds'], r['episode_id']))[:200]:
            wanted[(task, item['episode_id'])] = item
    actual = {(r['task_id'], r['episode_id']): r for r in rows}
    assert actual == wanted and len(actual) == 10000
    membership = [(s['task_id'], episode) for s in shards for episode in s['episodes']]
    assert len(membership) == len(set(membership)) == 10000 and set(membership) == set(actual)
    assert Counter(task for task, _ in membership) == {f'{i:02d}': 200 for i in range(1, 51)}
    assert sum(s['source_bytes'] for s in shards) == summary['selected_source_bytes']
    assert sum(s['source_file_count'] for s in shards) == summary['selected_source_files']
    assert all(s['source_md5_verified'] and s['remote_lfs_sha256_verified'] for s in shards)
    api = HfApi()
    repo = 'Cyan666/HandsOn-Datasets'
    revision = api.dataset_info(repo).sha
    remote = {}
    for start in range(0, len(shards), 50):
        info = api.get_paths_info(repo, [s['path'] for s in shards[start:start+50]],
                                  repo_type='dataset', revision=revision)
        remote.update({item.path: item for item in info})
    for shard in shards:
        item = remote[shard['path']]
        assert item.size == shard['bytes'] and item.lfs.sha256 == shard['sha256'], shard['path']
    extras = json.loads((a.release / 'EXTRAS_COMPLETE.json').read_text())
    for item in extras:
        found = api.get_paths_info(repo, [item['path']], repo_type='dataset', revision=revision)[0]
        assert found.size == item['bytes'], item['path']
        if found.lfs:
            assert found.lfs.sha256 == item['sha256'], item['path']
        else:
            downloaded = Path(hf_hub_download(repo, item['path'], repo_type='dataset', revision=revision))
            assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == item['sha256'], item['path']
    proof = {'status': 'complete', 'verified_hub_revision': revision, 'tasks': 50, 'episodes': 10000,
        'shards': 200, 'source_files': summary['selected_source_files'],
        'source_bytes': summary['selected_source_bytes'], 'frames': summary['selected_frames'],
        'selection_sha256': summary['selection_sha256'],
        'verification': 'Per-task shortest-200 membership independently recomputed; all remote archive sizes and LFS SHA256s checked at one immutable Hub revision; every archived source file MD5 matched the transfer inventory.',
        'renderer_and_example_files': extras}
    (folder / 'RELEASE_COMPLETE.json').write_text(json.dumps(proof, indent=2) + '\n')
    operations = [CommitOperationAdd(path_in_repo='newton_wuji2/' + name, path_or_fileobj=folder / name)
        for name in ['selection.jsonl', 'candidates.jsonl', 'SELECTION.json', 'SHARDS.json', 'RELEASE_COMPLETE.json']]
    operations.append(CommitOperationAdd(path_in_repo='newton_wuji2/README.md', path_or_fileobj=a.readme))
    if a.publish_card:
        card = Path(hf_hub_download(repo, 'README.md', repo_type='dataset', revision=revision)).read_text()
        heading = '## Newton + Wuji Hand 2 addition (2026-11-08)'
        if heading not in card:
            addition = '''## Newton + Wuji Hand 2 addition (2026-11-08)

The [Newton/Wuji2 release](newton_wuji2/README.md) adds **10,000 complete raw
trajectories: 200 per Core task**, with 7,077,577 timesteps at 30 Hz. Within
each task, these are the shortest 200 by recorded duration (episode ID breaks
ties). The addition contains 9,800 Newton v1 episodes and 200 Newton v2
episodes for task 09; original generation markers are preserved.

These are source Zarr archives with replay assets and metadata. The matching
[V16/Wuji2 rendering code and instructions](https://github.com/TsinghuaC3I/HandsOn/blob/main/docs/newton-wuji2-dataset.md)
produce the main and two wrist videos. See the
[complete three-view acceptance example](newton_wuji2/examples/task03/three_views.mp4).
The raw addition has no assigned train/validation split and is not a claim of
10,000 pre-rendered LeRobot episodes. The PhysX/Wuji1 corpus described below
retains its own splits and normalization statistics.

'''
            assert '## Dataset summary' in card
            card = card.replace('## Dataset summary', addition + '## Dataset summary', 1)
        operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card.encode()))
    commit = api.create_commit(repo_id=repo, repo_type='dataset', operations=operations, parent_commit=revision,
        commit_message='Release verified Newton Wuji2 shortest-200 dataset: 50 tasks, 10,000 trajectories')
    (a.release / 'PUBLISHED.json').write_text(json.dumps({'commit': commit.oid, 'url': commit.commit_url,
        'dataset_card_updated': a.publish_card, 'proof': proof}, indent=2) + '\n')
    print(commit.commit_url)


if __name__ == '__main__':
    main()
