#!/usr/bin/env python3
"""Download and verify one task (or all 50) from the Newton/Wuji2 release."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=int, help='Task number 1..50; omitted means all 10,000 episodes')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', default='main')
    a = p.parse_args()
    if a.task is not None and not 1 <= a.task <= 50:
        p.error('task must be 1..50')
    destination = a.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repo = 'Cyan666/HandsOn-Datasets'
    manifest = Path(hf_hub_download(repo, 'newton_wuji2/SHARDS.json', repo_type='dataset', revision=a.revision))
    shards = json.loads(manifest.read_text())['shards']
    chosen = [s for s in shards if a.task is None or int(s['task_id']) == a.task]
    assert sum(s['episode_count'] for s in chosen) == (10000 if a.task is None else 200)
    for row in chosen:
        path = Path(hf_hub_download(repo, row['path'], repo_type='dataset', revision=a.revision))
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                h.update(block)
        if path.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
            raise RuntimeError('Shard checksum mismatch: ' + row['path'])
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or destination not in (destination / member.name).resolve().parents:
                    raise RuntimeError('Unsafe archive member: ' + member.name)
            archive.extractall(destination)
        print('Verified and extracted', row['path'], flush=True)


if __name__ == '__main__':
    main()
