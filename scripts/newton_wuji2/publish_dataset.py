#!/usr/bin/env python3
"""Stream verified source episodes into bounded tar shards and publish to the Hub.

Requires huggingface_hub and an existing write token (HF_TOKEN or hf auth login).
Only this release's scratch archives are removed, after remote hash readback.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time

TOKEN = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def retry_read(call):
    for attempt in range(5):
        try:
            return call()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 10))


class VerifiedReader:
    def __init__(self, stream):
        self.stream, self.md5 = stream, hashlib.md5()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.md5.update(block)
        return block


def main():
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--repo', default='Cyan666/HandsOn-Datasets')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--shard-episodes', type=int, default=50)
    parser.add_argument('--tasks', type=int, nargs='+', choices=range(1, 51),
                        help='Upload only these tasks; keep subset progress separate from the full release.')
    args = parser.parse_args()
    task_filter = {f'{task:02d}' for task in args.tasks} if args.tasks else None
    scope = 'tasks-' + '-'.join(sorted(task_filter)) if task_filter else None
    folder = args.release / 'package' / 'newton_wuji2'
    rows = [json.loads(line) for line in (folder / 'selection.jsonl').read_text().splitlines()]
    selected = {r['source_relative_path']: r for r in rows}
    assert len(rows) == len(selected) == 10000
    files = defaultdict(list)
    for line in (args.source / 'transfer_manifest.jsonl').read_text().splitlines():
        item = json.loads(line)
        episode = '/'.join(item['relpath'].split('/')[:3])
        if episode in selected:
            files[episode].append(item)
    for episode, row in selected.items():
        assert len(files[episode]) == row['source_file_count'], episode
        assert sum(f['size'] for f in files[episode]) == row['source_bytes'], episode
    # Check plaintext metadata before any public upload. Report paths, never values.
    metadata = [item for episode in selected for item in files[episode]
                if Path(item['relpath']).suffix in {'.json', '.yaml', '.yml', '.txt'}]
    def inspect_metadata(item):
        content = (args.source / item['relpath']).read_bytes()
        if TOKEN.search(content):
            raise RuntimeError('Credential-like content in ' + item['relpath'])
    print(json.dumps({'event': 'checking_metadata', 'files': len(metadata)}), flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(inspect_metadata, metadata):
            pass
    print(json.dumps({'event': 'metadata_checked', 'files': len(metadata)}), flush=True)
    api = HfApi()
    identity = retry_read(lambda: api.whoami())['name']
    print(json.dumps({'event': 'authenticated', 'account': identity, 'episodes': len(rows)}), flush=True)
    scratch = args.release / ('scratch-' + scope if scope else 'scratch')
    receipts = args.release / 'receipts'
    scratch.mkdir(exist_ok=True)
    receipts.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['task_id']].append(row)
    jobs = []
    for task, episodes in sorted(groups.items()):
        assert len(episodes) == 200
        if task_filter is not None and task not in task_filter:
            continue
        for start in range(0, len(episodes), args.shard_episodes):
            jobs.append((task, start // args.shard_episodes, episodes[start:start + args.shard_episodes]))
    print(json.dumps({'event': 'scheduled', 'scope': scope or 'all', 'shards': len(jobs)}), flush=True)

    def remote_match(path, sha, size):
        info = retry_read(lambda: api.get_paths_info(
            args.repo, paths=[path], repo_type='dataset', revision=args.revision))
        return bool(info and info[0].size == size and info[0].lfs and info[0].lfs.sha256 == sha)

    def worker(job):
        task, part, episodes = job
        name = f'task-{task}-part-{part:03d}.tar'
        remote = 'newton_wuji2/raw/' + name
        receipt = receipts / (name + '.json')
        archive_path = scratch / name
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if remote_match(remote, prior['sha256'], prior['bytes']):
                archive_path.unlink(missing_ok=True)
                return prior
        began = time.time()
        print(json.dumps({'event': 'packing', 'shard': name}), flush=True)
        entries = [item for row in episodes for item in sorted(files[row['source_relative_path']], key=lambda f: f['relpath'])]
        temporary = archive_path.with_suffix('.partial')
        with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT) as archive:
            for item in entries:
                source = args.source / item['relpath']
                assert source.is_file() and not source.is_symlink() and source.stat().st_size == item['size'], str(source)
                info = tarfile.TarInfo(item['relpath'])
                info.size, info.mode = item['size'], 0o644
                info.mtime, info.uid, info.gid = 0, 0, 0
                with source.open('rb') as stream:
                    reader = VerifiedReader(stream)
                    archive.addfile(info, reader)
                    assert reader.md5.hexdigest() == item['etag'].strip('"'), 'Source MD5 mismatch: ' + item['relpath']
        temporary.replace(archive_path)
        sha, size = digest(archive_path), archive_path.stat().st_size
        print(json.dumps({'event': 'uploading', 'shard': name, 'bytes': size, 'sha256': sha}), flush=True)
        commit = None
        for attempt in range(5):
            try:
                if not remote_match(remote, sha, size):
                    commit = api.upload_file(path_or_fileobj=archive_path, path_in_repo=remote,
                        repo_id=args.repo, repo_type='dataset', revision=args.revision,
                        commit_message=f'Add Newton Wuji2 task {task} shard {part + 1}/4')
                assert remote_match(remote, sha, size), 'Remote size/SHA256 mismatch'
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(5 * 2 ** attempt, 30))
        result = {'path': remote, 'sha256': sha, 'bytes': size, 'task_id': task,
            'episodes': [r['episode_id'] for r in episodes], 'episode_count': len(episodes),
            'source_file_count': len(entries), 'source_bytes': sum(x['size'] for x in entries),
            'source_md5_verified': True, 'remote_lfs_sha256_verified': True,
            'commit': commit.oid if commit else None, 'elapsed_seconds': round(time.time() - began, 2)}
        write_json(receipt, result)
        archive_path.unlink()
        print(json.dumps({'event': 'verified', 'shard': name, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        return result

    complete = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, job) for job in jobs]):
            complete.append(future.result())
            progress_name = 'PROGRESS.' + scope + '.json' if scope else 'PROGRESS.json'
            write_json(args.release / progress_name, {'completed_shards': len(complete),
                'total_shards': len(jobs), 'verified_episodes': sum(r['episode_count'] for r in complete),
                'verified_archive_bytes': sum(r['bytes'] for r in complete), 'updated_unix': time.time()})
    complete.sort(key=lambda r: r['path'])
    manifest_path = args.release / ('SHARDS.' + scope + '.json') if scope else folder / 'SHARDS.json'
    complete_name = 'UPLOAD_COMPLETE.' + scope + '.json' if scope else 'UPLOAD_COMPLETE.json'
    write_json(manifest_path, {'format': 'handson.newton_wuji2_tar_shards.v1', 'shards': complete})
    write_json(args.release / complete_name, {'repo': args.repo, 'revision': args.revision,
        'episodes': sum(r['episode_count'] for r in complete), 'shards': len(complete),
        'sha256': digest(manifest_path), 'completed_unix': time.time()})


if __name__ == '__main__':
    main()
