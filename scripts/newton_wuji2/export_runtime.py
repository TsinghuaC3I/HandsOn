#!/usr/bin/env python3
"""Freeze the validated replay source, native assets and V16/Wuji2 visuals.

This is a maintainer export tool. Consumers use render_episode.py with the
published archive. Symlinks are dereferenced so the result is self-contained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview-root', type=Path, required=True)
    p.add_argument('--v16-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    entries = []
    for relative in ['source', 'assets/wuji2_photo_glove_link_v14', 'cloth_joint_profile_v14.json']:
        base = a.preview_root / relative
        for path in sorted(base.rglob('*')) if base.is_dir() else [base]:
            if '__pycache__' in path.parts or not path.is_file() or path.suffix in {'.pyc', '.baseline'}:
                continue
            entries.append((path, str(path.relative_to(a.preview_root))))
    for relative in ['scripts/v16_appearance.py', 'scripts/V16_APPEARANCE.json', 'appearance_assets']:
        base = a.v16_runtime / relative
        for path in sorted(base.rglob('*')) if base.is_dir() else [base]:
            if path.is_file():
                entries.append((path, 'v16/' + str(path.relative_to(a.v16_runtime))))
    # A public runtime contains no operational queues, GPU claims or credentials.
    token = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    manifest = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output.with_suffix('.partial'), 'w', dereference=True) as tar:
        for path, relative in entries:
            content = path.read_bytes()
            if path.suffix in {'.py', '.sh', '.json', '.toml', '.yaml', '.yml', '.md', '.txt'} and token.search(content):
                raise RuntimeError('Credential-like content in ' + relative)
            info = tar.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            info.mode = 0o644
            with path.open('rb') as stream:
                tar.addfile(info, stream)
            manifest.append({'path': relative, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    a.output.with_suffix('.partial').replace(a.output)
    h = hashlib.sha256()
    with a.output.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    result = {'format': 'handson.newton_wuji2_replay_runtime.v1', 'archive_sha256': h.hexdigest(),
              'archive_bytes': a.output.stat().st_size, 'files': manifest}
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
