"""Export a hash-bound sparse snapshot of this branch for isolated GPU audits."""
from pathlib import Path
import subprocess
import json
import tarfile
import hashlib

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent / 'evidence'
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
rows = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', 'HEAD'], cwd=ROOT).split(b'\0')
entries = []
for row in rows:
    if not row:
        continue
    info, path = row.split(b'\t', 1)
    mode, kind, oid = info.split()
    if kind == b'blob':
        entries.append((oid.decode(), path.decode()))

def needed(path):
    if path.startswith('taskdev/tasks/'):
        return False
    if path.startswith('handson/handson/sim/envs/xmls/'):
        return path.endswith(('.xml', '.json', '.py', '.yaml', '.toml'))
    if path.startswith('teleoperation/manus/vendor/'):
        return False
    return (ROOT / path).stat().st_size < 5_000_000

selected = [(oid, path) for oid, path in entries if needed(path)]
trees = subprocess.check_output(['git', 'ls-tree', '-rt', 'HEAD'], cwd=ROOT, text=True).splitlines()
objects = {head, *[line.split()[2] for line in trees if line.split()[1] == 'tree'],
           *[oid for oid, path in selected]}
objects.add(subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip())
with (EVIDENCE / 'gen2-source.pack').open('wb') as stream:
    subprocess.run(['git', 'pack-objects', '--stdout'], cwd=ROOT,
                   input=('\n'.join(sorted(objects)) + '\n').encode(), stdout=stream, check=True)
modified = subprocess.check_output(['git', 'ls-files', '-m', '-o', '--exclude-standard', '-z'], cwd=ROOT).split(b'\0')
paths = {path for oid, path in selected} | {
    path.decode() for path in modified if path and
    not path.decode().startswith('taskdev/tasks/eval_gen01_cover_lunchbox/evidence/')}
paths |= {str(path.relative_to(ROOT)) for path in (EVIDENCE / 'xml_mirror').rglob('*') if path.is_file()}
manifest = dict(branch='codex/eval_gen01_cover_lunchbox', base_commit=head, files={})
with tarfile.open(EVIDENCE / 'gen2-runtime.tar.gz', 'w:gz') as archive:
    for name in sorted(paths):
        path = ROOT / name
        if path.is_file():
            archive.add(path, arcname=name, recursive=False)
            manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
(EVIDENCE / 'remote_source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(EVIDENCE / 'sparse_paths.txt').write_text('\n'.join('/' + path for oid, path in selected) + '\n')
print(dict(pack_bytes=(EVIDENCE / 'gen2-source.pack').stat().st_size,
           runtime_bytes=(EVIDENCE / 'gen2-runtime.tar.gz').stat().st_size, files=len(paths)))
