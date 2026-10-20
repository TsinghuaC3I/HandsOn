"""Build a task-local dependency mirror; hydrate only verified existing payloads.

Does not rewrite the repository's LFS pointers or any other worktree. GPU tests
must use a fully hydrated, equally hash-checked dependency closure too.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
XML=ROOT/'handson/handson/sim/envs/xmls'
OUT=ROOT/'taskdev/tasks/eval_gen01_cover_lunchbox/evidence/xml_mirror'
parser=argparse.ArgumentParser()
parser.add_argument('--source-repo',type=Path,required=True)
args=parser.parse_args()
seen=set(); files=[]; missing=[]

def copy(path):
    path=path.resolve()
    if path in seen:return
    seen.add(path)
    relative=path.relative_to(XML)
    data=path.read_bytes()
    pointer=data.startswith(b'version https://git-lfs.github.com/spec/v1')
    if pointer:
        expected=re.search(rb'oid sha256:([0-9a-f]+)',data).group(1).decode()
        size=int(re.search(rb'size (\d+)',data).group(1))
        source=args.source_repo/'handson/handson/sim/envs/xmls'/relative
        if source.is_file():
            payload=source.read_bytes()
            if len(payload)==size and hashlib.sha256(payload).hexdigest()==expected:
                data=payload
            else:
                missing.append(dict(file=str(relative),reason='available payload does not match LFS identity',oid=expected,size=size))
                return
        else:
            missing.append(dict(file=str(relative),reason='no local payload',oid=expected,size=size))
            return
    target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    files.append(dict(file=str(relative),sha256=hashlib.sha256(data).hexdigest(),size=len(data),hydrated=pointer))
    if path.suffix=='.xml':
        tree=ET.fromstring(data)
        for node in tree.iter():
            filename=node.get('file')
            if filename:
                dep=path.parent/filename
                copy(dep)

copy(XML/'arena_eval_gen01_cover_lunchbox_rokae_wuji1.xml')
# The shared importer/cache digest also needs the existing primitive source files.
# GEN2 selects the unchanged detailed source mesh colliders at runtime.
for side in ('right','left'):
    copy(XML/'wuji_hand1/mjcf'/f'{side}_primitive.xml')
report=dict(scope='GEN2 source dependencies, verified local payloads only',files=files,missing=missing)
(ROOT/'taskdev/tasks/eval_gen01_cover_lunchbox/evidence/asset_dependencies.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(files=len(files),hydrated=sum(f['hydrated'] for f in files),bytes=sum(f['size'] for f in files),missing=missing),indent=2))
raise SystemExit(bool(missing))
