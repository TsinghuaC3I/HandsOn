"""Package this branch's source and verified GEN1 asset closure for a task-only runtime mirror."""
from pathlib import Path
import hashlib,json,tarfile,os
from hydrate_cpu_scene import build
ROOT=Path(__file__).resolve().parents[3]
TARGET=ROOT/'.gen1-cache/gen1_runtime.tar.gz'
include_dirs=['handson/handson','collection_ui','scripts','configs','teleoperation','taskdev/tasks/eval_gen03_orient_bottle_upright']
files={}
# The ordinary collection server imports this existing task contract eagerly.
# Include its source dependency without copying the unrelated task's assets.
for p in (ROOT/'taskdev').glob('*.py'):
 files[p.relative_to(ROOT).as_posix()]=p
for name in ['taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/scene_contract.py','docs/pics/eval_gen03_orient_bottle_upright.png']:
 p=ROOT/name
 if p.is_file():files[name]=p
for prefix in include_dirs:
 for p in (ROOT/prefix).rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(ROOT).as_posix()
  if '/__pycache__/' in rel or '/.pytest_cache/' in rel:continue
  if rel.startswith(('handson/handson/sim/envs/usd/','handson/handson/sim/envs/xmls/')):continue
  if '/evidence/' in rel or '/vendor/' in rel or rel.endswith('/runtime_source_manifest.json'):continue
  if p.stat().st_size>20_000_000:continue
  files[rel]=p
for p in (ROOT/'handson/handson/sim/envs/xmls').rglob('*.xml'):
 files[p.relative_to(ROOT).as_posix()]=p
for folder in ['table_arena/textures','photograph']:
 for p in (ROOT/'handson/handson/sim/envs/xmls'/folder).rglob('*'):
  if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
for name in ['pyproject.toml','handson/pyproject.toml','requirements.txt']:
 p=ROOT/name
 if p.is_file():files[name]=p
xml=build().parent
for p in xml.rglob('*'):
 if p.is_file():files['handson/handson/sim/envs/xmls/'+p.relative_to(xml).as_posix()]=p
for p in (ROOT/'handson/handson/sim/envs/xmls/assets/eval_gen03_orient_bottle_upright').rglob('*'):
 if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
manifest={'source_branch':'codex/eval_gen03_orient_bottle_upright','baseline':'282390f0fec08daca53e2d6b03f6ad8772b609f4','role':'Task-only runtime mirror; all implementation edits are made on the dedicated source branch','files':{rel:hashlib.sha256(p.read_bytes()).hexdigest() for rel,p in files.items()}}
mp=ROOT/'taskdev/tasks/eval_gen03_orient_bottle_upright/runtime_source_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');files[mp.relative_to(ROOT).as_posix()]=mp
if os.environ.get('GEN1_PACKAGE_MANIFEST_ONLY')!='1':
 with tarfile.open(TARGET,'w:gz') as archive:
  for rel,p in sorted(files.items()):archive.add(p,arcname=rel,recursive=False)
 print('runtime_files',len(files),'archive_bytes',TARGET.stat().st_size)
else:print('manifest_files',len(files)-1)
