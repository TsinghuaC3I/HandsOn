"""Build a GEN1-only hydrated XML closure without changing tracked LFS pointers."""
from pathlib import Path
import hashlib
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
XML=ROOT/'handson/handson/sim/envs/xmls'
CACHE=ROOT/'.gen1-cache'

def build():
 target=CACHE/'gen1_cpu_xml'
 queue=[Path('arena_arm_hand_eval_gen03_orient_bottle_upright_rokae_wuji1.xml')]
 seen=set()
 while queue:
  relative=queue.pop()
  if relative in seen: continue
  seen.add(relative)
  source=XML/relative;dest=target/relative
  dest.parent.mkdir(parents=True,exist_ok=True)
  raw=source.read_bytes()
  if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
   oid=raw.decode().split('oid sha256:')[1].splitlines()[0]
   blob=CACHE/'lfs'/oid
   if not blob.exists() or hashlib.sha256(blob.read_bytes()).hexdigest()!=oid:
    raise RuntimeError(f'Missing verified LFS source: {relative}; hydrate this OID in .gen1-cache/lfs')
   shutil.copyfile(blob,dest)
  else:dest.write_bytes(raw)
  if source.suffix=='.xml':
   for node in ET.fromstring(raw).iter():
    filename=node.get('file')
    if filename: queue.append(Path(filename))
 return target/'arena_arm_hand_eval_gen03_orient_bottle_upright_rokae_wuji1.xml'

if __name__=='__main__':print(build())
