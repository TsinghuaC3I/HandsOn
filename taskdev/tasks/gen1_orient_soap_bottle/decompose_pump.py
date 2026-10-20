"""Derive compact convex pieces from the existing pump source meshes."""
import json
from pathlib import Path
import coacd
import trimesh
from prepare_asset import source_meshes, ROOT

parts,_=source_meshes()
coacd.set_log_level('warn')
result={}
for name in ['Object_21','Object_17']:
 m=parts[name].copy();m.visual=trimesh.visual.ColorVisuals();m.merge_vertices(merge_tex=True,merge_norm=True)
 solids=m.split(only_watertight=True)
 result[name]=[]
 for solid in solids:
  decomposition=coacd.run_coacd(coacd.Mesh(solid.vertices,solid.faces),threshold=.02,max_convex_hull=16,preprocess_mode='off',resolution=2000,mcts_nodes=15,mcts_iterations=80,mcts_max_depth=3,decimate=True,max_ch_vertex=128,seed=7)
  result[name].extend({'vertices':v.tolist(),'faces':f.tolist()} for v,f in decomposition)
 print(name,len(result[name]),flush=True)
 if not result[name]:raise RuntimeError('No closed source component available')
(ROOT/'.gen1-cache/gen1_pump_decomposition.json').write_text(json.dumps(result))
