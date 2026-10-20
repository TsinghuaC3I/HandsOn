#!/usr/bin/env python3
"""CPU views of the actual converted source triangles; not live evidence."""
from pathlib import Path
import os
import sys
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade
REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(REPO / '.cache-gen3/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, str(REPO / 'handson'))
from handson.sim.isaaclab.gen3_hinged_trash_bin_scene import GEN3_ASSET_DIRECTORY
stage = Usd.Stage.Open(str(GEN3_ASSET_DIRECTORY / 'runtime/trash_bin.usda'))
hinge = np.array(UsdGeom.Xform(stage.GetPrimAtPath('/TrashBin/trash_bin_lid')).GetLocalTransformation().ExtractTranslation())
meshes = []
for prim in stage.Traverse():
    if '/Visuals/' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):
        continue
    usd = UsdGeom.Mesh(prim)
    mesh = trimesh.Trimesh(vertices=np.array(usd.GetPointsAttr().Get()), faces=np.array(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1,3), process=False)
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    color = stage.GetPrimAtPath(str(material.GetPath()) + '/Surface').GetAttribute('inputs:diffuseColor').Get()
    meshes.append((str(prim.GetPath()), mesh, np.asarray(color if color is not None else [.3,.3,.3])))
light = np.array([-1., -1., 2.]); light /= np.linalg.norm(light)


def draw(ax, angle, elev, azim):
    all_triangles, all_colors = [], []
    a = np.radians(-angle)
    rotation = np.array([[1,0,0], [0,np.cos(a),-np.sin(a)], [0,np.sin(a),np.cos(a)]])
    for name, source, color in meshes:
        mesh = source.copy()
        if '/trash_bin_lid/' in name:
            mesh.vertices = mesh.vertices @ rotation.T + hinge
        colors = np.clip(color*.75+.12,0,1)[None,:] * (0.45+0.55*np.abs(mesh.face_normals@light))[:,None]
        all_triangles.append(mesh.triangles); all_colors.append(colors)
    ax.add_collection3d(Poly3DCollection(np.concatenate(all_triangles), facecolors=np.concatenate(all_colors), edgecolors='none', rasterized=True))
    ax.set(xlim=(-.13,.13), ylim=(-.15,.15), zlim=(0,.44))
    ax.set_box_aspect((.26,.30,.44)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()

fig = plt.figure(figsize=(13,8), facecolor='white')
for i,(angle,elev,azim) in enumerate([(0,25,-135),(75,25,-135),(90,70,-90),(0,5,-90)]):
    ax = fig.add_subplot(2,2,i+1,projection='3d'); draw(ax,angle,elev,azim)
    ax.set_title(f'Lid opening: {angle} degrees | CPU geometry view',fontsize=11)
fig.suptitle('GEN3 — existing animated trash-can asset\nSource triangles and materials; passive hinge; live physics pending',fontsize=15)
fig.tight_layout(); out=REPO/'taskdev/tasks/eval_gen02_open_trash_bin/evidence/source_geometry_views.png'; fig.savefig(out,dpi=140)
fig2=plt.figure(figsize=(6,5),facecolor='white'); ax=fig2.add_subplot(111,projection='3d');draw(ax,0,25,-135)
fig2.suptitle('GEN3 Open the Hinged Trash Bin',fontsize=14);fig2.savefig(REPO/'docs/pics/eval_gen02_open_trash_bin.png',dpi=120)
print(out)
