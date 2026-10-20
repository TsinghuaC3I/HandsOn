"""Orthographic CPU rasterization of final USD meshes and source texture UVs.

Used only for asset inspection. It is not an Isaac/RTX or recording-camera test.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))
import json
import numpy as np
import trimesh
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade
from build_assets import OUTPUT, SOURCE, TASK_DIR


def batches():
    data = json.loads((SOURCE / 'scene.gltf').read_text())
    materials = {item['name']: item for item in data['materials']}
    geometry = json.loads((OUTPUT / 'asset_manifest.json').read_text())['geometry']
    sources = [('mixer_fixed_visual.usda', np.eye(4))]
    for side in ('left', 'right'):
        pose = trimesh.transformations.quaternion_matrix(geometry['knob_orientation_wxyz'])
        pose[:3, 3] = geometry['knob_centers_m'][side]
        sources.append(('knob_visual.usda', pose))
    result = []
    for filename, pose in sources:
        s = Usd.Stage.Open(str(OUTPUT / filename))
        for prim in s.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(prim)
            assert m.ComputeVisibility() == 'inherited'
            v = trimesh.transform_points(np.asarray(m.GetPointsAttr().Get()), pose)
            f = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            item = materials[material.GetPrim().GetName()]
            pbr = item.get('pbrMetallicRoughness', {})
            color = np.asarray(pbr.get('baseColorFactor', [1., 1., 1., 1.]))[:3]
            tex, uv = None, None
            spec = pbr.get('baseColorTexture')
            if spec is not None:
                uri = data['images'][data['textures'][spec['index']]['source']]['uri']
                tex = np.asarray(Image.open(OUTPUT / uri).convert('RGB')).astype(float) / 255.
                uv = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar('st').Get())[f]
            else:
                color = np.power(color, 1/2.2)
            result.append((v[f], uv, tex, color))
    return result


def render(data, elev, azim, width=900, height=750):
    el, az = np.radians([elev, azim])
    view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.cross([0., 0., 1.], view); right /= np.linalg.norm(right)
    up = np.cross(view, right)
    basis = np.array([right, -up, view])
    allv = np.concatenate([item[0].reshape(-1, 3) for item in data])
    projected = allv @ basis.T
    low, high = projected.min(0), projected.max(0)
    scale = min((width-80)/(high[0]-low[0]), (height-80)/(high[1]-low[1]))
    center = (low+high)/2
    zbuffer = np.full((height, width), -np.inf)
    image = np.full((height, width, 3), [243., 245., 247.]) / 255.
    light = np.array([-.3, -.2, .9327])
    for triangles, uvs, texture, color in data:
        screen = triangles @ basis.T
        screen[..., :2] = (screen[..., :2] - center[:2]) * scale + [width/2, height/2]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-15)
        shades = .4 + .6*np.abs(normal@light)
        for i, t in enumerate(screen):
            x0, y0 = np.maximum(np.ceil(t[:, :2].min(0)-.5).astype(int), 0)
            x1, y1 = np.minimum(np.floor(t[:, :2].max(0)-.5).astype(int), [width-1, height-1])
            if x1 < x0 or y1 < y0:
                continue
            a, b, c = t
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-10:
                continue
            x, y = np.meshgrid(np.arange(x0, x1+1)+.5, np.arange(y0, y1+1)+.5)
            w0 = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
            w1 = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
            w2 = 1.-w0-w1
            depth = w0*a[2]+w1*b[2]+w2*c[2]
            region = zbuffer[y0:y1+1, x0:x1+1]
            mask = (w0 >= -1e-8)&(w1 >= -1e-8)&(w2 >= -1e-8)&(depth > region)
            if not mask.any():
                continue
            rgb = color * shades[i]
            if texture is not None:
                weights = np.column_stack([w0[mask], w1[mask], w2[mask]])
                uv = weights @ uvs[i]
                tx = np.clip(np.round(uv[:, 0]*(texture.shape[1]-1)).astype(int), 0, texture.shape[1]-1)
                ty = np.clip(np.round((1.-uv[:, 1])*(texture.shape[0]-1)).astype(int), 0, texture.shape[0]-1)
                rgb = texture[ty, tx] * rgb
            image[y0:y1+1, x0:x1+1][mask] = rgb
            region[mask] = depth[mask]
    return Image.fromarray(np.uint8(np.clip(image, 0., 1.) * 255))


if __name__ == '__main__':
    data = batches()
    for label, elev, azim in [('top', 70., 180.), ('approach', 35., 135.)]:
        render(data, elev, azim).save(TASK_DIR / f'evidence/asset_cpu_{label}.png')
        print(f'Wrote {label} view', flush=True)
