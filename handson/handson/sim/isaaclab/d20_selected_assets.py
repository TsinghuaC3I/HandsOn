"""Visual/material boundary for the user-selected D20 meshes."""
from functools import lru_cache
import json
import hashlib
import math
from pathlib import Path
import numpy as np

ASSET_ROOT = Path(__file__).resolve().parents[1]/'envs/xmls/assets/d20_selected'


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ASSET_ROOT/'manifest.json').read_text())


@lru_cache(maxsize=1)
def hot_dog_collision_vertices():
    """Use the selected collider surface for containment, including rest tilt."""
    points = []
    for relative in manifest()['assets']['hot_dog']['collision']:
        for line in (ASSET_ROOT / relative).read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == 'v':
                points.append([float(value) for value in fields[1:4]])
    vertices = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('D20 hot dog collision vertices are missing or invalid')
    vertices.setflags(write=False)
    return vertices


def asset_content_digest():
    """Include selected mesh/texture bytes in the converter cache identity."""
    payload=(ASSET_ROOT/'manifest.json').read_bytes()
    data=json.loads(payload)
    digest=hashlib.sha256(payload)
    paths=set()
    for entry in data['assets'].values():
        paths.add(entry['visual'])
        paths.update(entry['collision'])
    for material in data.get('materials',{}).values():
        paths.update(material[key] for key in ('baseColor','metallicRoughness','normal') if key in material)
    for relative in sorted(paths):
        label=relative.encode('utf-8')
        content=(ASSET_ROOT/relative).read_bytes()
        digest.update(len(label).to_bytes(8,'big'))
        digest.update(label)
        digest.update(len(content).to_bytes(8,'big'))
        digest.update(content)
    return digest.hexdigest()


def bounds_corners(bounds):
    b=np.asarray(bounds,dtype=np.float64)
    return np.asarray([[x,y,z] for x in b[:,0] for y in b[:,1] for z in b[:,2]])


def reset_points(name, xy, yaw):
    data=manifest();key=name.removeprefix('d20_')
    if key=='tongs':
        fixed=bounds_corners(data['assets']['tongs_fixed']['bounds'])
        moving=bounds_corners(data['assets']['tongs_moving']['bounds'])
        c,s=math.cos(.32),math.sin(.32)
        moving=moving@np.array([[c,s,0],[-s,c,0],[0,0,1]])+data['positions']['tongs_hinge']
        points=np.concatenate((fixed,moving))
    else:points=bounds_corners(data['assets'][key]['bounds'])
    points=points@np.asarray(data['reset'][name].get('rest_rotation',np.eye(3))).T
    a=math.radians(yaw);c,s=math.cos(a),math.sin(a)
    points=points@np.array([[c,s,0],[-s,c,0],[0,0,1]])
    points+=np.asarray([*xy,data['reset'][name]['z']])
    return points


def reset_bounds(name, xy, yaw):
    points=reset_points(name,xy,yaw)
    return np.array([points.min(0),points.max(0)])


def layout_clear(xy, yaw, hand_points=()):
    from . import d20_microwave_hot_dog as d20
    bounds={n:reset_bounds(n,xy[n],yaw[n]) for n in d20.FREE_BODIES}
    gap=.006
    tray_center=d20.TRAY_XY;tray_half=np.asarray(d20.TRAY_HALF_XY)
    oven=np.asarray(manifest()['fixture_poses']['microwave'])
    oven_low=oven[:2]+[-.156,-.209];oven_high=oven[:2]+[.156,.209]
    for name,b in bounds.items():
        if np.any(b[0,:2]<[-.54,-.79]) or np.any(b[1,:2]>[.24,.79]):return False
        if name in (*d20.TABLEWARE,d20.HOT_DOG):
            if np.any(b[0,:2]<tray_center-tray_half+gap) or np.any(b[1,:2]>tray_center+tray_half-gap):return False
        else:
            if not np.any((b[1,:2]+gap < tray_center-[.193,.300]) | (b[0,:2]-gap > tray_center+[.193,.300])):return False
        if not np.any((b[1,:2]+gap < oven_low) | (b[0,:2]-gap > oven_high)):return False
        for point in hand_points:
            # Sampling uses all live palm/finger origins with a conservative
            # 25 mm link envelope. Final acceptance uses actual shape queries.
            if np.all(np.asarray(point)>=b[0]-.025) and np.all(np.asarray(point)<=b[1]+.025):return False
    names=list(bounds)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if {a,b}=={d20.HOT_DOG,d20.PLATE}:continue
            if not np.any((bounds[a][1,:2]+gap<bounds[b][0,:2]) | (bounds[b][1,:2]+gap<bounds[a][0,:2])):return False
    food=reset_points(d20.HOT_DOG,xy[d20.HOT_DOG],yaw[d20.HOT_DOG])
    # The food must remain inside the dish's rim. Its actual underside rests
    # on the measured central surface; the bun's upper corners may overhang
    # that flat region while remaining well inside the plate itself.
    radius=manifest()['assets']['plate']['bounds'][1][0]-.006
    if any(np.linalg.norm(p-np.asarray(xy[d20.PLATE]))>radius for p in food[:,:2]):return False
    return True


def sample_layout(rng, *, hand_points=(), reach=None):
    from . import d20_microwave_hot_dog as d20
    for _ in range(2048):
        xy={n:base+rng.uniform(-d20.RESET_XY_JITTER[n],d20.RESET_XY_JITTER[n],2) for n,base in d20.RESET_XY.items()}
        xy[d20.HOT_DOG]=xy[d20.PLATE]+rng.uniform(-d20.RESET_XY_JITTER[d20.HOT_DOG],d20.RESET_XY_JITTER[d20.HOT_DOG],2)
        yaw = {}
        for n in xy:
            choices = getattr(d20, "RESET_YAW_CHOICES_DEG", {}).get(n)
            if choices:
                # Keep the food orientation visibly varied while remaining
                # deterministic under the task RNG and within the declared
                # yaw envelope used by reset audits.
                yaw[n] = float(rng.choice(np.asarray(choices, dtype=np.float64)))
            else:
                yaw[n] = d20.RESET_BASE_YAW_DEG[n] + float(
                    rng.uniform(-d20.RESET_YAW_JITTER_DEG[n], d20.RESET_YAW_JITTER_DEG[n])
                )
        if layout_clear(xy,yaw,hand_points) and (reach is None or reach(xy)):
            return xy,yaw
    raise RuntimeError('D20 randomized layout has no collision-clear supported sample; inspect hand/fixture poses')


def configure_visuals(stage, task_root):
    """Bind source PBR maps and prove collision-only prims are invisible."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    data=manifest();materials={}
    material_root=f'{task_root.GetPath()}/d20_materials'
    for key, definition in data['materials'].items():
        path=f'{material_root}/{key}'
        material=UsdShade.Material.Define(stage,path)
        shader=UsdShade.Shader.Define(stage,f'{path}/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(definition.get('roughnessFactor',.5)))
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(definition.get('metallicFactor',0)))
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*definition.get('baseColorFactor',[1,1,1,1])[:3]))
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        reader=UsdShade.Shader.Define(stage,f'{path}/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        for slot in ('baseColor','metallicRoughness','normal'):
            if slot not in definition:continue
            tex=UsdShade.Shader.Define(stage,f'{path}/{slot}')
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str((ASSET_ROOT/definition[slot]).resolve()))
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if slot=='baseColor' else 'raw')
            for channel,typ in (('rgb',Sdf.ValueTypeNames.Float3),('r',Sdf.ValueTypeNames.Float),('g',Sdf.ValueTypeNames.Float),('b',Sdf.ValueTypeNames.Float)):
                tex.CreateOutput(channel,typ)
            if slot=='baseColor':
                shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
            elif slot=='metallicRoughness':
                shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),'g')
                shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),'b')
            else:
                tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
                tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
                shader.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        # Sim 5.1 names imported OBJ prims after the MJCF mesh asset, rather
        # than the geom name. Bind the actual Mesh leaf (not just its Xform).
        materials[f'd20_{key}_mesh']=material
    hidden=[];bound=[]
    for prim in list(Usd.PrimRange(task_root)):
        if prim.IsInstance():prim.SetInstanceable(False)
    for prim in Usd.PrimRange(task_root):
        name=prim.GetName()
        if name.startswith('d20_') and '_collision_' in name:
            imageable=UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible();hidden.append(str(prim.GetPath()))
        if name in materials and prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[name])
            bound.append(name)
    missing=set(materials)-set(bound)
    if missing:
        raise RuntimeError(f'D20 selected render meshes missing: {sorted(missing)}')
    for path in hidden:
        if UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility() != UsdGeom.Tokens.invisible:
            raise RuntimeError(f'D20 collider is visible: {path}')
    print(f'[D20 assets] PBR meshes={len(bound)} invisible collision prims={len(hidden)}',flush=True)
