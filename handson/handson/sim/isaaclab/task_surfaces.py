"""Archived Core task surfaces applied to the existing native physical scene.

The original deployment copies shaders, render attributes and material subsets.
Transfer tasks retain their own native appearance. Assets are selected by the
paper's recorded task release, independent of policy output or success.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'assets/task_surfaces'
MANIFEST_SHA256 = 'c2d070bb65402330922a6f016645c2a3f2cfa7099fecc5baa0761d619f57ec14'


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def _manifest():
    raw = (ROOT / 'manifest.json').read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, 'Task surface manifest changed')
    value = json.loads(raw)
    _require(value['schema'] == 'handson_paper_task_surfaces_v1' and
             value['verified_trials'] == 45000, 'Unknown task surface manifest')
    from ...evaluation.paper_trials import MANIFEST_SHA256 as trials_sha256
    _require(value['paper_trial_manifest_sha256'] == trials_sha256, 'Task surfaces refer to different paper trials')
    return value


@lru_cache(maxsize=None)
def _verify_file(relative):
    path = Path(relative)
    _require(not path.is_absolute() and '..' not in path.parts, 'Task surface asset escapes bundle')
    expected = _manifest()['files'][relative]
    _require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
             'Task surface asset changed: ' + relative)
    return ROOT / path


def apply(environment):
    task = environment.task_name
    if getattr(environment, '_physics_backend', None) != 'physx':
        return dict(mode='native', reason='not the archived PhysX evaluator')
    catalog = _manifest()
    if task in catalog['transfer_tasks']:
        return dict(mode='native_task_objects', manifest_sha256=MANIFEST_SHA256)
    _require(task in catalog['tasks'], 'No archived task surface contract: ' + task)
    profile_hash = catalog['tasks'][task]
    profile = catalog['profiles'][profile_hash]
    if profile['mode'] == 'recorded_native_git_assets':
        return dict(mode=profile['mode'], source_asset_release_sha256=profile_hash,
                    manifest_sha256=MANIFEST_SHA256)

    from pxr import Sdf, Usd, UsdGeom, UsdShade
    stage = environment._stage
    source_path = _verify_file(profile['layer'])
    source = Usd.Stage.Open(str(source_path))
    _require(bool(source), 'Could not open archived task surfaces')
    # Resolve and verify only this layer's referenced textures/MDL. Built-in
    # OmniPBR/OmniSurface modules remain owned by the pinned Isaac Sim runtime.
    for prim in source.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                if value.path in {'OmniPBR.mdl', 'OmniSurface.mdl'}:
                    continue
                path = (source_path.parent / value.path).resolve()
                _require(path.is_relative_to(ROOT.resolve()), 'Surface dependency escapes package')
                _verify_file(str(path.relative_to(ROOT.resolve())))

    target_layer = stage.GetSessionLayer()
    present, absent = [], []
    with Usd.EditContext(stage, target_layer):
        for row in profile['bindings']:
            path = row['subset']['parent'] if row.get('subset') else row['target_prim']
            prim = stage.GetPrimAtPath(path)
            if not prim:
                absent.append(path)
                continue
            # Same pre-physics deinstancing boundary as the archived helper.
            chain, parent = [], prim
            while parent and not parent.IsPseudoRoot():
                chain.append(str(parent.GetPath()))
                parent = parent.GetParent()
            for name in reversed(chain):
                parent = stage.GetPrimAtPath(name)
                if parent.IsInstance():
                    parent.SetInstanceable(False)
            present.append(row)
        _require(bool(present), 'No archived task visual targets exist: ' + task)
        flattened = source.Flatten()
        UsdGeom.Scope.Define(stage, '/World/V14Materials')
        _require(Sdf.CopySpec(flattened, Sdf.Path('/World/V14Materials'), target_layer,
                             Sdf.Path('/World/V14Materials')), 'Could not copy archived task materials')
        for row in present:
            if row.get('subset'):
                part = row['subset']
                parent = stage.GetPrimAtPath(part['parent'])
                _require(parent.IsA(UsdGeom.Mesh) and
                         len(UsdGeom.Mesh(parent).GetFaceVertexCountsAttr().Get()) == part['source_face_count'],
                         'Archived material face groups differ: ' + part['parent'])
                subset = UsdGeom.Subset.Define(stage, row['target_prim'])
                subset.CreateElementTypeAttr().Set('face')
                subset.CreateFamilyNameAttr().Set('materialBind')
                subset.CreateIndicesAttr().Set(part['indices'])
            prim = stage.GetPrimAtPath(row['target_prim'])
            src, dst = source.GetPrimAtPath(row['visual_target']), stage.GetPrimAtPath(row['visual_target'])
            if src and dst:
                for attr in src.GetAttributes():
                    value = attr.Get()
                    if value is None:
                        continue
                    target = dst.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom())
                    target.Set(value)
                    for key in ('interpolation', 'elementSize'):
                        if attr.HasMetadata(key):
                            target.SetMetadata(key, attr.GetMetadata(key))
            material = UsdShade.Material(stage.GetPrimAtPath(row['material']))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
            for _ in range(64):
                actual, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if actual and actual.GetPath() == material.GetPath():
                    break
                _require(relation and relation.GetPrim() != prim,
                         'Archived material binding unresolved: ' + str(prim.GetPath()))
                UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(relation, UsdShade.Tokens.weakerThanDescendants)
            else:
                raise RuntimeError('Archived material binding did not converge: ' + str(prim.GetPath()))
    return dict(mode=profile['mode'], manifest_sha256=MANIFEST_SHA256,
                source_asset_release_sha256=profile_hash,
                source_visuals_sha256=profile['source_visuals_sha256'],
                portable_layer_sha256=catalog['files'][profile['layer']],
                applied_bindings=len(present), materials=profile['material_count'],
                absent_in_this_native_scene=absent,
                bindings=[dict(target_prim=r['target_prim'], material=r['material']) for r in present])
