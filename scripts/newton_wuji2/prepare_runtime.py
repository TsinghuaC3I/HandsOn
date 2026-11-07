#!/usr/bin/env python3
"""Resolve machine-local shader paths in the checked runtime without changing geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    try:
        from pxr import Sdf
    except ModuleNotFoundError:
        # Pip Isaac Sim exposes USD only after Kit initializes. For this CPU
        # path-only preparation, load its own USD extension in a fresh process.
        import importlib.util
        import os
        import subprocess
        if os.environ.get('HANDSON_USD_PREPARE_BOOTSTRAPPED'):
            raise
        spec = importlib.util.find_spec('isaacsim')
        candidates = [path for base in ((spec.submodule_search_locations or []) if spec else [])
                      for path in Path(base).glob('extscache/omni.usd.libs-*')
                      if (path / 'pxr').is_dir()]
        if not candidates:
            raise RuntimeError('Isaac Sim USD extension is missing; install Isaac Sim 5.1')
        usd = sorted(candidates)[-1]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(usd) + os.pathsep + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(usd / 'bin'),
            str(Path(sys.base_prefix) / 'lib'), env.get('LD_LIBRARY_PATH', '')])
        env['HANDSON_USD_PREPARE_BOOTSTRAPPED'] = '1'
        subprocess.run([sys.executable, __file__, str(root)], env=env, check=True)
        return
    visuals = Path(__file__).resolve().parent / 'visuals'
    sys.path.insert(0, str(visuals))
    from soft_glove_material_v12 import mdl_source
    mdl = mdl_source()
    bundle = root / 'assets/wuji2_photo_glove_link_v14'
    manifest_path = bundle / 'manifest.json'
    original_path = bundle / 'manifest.original.json'
    if not original_path.exists():
        shutil.copyfile(manifest_path, original_path)
    original = json.loads(original_path.read_text())
    if sha(mdl) != original['soft_cloth_v12']['mdl_source_sha256']:
        raise RuntimeError('OmniSurface.mdl differs from the validated Isaac Sim 5.1 shader')
    manifest = json.loads(manifest_path.read_text())
    relocations = []
    for item in manifest['links']:
        asset = bundle / item['asset']
        if sha(asset) != item['sha256']:
            raise RuntimeError('Changed asset: ' + str(asset))
        layer = Sdf.Layer.FindOrOpen(str(asset))
        changed = []
        def visit(path):
            spec = layer.GetObjectAtPath(path)
            if not isinstance(spec, Sdf.AttributeSpec) or not isinstance(spec.default, Sdf.AssetPath):
                return
            value = spec.default.path
            target = None
            if value.endswith('/OmniSurface.mdl'):
                target = str(mdl)
            elif value.startswith('/') and '/textures/' in value:
                texture = 'textures/' + value.rsplit('/textures/', 1)[-1]
                if texture not in manifest['dependencies']:
                    raise RuntimeError('Unexpected texture dependency: ' + value)
                target = '../' + texture
            if target is not None and target != value:
                spec.default = Sdf.AssetPath(target)
                changed.append(str(path))
        layer.Traverse('/', visit)
        if changed:
            layer.Save()
        item['sha256'] = sha(asset)
        relocations.append({'asset': item['asset'], 'asset_path_attributes_relocated': changed})
    manifest['portable_source_manifest_sha256'] = sha(original_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    profile_path = root / 'cloth_joint_profile_v14.json'
    profile = json.loads(profile_path.read_text())
    profile['bundle'] = str(bundle)
    profile['bundle_manifest_sha256'] = sha(manifest_path)
    profile_path.write_text(json.dumps(profile, indent=2) + '\n')
    for source in visuals.glob('*.py'):
        target = root / source.name
        if source.name == 'wuji2_appearance.py':
            target = root / 'source/dexjoco/dexjoco/sim/isaaclab/wuji2_appearance.py'
        shutil.copyfile(source, target)
    replay = root / 'source/scripts/replay_demos_zarr.py'
    text = replay.read_text()
    text = text.replace('task_root = Path(os.environ["DEXJOCO_WUJI2_VISUAL_REPORT"]).parents[2]',
                        'task_root = Path(os.environ["HANDSON_WUJI2_RUNTIME"])')
    replay.write_text(text)
    receipt = {'source_manifest_sha256': sha(original_path), 'manifest_sha256': sha(manifest_path),
        'profile_sha256': sha(profile_path), 'mdl_sha256': sha(mdl),
        'relocations': relocations, 'change_scope': 'Asset path attributes and runtime path lookup only; no mesh points, normals, UVs, materials or physics values modified.'}
    (root / 'PORTABLE_RUNTIME.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'relocations'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    prepare(parser.parse_args().runtime.resolve(strict=True))
