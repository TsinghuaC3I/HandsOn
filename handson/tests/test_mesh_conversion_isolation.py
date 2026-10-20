"""Isaac's sidecar conversion must never write beside shared source meshes."""
import ast
import hashlib
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def test_first_use_and_duplicate_meshes_have_private_sidecar_directories(tmp_path):
    # Load the file-only preparation helpers without starting Isaac Sim.
    source = Path(__file__).parents[1] / 'handson/sim/isaaclab/task_preview.py'
    module = ast.parse(source.read_text())
    names = {'_alias_duplicate_mesh_sources', '_missing_obj_material_libraries',
             '_write_missing_obj_material_libraries'}
    module.body = [node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Path=Path, ET=ET, hashlib=hashlib, shutil=shutil, os=os)
    exec(compile(module, str(source), 'exec'), namespace)
    shared = tmp_path / 'shared'
    shared.mkdir()
    mesh = shared / 'body.obj'
    mesh.write_text('mtllib material.mtl\nv 0 0 0\n')
    (shared / 'material.mtl').write_text('newmtl body\n')
    (shared / 'body_tmp').mkdir()
    (shared / 'body_tmp/stale.usd').write_text('must not propagate conversion residue')
    outputs = []
    for index in range(2):
        cache = tmp_path / f'attempt{index}'
        cache.mkdir()
        root = ET.fromstring(f'<asset><mesh name="one" file="{mesh}"/>'
                             f'<mesh name="two" file="{mesh}"/></asset>')
        namespace['_alias_duplicate_mesh_sources'](root, cache)
        paths = [(cache / node.attrib['file']).resolve() for node in root.iter('mesh')]
        assert len(set(paths)) == 2
        for path in paths:
            assert path.is_relative_to(cache)
            assert path.read_bytes() == mesh.read_bytes()
            assert (path.parent / 'material.mtl').is_file()
            assert not (path.parent / 'body_tmp').exists()
        outputs.extend(paths)
    assert len(set(outputs)) == 4


def test_center_rig_meshes_are_converted_inside_the_attempt_cache(tmp_path):
    package = Path(__file__).parents[1] / 'handson/sim/isaaclab'
    namespace = dict(Path=Path, ET=ET, shutil=shutil, os=os)
    scene = ast.parse((package / 'bimanual_assembly_scene.py').read_text())
    scene.body = [node for node in scene.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_prepare_center_mjcf']
    shared = tmp_path / 'shared'
    shared.mkdir()
    body, head = shared / 'body.stl', shared / 'head.stl'
    body.write_bytes(b'body mesh')
    head.write_bytes(b'head mesh')
    namespace.update(__package__='handson.sim.isaaclab', _CACHE_DIR=tmp_path / 'attempt',
                     _CENTER_BODY_MESH=body, _CENTER_HEAD_MESH=head)
    exec(compile(scene, 'center_scene', 'exec'), namespace)
    xml = namespace['_prepare_center_mjcf']()
    meshes = list(ET.parse(xml).getroot().iter('mesh'))
    assert len(meshes) == 2
    for mesh, original in zip(meshes, [body, head]):
        resolved = (xml.parent / mesh.attrib['file']).resolve()
        assert resolved.is_relative_to(tmp_path / 'attempt')
        assert resolved.read_bytes() == original.read_bytes()
    assert sorted(p.name for p in shared.iterdir()) == ['body.stl', 'head.stl']
