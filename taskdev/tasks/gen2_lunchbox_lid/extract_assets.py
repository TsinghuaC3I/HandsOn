"""Extract existing D15 geometry without designing or adding any visible shape."""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
XML = ROOT / 'handson/handson/sim/envs/xmls'
source = XML / 'd_long_horizon_objects.xml'
root = ET.parse(source).getroot()
asset = ET.Element('mujoco', model='GEN2_D15_lunchbox_assets')
asset.append(ET.Comment('Existing D15 assets; geometry unchanged. MIT, HandsOn contributors. See GEN2_ASSET_PROVENANCE.json.'))
default = ET.SubElement(asset, 'default')
default.append(copy.deepcopy(root.find("./default/default[@class='dlong_object']")))
materials = ET.SubElement(asset, 'asset')
materials.append(copy.deepcopy(root.find("./asset/material[@name='dlong_cyan']")))
world = ET.SubElement(asset, 'worldbody')
records = []
for src, dst, mass, pos in [
    ('d15_box_left', 'gen2_lunchbox', .09, '-.20 .155 .903'),
    ('d15_lid_left', 'gen2_lid', .04, '-.275 -.18 .899'),
]:
    original = root.find(f"./worldbody/body[@name='{src}']")
    body = copy.deepcopy(original)
    body.set('name', dst)
    body.set('pos', pos)
    body.find('freejoint').set('name', dst + '_joint')
    geoms = body.findall('geom')
    source_mass = sum(float(g.get('mass')) for g in geoms)
    parts = []
    for i, g in enumerate(geoms):
        g.set('name', f'{dst}_surface_{i}')
        m = float(g.get('mass')) * mass / source_mass
        g.set('mass', format(m, '.15g'))
        centre = np.fromstring(g.get('pos', '0 0 0'), sep=' ')
        half = np.fromstring(g.get('size'), sep=' ')
        parts.append((m, centre, half))
    com = sum(m * p for m, p, _ in parts) / mass
    inertia = sum(
        m / 3 * (np.sum(h*h) - h*h)
        + m * (np.sum((p-com)**2) - (p-com)**2)
        for m, p, h in parts
    )
    body.insert(0, ET.Element('inertial',
        pos=' '.join(format(v, '.15g') for v in com),
        mass=str(mass), diaginertia=' '.join(format(v, '.15g') for v in inertia)))
    world.append(body)
    # Ignore identity, physics and placement; bind exact authored local geometry.
    signature = [{k:g.get(k, '') for k in ('type','size','pos','quat','euler','material')}
                 for g in original.findall('geom')]
    records.append(dict(source_body=src, body=dst, source_mass_kg=source_mass,
                        mass_kg=mass, centre_of_mass_m=com.tolist(),
                        diagonal_inertia_kg_m2=inertia.tolist(), components=signature,
                        geometry_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()))
ET.indent(asset)
ET.ElementTree(asset).write(XML/'eval_gen01_cover_lunchbox_objects.xml', encoding='utf-8', xml_declaration=True)
scene = ET.parse(XML/'arena_arm_hand_ycb_tasks_rokae_wuji1.xml').getroot()
scene.set('model', 'GEN2_Lunchbox_Lid_Placement')
for inc in list(scene.findall('include')):
    if inc.get('file') != 'rokae_wuji1_bimanual.xml':
        scene.remove(inc)
scene.insert(1, ET.Element('include', file='eval_gen01_cover_lunchbox_objects.xml'))
scene.remove(scene.find('sensor'))
# Remove the unused inherited mocap helper explicitly: the 5.1 importer
# otherwise turns it into an untracked dynamic body, despite its invisible geom.
worldbody = scene.find('worldbody')
for helper in list(worldbody.findall("body[@name='target']")):
    worldbody.remove(helper)
ET.indent(scene)
ET.ElementTree(scene).write(XML/'arena_eval_gen01_cover_lunchbox_rokae_wuji1.xml', encoding='utf-8', xml_declaration=True)
provenance = dict(source_repository='https://github.com/llltttwww/handson',
    source_revision='282390f0fec08daca53e2d6b03f6ad8772b609f4',
    source_file=str(source.relative_to(ROOT)),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    license='MIT', copyright='Copyright (c) 2026 HandsOn contributors',
    license_file='LICENSE', geometry_changes='none', bodies=records)
(XML/'GEN2_ASSET_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(records, indent=2))
