"""The Core box collider must match its receipt-bound paper asset."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

import handson


# Exercise the actual XML transformer without importing IsaacLab's GPU-only
# scene bootstrap. Its complete function body depends only on ElementTree.
scene_path = Path(handson.__file__).parent / 'sim/isaaclab/task_preview.py'
tree = ast.parse(scene_path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == '_apply_b05_paper_lid_collision')
namespace = {'ET': ET}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(scene_path), 'exec'), namespace)
scene = SimpleNamespace(_apply_b05_paper_lid_collision=namespace[node.name])


def asset():
    return ET.parse(scene_path.parents[1] / 'envs/xmls/a_series_ycb_task_objects.xml').getroot()


def test_core_restores_paper_box_without_changing_other_bodies_or_visuals():
    root = asset()
    before = ET.tostring(root)
    expected = ET.fromstring(before)
    expected.find(".//geom[@name='Box_lid_collision_1']").attrib = {
        'name': 'Box_lid_collision_1', 'class': 'a_series_ycb_collision',
        'type': 'box', 'size': '0.146 0.2145 0.010', 'mass': '0.04',
    }
    assert scene._apply_b05_paper_lid_collision(root, 't03_cover_box') == 1
    assert ET.canonicalize(ET.tostring(root, encoding='unicode')) == ET.canonicalize(
        ET.tostring(expected, encoding='unicode'))
    assert scene._apply_b05_paper_lid_collision(root, 't03_cover_box') == 1
    assert root.find(".//geom[@name='Box_lid_visual']").get('mesh') == 'Box_lid_vis'


@pytest.mark.parametrize('task', ['eval_comb03_pass_block_and_close_box', 't34_roll_ball_through_gate'])
def test_other_scenes_keep_their_own_asset_contract(task):
    root = asset()
    before = ET.tostring(root)
    assert scene._apply_b05_paper_lid_collision(root, task) == 0
    assert ET.tostring(root) == before


def test_missing_core_collider_fails_instead_of_silently_changing_physics():
    with pytest.raises(ValueError, match='missing its paper lid collider'):
        scene._apply_b05_paper_lid_collision(ET.fromstring('<mujoco/>'), 't03_cover_box')
