"""The micropipette helper anchors retain their paper asset positions."""
import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import handson


source = Path(handson.__file__).parent / 'sim/isaaclab/task_preview.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == '_apply_t18_paper_target_positions')
namespace = {'ET': ET}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
apply_positions = namespace[node.name]


def asset():
    return ET.parse(source.parents[1] / 'envs/xmls/arena_arm_hand_lw_native_articulation_rokae_wuji1.xml').getroot()


def test_paper_helpers_change_without_moving_scene_objects():
    root = asset()
    expected = ET.fromstring(ET.tostring(root))
    # Original arena f1306e34, loaded by the bb00df54 paper source.
    expected.find(".//body[@name='target_right']").set('pos', '-.10 -.18 1.16')
    expected.find(".//body[@name='target_left']").set('pos', '-.10 .18 1.16')
    assert apply_positions(root, 't18_open_micropipette_box') == 2
    assert ET.tostring(root) == ET.tostring(expected)


@pytest.mark.parametrize('task', ['t48_unfold_fan', 't49_unfold_ruler'])
def test_other_articulated_tasks_retain_their_source_asset(task):
    root = asset()
    before = ET.tostring(root)
    assert apply_positions(root, task) == 0
    assert ET.tostring(root) == before


def test_missing_paper_anchor_is_reported():
    with pytest.raises(ValueError, match='missing its paper helper'):
        apply_positions(ET.fromstring('<mujoco/>'), 't18_open_micropipette_box')
