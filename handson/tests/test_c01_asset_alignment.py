from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

_XML_ROOT = (
    Path(__file__).resolve().parents[1] / "handson" / "sim" / "envs" / "xmls"
)

_HAMMER_COLLIDER_NAMES = (
    "handle",
    "hammer_collision_02",
    "hammer_collision_03",
    "hammer_collision_04",
    "head",
    "hammer_collision_06",
    "neck",
    "face",
    "hammer_collision_09",
    "hammer_collision_10",
    "claw",
    "hammer_collision_12",
)


def _named_elements(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    return {
        element.attrib["name"]: element
        for element in root.iter(tag)
        if "name" in element.attrib
    }


def _mesh_geom_points(model: mujoco.MjModel, geom_name: str) -> np.ndarray:
    geom_id = model.geom(geom_name).id
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_adr = int(model.mesh_vertadr[mesh_id])
    vertex_num = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_adr : vertex_adr + vertex_num],
        dtype=np.float64,
    )
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    return vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]


def _mesh_bounds(
    model: mujoco.MjModel,
    geom_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [_mesh_geom_points(model, geom_name) for geom_name in geom_names],
        axis=0,
    )
    return points.min(axis=0), points.max(axis=0)


def test_hammer_uses_all_ycb_convex_colliders_at_the_visual_transform() -> None:
    root = ET.parse(_XML_ROOT / "hammer.xml").getroot()
    meshes = _named_elements(root, "mesh")
    geoms = _named_elements(root, "geom")
    visual = geoms["hammer_visual"]

    assert visual.attrib["quat"] == "0.7071068 0.7071068 0 0"
    assert visual.attrib["pos"] == "0 0 0.0308"
    assert geoms["face"].attrib["mesh"] == "c01_ycb_hammer_col_08"

    for index, geom_name in enumerate(_HAMMER_COLLIDER_NAMES, start=1):
        mesh_name = f"c01_ycb_hammer_col_{index:02d}"
        assert meshes[mesh_name].attrib["file"] == (
            "lightwheel_ycb/hammer/collision/"
            f"hammer_Collider{index}_m.obj"
        )
        collider = geoms[geom_name]
        assert collider.attrib["type"] == "mesh"
        assert collider.attrib["mesh"] == mesh_name
        assert collider.attrib["pos"] == visual.attrib["pos"]
        assert collider.attrib["quat"] == visual.attrib["quat"]
        assert float(collider.attrib["density"]) > 0.0
        assert collider.attrib["rgba"] == "0 0 0 0"
        assert collider.attrib["contype"] == "3"
        assert collider.attrib["conaffinity"] == "3"


def test_nail_ycb_visual_and_collider_share_scale_and_transform() -> None:
    root = ET.parse(_XML_ROOT / "c01_wood_nail.xml").getroot()
    meshes = _named_elements(root, "mesh")
    geoms = _named_elements(root, "geom")
    visual = geoms["nail_visual"]
    collider = geoms["nail_head"]

    assert meshes["c01_ycb_nail_collision"].attrib["file"] == (
        "lightwheel_ycb/a_Nails/collision/a_Nails_Collider1_m.obj"
    )
    assert meshes["c01_ycb_nail_collision"].attrib["scale"] == (
        meshes["c01_ycb_nail_vis"].attrib["scale"]
    )
    assert collider.attrib["type"] == "mesh"
    assert collider.attrib["mesh"] == "c01_ycb_nail_collision"
    assert collider.attrib["pos"] == visual.attrib["pos"]
    assert collider.attrib["quat"] == visual.attrib["quat"]
    assert float(collider.attrib["density"]) > 0.0
    assert collider.attrib["rgba"] == "0 0 0 0"

    compatibility_shaft = geoms["nail_shaft"]
    assert compatibility_shaft.attrib["mass"] == "0"
    assert compatibility_shaft.attrib["size"] == "0.0001 0.0001"
