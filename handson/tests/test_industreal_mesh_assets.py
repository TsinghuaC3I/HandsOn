from __future__ import annotations

from pathlib import Path

import pytest


_MESH_ROOT = (
    Path(__file__).resolve().parents[1]
    / "handson"
    / "sim"
    / "envs"
    / "xmls"
    / "industreal"
    / "mesh"
    / "industreal_franka_table"
)
_FINGER_MESHES = (
    _MESH_ROOT / "industreal_franka_finger_rubber_coll.obj",
    _MESH_ROOT / "industreal_franka_finger_rubber_viz.obj",
)


@pytest.mark.parametrize("mesh_path", _FINGER_MESHES, ids=lambda path: path.name)
def test_industreal_finger_mesh_is_resolved_and_index_valid(
    mesh_path: Path,
) -> None:
    text = mesh_path.read_text(encoding="utf-8")
    assert "<<<<<<<" not in text
    assert "=======" not in text
    assert ">>>>>>>" not in text
    assert "git-lfs.github.com/spec" not in text

    vertex_count = 0
    normal_count = 0
    faces: list[list[str]] = []
    for raw_line in text.splitlines():
        fields = raw_line.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            assert len(fields) == 4
            tuple(float(value) for value in fields[1:])
            vertex_count += 1
        elif fields[0] == "vn":
            assert len(fields) == 4
            tuple(float(value) for value in fields[1:])
            normal_count += 1
        elif fields[0] == "f":
            assert len(fields) >= 4
            faces.append(fields[1:])

    assert vertex_count > 0
    assert faces
    for face in faces:
        for token in face:
            parts = token.split("/")
            vertex_index = int(parts[0])
            assert 1 <= vertex_index <= vertex_count
            if len(parts) >= 3 and parts[2]:
                normal_index = int(parts[2])
                assert 1 <= normal_index <= normal_count
