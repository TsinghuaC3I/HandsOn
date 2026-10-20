"""Read the licensed OBJ groups with the companion glTF's original materials."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import trimesh

TASK_DIR = Path(__file__).resolve().parent


def read_source():
    vertices, faces, labels, names = [], [], [], []
    for line in (TASK_DIR / "source/audio_mixer/03 test.obj").open():
        row = line.split()
        if not row:
            continue
        if row[0] == "v":
            vertices.append(list(map(float, row[1:4])))
        elif row[0] == "g":
            names.append(row[1])
        elif row[0] == "f":
            assert len(row) == 4, "Source must remain the original triangulated OBJ"
            faces.append([int(i.split('/')[0]) - 1 for i in row[1:]])
            labels.append(len(names) - 1)
    vertices = np.asarray(vertices)
    faces, labels = np.asarray(faces), np.asarray(labels)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = area > 0.0
    removed_degenerate = int((~valid).sum())
    # Sketchfab omits exactly the 60 zero-area OBJ faces in its glTF export.
    faces, labels = faces[valid], labels[valid]
    tree = cKDTree(vertices[faces].mean(axis=1))
    scene = trimesh.load(TASK_DIR / "source/audio_mixer_gltf/scene.gltf",
                         force="scene", process=False)
    groups = {name: [] for name in names}
    max_error, face_count = 0.0, 0
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        original = scene.geometry[key]
        if not isinstance(original, trimesh.Trimesh):
            continue  # Original OBJ guide lines are not renderable surfaces.
        mesh = original.copy()
        mesh.apply_transform(transform)
        distance, match = tree.query(mesh.triangles_center)
        max_error = max(max_error, float(distance.max()))
        assert distance.max() < 0.001, "glTF geometry does not match the source OBJ"
        group_ids = labels[match]
        for group_id in np.unique(group_ids):
            selected = np.flatnonzero(group_ids == group_id)
            source_faces = mesh.faces[selected]
            used, inverse = np.unique(source_faces, return_inverse=True)
            visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv[used] if mesh.visual.uv is not None else None,
                material=mesh.visual.material)
            part = trimesh.Trimesh(vertices=mesh.vertices[used],
                                   faces=inverse.reshape(-1, 3),
                                   vertex_normals=mesh.vertex_normals[used],
                                   visual=visual, process=False)
            groups[names[group_id]].append(part)
            face_count += len(part.faces)
    assert face_count == len(faces), "Source face count changed during material matching"
    return groups, {"obj_faces": len(faces) + removed_degenerate,
                    "source_zero_area_faces_omitted_by_gltf": removed_degenerate,
                    "gltf_faces": face_count,
                    "group_count": len(names), "max_centroid_match_error_source_units": max_error}


if __name__ == "__main__":
    groups, report = read_source()
    for name in ("Cylinder098", "Cylinder115", "Box016"):
        mesh = trimesh.util.concatenate(groups[name])
        mesh.merge_vertices()
        parts = mesh.split(only_watertight=False, repair=False)
        report[name] = {"bounds": mesh.bounds.tolist(), "components": [
            {"faces": len(p.faces), "bounds": p.bounds.tolist(), "watertight": p.is_watertight}
            for p in parts if len(p.faces) > 8]}
    print(json.dumps(report, indent=2))
    (TASK_DIR / "evidence/source_material_matching.json").write_text(json.dumps(report, indent=2))
