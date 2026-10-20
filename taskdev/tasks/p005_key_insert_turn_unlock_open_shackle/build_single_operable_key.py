#!/usr/bin/env python3
"""Keep the primary YCB blade and original rear ring, with UVs unchanged.

The archived single_ring derivative actually contains two rigid blades. Its
spare blade cannot enter the functional keyway alongside the primary blade.
This derivative removes that whole connected component, not just its collider.
No source asset or ring dimensions are changed. Run from any working directory.
"""
from pathlib import Path
import hashlib
import json

REPO = Path(__file__).resolve().parents[3]
ASSET_ROOT = REPO / "handson/handson/sim/envs/xmls/assets/p005_ycb_key_ring"
SOURCE = ASSET_ROOT / "visuals/key_berkeley_meshes_single_ring.obj"
DESTINATION = ASSET_ROOT / "visuals/key_berkeley_meshes_single_operable_key.obj"


def build_text(text: str) -> tuple[str, dict]:
    lines = text.splitlines()
    vertices = [tuple(map(float, line.split()[1:4])) for line in lines if line.startswith("v ")]
    parent = list(range(len(vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    faces = [[int(token.split("/")[0]) - 1 for token in line.split()[1:]]
             for line in lines if line.startswith("f ")]
    for face in faces:
        for i in face[1:]:
            parent[find(i)] = find(face[0])
    components = {}
    for i in range(len(vertices)):
        components.setdefault(find(i), set()).add(i)
    if len(components) != 3:
        raise ValueError(f"Expected the audited two blades + ring, got {len(components)}")
    ring = max(components.values(), key=lambda c: max(vertices[i][2] for i in c) - min(vertices[i][2] for i in c))
    blades = [c for c in components.values() if c is not ring]
    primary = min(blades, key=lambda c: sum(vertices[i][2] for i in c) / len(c))
    spare = next(c for c in blades if c is not primary)
    if not (max(vertices[i][2] for i in primary) < 0 < min(vertices[i][2] for i in spare)):
        raise ValueError("Source blade identity changed; re-audit before generating")
    keep = ring | primary
    remap = {i: n + 1 for n, i in enumerate(sorted(keep))}
    output = ["# P0-05: one operable primary YCB key and its original rear ring.",
              "# Spare rigid blade and its faces removed; retained positions/UVs/normals unchanged."]
    vertex_index = 0
    kept_faces = 0
    for line in lines:
        if line.startswith("#"):
            continue
        if line.startswith("v "):
            if vertex_index in keep:
                output.append(line)
            vertex_index += 1
        elif line.startswith("f "):
            tokens = [token.split("/") for token in line.split()[1:]]
            indices = [int(token[0]) - 1 for token in tokens]
            if all(i in keep for i in indices):
                for token, i in zip(tokens, indices):
                    token[0] = str(remap[i])
                output.append("f " + " ".join("/".join(token) for token in tokens))
                kept_faces += 1
            elif any(i in keep for i in indices):
                raise ValueError("Unexpected face shared between source components")
        else:
            output.append(line)
    return "\n".join(output) + "\n", {
        "source_vertices": len(vertices), "retained_vertices": len(keep),
        "removed_spare_vertices": len(spare), "source_faces": len(faces),
        "retained_faces": kept_faces, "retained_components": 2,
        "retained_ring_vertices": len(ring), "ring_geometry_changed": False,
        "excluded_collision_mesh": "p005_key_collider03",
    }


def main():
    source = SOURCE.read_bytes()
    text, report = build_text(source.decode())
    DESTINATION.write_text(text, encoding="utf-8")
    report.update(source=str(SOURCE.relative_to(REPO)), output=str(DESTINATION.relative_to(REPO)),
                  source_sha256=hashlib.sha256(source).hexdigest(),
                  output_sha256=hashlib.sha256(text.encode()).hexdigest())
    (ASSET_ROOT / "single_operable_key_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
