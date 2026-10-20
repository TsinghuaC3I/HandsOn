#!/usr/bin/env python3
"""Build P0-05's single-ring and baked-scale derivatives of the YCB key.

The source visual contains two flat key blades and one separate, perpendicular
ring component. P0-05 keeps that topology and preserves the ring's tube
thickness. The visual remains instantiated at the task-local 1.2x scale, while
all dynamic collision meshes are baked to 1.2x here so Isaac/PhysX does not
need to simulate a scaled convex mesh.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "handson/handson/sim/envs/xmls/lightwheel_ycb/key_berkeley_meshes"
OUTPUT_ROOT = REPOSITORY_ROOT / "handson/handson/sim/envs/xmls/assets/p005_ycb_key_ring"
RING_RADIAL_EXPANSION_M = 0.006
ASSET_SCALE = 1.2
ALL_COLLIDER_INDICES = range(1, 17)
RING_COLLIDER_INDICES = range(7, 17)


def _vertex(line: str) -> tuple[float, float, float]:
    values = line.split()
    return float(values[1]), float(values[2]), float(values[3])


def _face_vertices(line: str) -> list[int]:
    return [int(token.split("/", 1)[0]) - 1 for token in line.split()[1:]]


def _find_ring_indices(lines: list[str]) -> set[int]:
    vertices = [_vertex(line) for line in lines if line.startswith("v ")]
    parent = list(range(len(vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for line in lines:
        if line.startswith("f "):
            face = _face_vertices(line)
            for index in face[1:]:
                union(face[0], index)
    components: dict[int, set[int]] = {}
    for index in range(len(vertices)):
        components.setdefault(find(index), set()).add(index)
    if len(components) != 3:
        raise RuntimeError(f"expected two key blades and one ring, got {len(components)} components")
    ring = max(components.values(), key=len)
    if min(vertices[index][1] for index in ring) < 0.010:
        raise RuntimeError("the largest YCB key component is not the bow ring")
    return ring


def _expand(point: tuple[float, float, float], center_y: float, center_z: float):
    x, y, z = point
    dy, dz = y - center_y, z - center_z
    radius = (dy * dy + dz * dz) ** 0.5
    if radius <= 1.0e-12:
        raise RuntimeError("ring vertex lies on the expansion center")
    factor = (radius + RING_RADIAL_EXPANSION_M) / radius
    return x, center_y + factor * dy, center_z + factor * dz


def _format(point) -> str:
    return "v " + " ".join(f"{value:.8f}" for value in point)


def _scale(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(value * ASSET_SCALE for value in point)


def main() -> None:
    source = SOURCE_ROOT / "visuals/key_berkeley_meshes.obj"
    lines = source.read_text(encoding="utf-8").splitlines()
    vertices = [_vertex(line) for line in lines if line.startswith("v ")]
    ring = _find_ring_indices(lines)
    center_y = (min(vertices[i][1] for i in ring) + max(vertices[i][1] for i in ring)) / 2
    center_z = (min(vertices[i][2] for i in ring) + max(vertices[i][2] for i in ring)) / 2

    output = [
        "# P0-05 derivative: YCB key body at 1.0 scale; only its existing",
        "# ring component is expanded radially by 0.006 m without thickening.",
    ]
    vertex_index = 0
    for line in lines:
        if line.startswith("v "):
            point = vertices[vertex_index]
            if vertex_index in ring:
                point = _expand(point, center_y, center_z)
            output.append(_format(point))
            vertex_index += 1
        else:
            output.append(line)
    (OUTPUT_ROOT / "visuals").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "visuals/key_berkeley_meshes_single_ring.obj").write_text(
        "\n".join(output) + "\n", encoding="utf-8"
    )

    (OUTPUT_ROOT / "collision").mkdir(parents=True, exist_ok=True)
    for index in ALL_COLLIDER_INDICES:
        source = SOURCE_ROOT / f"collision/key_berkeley_meshes_Collider{index}_m.obj"
        collider = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                point = _vertex(line)
                if index in RING_COLLIDER_INDICES:
                    point = _expand(point, center_y, center_z)
                point = _scale(point)
                collider.append(_format(point))
            else:
                collider.append(line)
        (OUTPUT_ROOT / f"collision/{source.name}").write_text(
            "\n".join(collider) + "\n", encoding="utf-8"
        )
    print(f"wrote {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
