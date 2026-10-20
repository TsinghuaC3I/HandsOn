"""Extract exact GLB material/image data without touching task mesh assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "handson/handson/sim/envs/usd/eval_gen06_spray_shoe"
OUTPUT = ASSETS / "appearance_source"


def extract(name, mesh_nodes):
    data = (ASSETS / "sources" / name).read_bytes()
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    document = json.loads(data[20:20 + length])
    binary_length, binary_kind = struct.unpack_from("<II", data, 20 + length)
    assert binary_kind == 0x004E4942
    binary = data[28 + length:28 + length + binary_length]
    output = {}
    for destination, source_node in mesh_nodes.items():
        node = next(n for n in document["nodes"] if n.get("name") == source_node)
        primitives = document["meshes"][node["mesh"]]["primitives"]
        assert len(primitives) == 1
        material = document["materials"][primitives[0]["material"]]
        textures = {}
        def walk(value):
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                if key.endswith("Texture") and isinstance(child, dict) and "index" in child:
                    texture = document["textures"][child["index"]]
                    image_index = texture["source"]
                    image = document["images"][image_index]
                    view = document["bufferViews"][image["bufferView"]]
                    payload = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
                    extension = {"image/png": "png", "image/jpeg": "jpg"}[image["mimeType"]]
                    filename = f"{Path(name).stem}_{image_index}.{extension}"
                    (OUTPUT / filename).write_bytes(payload)
                    textures[key] = {"file": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                                     "source_texture_info": child,
                                     "sampler": document.get("samplers", [{}])[texture.get("sampler", 0)]}
                else:
                    walk(child)
        walk(material)
        output[destination] = {"source_glb": name, "source_node": source_node,
                               "material_index": primitives[0]["material"],
                               "material": material, "textures": textures}
    return output


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    materials = extract("spray_bottle.glb", {
        "gen7_bottle_0": "Cube_Spray Body_0",
        "gen7_bottle_1": "Plane_Spray Top_0",
        "gen7_bottle_2": "Cylinder_Spray Inside_0",
        "gen7_trigger": "Plane.001_Spray Top_0",
    })
    materials.update(extract("sneakers.glb", {f"gen7_shoe_object_{i}": f"Object_{i}" for i in range(81, 100, 2)}))
    report = {"profile": "gen7_source_pbr_v1", "uv_convention": "Existing OBJ st = (gltf_u, 1-gltf_v); transform is F*T_gltf*F",
              "geometry_modified": False, "source_sha256": {
                  name: hashlib.sha256((ASSETS / "sources" / name).read_bytes()).hexdigest()
                  for name in ("spray_bottle.glb", "sneakers.glb")}, "materials": materials}
    (OUTPUT / "materials.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
