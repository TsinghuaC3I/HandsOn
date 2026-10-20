"""Restore the existing GEN3 GLB's PBR defaults without changing its geometry.

The original import treats missing trimesh metallic/roughness values as 0/0.5.
glTF specifies 1/1 instead. Keep the existing gray/black RGB values, bindings,
and source USD intact; author only the omitted PBR factors in the live layer.
The shared visual profile owns robot, table, lighting and render validation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
import struct

from ...sim.isaaclab.gen3_hinged_trash_bin_scene import (
    GEN3_ASSET_DIRECTORY,
    GEN3_SOURCE_UID,
    GEN3_SOURCE_URL,
    GEN3_TASK_NAME,
)

_PROFILE_ENV = "HANDSON_ISAACSIM_TASK_VISUAL_PROFILE"
_SOURCE_SHA256 = "99274599f28b909fb5d56cf8a4646c433037559b4969751d8aee275b6a5d9626"
_PBR_SCHEMA_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF/main/"
    "specification/2.0/schema/material.pbrMetallicRoughness.schema.json"
)


@lru_cache(maxsize=1)
def _source_contract() -> tuple[dict, ...]:
    path = GEN3_ASSET_DIRECTORY / "source" / f"{GEN3_SOURCE_UID}.glb"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _SOURCE_SHA256:
        raise ValueError("GEN3 visual source changed; inspect its PBR contract")
    magic, version, length = struct.unpack_from("<4sII", data)
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if (magic, version, length, chunk_type) != (b"glTF", 2, len(data), 0x4E4F534A):
        raise ValueError("GEN3 visual source is not the expected glTF 2 GLB")
    gltf = json.loads(data[20 : 20 + chunk_length])
    rows = []
    for index, mesh in enumerate(gltf["meshes"]):
        if len(mesh["primitives"]) != 1:
            raise ValueError("GEN3 source mesh/material mapping changed")
        material_index = mesh["primitives"][0]["material"]
        material = gltf["materials"][material_index]
        pbr = material.get("pbrMetallicRoughness", {})
        rows.append({
            "source_mesh_index": index,
            "source_material_index": material_index,
            "source_material_name": material.get("name", ""),
            "source_rgb": pbr.get("baseColorFactor", [1.0] * 4)[:3],
            "metallic": float(pbr.get("metallicFactor", 1.0)),
            "roughness": float(pbr.get("roughnessFactor", 1.0)),
            "omitted_source_factors": [
                key for key in ("metallicFactor", "roughnessFactor") if key not in pbr
            ],
        })
    return tuple(rows)


def apply_visual_appearance(stage, task_root_path="/World/Task") -> dict:
    """Return JSON-safe provenance; change only GEN3 surface PBR inputs.

    Validate all eight bindings before authoring, so a changed scene cannot
    silently use unrelated material paths or receive a partial override.
    ``source`` preserves the existing runtime USD for independent-process A/B.
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade

    profile = os.environ.get(_PROFILE_ENV, "real_v12").strip().lower()
    if profile not in {"source", "real_v12"}:
        raise ValueError(f"Unsupported GEN3 task visual profile: {profile!r}")
    asset_path = str(task_root_path).rstrip("/") + "/LWNativeArticulation/trash_bin"
    source_rows = _source_contract()
    pending = []
    for source in source_rows:
        index = source["source_mesh_index"]
        body = "trash_bin_lid" if index in (1, 2) else "trash_bin_base"
        prim_path = f"{asset_path}/{body}/Visuals/SourceMesh{index:02d}"
        prim = stage.GetPrimAtPath(prim_path)
        if (not prim or not prim.IsA(UsdGeom.Mesh)
                or prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.GetCustomDataByKey("handson:sourceNode") != f"SourceMesh{index:02d}"):
            raise ValueError(f"GEN3 source visual is missing or changed: {prim_path}")
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        material_path = f"{asset_path}/Materials/Source{index}"
        if not material or str(material.GetPath()) != material_path:
            raise ValueError(f"GEN3 source visual material binding changed: {prim_path}")
        shader = UsdShade.Shader(stage.GetPrimAtPath(material_path + "/Surface"))
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            raise ValueError(f"GEN3 source surface shader changed: {material_path}")
        inputs = {key: shader.GetInput(key) for key in ("diffuseColor", "metallic", "roughness")}
        if any(not value or value.HasConnectedSource() for value in inputs.values()):
            raise ValueError(f"GEN3 expected constant PBR inputs: {material_path}")
        before = {
            "diffuseColor": [float(value) for value in inputs["diffuseColor"].Get()],
            "metallic": float(inputs["metallic"].Get()),
            "roughness": float(inputs["roughness"].Get()),
        }
        pending.append((source, prim_path, shader, inputs, before))

    changes, materials = [], []
    for source, prim_path, shader, inputs, before in pending:
        for key in ("metallic", "roughness"):
            target = source[key]
            if profile == "real_v12" and not math.isclose(before[key], target, abs_tol=1e-7):
                inputs[key].Set(target)
                changes.append({
                    "prim_path": str(shader.GetPath()),
                    "attribute": "inputs:" + key,
                    "before": before[key],
                    "after": float(inputs[key].Get()),
                    "reason": "Restore effective source glTF PBR factor, including its specified default",
                })
        materials.append({
            **source,
            "matched_prim": prim_path,
            "material_path": str(shader.GetPath().GetParentPath()),
            "shader_path": str(shader.GetPath()),
            "before": before,
            "after": {
                "diffuseColor": before["diffuseColor"],
                "metallic": float(inputs["metallic"].Get()),
                "roughness": float(inputs["roughness"].Get()),
            },
        })
    return {
        "task_name": GEN3_TASK_NAME,
        "profile": profile,
        "policy": "source_gltf_pbr_defaults_v1" if profile == "real_v12" else "preserve_runtime_source",
        "source": {"url": GEN3_SOURCE_URL, "glb_sha256": _SOURCE_SHA256, "pbr_schema": _PBR_SCHEMA_URL},
        "matched_prim_paths": [row[1] for row in pending],
        "materials": materials,
        "changed_attributes": changes,
        "preserved": ["gray_and_black_RGB", "source_textures_absent", "material_bindings", "mesh_geometry", "all_physics_attributes"],
        "native_visual_validation": "pending_independent_process_three_camera_AB",
    }
