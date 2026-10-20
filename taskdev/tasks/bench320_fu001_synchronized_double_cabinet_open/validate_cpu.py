#!/usr/bin/env python3
"""CPU-only fail-closed checks for the FU001 core-scene package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from taskdev.manifest import ManifestValidator
from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
    DOOR_JOINTS,
    EXPECTED_SHA256,
    REPOSITORY_ROOT,
    SELECTED_ASSET_ARCHIVE_RELATIVE,
    SELECTED_ASSET_ARCHIVE_SHA256,
    SELECTED_ASSET_COMPONENTS,
    SELECTED_ASSET_MANIFEST_RELATIVE,
    SELECTED_ASSET_RUNTIME_RELATIVE,
    SELECTED_ASSET_SOURCE_RELATIVE,
    SOURCE_ROWS_RELATIVE,
    TASK_DIR,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_asset_check() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the reviewed source closure and canonical runtime outputs.

    This deliberately stays CPU-only: USD composition and PhysX cooking are
    separate live gates.  The check nevertheless binds every runtime file to
    the downloaded archive and rejects a silently substituted old bundle.
    """

    archive = REPOSITORY_ROOT / SELECTED_ASSET_ARCHIVE_RELATIVE
    source_dir = REPOSITORY_ROOT / SELECTED_ASSET_SOURCE_RELATIVE
    manifest_path = REPOSITORY_ROOT / SELECTED_ASSET_MANIFEST_RELATIVE
    if not archive.is_file():
        return {"pass": False, "error": f"missing archive: {archive}"}, {}
    actual_archive_hash = _sha256(archive)
    if actual_archive_hash != SELECTED_ASSET_ARCHIVE_SHA256:
        return {
            "pass": False,
            "error": "selected archive SHA-256 mismatch",
            "expected_archive_sha256": SELECTED_ASSET_ARCHIVE_SHA256,
            "actual_archive_sha256": actual_archive_hash,
        }, {}
    if not manifest_path.is_file():
        return {"pass": False, "error": f"missing runtime manifest: {manifest_path}"}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("provenance", {})
    provenance_pass = bool(
        provenance.get("archive_sha256") == SELECTED_ASSET_ARCHIVE_SHA256
        and provenance.get("source_url") == "https://polyhaven.com/a/painted_wooden_cabinet"
        and provenance.get("license") == "CC0 1.0"
    )
    members = manifest.get("source", {}).get("members", [])
    source_pass = True
    source_hashes: dict[str, str] = {}
    for member in members:
        relative = str(member.get("path", ""))
        path = source_dir / relative
        if not relative or not path.is_file():
            source_pass = False
            continue
        actual = _sha256(path)
        source_hashes[relative] = actual
        if actual != member.get("sha256") or path.stat().st_size != member.get("size_bytes"):
            source_pass = False
    components = manifest.get("runtime_components", [])
    runtime_pass = True
    runtime_hashes: dict[str, str] = {}
    for logical, relative_path in SELECTED_ASSET_COMPONENTS.items():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file() or path.stat().st_size <= 1024:
            runtime_pass = False
            continue
        actual = _sha256(path)
        runtime_hashes[str(relative_path)] = actual
        expected = next(
            (entry.get("sha256") for entry in components if entry.get("path") == path.name),
            None,
        )
        if expected != actual:
            runtime_pass = False
    texture_pass = all(
        (REPOSITORY_ROOT / SELECTED_ASSET_RUNTIME_RELATIVE / "textures" / name).is_file()
        and (REPOSITORY_ROOT / SELECTED_ASSET_RUNTIME_RELATIVE / "textures" / name).stat().st_size > 0
        for name in (
            "painted_wooden_cabinet_diff_4k.jpg",
            "painted_wooden_cabinet_nor_gl_4k.jpg",
            "painted_wooden_cabinet_arm_4k.jpg",
        )
    )
    structure = manifest.get("source", {})
    scene_frame = manifest.get("scene_frame", {})
    geometry_pass = bool(
        set(structure.get("independent_nodes", []))
        >= {
            "/World/painted_wooden_cabinet",
            "/World/painted_wooden_cabinet_door_01",
            "/World/painted_wooden_cabinet_door_02",
        }
        and structure.get("source_up_axis") == "Y"
        and structure.get("hinge_axis_source") == "+Y"
        and structure.get("hinge_axis_runtime") == "+Z after +90 degrees about X"
        and scene_frame.get("converter_source_to_runtime")
        == "(x,y,z) -> (x,-z,y)"
        and scene_frame.get("extra_rotation") == "Rz(-90 degrees)"
        and scene_frame.get("source_to_world") == "(x,y,z) -> (-z,-x,y)"
        and scene_frame.get("orientation_wxyz")
        == [0.5, 0.5, -0.5, -0.5]
    )
    collision_pass = bool(
        manifest.get("collision", {}).get("placeholder_primitives") is False
        and manifest.get("collision", {}).get("approximation") == "convexDecomposition"
        and all(entry.get("mesh_count", 0) > 0 for entry in components)
    )
    checks = {
        "archive_hash": actual_archive_hash == SELECTED_ASSET_ARCHIVE_SHA256,
        "provenance": provenance_pass,
        "source_closure": source_pass,
        "runtime_components": runtime_pass,
        "runtime_textures": texture_pass,
        "source_geometry": geometry_pass,
        "collision_derivation": collision_pass,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "archive_sha256": actual_archive_hash,
        "source_hashes": source_hashes,
        "runtime_hashes": runtime_hashes,
    }, manifest


def validate() -> dict[str, Any]:
    hashes = {
        relative: _sha256(REPOSITORY_ROOT / relative)
        for relative in EXPECTED_SHA256
    }
    hash_pass = all(
        hashes[path] == EXPECTED_SHA256[path] for path in EXPECTED_SHA256
    )
    meta = json.loads(
        (TASK_DIR / "assets/dv_double_cabinet_44781/meta.json").read_text(
            encoding="utf-8"
        )
    )
    slots = meta.get("slots", [])
    real_revolute = {
        slot.get("slot_joint")
        for slot in slots
        if slot.get("type") == "revolute" and slot.get("is_dummy") is False
    }
    source_rows = json.loads(
        (REPOSITORY_ROOT / SOURCE_ROWS_RELATIVE).read_text(encoding="utf-8")
    )
    source = next(
        item for item in source_rows["tasks"] if item["task_id"] == "B320-FU001"
    )
    manifest_path = TASK_DIR / "task.yaml"
    manifest_result = ManifestValidator(REPOSITORY_ROOT).validate_path(manifest_path)
    runtime_source = (TASK_DIR / "scene_runtime.py").read_text(encoding="utf-8")
    spec_source = (TASK_DIR / "scene_spec.py").read_text(encoding="utf-8")
    bound_scene_source = runtime_source + "\n" + spec_source
    forbidden = [
        token
        for token in ("omni.isaac.core", "/home/", "C:\\Users\\")
        if token in bound_scene_source
    ]
    required_runtime_tokens = (
        "AppLauncher",
        "RayTracedLighting",
        "DomeLight",
        "DistantLight",
        "target_joint_revolute",
        "joint_11",
        "task_complete",
        "core_scene_pass",
    )
    missing_runtime_tokens = [
        token for token in required_runtime_tokens if token not in bound_scene_source
    ]
    selected_asset, selected_manifest = _selected_asset_check()
    checks = {
        "asset_hashes": hash_pass,
        "source_metadata_two_real_revolute_joints": real_revolute == set(DOOR_JOINTS),
        "authoritative_source_row": bool(
            source.get("excel_row") == 215
            and source.get("left_role") == "驱动左门target_joint_revolute"
            and source.get("right_role") == "驱动右门joint_11"
            and source.get("exact_assets", "").startswith(
                "dv_double_cabinet_44781: "
            )
        ),
        "manifest": manifest_result.valid,
        "runtime_structure": not forbidden and not missing_runtime_tokens,
        "selected_asset_provenance": bool(selected_asset.get("pass")),
        "selected_runtime_closure": bool(
            selected_asset.get("checks", {}).get("runtime_components", False)
            and selected_asset.get("checks", {}).get("runtime_textures", False)
        ),
        "selected_source_geometry_and_collision": bool(
            selected_asset.get("checks", {}).get("source_geometry", False)
            and selected_asset.get("checks", {}).get("collision_derivation", False)
        ),
    }
    return {
        "schema": "fu001_cpu_contract_v1",
        "task_id": "B320-FU001",
        "checks": checks,
        "pass": all(checks.values()),
        "asset_hashes": hashes,
        "real_revolute_joint_names": sorted(real_revolute),
        "authoritative_source_row": source,
        "manifest": manifest_result.as_dict(),
        "forbidden_runtime_tokens": forbidden,
        "missing_runtime_tokens": missing_runtime_tokens,
        "selected_asset": selected_asset,
        "selected_asset_manifest": selected_manifest,
        "task_complete": False,
        "live_isaac_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate()
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
