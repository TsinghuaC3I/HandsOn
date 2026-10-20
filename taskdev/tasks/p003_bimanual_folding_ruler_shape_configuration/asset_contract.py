"""Pinned identity and topology contract for the self-authored Ruler035 bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct


TASK_RELATIVE_ROOT = Path(
    "taskdev/tasks/p003_bimanual_folding_ruler_shape_configuration"
)
ASSET_RELATIVE_ROOT = TASK_RELATIVE_ROOT / "assets/Ruler035"

ASSETS: dict[str, dict[str, object]] = {
    "Ruler035.usd": {
        "size": 33_679,
        "sha256": "55f3823f9ae46ba34f5bfcc12332aaa512cf99eb344e2db8bfa92a8825ca0369",
        "kind": "usd",
    },
    "texture/T_Ruler035_BC001.png": {
        "size": 2_316_649,
        "sha256": "39d63de657476f0e9a90d1a3aa7538a90f7f2654748aaf29c89b06f364178fbf",
        "kind": "texture",
        "dimensions": (2048, 2048),
    },
    "texture/T_Ruler035_N001.png": {
        "size": 28_323,
        "sha256": "e2f800d12ca234ae2adbc7f8b79fa58998e0bb9a219cd43f31029d8fdfeb1ff8",
        "kind": "texture",
        "dimensions": (2048, 2048),
    },
    "texture/T_Ruler035_ORM001.png": {
        "size": 28_298,
        "sha256": "3d6fef33e1e7638e39acb8dba212f6cf9ec76b9175b0c0da7ba0b1dc78e23da0",
        "kind": "texture",
        "dimensions": (2048, 2048),
    },
}

USD_DEFAULT_PRIM = "/root"
USD_RIGID_BODIES = (
    "/root/Ruler035",
    "/root/Ruler035_Prop001",
    "/root/Ruler035_Prop002",
)
USD_REVOLUTE_JOINTS = {
    "/root/Ruler035_Prop001/Prop001_joint": {
        "body0": "/root/Ruler035",
        "body1": "/root/Ruler035_Prop001",
        "axis": "Z",
        "limits_deg": (-180.0, 0.0),
    },
    "/root/Ruler035_Prop002/Prop002_joint": {
        "body0": "/root/Ruler035_Prop001",
        "body1": "/root/Ruler035_Prop002",
        "axis": "Z",
        "limits_deg": (0.0, 180.0),
    },
}
USD_COLLISION_MESH_COUNT = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG with an IHDR header")
    return struct.unpack(">II", data[16:24])


def audit_assets(repository_root: Path) -> dict[str, object]:
    """Fail closed unless every asset byte matches the delivered bundle."""

    files: dict[str, object] = {}
    for relative_name, expected in ASSETS.items():
        path = repository_root / ASSET_RELATIVE_ROOT / relative_name
        if not path.is_file():
            raise ValueError(f"missing delivered Ruler035 file: {path}")
        data = path.read_bytes()
        actual_size = len(data)
        actual_sha = _sha256(path)
        if actual_size != expected["size"] or actual_sha != expected["sha256"]:
            raise ValueError(f"Ruler035 identity mismatch: {relative_name}")
        row = {
            **expected,
            "relative_path": str((ASSET_RELATIVE_ROOT / relative_name).as_posix()),
        }
        if expected["kind"] == "usd":
            if not data.startswith(b"PXR-USDC\x00"):
                raise ValueError("Ruler035.usd is not the delivered binary USD crate")
            row["crate_magic"] = "PXR-USDC"
            row["topology_validation"] = "required in parsed USD and live PhysX canary"
        else:
            dimensions = _png_dimensions(data)
            if dimensions != expected["dimensions"]:
                raise ValueError(
                    f"Ruler035 texture dimensions mismatch for {relative_name}: {dimensions}"
                )
        files[relative_name] = row
    return {
        "provenance": "self-authored ZRYZ_LW_Assets_20260820/Ruler035 bundle supplied by the benchmark owner",
        "modification": "none; byte-identical copy",
        "files": files,
        "usd_contract": {
            "default_prim": USD_DEFAULT_PRIM,
            "rigid_bodies": list(USD_RIGID_BODIES),
            "revolute_joints": USD_REVOLUTE_JOINTS,
            "collision_mesh_count": USD_COLLISION_MESH_COUNT,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
    }
