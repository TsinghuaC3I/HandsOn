from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


_HERE = Path(__file__).resolve()
_SIM_DIR = _HERE.parents[1]
_ASSET_ROOT = _SIM_DIR / "envs" / "rokae_wuji_ik_replay" / "assets" / "rokae"
_URDF_DIR = _ASSET_ROOT / "models"
_MESH_DIR = _ASSET_ROOT / "meshes"

_ROKAE_URDFS = {
    "right": _URDF_DIR / "AR5-5_08R-W4C1C9-ZY2.urdf",
    "left": _URDF_DIR / "AR5-5_08L-W4C6C11.urdf",
}

_PACKAGE_MESH_ROOTS = {
    "AR5-5_08R-W4C1C9-ZY2_description": _MESH_DIR / "AR5-5_08R-W4C1C9-ZY2" / "visual",
    "AR5-5_08L-W4C6C11_description": _MESH_DIR / "AR5-5_08L-W4C6C11" / "visual",
}


def _repo_root() -> Path:
    for parent in _HERE.parents:
        if (parent / "handson" / "pyproject.toml").exists():
            return parent
    return _HERE.parents[4]


def wuji_usd_path(side: str) -> Path:
    side_key = side.lower()
    if side_key not in {"right", "left"}:
        raise ValueError(f"Unsupported Wuji side {side!r}; expected 'right' or 'left'.")
    return (
        _repo_root()
        / "teleoperation"
        / "manus"
        / "vendor"
        / "wuji-hand-teleop"
        / "src"
        / "wujihandros2"
        / "external"
        / "wuji-description"
        / "hand"
        / "body"
        / "usd"
        / side_key
        / "wujihand.usd"
    )


def rokae_urdf_path(side: str) -> Path:
    side_key = side.lower()
    if side_key not in _ROKAE_URDFS:
        raise ValueError(f"Unsupported ROKAE side {side!r}; expected 'right' or 'left'.")
    return _ROKAE_URDFS[side_key]


def _resolve_mesh_uri(uri: str, source_urdf: Path) -> str:
    if uri.startswith("package://"):
        rest = uri[len("package://") :]
        package, _, rel = rest.partition("/")
        mesh_root = _PACKAGE_MESH_ROOTS.get(package)
        if mesh_root is None:
            return uri
        filename = Path(rel).name
        return str((mesh_root / filename).resolve())

    if uri.startswith("../"):
        return str((source_urdf.parent / uri).resolve())

    return uri


def prepare_rokae_urdf_for_isaac(side: str, output_dir: str | Path) -> Path:
    """Write an Isaac-importer friendly copy of a ROKAE URDF.

    The source URDFs use ROS package:// mesh URLs. The standalone Isaac importer
    is much more predictable with absolute filesystem paths, so the copy rewrites
    only mesh filenames and leaves the rest of the robot description untouched.
    """

    source = rokae_urdf_path(side)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / source.name

    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            mesh.attrib["filename"] = _resolve_mesh_uri(filename, source)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
