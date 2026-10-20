"""Native Isaac Lab scene for the ROKAE + Wuji bimanual assembly task.

The configuration follows DexVerse's native Isaac Lab scene pattern while
reusing HandsOn's existing MJCF and mesh assets.  Robots are real PhysX
articulations and the assembly parts are real rigid bodies; no stage-Xform
teleporting is used during simulation.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from ..envs.hand_specs import native_hand_joint_names, normalize_hand_type
from .compat import (
    isaacsim_cache_namespace,
    make_mjcf_file_cfg,
    mjcf_robot_articulation_root_prim_path,
    mjcf_robot_root_link_path,
    quat_wxyz_to_isaaclab,
)
from .recording_cameras import (
    HEAD_RECORDING_CAMERA_FOVY_DEG,
    RECORDING_CAMERA_VERTICAL_APERTURE,
    WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE,
    WRIST_RECORDING_CAMERA_NEAR_M,
    WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
    focal_length_for_horizontal_fov,
    focal_length_for_vertical_fov,
)


_XML_DIR = Path(
    os.environ.get("HANDSON_ISAACLAB_XML_DIR")
    or Path(__file__).resolve().parents[1] / "envs" / "xmls"
).expanduser()
_CACHE_ROOT = Path(
    os.environ.get(
        "HANDSON_ISAACLAB_CACHE",
        Path.home() / ".cache" / "handson" / "isaaclab",
    )
)
_CACHE_DIR = _CACHE_ROOT / isaacsim_cache_namespace()

ROBOT_MJCF_BY_HAND = {
    "wuji1": {
        "right": _XML_DIR / "rokae_wuji1_right.xml",
        "left": _XML_DIR / "rokae_wuji1_left.xml",
    },
    "wuji2": {
        "right": _XML_DIR / "rokae_wuji2_right.xml",
        "left": _XML_DIR / "rokae_wuji2_left.xml",
    },
}
RIGHT_ROBOT_MJCF = ROBOT_MJCF_BY_HAND["wuji1"]["right"]
LEFT_ROBOT_MJCF = ROBOT_MJCF_BY_HAND["wuji1"]["left"]
PEG_MJCF = _XML_DIR / "industreal_round_peg_8mm.xml"
SOCKET_MJCF = _XML_DIR / "industreal_tray_insert_round_peg_8mm.xml"

# Bump this only when the generated robot-MJCF contract changes.  The source
# and primitive MJCF bytes are hashed separately below so an edited asset gets
# a new Isaac converter cache even when XML parsing normalizes the edit away.
_ROBOT_MJCF_GENERATOR_SCHEMA = (
    b"handson-rokae-wuji-isaac-mjcf-v3-mask-1-1-content-assets"
)

_TEXTURE_DIR = _XML_DIR / "table_arena" / "textures"
_FLOOR_TEXTURE = _TEXTURE_DIR / "light-gray-floor-tile.png"
_WALL_TEXTURE = _TEXTURE_DIR / "light-gray-plaster.png"
_LEG_TEXTURE = _TEXTURE_DIR / "steel-brushed.png"
_TABLE_TEXTURE_NAME = os.environ.get("HANDSON_ISAACLAB_TABLE_TEXTURE", "ceramic.png").strip()
_TABLE_TEXTURE = None
if _TABLE_TEXTURE_NAME:
    _table_texture_candidate = Path(_TABLE_TEXTURE_NAME).expanduser()
    _TABLE_TEXTURE = _table_texture_candidate if _table_texture_candidate.is_absolute() else _TEXTURE_DIR / _TABLE_TEXTURE_NAME
_EXTERNAL_TEXTURE_DIRS = tuple(
    Path(path).expanduser()
    for path in (
        os.environ.get(
            "HANDSON_ISAACLAB_SHARETEXTURES_DIR",
            str(Path.home() / "handson_assets" / "sharetextures" / "table"),
        ),
        os.environ.get("HANDSON_ISAACLAB_TABLE_TEXTURE_DIR", ""),
    )
    if path.strip()
)
_TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}
_COLOR_MAP_HINTS = ("basecolor", "base_color", "albedo", "diffuse", "color", "colour", "col")
_NON_COLOR_MAP_HINTS = (
    "normal",
    "rough",
    "metal",
    "height",
    "disp",
    "displace",
    "ao",
    "ambient",
    "opacity",
    "spec",
    "gloss",
    "bump",
    "mask",
)
_RANDOM_TABLE_TEXTURES = tuple(
    _TEXTURE_DIR / name
    for name in (
        "blue-wood.png",
        "brass-ambra.png",
        "ceramic.png",
        "clay.png",
        "dark-wood.png",
        "gray-felt.png",
        "gray-woodgrain.png",
        "green-wood.png",
        "light-wood.png",
        "red-wood.png",
        "steel-scratched.png",
        "white-plaster.png",
        "wood-tiles.png",
        "wood-varnished-panels.png",
    )
)
_BACKGROUND_WALL_COLORS = (
    (0.78, 0.80, 0.78),
    (0.72, 0.76, 0.82),
    (0.74, 0.70, 0.65),
    (0.66, 0.73, 0.70),
    (0.80, 0.75, 0.77),
)
_BACKGROUND_FLOOR_COLORS = (
    (0.52, 0.54, 0.55),
    (0.42, 0.45, 0.48),
    (0.58, 0.53, 0.46),
    (0.49, 0.55, 0.51),
    (0.62, 0.61, 0.57),
)

_CENTER_BODY_MESH = _XML_DIR / "rokae_arm" / "meshes" / "center" / "body.stl"
_CENTER_HEAD_MESH = _XML_DIR / "rokae_arm" / "meshes" / "center" / "head_fixed.stl"

_RIGHT_ARM_HOME = (
    -1.5707963,
    1.5707963,
    1.5707963,
    1.5707963,
    -1.5707963,
    0.0,
    0.0,
)
_LEFT_ARM_MIRROR_SIGNS = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
_ARM_HOME = {
    "right": _RIGHT_ARM_HOME,
    "left": tuple(
        sign * position
        for sign, position in zip(_LEFT_ARM_MIRROR_SIGNS, _RIGHT_ARM_HOME)
    ),
}

_BASE_POSE = {
    "right": ((-0.75, -0.065, 1.34), (0.7071067812, 0.7071067812, 0.0, 0.0)),
    "left": ((-0.75, 0.065, 1.34), (0.7071067812, -0.7071067812, 0.0, 0.0)),
}


def _is_external_color_texture(path: Path) -> bool:
    lower_name = path.name.lower()
    if path.suffix.lower() not in _TEXTURE_EXTENSIONS:
        return False
    if any(hint in lower_name for hint in _NON_COLOR_MAP_HINTS):
        return False
    return any(hint in lower_name for hint in _COLOR_MAP_HINTS)


def _external_table_textures() -> tuple[Path, ...]:
    textures: list[Path] = []
    seen: set[Path] = set()
    for directory in _EXTERNAL_TEXTURE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or not _is_external_color_texture(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            textures.append(resolved)
    return tuple(textures)


def _available_table_textures() -> tuple[Path, ...]:
    external = _external_table_textures()
    return external or _RANDOM_TABLE_TEXTURES


def _texture_companion(texture: Path | None, map_name: str) -> Path | None:
    if texture is None:
        return None
    replacements = {
        "normal": (
            ("_BaseColor_2K", "_NormalGL_2K"),
            ("_2K-JPG_Color", "_2K-JPG_NormalGL"),
        ),
        "roughness": (
            ("_BaseColor_2K", "_Roughness_2K"),
            ("_2K-JPG_Color", "_2K-JPG_Roughness"),
        ),
    }
    for old, new in replacements[map_name]:
        if old not in texture.stem:
            continue
        candidate = texture.with_name(texture.name.replace(old, new))
        if candidate.is_file():
            return candidate
    return None


def _configure_pbr_texture_maps(material, texture: Path | None) -> tuple[bool, bool]:
    if texture is None:
        return False, False
    from pxr import Sdf

    shader = material.shaders_list[0]
    normal = _texture_companion(texture, "normal")
    roughness = _texture_companion(texture, "roughness")
    if normal is not None:
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(str(normal.resolve()))
        shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.35)
    if roughness is not None:
        shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(str(roughness.resolve()))
        shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(0.85)
    return normal is not None, roughness is not None


def _prepare_center_mjcf() -> Path:
    """Extract the fixed center body and camera head from the MuJoCo arena."""
    output_dir = _CACHE_DIR / "mjcf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rokae_center_isaac.xml"

    root = ET.Element("mujoco", {"model": "rokae_center_isaac"})
    compiler = ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    compiler.tail = "\n  "
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "rokae_center_body",
            "file": os.path.relpath(_CENTER_BODY_MESH, output_dir),
            "scale": "0.001 0.001 0.001",
        },
    )
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "rokae_center_head",
            "file": os.path.relpath(_CENTER_HEAD_MESH, output_dir),
            "scale": "0.001 0.001 0.001",
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    center = ET.SubElement(worldbody, "body", {"name": "rokae_center"})
    ET.SubElement(
        center,
        "inertial",
        {"pos": "-0.815 0.065 0.30", "mass": "8", "diaginertia": "0.25 0.25 0.25"},
    )
    ET.SubElement(
        center,
        "geom",
        {
            "name": "center_body_visual",
            "type": "mesh",
            "mesh": "rokae_center_body",
            "pos": "-0.815 0.065 0.30",
            "rgba": "0.42 0.46 0.50 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    head = ET.SubElement(
        center,
        "body",
        {"name": "rokae_center_head", "pos": "-0.702 0 1.375", "quat": "0.5 0.5 -0.5 -0.5"},
    )
    ET.SubElement(
        head,
        "inertial",
        {"pos": "0 0 0", "mass": "1", "diaginertia": "0.02 0.02 0.02"},
    )
    ET.SubElement(
        head,
        "geom",
        {
            "name": "center_head_visual",
            "type": "mesh",
            "mesh": "rokae_center_head",
            "rgba": "0.62 0.68 0.74 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        head,
        "camera",
        {
            "name": "main",
            "mode": "fixed",
            "pos": "0.001819 -0.169567 -0.147706",
            "quat": "0.592907 -0.370957 0.399206 0.592252",
            "fovy": "70.875",
        },
    )

    # This function also runs during module-level scene configuration. Keep
    # it independent of task_preview, which imports this scene module.
    mesh_dir = output_dir / "center_meshes"
    mesh_dir.mkdir(exist_ok=True)
    for mesh in root.iter("mesh"):
        source = (output_dir / mesh.attrib["file"]).resolve()
        target = mesh_dir / source.name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copy2(source, target)
        mesh.attrib["file"] = os.path.relpath(target, output_dir)

    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not output_path.exists() or output_path.read_bytes() != content:
        output_path.write_bytes(content)
    return output_path


def _robot_mjcf_variant_digest(
    source: Path,
    primitive_source: Path,
    *,
    preserve_base_pose: bool,
    ensure_proximal_finger_collisions: bool,
    hand_contact_margin: float | None = None,
    hand_collision_inflation_m: float | None = None,
    hand_collision_source_meshes: bool = False,
) -> str:
    """Return a cache identity bound to every robot-MJCF input byte."""

    digest = hashlib.sha256()
    for label, payload in (
        (b"generator_schema", _ROBOT_MJCF_GENERATOR_SCHEMA),
        (b"source_mjcf", source.read_bytes()),
        (b"primitive_mjcf", primitive_source.read_bytes()),
        (b"preserve_base_pose", b"1" if preserve_base_pose else b"0"),
        (
            b"ensure_proximal_finger_collisions",
            b"1" if ensure_proximal_finger_collisions else b"0",
        ),
        (
            b"hand_contact_margin",
            (
                b"none"
                if hand_contact_margin is None
                else format(float(hand_contact_margin), ".9g").encode("ascii")
            ),
        ),
        # A task may request a collision-only envelope around the primitive
        # hand geoms.  Include it in the identity so Isaac cannot reuse a USD
        # cooked from the uninflated hand derivative.
        (
            b"hand_collision_inflation_m",
            (
                b"none"
                if hand_collision_inflation_m is None
                else format(
                    float(hand_collision_inflation_m), ".9g"
                ).encode("ascii")
            ),
        ),
    ):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    # Preserve every existing task's cache identity unless this option is used.
    if hand_collision_source_meshes:
        digest.update(b"hand_collision_source_meshes_v1")
    source_root = ET.parse(source).getroot()
    compiler = source_root.find("compiler")
    asset_dir = compiler.attrib.get("assetdir", "") if compiler is not None else ""
    mesh_dir = (
        compiler.attrib.get("meshdir", asset_dir)
        if compiler is not None
        else asset_dir
    )
    texture_dir = (
        compiler.attrib.get("texturedir", asset_dir)
        if compiler is not None
        else asset_dir
    )
    referenced_rows: list[tuple[str, bytes]] = []
    for element in source_root.iter():
        raw_reference = element.attrib.get("file")
        if not raw_reference or element.tag == "include":
            continue
        if element.tag in {"mesh", "skin"}:
            relative_dir = mesh_dir
        elif element.tag in {"texture", "hfield"}:
            relative_dir = texture_dir
        else:
            relative_dir = asset_dir
        candidate = Path(raw_reference)
        asset_path = (
            candidate
            if candidate.is_absolute()
            else source.parent / relative_dir / candidate
        ).resolve()
        referenced_rows.append(
            (f"{element.tag}:{raw_reference}", asset_path.read_bytes())
        )
    for reference, payload in sorted(referenced_rows, key=lambda row: row[0]):
        label = f"external_asset:{reference}".encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()[:16]


def _robot_usd_cache_dir(robot_mjcf: Path) -> Path:
    """Keep converted USDs isolated by the generated MJCF identity."""

    return _CACHE_DIR / "usd" / robot_mjcf.stem


def _framed_file_tree_identity(
    rows: list[tuple[str, Path]],
    *,
    format_name: str,
) -> dict[str, object]:
    """Hash labeled file bytes without depending on host-absolute paths."""

    digest = hashlib.sha256()
    files: list[dict[str, str]] = []
    for label, path in sorted(rows, key=lambda row: row[0]):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"B03 asset identity cannot read {label!r}: {path}: {exc}"
            ) from exc
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        files.append(
            {
                "path": label,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not files:
        raise RuntimeError(f"B03 asset identity has no files for {format_name}")
    return {
        "format": format_name,
        "version": 1,
        "file_count": len(files),
        "files": files,
        "combined_sha256": digest.hexdigest(),
    }


def _external_asset_dependency_paths(source: Path) -> list[tuple[str, Path]]:
    """Return an external asset plus OBJ/MTL dependencies with stable labels."""

    source = source.resolve()
    rows: dict[str, Path] = {f"asset{source.suffix.lower()}": source}
    if source.suffix.lower() != ".obj":
        return sorted(rows.items())

    material_paths: list[tuple[Path, Path]] = []
    with source.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped.startswith("mtllib "):
                continue
            relative = Path(stripped.split(maxsplit=1)[1].strip())
            material = (source.parent / relative).resolve()
            if not material.is_file():
                raise RuntimeError(
                    f"B03 OBJ dependency is missing: {source}: {material}"
                )
            rows[relative.as_posix()] = material
            material_paths.append((relative, material))

    for material_relative, material in material_paths:
        with material.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                token = stripped.split(maxsplit=1)[0].lower()
                if not token.startswith("map_"):
                    continue
                relative = material_relative.parent / stripped.split()[-1]
                texture = (source.parent / relative).resolve()
                if not texture.is_file():
                    raise RuntimeError(
                        f"B03 MTL dependency is missing: {material}: {texture}"
                    )
                rows[relative.as_posix()] = texture
    return sorted(rows.items())


def _mjcf_referenced_asset_identity(mjcf_path: Path) -> dict[str, object]:
    """Hash every external file referenced by a generated robot MJCF.

    The generated B03 MJCF rewrites mesh references relative to its cache
    directory.  Preserve the authored reference as the digest label so hosts
    may use different absolute checkout roots while still comparing the exact
    referenced bytes.
    """

    pending: list[tuple[str, Path]] = [("root", mjcf_path)]
    visited_xml: set[Path] = set()
    assets: dict[str, Path] = {}
    while pending:
        document_label, document_path = pending.pop()
        resolved_document = document_path.resolve()
        if resolved_document in visited_xml:
            continue
        visited_xml.add(resolved_document)
        try:
            root = ET.parse(resolved_document).getroot()
        except (OSError, ET.ParseError) as exc:
            raise RuntimeError(
                f"B03 asset identity cannot parse MJCF {resolved_document}: {exc}"
            ) from exc
        compiler = root.find("compiler")
        asset_dir = ""
        mesh_dir = ""
        texture_dir = ""
        if compiler is not None:
            asset_dir = compiler.attrib.get("assetdir", "")
            mesh_dir = compiler.attrib.get("meshdir", asset_dir)
            texture_dir = compiler.attrib.get("texturedir", asset_dir)
        reference_index = 0
        for element in root.iter():
            raw_reference = element.attrib.get("file")
            if not raw_reference:
                continue
            if element.tag == "include":
                base_dir = resolved_document.parent
            elif element.tag in {"mesh", "skin"}:
                base_dir = resolved_document.parent / mesh_dir
            elif element.tag in {"texture", "hfield"}:
                base_dir = resolved_document.parent / texture_dir
            else:
                base_dir = resolved_document.parent / asset_dir
            candidate = Path(raw_reference)
            resolved_asset = (
                candidate.resolve()
                if candidate.is_absolute()
                else (base_dir / candidate).resolve()
            )
            if not resolved_asset.is_file():
                raise RuntimeError(
                    "B03 generated MJCF references a missing asset: "
                    f"{document_label}:{element.tag}:{raw_reference}: "
                    f"{resolved_asset}"
                )
            payload_sha256 = hashlib.sha256(resolved_asset.read_bytes()).hexdigest()
            element_name = element.attrib.get("name", "")
            reference_label = (
                f"{document_label}:{reference_index}:{element.tag}:"
                f"{element_name}:{payload_sha256}"
            )
            reference_index += 1
            for dependency_label, dependency in _external_asset_dependency_paths(
                resolved_asset
            ):
                dependency_sha256 = hashlib.sha256(
                    dependency.read_bytes()
                ).hexdigest()
                assets[
                    f"{reference_label}:{dependency_label}:{dependency_sha256}"
                ] = dependency
            if element.tag == "include":
                pending.append((f"include:{payload_sha256}", resolved_asset))
    return _framed_file_tree_identity(
        list(assets.items()),
        format_name="handson.b03_mjcf_referenced_assets",
    )


def _loaded_usd_cache_identity(usd_dir: Path) -> dict[str, object]:
    """Hash every file below the exact loaded USD asset directory."""

    root = usd_dir.resolve()
    if not root.is_dir():
        raise RuntimeError(f"B03 generated USD cache is missing: {root}")
    rows = [
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file()
    ]
    return _framed_file_tree_identity(
        rows,
        format_name="handson.b03_loaded_usd_cache",
    )


def _strict_loaded_mjcf_usd_asset_dir(mjcf_path: Path, usd_dir: Path) -> Path:
    """Resolve the sole allowed Lab 3 MJCF asset tree and reject ``_<N>``."""

    expected = usd_dir / mjcf_path.stem
    if not expected.is_dir():
        raise RuntimeError(
            "B03 expected MJCF USD asset directory is missing: "
            f"{expected}"
        )
    prefix = f"{mjcf_path.stem}_"
    ambiguous = sorted(
        path.name
        for path in usd_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and path.name[len(prefix) :].isdigit()
    )
    if ambiguous:
        raise RuntimeError(
            "B03 MJCF USD cache contains ambiguous importer variants; "
            "discard and rebuild the content-addressed cache before "
            f"certification: {usd_dir}: {ambiguous}"
        )
    return expected


def b03_generated_robot_asset_identity(
    robot_mjcf: Path,
    usd_dir: Path,
) -> dict[str, object]:
    """Return the complete loaded robot-asset identity for certification."""

    payload = robot_mjcf.read_bytes()
    loaded_asset_dir = _strict_loaded_mjcf_usd_asset_dir(robot_mjcf, usd_dir)
    return {
        "format": "handson.b03_generated_robot_asset",
        "version": 3,
        "generated_mjcf_name": robot_mjcf.name,
        "generated_mjcf_sha256": hashlib.sha256(payload).hexdigest(),
        "usd_cache_key": usd_dir.name,
        "referenced_assets": _mjcf_referenced_asset_identity(robot_mjcf),
        "loaded_usd_cache": _loaded_usd_cache_identity(loaded_asset_dir),
    }


def _canonical_mjcf_sha256(mjcf_path: Path) -> str:
    """Hash MJCF semantics while replacing host-specific asset paths."""

    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    asset_dir = compiler.attrib.get("assetdir", "") if compiler is not None else ""
    mesh_dir = (
        compiler.attrib.get("meshdir", asset_dir)
        if compiler is not None
        else asset_dir
    )
    texture_dir = (
        compiler.attrib.get("texturedir", asset_dir)
        if compiler is not None
        else asset_dir
    )
    for element in root.iter():
        raw_reference = element.attrib.get("file")
        if not raw_reference:
            continue
        if element.tag == "include":
            relative_dir = ""
        elif element.tag in {"mesh", "skin"}:
            relative_dir = mesh_dir
        elif element.tag in {"texture", "hfield"}:
            relative_dir = texture_dir
        else:
            relative_dir = asset_dir
        candidate = Path(raw_reference)
        asset_path = (
            candidate
            if candidate.is_absolute()
            else mjcf_path.parent / relative_dir / candidate
        ).resolve()
        payload_sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        element.attrib["file"] = f"sha256:{payload_sha256}"
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def b03_loaded_task_asset_identity(
    task_mjcf: Path,
    usd_dir: Path,
) -> dict[str, object]:
    """Bind B03's peg/socket scene and the exact USD layers it loaded."""

    loaded_asset_dir = _strict_loaded_mjcf_usd_asset_dir(task_mjcf, usd_dir)
    payload = task_mjcf.read_bytes()
    return {
        "format": "handson.b03_loaded_task_asset",
        "version": 2,
        "generated_mjcf_name": task_mjcf.name,
        "generated_mjcf_sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_generated_mjcf_sha256": _canonical_mjcf_sha256(task_mjcf),
        "usd_cache_key": usd_dir.name,
        "referenced_assets": _mjcf_referenced_asset_identity(task_mjcf),
        "loaded_usd_cache": _loaded_usd_cache_identity(loaded_asset_dir),
    }


def _prepare_robot_mjcf(
    side: str,
    *,
    preserve_base_pose: bool = False,
    ensure_proximal_finger_collisions: bool = False,
    hand_contact_margin: float | None = None,
    hand_collision_inflation_m: float | None = None,
    hand_collision_source_meshes: bool = False,
    hand_type: str = "wuji1",
) -> Path:
    """Create an Isaac-oriented MJCF while retaining the detailed visuals.

    The source task uses the same high-resolution mesh for every Wuji visual and
    collision shape.  PhysX convex cooking for both hands then takes minutes.
    HandsOn already ships calibrated primitive Wuji collision MJCFs, so this
    cached derivative keeps the detailed visual geoms and substitutes only the
    hand collision geoms.  It also moves the authored base pose into
    ``ArticulationCfg.init_state`` so Isaac's fixed root joint is not disjoint.
    """
    normalized_hand_type = str(hand_type or "wuji1").strip().lower()
    if normalized_hand_type not in {"wuji1", "wuji2"}:
        raise ValueError(
            "Isaac ROKAE robot assets support only wuji1 or wuji2; "
            f"got {hand_type!r}."
        )
    if normalized_hand_type == "wuji1":
        source = RIGHT_ROBOT_MJCF if side == "right" else LEFT_ROBOT_MJCF
    else:
        source = _XML_DIR / f"rokae_wuji2_{side}.xml"
    # Wuji1 replaces expensive detailed collision meshes with its calibrated
    # primitive derivative.  The official Hand 2 asset already carries a
    # complete collision geom on every canonical finger link, so retain it.
    primitive_source = (
        _XML_DIR / "wuji_hand1" / "mjcf" / f"{side}_primitive.xml"
        if normalized_hand_type == "wuji1"
        else source
    )
    output_dir = _CACHE_DIR / "mjcf"
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_digest = _robot_mjcf_variant_digest(
        source,
        primitive_source,
        preserve_base_pose=preserve_base_pose,
        ensure_proximal_finger_collisions=ensure_proximal_finger_collisions,
        hand_contact_margin=hand_contact_margin,
        hand_collision_inflation_m=hand_collision_inflation_m,
        hand_collision_source_meshes=hand_collision_source_meshes,
    )

    tree = ET.parse(source)
    root = tree.getroot()

    # Generated files live in a cache, so resolve all source-relative meshes.
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("file")
        if filename and not Path(filename).is_absolute():
            source_mesh = (source.parent / filename).resolve()
            payload = source_mesh.read_bytes()
            content_sha256 = hashlib.sha256(payload).hexdigest()
            stable_dir = output_dir / "assets" / content_sha256
            stable_dir.mkdir(parents=True, exist_ok=True)
            stable_path = stable_dir / f"{side}_{source_mesh.name}"
            if not stable_path.exists() or stable_path.read_bytes() != payload:
                shutil.copy2(source_mesh, stable_path)
            # The generated MJCF must be byte-identical when the checkout and
            # cache live at different host paths.  Content-addressed cache
            # references also prevent stale USD reuse after a mesh edit.
            mesh.attrib["file"] = stable_path.relative_to(output_dir).as_posix()

    worldbody = root.find("worldbody")
    if worldbody is None or worldbody.find("body") is None:
        raise ValueError(f"Robot MJCF has no root body: {source}")
    root_body = worldbody.find("body")
    if not preserve_base_pose:
        root_body.attrib["pos"] = "0 0 0"
        root_body.attrib["quat"] = "1 0 0 0"

    # Isaac keeps fixed MJCF bodies as rigid links.  Give otherwise-empty mount
    # frames a negligible valid inertia to avoid invalid-mass PhysX bodies.
    for body in root.iter("body"):
        if body.find("inertial") is None and body.find("geom") is None:
            body.insert(
                0,
                ET.Element(
                    "inertial",
                    {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-12 1e-12 1e-12"},
                ),
            )

    if hand_collision_source_meshes:
        # GEN2: use existing detailed source collision meshes at their authored
        # link transforms. The legacy primitive shells protrude beyond those
        # visuals by centimetres. No visible asset or robot joint is changed.
        hand_root = next(body for body in root.iter("body")
                         if body.attrib.get("name") == f"{side}_palm_link")
        for body in hand_root.iter("body"):
            for index, geom in enumerate(body.findall("geom")):
                if geom.attrib.get("contype", "1") == "0":
                    continue
                geom.attrib.update(contype="1", conaffinity="1", group="3")
                geom.attrib.setdefault("name", f"{body.attrib['name']}_source_col_{index}")
    elif normalized_hand_type == "wuji1":
        primitive_root = ET.parse(primitive_source).getroot()
        primitive_bodies = {
            body.attrib["name"]: body
            for body in primitive_root.iter("body")
            if "name" in body.attrib
        }
        hand_root = next(
            (
                body
                for body in root.iter("body")
                if body.attrib.get("name") == f"{side}_palm_link"
            ),
            None,
        )
        if hand_root is None:
            raise ValueError(f"Could not locate {side} Wuji palm in {source}")

        proximal_finger_links = {
            f"{side}_finger{finger}_link1"
            for finger in range(1, 6)
        }
        for body in hand_root.iter("body"):
            name = body.attrib.get("name")
            primitive_body = primitive_bodies.get(name)
            if primitive_body is None:
                continue
            for geom in list(body.findall("geom")):
                if geom.attrib.get("contype", "1") != "0":
                    body.remove(geom)
            for geom in primitive_body.findall("geom"):
                if geom.attrib.get("contype", "1") != "0":
                    collision = deepcopy(geom)
                    if ensure_proximal_finger_collisions:
                        # The primitive hand mixes collision masks 1/1 and 2/2.
                        # Collection objects use the ordinary 1/1 external-contact
                        # group, so normalize the complete-hand derivative.
                        collision.attrib["contype"] = "1"
                        collision.attrib["conaffinity"] = "1"
                    body.append(collision)

            # The calibrated primitive asset labels both proximal-link capsules as
            # visual-only.  That leaves the base of every finger with no physical
            # geometry, so it can pass through fruit and the table even when the
            # distal links contact correctly.  Newton A01 needs a complete hand:
            # reuse the matching primitive capsule as an ordinary external-contact
            # collider.  Robot self-collision remains disabled at the articulation
            # level, so this does not create adjacent-finger-link constraints.
            if ensure_proximal_finger_collisions and name in proximal_finger_links:
                has_collision = any(
                    geom.attrib.get("contype", "1") != "0"
                    for geom in primitive_body.findall("geom")
                )
                if not has_collision:
                    visual = next(iter(primitive_body.findall("geom")), None)
                    if visual is None:
                        raise ValueError(
                            f"Primitive hand body has no geometry: {name}"
                        )
                    collision = deepcopy(visual)
                    collision.attrib.update(
                        {
                            "name": f"{name}_col2",
                            "contype": "1",
                            "conaffinity": "1",
                            "group": "3",
                        }
                    )
                    collision.attrib.pop("density", None)
                    collision.attrib.pop("mass", None)
                    body.append(collision)

    if hand_collision_inflation_m is not None:
        inflation = float(hand_collision_inflation_m)
        if not math.isfinite(inflation) or inflation <= 0.0:
            raise ValueError(
                "hand_collision_inflation_m must be a finite positive value; "
                f"got {hand_collision_inflation_m!r}"
            )
        hand_root = next(
            (
                body
                for body in root.iter("body")
                if body.attrib.get("name") == f"{side}_palm_link"
            ),
            None,
        )
        if hand_root is None:
            raise ValueError(
                f"Could not locate {side} hand collision root in {source}"
            )

        # These are the calibrated primitive collision geoms, while the
        # detailed source meshes remain visual-only.  Expand only the physical
        # primitives: a small conservative shell catches a finger before the
        # 60 Hz deformable narrowphase can tunnel, without changing the hand
        # mesh recorded by the renderer.
        for body in hand_root.iter("body"):
            for geom in body.findall("geom"):
                if geom.attrib.get("contype", "1") == "0":
                    continue
                if geom.attrib.get("conaffinity", "1") == "0":
                    continue
                geom_type = geom.attrib.get("type", "sphere").strip().lower()
                size_text = geom.attrib.get("size")
                if not size_text or geom_type == "mesh":
                    continue
                try:
                    size = [float(token) for token in size_text.split()]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {side} hand collision size: {size_text!r}"
                    ) from exc
                if not size or not all(math.isfinite(value) for value in size):
                    raise ValueError(
                        f"Invalid {side} hand collision size: {size_text!r}"
                    )
                if geom_type in {"sphere", "ellipsoid"}:
                    size = [value + inflation for value in size]
                elif geom_type in {"capsule", "cylinder"}:
                    size = [value + inflation for value in size]
                elif geom_type == "box":
                    size = [value + inflation for value in size]
                else:
                    # Plane/heightfield and future mesh-like geoms do not have
                    # a stable radial size contract; leave them untouched.
                    continue
                geom.attrib["size"] = " ".join(
                    format(float(value), ".9g") for value in size
                )

    if hand_contact_margin is not None:
        margin = float(hand_contact_margin)
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                "hand_contact_margin must be a finite positive value; "
                f"got {hand_contact_margin!r}"
            )
        hand_root = next(
            (
                body
                for body in root.iter("body")
                if body.attrib.get("name") == f"{side}_palm_link"
            ),
            None,
        )
        if hand_root is None:
            raise ValueError(
                f"Could not locate {side} hand contact root in {source}"
            )
        formatted_margin = format(margin, ".9g")
        for body in hand_root.iter("body"):
            for geom in body.findall("geom"):
                if geom.attrib.get("contype", "1") == "0":
                    continue
                if geom.attrib.get("conaffinity", "1") == "0":
                    continue
                geom.attrib["margin"] = formatted_margin
                geom.attrib["gap"] = "0"

    suffix = (
        "scene_isaac"
        if preserve_base_pose
        else "isaac_newton"
        if ensure_proximal_finger_collisions
        else "isaac"
    )
    output_path = output_dir / (
        f"rokae_{normalized_hand_type}_{side}_{suffix}_{variant_digest}.xml"
    )
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not output_path.exists() or output_path.read_bytes() != content:
        output_path.write_bytes(content)
    return output_path


def _robot_cfg(
    side: str,
    *,
    physics_backend: str = "physx",
    ensure_proximal_finger_collisions: bool = False,
    hand_contact_margin: float | None = None,
    hand_collision_inflation_m: float | None = None,
    hand_collision_source_meshes: bool = False,
    hand_type: str = "wuji1",
) -> ArticulationCfg:
    """Build one fixed-base ROKAE arm with its attached Wuji hand."""
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported robot side: {side!r}")
    normalized_hand_type = normalize_hand_type(hand_type)
    if normalized_hand_type not in ROBOT_MJCF_BY_HAND:
        raise ValueError(
            "Isaac ROKAE robot assets support only wuji1 or wuji2; "
            f"got {hand_type!r}."
        )

    # Newton currently validates/corrects several imported MJCF inertias and
    # applies substantially larger joint loads than PhysX for this robot.  The
    # source-tuned PhysX effort limits saturate immediately, so the arm falls
    # away from an unchanged position target and cannot follow teleoperation.
    # Keep the established PhysX drive unchanged and give Newton a dedicated
    # industrial-arm drive envelope.
    newton = str(physics_backend).strip().lower() == "newton"
    arm_effort = (
        {
            f"joint[1-4]_{side}(_1)?": 80.0,
            f"joint[5-7]_{side}(_1)?": 40.0,
        }
        if newton
        else {
            f"joint[1-2]_{side}(_1)?": 140.0,
            f"joint[3-4]_{side}(_1)?": 90.0,
            f"joint[5-7]_{side}(_1)?": 35.0,
        }
    )
    # Explicit torque PD needs substantially lower gains than a solver-side
    # implicit drive.  Use the Isaac Lab/Newton reference envelope for the
    # load-bearing joints and soften the wrist further; PhysX keeps its
    # established implicit-drive tuning.
    arm_stiffness = (
        {
            f"joint[1-4]_{side}(_1)?": 40.0,
            f"joint[5-7]_{side}(_1)?": 20.0,
        }
        if newton
        else {
            f"joint[1-2]_{side}(_1)?": 380.0,
            f"joint[3-4]_{side}(_1)?": 300.0,
            f"joint[5-7]_{side}(_1)?": 105.0,
        }
    )
    arm_damping = (
        {
            f"joint[1-4]_{side}(_1)?": 8.0,
            f"joint[5-7]_{side}(_1)?": 4.0,
        }
        if newton
        else {
            f"joint[1-2]_{side}(_1)?": 52.0,
            f"joint[3-4]_{side}(_1)?": 44.0,
            f"joint[5-7]_{side}(_1)?": 18.0,
        }
    )
    if newton:
        # MJWarp does not yet expose inverse dynamics through Isaac Lab, but it
        # does support MuJoCo's solver-native gravity compensation.  Route that
        # compensation through the explicit actuator channel so a moderate-gain
        # IdealPD drive can hold the arm without the old high implicit gains.
        from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg

        joint_drive_props = MujocoJointDrivePropertiesCfg(
            drive_type="force",
            actuatorgravcomp=True,
        )
    else:
        joint_drive_props = sim_utils.JointDrivePropertiesCfg(drive_type="force")

    # Sim 6's MJCF converter appends ``_1`` to the seven arm-joint names when
    # it resolves the duplicated MJCF/USD identifiers.  These expressions
    # intentionally match both the Sim 5 and Sim 6 layouts.
    joint_pos = {
        f"joint{index}_{side}(_1)?": value
        for index, value in enumerate(_ARM_HOME[side], start=1)
    }
    hand_joint_names = native_hand_joint_names(normalized_hand_type, side)
    if normalized_hand_type == "wuji1":
        # All non-thumb Wuji1 joints contain zero in their valid range.  The
        # thumb base has a small positive lower limit and starts slightly open.
        joint_pos[f"{side}_finger[2-5]_joint[1-4]"] = 0.0
        joint_pos[f"{side}_finger1_joint[2-4]"] = 0.0
        joint_pos[f"{side}_finger1_joint1"] = 0.10
        hand_joint_exprs = [f"{side}_finger[1-5]_joint[1-4]"]
    else:
        # Zero is inside every official Wuji2 joint range and matches the
        # bridge's neutral native command before the first glove packet.
        joint_pos.update({name: 0.0 for name in hand_joint_names})
        hand_joint_exprs = list(hand_joint_names)

    arm_expr = f"joint[1-7]_{side}(_1)?"
    base_pos, base_rot = _BASE_POSE[side]

    robot_mjcf = _prepare_robot_mjcf(
        side,
        ensure_proximal_finger_collisions=(
            newton or bool(ensure_proximal_finger_collisions)
        ),
        hand_contact_margin=hand_contact_margin,
        hand_collision_inflation_m=hand_collision_inflation_m,
        hand_collision_source_meshes=hand_collision_source_meshes,
        hand_type=normalized_hand_type,
    )
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{side.capitalize()}Robot",
        # Isaac Sim 5.1's MJCF importer also tags the fixed base link with
        # ArticulationRootAPI.  Select the actual importer articulation scope
        # explicitly so Isaac Lab does not see two candidate roots.
        articulation_root_prim_path=mjcf_robot_articulation_root_prim_path(side),
        spawn=make_mjcf_file_cfg(
            sim_utils,
            asset_path=str(robot_mjcf),
            usd_dir=str(_robot_usd_cache_dir(robot_mjcf)),
            # Isaac Sim 5.1's MJCF-generated world joint is anchored at world
            # zero even when the referenced asset is spawned elsewhere.  Spawn
            # floating here and create a correctly aligned fixed joint after
            # the environment Xform is in place (see ``fix_robot_bases``).
            fix_base=False,
            import_sites=True,
            self_collision=False,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=3.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=joint_drive_props,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base_pos,
            rot=quat_wxyz_to_isaaclab(base_rot),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators={
            # MuJoCo's ROKAE path sends finite joint torques into the solver.
            # Match that contract on Newton: IdealPD computes/clamps torque
            # from the DLS joint target, while PhysX keeps its established
            # implicit drive.  Neither path restores joint state after reset.
            "arm": (IdealPDActuatorCfg if newton else ImplicitActuatorCfg)(
                joint_names_expr=[arm_expr],
                **(
                    {"effort_limit": arm_effort}
                    if newton
                    else {"effort_limit_sim": arm_effort}
                ),
                stiffness=arm_stiffness,
                damping=arm_damping,
                friction=0.01,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=hand_joint_exprs,
                effort_limit_sim=25.0 if newton else 6.0,
                stiffness=40.0 if newton else 8.0,
                damping=1.2 if newton else 0.12,
                friction=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def configure_newton_robot_gravity_compensation(
    stage,
    robot_root_paths: tuple[str, ...],
) -> int:
    """Author MJWarp gravity compensation on every imported robot link.

    Isaac Lab's nested schema writer stops at the imported articulation root
    in this direct, non-cloned scene, so its automatic fallback only marks the
    fixed base body.  MuJoCo gravity compensation is body-local; write the
    attribute on each real link before Newton imports the USD stage.
    """

    from pxr import Sdf, Usd, UsdPhysics

    configured = 0
    for root_path in robot_root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            attribute = prim.GetAttribute("mjc:gravcomp")
            if not attribute.IsValid():
                attribute = prim.CreateAttribute(
                    "mjc:gravcomp",
                    Sdf.ValueTypeNames.Float,
                )
            attribute.Set(1.0)
            configured += 1
    return configured


def _assembly_part_cfg(name: str, mjcf_path: Path, pos: tuple[float, float, float], rot=(1.0, 0.0, 0.0, 0.0)):
    """Build a dynamic assembly part from its original HandsOn MJCF."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=make_mjcf_file_cfg(
            sim_utils,
            asset_path=str(mjcf_path),
            usd_dir=str(_CACHE_DIR / "usd" / mjcf_path.stem),
            fix_base=False,
            import_sites=True,
            self_collision=False,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=5.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            # A MuJoCo freejoint is imported with ArticulationRootAPI in Isaac
            # Sim 5.1.  Disable that API so Isaac Lab exposes one RigidObject.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


@configclass
class BimanualAssemblySceneCfg(InteractiveSceneCfg):
    """One or more tiled instances of the bimanual insertion preview."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        # A local static cuboid avoids the Nucleus-backed GroundPlane asset.
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.56, 0.58, 0.61), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.92, 0.94, 1.0)),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=950.0, color=(1.0, 0.96, 0.90), angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9238795, 0.0, 0.3826834, 0.0)),
    )

    # Visual-only room boundaries from arena_arm_hand_bimanual_assembly_rokae_wuji1.xml.
    wall_left_corner = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallLeftCorner",
        spawn=sim_utils.CuboidCfg(
            size=(2.12, 3.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.70, 0.73), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.25, 2.25, 1.5), rot=(0.6532815, 0.6532815, 0.2705981, 0.2705981)
        ),
    )
    wall_right_corner = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallRightCorner",
        spawn=sim_utils.CuboidCfg(
            size=(2.12, 3.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.70, 0.73), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.25, -2.25, 1.5), rot=(0.6532815, 0.6532815, -0.2705981, -0.2705981)
        ),
    )
    wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(3.5, 3.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.70, 0.73), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.25, 3.0, 1.5), rot=(0.7071068, 0.7071068, 0.0, 0.0)
        ),
    )
    wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallRight",
        spawn=sim_utils.CuboidCfg(
            size=(3.5, 3.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.70, 0.73), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.25, -3.0, 1.5), rot=(0.7071068, -0.7071068, 0.0, 0.0)
        ),
    )
    wall_rear = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallRear",
        spawn=sim_utils.CuboidCfg(
            size=(3.0, 3.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.70, 0.73), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, 0.0, 1.5), rot=(0.5, 0.5, 0.5, 0.5)),
    )
    wall_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallFront",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 3.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.70, 0.73), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 0.0, 1.5), rot=(0.5, 0.5, -0.5, -0.5)),
    )

    # The top is z=0.92, matching the original assembly-part reset height.
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.6, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.72, 0.68), roughness=0.65, metallic=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.89)),
    )

    table_leg_front_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontLeft",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.86,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.37, 0.40), metallic=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.17, 0.72, 0.43)),
    )
    table_leg_front_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontRight",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.86,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.37, 0.40), metallic=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.17, -0.72, 0.43)),
    )
    table_leg_rear_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegRearLeft",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.86,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.37, 0.40), metallic=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.47, 0.72, 0.43)),
    )
    table_leg_rear_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegRearRight",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.86,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.37, 0.40), metallic=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.47, -0.72, 0.43)),
    )

    center_rig = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CenterRig",
        spawn=make_mjcf_file_cfg(
            sim_utils,
            asset_path=str(_prepare_center_mjcf()),
            usd_dir=str(_CACHE_DIR / "usd" / "rokae_center"),
            fix_base=True,
            import_sites=True,
            make_instanceable=False,
        ),
    )

    right_robot: ArticulationCfg = _robot_cfg("right")
    left_robot: ArticulationCfg = _robot_cfg("left")

    peg: RigidObjectCfg = _assembly_part_cfg(
        "Peg",
        PEG_MJCF,
        pos=(-0.10, -0.16, 0.94),
        rot=(0.6964, -0.1228, 0.6964, 0.1228),
    )
    socket: RigidObjectCfg = _assembly_part_cfg(
        "Socket",
        SOCKET_MJCF,
        pos=(-0.10, 0.16, 0.92),
    )

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/HeadRecordCamera",
        update_period=0.0,
        height=640,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_for_vertical_fov(
                HEAD_RECORDING_CAMERA_FOVY_DEG
            ),
            focus_distance=1.2,
            horizontal_aperture=RECORDING_CAMERA_VERTICAL_APERTURE,
            vertical_aperture=RECORDING_CAMERA_VERTICAL_APERTURE,
            clipping_range=(0.0008, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    wrist_left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/LeftWristRecordCamera",
        update_period=0.0,
        height=WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1],
        width=WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_for_horizontal_fov(
                WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE["left"],
                RECORDING_CAMERA_VERTICAL_APERTURE
                * WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0]
                / WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1],
            ),
            focus_distance=0.55,
            horizontal_aperture=(
                RECORDING_CAMERA_VERTICAL_APERTURE
                * WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0]
                / WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1]
            ),
            vertical_aperture=RECORDING_CAMERA_VERTICAL_APERTURE,
            clipping_range=(WRIST_RECORDING_CAMERA_NEAR_M, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    wrist_right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RightWristRecordCamera",
        update_period=0.0,
        height=WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1],
        width=WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_for_horizontal_fov(
                WRIST_RECORDING_CAMERA_HFOV_DEG_BY_SIDE["right"],
                RECORDING_CAMERA_VERTICAL_APERTURE
                * WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0]
                / WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1],
            ),
            focus_distance=0.55,
            horizontal_aperture=(
                RECORDING_CAMERA_VERTICAL_APERTURE
                * WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0]
                / WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1]
            ),
            vertical_aperture=RECORDING_CAMERA_VERTICAL_APERTURE,
            clipping_range=(WRIST_RECORDING_CAMERA_NEAR_M, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def make_bimanual_assembly_scene_cfg(
    *,
    num_envs: int,
    env_spacing: float,
    physics_backend: str = "physx",
    hand_type: str = "wuji1",
) -> BimanualAssemblySceneCfg:
    """Build the scene with robot drives for the selected physics backend.

    The config-class robot fields are evaluated when this module is imported,
    before an environment chooses PhysX or Newton.  Replacing the two robot
    configs here prevents Newton from silently inheriting the much weaker
    PhysX-only actuator limits.
    """

    cfg = BimanualAssemblySceneCfg(
        num_envs=int(num_envs),
        env_spacing=float(env_spacing),
    )
    cfg.right_robot = _robot_cfg(
        "right",
        physics_backend=physics_backend,
        hand_type=hand_type,
    )
    cfg.left_robot = _robot_cfg(
        "left",
        physics_backend=physics_backend,
        hand_type=hand_type,
    )
    return cfg


_ROBOT_COLLISION_VISIBLE_PATH_TOKENS = (
    "/rightrobot/",
    "/leftrobot/",
    "/base_right/",
    "/base_left/",
    "/centerrig/",
    "/rokae_center",
    "/right_robot/",
    "/left_robot/",
    "/right_hand/",
    "/left_hand/",
    "rokae",
    "wuji",
)


def _is_robot_collision_render_path(prim_path: str) -> bool:
    normalized = prim_path.lower()
    return any(token in normalized for token in _ROBOT_COLLISION_VISIBLE_PATH_TOKENS)


def spawn_center_rig(prim_path: str = "/World/HandsOnRobots/CenterRig") -> str:
    spawn_cfg = make_mjcf_file_cfg(
        sim_utils,
        asset_path=str(_prepare_center_mjcf()),
        usd_dir=str(_CACHE_DIR / "usd" / "rokae_center"),
        fix_base=True,
        import_sites=True,
        make_instanceable=False,
    )
    spawn_cfg.func(prim_path, spawn_cfg)
    return prim_path


def _repair_imported_visual_materials(stage, root_path: str) -> tuple[int, int]:
    """Restore MJCF materials hidden by nested OBJ/STL DefaultMaterial bindings.

    Isaac Sim 5.1 authors the MJCF ``rgba`` material on a visual Xform but also
    binds a white ``DefaultMaterial`` (or an OBJ material) directly on its child
    Mesh.  USD gives the child binding precedence.  Rebinding the authored
    parent material strongly on the render mesh restores MuJoCo's intended arm,
    hand, peg, and socket colors without changing collision geometry.
    """
    from pxr import Usd, UsdGeom, UsdShade

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return 0, 0

    def intended_material_for(prim):
        ancestor = prim.GetParent()
        while ancestor.IsValid() and str(ancestor.GetPath()).startswith(root_path):
            binding = UsdShade.MaterialBindingAPI(ancestor).GetDirectBinding()
            material = binding.GetMaterial()
            if (
                material
                and material.GetPrim().IsValid()
                and material.GetPrim().GetName() != "DefaultMaterial"
            ):
                return material
            ancestor = ancestor.GetParent()
        return None

    def editable_target_for(prim):
        while prim.IsInstanceProxy():
            prim = prim.GetParent()
        return prim

    # Imported MJCF visual groups are often instance roots.  A single group can
    # contain several source materials, so binding the first material on that
    # shared root makes every descendant inherit the same color.  Detect only
    # those heterogeneous visual groups and expand them before rebinding; robot
    # or asset roots outside a ``visuals`` subtree remain instanceable.
    materials_by_target: dict[str, set[str]] = {}
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())
        if "/visuals/" not in prim_path or not prim.IsA(UsdGeom.Gprim):
            continue
        intended_material = intended_material_for(prim)
        if intended_material is None:
            continue
        target_path = str(editable_target_for(prim).GetPath())
        materials_by_target.setdefault(target_path, set()).add(
            str(intended_material.GetPath())
        )

    heterogeneous_target_paths = {
        target_path
        for target_path, material_paths in materials_by_target.items()
        if len(material_paths) > 1
    }
    for target_path in heterogeneous_target_paths:
        target_prim = stage.GetPrimAtPath(target_path)
        # Expanding a scope named exactly ``visuals`` is narrow and safe even
        # when it is nested below a task object (for example D02's microwave).
        # If the nearest editable ancestor is broader than that, leave the
        # instance intact and let the second pass skip the unsafe broad bind.
        if (
            target_prim.IsValid()
            and target_prim.GetName() == "visuals"
            and target_prim.IsInstance()
        ):
            target_prim.SetInstanceable(False)

    # Re-fetch after changing instanceability so the traversal sees editable
    # descendants rather than stale instance proxies.
    root_prim = stage.GetPrimAtPath(root_path)

    rebound = 0
    hidden_colliders = 0
    rebound_paths: set[str] = set()
    hidden_paths: set[str] = set()
    debug_materials = os.environ.get("HANDSON_ISAACLAB_DEBUG_MATERIALS") == "1"
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())

        if debug_materials and prim.IsA(UsdGeom.Gprim):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            material_path = str(material.GetPath()) if material and material.GetPrim().IsValid() else "<none>"
            print(f"[material-debug] {prim_path} -> {material_path}")

        if (
            "/collisions/" in prim_path
            and prim.IsA(UsdGeom.Gprim)
            and not _is_robot_collision_render_path(prim_path)
        ):
            collision_prim = prim
            while collision_prim.IsInstanceProxy():
                collision_prim = collision_prim.GetParent()
            collision_path = str(collision_prim.GetPath())
            if collision_path not in hidden_paths:
                UsdGeom.Imageable(collision_prim).MakeInvisible()
                hidden_paths.add(collision_path)
                hidden_colliders += 1

        if "/visuals/" not in prim_path or not prim.IsA(UsdGeom.Gprim):
            continue

        intended_material = intended_material_for(prim)
        if intended_material is None:
            continue
        # Bind the authored MJCF material on the render mesh itself. Some OBJ
        # imports retain a child material binding that still wins even when the
        # parent Xform is authored as stronger than descendants.
        target_prim = editable_target_for(prim)
        binding_path = str(target_prim.GetPath())
        # If a broad/non-visual instance root could not be safely expanded, do
        # not reintroduce the first-material-wins corruption on that shared root.
        if binding_path in heterogeneous_target_paths:
            continue
        if binding_path in rebound_paths:
            continue
        UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            intended_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        rebound_paths.add(binding_path)
        rebound += 1
    return rebound, hidden_colliders


def apply_reference_robot_appearance(stage, *, robot_root_paths, appearance_profile=None) -> int:
    """Install the packaged default robot surfaces before physics starts."""
    from .default_appearance import apply_robot
    if appearance_profile not in (None, "handson"):
        raise ValueError("Only the packaged HandsOn appearance is supported")
    return int(apply_robot(stage, robot_root_paths)["links"])



def apply_cinematic_visual_materials(
    stage,
    *,
    robot_root_paths: tuple[str, ...],
    task_root_path: str,
    task_name: str,
) -> int:
    """Add a visual-only PBR finish to the B03 studio render.

    This only authors shader inputs on already-bound render materials.  It
    never touches collision prims, physics materials, mass, or friction, so a
    replay's physical state remains byte-for-byte independent of this profile.
    """

    if task_name != "t09_insert_peg":
        return 0

    from pxr import Sdf, Usd, UsdGeom, UsdShade

    # Robot materials are owned by the explicit real-v1/source appearance
    # profile applied immediately after this helper.  Keep the parameter for
    # caller compatibility, but do not author a second, conflicting finish.
    _ = robot_root_paths

    def bound_shader(prim):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material is None or not material.GetPrim().IsValid():
            return None
        for candidate in Usd.PrimRange(
            material.GetPrim(),
            Usd.TraverseInstanceProxies(),
        ):
            if candidate.IsA(UsdShade.Shader):
                return UsdShade.Shader(candidate)
        return None

    # One imported shader can be bound to many visual meshes.  Profile each
    # task-metal shader once; robot materials use the packaged default appearance.
    profiles: dict[str, tuple[object, str]] = {}
    task_root = stage.GetPrimAtPath(task_root_path)
    if task_root.IsValid():
        for prim in Usd.PrimRange(task_root, Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())
            lowered = prim_path.lower()
            if (
                "/visuals/" not in lowered
                or "/collisions/" in lowered
                or not prim.IsA(UsdGeom.Gprim)
            ):
                continue
            profile = None
            if "industreal_tray_insert_round_peg_8mm" in lowered:
                profile = "socket"
            elif "industreal_round_peg_8mm" in lowered:
                profile = "peg"
            if profile is None:
                continue
            shader = bound_shader(prim)
            if shader is not None and not shader.GetPrim().IsInstanceProxy():
                profiles[str(shader.GetPath())] = (shader, profile)

    def set_float(shader, name: str, value: float) -> None:
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(value))

    configured = 0
    for shader, profile in profiles.values():
        if profile == "peg":
            roughness, metallic, specular = 0.20, 0.70, 0.80
        else:
            roughness, metallic, specular = 0.26, 0.75, 0.85

        set_float(shader, "reflection_roughness_constant", roughness)
        set_float(shader, "metallic_constant", metallic)
        set_float(shader, "specular_level", specular)
        configured += 1

    print(
        "[HandsOn/IsaacLab] B03 studio PBR materials: "
        f"{configured} task shaders enhanced (visual only)",
        flush=True,
    )
    return configured


def apply_scene_visuals(num_envs: int) -> None:
    """Bind the source arena textures and repair imported asset materials."""
    import numpy as np
    import omni.usd
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()

    def create_material(
        path: str,
        color: tuple[float, float, float],
        roughness: float,
        texture: Path | None = None,
        texture_scale: tuple[float, float] | None = None,
        metallic: float = 0.0,
    ):
        material = OmniPBR(
            prim_path=path,
            texture_path=str(texture.resolve()) if texture is not None else None,
            texture_scale=np.asarray(texture_scale, dtype=np.float32) if texture_scale is not None else None,
            color=np.asarray(color, dtype=np.float32),
        )
        material.set_reflection_roughness(roughness)
        material.set_metallic_constant(metallic)
        _configure_pbr_texture_maps(material, texture)
        return material.material

    def bind(path: str, material) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Cannot bind material; missing scene prim: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    floor = create_material(
        "/World/Looks/HandsOnFloor",
        color=(0.62, 0.64, 0.67),
        roughness=0.92,
        texture=_FLOOR_TEXTURE,
        texture_scale=(2.0, 2.0),
    )
    walls = create_material(
        "/World/Looks/HandsOnWalls",
        color=(0.78, 0.79, 0.80),
        roughness=0.88,
        texture=_WALL_TEXTURE,
        texture_scale=(3.0, 3.0),
    )
    table = create_material(
        "/World/Looks/HandsOnTable",
        color=(0.72, 0.72, 0.68),
        roughness=0.68,
        texture=_TABLE_TEXTURE,
        texture_scale=(1.0, 2.0) if _TABLE_TEXTURE is not None else None,
    )
    legs = create_material(
        "/World/Looks/HandsOnTableLegs",
        color=(0.43, 0.45, 0.48),
        roughness=0.32,
        texture=_LEG_TEXTURE,
        texture_scale=(1.0, 1.0),
        metallic=0.82,
    )

    bind("/World/Ground", floor)
    wall_names = (
        "WallLeftCorner",
        "WallRightCorner",
        "WallLeft",
        "WallRight",
        "WallRear",
        "WallFront",
    )
    leg_names = (
        "TableLegFrontLeft",
        "TableLegFrontRight",
        "TableLegRearLeft",
        "TableLegRearRight",
    )
    rebound = 0
    hidden_colliders = 0
    robot_visual_roots: list[str] = []
    for env_index in range(num_envs):
        env_root = f"/World/envs/env_{env_index}"
        for name in wall_names:
            bind(f"{env_root}/{name}", walls)
        bind(f"{env_root}/Table", table)
        for name in leg_names:
            bind(f"{env_root}/{name}", legs)

        for name in ("RightRobot", "LeftRobot", "Peg", "Socket", "CenterRig"):
            visual_root = f"{env_root}/{name}"
            repaired, hidden = _repair_imported_visual_materials(stage, visual_root)
            rebound += repaired
            hidden_colliders += hidden
            if name in {"RightRobot", "LeftRobot", "CenterRig"}:
                robot_visual_roots.append(visual_root)

    apply_reference_robot_appearance(
        stage,
        robot_root_paths=tuple(robot_visual_roots),
    )

    print(
        f"[HandsOn/IsaacLab] Visual migration ready: {rebound} imported meshes rebound, "
        f"{hidden_colliders} collider meshes hidden, arena PBR materials bound"
    )


def randomize_scene_visuals(num_envs: int, random_state) -> None:
    """Apply HandsOn-style visual domain randomization to the live USD stage."""
    import numpy as np
    import omni.usd
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()

    def choice(items):
        return items[int(random_state.randint(0, len(items)))]

    def jitter_color(color: tuple[float, float, float], amount: float = 0.06) -> tuple[float, float, float]:
        base = np.asarray(color, dtype=np.float32)
        jitter = random_state.uniform(-amount, amount, size=3).astype(np.float32)
        return tuple(np.clip(base + jitter, 0.03, 0.95).tolist())

    def create_material(
        path: str,
        color: tuple[float, float, float],
        roughness: float,
        texture: Path | None = None,
        texture_scale: tuple[float, float] | None = None,
        metallic: float = 0.0,
    ):
        material = OmniPBR(
            prim_path=path,
            texture_path=str(texture.resolve()) if texture is not None else None,
            texture_scale=np.asarray(texture_scale, dtype=np.float32) if texture_scale is not None else None,
            color=np.asarray(color, dtype=np.float32),
        )
        material.set_reflection_roughness(float(roughness))
        material.set_metallic_constant(float(metallic))
        _configure_pbr_texture_maps(material, texture)
        return material.material

    def bind(path: str, material) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    def set_attr(prim_path: str, attr_names: tuple[str, ...], value) -> None:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        for attr_name in attr_names:
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(value)
                return

    table_textures = _available_table_textures()
    table_texture = choice(table_textures)
    external_table_texture = table_texture not in _RANDOM_TABLE_TEXTURES
    table_material = create_material(
        "/World/Looks/HandsOnTableRandom",
        color=jitter_color((0.72, 0.72, 0.68), amount=0.04),
        roughness=float(random_state.uniform(0.45, 0.9)),
        texture=table_texture,
        texture_scale=(
            float(random_state.uniform(3.0, 5.0))
            if external_table_texture
            else float(random_state.uniform(0.8, 1.8)),
            float(random_state.uniform(6.0, 10.0))
            if external_table_texture
            else float(random_state.uniform(1.2, 3.0)),
        ),
    )
    floor_material = create_material(
        "/World/Looks/HandsOnFloorRandom",
        color=jitter_color(choice(_BACKGROUND_FLOOR_COLORS), amount=0.06),
        roughness=float(random_state.uniform(0.78, 0.96)),
        texture=_FLOOR_TEXTURE,
        texture_scale=(float(random_state.uniform(1.5, 3.5)), float(random_state.uniform(1.5, 3.5))),
    )
    wall_material = create_material(
        "/World/Looks/HandsOnWallsRandom",
        color=jitter_color(choice(_BACKGROUND_WALL_COLORS), amount=0.06),
        roughness=float(random_state.uniform(0.78, 0.96)),
        texture=_WALL_TEXTURE,
        texture_scale=(float(random_state.uniform(2.0, 4.0)), float(random_state.uniform(2.0, 4.0))),
    )

    bind("/World/Ground", floor_material)
    wall_names = (
        "WallLeftCorner",
        "WallRightCorner",
        "WallLeft",
        "WallRight",
        "WallRear",
        "WallFront",
    )
    for env_index in range(num_envs):
        env_root = f"/World/envs/env_{env_index}"
        bind(f"{env_root}/Table", table_material)
        for name in wall_names:
            bind(f"{env_root}/{name}", wall_material)

    tint = np.asarray(random_state.uniform(0.82, 1.16, size=3), dtype=np.float32)
    dome_color = tuple(np.clip(tint * random_state.uniform(0.86, 1.08), 0.65, 1.25).tolist())
    key_color = tuple(np.clip(tint * random_state.uniform(0.90, 1.18), 0.65, 1.25).tolist())
    set_attr("/World/DomeLight", ("inputs:intensity", "intensity"), float(random_state.uniform(280.0, 900.0)))
    set_attr("/World/DomeLight", ("inputs:color", "color"), dome_color)
    set_attr("/World/KeyLight", ("inputs:intensity", "intensity"), float(random_state.uniform(550.0, 1700.0)))
    set_attr("/World/KeyLight", ("inputs:color", "color"), key_color)

    print(
        "[HandsOn/IsaacLab] Randomized visuals: "
        f"table_texture={table_texture.name}, "
        f"texture_pool={'external' if table_texture not in _RANDOM_TABLE_TEXTURES else 'bundled'}, "
        "dome/key lighting updated",
        flush=True,
    )


def randomize_task_scene_visuals(root_path: str, random_state) -> Path:
    """Bind the same external PBR table pool to an imported A/B/C task scene."""
    import numpy as np
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Cannot randomize missing task root: {root_path}")

    def preview_texture_material(
        path: str,
        *,
        texture_path: Path,
        texture_scale: tuple[float, float],
        color: tuple[float, float, float],
        roughness: float,
    ) -> UsdShade.Material:
        """Author a portable textured material for Isaac Sim 5/6."""

        material = UsdShade.Material.Define(stage, path)
        surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput(
            "roughness",
            Sdf.ValueTypeNames.Float,
        ).Set(float(roughness))
        surface.CreateInput(
            "metallic",
            Sdf.ValueTypeNames.Float,
        ).Set(0.0)

        primvar = UsdShade.Shader.Define(stage, f"{path}/PrimvarReader")
        primvar.CreateIdAttr("UsdPrimvarReader_float2")
        primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        primvar_output = primvar.CreateOutput(
            "result",
            Sdf.ValueTypeNames.Float2,
        )

        transform = UsdShade.Shader.Define(stage, f"{path}/UVTransform")
        transform.CreateIdAttr("UsdTransform2d")
        transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            primvar_output
        )
        transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(*map(float, texture_scale))
        )
        transformed_st = transform.CreateOutput(
            "result",
            Sdf.ValueTypeNames.Float2,
        )

        texture = UsdShade.Shader.Define(stage, f"{path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture_path.resolve()))
        )
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            transformed_st
        )
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB"
        )
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(float(color[0]), float(color[1]), float(color[2]), 1.0)
        )
        texture_rgb = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).ConnectToSource(texture_rgb)
        material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(),
            "surface",
        )
        return material

    table_prim = None
    for prim in Usd.PrimRange(root):
        if (
            prim.GetName().lower() in {"table", "worktable"}
            and prim.IsA(UsdGeom.Xformable)
        ):
            table_prim = prim
            break
    b320_root = stage.GetPrimAtPath(f"{root_path}/B320Benchmark")
    if table_prim is None and not b320_root.IsValid():
        raise RuntimeError(f"Cannot find table prim below {root_path}")

    textures = _available_table_textures()
    texture = textures[int(random_state.randint(0, len(textures)))]
    external = texture not in _RANDOM_TABLE_TEXTURES
    scale = (
        float(random_state.uniform(3.0, 5.0))
        if external
        else float(random_state.uniform(0.8, 1.8)),
        float(random_state.uniform(6.0, 10.0))
        if external
        else float(random_state.uniform(1.2, 3.0)),
    )
    material = preview_texture_material(
        "/World/Looks/HandsOnTaskTableRandom",
        texture_path=texture,
        texture_scale=scale,
        color=(0.74, 0.74, 0.71),
        roughness=float(random_state.uniform(0.45, 0.8)),
    )
    normal_bound, roughness_bound = False, False
    if table_prim is not None:
        table_visual_prim = next(
            (
                prim
                for prim in Usd.PrimRange(
                    table_prim,
                    Usd.TraverseInstanceProxies(),
                )
                if prim.GetName().lower() == "table_visual"
                and prim.IsA(UsdGeom.Gprim)
            ),
            None,
        )
        material_target = (
            table_visual_prim
            if table_visual_prim is not None
            and not table_visual_prim.IsInstanceProxy()
            else table_prim
        )
        UsdShade.MaterialBindingAPI.Apply(material_target).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    floor_color = np.asarray(
        _BACKGROUND_FLOOR_COLORS[
            int(random_state.randint(0, len(_BACKGROUND_FLOOR_COLORS)))
        ],
        dtype=np.float32,
    )
    floor_scale = np.asarray(
        [
            random_state.uniform(2.5, 6.0),
            random_state.uniform(2.5, 6.0),
        ],
        dtype=np.float32,
    )
    floor_material = preview_texture_material(
        "/World/Looks/HandsOnTaskBackgroundRandom",
        texture_path=_FLOOR_TEXTURE,
        texture_scale=(float(floor_scale[0]), float(floor_scale[1])),
        color=(
            float(floor_color[0]),
            float(floor_color[1]),
            float(floor_color[2]),
        ),
        roughness=float(random_state.uniform(0.58, 0.92)),
    )
    floor_targets = []
    for prim in Usd.PrimRange(root):
        if prim.GetName().lower() not in {"floor", "ground", "background"}:
            continue
        if prim.IsInstanceProxy():
            continue
        floor_targets.append(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            floor_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    tint = np.asarray(random_state.uniform(0.82, 1.16, size=3), dtype=np.float32)
    dome_color = tuple(
        np.clip(tint * random_state.uniform(0.86, 1.08), 0.65, 1.25).tolist()
    )
    key_color = tuple(
        np.clip(tint * random_state.uniform(0.90, 1.18), 0.65, 1.25).tolist()
    )
    dome_intensity = float(random_state.uniform(280.0, 900.0))
    key_intensity = float(random_state.uniform(1800.0, 5200.0))
    key_position = tuple(
        np.asarray(
            [
                random_state.uniform(-0.75, 0.25),
                random_state.uniform(-0.75, 0.75),
                random_state.uniform(1.65, 2.65),
            ],
            dtype=np.float64,
        ).tolist()
    )

    dome_light = stage.GetPrimAtPath("/World/HandsOnDomeLight")
    if dome_light.IsValid():
        intensity_attr = dome_light.GetAttribute("inputs:intensity")
        color_attr = dome_light.GetAttribute("inputs:color")
        if intensity_attr and intensity_attr.IsValid():
            intensity_attr.Set(dome_intensity)
        if color_attr and color_attr.IsValid():
            color_attr.Set(dome_color)

    key_light = stage.GetPrimAtPath("/World/HandsOnKeyLight")
    if key_light.IsValid():
        intensity_attr = key_light.GetAttribute("inputs:intensity")
        color_attr = key_light.GetAttribute("inputs:color")
        radius_attr = key_light.GetAttribute("inputs:radius")
        if intensity_attr and intensity_attr.IsValid():
            intensity_attr.Set(key_intensity)
        if color_attr and color_attr.IsValid():
            color_attr.Set(key_color)
        if radius_attr and radius_attr.IsValid():
            radius_attr.Set(float(random_state.uniform(0.08, 0.30)))
        translate_attr = key_light.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            translate_attr.Set(Gf.Vec3d(*key_position))

    print(
        "[HandsOn/IsaacLab] Randomized ABC visuals: "
        f"table_texture={texture.name}, "
        f"texture_pool={'external' if external else 'bundled'}, "
        f"normal={'yes' if normal_bound else 'no'}, "
        f"roughness={'yes' if roughness_bound else 'no'}, "
        f"texture_scale=({scale[0]:.2f}, {scale[1]:.2f}), "
        f"background_rgb=({floor_color[0]:.2f},{floor_color[1]:.2f},{floor_color[2]:.2f}), "
        f"background_prims={len(floor_targets)}, "
        f"dome_rgb=({dome_color[0]:.2f},{dome_color[1]:.2f},{dome_color[2]:.2f}), "
        f"key_pos=({key_position[0]:.2f},{key_position[1]:.2f},{key_position[2]:.2f})",
        flush=True,
    )
    return texture


def validate_asset_files() -> None:
    """Fail early with a useful error when an asset checkout is incomplete."""
    required = (
        RIGHT_ROBOT_MJCF,
        LEFT_ROBOT_MJCF,
        PEG_MJCF,
        SOCKET_MJCF,
        _CENTER_BODY_MESH,
        _CENTER_HEAD_MESH,
        _FLOOR_TEXTURE,
        _WALL_TEXTURE,
        _LEG_TEXTURE,
    )
    required += tuple(_RANDOM_TABLE_TEXTURES)
    if _TABLE_TEXTURE is not None:
        required += (_TABLE_TEXTURE,)
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing Isaac Lab preview assets:\n{formatted}")


def fix_robot_base_paths(root_link_paths: tuple[str, ...]) -> None:
    """Fix robot root links after they have reached their authored world poses."""
    cfg = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
    for root_link_path in root_link_paths:
        sim_utils.modify_articulation_root_properties(root_link_path, cfg)


def fix_robot_bases(num_envs: int) -> None:
    """Create world-fixed joints after each robot has its final world pose."""
    root_link_paths = []
    for env_index in range(num_envs):
        for side in ("right", "left"):
            root_link_paths.append(
                mjcf_robot_root_link_path(
                    f"/World/envs/env_{env_index}/{side.capitalize()}Robot",
                    side,
                )
            )
    fix_robot_base_paths(tuple(root_link_paths))
