from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
_SCENE_SOURCE = (
    _PACKAGE_ROOT / "sim" / "isaaclab" / "bimanual_assembly_scene.py"
)
_XML_DIR = _PACKAGE_ROOT / "sim" / "envs" / "xmls"


def _load_robot_mjcf_generator(
    cache_dir: Path,
    *,
    xml_dir: Path = _XML_DIR,
    right_robot: Path | None = None,
    left_robot: Path | None = None,
) -> SimpleNamespace:
    """Load the pure MJCF generator without importing Isaac Lab."""

    source = _SCENE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SCENE_SOURCE))
    function_names = {
        "_robot_mjcf_variant_digest",
        "_robot_usd_cache_dir",
        "_prepare_robot_mjcf",
        "_framed_file_tree_identity",
        "_external_asset_dependency_paths",
        "_mjcf_referenced_asset_identity",
        "_loaded_usd_cache_identity",
        "_strict_loaded_mjcf_usd_asset_dir",
        "b03_generated_robot_asset_identity",
        "_canonical_mjcf_sha256",
        "b03_loaded_task_asset_identity",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "_ROBOT_MJCF_GENERATOR_SCHEMA"
            for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            body.append(node)

    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {
        "Path": Path,
        "deepcopy": deepcopy,
        "hashlib": hashlib,
        "math": math,
        "os": os,
        "shutil": shutil,
        "ET": ET,
        "_CACHE_DIR": cache_dir,
        "_XML_DIR": xml_dir,
        "RIGHT_ROBOT_MJCF": right_robot
        or xml_dir / "rokae_wuji1_right.xml",
        "LEFT_ROBOT_MJCF": left_robot
        or xml_dir / "rokae_wuji1_left.xml",
    }
    exec(compile(module, str(_SCENE_SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _hand_colliders(path: Path, side: str) -> list[ET.Element]:
    root = ET.parse(path).getroot()
    hand = root.find(f".//body[@name='{side}_palm_link']")
    assert hand is not None
    return [
        geom
        for body in hand.iter("body")
        for geom in body.findall("geom")
        if geom.attrib.get("contype", "1") != "0"
    ]


@pytest.mark.usefixtures("rokae_assets")
def test_b03_newton_installs_52_hand_colliders_in_mask_one(
    tmp_path: Path,
) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")

    colliders = []
    for side in ("right", "left"):
        generated = generator._prepare_robot_mjcf(
            side,
            ensure_proximal_finger_collisions=True,
        )
        side_colliders = _hand_colliders(generated, side)
        assert len(side_colliders) == 26
        colliders.extend(side_colliders)

    assert len(colliders) == 52
    assert {
        (geom.attrib.get("contype"), geom.attrib.get("conaffinity"))
        for geom in colliders
    } == {("1", "1")}


@pytest.mark.usefixtures("rokae_assets")
def test_robot_variant_installs_complete_hand_with_contact_margin(
    tmp_path: Path,
) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")

    baseline = generator._prepare_robot_mjcf("right")
    generated = generator._prepare_robot_mjcf(
        "right",
        ensure_proximal_finger_collisions=True,
        hand_contact_margin=0.002,
    )
    baseline_colliders = _hand_colliders(baseline, "right")
    colliders = _hand_colliders(generated, "right")

    assert generated != baseline
    assert len(baseline_colliders) < 26
    assert all("margin" not in geom.attrib for geom in baseline_colliders)
    assert len(colliders) == 26
    assert {
        (geom.attrib.get("contype"), geom.attrib.get("conaffinity"))
        for geom in colliders
    } == {("1", "1")}
    assert {geom.attrib.get("margin") for geom in colliders} == {"0.002"}
    assert {geom.attrib.get("gap") for geom in colliders} == {"0"}


@pytest.mark.usefixtures("rokae_assets")
def test_wuji2_robot_derivative_keeps_official_joints_and_canonical_links(
    tmp_path: Path,
) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")

    generated = generator._prepare_robot_mjcf(
        "right",
        hand_type="wuji2",
    )
    root = ET.parse(generated).getroot()
    joint_names = {
        joint.attrib.get("name", "")
        for joint in root.iter("joint")
    }

    assert root.find(".//body[@name='right_palm_link']") is not None
    assert root.find(".//body[@name='right_finger5_link4']") is not None
    assert {
        "r_thumb_cmc_flex",
        "r_index_finger_mcp_abd",
        "r_middle_finger_pip",
        "r_ring_finger_dip",
        "r_pinky_dip",
    } <= joint_names
    assert all(
        mesh.attrib["file"].startswith("assets/")
        for mesh in root.iter("mesh")
        if mesh.attrib.get("file")
    )


@pytest.mark.usefixtures("rokae_assets")
def test_robot_mjcf_and_usd_cache_identity_tracks_raw_input_bytes(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"
    primitive_dir = input_root / "wuji_hand1" / "mjcf"
    primitive_dir.mkdir(parents=True)
    robot = input_root / "rokae_wuji1_right.xml"
    primitive = primitive_dir / "right_primitive.xml"
    robot_bytes = (_XML_DIR / robot.name).read_bytes()
    primitive_bytes = (
        _XML_DIR / "wuji_hand1" / "mjcf" / primitive.name
    ).read_bytes()
    robot.write_bytes(robot_bytes)
    primitive.write_bytes(primitive_bytes)
    for relative in (
        Path("wuji_hand1/meshes"),
        Path("wuji_attachment"),
        Path("wrist_fisheye"),
        Path("rokae_arm"),
    ):
        link = input_root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(_XML_DIR / relative, target_is_directory=True)

    generator = _load_robot_mjcf_generator(
        tmp_path / "cache",
        xml_dir=input_root,
        right_robot=robot,
    )

    def generate() -> Path:
        return generator._prepare_robot_mjcf(
            "right",
            ensure_proximal_finger_collisions=True,
        )

    baseline = generate()
    baseline_mesh_references = [
        mesh.attrib["file"]
        for mesh in ET.parse(baseline).getroot().iter("mesh")
        if mesh.attrib.get("file")
    ]
    robot.write_bytes(robot_bytes + b"\n<!-- source-byte-only mutation -->\n")
    source_changed = generate()
    robot.write_bytes(robot_bytes)
    primitive.write_bytes(
        primitive_bytes + b"\n<!-- primitive-byte-only mutation -->\n"
    )
    primitive_changed = generate()

    assert len({baseline, source_changed, primitive_changed}) == 3
    assert baseline.read_bytes() == source_changed.read_bytes()
    assert baseline.read_bytes() == primitive_changed.read_bytes()
    assert generate() == primitive_changed
    assert baseline_mesh_references
    assert all(
        reference.startswith("assets/")
        and ".." not in Path(reference).parts
        for reference in baseline_mesh_references
    )

    other_generator = _load_robot_mjcf_generator(
        tmp_path / "different" / "cache" / "depth",
        xml_dir=input_root,
        right_robot=robot,
    )
    other_generated = other_generator._prepare_robot_mjcf(
        "right",
        ensure_proximal_finger_collisions=True,
    )
    assert other_generated.name == generate().name
    assert other_generated.read_bytes() == generate().read_bytes()

    usd_dirs = {
        generator._robot_usd_cache_dir(path)
        for path in (baseline, source_changed, primitive_changed)
    }
    assert len(usd_dirs) == 3
    assert all(path.parent == tmp_path / "cache" / "usd" for path in usd_dirs)


def test_b03_asset_identity_hashes_recursive_mjcf_assets_and_loaded_usd(
    tmp_path: Path,
) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    mesh = asset_root / "collision.bin"
    mesh.write_bytes(b"collision-v1")
    included = asset_root / "included.xml"
    included.write_text(
        '<mujoco><asset><mesh name="collision" file="collision.bin"/>'
        "</asset></mujoco>",
        encoding="utf-8",
    )
    mjcf = asset_root / "robot.xml"
    mjcf.write_text(
        '<mujoco><include file="included.xml"/><worldbody/></mujoco>',
        encoding="utf-8",
    )
    usd_dir = tmp_path / "usd"
    loaded_asset = usd_dir / mjcf.stem
    (loaded_asset / "payloads").mkdir(parents=True)
    root_usd = loaded_asset / "robot.usda"
    payload_usd = loaded_asset / "payloads" / "physics.usdc"
    root_usd.write_bytes(b"root-usd-v1")
    payload_usd.write_bytes(b"physics-usd-v1")

    baseline = generator.b03_generated_robot_asset_identity(mjcf, usd_dir)
    mesh.write_bytes(b"collision-v2")
    mesh_changed = generator.b03_generated_robot_asset_identity(mjcf, usd_dir)
    mesh.write_bytes(b"collision-v1")
    payload_usd.write_bytes(b"physics-usd-v2")
    usd_changed = generator.b03_generated_robot_asset_identity(mjcf, usd_dir)

    assert baseline["referenced_assets"]["file_count"] == 2
    assert baseline["loaded_usd_cache"]["file_count"] == 2
    assert baseline["referenced_assets"]["combined_sha256"] != (
        mesh_changed["referenced_assets"]["combined_sha256"]
    )
    assert baseline["loaded_usd_cache"]["combined_sha256"] != (
        usd_changed["loaded_usd_cache"]["combined_sha256"]
    )


def test_b03_asset_identity_rejects_empty_usd_cache(tmp_path: Path) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")
    mesh = tmp_path / "mesh.bin"
    mesh.write_bytes(b"mesh")
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text(
        '<mujoco><asset><mesh name="mesh" file="mesh.bin"/></asset></mujoco>',
        encoding="utf-8",
    )
    usd_dir = tmp_path / "usd"
    (usd_dir / mjcf.stem).mkdir(parents=True)

    with pytest.raises(RuntimeError, match="has no files"):
        generator.b03_generated_robot_asset_identity(mjcf, usd_dir)


def test_b03_task_identity_is_stable_and_rejects_ambiguous_usd_variants(
    tmp_path: Path,
) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")
    mesh = tmp_path / "peg.obj"
    mesh.write_bytes(b"peg-v1")
    task_mjcf = tmp_path / "t09_insert_peg_scene_isaac.xml"
    task_mjcf.write_text(
        '<mujoco><asset><mesh name="peg" file="peg.obj"/></asset></mujoco>',
        encoding="utf-8",
    )
    usd_dir = tmp_path / "usd"
    loaded_asset = usd_dir / task_mjcf.stem
    (loaded_asset / "payloads").mkdir(parents=True)
    (loaded_asset / f"{task_mjcf.stem}.usda").write_bytes(b"root-usd")
    (loaded_asset / "payloads" / "base.usda").write_bytes(b"base-usd")
    # Converter metadata is deliberately outside the loaded tree and may
    # contain a host-specific cache root or generation timestamp.
    (usd_dir / "config.yaml").write_text(
        "asset_path: /first/host/cache/task.xml\n",
        encoding="utf-8",
    )

    baseline = generator.b03_loaded_task_asset_identity(task_mjcf, usd_dir)
    (usd_dir / "config.yaml").write_text(
        "asset_path: /second/host/cache/task.xml\n",
        encoding="utf-8",
    )
    repeated = generator.b03_loaded_task_asset_identity(task_mjcf, usd_dir)

    assert repeated == baseline
    assert baseline["version"] == 2
    assert baseline["generated_mjcf_sha256"] == hashlib.sha256(
        task_mjcf.read_bytes()
    ).hexdigest()

    ambiguous = usd_dir / f"{task_mjcf.stem}_1"
    ambiguous.mkdir()
    with pytest.raises(RuntimeError, match="ambiguous importer variants"):
        generator.b03_loaded_task_asset_identity(task_mjcf, usd_dir)


def test_robot_variant_digest_changes_when_external_asset_bytes_change(
    tmp_path: Path,
) -> None:
    generator = _load_robot_mjcf_generator(tmp_path / "cache")
    mesh = tmp_path / "mesh.bin"
    mesh.write_bytes(b"mesh-v1")
    source = tmp_path / "robot.xml"
    source.write_text(
        '<mujoco><asset><mesh name="mesh" file="mesh.bin"/></asset></mujoco>',
        encoding="utf-8",
    )
    primitive = tmp_path / "primitive.xml"
    primitive.write_text("<mujoco/>", encoding="utf-8")

    baseline = generator._robot_mjcf_variant_digest(
        source,
        primitive,
        preserve_base_pose=False,
        ensure_proximal_finger_collisions=True,
    )
    mesh.write_bytes(b"mesh-v2")
    changed = generator._robot_mjcf_variant_digest(
        source,
        primitive,
        preserve_base_pose=False,
        ensure_proximal_finger_collisions=True,
    )

    assert baseline != changed
