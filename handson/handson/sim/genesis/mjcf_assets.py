"""Backend-neutral MJCF preparation for Genesis IPC robot coupling."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Literal
import xml.etree.ElementTree as ET


HandCollisionProfile = Literal["full", "pbd_fingertips"]


def _default_cache_dir() -> Path:
    configured = os.environ.get("HANDSON_GENESIS_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(tempfile.gettempdir()) / "handson_genesis" / "mjcf"


def _resolve_mesh_paths(root: ET.Element, source: Path) -> None:
    """Make MJCF mesh paths independent of the generated cache location."""
    compiler = root.find("compiler")
    mesh_dir = source.parent
    if compiler is not None and compiler.attrib.get("meshdir"):
        mesh_dir = (source.parent / compiler.attrib["meshdir"]).resolve()
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("file")
        if not filename or Path(filename).is_absolute():
            continue
        source_relative = (source.parent / filename).resolve()
        candidate = (
            source_relative if source_relative.is_file() else mesh_dir / filename
        )
        mesh.attrib["file"] = str(candidate.resolve())


def _replace_hand_colliders(
    hand_root: ET.Element,
    primitive_source: Path,
    *,
    side: str,
    collision_profile: HandCollisionProfile = "full",
    ensure_proximal_finger_collisions: bool = False,
) -> None:
    if collision_profile not in {"full", "pbd_fingertips"}:
        raise ValueError(
            f"Unsupported hand collision profile: {collision_profile!r}"
        )

    primitive_root = ET.parse(primitive_source).getroot()
    primitive_bodies = {
        body.attrib["name"]: body
        for body in primitive_root.iter("body")
        if "name" in body.attrib
    }

    replaced = 0
    inserted: list[tuple[str, str, str]] = []
    for body in hand_root.iter("body"):
        body_name = body.attrib.get("name", "")
        primitive_body = primitive_bodies.get(body_name)
        if primitive_body is None:
            continue
        for geom in list(body.findall("geom")):
            if geom.attrib.get("contype", "1") != "0":
                body.remove(geom)
                replaced += 1
        for geom in primitive_body.findall("geom"):
            if geom.attrib.get("contype", "1") == "0":
                continue
            geom_name = geom.attrib.get("name", "")
            geom_type = geom.attrib.get("type", "mesh")
            if collision_profile == "pbd_fingertips":
                palm_collider = (
                    body_name == f"{side}_palm_link"
                    and geom_name == f"{side}_palm_link_col2"
                    and geom_type == "box"
                )
                fingertip_collider = any(
                    body_name == f"{side}_finger{finger}_link4"
                    and geom_name == f"{side}_finger{finger}_tip_link_col2"
                    and geom_type == "sphere"
                    for finger in range(1, 6)
                )
                if not (palm_collider or fingertip_collider):
                    continue
            collision = deepcopy(geom)
            if ensure_proximal_finger_collisions:
                collision.attrib["contype"] = "1"
                collision.attrib["conaffinity"] = "1"
            body.append(collision)
            inserted.append((body_name, geom_name, geom_type))

        if (
            ensure_proximal_finger_collisions
            and body_name
            in {
                f"{side}_finger{finger}_link1"
                for finger in range(1, 6)
            }
            and not any(
                geom.attrib.get("contype", "1") != "0"
                for geom in primitive_body.findall("geom")
            )
        ):
            visual = next(iter(primitive_body.findall("geom")), None)
            if visual is None:
                raise ValueError(
                    f"Primitive hand body has no geometry: {body_name}"
                )
            collision = deepcopy(visual)
            collision.attrib.update(
                {
                    "name": f"{body_name}_col2",
                    "contype": "1",
                    "conaffinity": "1",
                    "group": "3",
                }
            )
            collision.attrib.pop("density", None)
            collision.attrib.pop("mass", None)
            body.append(collision)
            inserted.append(
                (
                    body_name,
                    collision.attrib["name"],
                    collision.attrib.get("type", "mesh"),
                )
            )

    if replaced == 0:
        raise RuntimeError(f"No {side} hand colliders were replaced")
    if collision_profile == "pbd_fingertips":
        expected = {
            (f"{side}_palm_link", f"{side}_palm_link_col2", "box"),
            *{
                (
                    f"{side}_finger{finger}_link4",
                    f"{side}_finger{finger}_tip_link_col2",
                    "sphere",
                )
                for finger in range(1, 6)
            },
        }
        if len(inserted) != len(expected) or set(inserted) != expected:
            raise RuntimeError(
                f"Incomplete {side} PBD fingertip collision profile: "
                f"expected {sorted(expected)!r}, got {sorted(inserted)!r}"
            )


def _find_hand_root(root: ET.Element, side: str, source: Path) -> ET.Element:
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
    return hand_root


def prepare_rokae_wuji_mjcf(
    source: Path,
    primitive_source: Path,
    *,
    side: str,
    floating_hand_base: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Keep authored robot visuals but use the shipped primitive hand colliders.

    The source Wuji MJCF uses detailed STL meshes for both rendering and
    collision.  libuipc would otherwise preprocess hundreds of thousands of
    hand triangles.  The replacement geometry is copied from HandsOn's authored
    ``*_primitive.xml`` hand assets; robot kinematics and visual meshes remain
    unchanged.
    """
    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported robot side: {side!r}")
    source = source.resolve()
    primitive_source = primitive_source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not primitive_source.is_file():
        raise FileNotFoundError(primitive_source)

    output_dir = (cache_dir or _default_cache_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source)
    root = tree.getroot()
    # The derived XML lives outside the source tree, so make mesh paths
    # absolute before handing the document to MuJoCo/Genesis.
    _resolve_mesh_paths(root, source)
    hand_root = _find_hand_root(root, side, source)

    if floating_hand_base:
        existing_root_joints = [
            child for child in hand_root if child.tag in {"joint", "freejoint"}
        ]
        if existing_root_joints:
            names = [
                child.attrib.get("name", child.tag) for child in existing_root_joints
            ]
            raise ValueError(
                f"Cannot add a floating {side} hand base; palm already has joints {names}"
            )
        hand_root.insert(
            0,
            ET.Element("freejoint", {"name": f"{side}_palm_free"}),
        )

    _replace_hand_colliders(hand_root, primitive_source, side=side)

    digest = hashlib.sha256(
        source.read_bytes()
        + primitive_source.read_bytes()
        + (b"floating-hand-base" if floating_hand_base else b"fixed-hand-base")
    ).hexdigest()[:12]
    output_path = output_dir / f"rokae_wuji1_{side}_genesis_{digest}.xml"
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not output_path.exists() or output_path.read_bytes() != content:
        output_path.write_bytes(content)
    return output_path


def prepare_detached_rokae_wuji_hand_mjcf(
    source: Path,
    primitive_source: Path,
    *,
    side: str,
    collision_profile: HandCollisionProfile = "full",
    cache_dir: Path | None = None,
    ensure_proximal_finger_collisions: bool = False,
) -> Path:
    """Extract the original Wuji palm subtree without loading the ROKAE arm.

    ``rokae_wuji1_{side}.xml`` is the canonical collection asset.  Its hand
    uses the original black visual treatment and full authored actuator force
    ranges, while the separately shipped standalone hand has a diagnostic
    multi-colour treatment.  This helper retains only the canonical palm and
    five finger chains, then adds a floating palm joint for Plan-style live
    control.  No ROKAE body, joint, actuator, sensor, or mesh is copied.
    """
    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported robot side: {side!r}")
    source = source.resolve()
    primitive_source = primitive_source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not primitive_source.is_file():
        raise FileNotFoundError(primitive_source)

    source_root = ET.parse(source).getroot()
    _resolve_mesh_paths(source_root, source)
    source_hand = _find_hand_root(source_root, side, source)
    hand_root = deepcopy(source_hand)

    # In the complete robot the parent hand-mount body supplies this class.
    # The detached palm must inherit it directly to preserve the authored
    # joint damping and armature values.
    hand_root.attrib["childclass"] = f"wuji_{side}"
    hand_root.insert(
        0,
        ET.Element("freejoint", {"name": f"{side}_palm_free"}),
    )

    output_root = ET.Element(
        "mujoco", {"model": f"rokae_wuji1_{side}_detached_hand"}
    )
    for tag in ("compiler", "option", "default"):
        element = source_root.find(tag)
        if element is not None:
            output_root.append(deepcopy(element))

    referenced_meshes = {
        geom.attrib["mesh"]
        for geom in hand_root.iter("geom")
        if "mesh" in geom.attrib
    }
    source_assets = source_root.find("asset")
    mesh_assets = {
        mesh.attrib.get("name"): mesh
        for mesh in (() if source_assets is None else source_assets.findall("mesh"))
    }
    missing_meshes = referenced_meshes - mesh_assets.keys()
    if missing_meshes:
        missing = ", ".join(sorted(missing_meshes))
        raise ValueError(f"Missing hand mesh assets in {source}: {missing}")
    output_assets = ET.SubElement(output_root, "asset")
    if source_assets is not None:
        for mesh in source_assets.findall("mesh"):
            if mesh.attrib.get("name") in referenced_meshes:
                output_assets.append(deepcopy(mesh))

    output_worldbody = ET.SubElement(output_root, "worldbody")
    output_worldbody.append(hand_root)

    body_names = {
        body.attrib["name"]
        for body in hand_root.iter("body")
        if "name" in body.attrib
    }
    source_contact = source_root.find("contact")
    if source_contact is not None:
        output_contact = ET.Element("contact")
        for exclude in source_contact.findall("exclude"):
            if (
                exclude.attrib.get("body1") in body_names
                and exclude.attrib.get("body2") in body_names
            ):
                output_contact.append(deepcopy(exclude))
        if len(output_contact):
            output_root.append(output_contact)

    source_actuator = source_root.find("actuator")
    if source_actuator is not None:
        output_actuator = ET.Element("actuator")
        joint_prefix = f"{side}_finger"
        for actuator in source_actuator:
            if actuator.attrib.get("joint", "").startswith(joint_prefix):
                output_actuator.append(deepcopy(actuator))
        if len(output_actuator):
            output_root.append(output_actuator)

    _replace_hand_colliders(
        hand_root,
        primitive_source,
        side=side,
        collision_profile=collision_profile,
        ensure_proximal_finger_collisions=ensure_proximal_finger_collisions,
    )

    output_dir = (cache_dir or _default_cache_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        source.read_bytes()
        + primitive_source.read_bytes()
        + b"detached-canonical-palm-v1"
        + (
            b""
            if collision_profile == "full"
            else b"\0collision-profile:pbd-fingertips-v1"
        )
        + (
            b"\0proximal-finger-collisions-v1"
            if ensure_proximal_finger_collisions
            else b""
        )
    ).hexdigest()[:12]
    output_path = output_dir / (
        f"rokae_wuji1_{side}_detached_hand_genesis_{digest}.xml"
    )
    content = ET.tostring(output_root, encoding="utf-8", xml_declaration=True)
    if not output_path.exists() or output_path.read_bytes() != content:
        output_path.write_bytes(content)
    return output_path


__all__ = [
    "prepare_detached_rokae_wuji_hand_mjcf",
    "prepare_rokae_wuji_mjcf",
]
