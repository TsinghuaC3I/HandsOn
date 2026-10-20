#!/usr/bin/env python3
"""Vendor the official Wuji Hand 2 Beta 1 MJCF into the collection models.

The generated robot XMLs retain HandsOn's historical palm/finger body aliases so
task contact logic can be shared by Wuji Hand 1 and Hand 2.  The Hand 2 joint and
actuator names remain the official SDK/model names.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


DESCRIPTION_TAG = "v2026.8.3"
DESCRIPTION_COMMIT = "8271644a78d69ed9a4adcf9165d882c64ad33dfa"
DESCRIPTION_URL = "https://github.com/wuji-technology/wuji-description"


def _body_aliases(side: str) -> dict[str, str]:
    prefix = "l" if side == "left" else "r"
    word = side
    aliases = {f"{prefix}_wrist": f"{word}_wuji2_wrist_link"}
    groups = (
        ("thumb", 1),
        ("index_finger", 2),
        ("middle_finger", 3),
        ("ring_finger", 4),
        ("pinky", 5),
    )
    link_parts = ("proximal", "proximal_abd", "middle", "distal")
    for finger, finger_index in groups:
        for link_index, part in enumerate(link_parts, start=1):
            aliases[f"{prefix}_{finger}_{part}"] = (
                f"{word}_finger{finger_index}_link{link_index}"
            )
    return aliases


def _find_named_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"Body {name!r} not found")
    return body


def _rename_hand_bodies(hand_root: ET.Element, side: str) -> None:
    aliases = _body_aliases(side)
    for body in hand_root.iter("body"):
        old_name = body.get("name", "")
        if old_name in aliases:
            body.set("name", aliases[old_name])

    # Preserve the fingertip collision names consumed by task success/contact code.
    prefix = "l" if side == "left" else "r"
    fingers = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
    for finger_index, finger in enumerate(fingers, start=1):
        distal = _find_named_body(hand_root, f"{side}_finger{finger_index}_link4")
        visual_and_collision = list(distal.findall("geom"))
        for geom in visual_and_collision:
            suffix = "vis" if geom.get("contype") == "0" else "col2"
            geom.set("name", f"{side}_finger{finger_index}_tip_link_{suffix}")
        site = distal.find(f"site[@name='{prefix}_{finger}_tip']")
        if site is not None:
            site.set("name", f"{side}_finger{finger_index}_tip")


def _configure_collision_geoms(hand_root: ET.Element, side: str) -> None:
    contype, conaffinity = (("4", "3") if side == "left" else ("2", "5"))
    for geom in hand_root.iter("geom"):
        if geom.get("contype") == "0":
            geom.set("group", "1")
            geom.set("rgba", "0.02 0.02 0.02 1")
            continue
        geom.set("group", "2")
        geom.set("contype", contype)
        geom.set("conaffinity", conaffinity)
        geom.set("rgba", "0.02 0.02 0.02 1")
        geom.set("friction", "3 0.5 0.02")
        geom.set("condim", "4")
        geom.set("solimp", "0.95 0.99 0.001")
        geom.set("solref", "0.004 1")
        geom.set("priority", "1")


def _replace_hand_assets(target_root: ET.Element, source_root: ET.Element, side: str) -> None:
    assets = target_root.find("asset")
    source_assets = source_root.find("asset")
    if assets is None or source_assets is None:
        raise RuntimeError("Missing MJCF asset section")
    for mesh in list(assets.findall("mesh")):
        if (mesh.get("file") or "").startswith("wuji_hand1/"):
            assets.remove(mesh)
    for mesh in source_assets.findall("mesh"):
        vendored = copy.deepcopy(mesh)
        vendored.set("file", f"wuji_hand2/meshes/{side}/{mesh.get('file')}")
        assets.insert(0, vendored)


def _replace_hand_contacts(target_root: ET.Element, source_root: ET.Element, side: str) -> None:
    contacts = target_root.find("contact")
    if contacts is None:
        contacts = ET.SubElement(target_root, "contact")
    aliases = _body_aliases(side)
    old_prefix = f"{side}_"
    for exclude in list(contacts.findall("exclude")):
        if any((exclude.get(key) or "").startswith(old_prefix) for key in ("body1", "body2")):
            contacts.remove(exclude)
    source_contacts = source_root.find("contact")
    if source_contacts is None:
        return
    for exclude in source_contacts.findall("exclude"):
        vendored = copy.deepcopy(exclude)
        for key in ("body1", "body2"):
            name = vendored.get(key, "")
            if name in aliases:
                vendored.set(key, aliases[name])
        contacts.append(vendored)


def _replace_hand_actuators(target_root: ET.Element, source_root: ET.Element, side: str) -> None:
    actuators = target_root.find("actuator")
    source_actuators = source_root.find("actuator")
    if actuators is None or source_actuators is None:
        raise RuntimeError("Missing MJCF actuator section")
    for actuator in list(actuators):
        if (actuator.get("joint") or "").startswith(f"{side}_finger"):
            actuators.remove(actuator)
    for actuator in source_actuators:
        actuators.append(copy.deepcopy(actuator))


def _replace_hand_sensors(target_root: ET.Element, source_root: ET.Element, side: str) -> None:
    sensors = target_root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(target_root, "sensor")
    for sensor in list(sensors):
        if (sensor.get("name") or "").startswith(f"wuji1_{side}/"):
            sensors.remove(sensor)
    source_body = source_root.find("worldbody/body")
    if source_body is None:
        raise RuntimeError("Missing official hand body")
    for joint in source_body.iter("joint"):
        joint_name = joint.get("name")
        if joint_name:
            ET.SubElement(
                sensors,
                "jointpos",
                name=f"wuji2_{side}/{joint_name}_pos",
                joint=joint_name,
            )


def _build_side(xml_dir: Path, source_dir: Path, side: str) -> Path:
    target_path = xml_dir / f"rokae_wuji1_{side}.xml"
    source_path = source_dir / "hand2" / "hand2_beta1" / "body" / "mjcf" / f"{side}.xml"
    target_root = ET.parse(target_path).getroot()
    source_root = ET.parse(source_path).getroot()

    target_root.set("model", f"rokae_wuji2_{side}")
    for default in target_root.findall("./default/default"):
        if default.get("class") == f"wuji_{side}":
            default.set("class", f"wuji2_{side}")

    _replace_hand_assets(target_root, source_root, side)
    mount = _find_named_body(target_root, f"{side}_hand_mount")
    mount.set("childclass", f"wuji2_{side}")
    for child in list(mount.findall("body")):
        mount.remove(child)
    official_hand = source_root.find("worldbody/body")
    if official_hand is None:
        raise RuntimeError(f"Official {side} Hand 2 body missing")
    hand_root = copy.deepcopy(official_hand)
    _rename_hand_bodies(hand_root, side)
    _configure_collision_geoms(hand_root, side)
    # Keep the historical palm body at the mount frame because task-space controllers
    # read its pose.  The official wrist is nested below it and carries the model-frame
    # rotation needed to make Hand 2's fingers point along the HandsOn +Z convention.
    compatibility_palm = ET.Element("body", {"name": f"{side}_palm_link"})
    hand_root.set(
        "quat",
        "0 0.7071067812 -0.7071067812 0"
        if side == "left"
        else "0 0.7071067812 0.7071067812 0",
    )
    compatibility_palm.append(hand_root)
    mount.append(compatibility_palm)

    _replace_hand_contacts(target_root, source_root, side)
    _replace_hand_actuators(target_root, source_root, side)
    _replace_hand_sensors(target_root, source_root, side)

    output_path = xml_dir / f"rokae_wuji2_{side}.xml"
    tree = ET.ElementTree(target_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _build_bimanual(xml_dir: Path) -> Path:
    root = ET.parse(xml_dir / "rokae_wuji1_bimanual.xml").getroot()
    root.set("model", "rokae_wuji2_bimanual")
    for include in root.findall("include"):
        filename = include.get("file", "")
        include.set("file", filename.replace("wuji1", "wuji2"))
    output_path = xml_dir / "rokae_wuji2_bimanual.xml"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _build_scene_variants(xml_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for source in sorted(xml_dir.glob("*_rokae_wuji1.xml")):
        output = source.with_name(source.name.replace("_rokae_wuji1.xml", "_rokae_wuji2.xml"))
        transformed = (
            source.read_text().replace("wuji1", "wuji2").replace("Wuji1", "Wuji2")
        )
        output.write_text(
            "\n".join(line.rstrip() for line in transformed.splitlines()) + "\n"
        )
        outputs.append(output)
    return outputs


def _copy_sources(xml_dir: Path, source_dir: Path) -> None:
    package_dir = source_dir / "hand2" / "hand2_beta1" / "body"
    destination = xml_dir / "wuji_hand2"
    for side in ("left", "right"):
        source_xml = package_dir / "mjcf" / f"{side}.xml"
        root = ET.parse(source_xml).getroot()
        referenced = {mesh.get("file") for mesh in root.findall("asset/mesh")}
        mesh_destination = destination / "meshes" / side
        mesh_destination.mkdir(parents=True, exist_ok=True)
        for filename in sorted(name for name in referenced if name):
            shutil.copy2(package_dir / "meshes" / side / filename, mesh_destination / filename)
        shutil.copy2(source_xml, destination / f"hand2_beta1_{side}.xml")
    shutil.copy2(source_dir / "LICENSE", destination / "LICENSE.wuji-description")
    metadata = {
        "source": DESCRIPTION_URL,
        "tag": DESCRIPTION_TAG,
        "commit": DESCRIPTION_COMMIT,
        "model": "hand2/hand2_beta1/body",
        "license": "MIT",
    }
    (destination / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="checkout of wuji-technology/wuji-description")
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "handson/handson/sim/envs/xmls",
    )
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != DESCRIPTION_COMMIT:
        raise SystemExit(f"Expected wuji-description {DESCRIPTION_COMMIT}, got {head}")

    _copy_sources(args.xml_dir, args.source)
    outputs = [_build_side(args.xml_dir, args.source, side) for side in ("right", "left")]
    outputs.append(_build_bimanual(args.xml_dir))
    outputs.extend(_build_scene_variants(args.xml_dir))
    print(f"Generated {len(outputs)} Wuji Hand 2 XML files")


if __name__ == "__main__":
    main()
