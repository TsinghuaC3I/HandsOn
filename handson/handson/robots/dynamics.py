"""Explicit estimated dynamics for standalone robot simulation."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET

from .assets import PACKAGE_ROOT, RobotAsset


def load_simulation_profile(asset: RobotAsset) -> dict:
    path = asset.source_record.parent / "simulation.json"
    if not path.is_file():
        raise ValueError(f"No standalone simulation profile for {asset.robot_id}")
    record = json.loads(asset.source_record.read_text())
    expected = record["packaged_files"].get(path.relative_to(PACKAGE_ROOT).as_posix())
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Simulation profile checksum mismatch")
    profile = json.loads(path.read_text())
    if hashlib.sha256(asset.path.read_bytes()).hexdigest() != profile["source_sha256"]:
        raise ValueError("Simulation profile does not match the source description")
    return profile


def apply_simulation_profile(asset: RobotAsset, tree: ET.ElementTree) -> None:
    profile = load_simulation_profile(asset)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    for name, spec in profile["inertial_overrides"].items():
        link = links[name]
        if link.find("inertial") is not None:
            raise ValueError(f"Refusing to replace existing upstream inertia: {name}")
        mass = spec["mass_kg"]
        inertia = spec["diagonal_inertia_kg_m2"]
        if not all(math.isfinite(v) and v > 0 for v in [mass, *inertia]):
            raise ValueError(f"Invalid estimated mass/inertia: {name}")
        if 2 * max(inertia) > sum(inertia) + 1e-12:
            raise ValueError(f"Nonphysical estimated inertia: {name}")
        node = ET.SubElement(link, "inertial")
        ET.SubElement(node, "origin", xyz=" ".join(map(str, spec["com_xyz_m"])), rpy="0 0 0")
        ET.SubElement(node, "mass", value=str(mass))
        ET.SubElement(node, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]),
                      ixy="0", ixz="0", iyz="0")
    for joint in root.findall("joint"):
        if joint.get("name") not in profile["joint_drives"]:
            continue
        spec = profile["joint_drives"][joint.get("name")]
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Missing joint limits: {joint.get('name')}")
        # Preserve specified manufacturer efforts; fill only zero placeholders.
        if float(limit.get("effort", "0")) == 0:
            limit.set("effort", str(spec["effort_limit_nm"]))
