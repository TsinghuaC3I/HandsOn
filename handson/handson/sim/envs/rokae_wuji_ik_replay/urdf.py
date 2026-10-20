from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .kinematics import JointSpec, SerialChainModel
from .spatial import pose_from_xyz_rpy


def load_serial_chain_model(
    urdf_path: str | Path,
    *,
    joint_pattern: str,
    joint_count: int,
    end_link: str,
) -> SerialChainModel:
    root = ET.parse(urdf_path).getroot()
    compiled = re.compile(joint_pattern)
    joints: list[JointSpec] = []
    fixed: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    for node in root.findall("joint"):
        name = node.attrib.get("name", "")
        jtype = node.attrib.get("type", "")
        parent = node.find("parent")
        child = node.find("child")
        origin = node.find("origin")
        xyz = _floats(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3)
        rpy = _floats(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3)
        if parent is None or child is None:
            continue
        if jtype == "fixed":
            fixed.append((parent.attrib["link"], child.attrib["link"], xyz, rpy))
            continue
        if jtype != "revolute" or not compiled.fullmatch(name):
            continue
        axis_node = node.find("axis")
        limit = node.find("limit")
        if axis_node is None or limit is None:
            raise ValueError(f"revolute joint {name!r} missing axis or limit")
        axis = _floats(axis_node.attrib["xyz"])
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError(f"joint {name!r} has zero axis")
        joints.append(
            JointSpec(
                name=name,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=xyz,
                rpy=rpy,
                axis=axis / norm,
                lower=float(limit.attrib["lower"]),
                upper=float(limit.attrib["upper"]),
            )
        )

    joints.sort(key=lambda j: (int(re.search(r"(\d+)$", j.name).group(1)) if re.search(r"(\d+)$", j.name) else 999, j.name))
    if len(joints) != joint_count:
        raise ValueError(f"{urdf_path}: expected {joint_count} joints, got {len(joints)}")
    return SerialChainModel(joints, end_link=end_link, end_transform=_fixed_transform(fixed, joints[-1].child, end_link))


def _fixed_transform(fixed, start_link: str, end_link: str) -> np.ndarray:
    if start_link == end_link:
        return np.eye(4)
    by_parent = {parent: (child, xyz, rpy) for parent, child, xyz, rpy in fixed}
    transform = np.eye(4)
    current = start_link
    for _ in range(len(fixed) + 1):
        if current == end_link:
            return transform
        step = by_parent.get(current)
        if step is None:
            break
        child, xyz, rpy = step
        transform = transform @ pose_from_xyz_rpy(xyz, rpy)
        current = child
    raise ValueError(f"no fixed chain from {start_link!r} to {end_link!r}")


def _floats(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=float)
