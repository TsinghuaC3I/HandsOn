"""Regenerate declared simulation estimates; build dependency: trimesh==4.8.3.

These estimates are not identified manufacturer parameters. Existing inertias
are retained. See docs/robot-assets.md for assumptions and validation scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

PACKAGE = Path(__file__).resolve().parents[1] / "handson" / "handson"
ASSETS = PACKAGE / "assets" / "robots"
MASS_KG = {
    "kuka_kr6_r700": [14, 10, 12, 7, 4, 2, 1],
    "fanuc_lrmate200id": [7, 5, 5, 4, 2, 1.5, .5],
}
EFFORT_NM = {
    "kuka_kr6_r700": [180, 180, 100, 40, 40, 25],
    "abb_irb120": [100, 100, 60, 20, 20, 12],
    "fanuc_lrmate200id": [100, 100, 60, 20, 20, 12],
}
STIFFNESS = [2000, 2000, 1500, 600, 400, 300]
DAMPING = [100, 100, 80, 40, 30, 20]


def estimate_inertial(link: ET.Element, folder: Path, mass: float) -> dict:
    if link.find("inertial") is not None:
        raise ValueError(f"Upstream inertia already exists: {link.get('name')}")
    points = []
    for collision in link.findall("collision"):
        node = collision.find("geometry/mesh")
        mesh = trimesh.load_mesh(folder / node.get("filename"))
        vertices = np.asarray(mesh.vertices) * np.fromstring(node.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        transform = np.eye(4)
        if origin is not None:
            transform = trimesh.transformations.euler_matrix(*np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        points.append(trimesh.transform_points(vertices, transform))
    vertices = np.concatenate(points)
    lower, upper = vertices.min(0), vertices.max(0)
    dimensions = upper - lower
    if not np.all(np.isfinite(dimensions) & (dimensions > 0)):
        raise ValueError(f"Invalid collision dimensions: {link.get('name')}")
    inertia = mass / 12 * (np.sum(dimensions**2) - dimensions**2)
    return {"mass_kg": mass, "com_xyz_m": ((lower + upper) / 2).tolist(),
            "box_dimensions_m": dimensions.tolist(), "diagonal_inertia_kg_m2": inertia.tolist(),
            "provenance": "HandsOn simulation estimate"}


def main() -> None:
    for robot_id, efforts in EFFORT_NM.items():
        folder = ASSETS / robot_id
        source = folder / "model_right.urdf"
        root = ET.parse(source).getroot()
        profile = {
            "schema": 1, "robot_id": robot_id,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "status": "estimated simulation profile; not manufacturer-identified dynamics",
            "drive_type": "force", "gravity_m_s2": 9.81, "fixed_base": True,
            "physics_dt": 1/240, "control_dt": 1/60,
            "inertial_overrides": {}, "joint_drives": {},
            "estimation_method": "For links lacking inertials: authored per-link mass allocation, collision-mesh axis-aligned bounds in link coordinates, center at box midpoint, Ixx=m*(dy^2+dz^2)/12 and cyclic permutations. Existing upstream inertia is retained. Zero effort limits receive explicit simulation estimates.",
        }
        if robot_id in MASS_KG:
            links = {link.get("name"): link for link in root.findall("link")}
            names = ["base_link", *[f"link_{i}" for i in range(1, 7)]]
            for name, mass in zip(names, MASS_KG[robot_id], strict=True):
                profile["inertial_overrides"][name] = estimate_inertial(links[name], folder, mass)
        joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
        for joint, effort, stiffness, damping in zip(joints, efforts, STIFFNESS, DAMPING, strict=True):
            if float(joint.find("limit").get("effort")) != 0:
                raise ValueError(f"Upstream effort already specified: {joint.get('name')}")
            profile["joint_drives"][joint.get("name")] = {
                "effort_limit_nm": effort, "stiffness_nm_rad": stiffness,
                "damping_nm_s_rad": damping, "home_rad": 0.0, "test_amplitude_rad": .12,
                "provenance": "HandsOn standalone-simulation tuning; upstream effort was zero",
            }
        target = folder / "simulation.json"
        target.write_text(json.dumps(profile, indent=2) + "\n")
        record_path = folder / "SOURCE.json"
        record = json.loads(record_path.read_text())
        record["packaged_files"][target.relative_to(PACKAGE).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{robot_id}: {len(profile['inertial_overrides'])} estimated inertias, {len(joints)} drives")


if __name__ == "__main__":
    main()
