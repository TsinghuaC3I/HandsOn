"""Resolve and validate robot descriptions without importing a simulator."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from urllib.parse import unquote
import xml.etree.ElementTree as ET

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "assets" / "robots" / "catalog.json"


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def _package_path(relative: str) -> Path:
    path = (PACKAGE_ROOT / relative).resolve()
    if not path.is_relative_to(PACKAGE_ROOT.resolve()):
        raise ValueError(f"Robot asset escapes package: {relative}")
    return path


def _resource(description: Path, uri: str) -> Path:
    if "://" in uri or Path(uri).is_absolute():
        raise ValueError(f"Unresolved external resource: {uri}")
    path = (description.parent / unquote(uri)).resolve()
    if not path.is_relative_to(PACKAGE_ROOT.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing or external robot resource: {uri}")
    return path


@dataclass(frozen=True)
class RobotAsset:
    robot_id: str
    side: str
    kind: str
    model: str
    format: str
    path: Path
    source_record: Path
    benchmark_qualified: bool

    def prepare_urdf(self, output: str | Path, *, simulation: bool = False) -> Path:
        """Write an importer-ready URDF; only resource locations change.

        By default joint limits, inertias, origins and collisions are preserved.
        simulation=True explicitly applies the recorded estimated dynamics profile.
        This does not supply an arm controller or hand retargeting.
        """
        if self.format != "urdf":
            raise ValueError(f"{self.robot_id} provides {self.format}, not URDF")
        tree = ET.parse(self.path)
        if simulation:
            from .dynamics import apply_simulation_profile
            apply_simulation_profile(self, tree)
        for node in tree.getroot().iter():
            if node.tag in {"mesh", "texture"} and "filename" in node.attrib:
                node.set("filename", str(_resource(self.path, node.attrib["filename"])))
        output = Path(output).resolve()
        if output.is_relative_to(PACKAGE_ROOT.resolve()):
            raise ValueError("Write prepared URDFs outside the installed package")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            tree.write(stream, encoding="utf-8", xml_declaration=True)
        return output


def list_robot_assets() -> tuple[str, ...]:
    return tuple(sorted(_catalog()["robots"]))


def get_robot_asset(robot_id: str, *, side: str = "right") -> RobotAsset:
    """Resolve an explicit model/side; never silently substitute an asset."""
    robots = _catalog()["robots"]
    if robot_id not in robots:
        raise ValueError(f"Unknown robot {robot_id!r}; choose from {', '.join(sorted(robots))}")
    robot = robots[robot_id]
    if side not in robot["entries"]:
        raise ValueError(f"{robot_id} has sides {sorted(robot['entries'])}, not {side!r}")
    entry = robot["entries"][side]
    path = _package_path(entry["path"])
    if not path.is_file():
        if robot.get("distribution") == "external":
            from .external import acquisition_message
            raise FileNotFoundError(acquisition_message(robot_id))
        raise FileNotFoundError(path)
    return RobotAsset(robot_id, side, robot["kind"], robot["model"], entry["format"],
                      path, _package_path(robot["source_record"]), robot["benchmark_qualified"])


def _mesh_dependencies(path: Path) -> set[Path]:
    """Follow texture/buffer files referenced inside COLLADA and glTF meshes."""
    uris = []
    if path.suffix.lower() == ".dae":
        root = ET.parse(path).getroot()
        # Other init_from nodes name internal samplers, not files.
        for image in root.findall(".//{*}library_images/{*}image"):
            for node in image.iter():
                if node.tag.split("}")[-1] in {"init_from", "ref"} and node.text:
                    if not list(node):
                        uris.append(node.text.strip())
    elif path.suffix.lower() in {".glb", ".gltf"}:
        raw = path.read_bytes()
        if path.suffix.lower() == ".glb":
            if len(raw) < 20:
                raise ValueError(f"Truncated GLB: {path}")
            magic, version, size, json_size, chunk_type = struct.unpack_from("<4sIIII", raw)
            if magic != b"glTF" or version != 2 or size != len(raw) or chunk_type != 0x4E4F534A:
                raise ValueError(f"Invalid GLB header: {path}")
            raw = raw[20:20 + json_size]
        model = json.loads(raw)
        uris = [item["uri"] for key in ("buffers", "images")
                for item in model.get(key, []) if "uri" in item]
    return {_resource(path, uri) for uri in uris if not uri.startswith("data:")}


def _check_urdf(path: Path, hashed_paths: set[Path]) -> dict:
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Not a URDF robot: {path}")
    links = [n.attrib["name"] for n in root.findall("link")]
    joints = root.findall("joint")
    names = [n.attrib["name"] for n in joints]
    if len(set(links)) != len(links) or len(set(names)) != len(names):
        raise ValueError(f"Duplicate link or joint names: {path}")
    parents = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent not in links or child not in links or child in parents:
            raise ValueError(f"Invalid joint connection: {joint.attrib['name']}")
        parents[child] = parent
        mimic = joint.find("mimic")
        if mimic is not None and mimic.attrib["joint"] not in names:
            raise ValueError(f"Unknown mimic joint: {mimic.attrib['joint']}")
    roots = set(links) - set(parents)
    if len(roots) != 1:
        raise ValueError(f"Expected one robot root in {path}, got {sorted(roots)}")
    for link in links:
        visited = set()
        while link in parents:
            if link in visited:
                raise ValueError(f"Cyclic robot graph: {path}")
            visited.add(link)
            link = parents[link]
    resources = {_resource(path, node.attrib["filename"])
                 for node in root.iter() if node.tag in {"mesh", "texture"} and "filename" in node.attrib}
    dependencies = {dependency for resource in resources for dependency in _mesh_dependencies(resource)}
    if not (resources | dependencies) <= hashed_paths:
        raise ValueError(f"Unhashed robot resource in {path}")
    return {"links": len(links), "movable_joints": sum(j.get("type") != "fixed" for j in joints),
            "independent_joints": sum(j.get("type") != "fixed" and j.find("mimic") is None for j in joints),
            "mesh_resources": len(resources), "mesh_file_dependencies": len(dependencies),
            "links_with_inertia": sum(link.find("inertial") is not None for link in root.findall("link"))}


def validate_robot_assets(robot_id: str | None = None) -> dict:
    """Check hashes, resource closure and URDF kinematics, CPU only."""
    catalog = _catalog()
    selected = [robot_id] if robot_id is not None else list_robot_assets()
    reports = []
    for key in selected:
        if key not in catalog["robots"]:
            raise ValueError(f"Unknown robot {key!r}")
        spec = catalog["robots"][key]
        record = json.loads(_package_path(spec["source_record"]).read_text())
        files = dict(record["packaged_files"])
        if not files:
            raise ValueError(f"Empty robot asset manifest: {key}")
        for relative, expected in files.items():
            path = _package_path(relative)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Robot asset checksum mismatch: {relative}")
        availability = "bundled"
        if spec.get("distribution") == "external":
            from .external import external_status, require_external_assets
            status = external_status(key)
            if status["changed"]:
                raise ValueError(f"External robot asset checksum mismatch: {status['changed'][0]}")
            if not status["available"]:
                if robot_id is not None:
                    require_external_assets(key)
                reports.append({"robot": key, "verified_files": len(files), "entries": {},
                                "availability": "user-supplied-missing", "external": status,
                                "benchmark_qualified": spec["benchmark_qualified"], "license": record["license"]})
                continue
            files.update(record["external_files"])
            availability = "user-supplied-verified"
        entries = {}
        for side, entry in spec["entries"].items():
            if entry["path"] not in files:
                raise ValueError(f"Unhashed robot description: {entry['path']}")
            path = _package_path(entry["path"])
            hashed_paths = {_package_path(p) for p in files}
            if entry["format"] == "urdf":
                entries[side] = _check_urdf(path, hashed_paths)
            elif entry["format"] == "mjcf":
                root = ET.parse(path).getroot()
                compiler = root.find("compiler")
                resources = set()
                for node in root.findall("./asset/*"):
                    if "file" not in node.attrib:
                        continue
                    directory = ""
                    if compiler is not None:
                        directory = compiler.get("meshdir" if node.tag == "mesh" else "texturedir", "")
                    resources.add(_resource(path, str(Path(directory) / node.get("file"))))
                if root.findall("include"):
                    raise ValueError(f"Catalog MJCF must be standalone: {path}")
                if not resources <= hashed_paths:
                    raise ValueError(f"Unhashed MJCF resource: {path}")
                entries[side] = {"format": "mjcf", "mesh_resources": len(resources)}
            else:
                raise ValueError(f"Unsupported robot description format: {entry['format']}")
        reports.append({"robot": key, "verified_files": len(files), "entries": entries,
                        "availability": availability,
                        "benchmark_qualified": spec["benchmark_qualified"], "license": record["license"]})
    return {"verified": True, "complete": all(r["availability"] != "user-supplied-missing" for r in reports),
            "scope": "Bundled asset integrity; absent user-supplied models are listed separately. No physics, controller or task rollout.",
            "robots": reports}


def require_benchmark_assets() -> None:
    """Verify the bundled paper robot and rendering resources before Kit starts."""
    validate_robot_assets("rokae_ar5")
    validate_robot_assets("wuji1")
    from handson.sim.isaaclab.render_assets import validate_assets
    validate_assets()
