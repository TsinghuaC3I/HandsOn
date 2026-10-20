"""Portable robot descriptions, provenance and safe importer preparation."""
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from handson.robots import get_robot_asset, list_robot_assets, validate_robot_assets
from handson.robots import assets


def test_all_packaged_robots_have_complete_hashed_resources():
    report = validate_robot_assets()
    assert report["verified"]
    assert len(report["robots"]) == 11
    assert {r["robot"] for r in report["robots"] if r["benchmark_qualified"]} == {"rokae_ar5", "wuji1"}
    joints = {r["robot"]: r["entries"].get("right", {}).get("independent_joints") for r in report["robots"]}
    assert {k: joints[k] for k in ("kuka_kr6_r700", "ur5e", "abb_irb120", "fanuc_lrmate200id", "sharpa_wave")} == {
        "kuka_kr6_r700": 6, "ur5e": 6,
        "abb_irb120": 6, "fanuc_lrmate200id": 6, "sharpa_wave": 22,
    }
    tianji = json.loads((assets.PACKAGE_ROOT / "assets/robots/tianji_m6_ccs/SOURCE.json").read_text())
    assert tianji["revision"] == "b9e73d4b50a071457fe08e5c710f0bcda7731b29"
    assert "KR6 R700" in get_robot_asset("kuka_kr6_r700").model


def test_selection_never_substitutes_a_different_robot_or_side():
    with pytest.raises(ValueError, match="Unknown robot"):
        get_robot_asset("iiwa14")
    with pytest.raises(ValueError, match="has sides"):
        get_robot_asset("ur5e", side="left")
    with pytest.raises(ValueError, match="not URDF"):
        get_robot_asset("wuji1").prepare_urdf("unused.urdf")


@pytest.mark.parametrize("robot", [r for r,s in assets._catalog()["robots"].items()
                                  if s["entries"]["right"]["format"] == "urdf" and s.get("distribution") != "external"])
def test_preparation_preserves_physics_and_resolves_resources(robot, tmp_path):
    asset = get_robot_asset(robot)
    original = ET.parse(asset.path)
    output = asset.prepare_urdf(tmp_path / f"{robot}.urdf")
    prepared = ET.parse(output)
    for before, after in zip(original.getroot().iter(), prepared.getroot().iter(), strict=True):
        assert before.tag == after.tag
        assert before.text == after.text
        if before.tag in {"mesh", "texture"} and "filename" in before.attrib:
            assert Path(after.get("filename")).is_absolute()
            assert Path(after.get("filename")).is_file()
            assert Path(after.get("filename")) == (asset.path.parent / before.get("filename")).resolve()
            after.set("filename", before.get("filename"))
        assert before.attrib == after.attrib
    with pytest.raises(FileExistsError):
        asset.prepare_urdf(output)
    with pytest.raises(ValueError, match="outside the installed package"):
        asset.prepare_urdf(asset.path)


@pytest.fixture
def sample_package(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    mesh = root / "mesh.stl"
    mesh.write_bytes(b"test mesh")
    urdf = root / "model.urdf"
    urdf.write_text('<robot name="sample"><link name="base"><visual><geometry><mesh filename="mesh.stl"/></geometry></visual></link></robot>')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (mesh, urdf)}
    record = root / "SOURCE.json"
    record.write_text(json.dumps({"license": "test", "packaged_files": files}))
    catalog = root / "catalog.json"
    catalog.write_text(json.dumps({"robots": {"sample": {"source_record": "SOURCE.json", "benchmark_qualified": False,
        "entries": {"right": {"format": "urdf", "path": "model.urdf"}}}}}))
    monkeypatch.setattr(assets, "PACKAGE_ROOT", root)
    monkeypatch.setattr(assets, "CATALOG_PATH", catalog)
    return root


def test_checksum_missing_file_and_unhashed_dependency_are_rejected(sample_package):
    validate_robot_assets("sample")
    mesh = sample_package / "mesh.stl"
    mesh.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_robot_assets("sample")
    mesh.unlink()
    with pytest.raises(FileNotFoundError):
        validate_robot_assets("sample")
    mesh.write_bytes(b"test mesh")
    path = sample_package / "SOURCE.json"
    record = json.loads(path.read_text())
    del record["packaged_files"]["mesh.stl"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Unhashed robot resource"):
        validate_robot_assets("sample")


@pytest.mark.parametrize("uri", ["../outside.stl", "package://unknown/mesh.stl", "https://example.com/mesh.stl", "/missing.stl"])
def test_external_resources_are_rejected(sample_package, uri):
    with pytest.raises((ValueError, FileNotFoundError)):
        assets._resource(sample_package / "model.urdf", uri)


def test_mesh_embedded_external_references_are_checked(sample_package):
    dae = sample_package / "mesh.dae"
    dae.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_images><image><init_from>missing.png</init_from></image></library_images></COLLADA>')
    with pytest.raises(FileNotFoundError):
        assets._mesh_dependencies(dae)
    payload = json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "missing.bin"}]}).encode()
    glb = sample_package / "mesh.glb"
    glb.write_bytes(struct.pack("<4sIIII", b"glTF", 2, 20 + len(payload), len(payload), 0x4E4F534A) + payload)
    with pytest.raises(FileNotFoundError):
        assets._mesh_dependencies(glb)


@pytest.mark.parametrize("robot,side", [(r, s) for r, sides in [("wuji1", ("left", "right")), ("wuji2", ("left", "right")), ("allegro", ("left", "right")), ("panda", ("right",))] for s in sides])
def test_mjcf_catalog_entries_load_without_path_patches(robot, side):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(get_robot_asset(robot, side=side).path))
    assert model.nv == {"wuji1": 20, "wuji2": 20, "allegro": 16, "panda": 9}[robot]


@pytest.mark.parametrize("robot", ["kuka_kr6_r700", "abb_irb120", "fanuc_lrmate200id"])
def test_estimated_dynamics_are_explicit_and_preserve_supplied_parameters(robot, tmp_path):
    import numpy as np
    from handson.robots.dynamics import load_simulation_profile
    asset = get_robot_asset(robot)
    profile = load_simulation_profile(asset)
    assert "estimate" in profile["status"]
    original = ET.parse(asset.path).getroot()
    prepared = ET.parse(asset.prepare_urdf(tmp_path / "simulation.urdf", simulation=True)).getroot()
    for before, after in zip(original.findall("link"), prepared.findall("link"), strict=True):
        inertial = after.find("inertial")
        if before.find("inertial") is not None:
            assert ET.tostring(before.find("inertial")) == ET.tostring(inertial)
        if before.find("collision") is not None:
            assert float(inertial.find("mass").get("value")) > 0
            values = inertial.find("inertia").attrib
            tensor = np.array([[float(values[k]) for k in row] for row in [("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz")]])
            eig = np.linalg.eigvalsh(tensor)
            assert np.all(eig > 0)
            assert 2*max(eig) <= sum(eig) + 1e-8
    for before, after in zip(original.findall("joint"), prepared.findall("joint"), strict=True):
        if before.get("type") != "fixed":
            expected = dict(before.find("limit").attrib)
            expected["effort"] = str(profile["joint_drives"][before.get("name")]["effort_limit_nm"])
            assert after.find("limit").attrib == expected
    # The source description retains the original zero effort placeholders.
    assert all(float(j.find("limit").get("effort")) == 0 for j in original.findall("joint") if j.get("type") != "fixed")
