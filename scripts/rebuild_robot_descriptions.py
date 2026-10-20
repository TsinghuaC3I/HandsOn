#!/usr/bin/env python3
"""Rebuild bundled robot descriptions using only the packaged sources.

Install xacro==2.1.1 in a build environment, then run with --check to compare
without writing. Runtime asset loading does not depend on ROS or xacro.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
import xacro.substitution_args

PACKAGE = Path(__file__).resolve().parents[1] / "handson" / "handson"
ASSETS = PACKAGE / "assets" / "robots"
RECIPES = {
    "kuka_kr6_r700": "kuka_kr6_support/urdf/kr6r700sixx.xacro",
    "ur5e": "urdf/ur.urdf.xacro",
    "abb_irb120": "abb_irb120_support/urdf/irb120_3_58.xacro",
    "fanuc_lrmate200id": "fanuc_lrmate200id_support/urdf/lrmate200id.xacro",
    "sharpa_wave": None,
}


def build(robot: str, side: str) -> bytes:
    folder = ASSETS / robot
    source = folder / "source"
    packages = {p.name: p for p in source.iterdir() if p.is_dir()}
    if robot == "ur5e":
        packages["ur_description"] = source
    if robot == "sharpa_wave":
        packages = {f"{s}_sharpa_wave": source / "wave_01" / f"{s}_sharpa_wave"
                    for s in ("left", "right")}
    original_find = xacro.substitution_args._eval_find
    try:
        # Explicit package mapping: no ROS installation or ambient ROS search path.
        xacro.substitution_args._eval_find = lambda package: str(packages[package])
        if robot == "sharpa_wave":
            xml = (packages[f"{side}_sharpa_wave"] / f"{side}_sharpa_wave.urdf").read_text()
        else:
            mappings = {"ur_type": "ur5e", "name": "ur5e"} if robot == "ur5e" else {}
            xml = xacro.process_file(str(source / RECIPES[robot]), mappings=mappings).toxml()
    finally:
        xacro.substitution_args._eval_find = original_find
    tree = ET.ElementTree(ET.fromstring(xml))
    for node in tree.getroot().iter():
        if node.tag in {"mesh", "texture"} and "filename" in node.attrib:
            uri = node.attrib["filename"]
            if uri.startswith("package://"):
                package, inner = uri[10:].split("/", 1)
                path = packages[package] / inner
            elif uri.startswith("file://"):
                path = Path(uri[7:])
            elif Path(uri).is_absolute():
                path = Path(uri)
            else:
                raise ValueError(f"Unexpected source resource: {uri}")
            path = path.resolve()
            if not path.is_relative_to(source.resolve()) or not path.is_file():
                raise ValueError(f"Missing or external source resource: {uri}")
            node.set("filename", os.path.relpath(path, folder))
        if node.tag == "compiler":
            node.set("meshdir", ".")
            node.set("strippath", "false")
    ET.indent(tree, space="  ")
    stream = io.BytesIO()
    tree.write(stream, encoding="utf-8", xml_declaration=True)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare without writing")
    args = parser.parse_args()
    if version("xacro") != "2.1.1":
        parser.error("Reproducible expansion requires xacro==2.1.1")
    catalog = json.loads((ASSETS / "catalog.json").read_text())
    count = 0
    for robot in RECIPES:
        record_path = ASSETS / robot / "SOURCE.json"
        record = json.loads(record_path.read_text())
        for relative, expected in record["upstream_blob_ids"].items():
            data = (ASSETS / robot / "source" / relative).read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if actual != expected:
                raise ValueError(f"Modified pinned source: {robot}/{relative}")
        for side, entry in catalog["robots"][robot]["entries"].items():
            data = build(robot, side)
            target = PACKAGE / entry["path"]
            if args.check:
                if target.read_bytes() != data:
                    raise ValueError(f"Generated description differs: {robot}/{side}")
            else:
                target.write_bytes(data)
                record["packaged_files"][entry["path"]] = hashlib.sha256(data).hexdigest()
            count += 1
        if not args.check:
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"{'Verified' if args.check else 'Rebuilt'} {count} URDF descriptions from pinned sources")


if __name__ == "__main__":
    main()
