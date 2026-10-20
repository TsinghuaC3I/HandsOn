"""Import user-obtained robot files locally; never download or grant a license."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from . import assets


def external_catalog() -> dict:
    return json.loads((assets.PACKAGE_ROOT / "assets/robots/external.json").read_text())["robots"]


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"External asset path escapes destination: {relative}")
    return path


def acquisition_message(robot_id: str) -> str:
    spec = external_catalog()[robot_id]
    url = spec.get("download_url") or spec["source_url"]
    return (f"{spec['model']} requires user-supplied assets. {spec['download_status']} "
            f"Source: {url} . See docs/robot-assets.md#external-assets. "
            f"Import an authorized local copy with: python -m handson.robots "
            f"--robot {robot_id} --install-from /path/to/unpacked-assets")


def external_status(robot_id: str, *, root: Path | None = None, hashes: bool = True) -> dict:
    root = Path(root) if root is not None else assets.PACKAGE_ROOT
    spec = external_catalog()[robot_id]
    missing, changed = [], []
    for relative, expected in spec["files"].items():
        path = _path(root, relative)
        if not path.is_file():
            missing.append(relative)
        elif hashes and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    return dict(robot=robot_id, distribution="external", available=not (missing or changed),
                expected_files=len(spec["files"]), missing=missing, changed=changed,
                source_url=spec["source_url"], download_url=spec.get("download_url"),
                download_status=spec["download_status"])


def require_external_assets(robot_id: str) -> None:
    status = external_status(robot_id)
    if not status["available"]:
        raise FileNotFoundError(acquisition_message(robot_id) +
                                f" Missing: {len(status['missing'])}; changed: {len(status['changed'])}.")


def _expand_tianji(stage: Path) -> None:
    """Expand the pinned, user-provided source without ROS or a network call."""
    from importlib.metadata import version
    try:
        import xacro
        import xacro.substitution_args
        if version("xacro") != "2.1.1":
            raise ImportError("wrong xacro version")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError("Install xacro==2.1.1 to prepare the Tianji source archive") from exc
    folder = stage / "assets/robots/tianji_m6_ccs"
    source = folder / "source"
    original_find = xacro.substitution_args._eval_find
    try:
        xacro.substitution_args._eval_find = lambda package: str(source / package)
        for side in ("left", "right"):
            wrapper = folder / f"expand_{side}.xacro"
            shutil.copyfile(assets.PACKAGE_ROOT / "assets/robots/tianji_m6_ccs" / wrapper.name, wrapper)
            tree = ET.ElementTree(ET.fromstring(xacro.process_file(str(wrapper)).toxml()))
            for node in tree.getroot().iter():
                if node.tag in {"mesh", "texture"} and "filename" in node.attrib:
                    uri = node.get("filename")
                    if uri.startswith("package://"):
                        path = _path(source, uri[10:])
                    elif uri.startswith("file://"):
                        path = Path(uri[7:]).resolve()
                    else:
                        raise ValueError(f"Unexpected Tianji resource: {uri}")
                    if not path.is_relative_to(source.resolve()) or not path.is_file():
                        raise FileNotFoundError(path)
                    node.set("filename", os.path.relpath(path, folder))
            ET.indent(tree, space="  ")
            stream = io.BytesIO()
            tree.write(stream, encoding="utf-8", xml_declaration=True)
            (folder / f"model_{side}.urdf").write_bytes(stream.getvalue())
    finally:
        xacro.substitution_args._eval_find = original_find


def install_external_assets(robot_id: str, source: str | Path, *, destination: Path | None = None) -> dict:
    """Verify all inputs before copying. Accept an unpacked, authorized local source.

    Identical files may have different supplier names. Every destination is bound
    to a published SHA256; partial/wrong models and conflicting installs fail.
    Existing files are never overwritten. No archive extraction or network access.
    """
    spec = external_catalog()[robot_id]
    source = Path(source).expanduser().resolve()
    destination = Path(destination).resolve() if destination is not None else assets.PACKAGE_ROOT.resolve()
    if not source.is_dir():
        raise ValueError(f"Provide an unpacked asset directory: {source}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate directories")
    expected_hashes = set(spec["files"].values())
    found = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file() or ".git" in path.relative_to(source).parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in expected_hashes:
            found[digest] = path
    # Stage beside the destination so all installs remain on the same filesystem.
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robot-import-", dir=destination.parent) as work:
        stage = Path(work)
        generated = {f"assets/robots/tianji_m6_ccs/model_{s}.urdf" for s in ("left", "right")}
        missing = []
        for relative, digest in spec["files"].items():
            if digest not in found:
                if robot_id != "tianji_m6_ccs" or relative not in generated:
                    missing.append(relative)
                continue
            target = _path(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found[digest], target)
        if missing:
            raise ValueError(f"Incomplete or different {robot_id} source ({len(missing)} files): " +
                             ", ".join(missing[:5]) + ". " + acquisition_message(robot_id))
        if robot_id == "tianji_m6_ccs" and any(not (stage / p).is_file() for p in generated):
            _expand_tianji(stage)
        pending = []
        for relative, digest in spec["files"].items():
            staged = _path(stage, relative)
            if hashlib.sha256(staged.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Prepared robot checksum mismatch: {relative}")
            target = _path(destination, relative)
            if target.exists():
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Refusing to overwrite conflicting asset: {target}")
            else:
                pending.append((staged, target))
        created = []
        try:
            for staged, target in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also prevents a concurrent installer overwrite.
                with target.open("xb") as out:
                    created.append(target)
                    with staged.open("rb") as inp:
                        shutil.copyfileobj(inp, out)
        except BaseException:
            for target in reversed(created):
                target.unlink()
            raise
    return external_status(robot_id, root=destination)
