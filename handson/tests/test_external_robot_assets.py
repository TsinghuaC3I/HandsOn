"""Optional user-supplied robot assets are imported safely and never repackaged."""
import hashlib
import json
from pathlib import Path

import pytest

from handson.robots import assets, external


@pytest.fixture
def import_case(tmp_path, monkeypatch):
    package = tmp_path / "package"
    folder = package / "assets/robots"
    folder.mkdir(parents=True)
    files = {"robots/left.stl": hashlib.sha256(b"left").hexdigest(),
             "robots/right.stl": hashlib.sha256(b"right").hexdigest()}
    manifest = {"robots": {"sample": dict(model="Sample", files=files,
                source_url="https://example.com/robots", download_status="User supplied")}}
    (folder / "external.json").write_text(json.dumps(manifest))
    source = tmp_path / "source"
    source.mkdir()
    (source / "renamed-left.stl").write_bytes(b"left")
    (source / "renamed-right.stl").write_bytes(b"right")
    monkeypatch.setattr(assets, "PACKAGE_ROOT", package)
    return package, source, manifest


def test_import_matches_content_and_is_idempotent(import_case):
    package, source, _ = import_case
    assert not external.external_status("sample")["available"]
    assert external.install_external_assets("sample", source)["available"]
    assert (package / "robots/left.stl").read_bytes() == b"left"
    assert external.install_external_assets("sample", source)["available"]
    external.require_external_assets("sample")


def test_partial_or_wrong_model_never_installs_partial_files(import_case):
    package, source, _ = import_case
    (source / "renamed-right.stl").write_bytes(b"different model")
    with pytest.raises(ValueError, match="Incomplete or different"):
        external.install_external_assets("sample", source)
    assert not (package / "robots").exists()


def test_conflicting_local_install_is_not_overwritten(import_case):
    package, source, _ = import_case
    (package / "robots").mkdir()
    path = package / "robots/right.stl"
    path.write_bytes(b"keep this")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        external.install_external_assets("sample", source)
    assert path.read_bytes() == b"keep this"
    assert not (package / "robots/left.stl").exists()
    with pytest.raises(FileNotFoundError, match="user-supplied"):
        external.require_external_assets("sample")


def test_manifest_paths_and_symlink_inputs_cannot_escape(import_case, tmp_path):
    package, source, manifest = import_case
    outside = tmp_path / "outside.stl"
    outside.write_bytes(b"left")
    (source / "renamed-left.stl").unlink()
    (source / "renamed-left.stl").symlink_to(outside)
    with pytest.raises(ValueError, match="Incomplete"):
        external.install_external_assets("sample", source)
    manifest["robots"]["sample"]["files"] = {"../outside.stl": hashlib.sha256(b"right").hexdigest()}
    (package / "assets/robots/external.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        external.install_external_assets("sample", source)
    assert outside.read_bytes() == b"left"


def test_missing_external_assets_do_not_disappear_from_catalog():
    catalog = assets._catalog()["robots"]
    for key in ("tianji_m6_ccs",):
        assert catalog[key]["distribution"] == "external"
        status = external.external_status(key)
        if not status["available"]:
            with pytest.raises(FileNotFoundError, match="--install-from"):
                assets.get_robot_asset(key)
            with pytest.raises(FileNotFoundError, match="--install-from"):
                assets.validate_robot_assets(key)
    report = assets.validate_robot_assets()
    assert report["complete"] == all(external.external_status(k)["available"] for k in external.external_catalog())


def test_every_external_file_is_excluded_from_both_build_targets():
    import tomllib
    import fnmatch
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    build = config["tool"]["hatch"]["build"]
    excluded = build["exclude"]
    artifact_excludes = [p[1:] for p in build["targets"]["wheel"]["artifacts"] if p.startswith("!")]
    for spec in external.external_catalog().values():
        for relative in spec["files"]:
            path = "handson/" + relative
            assert any(fnmatch.fnmatchcase(path, p) for p in excluded), path
            assert any(fnmatch.fnmatchcase(path, p) for p in artifact_excludes), path


def test_build_exclusions_preserve_bundled_files_inside_robot_directories():
    import re
    import tomllib
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    patterns = config["tool"]["hatch"]["build"]["exclude"]
    # These build patterns use * (one path component) and ** (recursive).
    patterns = [re.compile(re.escape(p).replace(r"\*\*", ".*").replace(r"\*", "[^/]*"))
                for p in patterns]
    for spec in assets._catalog()["robots"].values():
        record = json.loads(assets._package_path(spec["source_record"]).read_text())
        for relative in record["packaged_files"]:
            path = Path("handson") / relative
            # Git-style excludes can match a directory and exclude all children.
            for candidate in [path, *path.parents]:
                assert not any(p.fullmatch(candidate.as_posix()) for p in patterns), relative
