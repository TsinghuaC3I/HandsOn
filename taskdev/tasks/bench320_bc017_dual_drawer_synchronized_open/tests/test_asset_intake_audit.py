from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "asset_intake_audit.py"
SPEC = importlib.util.spec_from_file_location("bc017_asset_intake_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_zip(path: Path, members: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _write_stored_zip(path: Path, members: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _corrupt_first_member_payload(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.infolist()[0]
    payload = bytearray(path.read_bytes())
    local = info.header_offset
    assert payload[local : local + 4] == b"PK\x03\x04"
    name_length = int.from_bytes(payload[local + 26 : local + 28], "little")
    extra_length = int.from_bytes(payload[local + 28 : local + 30], "little")
    data_offset = local + 30 + name_length + extra_length
    payload[data_offset] ^= 0xFF
    path.write_bytes(payload)


def _patch_first_member_declared_size(path: Path, declared_size: int) -> None:
    assert declared_size < 2**32
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.infolist()[0]
    payload = bytearray(path.read_bytes())
    encoded = declared_size.to_bytes(4, "little")
    local = info.header_offset
    assert payload[local : local + 4] == b"PK\x03\x04"
    payload[local + 22 : local + 26] = encoded
    central = payload.find(b"PK\x01\x02")
    assert central >= 0
    payload[central + 24 : central + 28] = encoded
    path.write_bytes(payload)


def _gltf(
    *,
    buffer_uri: str | None = None,
    buffer_byte_length: int = 4,
    image_uri: str | None = None,
    nodes: list[dict[str, object]] | None = None,
    meshes: list[dict[str, object]] | None = None,
) -> str:
    document: dict[str, object] = {"asset": {"version": "2.0"}}
    if buffer_uri is not None:
        document["buffers"] = [
            {"byteLength": buffer_byte_length, "uri": buffer_uri}
        ]
    if image_uri is not None:
        document["images"] = [{"uri": image_uri}]
    if nodes is not None:
        document["nodes"] = nodes
    if meshes is not None:
        document["meshes"] = meshes
    return json.dumps(document)


def test_normal_archive_is_read_only_and_reports_assets_and_names(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "desk.zip",
        {
            "models/desk.gltf": _gltf(
                buffer_uri="desk.bin",
                image_uri="../textures/wood.png",
                nodes=[
                    {"name": "Left_Drawer_Handle"},
                    {"name": "DeskBody"},
                ],
                meshes=[{"name": "RightDrawerMesh"}, {"name": "Legs"}],
            ),
            "models/desk.bin": b"\x00\x01\x02\x03",
            "textures/wood.png": b"not-a-real-image",
            "materials/desk.mtl": "newmtl wood\n",
        },
    )
    original_bytes = archive_path.read_bytes()

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_OK
    assert report["status"] == "ok"
    assert report["security"]["is_safe"] is True
    assert report["assets"]["formats"] == ["gltf"]
    assert report["assets"]["textures"] == ["textures/wood.png"]
    assert report["assets"]["materials"] == ["materials/desk.mtl"]
    assert report["gltf"]["external_uri_closure"] is True
    assert [(hit["kind"], hit["name"], hit["keywords"]) for hit in report["name_hits"]] == [
        ("mesh", "RightDrawerMesh", ["drawer"]),
        ("node", "Left_Drawer_Handle", ["drawer", "handle"]),
        ("node", "DeskBody", ["body"]),
    ]
    assert archive_path.read_bytes() == original_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == ["desk.zip"]


@pytest.mark.parametrize(
    ("member", "security_key"),
    [
        ("../escape/model.fbx", "path_traversal"),
        ("/absolute/model.fbx", "absolute_paths"),
        ("C:\\absolute\\model.fbx", "absolute_paths"),
    ],
)
def test_unsafe_member_paths_are_rejected(
    tmp_path: Path, member: str, security_key: str
) -> None:
    report = AUDIT.audit_archive(_write_zip(tmp_path / "unsafe.zip", {member: b"fbx"}))

    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["security"]["is_safe"] is False
    assert report["security"][security_key] == [member]
    assert "unsafe_archive_members" in report["rejection_reasons"]


def test_zip_symlink_is_reported_and_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("models/linked.fbx")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "../../outside.fbx")

    report = AUDIT.audit_archive(archive_path)

    assert report["security"]["symlinks"] == ["models/linked.fbx"]
    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert "empty_archive" in report["rejection_reasons"]
    assert "no_supported_model" in report["rejection_reasons"]


def test_empty_zip_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w"):
        pass

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["rejection_reasons"] == ["empty_archive", "no_supported_model"]


def test_archive_without_supported_model_is_rejected(tmp_path: Path) -> None:
    report = AUDIT.audit_archive(
        _write_zip(tmp_path / "texture-only.zip", {"wood.png": b"pixels"})
    )

    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["rejection_reasons"] == ["no_supported_model"]


def test_missing_gltf_buffer_and_texture_are_rejected(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "missing.zip",
        {
            "model/desk.gltf": _gltf(
                buffer_uri="desk.bin", image_uri="textures/wood%20oak.png"
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["gltf"]["external_uri_closure"] is False
    assert [
        (item["kind"], item["uri"], item["resolved_path"], item["status"])
        for item in report["gltf"]["missing_external_references"]
    ] == [
        ("buffer", "desk.bin", "model/desk.bin", "missing"),
        (
            "image",
            "textures/wood%20oak.png",
            "model/textures/wood oak.png",
            "missing",
        ),
    ]
    assert "gltf_external_dependency_missing" in report["rejection_reasons"]


def test_complete_gltf_accepts_relative_and_embedded_dependencies(tmp_path: Path) -> None:
    document = json.loads(_gltf(buffer_uri="../binary/desk.bin"))
    document["images"] = [
        {"uri": "../textures/wood%20oak.png"},
        {"uri": "data:image/png;base64,AA=="},
    ]
    archive_path = _write_zip(
        tmp_path / "complete.zip",
        {
            "scene/desk.gltf": json.dumps(document),
            "binary/desk.bin": b"1234",
            "textures/wood oak.png": b"pixels",
        },
    )

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_OK
    gltf = report["gltf"]["documents"][0]
    assert gltf["external_uri_closure"] is True
    assert gltf["embedded_data_uri_count"] == 1
    assert [item["status"] for item in gltf["external_references"]] == [
        "present",
        "present",
    ]


def test_json_gltf_buffer_without_uri_is_rejected(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "missing-uri.zip",
        {
            "scene.gltf": json.dumps(
                {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]}
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["gltf"]["missing_external_references"] == [
        {
            "document": "scene.gltf",
            "index": 0,
            "kind": "buffer",
            "resolved_path": None,
            "status": "missing_uri",
            "uri": None,
        }
    ]


def test_detects_all_supported_model_formats_case_insensitively(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "formats.zip",
        {
            "a/model.BLEND": b"blend",
            "b/model.FBX": b"fbx",
            "c/model.GLB": _minimal_glb(),
            "d/model.GLTF": _gltf(),
            "e/model.USD": b"usd",
            "f/model.USDA": b"usda",
            "g/model.USDC": b"usdc",
        },
    )

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_OK
    assert report["assets"]["formats"] == [
        "blend",
        "fbx",
        "glb",
        "gltf",
        "usd",
        "usda",
        "usdc",
    ]


def _glb(
    document: dict[str, object],
    *,
    bin_chunks: tuple[bytes, ...] = (),
    raw_suffix: bytes = b"",
) -> bytes:
    json_payload = json.dumps(document, separators=(",", ":")).encode()
    json_payload += b" " * ((-len(json_payload)) % 4)
    body = (
        len(json_payload).to_bytes(4, "little")
        + AUDIT.GLB_JSON_CHUNK.to_bytes(4, "little")
        + json_payload
    )
    for payload in bin_chunks:
        assert len(payload) % 4 == 0
        body += (
            len(payload).to_bytes(4, "little")
            + AUDIT.GLB_BIN_CHUNK.to_bytes(4, "little")
            + payload
        )
    body += raw_suffix
    total_length = 12 + len(body)
    return (
        b"glTF"
        + (2).to_bytes(4, "little")
        + total_length.to_bytes(4, "little")
        + body
    )


def _minimal_glb() -> bytes:
    return _glb({"asset": {"version": "2.0"}})


def test_hash_and_json_are_stable(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "stable.zip", {"scene.gltf": _gltf(nodes=[{"name": "Door"}])}
    )

    first = AUDIT.audit_archive(archive_path)
    second = AUDIT.audit_archive(archive_path)

    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert AUDIT.stable_json(first) == AUDIT.stable_json(second)
    assert len(first["archive"]["sha256"]) == 64


def test_duplicate_and_portable_path_collisions_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Models/Desk.FBX", b"first")
        archive.writestr("models/desk.fbx", b"second")

    report = AUDIT.audit_archive(archive_path)

    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["security"]["portable_path_collisions"] == [
        {
            "members": ["Models/Desk.FBX", "models/desk.fbx"],
            "portable_path": "models/desk.fbx",
        }
    ]
    assert "unsafe_archive_members" in report["rejection_reasons"]


def test_stable_json_escapes_isolated_surrogate(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "surrogate.zip",
        {"scene.gltf": '{"asset":{"version":"2.0"},"nodes":[{"name":"Door\\ud800"}]}'},
    )

    rendered = AUDIT.stable_json(AUDIT.audit_archive(archive_path))

    assert "Door\\ud800" in rendered
    assert json.loads(rendered)["name_hits"][0]["name"].startswith("Door")


def test_cli_emits_json_and_uses_nonzero_exit_for_rejection(tmp_path: Path) -> None:
    archive_path = _write_zip(tmp_path / "no-model.zip", {"readme.txt": "hello"})

    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(archive_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == AUDIT.EXIT_REJECTED
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["rejection_reasons"] == ["no_supported_model"]


def test_cli_uses_zero_for_success_and_two_for_missing_input(tmp_path: Path) -> None:
    valid_path = _write_zip(tmp_path / "valid.zip", {"scene.gltf": _gltf()})
    success = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(valid_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    missing = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(tmp_path / "absent.zip")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert success.returncode == AUDIT.EXIT_OK
    assert json.loads(success.stdout)["status"] == "ok"
    assert missing.returncode == AUDIT.EXIT_INPUT_ERROR
    assert json.loads(missing.stdout)["status"] == "input_error"


def test_crc_damaged_member_is_a_structured_audit_rejection(tmp_path: Path) -> None:
    archive_path = _write_stored_zip(
        tmp_path / "bad-crc.zip", {"model.fbx": b"valid-looking-fbx"}
    )
    _corrupt_first_member_payload(archive_path)

    report = AUDIT.audit_archive(archive_path)

    assert report["status"] == "rejected"
    assert report["exit_code"] == AUDIT.EXIT_REJECTED
    assert report["payload_validation"]["member_read_errors"] == [
        {"archive_index": 0, "error": "crc_mismatch", "path": "model.fbx"}
    ]
    assert report["members"][0]["payload_status"] == "read_error"
    assert "archive_member_read_failed" in report["rejection_reasons"]


def test_zip_bomb_member_size_metadata_is_rejected_without_reading(
    tmp_path: Path,
) -> None:
    archive_path = _write_stored_zip(
        tmp_path / "metadata-bomb.zip", {"model.fbx": b"x"}
    )
    _patch_first_member_declared_size(
        archive_path, AUDIT.MAX_MEMBER_UNCOMPRESSED_BYTES + 1
    )

    report = AUDIT.audit_archive(archive_path)

    violation_ids = {
        item["id"]
        for item in report["payload_validation"]["resource_limit_violations"]
    }
    assert "member_size_limit" in violation_ids
    assert report["members"][0]["payload_bytes_read"] is None
    assert report["members"][0]["payload_status"] == (
        "skipped_member_resource_limit"
    )
    assert "archive_resource_limits_exceeded" in report["rejection_reasons"]


def test_high_compression_ratio_is_rejected_before_decompression(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(
        tmp_path / "ratio-bomb.zip", {"model.fbx": b"\x00" * (1024 * 1024)}
    )

    report = AUDIT.audit_archive(archive_path)

    assert [
        item["id"]
        for item in report["payload_validation"]["resource_limit_violations"]
    ] == ["compression_ratio_limit"]
    assert report["members"][0]["payload_bytes_read"] is None
    assert report["status"] == "rejected"


def test_external_buffer_must_be_at_least_declared_length(tmp_path: Path) -> None:
    short_report = AUDIT.audit_archive(
        _write_zip(
            tmp_path / "short-buffer.zip",
            {"scene.gltf": _gltf(buffer_uri="scene.bin"), "scene.bin": b"123"},
        )
    )
    longer_report = AUDIT.audit_archive(
        _write_zip(
            tmp_path / "longer-buffer.zip",
            {
                "scene.gltf": _gltf(buffer_uri="scene.bin"),
                "scene.bin": b"12345",
            },
        )
    )

    reference = short_report["gltf"]["missing_external_references"][0]
    assert reference["status"] == "buffer_too_short"
    assert reference["actual_length_bytes"] == 3
    assert reference["declared_byte_length"] == 4
    assert short_report["status"] == "rejected"
    # Khronos glTF 2.0 requires >= byteLength for external resources, not
    # equality, so harmless trailing bytes are accepted.
    assert longer_report["status"] == "ok"


def test_external_buffer_target_must_be_unique(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate-buffer.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("scene.gltf", _gltf(buffer_uri="scene.bin"))
        archive.writestr("scene.bin", b"1234")
        archive.writestr("scene.bin", b"5678")

    report = AUDIT.audit_archive(archive_path)

    assert report["gltf"]["missing_external_references"][0]["status"] == (
        "ambiguous_member"
    )
    assert report["status"] == "rejected"


@pytest.mark.parametrize(
    "uri",
    [
        "data:application/octet-stream;base64,@@@@",
        "data:application/octet-stream;base64,AA%ZZ",
    ],
)
def test_invalid_data_uri_encoding_is_rejected(tmp_path: Path, uri: str) -> None:
    report = AUDIT.audit_archive(
        _write_zip(tmp_path / "bad-data.zip", {"scene.gltf": _gltf(buffer_uri=uri)})
    )

    assert report["gltf"]["missing_external_references"][0]["status"] == (
        "invalid_data_uri"
    )
    assert report["status"] == "rejected"


def test_data_uri_buffer_length_uses_decoded_bytes(tmp_path: Path) -> None:
    short_report = AUDIT.audit_archive(
        _write_zip(
            tmp_path / "short-data.zip",
            {
                "scene.gltf": _gltf(
                    buffer_uri="data:application/octet-stream;base64,AQID"
                )
            },
        )
    )
    long_report = AUDIT.audit_archive(
        _write_zip(
            tmp_path / "long-data.zip",
            {
                "scene.gltf": _gltf(
                    buffer_uri="data:application/gltf-buffer;base64,AQIDBAU="
                )
            },
        )
    )

    assert short_report["gltf"]["missing_external_references"][0]["status"] == (
        "buffer_too_short"
    )
    assert long_report["status"] == "ok"


def test_glb_uri_less_buffer_requires_bin_chunk(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "missing-bin.zip",
        {
            "scene.glb": _glb(
                {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]}
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    assert report["gltf"]["missing_external_references"][0]["status"] == (
        "missing_bin"
    )
    assert report["status"] == "rejected"


def test_glb_bin_must_cover_declared_buffer_length(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "short-bin.zip",
        {
            "scene.glb": _glb(
                {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 8}]},
                bin_chunks=(b"1234",),
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    reference = report["gltf"]["missing_external_references"][0]
    assert reference["status"] == "buffer_too_short"
    assert reference["actual_length_bytes"] == 4
    assert report["status"] == "rejected"


def test_glb_duplicate_bin_chunk_is_rejected(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "duplicate-bin.zip",
        {
            "scene.glb": _glb(
                {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]},
                bin_chunks=(b"1234", b"5678"),
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    document = report["gltf"]["documents"][0]
    assert document["parsed"] is False
    assert "duplicate GLB BIN chunk" in document["parse_error"]
    assert report["status"] == "rejected"


def test_glb_truncated_bin_chunk_is_rejected(tmp_path: Path) -> None:
    malformed_bin = (
        (8).to_bytes(4, "little")
        + AUDIT.GLB_BIN_CHUNK.to_bytes(4, "little")
        + b"1234"
    )
    archive_path = _write_zip(
        tmp_path / "truncated-bin.zip",
        {
            "scene.glb": _glb(
                {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 8}]},
                raw_suffix=malformed_bin,
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    document = report["gltf"]["documents"][0]
    assert document["parsed"] is False
    assert "truncated GLB chunk" in document["parse_error"]


def test_glb_trailing_garbage_is_rejected(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "trailing-glb.zip",
        {"scene.glb": _glb({"asset": {"version": "2.0"}}, raw_suffix=b"x")},
    )

    report = AUDIT.audit_archive(archive_path)

    document = report["gltf"]["documents"][0]
    assert document["parsed"] is False
    assert "trailing bytes after GLB chunks" in document["parse_error"]


def test_glb_complete_padded_bin_is_accepted(tmp_path: Path) -> None:
    archive_path = _write_zip(
        tmp_path / "complete-bin.zip",
        {
            "scene.glb": _glb(
                {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 3}]},
                bin_chunks=(b"abc\x00",),
            )
        },
    )

    report = AUDIT.audit_archive(archive_path)

    assert report["status"] == "ok"
    assert report["gltf"]["documents"][0]["glb_bin_buffer_count"] == 1


@pytest.mark.parametrize("extension", ["blend", "fbx", "usd", "usda", "usdc"])
def test_empty_non_gltf_model_is_rejected(tmp_path: Path, extension: str) -> None:
    report = AUDIT.audit_archive(
        _write_zip(tmp_path / f"empty-{extension}.zip", {f"model.{extension}": b""})
    )

    assert report["assets"]["empty_models"] == [f"model.{extension}"]
    assert "empty_model_file" in report["rejection_reasons"]
    assert report["status"] == "rejected"


def test_unexpected_internal_exception_is_stable_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _write_zip(tmp_path / "valid.zip", {"model.fbx": b"fbx"})

    def _raise_internal(_path: object) -> dict[str, object]:
        raise RuntimeError("host secret that must not be emitted")

    monkeypatch.setattr(AUDIT, "_audit_archive_impl", _raise_internal)
    first = AUDIT.audit_archive(archive_path)
    second = AUDIT.audit_archive(archive_path)
    rendered = AUDIT.stable_json(first)

    assert first == second
    assert first["status"] == "input_error"
    assert first["exit_code"] == AUDIT.EXIT_INPUT_ERROR
    assert first["rejection_reasons"] == ["internal_audit_error:RuntimeError"]
    assert "host secret" not in rendered
    assert "Traceback" not in rendered
