#!/usr/bin/env python3
"""Read-only, deterministic audit for a user-downloaded 3D asset ZIP.

The archive is never extracted.  The report intentionally contains no clock time,
temporary path, or other run-specific value so the same archive and path produce
the same JSON on repeated audits.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import posixpath
import re
import stat
import struct
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable
from urllib.parse import unquote, unquote_to_bytes, urlsplit


SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INPUT_ERROR = 2

SUPPORTED_MODEL_EXTENSIONS = (
    ".blend",
    ".fbx",
    ".glb",
    ".gltf",
    ".usd",
    ".usda",
    ".usdc",
)
TEXTURE_EXTENSIONS = (
    ".bmp",
    ".dds",
    ".exr",
    ".gif",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".ktx",
    ".ktx2",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
)
MATERIAL_EXTENSIONS = (
    ".mat",
    ".material",
    ".mdl",
    ".mtl",
    ".sbsar",
)
NAME_KEYWORDS = ("body", "door", "drawer", "handle")

# A JSON chunk this large is already unsuitable for an intake manifest.  Bounding
# structured reads prevents a small compressed entry from becoming a memory bomb.
MAX_GLTF_JSON_BYTES = 64 * 1024 * 1024
MAX_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
STREAM_CHUNK_BYTES = 1024 * 1024
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:/")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_VALID_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def _absolute_path(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _normalized_slashes(path: str) -> str:
    return path.replace("\\", "/")


def _is_absolute_member_path(path: str) -> bool:
    normalized = _normalized_slashes(path)
    return (
        normalized.startswith("/")
        or normalized.startswith("//")
        or bool(_WINDOWS_ABSOLUTE_RE.match(normalized))
    )


def _has_path_traversal(path: str) -> bool:
    return ".." in _normalized_slashes(path).split("/")


def _canonical_member_path(path: str) -> str:
    normalized = posixpath.normpath(_normalized_slashes(path))
    return "" if normalized == "." else normalized


def _portable_member_key(path: str) -> str:
    """Approximate the strictest common path identity across target platforms."""

    return unicodedata.normalize("NFC", _canonical_member_path(path)).casefold()


def _has_windows_unsafe_component(path: str) -> bool:
    normalized = _normalized_slashes(path)
    if _WINDOWS_DRIVE_RE.match(normalized):
        return True
    return any(":" in component for component in normalized.split("/") if component)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    # Treat symlink mode bits as suspicious even when a crafted entry lies about
    # its creator OS.  This is intentionally stricter than Python's extractor.
    return stat.S_ISLNK(unix_mode)


def _is_zip_special_file(info: zipfile.ZipInfo) -> bool:
    """Return whether mode bits explicitly describe a non-regular file."""

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    return not info.is_dir() and file_type not in (0, stat.S_IFREG, stat.S_IFLNK)


def _is_zip_regular_file(info: zipfile.ZipInfo) -> bool:
    return (
        not info.is_dir()
        and not _is_zip_symlink(info)
        and not _is_zip_special_file(info)
    )


def _extension(path: str) -> str:
    return posixpath.splitext(_normalized_slashes(path))[1].lower()


def _format_name(path: str) -> str:
    return _extension(path).lstrip(".")


def _read_limited(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limit: int = MAX_GLTF_JSON_BYTES,
) -> bytes:
    if info.file_size > limit:
        raise ValueError(f"uncompressed JSON exceeds {limit} bytes")
    with archive.open(info, "r") as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"uncompressed JSON exceeds {limit} bytes")
    return payload


def _read_exact(source: BinaryIO, size: int) -> bytes:
    payload = source.read(size)
    if len(payload) != size:
        raise ValueError("truncated GLB")
    return payload


def _discard_exact(source: BinaryIO, size: int) -> bytes:
    remaining = size
    tail = b""
    while remaining:
        chunk = source.read(min(STREAM_CHUNK_BYTES, remaining))
        if not chunk:
            raise ValueError("truncated GLB")
        remaining -= len(chunk)
        tail = (tail + chunk)[-3:]
    return tail


def _load_glb_json(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> tuple[Any, int | None, bytes]:
    with archive.open(info, "r") as source:
        header = _read_exact(source, 12)
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != b"glTF":
            raise ValueError("invalid GLB magic")
        if version != 2:
            raise ValueError(f"unsupported GLB version {version}")
        if declared_length != info.file_size:
            raise ValueError("GLB declared length does not match ZIP member size")
        remaining = declared_length - 12
        chunk_index = 0
        json_payload: bytes | None = None
        bin_chunk_length: int | None = None
        bin_chunk_tail = b""
        while remaining:
            if remaining < 8:
                raise ValueError("trailing bytes after GLB chunks")
            chunk_length, chunk_type = struct.unpack(
                "<II", _read_exact(source, 8)
            )
            remaining -= 8
            if chunk_length % 4:
                raise ValueError("GLB chunk length is not 4-byte aligned")
            if chunk_length > remaining:
                raise ValueError("truncated GLB chunk")
            if chunk_index == 0 and chunk_type != GLB_JSON_CHUNK:
                raise ValueError("first GLB chunk is not JSON")
            if chunk_type == GLB_JSON_CHUNK:
                if chunk_index != 0 or json_payload is not None:
                    raise ValueError("duplicate or misplaced GLB JSON chunk")
                if chunk_length > MAX_GLTF_JSON_BYTES:
                    raise ValueError(
                        f"uncompressed JSON exceeds {MAX_GLTF_JSON_BYTES} bytes"
                    )
                json_payload = _read_exact(source, chunk_length)
            elif chunk_type == GLB_BIN_CHUNK:
                if bin_chunk_length is not None:
                    raise ValueError("duplicate GLB BIN chunk")
                if chunk_index != 1:
                    raise ValueError("GLB BIN chunk is not second")
                bin_chunk_length = chunk_length
                bin_chunk_tail = _discard_exact(source, chunk_length)
            else:
                # The glTF 2.0 forward-compatibility rule requires consumers to
                # ignore unknown chunks after the standard chunks, but they still
                # must be structurally valid and fully consumed.
                _discard_exact(source, chunk_length)
            remaining -= chunk_length
            chunk_index += 1
        if json_payload is None:
            raise ValueError("missing GLB JSON chunk")
    document = json.loads(json_payload.rstrip(b" ").decode("utf-8"))
    return document, bin_chunk_length, bin_chunk_tail


def _load_gltf_json(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> tuple[Any, int | None, bytes]:
    if _extension(info.filename) == ".glb":
        document, bin_chunk_length, bin_chunk_tail = _load_glb_json(archive, info)
    else:
        document = json.loads(
            _read_limited(archive, info).decode("utf-8-sig")
        )
        bin_chunk_length = None
        bin_chunk_tail = b""
    if not isinstance(document, dict):
        raise ValueError("glTF root must be a JSON object")
    return document, bin_chunk_length, bin_chunk_tail


def _iter_uri_fields(
    document: dict[str, Any],
) -> Iterable[tuple[str, int, str | None, int | None]]:
    buffers = document.get("buffers", [])
    if not isinstance(buffers, list):
        raise ValueError("glTF buffers must be an array")
    for index, item in enumerate(buffers):
        if not isinstance(item, dict):
            raise ValueError(f"glTF buffer {index} must be an object")
        byte_length = item.get("byteLength")
        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 1
        ):
            raise ValueError(f"glTF buffer {index} has invalid byteLength")
        uri = item.get("uri")
        if uri is not None and not isinstance(uri, str):
            raise ValueError(f"glTF buffer {index} has a non-string URI")
        yield "buffer", index, uri, byte_length

    images = document.get("images", [])
    if not isinstance(images, list):
        raise ValueError("glTF images must be an array")
    for index, item in enumerate(images):
        if not isinstance(item, dict):
            raise ValueError(f"glTF image {index} must be an object")
        uri = item.get("uri")
        if uri is not None:
            if not isinstance(uri, str):
                raise ValueError(f"glTF image {index} has a non-string URI")
            yield "image", index, uri, None


def _validate_percent_encoding(value: str) -> None:
    offset = 0
    while True:
        offset = value.find("%", offset)
        if offset < 0:
            return
        if _VALID_PERCENT_ESCAPE_RE.match(value, offset) is None:
            raise ValueError("invalid percent escape")
        offset += 3


def _decode_data_uri(uri: str, kind: str) -> tuple[bytes, str, str]:
    if not uri[:5].casefold() == "data:":
        raise ValueError("not a data URI")
    body = uri[5:]
    if "," not in body or "#" in body:
        raise ValueError("malformed data URI")
    header, encoded_payload = body.split(",", 1)
    _validate_percent_encoding(header)
    _validate_percent_encoding(encoded_payload)
    try:
        decoded_header = unquote_to_bytes(header).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("non-ASCII data URI header") from exc

    fields = decoded_header.split(";")
    base64_positions = [
        index for index, field in enumerate(fields) if field.casefold() == "base64"
    ]
    if len(base64_positions) > 1 or (
        base64_positions and base64_positions[0] != len(fields) - 1
    ):
        raise ValueError("misplaced base64 marker")
    if base64_positions and fields[-1] != "base64":
        raise ValueError("non-canonical base64 marker")
    is_base64 = bool(base64_positions)
    media_type = fields[0].casefold() if fields[0] else "text/plain"

    encoded_bytes = unquote_to_bytes(encoded_payload)
    if is_base64:
        try:
            payload = base64.b64decode(encoded_bytes, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64 data") from exc
        encoding = "base64"
    else:
        payload = encoded_bytes
        encoding = "percent"

    if kind == "buffer":
        if not is_base64:
            raise ValueError("glTF buffer data URI is not base64 encoded")
        if media_type not in ("application/octet-stream", "application/gltf-buffer"):
            raise ValueError("invalid glTF buffer data URI media type")
    return payload, media_type, encoding


def _resolve_external_uri(
    document_path: str,
    kind: str,
    index: int,
    uri: str,
    declared_byte_length: int | None,
    member_lookup: dict[str, list[tuple[int, zipfile.ZipInfo]]],
    safe_member_indices: set[int],
    validated_sizes: dict[int, int],
) -> dict[str, Any]:
    base = {
        "index": index,
        "kind": kind,
        "uri": uri,
    }
    if declared_byte_length is not None:
        base["declared_byte_length"] = declared_byte_length
    uri_with_slashes = _normalized_slashes(uri)
    split = urlsplit(uri_with_slashes)
    scheme = split.scheme.lower()

    if scheme == "data":
        try:
            payload, media_type, encoding = _decode_data_uri(uri, kind)
        except ValueError:
            return {
                **base,
                "actual_length_bytes": None,
                "data_encoding": None,
                "media_type": None,
                "status": "invalid_data_uri",
                "resolved_path": None,
            }
        embedded = {
            **base,
            "actual_length_bytes": len(payload),
            "data_encoding": encoding,
            "media_type": media_type,
            "resolved_path": None,
        }
        if (
            kind == "buffer"
            and declared_byte_length is not None
            and len(payload) < declared_byte_length
        ):
            return {**embedded, "status": "buffer_too_short"}
        return {**embedded, "status": "embedded_data"}
    if scheme or split.netloc:
        return {**base, "status": "non_archive_uri", "resolved_path": None}
    if split.query or split.fragment or "\\" in uri:
        return {**base, "status": "unsafe_uri", "resolved_path": None}

    try:
        _validate_percent_encoding(split.path)
    except ValueError:
        return {**base, "status": "invalid_uri_encoding", "resolved_path": None}

    decoded_path = _normalized_slashes(unquote(split.path))
    if not decoded_path:
        return {**base, "status": "missing", "resolved_path": ""}
    if _is_absolute_member_path(decoded_path) or _has_windows_unsafe_component(
        decoded_path
    ):
        return {**base, "status": "unsafe_uri", "resolved_path": decoded_path}

    document_directory = posixpath.dirname(_canonical_member_path(document_path))
    resolved = posixpath.normpath(posixpath.join(document_directory, decoded_path))
    if resolved == ".." or resolved.startswith("../"):
        return {**base, "status": "unsafe_uri", "resolved_path": resolved}
    candidates = member_lookup.get(resolved, [])
    if not candidates:
        return {**base, "status": "missing", "resolved_path": resolved}
    if len(candidates) != 1:
        return {**base, "status": "ambiguous_member", "resolved_path": resolved}
    archive_index, _target_info = candidates[0]
    if archive_index not in safe_member_indices:
        return {**base, "status": "unsafe_member", "resolved_path": resolved}
    if archive_index not in validated_sizes:
        return {**base, "status": "unreadable_member", "resolved_path": resolved}
    actual_length = validated_sizes[archive_index]
    resolved_reference = {
        **base,
        "actual_length_bytes": actual_length,
        "resolved_path": resolved,
    }
    if (
        kind == "buffer"
        and declared_byte_length is not None
        and actual_length < declared_byte_length
    ):
        return {**resolved_reference, "status": "buffer_too_short"}
    return {**resolved_reference, "status": "present"}


def _name_hits(document_path: str, document: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for kind, collection_name in (("node", "nodes"), ("mesh", "meshes")):
        collection = document.get(collection_name, [])
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = item["name"]
            folded = name.casefold()
            keywords = [keyword for keyword in NAME_KEYWORDS if keyword in folded]
            if keywords:
                hits.append(
                    {
                        "document": document_path,
                        "index": index,
                        "keywords": keywords,
                        "kind": kind,
                        "name": name,
                    }
                )
    return hits


def _inspect_gltf_document(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    member_lookup: dict[str, list[tuple[int, zipfile.ZipInfo]]],
    safe_member_indices: set[int],
    validated_sizes: dict[int, int],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "external_references": [],
        "external_uri_closure": False,
        "format": _format_name(info.filename),
        "name_hits": [],
        "parse_error": None,
        "parsed": False,
        "path": info.filename,
    }
    try:
        document, bin_chunk_length, bin_chunk_tail = _load_gltf_json(archive, info)
        document_format = _format_name(info.filename)
        references: list[dict[str, Any]] = []
        glb_bin_claimed = False
        for kind, index, uri, declared_byte_length in _iter_uri_fields(document):
            if uri is not None:
                references.append(
                    _resolve_external_uri(
                        info.filename,
                        kind,
                        index,
                        uri,
                        declared_byte_length,
                        member_lookup,
                        safe_member_indices,
                        validated_sizes,
                    )
                )
                continue

            reference: dict[str, Any] = {
                "index": index,
                "kind": kind,
                "resolved_path": None,
                "uri": None,
            }
            if kind == "buffer" and document_format == "glb" and index == 0:
                glb_bin_claimed = True
                reference["declared_byte_length"] = declared_byte_length
                reference["actual_length_bytes"] = bin_chunk_length
                if bin_chunk_length is None:
                    reference["status"] = "missing_bin"
                elif bin_chunk_length < declared_byte_length:
                    reference["status"] = "buffer_too_short"
                elif bin_chunk_length > declared_byte_length + 3:
                    reference["status"] = "bin_length_mismatch"
                else:
                    padding_length = bin_chunk_length - declared_byte_length
                    if padding_length and bin_chunk_tail[-padding_length:] != (
                        b"\x00" * padding_length
                    ):
                        reference["status"] = "invalid_bin_padding"
                    else:
                        reference["status"] = "glb_bin"
            else:
                reference["status"] = "missing_uri"
            references.append(reference)

        if bin_chunk_length is not None and not glb_bin_claimed:
            raise ValueError("GLB BIN chunk has no matching URI-less buffer 0")
        references.sort(
            key=lambda item: (item["kind"], item["index"], str(item["uri"]))
        )
        internal_statuses = {"embedded_data", "glb_bin"}
        accepted_statuses = {"embedded_data", "glb_bin", "present"}
        external = [
            item for item in references if item["status"] not in internal_statuses
        ]
        base.update(
            {
                "embedded_data_uri_count": sum(
                    item["status"] == "embedded_data" for item in references
                ),
                "external_references": external,
                "external_uri_closure": all(
                    item["status"] in accepted_statuses for item in references
                ),
                "glb_bin_buffer_count": sum(
                    item["status"] == "glb_bin" for item in references
                ),
                "name_hits": _name_hits(info.filename, document),
                "parsed": True,
            }
        )
    except (
        json.JSONDecodeError,
        NotImplementedError,
        UnicodeDecodeError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
    ) as exc:
        base["parse_error"] = f"{type(exc).__name__}: {exc}"
    return base


def _member_report(index: int, info: zipfile.ZipInfo) -> dict[str, Any]:
    is_symlink = _is_zip_symlink(info)
    is_special_file = _is_zip_special_file(info)
    is_regular_file = _is_zip_regular_file(info)
    return {
        "absolute_path": _is_absolute_member_path(info.filename),
        "archive_index": index,
        "compressed_size_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "is_directory": info.is_dir(),
        "is_regular_file": is_regular_file,
        "is_special_file": is_special_file,
        "is_symlink": is_symlink,
        "path": info.filename,
        "path_traversal": _has_path_traversal(info.filename),
        "payload_bytes_read": None,
        "payload_status": "pending" if is_regular_file else "not_applicable",
        "uncompressed_size_bytes": info.file_size,
    }


def _member_read_error_code(exc: Exception) -> str:
    if isinstance(exc, zipfile.BadZipFile) and "CRC" in str(exc).upper():
        return "crc_mismatch"
    if isinstance(exc, RuntimeError) and "encrypt" in str(exc).casefold():
        return "encrypted_member"
    if isinstance(exc, NotImplementedError):
        return "unsupported_compression"
    if isinstance(exc, zipfile.BadZipFile):
        return "invalid_member_data"
    return f"member_read_failed:{type(exc).__name__}"


def _stream_validate_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    total_bytes_before: int,
) -> tuple[int, str | None, dict[str, Any] | None]:
    bytes_read = 0
    checksum = 0
    try:
        with archive.open(info, "r") as source:
            while True:
                chunk = source.read(STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_MEMBER_UNCOMPRESSED_BYTES:
                    return (
                        bytes_read,
                        None,
                        {
                            "actual_bytes": bytes_read,
                            "id": "member_actual_size_limit",
                            "limit_bytes": MAX_MEMBER_UNCOMPRESSED_BYTES,
                        },
                    )
                if total_bytes_before + bytes_read > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    return (
                        bytes_read,
                        None,
                        {
                            "actual_bytes": total_bytes_before + bytes_read,
                            "id": "archive_actual_size_limit",
                            "limit_bytes": MAX_TOTAL_UNCOMPRESSED_BYTES,
                        },
                    )
                checksum = binascii.crc32(chunk, checksum)
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        return bytes_read, _member_read_error_code(exc), None

    if bytes_read != info.file_size:
        return bytes_read, "uncompressed_size_mismatch", None
    if checksum & 0xFFFFFFFF != info.CRC:
        return bytes_read, "crc_mismatch", None
    return bytes_read, None, None


def _validate_member_payloads(
    archive: zipfile.ZipFile,
    indexed_infos: list[tuple[int, zipfile.ZipInfo]],
    member_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, int]]:
    rows_by_index = {row["archive_index"]: row for row in member_rows}
    regular = [
        (index, info)
        for index, info in indexed_infos
        if _is_zip_regular_file(info)
    ]
    declared_total = sum(info.file_size for _index, info in regular)
    violations: list[dict[str, Any]] = []
    member_violation_indices: set[int] = set()
    for index, info in regular:
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            violations.append(
                {
                    "archive_index": index,
                    "declared_bytes": info.file_size,
                    "id": "member_size_limit",
                    "limit_bytes": MAX_MEMBER_UNCOMPRESSED_BYTES,
                    "path": info.filename,
                }
            )
            member_violation_indices.add(index)
        if info.file_size and (
            info.compress_size <= 0
            or info.file_size > MAX_COMPRESSION_RATIO * info.compress_size
        ):
            violations.append(
                {
                    "archive_index": index,
                    "compressed_bytes": info.compress_size,
                    "id": "compression_ratio_limit",
                    "limit_ratio": MAX_COMPRESSION_RATIO,
                    "path": info.filename,
                    "uncompressed_bytes": info.file_size,
                }
            )
            member_violation_indices.add(index)
    archive_size_violation = declared_total > MAX_TOTAL_UNCOMPRESSED_BYTES
    if archive_size_violation:
        violations.append(
            {
                "declared_bytes": declared_total,
                "id": "archive_size_limit",
                "limit_bytes": MAX_TOTAL_UNCOMPRESSED_BYTES,
            }
        )

    read_errors: list[dict[str, Any]] = []
    validated_sizes: dict[int, int] = {}
    total_bytes_read = 0
    for index, info in regular:
        row = rows_by_index[index]
        if archive_size_violation:
            row["payload_status"] = "skipped_archive_size_limit"
            continue
        if index in member_violation_indices:
            row["payload_status"] = "skipped_member_resource_limit"
            continue
        bytes_read, error_code, runtime_violation = _stream_validate_member(
            archive, info, total_bytes_read
        )
        row["payload_bytes_read"] = bytes_read
        total_bytes_read += bytes_read
        if runtime_violation is not None:
            violation = {
                "archive_index": index,
                "path": info.filename,
                **runtime_violation,
            }
            violations.append(violation)
            row["payload_status"] = "resource_limit_exceeded"
        elif error_code is not None:
            read_errors.append(
                {
                    "archive_index": index,
                    "error": error_code,
                    "path": info.filename,
                }
            )
            row["payload_status"] = "read_error"
        else:
            validated_sizes[index] = bytes_read
            row["payload_status"] = "ok"

    violations.sort(
        key=lambda item: (
            str(item.get("path", "")),
            int(item.get("archive_index", -1)),
            item["id"],
        )
    )
    read_errors.sort(key=lambda item: (item["path"], item["archive_index"]))
    return (
        {
            "all_regular_members_readable": not violations and not read_errors,
            "declared_total_uncompressed_bytes": declared_total,
            "limits": {
                "max_compression_ratio": MAX_COMPRESSION_RATIO,
                "max_member_uncompressed_bytes": MAX_MEMBER_UNCOMPRESSED_BYTES,
                "max_total_uncompressed_bytes": MAX_TOTAL_UNCOMPRESSED_BYTES,
            },
            "member_read_errors": read_errors,
            "regular_member_count": len(regular),
            "resource_limit_violations": violations,
            "total_uncompressed_bytes_read": total_bytes_read,
            "validated_member_count": len(validated_sizes),
        },
        validated_sizes,
    )


def _empty_report(path: Path) -> dict[str, Any]:
    return {
        "archive": {
            "path": str(path),
            "sha256": None,
            "size_bytes": None,
        },
        "assets": {
            "empty_models": [],
            "formats": [],
            "materials": [],
            "models": [],
            "textures": [],
        },
        "checks": [],
        "exit_code": EXIT_INPUT_ERROR,
        "gltf": {
            "documents": [],
            "external_uri_closure": False,
            "missing_external_references": [],
        },
        "members": [],
        "name_hits": [],
        "payload_validation": {
            "all_regular_members_readable": False,
            "declared_total_uncompressed_bytes": 0,
            "limits": {
                "max_compression_ratio": MAX_COMPRESSION_RATIO,
                "max_member_uncompressed_bytes": MAX_MEMBER_UNCOMPRESSED_BYTES,
                "max_total_uncompressed_bytes": MAX_TOTAL_UNCOMPRESSED_BYTES,
            },
            "member_read_errors": [],
            "regular_member_count": 0,
            "resource_limit_violations": [],
            "total_uncompressed_bytes_read": 0,
            "validated_member_count": 0,
        },
        "rejection_reasons": [],
        "schema_version": SCHEMA_VERSION,
        "security": {
            "absolute_paths": [],
            "backslash_paths": [],
            "duplicate_paths": [],
            "is_safe": False,
            "portable_path_collisions": [],
            "path_traversal": [],
            "special_files": [],
            "symlinks": [],
            "windows_unsafe_paths": [],
        },
        "status": "input_error",
    }


def _input_error_report(path: Path, reason: str) -> dict[str, Any]:
    report = _empty_report(path)
    report["checks"] = [{"id": "zip_readable", "passed": False}]
    report["rejection_reasons"] = [reason]
    return report


def _audit_archive_impl(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Return a deterministic audit report without writing to the archive or disk."""

    archive_path = _absolute_path(path)
    report = _empty_report(archive_path)
    try:
        source = archive_path.open("rb")
    except OSError as exc:
        return _input_error_report(
            archive_path, f"archive_open_failed:{type(exc).__name__}"
        )

    try:
        with source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                return _input_error_report(archive_path, "archive_is_not_regular_file")
            sha256 = _sha256_stream(source)
            source.seek(0)
            report["archive"].update(
                {"sha256": sha256, "size_bytes": before.st_size}
            )

            try:
                with zipfile.ZipFile(source, "r") as archive:
                    infos = archive.infolist()
                    indexed_infos = list(enumerate(infos))
                    member_rows = [
                        _member_report(index, info) for index, info in indexed_infos
                    ]
                    payload_validation, validated_sizes = _validate_member_payloads(
                        archive, indexed_infos, member_rows
                    )
                    member_rows.sort(key=lambda row: (row["path"], row["archive_index"]))
                    report["members"] = member_rows
                    report["payload_validation"] = payload_validation

                    regular_infos = [
                        info
                        for info in infos
                        if _is_zip_regular_file(info)
                    ]
                    member_lookup: dict[
                        str, list[tuple[int, zipfile.ZipInfo]]
                    ] = {}
                    portable_lookup: dict[str, list[int]] = {}
                    for index, info in indexed_infos:
                        member_lookup.setdefault(
                            _canonical_member_path(info.filename), []
                        ).append((index, info))
                        portable_lookup.setdefault(
                            _portable_member_key(info.filename), []
                        ).append(index)
                    safe_member_indices = {
                        index
                        for index, info in indexed_infos
                        if _is_zip_regular_file(info)
                        and bool(_canonical_member_path(info.filename))
                        and not _is_absolute_member_path(info.filename)
                        and not _has_path_traversal(info.filename)
                        and "\\" not in info.filename
                        and not _has_windows_unsafe_component(info.filename)
                        and len(
                            member_lookup[_canonical_member_path(info.filename)]
                        )
                        == 1
                        and len(portable_lookup[_portable_member_key(info.filename)])
                        == 1
                    }
                    info_indices = {id(info): index for index, info in indexed_infos}

                    model_infos = [
                        info
                        for info in regular_infos
                        if _extension(info.filename) in SUPPORTED_MODEL_EXTENSIONS
                    ]
                    model_infos.sort(key=lambda info: info.filename)
                    models = [
                        {"format": _format_name(info.filename), "path": info.filename}
                        for info in model_infos
                    ]
                    report["assets"] = {
                        "empty_models": sorted(
                            info.filename for info in model_infos if info.file_size == 0
                        ),
                        "formats": sorted({item["format"] for item in models}),
                        "materials": sorted(
                            info.filename
                            for info in regular_infos
                            if _extension(info.filename) in MATERIAL_EXTENSIONS
                        ),
                        "models": models,
                        "textures": sorted(
                            info.filename
                            for info in regular_infos
                            if _extension(info.filename) in TEXTURE_EXTENSIONS
                        ),
                    }

                    gltf_infos = [
                        info
                        for info in model_infos
                        if _extension(info.filename) in (".gltf", ".glb")
                    ]
                    documents: list[dict[str, Any]] = []
                    for info in gltf_infos:
                        if info_indices[id(info)] not in validated_sizes:
                            documents.append(
                                {
                                    "external_references": [],
                                    "external_uri_closure": False,
                                    "format": _format_name(info.filename),
                                    "name_hits": [],
                                    "parse_error": (
                                        "archive member payload was not validated"
                                    ),
                                    "parsed": False,
                                    "path": info.filename,
                                }
                            )
                        else:
                            documents.append(
                                _inspect_gltf_document(
                                    archive,
                                    info,
                                    member_lookup,
                                    safe_member_indices,
                                    validated_sizes,
                                )
                            )
                    documents.sort(key=lambda document: document["path"])
                    missing = [
                        {"document": document["path"], **reference}
                        for document in documents
                        for reference in document["external_references"]
                        if reference["status"] != "present"
                    ]
                    missing.sort(
                        key=lambda item: (
                            item["document"],
                            item["kind"],
                            item["index"],
                            str(item["uri"]),
                        )
                    )
                    name_hits = [
                        hit for document in documents for hit in document["name_hits"]
                    ]
                    name_hits.sort(
                        key=lambda item: (
                            item["document"],
                            item["kind"],
                            item["index"],
                            item["name"],
                        )
                    )
                    gltf_closure = all(
                        document["parsed"] and document["external_uri_closure"]
                        for document in documents
                    )
                    report["gltf"] = {
                        "documents": documents,
                        "external_uri_closure": gltf_closure,
                        "missing_external_references": missing,
                    }
                    report["name_hits"] = name_hits
            except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as exc:
                report.update(
                    {
                        "checks": [{"id": "zip_readable", "passed": False}],
                        "exit_code": EXIT_INPUT_ERROR,
                        "rejection_reasons": [
                            f"invalid_zip:{type(exc).__name__}"
                        ],
                        "status": "input_error",
                    }
                )
                return report

            after = os.fstat(source.fileno())
    except OSError as exc:
        return _input_error_report(
            archive_path, f"archive_read_failed:{type(exc).__name__}"
        )

    changed_during_audit = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    absolute_paths = sorted(
        row["path"] for row in report["members"] if row["absolute_path"]
    )
    traversals = sorted(
        row["path"] for row in report["members"] if row["path_traversal"]
    )
    symlinks = sorted(row["path"] for row in report["members"] if row["is_symlink"])
    special_files = sorted(
        row["path"] for row in report["members"] if row["is_special_file"]
    )
    backslash_paths = sorted(
        row["path"] for row in report["members"] if "\\" in row["path"]
    )
    windows_unsafe_paths = sorted(
        row["path"]
        for row in report["members"]
        if _has_windows_unsafe_component(row["path"])
    )
    paths_to_indices: dict[str, list[int]] = {}
    portable_paths: dict[str, list[str]] = {}
    for row in report["members"]:
        paths_to_indices.setdefault(row["path"], []).append(row["archive_index"])
        portable_paths.setdefault(_portable_member_key(row["path"]), []).append(
            row["path"]
        )
    duplicate_paths = sorted(
        path for path, indices in paths_to_indices.items() if len(indices) > 1
    )
    portable_path_collisions = [
        {"members": sorted(paths), "portable_path": portable_path}
        for portable_path, paths in portable_paths.items()
        if len(paths) > 1
    ]
    portable_path_collisions.sort(key=lambda item: item["portable_path"])
    security_safe = not any(
        (
            absolute_paths,
            backslash_paths,
            duplicate_paths,
            portable_path_collisions,
            special_files,
            traversals,
            symlinks,
            windows_unsafe_paths,
        )
    )
    report["security"] = {
        "absolute_paths": absolute_paths,
        "backslash_paths": backslash_paths,
        "duplicate_paths": duplicate_paths,
        "is_safe": security_safe,
        "portable_path_collisions": portable_path_collisions,
        "path_traversal": traversals,
        "special_files": special_files,
        "symlinks": symlinks,
        "windows_unsafe_paths": windows_unsafe_paths,
    }

    archive_nonempty = any(
        row["is_regular_file"] for row in report["members"]
    )
    supported_model_present = bool(report["assets"]["models"])
    model_files_nonempty = not report["assets"]["empty_models"]
    resource_limits_ok = not report["payload_validation"][
        "resource_limit_violations"
    ]
    member_payloads_readable = report["payload_validation"][
        "all_regular_members_readable"
    ]
    gltf_parsed = all(document["parsed"] for document in report["gltf"]["documents"])
    gltf_closure = report["gltf"]["external_uri_closure"]
    report["checks"] = [
        {"id": "zip_readable", "passed": True},
        {"id": "archive_unchanged_during_audit", "passed": not changed_during_audit},
        {"id": "archive_nonempty", "passed": archive_nonempty},
        {"id": "archive_members_safe", "passed": security_safe},
        {"id": "archive_resource_limits", "passed": resource_limits_ok},
        {
            "id": "archive_member_payloads_readable",
            "passed": member_payloads_readable,
        },
        {"id": "supported_model_present", "passed": supported_model_present},
        {"id": "model_files_nonempty", "passed": model_files_nonempty},
        {"id": "gltf_documents_parse", "passed": gltf_parsed},
        {"id": "gltf_external_uri_closure", "passed": gltf_closure},
    ]

    reasons: list[str] = []
    if changed_during_audit:
        reasons.append("archive_changed_during_audit")
    if not archive_nonempty:
        reasons.append("empty_archive")
    if not security_safe:
        reasons.append("unsafe_archive_members")
    if not resource_limits_ok:
        reasons.append("archive_resource_limits_exceeded")
    if report["payload_validation"]["member_read_errors"]:
        reasons.append("archive_member_read_failed")
    if not supported_model_present:
        reasons.append("no_supported_model")
    if not model_files_nonempty:
        reasons.append("empty_model_file")
    if not gltf_parsed:
        reasons.append("invalid_gltf_document")
    if not gltf_closure:
        reasons.append("gltf_external_dependency_missing")

    report["rejection_reasons"] = reasons
    report["exit_code"] = EXIT_REJECTED if reasons else EXIT_OK
    report["status"] = "rejected" if reasons else "ok"
    return report


def _safe_error_path(path: os.PathLike[str] | str) -> Path:
    try:
        return _absolute_path(path)
    except Exception:
        return Path("<invalid-path>")


def audit_archive(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Return stable JSON-compatible input errors for unexpected audit failures."""

    try:
        return _audit_archive_impl(path)
    except Exception as exc:
        # Do not expose exception messages or tracebacks: those may contain host
        # paths or data from an untrusted archive.  Expected policy violations are
        # handled inside the implementation and remain status="rejected".
        return _input_error_report(
            _safe_error_path(path), f"internal_audit_error:{type(exc).__name__}"
        )


def stable_json(report: dict[str, Any]) -> str:
    # ASCII escaping keeps CLI output valid even if untrusted JSON contains an
    # isolated UTF-16 surrogate that cannot be encoded directly as UTF-8.
    return json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of a downloaded 3D asset ZIP"
    )
    parser.add_argument("archive", help="path to the ZIP archive to audit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = audit_archive(args.archive)
        rendered = stable_json(report)
    except Exception as exc:
        report = _input_error_report(
            _safe_error_path(args.archive),
            f"internal_cli_error:{type(exc).__name__}",
        )
        rendered = stable_json(report)
    sys.stdout.write(rendered)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
