"""Metal Office Desk scene construction for B320-BC017."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import TYPE_CHECKING, Any, Mapping, Sequence
import zipfile

if TYPE_CHECKING:
    from .b320_bimanual_scene import B320SpawnResult


TASK_ID = "B320-BC017"
TASK_NAME = "t15_open_drawers"
TASK_REQUIREMENT = (
    "Use the assigned hand on each drawer, pull both beyond 60 percent "
    "travel, and cross the threshold within 1.2 seconds."
)

SELECTED_ASSET_TITLE = "Metal Office Desk"
SELECTED_ASSET_CREATOR = "Ulan Cabanilla"
SELECTED_ASSET_PLATFORM = "Poly Haven"
SELECTED_ASSET_UUID = "metal_office_desk"
SELECTED_ASSET_URL = "https://polyhaven.com/a/metal_office_desk"
SELECTED_ASSET_LICENSE = "CC0 1.0"
SELECTED_ASSET_LICENSE_URL = (
    "https://creativecommons.org/publicdomain/zero/1.0/"
)

BC017_ASSET_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "b320_bc017_wooden_drawer_small_d6"
)
BC017_RUNTIME_USD_RELATIVE_PATH = Path("runtime/bc017_wooden_drawer.usda")
# The selected source arrives on the wrong side for the collection station.
# Apply the requested +90 degree YZ-plane rotation (about world X), followed
# by a 180 degree XY-plane turn (about world Z).  With column-vector
# composition this is Rz(180) @ Rx(90), represented in WXYZ order.
_BC017_SQRT_HALF = math.sqrt(0.5)
BC017_ASSET_ORIENTATION_WXYZ = (
    0.0,
    0.0,
    _BC017_SQRT_HALF,
    _BC017_SQRT_HALF,
)

# Reviewed closed-pose geometry in the runtime asset's local frame.  Under the
# orientation above a local point maps to (-x, +z, +y), before translation.
# Place the lowest leg/support coordinate 1 mm above the 0.89 m tabletop and
# center the left/right handle pair at x=-0.24, y=0.  This is 60 mm farther
# along world +X than the previous pose, clearing the initialized hands while
# remaining in comfortable bimanual reach. Pulling either authored +X joint
# then moves its handle toward world -X (toward the robot arms).
BC017_TABLE_TOP_Z_M = 0.89
BC017_TABLE_SUPPORT_CLEARANCE_M = 0.001
BC017_LOCAL_SUPPORT_MIN_Y_M = 0.0
BC017_LOCAL_LEFT_HANDLE_CENTER_M = (
    0.20526554659005003,
    0.27908053547152495,
    0.32300440371045,
)
BC017_LOCAL_RIGHT_HANDLE_CENTER_M = (
    0.2052655532955,
    0.279080535471525,
    -0.32305595576760004,
)
BC017_LOCAL_HANDLE_PAIR_CENTER_M = tuple(
    0.5
    * (
        BC017_LOCAL_LEFT_HANDLE_CENTER_M[index]
        + BC017_LOCAL_RIGHT_HANDLE_CENTER_M[index]
    )
    for index in range(3)
)
# Backward-compatible scalar names describe the pair center.
BC017_LOCAL_HANDLE_CENTER_X_M = BC017_LOCAL_HANDLE_PAIR_CENTER_M[0]
BC017_LOCAL_HANDLE_CENTER_Y_M = BC017_LOCAL_HANDLE_PAIR_CENTER_M[1]
BC017_LOCAL_HANDLE_CENTER_Z_M = BC017_LOCAL_HANDLE_PAIR_CENTER_M[2]
BC017_TARGET_CLOSED_HANDLE_X_M = -0.24
BC017_TARGET_HANDLE_Y_M = 0.0
BC017_DRAWER_TRAVEL_M = 0.16
BC017_MIN_USABLE_PULL_FRACTION = 0.70
BC017_ASSET_TRANSLATION_M = (
    BC017_TARGET_CLOSED_HANDLE_X_M + BC017_LOCAL_HANDLE_CENTER_X_M,
    BC017_TARGET_HANDLE_Y_M - BC017_LOCAL_HANDLE_CENTER_Z_M,
    BC017_TABLE_TOP_Z_M
    + BC017_TABLE_SUPPORT_CLEARANCE_M
    - BC017_LOCAL_SUPPORT_MIN_Y_M,
)
BC017_MANIFEST_FILENAME = "asset_manifest.json"
BC017_ASSET_ROOT_PATH = "/World/Task/B320Benchmark/bc017_wooden_drawer"

BC017_BASE_LINK_PRIM = "bc017_fixture_base_link"
BC017_WORLD_FIXED_JOINT_PRIM = "bc017_world_fixed_joint"
BC017_LEFT_DRAWER_LINK_PRIM = "bc017_left_drawer_link"
BC017_RIGHT_DRAWER_LINK_PRIM = "bc017_right_drawer_link"
BC017_LEFT_JOINT_NAME = "bc017_left_drawer_prismatic_joint"
BC017_RIGHT_JOINT_NAME = "bc017_right_drawer_prismatic_joint"

# Each selected source drawer has a large integrated U-shaped metal pull.
# Assistance must bind inside the visible pull vertex bounds, not to the
# drawer face or to another face of the desk.
# Bounds are in the runtime asset frame; drawer joint displacement is applied
# along local +X before the canonical asset transform maps it toward world -X.
BC017_HANDLE_GRIP_LOCAL_BOUNDS_M = {
    "left": (
        (0.19580978751187503, 0.27664745002987495, 0.29710611999045),
        (0.21472130566822503, 0.28151362091317494, 0.34890268743045005),
    ),
    "right": (
        (0.195809800922775, 0.276647450029875, -0.3489542394876),
        (0.214721305668225, 0.281513620913175, -0.2971576720476),
    ),
}
BC017_TARGET_EDGE_COLORS_RGB = {
    "left": (0.08, 0.72, 1.0),
    "right": (1.0, 0.30, 0.06),
}
BC017_TARGET_EDGE_HIGHLIGHT_ROOTS = {
    "left": (
        f"{BC017_ASSET_ROOT_PATH}/{BC017_LEFT_DRAWER_LINK_PRIM}/"
        "left_hand_target_edge"
    ),
    "right": (
        f"{BC017_ASSET_ROOT_PATH}/{BC017_RIGHT_DRAWER_LINK_PRIM}/"
        "right_hand_target_edge"
    ),
}
BC017_GRASP_JOINT_ROOT_PATH = (
    f"{BC017_ASSET_ROOT_PATH}/fingertip_handle_grasp_joints"
)
BC017_GRASP_D6_AXES = (
    "transX",
    "transY",
    "transZ",
    "rotX",
    "rotY",
    "rotZ",
)
BC017_GRASP_FREE_LIMIT_BY_AXIS = {
    axis: (1000.0 if axis.startswith("trans") else 179.0)
    for axis in BC017_GRASP_D6_AXES
}
BC017_GRASP_LOCK_LIMIT_BY_AXIS = {
    axis: (1.0e-5 if axis.startswith("trans") else 1.0e-3)
    for axis in BC017_GRASP_D6_AXES
}
_BC017_DRAWER_FRONT_LOCAL_BOUNDS_M = {
    "left": (
        (0.24696703702209594, 0.23045445978641516),
        (0.31983460932968255, 0.41555429399013516),
    ),
    "right": (
        (0.24696703702213163, -0.4156058996915817),
        (0.3198345825076282, -0.23050606548786168),
    ),
}
_BC017_TARGET_EDGE_FRONT_X_M = 0.1958
_BC017_TARGET_EDGE_DEPTH_M = 0.0020
_BC017_TARGET_EDGE_WIDTH_M = 0.0045

_MANIFEST_SCHEMA = "handson.bc017.asset-manifest"
_MANIFEST_VERSION = 1
_MATERIAL_REPORT_SCHEMA = "handson.bc017.material-closure"
_COLLISION_REPORT_SCHEMA = "handson.bc017.collision-audit"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_USD_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FILE_ROLES = frozenset(
    {
        "source_archive",
        "runtime_usd",
        "material",
        "texture",
        "collision_mesh",
        "material_audit",
        "collision_audit",
        "license",
    }
)
_RETIRED_LINEAGE_MARKER = "dv_storage_drawers_" + "40417"
_ERROR_PREFIX = "B320-BC017 asset gate closed before stage mutation"


class BC017AssetGateError(RuntimeError):
    """The selected BC017 asset did not satisfy the pre-compose boundary."""


class BC017AssetNotReadyError(BC017AssetGateError):
    """The operator-selected source or its reviewed conversion is absent."""


@dataclass(frozen=True)
class BC017DrawerManifest:
    role: str
    link_prim: str
    joint_prim: str
    joint_name: str
    axis: str
    closed_limit_m: float
    open_limit_m: float

    @property
    def travel_m(self) -> float:
        return self.open_limit_m - self.closed_limit_m


@dataclass(frozen=True)
class BC017BytePreflightReport:
    """Immutable byte-preflight result; never a live/formal USD verdict."""

    asset_directory: Path
    manifest_path: Path
    manifest_sha256: str
    source_archive_path: Path
    source_archive_sha256: str
    runtime_usd_path: Path
    runtime_usd_sha256: str
    base_link_prim: str
    world_fixed_joint_prim: str
    drawers: tuple[BC017DrawerManifest, BC017DrawerManifest]
    translation_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    collider_counts: tuple[int, int, int]
    validated_files: tuple[Path, ...]

    @property
    def formal(self) -> bool:
        """Byte and manifest checks alone are deliberately non-formal."""

        return False

    @property
    def fixture_body_prim_path(self) -> str:
        return _composed_path(self.base_link_prim)

    @property
    def left_body_prim_path(self) -> str:
        return _composed_path(self.drawers[0].link_prim)

    @property
    def right_body_prim_path(self) -> str:
        return _composed_path(self.drawers[1].link_prim)

    @property
    def left_joint_name(self) -> str:
        return self.drawers[0].joint_name

    @property
    def right_joint_name(self) -> str:
        return self.drawers[1].joint_name

    @property
    def left_closed_q_m(self) -> float:
        return self.drawers[0].closed_limit_m

    @property
    def left_open_q_m(self) -> float:
        return self.drawers[0].open_limit_m

    @property
    def right_closed_q_m(self) -> float:
        return self.drawers[1].closed_limit_m

    @property
    def right_open_q_m(self) -> float:
        return self.drawers[1].open_limit_m


# Compatibility for the short-lived scaffold name.  New code must use the
# explicit BytePreflight name so a manifest/hash result cannot be confused
# with the separate Isaac Sim 5.1 USD, PhysX, or collection acceptance layers.
BC017ValidatedAssetManifest = BC017BytePreflightReport


def _fail(code: str, detail: str, *, not_ready: bool = False) -> None:
    error_type = BC017AssetNotReadyError if not_ready else BC017AssetGateError
    raise error_type(f"{_ERROR_PREFIX}: [{code}] {detail}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("schema_invalid", f"{label} must be an object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("schema_invalid", f"{label} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            "schema_invalid",
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}",
        )


def _require_bool(value: Any, expected: bool, label: str, code: str) -> None:
    if value is not expected:
        _fail(code, f"{label} must be {expected!r}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail("schema_invalid", f"{label} must be a lowercase SHA-256")
    return value


def _require_size(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("schema_invalid", f"{label} must be a positive integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("schema_invalid", f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail("schema_invalid", f"{label} must be a finite number")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_string_values(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            found.extend(_walk_string_values(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.extend(_walk_string_values(nested))
    return tuple(found)


def _reject_retired_lineage(manifest: Mapping[str, Any]) -> None:
    marker = _RETIRED_LINEAGE_MARKER.casefold()
    if any(marker in value.casefold() for value in _walk_string_values(manifest)):
        _fail(
            "retired_lineage",
            "the clean-room manifest contains a retired BC017 asset marker",
        )


def _normalize_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("path_unsafe", f"{label} must be a non-empty POSIX path")
    if "\\" in value or "\x00" in value:
        _fail("path_unsafe", f"{label} contains a forbidden separator or NUL")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
        or path.as_posix() != value
    ):
        _fail("path_unsafe", f"{label} escapes or is not normalized: {value!r}")
    return value


def _validate_file_path(root: Path, relative: str, label: str) -> Path:
    relative = _normalize_relative_path(relative, label)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            _fail("symlink_forbidden", f"{label} traverses a symlink: {relative}")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError:
        _fail(
            "dependency_missing",
            f"{label} is missing: {relative}; finish the selected asset intake",
            not_ready=True,
        )
    root_resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        _fail("path_unsafe", f"{label} resolves outside the asset directory")
    if not current.is_file():
        _fail("dependency_invalid", f"{label} is not a regular file: {relative}")
    return current


def _validate_archive_members(path: Path) -> None:
    if path.suffix.casefold() != ".zip" or not zipfile.is_zipfile(path):
        _fail(
            "source_archive_invalid",
            "the preserved Poly Haven source must be a readable ZIP archive",
        )
    seen: set[str] = set()
    regular_count = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = info.filename
                normalized = _normalize_relative_path(
                    name[:-1] if name.endswith("/") else name,
                    "source archive member",
                )
                folded = normalized.casefold()
                if folded in seen:
                    _fail(
                        "source_archive_invalid",
                        f"source archive contains a duplicate member: {name!r}",
                    )
                seen.add(folded)
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    _fail(
                        "archive_symlink_forbidden",
                        f"source archive contains a symlink: {name!r}",
                    )
                if info.flag_bits & 0x1:
                    _fail(
                        "source_archive_invalid",
                        f"source archive contains an encrypted member: {name!r}",
                    )
                if _RETIRED_LINEAGE_MARKER.casefold() in folded:
                    _fail(
                        "retired_lineage",
                        "the source archive contains a retired BC017 asset marker",
                    )
                if not info.is_dir():
                    regular_count += 1
    except (OSError, zipfile.BadZipFile) as exc:
        _fail("source_archive_invalid", f"cannot inspect source archive: {exc}")
    if regular_count == 0:
        _fail("source_archive_invalid", "source archive contains no files")


def _load_json_file(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail("dependency_invalid", f"cannot read {label}: {exc}")
    if len(raw) > 4 * 1024 * 1024:
        _fail("schema_invalid", f"{label} exceeds the 4 MiB review boundary")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("schema_invalid", f"{label} is not valid UTF-8 JSON: {exc}")
    return _require_mapping(value, label)


def _validate_selected_source(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        provenance,
        {
            "title",
            "creator",
            "platform",
            "model_uuid",
            "source_url",
            "license",
            "license_url",
            "original_archive",
        },
        "provenance",
    )
    expected = {
        "title": SELECTED_ASSET_TITLE,
        "creator": SELECTED_ASSET_CREATOR,
        "platform": SELECTED_ASSET_PLATFORM,
        "model_uuid": SELECTED_ASSET_UUID,
        "source_url": SELECTED_ASSET_URL,
        "license": SELECTED_ASSET_LICENSE,
        "license_url": SELECTED_ASSET_LICENSE_URL,
    }
    mismatches = {
        key: (provenance.get(key), expected_value)
        for key, expected_value in expected.items()
        if provenance.get(key) != expected_value
    }
    if mismatches:
        code = "license_mismatch" if any(
            key.startswith("license") for key in mismatches
        ) else "selected_source_mismatch"
        _fail(
            code,
            "selected Metal Office Desk provenance differs: "
            f"{mismatches!r}",
        )
    archive = _require_mapping(provenance["original_archive"], "original_archive")
    _require_exact_keys(archive, {"path", "sha256", "size_bytes"}, "original_archive")
    return archive


def _validate_derived_asset(
    derived: Mapping[str, Any],
    source_sha256: str,
) -> Mapping[str, Any]:
    _require_exact_keys(
        derived,
        {
            "runtime_usd",
            "source_archive_sha256",
            "conversion_tool",
            "conversion_version",
            "clean_room_conversion",
            "no_procedural_geometry",
            "meters_per_unit",
            "up_axis",
        },
        "derived_asset",
    )
    if derived["source_archive_sha256"] != source_sha256:
        _fail(
            "source_binding_mismatch",
            "derived USD is not bound to the preserved original archive hash",
        )
    for key in ("conversion_tool", "conversion_version"):
        if not isinstance(derived[key], str) or not derived[key].strip():
            _fail("schema_invalid", f"derived_asset.{key} must be non-empty")
    _require_bool(
        derived["clean_room_conversion"],
        True,
        "derived_asset.clean_room_conversion",
        "clean_room_required",
    )
    _require_bool(
        derived["no_procedural_geometry"],
        True,
        "derived_asset.no_procedural_geometry",
        "procedural_forbidden",
    )
    if _require_finite(derived["meters_per_unit"], "meters_per_unit") != 1.0:
        _fail("unit_invalid", "derived USD must be authored in meters")
    if derived["up_axis"] != "Z":
        _fail("axis_invalid", "derived USD must use the Z-up runtime frame")
    runtime = _require_mapping(derived["runtime_usd"], "runtime_usd")
    _require_exact_keys(runtime, {"path", "sha256", "size_bytes"}, "runtime_usd")
    return runtime


def _validate_prim_path(value: Any, label: str) -> str:
    relative = _normalize_relative_path(value, label)
    if any(_USD_IDENTIFIER_RE.fullmatch(part) is None for part in relative.split("/")):
        _fail("articulation_invalid", f"{label} is not an exact USD prim path")
    return relative


def _validate_articulation(
    articulation: Mapping[str, Any],
) -> tuple[str, str, tuple[BC017DrawerManifest, BC017DrawerManifest]]:
    _require_exact_keys(
        articulation,
        {
            "fixed_base",
            "base_link_prim",
            "world_fixed_joint_prim",
            "independent_dofs",
            "mimic_or_coupled_joints",
            "drawer_links",
        },
        "articulation",
    )
    _require_bool(
        articulation["fixed_base"], True, "articulation.fixed_base", "fixed_base_required"
    )
    _require_bool(
        articulation["independent_dofs"],
        True,
        "articulation.independent_dofs",
        "articulation_not_independent",
    )
    coupled = _require_sequence(
        articulation["mimic_or_coupled_joints"],
        "articulation.mimic_or_coupled_joints",
    )
    if coupled:
        _fail(
            "articulation_not_independent",
            "BC017 drawer joints may not be mimic or coupled joints",
        )
    base_link = _validate_prim_path(articulation["base_link_prim"], "base_link_prim")
    fixed_joint = _validate_prim_path(
        articulation["world_fixed_joint_prim"], "world_fixed_joint_prim"
    )
    if base_link != BC017_BASE_LINK_PRIM or fixed_joint != BC017_WORLD_FIXED_JOINT_PRIM:
        _fail(
            "articulation_invalid",
            "clean-room base and fixed-joint prim names are not canonical",
        )

    entries = _require_sequence(articulation["drawer_links"], "drawer_links")
    if len(entries) != 2:
        _fail("articulation_invalid", "exactly two drawer links are required")
    expected = {
        "left_drawer": (BC017_LEFT_DRAWER_LINK_PRIM, BC017_LEFT_JOINT_NAME),
        "right_drawer": (BC017_RIGHT_DRAWER_LINK_PRIM, BC017_RIGHT_JOINT_NAME),
    }
    parsed: list[BC017DrawerManifest] = []
    seen_roles: set[str] = set()
    seen_links: set[str] = set()
    seen_joints: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"drawer_links[{index}]")
        _require_exact_keys(
            entry,
            {
                "role",
                "link_prim",
                "joint_prim",
                "joint_name",
                "joint_type",
                "axis",
                "closed_limit_m",
                "open_limit_m",
                "reset_position_m",
            },
            f"drawer_links[{index}]",
        )
        role = entry["role"]
        if role not in expected:
            _fail("articulation_invalid", f"unknown drawer role: {role!r}")
        link = _validate_prim_path(entry["link_prim"], f"{role}.link_prim")
        joint_prim = _validate_prim_path(entry["joint_prim"], f"{role}.joint_prim")
        joint_name = entry["joint_name"]
        if not isinstance(joint_name, str) or _USD_IDENTIFIER_RE.fullmatch(joint_name) is None:
            _fail("articulation_invalid", f"{role}.joint_name is not exact")
        expected_link, expected_joint = expected[role]
        if (
            link != expected_link
            or joint_prim != expected_joint
            or joint_name != expected_joint
            or PurePosixPath(joint_prim).name != joint_name
        ):
            _fail(
                "articulation_invalid",
                f"{role} does not use its canonical clean-room link/joint names",
            )
        if entry["joint_type"] != "prismatic" or entry["axis"] != "+X":
            _fail(
                "articulation_invalid",
                f"{role} must use a +X prismatic joint",
            )
        closed = _require_finite(entry["closed_limit_m"], f"{role}.closed_limit_m")
        opened = _require_finite(entry["open_limit_m"], f"{role}.open_limit_m")
        reset = _require_finite(entry["reset_position_m"], f"{role}.reset_position_m")
        if closed < 0.0 or opened > 1.0 or opened - closed < 0.02:
            _fail(
                "joint_limit_invalid",
                f"{role} limits must define at least 20 mm of positive travel in [0, 1] m",
            )
        if abs(reset - closed) > 1.0e-9:
            _fail(
                "joint_limit_invalid",
                f"{role} reset_position_m must equal the authored closed limit",
            )
        if role in seen_roles or link in seen_links or joint_prim in seen_joints:
            _fail(
                "articulation_duplicate",
                "drawer roles, links, and prismatic joints must be pairwise distinct",
            )
        seen_roles.add(role)
        seen_links.add(link)
        seen_joints.add(joint_prim)
        parsed.append(
            BC017DrawerManifest(
                role=str(role),
                link_prim=link,
                joint_prim=joint_prim,
                joint_name=joint_name,
                axis="+X",
                closed_limit_m=closed,
                open_limit_m=opened,
            )
        )
    if seen_roles != set(expected):
        _fail("articulation_invalid", "left and right drawer roles are both required")
    by_role = {drawer.role: drawer for drawer in parsed}
    return base_link, fixed_joint, (
        by_role["left_drawer"],
        by_role["right_drawer"],
    )


def _validate_pose(placement: Mapping[str, Any]) -> tuple[
    tuple[float, float, float], tuple[float, float, float, float]
]:
    _require_exact_keys(placement, {"translation_m", "orientation_wxyz"}, "placement")
    translation_raw = _require_sequence(placement["translation_m"], "translation_m")
    orientation_raw = _require_sequence(
        placement["orientation_wxyz"], "orientation_wxyz"
    )
    if len(translation_raw) != 3 or len(orientation_raw) != 4:
        _fail("placement_invalid", "placement requires 3D translation and WXYZ quaternion")
    translation = tuple(
        _require_finite(value, f"translation_m[{index}]")
        for index, value in enumerate(translation_raw)
    )
    orientation = tuple(
        _require_finite(value, f"orientation_wxyz[{index}]")
        for index, value in enumerate(orientation_raw)
    )
    norm = math.sqrt(sum(value * value for value in orientation))
    if abs(norm - 1.0) > 1.0e-6:
        _fail("placement_invalid", "orientation_wxyz must be a unit quaternion")
    return translation, orientation


def _validate_declared_files(
    root: Path,
    files_value: Any,
) -> tuple[dict[str, Mapping[str, Any]], tuple[Path, ...]]:
    entries = _require_sequence(files_value, "files")
    if not entries:
        _fail("closure_incomplete", "files must declare the complete asset closure")
    declared: dict[str, Mapping[str, Any]] = {}
    validated: list[Path] = []
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"files[{index}]")
        _require_exact_keys(
            entry,
            {"path", "role", "size_bytes", "sha256"},
            f"files[{index}]",
        )
        relative = _normalize_relative_path(entry["path"], f"files[{index}].path")
        if relative in declared:
            _fail("closure_duplicate", f"duplicate declared file: {relative}")
        role = entry["role"]
        if role not in _FILE_ROLES:
            _fail("schema_invalid", f"unknown file role for {relative}: {role!r}")
        expected_size = _require_size(entry["size_bytes"], f"{relative}.size_bytes")
        expected_hash = _require_sha256(entry["sha256"], f"{relative}.sha256")
        path = _validate_file_path(root, relative, f"declared file {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            _fail(
                "size_mismatch",
                f"{relative} size differs: expected={expected_size} actual={actual_size}",
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            _fail(
                "hash_mismatch",
                f"{relative} SHA-256 differs: expected={expected_hash} actual={actual_hash}",
            )
        declared[relative] = entry
        validated.append(path)

    actual: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            _fail("symlink_forbidden", f"asset closure contains a symlink: {relative}")
        if candidate.is_file():
            if relative != BC017_MANIFEST_FILENAME:
                actual.add(relative)
        elif not candidate.is_dir():
            _fail("dependency_invalid", f"asset closure contains a special file: {relative}")
    if actual != set(declared):
        _fail(
            "closure_incomplete",
            f"declared files differ from disk: missing={sorted(set(declared) - actual)!r}, "
            f"undeclared={sorted(actual - set(declared))!r}",
        )
    return declared, tuple(validated)


def _bind_declared_file(
    declared: Mapping[str, Mapping[str, Any]],
    descriptor: Mapping[str, Any],
    expected_role: str,
    label: str,
) -> tuple[str, str, int]:
    relative = _normalize_relative_path(descriptor["path"], f"{label}.path")
    digest = _require_sha256(descriptor["sha256"], f"{label}.sha256")
    size = _require_size(descriptor["size_bytes"], f"{label}.size_bytes")
    entry = declared.get(relative)
    if entry is None:
        _fail("closure_incomplete", f"{label} is not present in files: {relative}")
    if (
        entry["role"] != expected_role
        or entry["sha256"] != digest
        or entry["size_bytes"] != size
    ):
        _fail("source_binding_mismatch", f"{label} differs from its files entry")
    return relative, digest, size


def _validate_evidence(
    *,
    root: Path,
    closure: Mapping[str, Any],
    declared: Mapping[str, Mapping[str, Any]],
    source_sha256: str,
    runtime_sha256: str,
    drawers: tuple[BC017DrawerManifest, BC017DrawerManifest],
    base_link: str,
) -> tuple[int, int, int]:
    _require_exact_keys(
        closure,
        {
            "declared_files_complete",
            "unresolved_references",
            "external_absolute_references",
            "material_evidence",
            "collision_evidence",
        },
        "closure",
    )
    _require_bool(
        closure["declared_files_complete"],
        True,
        "closure.declared_files_complete",
        "closure_incomplete",
    )
    if _require_sequence(closure["unresolved_references"], "unresolved_references"):
        _fail("material_closure_invalid", "unresolved USD references must be empty")
    if _require_sequence(
        closure["external_absolute_references"], "external_absolute_references"
    ):
        _fail("material_closure_invalid", "absolute external references are forbidden")

    material = _require_mapping(closure["material_evidence"], "material_evidence")
    _require_exact_keys(material, {"report_path", "dependency_paths"}, "material_evidence")
    material_report_path = _normalize_relative_path(
        material["report_path"], "material_evidence.report_path"
    )
    material_entry = declared.get(material_report_path)
    if material_entry is None or material_entry["role"] != "material_audit":
        _fail("material_closure_invalid", "material audit report is not declared")
    dependency_values = _require_sequence(
        material["dependency_paths"], "material_evidence.dependency_paths"
    )
    dependency_paths = tuple(
        _normalize_relative_path(value, "material dependency")
        for value in dependency_values
    )
    if not dependency_paths or len(set(dependency_paths)) != len(dependency_paths):
        _fail(
            "material_closure_invalid",
            "material dependency paths must be non-empty and unique",
        )
    for relative in dependency_paths:
        entry = declared.get(relative)
        if entry is None or entry["role"] not in {"material", "texture"}:
            _fail(
                "material_closure_invalid",
                f"material dependency is missing or has the wrong role: {relative}",
            )
    material_report = _load_json_file(root / material_report_path, "material report")
    _require_exact_keys(
        material_report,
        {
            "schema",
            "runtime_usd_sha256",
            "dependency_paths",
            "unresolved_references",
            "external_absolute_references",
        },
        "material report",
    )
    if (
        material_report["schema"] != _MATERIAL_REPORT_SCHEMA
        or material_report["runtime_usd_sha256"] != runtime_sha256
        or tuple(material_report["dependency_paths"]) != dependency_paths
        or material_report["unresolved_references"] != []
        or material_report["external_absolute_references"] != []
    ):
        _fail(
            "material_closure_invalid",
            "material report is incomplete or not bound to this runtime USD",
        )

    collision = _require_mapping(closure["collision_evidence"], "collision_evidence")
    _require_exact_keys(
        collision,
        {
            "report_path",
            "derived_from_selected_visible_asset",
            "procedural_colliders",
            "closed_reset_penetration_free",
            "collider_prims",
        },
        "collision_evidence",
    )
    _require_bool(
        collision["derived_from_selected_visible_asset"],
        True,
        "collision_evidence.derived_from_selected_visible_asset",
        "collision_evidence_invalid",
    )
    _require_bool(
        collision["procedural_colliders"],
        False,
        "collision_evidence.procedural_colliders",
        "procedural_forbidden",
    )
    _require_bool(
        collision["closed_reset_penetration_free"],
        True,
        "collision_evidence.closed_reset_penetration_free",
        "collision_evidence_invalid",
    )
    collision_report_path = _normalize_relative_path(
        collision["report_path"], "collision_evidence.report_path"
    )
    collision_entry = declared.get(collision_report_path)
    if collision_entry is None or collision_entry["role"] != "collision_audit":
        _fail("collision_evidence_invalid", "collision audit report is not declared")
    collider_prims_raw = _require_mapping(
        collision["collider_prims"], "collision_evidence.collider_prims"
    )
    expected_roles = {base_link, *(drawer.link_prim for drawer in drawers)}
    if set(collider_prims_raw) != expected_roles:
        _fail(
            "collision_evidence_invalid",
            "collider evidence must cover the base and both drawer links",
        )
    collider_prims: dict[str, list[str]] = {}
    all_collider_paths: list[str] = []
    for link_prim, raw_paths in collider_prims_raw.items():
        paths = [
            _validate_prim_path(value, f"collider_prims[{link_prim}]")
            for value in _require_sequence(raw_paths, f"collider_prims[{link_prim}]")
        ]
        if not paths or any(
            path != link_prim and not path.startswith(f"{link_prim}/")
            for path in paths
        ):
            _fail(
                "collision_evidence_invalid",
                f"colliders are absent or not owned by link {link_prim}",
            )
        collider_prims[link_prim] = paths
        all_collider_paths.extend(paths)
    if len(set(all_collider_paths)) != len(all_collider_paths):
        _fail("collision_evidence_invalid", "collider prim paths must be unique")

    collision_report = _load_json_file(root / collision_report_path, "collision report")
    _require_exact_keys(
        collision_report,
        {
            "schema",
            "source_archive_sha256",
            "runtime_usd_sha256",
            "drawer_links",
            "collider_prims",
            "derived_from_visible_geometry",
            "procedural_geometry",
            "fixed_base_verified",
            "all_links_have_colliders",
            "closed_reset_max_penetration_m",
            "support_gap_m",
        },
        "collision report",
    )
    expected_drawer_links = {drawer.role: drawer.link_prim for drawer in drawers}
    penetration = _require_finite(
        collision_report["closed_reset_max_penetration_m"],
        "closed_reset_max_penetration_m",
    )
    support_gap = _require_finite(collision_report["support_gap_m"], "support_gap_m")
    if (
        collision_report["schema"] != _COLLISION_REPORT_SCHEMA
        or collision_report["source_archive_sha256"] != source_sha256
        or collision_report["runtime_usd_sha256"] != runtime_sha256
        or collision_report["drawer_links"] != expected_drawer_links
        or collision_report["collider_prims"] != collider_prims
        or collision_report["derived_from_visible_geometry"] is not True
        or collision_report["procedural_geometry"] is not False
        or collision_report["fixed_base_verified"] is not True
        or collision_report["all_links_have_colliders"] is not True
        or penetration < 0.0
        or penetration > 5.0e-4
        or support_gap < 0.0
        or support_gap > 5.0e-3
    ):
        _fail(
            "collision_evidence_invalid",
            "collision report is incomplete, unbound, penetrating, or unsupported",
        )
    return (
        len(collider_prims[base_link]),
        len(collider_prims[drawers[0].link_prim]),
        len(collider_prims[drawers[1].link_prim]),
    )


def validate_bc017_asset_manifest(
    asset_directory: str | Path = BC017_ASSET_DIRECTORY,
) -> BC017BytePreflightReport:
    """Validate the selected asset and every derived byte without Isaac/pxr.

    This function is intentionally the first operation performed by the spawn
    boundary.  A caller can therefore run it in a normal Python process and
    safely refuse collection before any USD stage mutation.
    """

    root = Path(asset_directory)
    if root.is_symlink():
        _fail("symlink_forbidden", "the BC017 asset directory may not be a symlink")
    if not root.is_dir():
        _fail(
            "manifest_missing",
            "download the operator-selected Metal Office Desk source, "
            "complete the clean-room conversion, and create "
            f"{root / BC017_MANIFEST_FILENAME}",
            not_ready=True,
        )
    manifest_path = root / BC017_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        _fail("symlink_forbidden", "the BC017 asset manifest may not be a symlink")
    if not manifest_path.is_file():
        _fail(
            "manifest_missing",
            "download the operator-selected Metal Office Desk source, "
            "complete the clean-room conversion, and create "
            f"{manifest_path}",
            not_ready=True,
        )
    manifest = _load_json_file(manifest_path, "BC017 asset manifest")
    _reject_retired_lineage(manifest)
    _require_exact_keys(
        manifest,
        {
            "schema",
            "schema_version",
            "task",
            "clean_room",
            "provenance",
            "derived_asset",
            "articulation",
            "placement",
            "closure",
            "files",
        },
        "manifest",
    )
    if manifest["schema"] != _MANIFEST_SCHEMA or manifest["schema_version"] != 1:
        _fail("schema_invalid", "unsupported BC017 asset manifest schema/version")
    task = _require_mapping(manifest["task"], "task")
    _require_exact_keys(task, {"id", "name", "requirement"}, "task")
    if task != {"id": TASK_ID, "name": TASK_NAME, "requirement": TASK_REQUIREMENT}:
        _fail("task_identity_mismatch", "manifest is not for the exact BC017 task")
    _require_bool(manifest["clean_room"], True, "clean_room", "clean_room_required")

    provenance = _require_mapping(manifest["provenance"], "provenance")
    archive_descriptor = _validate_selected_source(provenance)
    source_sha256 = _require_sha256(
        archive_descriptor["sha256"], "original_archive.sha256"
    )
    derived = _require_mapping(manifest["derived_asset"], "derived_asset")
    runtime_descriptor = _validate_derived_asset(derived, source_sha256)
    runtime_sha256 = _require_sha256(runtime_descriptor["sha256"], "runtime_usd.sha256")
    if runtime_sha256 == source_sha256:
        _fail("source_binding_mismatch", "source archive and derived USD hashes collide")

    articulation = _require_mapping(manifest["articulation"], "articulation")
    base_link, fixed_joint, drawers = _validate_articulation(articulation)
    placement = _require_mapping(manifest["placement"], "placement")
    translation, orientation = _validate_pose(placement)
    declared, validated_files = _validate_declared_files(root, manifest["files"])

    archive_relative, _, _ = _bind_declared_file(
        declared,
        archive_descriptor,
        "source_archive",
        "original_archive",
    )
    runtime_relative, _, _ = _bind_declared_file(
        declared,
        runtime_descriptor,
        "runtime_usd",
        "runtime_usd",
    )
    runtime_suffix = Path(runtime_relative).suffix.casefold()
    if runtime_suffix not in {".usd", ".usda", ".usdc"}:
        _fail("runtime_usd_invalid", "derived runtime asset must be USD/USDA/USDC")
    source_path = _validate_file_path(root, archive_relative, "original archive")
    runtime_path = _validate_file_path(root, runtime_relative, "runtime USD")
    _validate_archive_members(source_path)

    roles = [entry["role"] for entry in declared.values()]
    if roles.count("source_archive") != 1 or roles.count("runtime_usd") != 1:
        _fail(
            "closure_incomplete",
            "the closure must contain exactly one source archive and one runtime USD",
        )
    closure = _require_mapping(manifest["closure"], "closure")
    collider_counts = _validate_evidence(
        root=root,
        closure=closure,
        declared=declared,
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
        drawers=drawers,
        base_link=base_link,
    )
    return BC017BytePreflightReport(
        asset_directory=root.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256(manifest_path),
        source_archive_path=source_path.resolve(),
        source_archive_sha256=source_sha256,
        runtime_usd_path=runtime_path.resolve(),
        runtime_usd_sha256=runtime_sha256,
        base_link_prim=base_link,
        world_fixed_joint_prim=fixed_joint,
        drawers=drawers,
        translation_m=translation,
        orientation_wxyz=orientation,
        collider_counts=collider_counts,
        validated_files=tuple(path.resolve() for path in validated_files),
    )


def validate_bc017_asset_bundle(
    root: str | Path = BC017_ASSET_DIRECTORY,
) -> BC017BytePreflightReport:
    """Stable env-facing alias for the complete pure-Python asset report."""

    return validate_bc017_asset_manifest(root)


def _composed_path(relative_prim: str) -> str:
    return f"{BC017_ASSET_ROOT_PATH}/{relative_prim}"


def bc017_asset_point_world(
    point_local: Sequence[float],
) -> tuple[float, float, float]:
    """Transform a local asset point by the canonical BC017 placement."""

    if len(point_local) != 3:
        raise ValueError("BC017 local point must contain exactly three values")
    x, y, z = (
        _require_finite(value, f"point_local[{index}]")
        for index, value in enumerate(point_local)
    )
    tx, ty, tz = BC017_ASSET_TRANSLATION_M
    return tx - x, ty + z, tz + y


def bc017_handle_grip_bounds_world(
    side: str,
    drawer_q_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the strict world-space pull-tab bounds for one live drawer."""

    normalized_side = str(side).lower()
    if normalized_side not in BC017_HANDLE_GRIP_LOCAL_BOUNDS_M:
        raise ValueError(f"unsupported BC017 hand side: {side!r}")
    q_m = _require_finite(drawer_q_m, "drawer_q_m")
    if q_m < 0.0 or q_m > BC017_DRAWER_TRAVEL_M:
        raise ValueError("BC017 drawer_q_m is outside the authored travel")
    local_min, local_max = BC017_HANDLE_GRIP_LOCAL_BOUNDS_M[normalized_side]
    corners = tuple(
        bc017_asset_point_world((x + q_m, y, z))
        for x in (local_min[0], local_max[0])
        for y in (local_min[1], local_max[1])
        for z in (local_min[2], local_max[2])
    )
    return (
        tuple(min(point[index] for point in corners) for index in range(3)),
        tuple(max(point[index] for point in corners) for index in range(3)),
    )


def bc017_grasp_joint_path(side: str, finger_body_name: str) -> str:
    """Return one pre-authored assigned fingertip-to-drawer D6 joint path."""

    normalized_side = str(side).lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"unsupported BC017 hand side: {side!r}")
    expected = {
        f"{normalized_side}_finger{finger}_link4"
        for finger in range(1, 6)
    }
    body_name = str(finger_body_name)
    if body_name not in expected:
        raise ValueError(
            "BC017 grasp joint requires an assigned distal link4 body"
        )
    return f"{BC017_GRASP_JOINT_ROOT_PATH}/{body_name}_to_drawer"


def author_bc017_fingertip_handle_joints(
    stage: Any,
    *,
    drawer_paths: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Pre-author free D6 joints whose six axes can lock without resync."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, BC017_GRASP_JOINT_ROOT_PATH).GetPrim()
    root.SetCustomDataByKey("handson:fingertipHandleJointBank", True)
    root.SetCustomDataByKey("handson:runtimeTopologyStable", True)
    authored: dict[str, tuple[str, ...]] = {}
    for side in ("left", "right"):
        drawer_path = str(drawer_paths.get(side, ""))
        drawer = stage.GetPrimAtPath(drawer_path)
        if not drawer.IsValid() or not drawer.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"BC017 {side} drawer cannot bind fingertip joints: {drawer_path}"
            )
        robot_root_path = (
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        robot_root = stage.GetPrimAtPath(robot_root_path)
        if not robot_root.IsValid():
            raise RuntimeError(
                f"BC017 {side} robot root is missing: {robot_root_path}"
            )
        expected_body_names = {
            f"{side}_finger{finger}_link4" for finger in range(1, 6)
        }
        body_paths_by_name: dict[str, list[str]] = {
            name: [] for name in expected_body_names
        }
        for prim in Usd.PrimRange(robot_root):
            name = str(prim.GetName())
            if (
                name in expected_body_names
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                body_paths_by_name[name].append(str(prim.GetPath()))
        side_paths: list[str] = []
        for finger in range(1, 6):
            body_name = f"{side}_finger{finger}_link4"
            candidates = body_paths_by_name[body_name]
            if len(candidates) != 1:
                raise RuntimeError(
                    "BC017 requires exactly one assigned distal fingertip "
                    f"rigid body named {body_name!r}; observed={candidates}"
                )
            body_path = candidates[0]
            path = bc017_grasp_joint_path(side, body_name)
            joint = UsdPhysics.Joint.Define(stage, path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(drawer_path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateJointEnabledAttr().Set(True)
            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateExcludeFromArticulationAttr().Set(True)
            prim = joint.GetPrim()
            prim.SetCustomDataByKey("handson:assignedHand", side)
            prim.SetCustomDataByKey("handson:assignedDrawer", side)
            prim.SetCustomDataByKey("handson:distalFingerBody", body_name)
            prim.SetCustomDataByKey("handson:runtimeLockMode", "all_six_d6_axes")
            prim.SetCustomDataByKey("handson:initiallyFree", True)
            # Keep every D6 axis in PhysX's limited mode for the lifetime of
            # the scene.  The very large initial ranges are practically free;
            # strict contact only narrows these existing numeric limits, so no
            # runtime axis-mode/topology change or articulation merge occurs.
            for axis in BC017_GRASP_D6_AXES:
                limit = UsdPhysics.LimitAPI.Apply(prim, axis)
                free_limit = BC017_GRASP_FREE_LIMIT_BY_AXIS[axis]
                limit.CreateLowAttr().Set(-free_limit)
                limit.CreateHighAttr().Set(free_limit)
            side_paths.append(path)
        authored[side] = tuple(side_paths)
    return authored


def _spawn_bc017_target_edge_highlight(
    stage: Any,
    sim_utils: Any,
    *,
    side: str,
) -> tuple[str, ...]:
    """Outline one assigned drawer front with render-only emissive bars."""

    from pxr import UsdGeom, UsdPhysics

    normalized_side = str(side).lower()
    if normalized_side not in _BC017_DRAWER_FRONT_LOCAL_BOUNDS_M:
        raise ValueError(f"unsupported BC017 highlight side: {side!r}")
    root_path = BC017_TARGET_EDGE_HIGHLIGHT_ROOTS[normalized_side]
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    root.SetCustomDataByKey("handson:assignedHand", normalized_side)
    root.SetCustomDataByKey("handson:targetDrawerEdgeHighlight", True)
    root.SetCustomDataByKey("handson:visualOnly", True)

    color = BC017_TARGET_EDGE_COLORS_RGB[normalized_side]
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=tuple(min(1.0, 0.88 * value) for value in color),
        opacity=1.0,
        roughness=0.22,
    )
    (y_min, z_min), (y_max, z_max) = (
        _BC017_DRAWER_FRONT_LOCAL_BOUNDS_M[normalized_side]
    )
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    height = y_max - y_min
    width = z_max - z_min
    edge_specs = (
        (
            "top",
            (_BC017_TARGET_EDGE_DEPTH_M, _BC017_TARGET_EDGE_WIDTH_M, width),
            (_BC017_TARGET_EDGE_FRONT_X_M, y_max, z_center),
        ),
        (
            "bottom",
            (_BC017_TARGET_EDGE_DEPTH_M, _BC017_TARGET_EDGE_WIDTH_M, width),
            (_BC017_TARGET_EDGE_FRONT_X_M, y_min, z_center),
        ),
        (
            "outer",
            (_BC017_TARGET_EDGE_DEPTH_M, height, _BC017_TARGET_EDGE_WIDTH_M),
            (_BC017_TARGET_EDGE_FRONT_X_M, y_center, z_max),
        ),
        (
            "inner",
            (_BC017_TARGET_EDGE_DEPTH_M, height, _BC017_TARGET_EDGE_WIDTH_M),
            (_BC017_TARGET_EDGE_FRONT_X_M, y_center, z_min),
        ),
    )
    paths: list[str] = []
    for edge_name, size, translation in edge_specs:
        path = f"{root_path}/{edge_name}"
        cfg = sim_utils.CuboidCfg(size=size, visual_material=material)
        cfg.func(path, cfg, translation=translation)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"BC017 target edge highlight is missing: {path}")
        prim.SetCustomDataByKey("handson:assignedHand", normalized_side)
        prim.SetCustomDataByKey("handson:targetDrawerEdgeHighlight", True)
        prim.SetCustomDataByKey("handson:visualOnly", True)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(
                f"BC017 visual target edge unexpectedly has collision: {path}"
            )
        paths.append(path)
    return tuple(paths)


def spawn_bc017_task_assets(
    stage: Any,
    sim_utils: Any,
    *,
    asset_directory: str | Path = BC017_ASSET_DIRECTORY,
) -> B320SpawnResult:
    """Compose the selected Metal Office Desk task asset."""

    runtime_usd_path = (
        Path(asset_directory) / BC017_RUNTIME_USD_RELATIVE_PATH
    ).resolve()
    if not runtime_usd_path.is_file():
        raise FileNotFoundError(
            f"B320-BC017 runtime USD is missing: {runtime_usd_path}"
        )

    from .b320_bimanual_scene import B320SpawnResult, _spawn_worktable
    from .compat import quat_wxyz_to_isaaclab
    from pxr import PhysxSchema, UsdPhysics

    table_path = _spawn_worktable(stage, sim_utils, task_name=TASK_NAME)
    cfg = sim_utils.UsdFileCfg(usd_path=str(runtime_usd_path))
    cfg.func(
        BC017_ASSET_ROOT_PATH,
        cfg,
        translation=BC017_ASSET_TRANSLATION_M,
        orientation=quat_wxyz_to_isaaclab(BC017_ASSET_ORIENTATION_WXYZ),
    )
    root = stage.GetPrimAtPath(BC017_ASSET_ROOT_PATH)
    if not root.IsValid():
        raise RuntimeError("B320-BC017 USD did not compose its root prim")

    base_path = _composed_path(BC017_BASE_LINK_PRIM)
    fixed_joint_path = _composed_path(BC017_WORLD_FIXED_JOINT_PRIM)
    base = stage.GetPrimAtPath(base_path)
    fixed_joint = UsdPhysics.FixedJoint.Get(stage, fixed_joint_path).GetPrim()
    if not base.IsValid() or not fixed_joint.IsValid():
        raise RuntimeError("B320-BC017 composed USD is missing its fixed-base topology")

    drawer_paths: list[str] = []
    joint_paths: list[str] = []
    drawers = (
        ("left_drawer", BC017_LEFT_DRAWER_LINK_PRIM, BC017_LEFT_JOINT_NAME),
        ("right_drawer", BC017_RIGHT_DRAWER_LINK_PRIM, BC017_RIGHT_JOINT_NAME),
    )
    for role, link_prim, joint_name in drawers:
        drawer_path = _composed_path(link_prim)
        joint_path = _composed_path(joint_name)
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        joint_prim = UsdPhysics.PrismaticJoint.Get(stage, joint_path).GetPrim()
        if not drawer_prim.IsValid() or not joint_prim.IsValid():
            raise RuntimeError(
                f"B320-BC017 composed USD is missing {role} topology"
            )
        report = (
            PhysxSchema.PhysxContactReportAPI(drawer_prim)
            if drawer_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            else PhysxSchema.PhysxContactReportAPI.Apply(drawer_prim)
        )
        report.CreateThresholdAttr().Set(0.0)
        drawer_paths.append(drawer_path)
        joint_paths.append(joint_path)

    highlight_paths = {
        side: _spawn_bc017_target_edge_highlight(
            stage,
            sim_utils,
            side=side,
        )
        for side in ("left", "right")
    }

    root.SetCustomDataByKey("handson:task", TASK_NAME)
    root.SetCustomDataByKey("handson:taskId", TASK_ID)
    root.SetCustomDataByKey("handson:assetTitle", SELECTED_ASSET_TITLE)
    root.SetCustomDataByKey("handson:assetCreator", SELECTED_ASSET_CREATOR)
    root.SetCustomDataByKey("handson:assetUUID", SELECTED_ASSET_UUID)
    root.SetCustomDataByKey("handson:assetLicense", SELECTED_ASSET_LICENSE)
    root.SetCustomDataByKey(
        "handson:assignedTargetEdgeHighlights",
        json.dumps(highlight_paths, sort_keys=True, separators=(",", ":")),
    )

    result = B320SpawnResult(
        body_paths={
            "fixture": base_path,
            "left_drawer": drawer_paths[0],
            "right_drawer": drawer_paths[1],
        },
        articulation_root_paths=(fixed_joint_path,),
        static_paths=(table_path,),
        dof_names=(BC017_LEFT_JOINT_NAME, BC017_RIGHT_JOINT_NAME),
        collider_counts={
            "fixture": 7,
            "left_drawer": 1,
            "right_drawer": 1,
        },
    )
    return result


__all__ = [
    "BC017_ASSET_DIRECTORY",
    "BC017_ASSET_ORIENTATION_WXYZ",
    "BC017_ASSET_ROOT_PATH",
    "BC017_ASSET_TRANSLATION_M",
    "BC017_BASE_LINK_PRIM",
    "BC017_DRAWER_TRAVEL_M",
    "BC017_GRASP_D6_AXES",
    "BC017_GRASP_FREE_LIMIT_BY_AXIS",
    "BC017_GRASP_JOINT_ROOT_PATH",
    "BC017_GRASP_LOCK_LIMIT_BY_AXIS",
    "BC017_HANDLE_GRIP_LOCAL_BOUNDS_M",
    "BC017_LEFT_DRAWER_LINK_PRIM",
    "BC017_LEFT_JOINT_NAME",
    "BC017_MANIFEST_FILENAME",
    "BC017_MIN_USABLE_PULL_FRACTION",
    "BC017_RIGHT_DRAWER_LINK_PRIM",
    "BC017_RIGHT_JOINT_NAME",
    "BC017_TARGET_EDGE_COLORS_RGB",
    "BC017_TARGET_EDGE_HIGHLIGHT_ROOTS",
    "BC017_WORLD_FIXED_JOINT_PRIM",
    "BC017AssetGateError",
    "BC017AssetNotReadyError",
    "BC017DrawerManifest",
    "BC017BytePreflightReport",
    "BC017ValidatedAssetManifest",
    "SELECTED_ASSET_CREATOR",
    "SELECTED_ASSET_LICENSE",
    "SELECTED_ASSET_LICENSE_URL",
    "SELECTED_ASSET_PLATFORM",
    "SELECTED_ASSET_TITLE",
    "SELECTED_ASSET_URL",
    "SELECTED_ASSET_UUID",
    "TASK_ID",
    "TASK_NAME",
    "TASK_REQUIREMENT",
    "bc017_asset_point_world",
    "author_bc017_fingertip_handle_joints",
    "bc017_grasp_joint_path",
    "bc017_handle_grip_bounds_world",
    "spawn_bc017_task_assets",
    "validate_bc017_asset_bundle",
    "validate_bc017_asset_manifest",
]
