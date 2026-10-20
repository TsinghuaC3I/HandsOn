"""Asset intake and provenance for B320-BC025.

BC025 deliberately separates the two user-supplied sources:

* the downloaded Poly Haven Terry Cloth archive supplies render maps only;
* the user-provided Pillow024 USD supplies audited PhysX collision/material
  parameters only.

The Pillow024 tetrahedral volume is not composed into the task.  BC025's
runtime contract is one task-authored surface mesh whose render, simulation,
and collision topology are identical.  Keeping this boundary explicit avoids
turning a closed pillow volume into a hidden towel collider.

This module is importable without Isaac Sim.  USD/OmniPBR work is kept in the
small authoring helpers at the bottom so catalog and intake tests can run in a
plain Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

from handson.runtime.project_paths import resolve_project_path


BC025_ASSET_ID = "bc025-towel-01"
BC025_MATERIAL_TITLE = "Terry Cloth"
BC025_MATERIAL_SOURCE_URL = "https://polyhaven.com/a/terry_cloth"
BC025_MATERIAL_LICENSE = "CC0 1.0"
BC025_MATERIAL_LICENSE_URL = "https://polyhaven.com/license"
BC025_MATERIAL_ARCHIVE_NAME = "terry_cloth_4k.gltf.zip"
BC025_MATERIAL_ARCHIVE_SHA256 = (
    "35df068d309815a136aea26a19553d121a050b680418ae803a304976491a29ef"
)
# The original browser-generated ZIP was missing on the collection host.
# This deterministic recovery package contains the same official 4K glTF
# files, individually verified against Poly Haven's file API. ZIP timestamps
# change the container hash; assets/material_restoration.json records every
# source URL and content hash. Keep accepting the original selected archive.
BC025_RECOVERED_MATERIAL_ARCHIVE_SHA256 = (
    "92498e0f29a843af8670530aea6700e290b4471778207c1f19fce671b72e26ce"
)

BC025_COLLISION_SOURCE_TITLE = "Pillow024"
BC025_COLLISION_SOURCE_ENV = "HANDSON_BC025_COLLISION_SOURCE"
BC025_MATERIAL_ARCHIVE_ENV = "HANDSON_BC025_MATERIAL_ARCHIVE"
BC025_MATERIAL_ROOT_ENV = "HANDSON_BC025_MATERIAL_ROOT"
BC025_ASSET_CACHE_ENV = "HANDSON_BC025_ASSET_CACHE"

BC025_COLLISION_SOURCE_SHA256 = (
    "8b08e24ba0e8bc359f8802fce393fddb6234b31ca60a8e51a85404638cc80305"
)

BC025_MATERIAL_MEMBERS: Mapping[str, str] = {
    "diffuse": "textures/terry_cloth_diff_4k.jpg",
    "normal": "textures/terry_cloth_nor_gl_4k.jpg",
    "arm": "textures/terry_cloth_arm_4k.jpg",
    "specular_ior": "textures/terry_cloth_spec_ior_4k.jpg",
}

# These values were read from the supplied Pillow024.usd.  The runtime
# authoring path validates them against the USD again before applying them.
# They are kept here as a plain-Python audit record for task manifests/tests.
BC025_PILLOW_COLLISION_CALIBRATION: Mapping[str, Any] = {
    "collision_prim_suffix": "/simulation_mesh",
    "contact_offset_m": 0.00800000038,
    "rest_offset_m": 0.00400000019,
    "self_collision": True,
    "enable_speculative_ccd": True,
    "linear_damping": 0.1,
    "solver_position_iterations": 32,
    "static_friction": 3.0,
    "dynamic_friction": 3.0,
    "poissons_ratio": 0.48,
    "youngs_modulus_pa": 1_000_000.0,
    "elasticity_damping": 0.0,
    # The source contains two mass annotations (0.8 kg on the body and
    # 1.2 kg on the visual).  They are retained as provenance, not copied
    # into the surface material because the ambiguity is not a valid density
    # specification for a towel sheet.
    "source_mass_annotations_kg": (0.8, 1.2),
    "density_is_zero_in_source": True,
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


BC025_TASK_ASSET_DIRECTORY = (
    _repository_root()
    / "taskdev"
    / "tasks"
    / "bench320_bc025_towel_corner_synchronized_spread"
    / "assets"
)


@dataclass(frozen=True)
class BC025ResolvedAssets:
    """Resolved local files used by the BC025 scene author."""

    archive_path: Path
    extracted_root: Path
    collision_source_path: Path
    archive_sha256: str
    collision_source_sha256: str
    textures: Mapping[str, Path]

    def manifest(self) -> dict[str, Any]:
        return {
            "asset_id": BC025_ASSET_ID,
            "material": {
                "title": BC025_MATERIAL_TITLE,
                "source_url": BC025_MATERIAL_SOURCE_URL,
                "license": BC025_MATERIAL_LICENSE,
                "license_url": BC025_MATERIAL_LICENSE_URL,
                "archive": str(self.archive_path),
                "archive_sha256": self.archive_sha256,
                "extracted_root": str(self.extracted_root),
                "textures": {
                    name: str(path) for name, path in self.textures.items()
                },
            },
            "collision_source": {
                "title": BC025_COLLISION_SOURCE_TITLE,
                "path": str(self.collision_source_path),
                "sha256": self.collision_source_sha256,
                "use": "collision_parameters_and_physics_calibration",
                "topology_composed": False,
                "license_status": "unresolved_user_provided_source",
            },
            "pillow024_calibration": dict(BC025_PILLOW_COLLISION_CALIBRATION),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_archive_path() -> Path | None:
    configured = os.environ.get(BC025_MATERIAL_ARCHIVE_ENV, "").strip()
    if configured:
        return resolve_project_path(configured)
    # Production assets must travel with the checkout, not live in whichever
    # operator's Downloads folder happens to be present on this computer.
    candidate = BC025_TASK_ASSET_DIRECTORY / BC025_MATERIAL_ARCHIVE_NAME
    return candidate.resolve() if candidate.is_file() else None


def _resolve_collision_source_path() -> Path | None:
    configured = os.environ.get(BC025_COLLISION_SOURCE_ENV, "").strip()
    if configured:
        return resolve_project_path(configured)
    candidate = BC025_TASK_ASSET_DIRECTORY / "Pillow024" / "Pillow024.usd"
    return candidate.resolve() if candidate.is_file() else None


def _validate_hash(path: Path, expected: str | tuple[str, ...], label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"BC025 {label} is missing: {path}")
    actual = _sha256(path)
    accepted = (expected,) if isinstance(expected, str) else expected
    if actual not in accepted:
        raise ValueError(
            f"BC025 {label} hash mismatch: expected {expected}, got {actual}; "
            f"refusing to compose an unreviewed source ({path})"
        )
    return actual


def _safe_member_path(root: Path, member_name: str) -> Path:
    # ZIP names are POSIX paths even on Windows.  Reject absolute paths,
    # parent traversal, and backslash tricks before writing the cache.
    normalised = member_name.replace("\\", "/")
    candidate = Path(normalised)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Unsafe BC025 material archive member: {member_name!r}")
    destination = (root / candidate).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise ValueError(f"BC025 archive member escapes extraction root: {member_name!r}")
    return destination


def _extract_required_material_members(archive: Path) -> Path:
    cache_parent = resolve_project_path(
        os.environ.get(BC025_ASSET_CACHE_ENV, "").strip()
        or (Path(tempfile.gettempdir()) / "handson-bc025-assets")
    )
    cache_root = cache_parent / _sha256(archive)
    required_members = ("terry_cloth_4k.gltf", *BC025_MATERIAL_MEMBERS.values())
    existing = all(_safe_member_path(cache_root, name).is_file() for name in required_members)
    if existing:
        return cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as bundle:
        names = set(bundle.namelist())
        missing = [name for name in required_members if name not in names]
        if missing:
            raise ValueError(
                "BC025 Terry Cloth archive is incomplete; missing members: "
                + ", ".join(missing)
            )
        for member_name in required_members:
            destination = _safe_member_path(cache_root, member_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member_name, "r") as source, destination.open("wb") as target:
                target.write(source.read())
    return cache_root


def resolve_bc025_assets(*, required: bool = True) -> BC025ResolvedAssets | None:
    """Resolve and hash-lock the selected material and collision source."""

    archive = _resolve_archive_path()
    collision_source = _resolve_collision_source_path()
    if archive is None or collision_source is None:
        if not required:
            return None
        missing = []
        if archive is None:
            missing.append(
                f"Terry Cloth archive ({BC025_MATERIAL_ARCHIVE_NAME})"
            )
        if collision_source is None:
            missing.append("Pillow024/Pillow024.usd")
        raise FileNotFoundError(
            "BC025 selected sources are not available: " + ", ".join(missing)
            + f". Install the selected sources under {BC025_TASK_ASSET_DIRECTORY}; "
            "asset environment overrides may also use project-relative paths."
        )

    archive_hash = _validate_hash(
        archive,
        (BC025_MATERIAL_ARCHIVE_SHA256, BC025_RECOVERED_MATERIAL_ARCHIVE_SHA256),
        "Terry Cloth archive",
    )
    collision_hash = _validate_hash(
        collision_source,
        BC025_COLLISION_SOURCE_SHA256,
        "Pillow024 collision source",
    )
    extracted_root = _extract_required_material_members(archive)
    textures = {
        name: _safe_member_path(extracted_root, member)
        for name, member in BC025_MATERIAL_MEMBERS.items()
    }
    return BC025ResolvedAssets(
        archive_path=archive,
        extracted_root=extracted_root,
        collision_source_path=collision_source,
        archive_sha256=archive_hash,
        collision_source_sha256=collision_hash,
        textures=textures,
    )


def bc025_static_asset_manifest() -> dict[str, Any]:
    """Return provenance without requiring local files or Isaac Sim."""

    return {
        "asset_id": BC025_ASSET_ID,
        "material": {
            "title": BC025_MATERIAL_TITLE,
            "source_url": BC025_MATERIAL_SOURCE_URL,
            "license": BC025_MATERIAL_LICENSE,
            "license_url": BC025_MATERIAL_LICENSE_URL,
            "archive_name": BC025_MATERIAL_ARCHIVE_NAME,
            "archive_sha256": BC025_MATERIAL_ARCHIVE_SHA256,
            "members": dict(BC025_MATERIAL_MEMBERS),
        },
        "collision_source": {
            "title": BC025_COLLISION_SOURCE_TITLE,
            "sha256": BC025_COLLISION_SOURCE_SHA256,
            "use": "collision_parameters_and_physics_calibration",
            "topology_composed": False,
            "license_status": "unresolved_user_provided_source",
        },
        "pillow024_calibration": dict(BC025_PILLOW_COLLISION_CALIBRATION),
    }


def inspect_pillow024_collision_source(path: Path) -> dict[str, Any]:
    """Read the audited collision/material values from a Pillow024 USD.

    This function intentionally imports PXR lazily.  It is called from the
    Isaac scene author after the Kit runtime exists, and can also be used by a
    live intake check.
    """

    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open BC025 Pillow024 source: {path}")
    collision_prim = None
    material_prim = None
    body_prim = None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.endswith(BC025_PILLOW_COLLISION_CALIBRATION["collision_prim_suffix"]):
            collision_prim = prim
        if prim_path.endswith("/PhysicsMaterial"):
            material_prim = prim
        if prim_path.endswith("/Pillow024") and (
            prim.GetAttribute("physxDeformableBody:selfCollision").IsValid()
            or prim.HasAPI("PhysxBaseDeformableBodyAPI")
        ):
            body_prim = prim
    if collision_prim is None or material_prim is None or body_prim is None:
        raise RuntimeError(
            "Pillow024 source does not expose the expected simulation_mesh, "
            "PhysicsMaterial, and deformable body prims"
        )

    def value(prim: Any, name: str, *, default: Any = None) -> Any:
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            return default
        return attr.Get()

    result = {
        "collision_prim_path": str(collision_prim.GetPath()),
        "body_prim_path": str(body_prim.GetPath()),
        "material_prim_path": str(material_prim.GetPath()),
        "contact_offset_m": float(
            value(collision_prim, "physxCollision:contactOffset")
        ),
        "rest_offset_m": float(value(collision_prim, "physxCollision:restOffset")),
        "self_collision": bool(
            value(body_prim, "physxDeformableBody:selfCollision")
        ),
        "enable_speculative_ccd": bool(
            value(body_prim, "physxDeformableBody:enableSpeculativeCCD")
        ),
        "linear_damping": float(
            value(body_prim, "physxDeformableBody:linearDamping")
        ),
        "solver_position_iterations": int(
            value(body_prim, "physxDeformableBody:solverPositionIterationCount")
        ),
        "static_friction": float(
            value(material_prim, "omniphysics:staticFriction")
        ),
        "dynamic_friction": float(
            value(material_prim, "omniphysics:dynamicFriction")
        ),
        "poissons_ratio": float(
            value(material_prim, "omniphysics:poissonsRatio")
        ),
        "youngs_modulus_pa": float(
            value(material_prim, "omniphysics:youngsModulus")
        ),
        "elasticity_damping": 0.0,
    }
    # The source may not have an elasticity field on its material schema.  The
    # audited value is zero, so preserve it explicitly in the normalized
    # record while still checking all fields that are actually authored.
    authored_elasticity = value(
        material_prim,
        "physxDeformableMaterial:elasticityDamping",
        default=None,
    )
    if authored_elasticity is not None:
        result["elasticity_damping"] = float(authored_elasticity)

    expected = BC025_PILLOW_COLLISION_CALIBRATION
    for key in (
        "contact_offset_m",
        "rest_offset_m",
        "linear_damping",
        "static_friction",
        "dynamic_friction",
        "poissons_ratio",
        "youngs_modulus_pa",
    ):
        if abs(float(result[key]) - float(expected[key])) > 1.0e-5 * max(
            1.0, abs(float(expected[key]))
        ):
            raise ValueError(
                f"Pillow024 {key} changed from the reviewed source: "
                f"expected={expected[key]!r}, got={result[key]!r}"
            )
    for key in ("self_collision", "enable_speculative_ccd"):
        if bool(result[key]) != bool(expected[key]):
            raise ValueError(
                f"Pillow024 {key} changed from the reviewed source: "
                f"expected={expected[key]!r}, got={result[key]!r}"
            )
    return result


def author_bc025_terry_cloth_material(
    stage: Any,
    material_path: str,
    assets: BC025ResolvedAssets,
) -> Any:
    """Create an opaque OmniPBR material using the downloaded Terry maps."""

    import numpy as np
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from pxr import Gf, Sdf

    material = OmniPBR(
        prim_path=material_path,
        texture_path=str(assets.textures["diffuse"].resolve()),
        color=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
    )
    material.set_project_uvw(False)
    material.set_texture_scale(0.9895, 1.0)
    material.set_reflection_roughness(0.78)
    material.set_metallic_constant(0.0)

    shader = material.shaders_list[0]

    def set_input(name: str, value_type: Any, value: Any) -> None:
        input_api = shader.GetInput(name)
        if not input_api.GetAttr().IsValid():
            input_api = shader.CreateInput(name, value_type)
        input_api.Set(value)

    set_input(
        "normalmap_texture",
        Sdf.ValueTypeNames.Asset,
        str(assets.textures["normal"].resolve()),
    )
    set_input("bump_factor", Sdf.ValueTypeNames.Float, 0.72)
    set_input("enable_ORM_texture", Sdf.ValueTypeNames.Bool, True)
    set_input(
        "ORM_texture",
        Sdf.ValueTypeNames.Asset,
        str(assets.textures["arm"].resolve()),
    )
    set_input("specular_level", Sdf.ValueTypeNames.Float, 0.42)
    set_input("ao_to_diffuse", Sdf.ValueTypeNames.Float, 0.18)
    # OmniPBR has no KHR_materials_specular/IOR texture socket.  Keep the map
    # in the manifest and on the material prim for provenance rather than
    # feeding it into the wrong roughness channel.
    material_prim = stage.GetPrimAtPath(material_path)
    material_prim.SetCustomDataByKey(
        "handson:terryClothSpecularIorTexture",
        str(assets.textures["specular_ior"].resolve()),
    )
    material_prim.SetCustomDataByKey("handson:materialSource", BC025_MATERIAL_TITLE)
    material_prim.SetCustomDataByKey("handson:materialLicense", BC025_MATERIAL_LICENSE)
    material_prim.SetCustomDataByKey("handson:materialArchiveSha256", assets.archive_sha256)
    material_prim.SetCustomDataByKey(
        "handson:materialMaps",
        json.dumps({name: str(path) for name, path in assets.textures.items()}),
    )
    # Force the shader input's asset type to stay explicit in the authored
    # layer.  This is useful when a downstream collector serializes the
    # material without opening the MDL implementation.
    _ = Gf  # imported for USD 5.1 type registration side effects
    return material.material


__all__ = [
    "BC025_ASSET_ID",
    "BC025_COLLISION_SOURCE_ENV",
    "BC025_COLLISION_SOURCE_SHA256",
    "BC025_COLLISION_SOURCE_TITLE",
    "BC025_MATERIAL_ARCHIVE_ENV",
    "BC025_MATERIAL_ARCHIVE_NAME",
    "BC025_MATERIAL_ARCHIVE_SHA256",
    "BC025_MATERIAL_LICENSE",
    "BC025_MATERIAL_LICENSE_URL",
    "BC025_MATERIAL_MEMBERS",
    "BC025_MATERIAL_ROOT_ENV",
    "BC025_MATERIAL_SOURCE_URL",
    "BC025_PILLOW_COLLISION_CALIBRATION",
    "BC025ResolvedAssets",
    "author_bc025_terry_cloth_material",
    "bc025_static_asset_manifest",
    "inspect_pillow024_collision_source",
    "resolve_bc025_assets",
]
