from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPOSITORY_ROOT / "scripts" / "prepare_g01_2_pillow_asset.py"
ASSET_DIRECTORY = (
    REPOSITORY_ROOT
    / "handson"
    / "handson"
    / "sim"
    / "envs"
    / "usd"
    / "g01_deformable_asset_inspection"
    / "Pillow024_v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(
        (ASSET_DIRECTORY / "asset_manifest.json").read_text(encoding="utf-8")
    )


def test_g01_2_builder_locks_source_and_single_collection_runtime() -> None:
    builder = runpy.run_path(str(BUILDER_PATH), run_name="g01_2_builder_contract")

    assert str(builder["SOURCE_USD"]) == "/home/user/下载/Pillow024/Pillow024.usd"
    assert builder["EXPECTED_SOURCE_SHA256"] == (
        "8b08e24ba0e8bc359f8802fce393fddb6234b31ca60a8e51a85404638cc80305"
    )
    assert builder["EXPECTED_SOURCE_SIZE_BYTES"] == 19_832_135
    assert builder["ASSET_DIRECTORY"] == ASSET_DIRECTORY
    source = BUILDER_PATH.read_text(encoding="utf-8")
    assert 'isaacsim_version != "5.1.0.0"' in source
    assert 'source_version != "2.3.2"' in source
    assert "toolchain-6" not in source
    assert "isaacsim_newton" not in source


def test_committed_pillow_v2_manifest_and_bytes_are_closed() -> None:
    manifest = _manifest()
    output = ASSET_DIRECTORY / "Pillow024.usd"
    readme = ASSET_DIRECTORY / "README.md"

    assert manifest["schema"] == "handson.g01.pillow024-v2.asset-manifest"
    assert manifest["schema_version"] == 1
    assert manifest["source"] == {
        "copied_into_repository": False,
        "immutable": True,
        "path": "/home/user/下载/Pillow024/Pillow024.usd",
        "sha256": "8b08e24ba0e8bc359f8802fce393fddb6234b31ca60a8e51a85404638cc80305",
        "size_bytes": 19_832_135,
    }
    assert output.is_file()
    assert readme.is_file()
    assert manifest["output"]["sha256"] == _sha256(output)
    assert manifest["output"]["size_bytes"] == output.stat().st_size
    assert manifest["output"]["deterministic_double_build_byte_identical"] is True
    assert manifest["output"]["source_texture_files_copied"] is False
    assert manifest["runtime"]["isaacsim"] == "5.1.0.0"
    assert manifest["runtime"]["isaaclab"] == "2.3.2"
    assert manifest["runtime"]["collection_runtime_family"] == "isaac_sim_5_1"

    files = {row["path"]: row for row in manifest["files"]}
    assert files["Pillow024.usd"]["sha256"] == _sha256(output)
    assert files["README.md"]["sha256"] == _sha256(readme)


def test_normalization_manifest_records_strict_rest_state_and_known_risk() -> None:
    manifest = _manifest()
    validation = manifest["validation"]
    visual = validation["visual"]
    simulation = validation["simulation"]
    physics = validation["physics"]

    assert (visual["point_count"], visual["face_count"], visual["corner_count"]) == (
        5_212,
        10_418,
        31_256,
    )
    assert visual["points_equal_bind_points"] is True
    assert visual["child_mass_api_removed"] is True
    assert visual["child_physics_material_binding_removed"] is True
    assert visual["unused_color_primvar_removed"] is True
    assert visual["closed_two_manifold"] is True
    assert visual["normal_interpolation"] == "faceVarying"
    assert visual["normal_preservation"]["changed_vector_count"] == 0
    assert visual["normal_preservation"]["preserved_byte_exact"] is True
    assert (
        visual["normal_preservation"]["source_array_sha256"]
        == "9ba5cbd99a85e20d293a06b9e08b2c6a01b04dbf35254db43dbce587892e692a"
    )
    assert (
        visual["normal_preservation"]["normalized_array_sha256"]
        == visual["normal_preservation"]["source_array_sha256"]
    )
    assert visual["bind_smooth_reference_alignment"]["minimum_corner_dot"] > 0.999999

    assert (
        simulation["point_count"],
        simulation["tet_count"],
        simulation["surface_face_count"],
    ) == (11_643, 36_327, 20_822)
    assert simulation["points_equal_rest_shape_points"] is True
    assert simulation["velocities_all_zero"] is True
    assert simulation["explicit_visibility"] == "invisible"
    assert simulation["rest_tets_all_positive"] is True
    assert simulation["surface_faces_match_derived_boundary"] is True
    assert simulation["surface_edge_incidence"] == {"2": 31_155, "4": 39}
    assert simulation["nonmanifold_surface_edge_count"] == 39
    assert simulation["topology_remeshed"] is False

    assert physics["root_mass_kg"] == 0.800000011920929
    assert physics["physics_material_bindings"] == {
        "root": "/root/Physics/PhysicsMaterial",
        "simulation": "/root/Physics/PhysicsMaterial",
    }
    assert manifest["normalization"]["tet_topology_changed"] is False
    assert manifest["normalization"]["material_parameters_changed"] is False
    assert manifest["normalization"]["root_mass_changed"] is False
    assert manifest["known_risks"][0]["count"] == 39
    assert manifest["known_risks"][0]["remeshed"] is False


def test_v2_reuses_shared_textures_without_copying_them() -> None:
    manifest = _manifest()
    assert not (ASSET_DIRECTORY / "texture").exists()
    dependencies = manifest["dependencies"]
    assert len(dependencies) == 3
    assert {row["authored_path"] for row in dependencies} == {
        "../Pillow024/texture/T_Pillow024_BC001.png",
        "../Pillow024/texture/T_Pillow024_N001.png",
        "../Pillow024/texture/T_Pillow024_ORM001.png",
    }
    for row in dependencies:
        dependency = ASSET_DIRECTORY / row["authored_path"]
        assert dependency.is_file()
        assert row["copied_into_v2"] is False
        assert row["sha256"] == _sha256(dependency)


def test_asset_provenance_integration_and_replay_boundaries_are_explicit() -> None:
    manifest = _manifest()
    assert manifest["repository_integration"] == {
        "authored_by_this_asset_builder": False,
        "formal_physical_replay_claimed": False,
        "g01_2_environment_wired": True,
        "g01_2_scene_wired": True,
        "g01_2_task_registration_wired": True,
    }
    provenance = manifest["provenance_and_license"]
    assert provenance["independent_license_metadata_in_download"] is False
    assert provenance["license_name"] is None
    assert provenance["new_license_grant_or_authorization"] is False
    assert provenance["inherited_supplier_archive_record"] == {
        "filename": "ZRYZ_LW_Assets_20260807.zip",
        "relationship": (
            "Inherited from the repository's existing Pillow024 supplier "
            "provenance record."
        ),
        "sha256": "90798bdad84d8b1de38eb625984f7fefa4ae3b56c7297fb4769bd35151189306",
    }
    readme = (ASSET_DIRECTORY / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    assert "39 boundary-surface edges" in normalized_readme
    assert "does not remesh" in normalized_readme
    assert "Companion repository changes wire this asset into the G01-2 scene" in normalized_readme
    assert "live deformable rendering must still be checked" in normalized_readme
    assert "did not include independent license metadata" in normalized_readme
    assert "not a new license grant or authorization" in normalized_readme
