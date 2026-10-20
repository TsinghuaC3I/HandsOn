from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import runpy
import stat
import sys
import types
import zipfile

import pytest

from handson.sim.isaaclab import b320_bc017_scene as scene
from handson.sim.isaaclab.b320_bimanual_scene import B320SpawnResult


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / scene.BC017_MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))


def _rewrite_declared_json(
    root: Path,
    manifest: dict[str, object],
    relative: str,
    value: object,
) -> None:
    (root / relative).write_bytes(_json_bytes(value))
    entry = next(
        item
        for item in manifest["files"]
        if isinstance(item, dict) and item["path"] == relative
    )
    entry["size_bytes"] = (root / relative).stat().st_size
    entry["sha256"] = _sha256(root / relative)
    _write_manifest(root, manifest)


def _build_bundle(
    tmp_path: Path,
    *,
    archive_member: str = "wooden_drawer/source.glb",
    archive_symlink: bool = False,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "bc017_bundle"
    for relative in (
        "source_archives",
        "runtime",
        "materials",
        "textures",
        "evidence",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    archive_path = root / "source_archives/wooden_drawer_original.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if archive_symlink:
            info = zipfile.ZipInfo(archive_member)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "wooden_drawer/source.glb")
        else:
            archive.writestr(archive_member, b"selected-sketchfab-source")
    runtime_path = root / "runtime/bc017_wooden_drawer.usda"
    runtime_path.write_text(
        "#usda 1.0\n(defaultPrim = \"WoodenDrawer\", metersPerUnit = 1, upAxis = \"Z\")\n",
        encoding="utf-8",
    )
    (root / "materials/wood.mdl").write_text(
        "mdl 1.7; export material wood() = material();\n",
        encoding="utf-8",
    )
    (root / "textures/wood_basecolor.png").write_bytes(b"not-a-render-test")

    source_sha256 = _sha256(archive_path)
    runtime_sha256 = _sha256(runtime_path)
    dependency_paths = [
        "materials/wood.mdl",
        "textures/wood_basecolor.png",
    ]
    material_report = {
        "schema": "handson.bc017.material-closure",
        "runtime_usd_sha256": runtime_sha256,
        "dependency_paths": dependency_paths,
        "unresolved_references": [],
        "external_absolute_references": [],
    }
    (root / "evidence/material_closure.json").write_bytes(
        _json_bytes(material_report)
    )

    collider_prims = {
        scene.BC017_BASE_LINK_PRIM: [
            f"{scene.BC017_BASE_LINK_PRIM}/colliders/base_hull"
        ],
        **{
            link: [f"{link}/colliders/{role}_hull"]
            for role, link in scene.BC017_DRAWER_LINK_PRIMS.items()
        },
    }
    collision_report = {
        "schema": "handson.bc017.collision-audit",
        "source_archive_sha256": source_sha256,
        "runtime_usd_sha256": runtime_sha256,
        "drawer_links": {
            role: scene.BC017_DRAWER_LINK_PRIMS[role]
            for side in ("left", "right")
            for role in scene.BC017_DRAWER_ROLES_BY_SIDE[side]
        },
        "collider_prims": collider_prims,
        "derived_from_visible_geometry": True,
        "procedural_geometry": False,
        "fixed_base_verified": True,
        "all_links_have_colliders": True,
        "closed_reset_max_penetration_m": 0.0001,
        "support_gap_m": 0.001,
    }
    (root / "evidence/collision_audit.json").write_bytes(
        _json_bytes(collision_report)
    )

    source_entry = _descriptor(
        root,
        "source_archives/wooden_drawer_original.zip",
        "source_archive",
    )
    runtime_entry = _descriptor(
        root,
        "runtime/bc017_wooden_drawer.usda",
        "runtime_usd",
    )
    manifest: dict[str, object] = {
        "schema": "handson.bc017.asset-manifest",
        "schema_version": 1,
        "task": {
            "id": scene.TASK_ID,
            "name": scene.TASK_NAME,
            "requirement": scene.TASK_REQUIREMENT,
        },
        "clean_room": True,
        "provenance": {
            "title": scene.SELECTED_ASSET_TITLE,
            "creator": scene.SELECTED_ASSET_CREATOR,
            "platform": scene.SELECTED_ASSET_PLATFORM,
            "model_uuid": scene.SELECTED_ASSET_UUID,
            "source_url": scene.SELECTED_ASSET_URL,
            "license": scene.SELECTED_ASSET_LICENSE,
            "license_url": scene.SELECTED_ASSET_LICENSE_URL,
            "original_archive": {
                key: source_entry[key]
                for key in ("path", "sha256", "size_bytes")
            },
        },
        "derived_asset": {
            "runtime_usd": {
                key: runtime_entry[key]
                for key in ("path", "sha256", "size_bytes")
            },
            "source_archive_sha256": source_sha256,
            "conversion_tool": "Blender USD exporter",
            "conversion_version": "reviewed-test-fixture-v1",
            "clean_room_conversion": True,
            "no_procedural_geometry": True,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "articulation": {
            "fixed_base": True,
            "base_link_prim": scene.BC017_BASE_LINK_PRIM,
            "world_fixed_joint_prim": scene.BC017_WORLD_FIXED_JOINT_PRIM,
            "independent_dofs": True,
            "mimic_or_coupled_joints": [],
            "drawer_links": [
                {
                    "role": role,
                    "link_prim": scene.BC017_DRAWER_LINK_PRIMS[role],
                    "joint_prim": scene.BC017_DRAWER_JOINT_NAMES[role],
                    "joint_name": scene.BC017_DRAWER_JOINT_NAMES[role],
                    "joint_type": "prismatic",
                    "axis": "+X",
                    "closed_limit_m": 0.0,
                    "open_limit_m": scene.BC017_DRAWER_TRAVEL_M,
                    "reset_position_m": 0.0,
                }
                for side in ("left", "right")
                for role in scene.BC017_DRAWER_ROLES_BY_SIDE[side]
            ],
        },
        "placement": {
            "translation_m": [-0.28, 0.0, 0.891],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "closure": {
            "declared_files_complete": True,
            "unresolved_references": [],
            "external_absolute_references": [],
            "material_evidence": {
                "report_path": "evidence/material_closure.json",
                "dependency_paths": dependency_paths,
            },
            "collision_evidence": {
                "report_path": "evidence/collision_audit.json",
                "derived_from_selected_visible_asset": True,
                "procedural_colliders": False,
                "closed_reset_penetration_free": True,
                "collider_prims": collider_prims,
            },
        },
        "files": [
            source_entry,
            runtime_entry,
            _descriptor(root, "materials/wood.mdl", "material"),
            _descriptor(root, "textures/wood_basecolor.png", "texture"),
            _descriptor(root, "evidence/material_closure.json", "material_audit"),
            _descriptor(root, "evidence/collision_audit.json", "collision_audit"),
        ],
    }
    _write_manifest(root, manifest)
    return root, manifest


def test_selected_asset_identity_is_exact_and_has_no_procedural_fallback() -> None:
    assert scene.SELECTED_ASSET_TITLE == "Metal Office Desk"
    assert scene.SELECTED_ASSET_CREATOR == "Ulan Cabanilla"
    assert scene.SELECTED_ASSET_PLATFORM == "Poly Haven"
    assert scene.SELECTED_ASSET_UUID == "metal_office_desk"
    assert scene.SELECTED_ASSET_LICENSE == "CC0 1.0"
    source = Path(scene.__file__).read_text(encoding="utf-8")
    # The desk remains the selected source USD.  Procedural cuboids are
    # restricted to render-only highlights and study dressing; neither can
    # replace source geometry or add colliders.
    assert "UsdFileCfg(" in source
    marker_source = inspect.getsource(scene._spawn_bc017_target_edge_highlight)
    assert marker_source.count("CuboidCfg(") == 1
    assert "visualOnly" in marker_source
    assert "CollisionAPI" in marker_source
    study_source = inspect.getsource(scene._spawn_bc017_study_room)
    assert "CuboidCfg(" in study_source
    assert "renderOnly" in study_source
    assert "Cube.Define(" not in source
    assert "spawn_provisional" not in source


def test_bc017_cabinet_faces_arms_on_support_plane_with_70_percent_clearance(
) -> None:
    half = math.sqrt(0.5)
    assert scene.BC017_ASSET_ORIENTATION_WXYZ == pytest.approx(
        (0.0, 0.0, half, half)
    )
    assert sum(value * value for value in scene.BC017_ASSET_ORIENTATION_WXYZ) == (
        pytest.approx(1.0)
    )

    # After Rz(180) @ Rx(90), local +X points toward the arms along world -X,
    # local +Z points across the tabletop along world +Y, and local +Y is up.
    origin = scene.bc017_asset_point_world((0.0, 0.0, 0.0))
    axis_x = scene.bc017_asset_point_world((1.0, 0.0, 0.0))
    axis_y = scene.bc017_asset_point_world((0.0, 1.0, 0.0))
    axis_z = scene.bc017_asset_point_world((0.0, 0.0, 1.0))
    assert tuple(axis_x[i] - origin[i] for i in range(3)) == pytest.approx(
        (-1.0, 0.0, 0.0)
    )
    assert tuple(axis_y[i] - origin[i] for i in range(3)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert tuple(axis_z[i] - origin[i] for i in range(3)) == pytest.approx(
        (0.0, 1.0, 0.0)
    )

    support_plane = scene.bc017_asset_point_world(
        (0.0, scene.BC017_LOCAL_SUPPORT_MIN_Y_M, 0.0)
    )
    assert support_plane[2] == pytest.approx(
        scene.BC017_TABLE_TOP_Z_M + scene.BC017_TABLE_SUPPORT_CLEARANCE_M
    )
    # The enlarged BC017 worktable is 0.92 x 1.52 m at x=-0.15 m.
    table_x = (-0.61, 0.31)
    table_y = (-0.76, 0.76)

    closed_handles = tuple(
        scene.bc017_asset_point_world(local)
        for local in (
            scene.BC017_LOCAL_LEFT_HANDLE_CENTER_M,
            scene.BC017_LOCAL_RIGHT_HANDLE_CENTER_M,
        )
    )
    assert tuple(handle[0] for handle in closed_handles) == pytest.approx(
        (scene.BC017_TARGET_CLOSED_HANDLE_X_M,) * 2
    )
    assert 0.49 <= closed_handles[0][1] <= 0.51
    assert -0.51 <= closed_handles[1][1] <= -0.49
    assert 0.5 * (closed_handles[0][1] + closed_handles[1][1]) == pytest.approx(
        scene.BC017_TARGET_HANDLE_Y_M
    )
    assert closed_handles[0][2] == pytest.approx(closed_handles[1][2])
    assert 1.31 <= closed_handles[0][2] <= 1.34
    assert all(
        table_y[0] <= handle[1] <= table_y[1]
        for handle in closed_handles
    )
    pull_70_m = (
        scene.BC017_DRAWER_TRAVEL_M
        * scene.BC017_MIN_USABLE_PULL_FRACTION
    )
    pulled_handles = tuple(
        scene.bc017_asset_point_world(
            (local[0] + pull_70_m, local[1], local[2])
        )
        for local in (
            scene.BC017_LOCAL_LEFT_HANDLE_CENTER_M,
            scene.BC017_LOCAL_RIGHT_HANDLE_CENTER_M,
        )
    )
    assert pull_70_m == pytest.approx(0.1736)
    for closed_handle, pulled_handle in zip(
        closed_handles, pulled_handles, strict=True
    ):
        assert pulled_handle[0] == pytest.approx(-0.4136)
        assert pulled_handle[0] < closed_handle[0]
        assert pulled_handle[0] - table_x[0] >= 0.15


def test_bc017_drawer_slot_and_role_contract_is_exact() -> None:
    assert scene.BC017_DRAWER_SLOT_NAMES == ("top", "middle", "bottom")
    expected_roles = tuple(
        f"{side}_{slot}_drawer"
        for side in ("left", "right")
        for slot in scene.BC017_DRAWER_SLOT_NAMES
    )
    assert tuple(
        role
        for side in ("left", "right")
        for role in scene.BC017_DRAWER_ROLES_BY_SIDE[side]
    ) == expected_roles
    assert tuple(scene.BC017_DRAWER_LINK_PRIMS) == expected_roles
    assert tuple(scene.BC017_DRAWER_JOINT_NAMES) == expected_roles
    assert scene.BC017_DOF_NAMES == tuple(
        scene.BC017_DRAWER_JOINT_NAMES[role] for role in expected_roles
    )
    for side in ("left", "right"):
        for slot, slot_name in enumerate(scene.BC017_DRAWER_SLOT_NAMES):
            expected = f"{side}_{slot_name}_drawer"
            assert scene.bc017_drawer_role(side, slot) == expected
            assert scene.bc017_drawer_role(side, slot_name) == expected


def test_bc017_strict_pull_tab_zones_move_with_each_drawer_role() -> None:
    roles = tuple(scene.BC017_HANDLE_GRIP_LOCAL_BOUNDS_M)
    closed = {
        role: scene.bc017_handle_grip_bounds_world(role, 0.0)
        for role in roles
    }
    opened_q_m = 0.60 * scene.BC017_DRAWER_TRAVEL_M
    opened = {
        role: scene.bc017_handle_grip_bounds_world(role, opened_q_m)
        for role in roles
    }
    for role in roles:
        minimum, maximum = closed[role]
        opened_minimum, opened_maximum = opened[role]
        assert tuple(
            opened_minimum[index] - minimum[index] for index in range(3)
        ) == pytest.approx((-opened_q_m, 0.0, 0.0))
        assert tuple(
            opened_maximum[index] - maximum[index] for index in range(3)
        ) == pytest.approx((-opened_q_m, 0.0, 0.0))
        # The strict zone encloses the source U-shaped pull but excludes the
        # drawer face and surrounding desk geometry.
        local_minimum, local_maximum = scene.BC017_HANDLE_GRIP_LOCAL_BOUNDS_M[
            role
        ]
        assert maximum[0] - minimum[0] == pytest.approx(
            local_maximum[0] - local_minimum[0]
        )
        assert maximum[1] - minimum[1] == pytest.approx(
            local_maximum[2] - local_minimum[2]
        )
        assert maximum[2] - minimum[2] == pytest.approx(
            local_maximum[1] - local_minimum[1]
        )
        assert maximum[0] - minimum[0] > 0.029
        assert maximum[1] - minimum[1] > 0.080
        assert maximum[2] - minimum[2] > 0.0075
    assert all(
        closed[role][0][1] > 0.0
        for role in scene.BC017_DRAWER_ROLES_BY_SIDE["left"]
    )
    assert all(
        closed[role][1][1] < 0.0
        for role in scene.BC017_DRAWER_ROLES_BY_SIDE["right"]
    )
    # Old side-only calls remain exact aliases for the original top pair.
    assert scene.bc017_handle_grip_bounds_world("left", 0.0) == closed[
        "left_top_drawer"
    ]
    assert scene.bc017_handle_grip_bounds_world("right", 0.0) == closed[
        "right_top_drawer"
    ]

    with pytest.raises(ValueError, match="unsupported BC017 drawer role"):
        scene.bc017_handle_grip_bounds_world("center", 0.0)
    with pytest.raises(ValueError, match="outside the authored travel"):
        scene.bc017_handle_grip_bounds_world(
            "left", scene.BC017_DRAWER_TRAVEL_M + 1.0e-6
        )


def test_bc017_six_target_drawer_edges_have_distinct_side_visual_roles() -> None:
    assert set(scene.BC017_TARGET_EDGE_COLORS_RGB) == {"left", "right"}
    assert (
        scene.BC017_TARGET_EDGE_COLORS_RGB["left"]
        != scene.BC017_TARGET_EDGE_COLORS_RGB["right"]
    )
    assert set(scene.BC017_TARGET_EDGE_HIGHLIGHT_ROOTS) == set(
        scene.BC017_DRAWER_LINK_PRIMS
    )
    for role, link in scene.BC017_DRAWER_LINK_PRIMS.items():
        side = role.partition("_")[0]
        path = scene.BC017_TARGET_EDGE_HIGHLIGHT_ROOTS[role]
        assert path.startswith(f"{scene.BC017_ASSET_ROOT_PATH}/{link}/")
        assert f"{side}_hand_target_edge" in path


def test_bc017_highlight_visibility_selects_one_slot_per_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Prim:
        def __init__(self) -> None:
            self.visible = False
            self.custom_data: dict[str, object] = {}

        def IsValid(self) -> bool:
            return True

        def SetCustomDataByKey(self, key: str, value: object) -> None:
            self.custom_data[key] = value

    prims = {
        path: Prim()
        for path in scene.BC017_TARGET_EDGE_HIGHLIGHT_ROOTS.values()
    }

    class Stage:
        @staticmethod
        def GetPrimAtPath(path: str) -> Prim:
            return prims[path]

    class Imageable:
        def __init__(self, prim: Prim) -> None:
            self.prim = prim

        def MakeVisible(self) -> None:
            self.prim.visible = True

        def MakeInvisible(self) -> None:
            self.prim.visible = False

    pxr = types.ModuleType("pxr")
    pxr.UsdGeom = types.SimpleNamespace(Imageable=Imageable)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    observed = scene.set_bc017_target_edge_highlight_visibility(
        Stage(), left_slot=2, right_slot=1
    )
    selected = {"left_bottom_drawer", "right_middle_drawer"}
    assert {role for role, visible in observed.items() if visible} == selected
    assert sum(prim.visible for prim in prims.values()) == 2
    for role, path in scene.BC017_TARGET_EDGE_HIGHLIGHT_ROOTS.items():
        assert prims[path].visible is (role in selected)
        assert prims[path].custom_data["handson:targetSelected"] is (
            role in selected
        )


def test_bc017_study_room_is_render_only_and_api_isolated() -> None:
    assert scene.BC017_STUDY_ROOM_ROOT_PATH.startswith(
        "/World/VisualEnvironment/"
    )
    assert not scene.BC017_STUDY_ROOM_ROOT_PATH.startswith(
        scene.BC017_ASSET_ROOT_PATH
    )
    source = inspect.getsource(scene._spawn_bc017_study_room)
    assert 'SetCustomDataByKey("handson:renderOnly", True)' in source
    assert "prim.IsA(UsdGeom.Camera)" in source
    assert "prim.IsA(UsdPhysics.Joint)" in source
    assert "UsdGeom.Camera.Define" not in source
    assert "UsdPhysics.Joint.Define" not in source
    for api_name in (
        "UsdPhysics.RigidBodyAPI",
        "UsdPhysics.CollisionAPI",
        "UsdPhysics.ArticulationRootAPI",
        "UsdPhysics.MassAPI",
        "PhysxSchema.PhysxContactReportAPI",
    ):
        assert f"prim.HasAPI({api_name})" in source
        assert f"{api_name}.Apply" not in source
    physical_spawn_source = inspect.getsource(scene.spawn_bc017_task_assets)
    assert "_spawn_bc017_study_room" not in physical_spawn_source


def test_runtime_spawn_has_no_preflight_or_formal_truth_markers() -> None:
    source = inspect.getsource(scene.spawn_bc017_task_assets)
    assert "validate_bc017_asset_manifest" not in source
    assert "manifestPreflight" not in source
    assert "formalTruth" not in source
    assert "Sha256" not in source


def test_shared_spawn_result_serializes_typed_evidence_without_dynamic_fields() -> None:
    result = B320SpawnResult(
        evidence={
            "schema": "handson.test.spawn-evidence",
            "formal": False,
            "manifest_sha256": "1" * 64,
        }
    )
    encoded = asdict(result)
    assert encoded["evidence"] == result.evidence
    assert set(encoded) == {
        "body_paths",
        "articulation_root_paths",
        "static_paths",
        "dof_names",
        "collider_counts",
        "evidence",
    }


def test_repository_checkpoint_contains_the_validated_selected_asset_bundle() -> None:
    manifest = scene.BC017_ASSET_DIRECTORY / scene.BC017_MANIFEST_FILENAME
    assert manifest.is_file()

    report = scene.validate_bc017_asset_bundle()

    assert report.formal is False
    assert report.manifest_path == manifest.resolve()
    assert report.source_archive_path == (
        scene.BC017_ASSET_DIRECTORY / "source" / "original.zip"
    ).resolve()
    assert report.source_archive_sha256 == (
        "63f1ef6d9ec9737a762e6e3433a2b843b9407568ae1506bfd97f2026910b021c"
    )
    assert report.runtime_usd_path.is_file()
    assert report.left_joint_name == scene.BC017_LEFT_JOINT_NAME
    assert report.right_joint_name == scene.BC017_RIGHT_JOINT_NAME
    assert report.left_open_q_m == pytest.approx(0.248)
    assert report.right_open_q_m == pytest.approx(0.248)
    assert tuple(drawer.role for drawer in report.drawers) == tuple(
        role
        for side in ("left", "right")
        for role in scene.BC017_DRAWER_ROLES_BY_SIDE[side]
    )
    assert report.translation_m == pytest.approx(scene.BC017_ASSET_TRANSLATION_M)
    assert report.orientation_wxyz == pytest.approx(
        scene.BC017_ASSET_ORIENTATION_WXYZ
    )
    assert report.collider_counts == (3, 4, 4, 4, 4, 4, 4)


def test_generator_manifest_and_lineage_share_the_scaled_asset_contract() -> None:
    root = scene.BC017_ASSET_DIRECTORY
    manifest = json.loads(
        (root / scene.BC017_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    lineage = json.loads(
        (root / "evidence/source_lineage.json").read_text(encoding="utf-8")
    )
    repository_root = Path(__file__).resolve().parents[2]
    builder_path = (
        repository_root / "scripts/prepare_bc017_wooden_drawer_asset.py"
    )
    builder = runpy.run_path(
        str(builder_path), run_name="bc017_builder_contract"
    )
    derivation = lineage["derivation"]
    expected_roles = tuple(
        role
        for side in ("left", "right")
        for role in scene.BC017_DRAWER_ROLES_BY_SIDE[side]
    )
    assert builder["TASK_REQUIREMENT"] == scene.TASK_REQUIREMENT
    assert manifest["task"]["requirement"] == scene.TASK_REQUIREMENT
    assert builder["DRAWER_ROLE_ORDER"] == expected_roles
    assert builder["DRAWER_LINK_PRIMS"] == scene.BC017_DRAWER_LINK_PRIMS
    assert builder["DRAWER_JOINT_NAMES"] == scene.BC017_DRAWER_JOINT_NAMES
    assert set(builder["SOURCE_BASE_NODES"]) == {
        "metal_office_desk",
        "metal_office_desk_tray_01",
        "metal_office_desk_tray_02",
    }
    assert set(builder["SOURCE_DRAWER_NODES_BY_ROLE"]) == set(expected_roles)
    for drawer_role, translation in builder[
        "DRAWER_SOURCE_TRANSLATION_M_BY_ROLE"
    ].items():
        if "_top_" in drawer_role:
            assert translation[:2] != (0.0, 0.0)
        else:
            assert translation == (0.0, 0.0, 0.0)
    assert builder["BASELINE_UNIFORM_SCALE"] == pytest.approx(0.45)
    assert builder["TASK_SCALE_MULTIPLIER"] == pytest.approx(1.55)
    assert builder["UNIFORM_SCALE"] == pytest.approx(0.6975)
    assert builder["UNIFORM_SCALE"] == pytest.approx(
        derivation["uniform_scale"]
    )
    assert builder["BASELINE_UNIFORM_SCALE"] == pytest.approx(
        derivation["baseline_uniform_scale"]
    )
    assert builder["TASK_SCALE_MULTIPLIER"] == pytest.approx(
        derivation["task_scale_multiplier"]
    )
    assert builder["DRAWER_TRAVEL_M"] == pytest.approx(0.248)
    assert scene.BC017_DRAWER_TRAVEL_M == pytest.approx(
        builder["DRAWER_TRAVEL_M"]
    )
    assert [
        row["open_limit_m"] for row in manifest["articulation"]["drawer_links"]
    ] == pytest.approx([builder["DRAWER_TRAVEL_M"]] * 6)
    assert tuple(manifest["placement"]["translation_m"]) == pytest.approx(
        scene.BC017_ASSET_TRANSLATION_M
    )
    assert tuple(manifest["placement"]["orientation_wxyz"]) == pytest.approx(
        scene.BC017_ASSET_ORIENTATION_WXYZ
    )
    assert derivation["asset_root_placement_m"] == pytest.approx(
        manifest["placement"]["translation_m"]
    )
    assert derivation["local_handle_centers_m"]["left"] == pytest.approx(
        scene.BC017_LOCAL_LEFT_HANDLE_CENTER_M
    )
    assert derivation["local_handle_centers_m"]["right"] == pytest.approx(
        scene.BC017_LOCAL_RIGHT_HANDLE_CENTER_M
    )
    for drawer_role in scene.BC017_DRAWER_LINK_PRIMS:
        assert derivation["local_handle_centers_m"][drawer_role] == pytest.approx(
            scene.BC017_LOCAL_HANDLE_CENTERS_M[drawer_role]
        )
        committed_bounds = derivation["runtime_handle_grip_bounds_m"][
            drawer_role
        ]
        expected_bounds = scene.BC017_HANDLE_GRIP_LOCAL_BOUNDS_M[
            drawer_role
        ]
        assert committed_bounds[0] == pytest.approx(expected_bounds[0])
        assert committed_bounds[1] == pytest.approx(expected_bounds[1])
    assert set(derivation["base_source_nodes"]) == set(
        builder["SOURCE_BASE_NODES"]
    )
    assert derivation["left_target_source_node"] == builder[
        "SOURCE_LEFT_DRAWER_NODE"
    ]
    assert derivation["right_target_source_node"] == builder[
        "SOURCE_RIGHT_DRAWER_NODE"
    ]
    assert lineage["source_archive_sha256"] == manifest["derived_asset"][
        "source_archive_sha256"
    ]
    assert lineage["runtime_usd_sha256"] == manifest["derived_asset"][
        "runtime_usd"
    ]["sha256"]
    assert derivation["collider_prims"] == manifest["closure"][
        "collision_evidence"
    ]["collider_prims"]


def test_repository_colliders_align_visible_structure_without_blocking_bays() -> None:
    root = scene.BC017_ASSET_DIRECTORY
    manifest = json.loads(
        (root / scene.BC017_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    lineage = json.loads(
        (root / "evidence/source_lineage.json").read_text(encoding="utf-8")
    )
    assert next(
        entry["role"]
        for entry in manifest["files"]
        if entry["path"] == "evidence/source_lineage.json"
    ) == "lineage_audit"
    generation = lineage["collision_generation"]
    assert generation["collider_count"] == 27
    assert generation["exact_mesh_decomposition_collider_count"] == 3
    assert generation["exact_mesh_sdf_collider_count"] == 6
    assert generation["source_handle_compound_convex_hull_collider_count"] == 18
    assert generation["drawer_filtered_pair_count"] == 15
    assert generation["drawer_filtered_pair_scope"] == (
        "drawer_body_to_drawer_body_only"
    )
    assert generation["fixture_contacts_preserved"] is True
    assert generation["hand_contacts_preserved"] is True
    assert generation["low_poly_convex_hull_collider_count"] == 0
    assert generation["gpu_convex_hull_vertex_limit"] == 64
    assert generation["generation"] == (
        "exact_source_decomposition_for_fixture_plus_welded_"
        "drawer_body_sdf_and_three_source_handle_hulls"
    )
    assert generation["source_derived"] is True
    assert generation["hand_authored_or_placeholder"] is False
    assert generation["primitive_or_generic_proxy_fallback"] is False
    derivations = lineage["derivation"]["collider_derivations"]
    assert len(derivations) == 27
    declared_by_link = manifest["closure"]["collision_evidence"][
        "collider_prims"
    ]
    assert {
        link: len(paths) for link, paths in declared_by_link.items()
    } == {
        scene.BC017_BASE_LINK_PRIM: 3,
        **{link: 4 for link in scene.BC017_DRAWER_LINK_PRIMS.values()},
    }
    declared = {
        path
        for paths in declared_by_link.values()
        for path in paths
    }
    assert declared == {row["collider_prim"] for row in derivations}
    assert (
        sum("/source_exact_mesh_colliders/" in path for path in declared)
        == 27
    )
    assert not any("/source_derived_colliders/" in path for path in declared)

    fixture_rows = [
        row
        for row in derivations
        if row["semantic_role"] == "fixture_structure_collider"
    ]
    drawer_body_rows = [
        row
        for row in derivations
        if row["semantic_role"] == "drawer_body_collider"
    ]
    handle_rows = [
        row
        for row in derivations
        if row["semantic_role"] == "integrated_u_pull_compound_collider"
    ]
    assert (len(fixture_rows), len(drawer_body_rows), len(handle_rows)) == (
        3,
        6,
        18,
    )
    for row in derivations:
        assert row["source_derived"] is True
        assert row["hand_authored_or_placeholder"] is False
        assert row["primitive_or_generic_proxy_fallback"] is False
        assert row["source_visible_prim"] != row["collider_prim"]
        assert row["source_visible_owner_relative_prim"].startswith("/")
    for row in fixture_rows:
        assert row["source_aabb_preserved"] is True
        assert row["source_topology_preserved"] is True
        assert row["authored_vertex_count"] == row["source_vertex_count"]
        assert row["authored_face_count"] == row["source_face_count"]
        assert (
            row["authored_face_counts_sha256"]
            == row["source_face_counts_sha256"]
        )
        assert (
            row["authored_face_indices_sha256"]
            == row["source_face_indices_sha256"]
        )
    assert all(
        row["mesh_collision_approximation"] == "convexDecomposition"
        for row in fixture_rows
    )
    assert {row["drawer_role"] for row in drawer_body_rows} == set(
        scene.BC017_DRAWER_LINK_PRIMS
    )
    assert all(
        row["mesh_collision_approximation"] == "sdf"
        for row in drawer_body_rows
    )
    assert all(row["sdf_resolution"] == 256 for row in drawer_body_rows)
    assert all(
        row["sdf_remeshing_enabled"] is True for row in drawer_body_rows
    )
    assert all(
        row["source_component_closed_manifold_verified"] is True
        for row in drawer_body_rows
    )
    assert {
        (row["drawer_role"], row["handle_segment"]) for row in handle_rows
    } == {
        (drawer_role, segment)
        for drawer_role in scene.BC017_DRAWER_LINK_PRIMS
        for segment in ("negative_leg", "crossbar", "positive_leg")
    }
    for row in handle_rows:
        assert row["mesh_collision_approximation"] == "convexHull"
        assert row["convex_verified"] is True
        assert row["closed_manifold_verified"] is True
        assert row["source_vertex_subset_verified"] is True
        assert row["handle_source_triangle_coverage_verified"] is True
        assert len(row["convex_halfspaces_local"]) >= 4
        for equation in row["convex_halfspaces_local"]:
            assert len(equation) == 4
            assert math.sqrt(sum(value * value for value in equation[:3])) == (
                pytest.approx(1.0, abs=1.0e-6)
            )
    derivation = lineage["derivation"]
    assert derivation["base_source_nodes"] == [
        "metal_office_desk",
        "metal_office_desk_tray_01",
        "metal_office_desk_tray_02",
    ]
    assert derivation["left_target_source_node"] == (
        "metal_office_desk_drawer_04"
    )
    assert derivation["right_target_source_node"] == (
        "metal_office_desk_drawer_01"
    )
    assert derivation["omitted_decorative_source_nodes"] == []
    bound_audit = derivation["visual_collision_bound_audit"]
    assert set(bound_audit) == {
        scene.BC017_BASE_LINK_PRIM,
        *scene.BC017_DRAWER_LINK_PRIMS.values(),
    }
    for row in bound_audit.values():
        assert row["aligned_within_tolerance"] is True
        assert row["tolerance_m"] == pytest.approx(1.0e-6)
        assert row["maximum_absolute_bound_error_m"] <= row["tolerance_m"]


def test_repository_filters_only_guided_drawer_body_pairs() -> None:
    root = scene.BC017_ASSET_DIRECTORY
    runtime = (
        root / scene.BC017_RUNTIME_USD_RELATIVE_PATH
    ).read_text(encoding="utf-8")
    lineage = json.loads(
        (root / "evidence/source_lineage.json").read_text(encoding="utf-8")
    )
    target_blocks = re.findall(
        r"rel physics:filteredPairs = \[(.*?)\n\s*\]",
        runtime,
        flags=re.DOTALL,
    )
    roles = tuple(
        role
        for side in ("left", "right")
        for role in scene.BC017_DRAWER_ROLES_BY_SIDE[side]
    )
    assert len(target_blocks) == len(roles) == 6
    all_links = set(scene.BC017_DRAWER_LINK_PRIMS.values())
    for role, block in zip(roles, target_blocks, strict=True):
        targets = set(
            re.findall(r"</WoodenDrawer/([^>]+)>", block)
        )
        assert targets == all_links - {scene.BC017_DRAWER_LINK_PRIMS[role]}
        assert scene.BC017_BASE_LINK_PRIM not in targets
        assert all("/" not in target for target in targets)

    derivation = lineage["derivation"]
    assert derivation["drawer_filtered_pair_count"] == 15
    assert len(derivation["drawer_filtered_pairs"]) == 15
    policy = derivation["drawer_filtered_pair_policy"]
    assert policy == {
        "scope": "drawer_body_to_drawer_body_only",
        "reason": (
            "parallel prismatic guides preserve row/pedestal offsets; "
            "suppress source-seam SDF/contact-offset chatter"
        ),
        "fixture_contacts_preserved": True,
        "hand_contacts_preserved": True,
        "joint_limits_or_locks_added": False,
    }
    audit = derivation["authored_closed_collision_audit"]
    assert audit["drawer_filtered_body_pair_count"] == 15
    problem_rows = [
        row
        for row in audit["cross_body_pairs"]
        if {row["left_body"], row["right_body"]}
        == {
            scene.BC017_DRAWER_LINK_PRIMS["right_top_drawer"],
            scene.BC017_DRAWER_LINK_PRIMS["right_middle_drawer"],
        }
        and not row["separated"]
    ]
    assert len(problem_rows) == 1
    assert problem_rows[0]["collision_filtered"] is True
    assert problem_rows[0]["aabb_result"] == (
        "excluded_by_guided_drawer_filtered_pair"
    )


def test_all_six_role_aware_handle_surface_filters_accept_source_hulls() -> None:
    lineage = json.loads(
        (
            scene.BC017_ASSET_DIRECTORY / "evidence/source_lineage.json"
        ).read_text(encoding="utf-8")
    )
    bounds = lineage["derivation"]["authored_closed_collision_audit"][
        "collider_bounds"
    ]
    for drawer_role, link in scene.BC017_DRAWER_LINK_PRIMS.items():
        for segment in ("negative_leg", "crossbar", "positive_leg"):
            row = next(
                value
                for path, value in bounds.items()
                if path.startswith(f"{link}/")
                and f"handle_{segment}_source_convex" in path
            )
            minimum = row["minimum_m"]
            maximum = row["maximum_m"]
            local = [
                0.5 * (float(minimum[index]) + float(maximum[index]))
                for index in range(3)
            ]
            # Minimum local Y is an exposed plane on each source-derived
            # convex hull and lies away from the compound seams.
            local[1] = float(minimum[1])
            assert scene.bc017_point_on_handle_surface_world(
                drawer_role,
                0.0,
                scene.bc017_asset_point_world(local),
                tolerance_m=1.0e-5,
            )


def test_bc017_sources_contain_no_legacy_runtime_constraint_tokens() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    sources = {
        "scene": Path(scene.__file__),
        "env": Path(scene.__file__).with_name("b320_bc017_env.py"),
        "builder": (
            repository_root / "scripts/prepare_bc017_wooden_drawer_asset.py"
        ),
        "runtime": (
            scene.BC017_ASSET_DIRECTORY
            / scene.BC017_RUNTIME_USD_RELATIVE_PATH
        ),
    }
    for label, path in sources.items():
        source = path.read_text(encoding="utf-8")
        for forbidden in ("PhysxD6", "LimitAPI", "runtimeLocked"):
            assert forbidden not in source, f"{label} retained {forbidden}"


def test_missing_runtime_usd_fails_before_stage_or_sim_utils_access(
    tmp_path: Path,
) -> None:
    class Untouchable:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"stage/sim access before validation: {name}")

    with pytest.raises(FileNotFoundError, match="runtime USD is missing"):
        scene.spawn_bc017_task_assets(
            Untouchable(),
            Untouchable(),
            asset_directory=tmp_path / "absent",
        )


def test_positive_temporary_bundle_returns_nonformal_byte_preflight_report(
    tmp_path: Path,
) -> None:
    root, _manifest = _build_bundle(tmp_path)
    report = scene.validate_bc017_asset_bundle(root)

    assert report.formal is False
    assert report.manifest_sha256 == _sha256(root / scene.BC017_MANIFEST_FILENAME)
    assert report.source_archive_sha256 == _sha256(report.source_archive_path)
    assert report.runtime_usd_sha256 == _sha256(report.runtime_usd_path)
    assert report.fixture_body_prim_path == (
        f"{scene.BC017_ASSET_ROOT_PATH}/{scene.BC017_BASE_LINK_PRIM}"
    )
    assert report.left_body_prim_path.endswith(scene.BC017_LEFT_DRAWER_LINK_PRIM)
    assert report.right_body_prim_path.endswith(scene.BC017_RIGHT_DRAWER_LINK_PRIM)
    assert report.left_joint_name == scene.BC017_LEFT_JOINT_NAME
    assert report.right_joint_name == scene.BC017_RIGHT_JOINT_NAME
    assert (report.left_closed_q_m, report.left_open_q_m) == (
        0.0,
        scene.BC017_DRAWER_TRAVEL_M,
    )
    assert (report.right_closed_q_m, report.right_open_q_m) == (
        0.0,
        scene.BC017_DRAWER_TRAVEL_M,
    )
    assert report.collider_counts == (1, 1, 1, 1, 1, 1, 1)
    assert len(report.validated_files) == 6


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    manifest["files"][0]["path"] = "../outside.zip"
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="path_unsafe"):
        scene.validate_bc017_asset_bundle(root)


def test_source_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    root, _manifest = _build_bundle(tmp_path, archive_member="../escape.glb")
    with pytest.raises(scene.BC017AssetGateError, match="path_unsafe"):
        scene.validate_bc017_asset_bundle(root)


def test_source_archive_symlink_is_rejected(tmp_path: Path) -> None:
    root, _manifest = _build_bundle(
        tmp_path,
        archive_member="wooden_drawer/source-link.glb",
        archive_symlink=True,
    )
    with pytest.raises(scene.BC017AssetGateError, match="archive_symlink_forbidden"):
        scene.validate_bc017_asset_bundle(root)


def test_filesystem_symlink_is_rejected(tmp_path: Path) -> None:
    root, _manifest = _build_bundle(tmp_path)
    texture = root / "textures/wood_basecolor.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(texture.read_bytes())
    texture.unlink()
    texture.symlink_to(outside)
    with pytest.raises(scene.BC017AssetGateError, match="symlink_forbidden"):
        scene.validate_bc017_asset_bundle(root)


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    root, _manifest = _build_bundle(tmp_path)
    runtime = root / "runtime/bc017_wooden_drawer.usda"
    modified = bytearray(runtime.read_bytes())
    modified[-2] ^= 0x01
    runtime.write_bytes(modified)
    with pytest.raises(scene.BC017AssetGateError, match="hash_mismatch"):
        scene.validate_bc017_asset_bundle(root)


def test_license_drift_is_rejected(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    manifest["provenance"]["license"] = "CC0-1.0"
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="license_mismatch"):
        scene.validate_bc017_asset_bundle(root)


def test_retired_asset_marker_is_rejected(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    retired = "dv_storage_drawers_" + "40417"
    manifest["provenance"]["title"] = retired
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="retired_lineage"):
        scene.validate_bc017_asset_bundle(root)


def test_procedural_geometry_or_colliders_are_rejected(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    manifest["derived_asset"]["no_procedural_geometry"] = False
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="procedural_forbidden"):
        scene.validate_bc017_asset_bundle(root)

    manifest["derived_asset"]["no_procedural_geometry"] = True
    manifest["closure"]["collision_evidence"]["procedural_colliders"] = True
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="procedural_forbidden"):
        scene.validate_bc017_asset_bundle(root)


def test_duplicate_drawer_link_and_joint_are_rejected(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    links = manifest["articulation"]["drawer_links"]
    links[1] = deepcopy(links[0])
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="articulation_duplicate"):
        scene.validate_bc017_asset_bundle(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open_limit_m", 0.0),
        ("open_limit_m", 0.019),
        ("closed_limit_m", -0.001),
        ("reset_position_m", 0.001),
    ],
)
def test_invalid_closed_open_limits_are_rejected(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    root, manifest = _build_bundle(tmp_path)
    manifest["articulation"]["drawer_links"][0][field] = value
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="joint_limit_invalid"):
        scene.validate_bc017_asset_bundle(root)


def test_material_closure_must_have_no_unresolved_reference(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    manifest["closure"]["unresolved_references"] = ["missing_texture.png"]
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="material_closure_invalid"):
        scene.validate_bc017_asset_bundle(root)


def test_material_report_must_bind_runtime_hash(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    report_path = root / "evidence/material_closure.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_usd_sha256"] = "0" * 64
    _rewrite_declared_json(
        root,
        manifest,
        "evidence/material_closure.json",
        report,
    )
    with pytest.raises(scene.BC017AssetGateError, match="material_closure_invalid"):
        scene.validate_bc017_asset_bundle(root)


def test_collision_report_must_prove_nonpenetrating_support(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    report_path = root / "evidence/collision_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["closed_reset_max_penetration_m"] = 0.01
    _rewrite_declared_json(
        root,
        manifest,
        "evidence/collision_audit.json",
        report,
    )
    with pytest.raises(scene.BC017AssetGateError, match="collision_evidence_invalid"):
        scene.validate_bc017_asset_bundle(root)


def test_duplicate_or_undeclared_files_break_byte_closure(tmp_path: Path) -> None:
    root, manifest = _build_bundle(tmp_path)
    manifest["files"].append(deepcopy(manifest["files"][0]))
    _write_manifest(root, manifest)
    with pytest.raises(scene.BC017AssetGateError, match="closure_duplicate"):
        scene.validate_bc017_asset_bundle(root)

    root, _manifest = _build_bundle(tmp_path / "second")
    (root / "textures/unreviewed.png").write_bytes(b"undeclared")
    with pytest.raises(scene.BC017AssetGateError, match="closure_incomplete"):
        scene.validate_bc017_asset_bundle(root)
