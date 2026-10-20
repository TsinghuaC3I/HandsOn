from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from taskdev.manifest import ManifestValidator
from taskdev.tasks.t46_store_charger import scene_spec


TASK_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_DIR.parents[2]


def test_static_asset_hashes_and_initial_layout() -> None:
    assert scene_spec.validate_static_contract(REPOSITORY_ROOT) == []
    assert scene_spec.DRAWER_TRANSLATION_M == (
        scene_spec.ORIGINAL_DRAWER_TRANSLATION_M[0]
        + scene_spec.DRAWER_BACK_SHIFT_M,
        scene_spec.ORIGINAL_DRAWER_TRANSLATION_M[1],
        scene_spec.DRAWER_SUPPORT_Z_M - scene_spec.DRAWER_SCALE[2] * 0.32,
    )
    layout = scene_spec.layout_contract()
    assert layout["drawer_table_clearance_m"] >= 0.0049
    assert layout["charger_table_clearance_m"] >= 0.0039
    assert layout["drawer_charger_separation_m"] >= 0.03
    assert tuple(round(value, 4) for value in layout["drawer_scaled_extent_m"]) == (
        0.4155,
        0.2836,
        0.5435,
    )
    assert abs(layout["drawer_opening_travel_m"] - 0.098) < 1.0e-9


def test_manifest_schema_identity_runtime_and_exact_assets() -> None:
    manifest_path = TASK_DIR / "task.yaml"
    result = ManifestValidator(REPOSITORY_ROOT).validate_path(manifest_path)
    assert result.valid, result.as_dict()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task"]["canonical_id"] == "B320-FU006"
    assert manifest["task"]["task_complete"] is False
    assert manifest["task"]["runtime_family"] == "isaac_sim_5_1"
    assert manifest["task"]["simulator_version"] == "5.1"
    assert manifest["task"]["physics_backend"] == "physx"
    assert manifest["task"]["newton_allowed"] is False
    asset_paths = {item["id"]: item["path"] for item in manifest["assets"]}
    assert asset_paths == {
        "drawer_45575": str(scene_spec.DRAWER_RELATIVE),
        "charger": str(scene_spec.CHARGER_RELATIVE),
    }
    assert manifest["task"]["action_shape"] == [54]
    assert [item["key"] for item in manifest["camera"]["canonical_streams"]] == [
        "ego",
        "wrist_left",
        "wrist_right",
    ]


def test_asset_contract_matches_scene_spec() -> None:
    contract = json.loads((TASK_DIR / "asset_contract.json").read_text(encoding="utf-8"))
    drawer = contract["assets"]["drawer_45575"]
    charger = contract["assets"]["dexverse_charger"]
    assert contract["task_complete"] is False
    assert drawer["entry_sha256"] == scene_spec.EXPECTED_SHA256[str(scene_spec.DRAWER_RELATIVE)]
    assert charger["entry_sha256"] == scene_spec.EXPECTED_SHA256[str(scene_spec.CHARGER_RELATIVE)]
    assert drawer["collision_api_count"] == scene_spec.DRAWER_COLLISION_COUNT
    assert drawer["rigid_body_api_count"] == scene_spec.DRAWER_RIGID_BODY_COUNT


def test_compatibility_runner_delegates_to_shared_scheduler_runtime() -> None:
    source = (TASK_DIR / "run_scene.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {name for name in imported if name.startswith(("isaac", "omni"))}
    assert "scripts/fu006_vertical_slice.py" in source
    assert 'args.device != "cuda:0"' in source
    assert "os.execv" in source
    assert "SimulationApp" not in source


def test_vertical_slice_never_restores_state_after_action_zero() -> None:
    source = (REPOSITORY_ROOT / "scripts" / "fu006_vertical_slice.py").read_text(
        encoding="utf-8"
    )
    assert '"state_writes_after_action_zero": False' in source
    assert '"state_restores_after_action_zero": 0' in source
    assert "restore_recorded_initial_state" not in source
    assert "_set_task_body_world_pose" not in source
    assert "physx_robot_joint_target" in source
    assert 'render_resolution=(480, 640)' in source
    assert "environment construction exited before returning a live scene" in source


def test_fu006_exact_assets_ignore_the_mjcf_overlay_root() -> None:
    source = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/task_preview.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _compose_fu006_assets")
    end = source.index("\ndef spawn_task_scene", start)
    function_source = source[start:end]
    assert (
        '_PACKAGED_XML_DIR = Path(__file__).resolve().parents[1] / "envs" / "xmls"'
        in source
    )
    assert (
        'asset_root = _PACKAGED_XML_DIR / "assets" / "bench320_fu006"'
        in function_source
    )
    assert 'asset_root = _XML_DIR / "assets"' not in function_source
    assert "_set_fu006_reference_transform" in function_source
    assert "translate=_FU006_DRAWER_TRANSLATION_M" in function_source
    assert (
        "_FU006_ORIGINAL_DRAWER_TRANSLATION_M = (-0.07, 0.0, 1.1931573060)"
        in source
    )
    assert "_FU006_DRAWER_BACK_SHIFT_M = 0.10" in source
    assert "_FU006_CHARGER_TRANSLATION_M = (-0.37, -0.22, 0.90750407)" in source
    assert "XformCommonAPI" not in function_source


def test_fu006_live_prim_names_match_the_exact_usd_schemas() -> None:
    source = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/abc_collection_env.py"
    ).read_text(encoding="utf-8")
    assert (
        '_FU006_DRAWER_ARTICULATION_ROOT = f"{_FU006_DRAWER_ROOT}/root_joint"'
        in source
    )
    assert '"fu006_charger": "E_body_562"' in source
    assert "drawer_articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI)" in source
    assert "scheduler-owned process exit on the original error" in source


def test_handoff_tracks_vertical_slice_without_false_completion() -> None:
    handoff = (TASK_DIR / "SCENE_HANDOFF.md").read_text(encoding="utf-8")
    assert "`task_complete=false`" in handoff
    assert "before the first PhysX" in handoff
    assert "54-D" in handoff
    assert "fresh-process" in handoff
